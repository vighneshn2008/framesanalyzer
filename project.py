"""
main.py
--------
CLI tool for "Experimental Investigation of Effect of Moment of Inertia
on Acceleration of Rolling Bodies".

Run:  python main.py

Menu:
1. Configure a solid   (name / mass / radius / incline angle / gate distance / I_theory)
2. Record a trial      (mark 2 points on a video, auto-track object, get time in ms)
3. View a solid's data
4. Generate & export comparison table for all 8 solids
5. Exit
"""

import os
import cv2

import data_manager as dm
import tracker
import perspective


def choose_slot(data):
    print("\nSolids:")
    for s in data["solids"]:
        label = s["name"] or "(unconfigured)"
        filled = sum(1 for t in s["trials_ms"] if t is not None)
        print(f"  {s['slot']}. {label}   [{filled}/3 trials recorded]")
    raw = input("Choose slot number (1-8): ").strip()
    try:
        slot = int(raw)
    except ValueError:
        print("Invalid input.")
        return None
    if slot < 1 or slot > dm.MAX_SOLIDS:
        print("Out of range.")
        return None
    return slot


def _read_float(prompt, current):
    v = input(f"{prompt} [{current}]: ").strip()
    if not v:
        return current
    try:
        return float(v)
    except ValueError:
        print("  Not a number, keeping previous value.")
        return current


def configure_solid(data):
    slot = choose_slot(data)
    if slot is None:
        return
    s = dm.get_solid(data, slot)

    print(f"\nConfiguring slot {slot} (press Enter to keep the current value)")
    name = input(f"Name [{s['name']}]: ").strip()
    if name:
        s["name"] = name

    s["mass_kg"] = _read_float("Mass (kg)", s["mass_kg"])
    s["radius_m"] = _read_float("Radius (m)", s["radius_m"])
    s["theta_deg"] = _read_float("Incline angle theta (degrees)", s["theta_deg"])
    s["distance_m"] = _read_float("Distance from release point to END gate (m)", s["distance_m"])
    s["I_theory"] = _read_float("Theoretical moment of inertia I (kg.m^2)", s["I_theory"])

    dm.save_data(data)
    print("Saved.")


def record_from_webcam(name, slot):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    safe_name = "".join(c if c.isalnum() else "_" for c in name)
    out_path = f"video_slot{slot}_{safe_name}.avi"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    print("\nWebcam preview open. Press 'r' to START recording, "
          "'s' to STOP & save, 'q' to abort.")
    recording = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        disp = frame.copy()
        status = "RECORDING (press s to stop)" if recording else "press r to start"
        color = (0, 0, 255) if recording else (0, 255, 0)
        cv2.putText(disp, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("Webcam - r:start  s:stop&save  q:abort", disp)
        if recording:
            writer.write(frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            recording = True
        elif key == ord('s'):
            break
        elif key == ord('q'):
            cap.release()
            writer.release()
            cv2.destroyAllWindows()
            if os.path.exists(out_path):
                os.remove(out_path)
            return None

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    return out_path


def _setup_perspective(first_frame):
    """Optional 4-corner perspective rectification. Returns H (or None)."""
    print("\nCamera angled / not square-on? Mark the 4 corners of a rectangle "
          "you can measure on the TABLE plane (e.g. the table top) to apply "
          "perspective correction so the END gate stays perpendicular to the "
          "motion.")
    ans = input("Set up perspective correction? (y/N): ").strip().lower()
    if ans != "y":
        return None
    pts = perspective.mark_rectangle(first_frame)
    if pts is None:
        print("Skipping perspective correction.")
        return None
    w = float(input("Width of the marked rectangle (m): ").strip())
    h = float(input("Height of the marked rectangle (m): ").strip())
    if w <= 0 or h <= 0:
        print("Invalid size - skipping perspective correction.")
        return None
    H = perspective.homography_from_rect(pts, w, h)
    print("Perspective correction active.")
    return H


def record_trial(data):
    slot = choose_slot(data)
    if slot is None:
        return
    s = dm.get_solid(data, slot)
    if not s["name"]:
        print("Configure this solid first (option 1) before recording trials.")
        return

    print("\nVideo source:")
    print("  1. Existing video file")
    print("  2. Live webcam (records a new video first)")
    src_choice = input("Choose (1/2): ").strip()

    if src_choice == "2":
        video_path = record_from_webcam(s["name"], slot)
        if video_path is None:
            print("Recording aborted.")
            return
    else:
        video_path = input("Path to video file: ").strip()
        if not os.path.exists(video_path):
            print("File not found.")
            return

    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    cap.release()
    if not ret:
        print("Could not read video.")
        return

    H = _setup_perspective(first_frame)

    print("\nMark the END gate point (where the crossing time is measured to) "
          "- 1 click, then press 'c'.")
    points = tracker.mark_points(first_frame, n_points=1)
    if points is None:
        print("Aborted.")
        return

    print("\nTracking mode: color-based (click on the object to sample its color).")
    cap = cv2.VideoCapture(video_path)
    sample_frame = None
    for _ in range(8):  # skip ahead a little so the object is visible
        ret, sample_frame = cap.read()
        if not ret:
            break
    cap.release()
    if sample_frame is None:
        print("Could not grab a frame for color sampling.")
        return
    color_range = tracker.pick_color(sample_frame)
    if color_range is None:
        print("Aborted.")
        return

    print("\nProcessing video, please wait...")
    trajectory, fps = tracker.track_object(video_path, color_range=color_range)
    result = tracker.compute_crossing_time_ms(trajectory, points, H=H)

    if result is None:
        print("Could not detect the object crossing the END gate clearly.\n"
              "Tips: re-mark the END point more precisely, improve lighting/contrast,\n"
              "or re-sample the object color.")
        return

    print(f"\nDetected crossing time: {result['elapsed_ms']} ms  "
          f"(release={result['start_time_ms']} ms, end={result['end_time_ms']} ms, "
          f"video FPS={fps:.2f})")

    trial_no = input("Save as trial 1, 2, or 3? (Enter to discard): ").strip()
    if trial_no not in ("1", "2", "3"):
        print("Not saved.")
        return
    s["trials_ms"][int(trial_no) - 1] = result["elapsed_ms"]
    dm.save_data(data)
    print("Trial saved.")


def view_solid(data):
    slot = choose_slot(data)
    if slot is None:
        return
    s = dm.get_solid(data, slot)
    print()
    for k, v in s.items():
        print(f"  {k}: {v}")


def generate_table(data):
    fmt = input("Export as (xlsx/csv) [xlsx]: ").strip().lower() or "xlsx"
    path = f"rolling_bodies_results.{fmt}"
    path, df = dm.export_table(data, path)
    print()
    print(df.to_string(index=False))
    print(f"\nSaved to: {os.path.abspath(path)}")


MENU = """
========================================
 Rolling Body Moment of Inertia Analyzer
========================================
1. Configure a solid (name/mass/radius/theta/distance/I_theory)
2. Record a trial (video -> automatic time measurement)
3. View a solid's data
4. Generate & export comparison table (all 8 solids)
5. Exit
"""


def main():
    data = dm.load_data()
    while True:
        print(MENU)
        choice = input("Choose an option (1-5): ").strip()
        if choice == "1":
            configure_solid(data)
        elif choice == "2":
            record_trial(data)
        elif choice == "3":
            view_solid(data)
        elif choice == "4":
            generate_table(data)
        elif choice == "5":
            print("Goodbye!")
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
