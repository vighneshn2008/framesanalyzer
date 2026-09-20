"""
friction_slope.py
-----------------
Standalone tool - measures the coefficient of friction (mu) and air drag
for a SQUARE BLOCK of known mass sliding down a flat inclined surface.

It is deliberately separate from the rolling-body app, but reuses the same
tracking / timing algorithm from tracker.py:

  1. Optionally load a video of the block sliding down the slope.
  2. Mark a single END gate point on the first frame (like the main app).
  3. Track the block (color-based) with the same algorithm and
     compute the release->gate time automatically ("you calculate the time").
  4. You then enter the ACTUAL measured time (stopwatch), the block mass,
     incline angle and slope distance; the coefficient of friction (and, if
     you know mu independently, the air drag on the block) is computed.

Physics (block sliding down an incline, released from rest):
    a = 2*d / t^2                                  (kinematics)
    mg sin(theta) - mu*mg cos(theta) - F_drag = m*a   (forces along slope)

    mu (assuming no drag):  mu = (g*sin(theta) - a) / (g*cos(theta))
    F_drag (if mu is known): F_drag = m*(g*sin(theta) - a - mu*g*cos(theta))
    C_d (if area given):     C_d = 2*F_drag / (rho*A*v_avg^2)

Run:  python friction_slope.py
"""

import os
import math

import cv2

import calculations as calc
import tracker


# ---------------------------------------------------------------------------
# Small input helpers
# ---------------------------------------------------------------------------
def _read_float(prompt, default=None, allow_blank=False):
    while True:
        raw = input(prompt).strip()
        if raw == "" and (allow_blank or default is not None):
            return default if raw == "" else None
        try:
            return float(raw)
        except ValueError:
            print("  Please enter a number.")


# ---------------------------------------------------------------------------
# END-gate marking (single point, like the main GUI's step 3)
# ---------------------------------------------------------------------------
def _mark_end_point(frame):
    """Left-click once, then press 'c' to confirm (or 'q' to abort)."""
    pt = {}
    window = "Click on the END gate point (1 click), c=confirm q=abort"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pt["p"] = (x, y)

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        disp = frame.copy()
        if "p" in pt:
            cv2.circle(disp, pt["p"], 8, (0, 0, 255), -1)
            cv2.putText(disp, "END", (pt["p"][0] + 12, pt["p"][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow(window, disp)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("c") and "p" in pt:
            cv2.destroyWindow(window)
            return pt["p"]
        if key == ord("q"):
            cv2.destroyWindow(window)
            return None


# ---------------------------------------------------------------------------
# Core physics
# ---------------------------------------------------------------------------
def compute_friction(mass_kg, theta_deg, distance_m, time_s):
    """Returns a dict with acceleration, mu (no-drag), and a sanity check."""
    if mass_kg <= 0 or distance_m <= 0 or time_s <= 0:
        raise ValueError("mass, distance and time must all be > 0")
    a = calc.experimental_acceleration(distance_m, time_s)
    theta = math.radians(theta_deg)
    mu = (calc.G * math.sin(theta) - a) / (calc.G * math.cos(theta))
    tan_theta = math.tan(theta)
    return {
        "a_exp": a,
        "mu_no_drag": mu,
        "will_slide": mu < tan_theta,
        "tan_theta": tan_theta,
        "mass_kg": mass_kg,
        "theta_deg": theta_deg,
        "distance_m": distance_m,
        "time_s": time_s,
        "v_avg": distance_m / time_s,
    }


def compute_drag_cd(mass_kg, theta_deg, distance_m, time_s, mu_known,
                    area_m2=None, rho=calc.AIR_DENSITY):
    """
    Given a known mu (measured independently), finds the mean drag force
    during the slide and (if the frontal area is given) C_d.
    """
    a = calc.experimental_acceleration(distance_m, time_s)
    theta = math.radians(theta_deg)
    f_drag = mass_kg * (calc.G * math.sin(theta)
                        - a - mu_known * calc.G * math.cos(theta))
    v_avg = distance_m / time_s
    cd = None
    if area_m2 and area_m2 > 0 and v_avg > 0 and f_drag > 0:
        cd = 2.0 * f_drag / (rho * area_m2 * v_avg ** 2)
    return {"f_drag": f_drag, "v_avg": v_avg, "cd": cd}


def predict_time(mu, theta_deg, distance_m):
    """Predicted slide time for a given mu (no drag): d = 0.5*a*t^2."""
    theta = math.radians(theta_deg)
    a = calc.G * (math.sin(theta) - mu * math.cos(theta))
    if a <= 0:
        return None
    return math.sqrt(2.0 * distance_m / a)


# ---------------------------------------------------------------------------
# Main interactive flow
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("FRICTION SLOPE ANALYSER  (square block sliding down a surface)")
    print("=" * 70)
    print("Reuses the same tracking/timing algorithm as the rolling-body app.")
    print()

    # ---- 1. Optional video tracking (same algorithm as the main app) ----
    tracked_time = None
    video = input("Video of the block sliding down (Enter to skip): ").strip().strip('"')
    if video and os.path.exists(video):
        color_range = None
        cap = cv2.VideoCapture(video)
        frame = None
        for _ in range(8):
            ret, frame = cap.read()
            if not ret:
                break
        cap.release()
        if frame is not None:
            color_range = tracker.pick_color(frame)
            if color_range is None:
                print("Aborted color pick.")
                return
        cap = cv2.VideoCapture(video)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("Could not read the video.")
            return
        end_pt = _mark_end_point(frame)
        if end_pt is None:
            print("Aborted.")
            return
        print("Tracking...")
        trajectory, fps = tracker.track_object(
            video, color_range=color_range, show=False)
        res = tracker.compute_crossing_time_ms(trajectory, end_pt)
        if res is not None:
            tracked_time = res["elapsed_ms"] / 1000.0
            print(f"  Tracked release->gate time: {res['elapsed_ms']:.1f} ms "
                  f"(~{fps:.1f} fps)")
        else:
            print("  No clear crossing detected from the video; "
                  "you'll enter the time manually.")
    else:
        print("  No video -> time will be entered manually.")

    # ---- 2. Actual / measured time ----
    print()
    if tracked_time is not None:
        print(f"Computed time from video = {tracked_time * 1000:.1f} ms")
        prompt = ("Enter the ACTUAL measured time in milliseconds "
                  "(Enter to use the tracked one): ")
        t_ms = _read_float(prompt, default=tracked_time * 1000)
    else:
        t_ms = _read_float("Enter the actual measured time in milliseconds: ")
    time_s = t_ms / 1000.0

    # ---- 3. Block / surface geometry ----
    print()
    mass_kg = _read_float("Block mass (kg): ")
    theta_deg = _read_float("Incline angle theta (degrees): ")
    distance_m = _read_float("Slope distance from RELEASE point to END gate (m): ")
    area_m2 = _read_float("Frontal area of the block (m^2, Enter for none): ",
                          default=0.0, allow_blank=True) or 0.0

    # ---- 4. Results ----
    print()
    try:
        f = compute_friction(mass_kg, theta_deg, distance_m, time_s)
    except ValueError as exc:
        print(f"Error: {exc}")
        return

    print("=" * 70)
    print(f"Measured time          : {time_s * 1000:.1f} ms")
    print(f"Acceleration (2d/t^2)  : {f['a_exp']:.4f} m/s^2")
    print(f"Average speed          : {f['v_avg']:.3f} m/s")
    print(f"mu (friction, no drag) : {f['mu_no_drag']:.4f}")
    if not f["will_slide"]:
        print("  WARNING: mu >= tan(theta) - the block would NOT slide; "
              "check the measured time/angle.")
    print()

    mu_known = _read_float("mu known independently (Enter to skip drag): ",
                           default=0.0, allow_blank=True) or 0.0
    if mu_known > 0:
        d = compute_drag_cd(mass_kg, theta_deg, distance_m, time_s,
                            mu_known, area_m2 or None)
        print(f"Mean drag force F_drag : {d['f_drag']:.4f} N")
        if d["cd"] is not None:
            print(f"Drag coefficient C_d   : {d['cd']:.4f}")
        else:
            print("C_d: needs a positive frontal area.")
        print()

    print("Time check (same algorithm, no drag):")
    t_pred = predict_time(f["mu_no_drag"], theta_deg, distance_m)
    if t_pred:
        print(f"  Predicted time with mu={f['mu_no_drag']:.4f}: "
              f"{t_pred * 1000:.1f} ms  (measured was {time_s * 1000:.1f} ms)")
        if t_pred < time_s:
            print("  Measured is slower -> drag/friction losses present.")
    print("=" * 70)


if __name__ == "__main__":
    main()