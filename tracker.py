"""
tracker.py
-----------
OpenCV-based module for:

1. Marking reference point(s) (END gate) on a video frame.
2. Tracking a rolling object frame-by-frame with HSV color tracking.
3. Computing sub-frame-accurate crossing times at the gate using
   linear interpolation between frames -> millisecond-level accuracy
   regardless of camera FPS.
"""

import cv2
import numpy as np

import perspective


# ---------------------------------------------------------------------------
# STEP 1: Mark points (END gate) on the first frame of a video
# ---------------------------------------------------------------------------
def mark_points(frame, n_points=2):
    """
    Opens a window showing `frame`. User left-clicks to place `n_points` points.
    Press 'c' to confirm once all points are placed, 'r' to reset, 'q' to abort.

    Returns: [(x1, y1), ...] in pixel coordinates, or None if aborted.
    """
    points = []
    clone = frame.copy()
    labels = ["START", "END"]
    window = (f"Mark {' and '.join(labels[:n_points])} gate "
              f"({n_points} click{'s' if n_points != 1 else ''}), "
              f"then c=confirm r=reset q=abort")

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < n_points:
            points.append((x, y))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        disp = clone.copy()
        for i, p in enumerate(points):
            cv2.circle(disp, p, 8, (0, 0, 255), -1)
            cv2.putText(disp, labels[i], (p[0] + 12, p[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if len(points) == 2:
            cv2.line(disp, points[0], points[1], (255, 0, 0), 2)

        cv2.imshow(window, disp)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('r'):
            points = []
        elif key == ord('c') and len(points) == n_points:
            cv2.destroyWindow(window)
            return points
        elif key == ord('q'):
            cv2.destroyWindow(window)
            return None


# ---------------------------------------------------------------------------
# STEP 2: Sample the object's color (for HSV color-based tracking mode)
# ---------------------------------------------------------------------------
def color_range_from_patch(patch):
    """
    Given an Nx3 array of HSV pixels (from clicking on the object), return
    (lower_hsv, upper_hsv) uint8 arrays usable with cv2.inRange.
    """
    h_med, s_med, v_med = np.median(patch, axis=0)
    lower = np.array([max(0, h_med - 12), max(30, s_med - 60), max(30, v_med - 60)],
                     dtype=np.uint8)
    upper = np.array([min(179, h_med + 12), 255, 255], dtype=np.uint8)
    return lower, upper


def pick_color(frame, box=6):
    """
    Lets the user click on the moving object to sample its color.
    Returns (lower_hsv, upper_hsv) numpy arrays usable with cv2.inRange,
    or None if aborted.
    """
    sample = {}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            sample['pt'] = (x, y)

    window = "Click on the OBJECT to sample its color, then press c (q=abort)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    while True:
        disp = frame.copy()
        if 'pt' in sample:
            cv2.circle(disp, sample['pt'], box, (0, 255, 0), 2)
        cv2.imshow(window, disp)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('c') and 'pt' in sample:
            cv2.destroyWindow(window)
            break
        if key == ord('q'):
            cv2.destroyWindow(window)
            return None

    x, y = sample['pt']
    h, w = hsv_frame.shape[:2]
    x0, x1 = max(0, x - box), min(w, x + box)
    y0, y1 = max(0, y - box), min(h, y + box)
    patch = hsv_frame[y0:y1, x0:x1].reshape(-1, 3)
    return color_range_from_patch(patch)


# ---------------------------------------------------------------------------
# STEP 3: Track the object across the whole video -> list of (t, cx, cy, idx)
# ---------------------------------------------------------------------------
def track_object(video_path, color_range=None, min_area=80,
                 bg_learn_frames=20, show=True, frame_cb=None, fps=None):
    """
    Tracks a specific HSV color blob through the whole video.

    Returns: (trajectory, fps)
        trajectory = list of (timestamp_sec, centroid_x, centroid_y, frame_index)

    `fps` optionally overrides the frame rate read from the video container
    (use this when the container FPS is wrong, e.g. slow-motion footage
    re-encoded without the true capture rate; None = use the video's own FPS).

    If `frame_cb` is given, it is called for EVERY frame as
    `frame_cb(frame, t_sec, centroid, frame_idx)` (a copy-free view of the
    frame; copy it if you keep it). This lets a GUI render a live preview
    without the OpenCV window (pass show=False).
    """
    if color_range is None:
        raise ValueError("track_object requires color_range (HSV color tracking).")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    if fps is None or fps <= 1:
        fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30.0  # sane fallback if the container doesn't report fps

    trajectory = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, color_range[0], color_range[1])
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centroid = None
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) >= min_area and frame_idx > bg_learn_frames:
                M = cv2.moments(largest)
                if M["m00"] != 0:
                    centroid = (M["m10"] / M["m00"], M["m01"] / M["m00"])

        t_sec = frame_idx / fps
        if centroid is not None:
            trajectory.append((t_sec, centroid[0], centroid[1], frame_idx))

        if frame_cb is not None:
            frame_cb(frame, t_sec, centroid, frame_idx)

        if show:
            disp = frame.copy()
            if centroid is not None:
                cv2.circle(disp, (int(centroid[0]), int(centroid[1])), 6, (0, 255, 0), -1)
            cv2.imshow("Tracking preview (press q to stop showing)", disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                show = False
                cv2.destroyWindow("Tracking preview (press q to stop showing)")

        frame_idx += 1

    cap.release()
    if show:
        cv2.destroyAllWindows()

    return trajectory, fps


# ---------------------------------------------------------------------------
# STEP 4: Compute crossing time from release to the END gate
# ---------------------------------------------------------------------------
def _fit_crossing(times, values, straddle_i, target, half_win=3, deg=2):
    """
    Estimates the exact time `values(t)` crosses `target` by least-squares
    fitting a low-degree polynomial to the ~2*half_win+2 samples straddling the
    gate. Kept generic over `times`/`values` so it works either on the raw
    centroid-X (historical) or on a perspective-rectified along-axis projection.

    Returns the crossing time in seconds, or None on failure (caller falls back
    to linear interpolation).
    """
    n = len(times)
    lo = max(0, straddle_i - half_win)
    hi = min(n, straddle_i + 2 + half_win)
    ts, xs = times[lo:hi], values[lo:hi]

    if len(ts) >= 4:
        try:
            base = ts[len(ts) // 2]                      # keep polyfit well-conditioned
            rt = np.array([t - base for t in ts])
            rx = np.array([x - target for x in xs])

            # Fit the smooth model, but BRACKET the crossing with the two raw
            # frames that actually straddle the gate, then interpolate the
            # fitted curve between them. Smoothing (jitter rejection) without
            # ever losing the guaranteed bracketing.
            p = np.polyfit(rt, rx, deg=min(deg, len(ts) - 2))
            off = straddle_i - lo
            r0, r1 = float(rt[off]), float(rt[off + 1])
            y0, y1 = float(np.polyval(p, r0)), float(np.polyval(p, r1))

            def interp(a, ya, b, yb):
                return float(base) + (a - ya * (b - a) / (yb - ya))

            if y0 == 0:
                return float(base) + r0
            if y1 == 0:
                return float(base) + r1
            if y0 * y1 < 0:
                return interp(r0, y0, r1, y1)
            # fitted curve doesn't bracket (rare, heavy noise) -> raw interpolation
            return interp(r0, rx[off], r1, rx[off + 1])
        except Exception:
            pass
    return None


def _fit_crossing_time(trajectory, straddle_i, target_x, half_win=3, deg=2):
    """Wrapper: _fit_crossing on the centroid-X of a trajectory."""
    return _fit_crossing(
        [pt[0] for pt in trajectory],
        [pt[1] for pt in trajectory],
        straddle_i, target_x, half_win=half_win, deg=deg)


def compute_crossing_time_ms(trajectory, gate_points, move_px=2.0,
                             fit_half_win=3, fit_deg=2, H=None):
    """
    trajectory: list of (t_sec, cx, cy, frame_idx) from track_object()
    gate_points: the single END gate - a point (x, y) or a list/tuple with
    the END gate as its last element. No START gate is used at all; the gate
    is where the crossing time is measured TO.
    move_px: how many pixels the blob must leave its resting spot before it
    counts as "released" (release-detection threshold).
    H: optional image -> table-plane homography. If given, every centroid and
    the gate point are rectified first and the same logic runs in the
    distortion-free table plane (crossing is then detected against the
    perpendicular gate line, not just against the image X axis).

    start_time = the moment the object first starts moving from its resting
    position (wherever it was left). end_time = the moment its centroid crosses
    the END gate. Both times are sub-frame accurate, independent of camera FPS:
    the gate crossing is least-squares fit to the straddling frames (averaging
    out per-frame centroid jitter) with linear interpolation as the fallback.

    Returns dict {start_time_ms, end_time_ms, elapsed_ms}, or None if the
    object never clearly crossed the END gate.
    """
    if len(trajectory) < 2:
        return None
    if isinstance(gate_points, tuple):
        end_x = gate_points[0]
    else:
        end_x = gate_points[-1][0]

    if H is not None:
        return _crossing_rectified(trajectory, end_x, gate_points, move_px,
                                   fit_half_win, fit_deg, H)

    # Resting position = where the blob is first detected (it starts at rest).
    init_x, init_y = float(trajectory[0][1]), float(trajectory[0][2])

    t_start = trajectory[0][0]
    prev_d, prev_t = 0.0, trajectory[0][0]
    for (t, cx, cy, fidx) in trajectory:
        d = float(np.hypot(cx - init_x, cy - init_y))
        if d >= move_px and prev_d < move_px:
            frac = (move_px - prev_d) / (d - prev_d)
            t_start = prev_t + frac * (t - prev_t)
            break
        prev_d, prev_t = d, t

    # END gate crossing on centroid X only.
    def find_crossing(target_x):
        for i in range(len(trajectory) - 1):
            t0, x0 = trajectory[i][0], trajectory[i][1]
            t1, x1 = trajectory[i + 1][0], trajectory[i + 1][1]
            if x0 == target_x:
                return t0
            if (x0 - target_x) * (x1 - target_x) < 0:  # sign change -> crossed here
                t_fit = _fit_crossing_time(trajectory, i, target_x,
                                           half_win=fit_half_win, deg=fit_deg)
                if t_fit is not None:
                    return t_fit
                frac = (target_x - x0) / (x1 - x0)   # fallback: linear interpolation
                return t0 + frac * (t1 - t0)
        return None

    t_end = find_crossing(end_x)

    if t_end is None or t_end <= t_start:
        return None

    return {
        "start_time_ms": round(t_start * 1000, 3),
        "end_time_ms": round(t_end * 1000, 3),
        "elapsed_ms": round((t_end - t_start) * 1000, 3),
    }


def _crossing_rectified(trajectory, end_x, gate_points, move_px,
                        fit_half_win, fit_deg, H):
    """
    Perspective-corrected variant of compute_crossing_time_ms. All centroids
    and the END gate point are rectified into the table plane first; the motion
    axis runs from the rectified resting centroid to the rectified gate, and
    crossing is detected when the along-axis projection reaches the gate level
    (i.e. against the physically-perpendicular gate line, not the image X).
    """
    pts = np.array([[cx, cy] for (_t, cx, cy, _f) in trajectory], dtype=float)
    rect = perspective.transform_points(H, pts)
    gate_rect = perspective.transform_point(H, gate_points[-1])
    mpp = perspective.local_mpp(H, pts[0][0], pts[0][1])

    origin = rect[0]
    ax = gate_rect - origin
    level = float(np.linalg.norm(ax))
    if level < 1e-6:
        return None
    ax = ax / level
    proj = (rect - origin) @ ax

    # release: distance (rectified, physical) from the resting spot
    thr = max(1.0, float(move_px)) * mpp
    t_start = trajectory[0][0]
    prev_d, prev_t = 0.0, trajectory[0][0]
    for i, (t, _cx, _cy, _f) in enumerate(trajectory):
        d = float(np.linalg.norm(rect[i] - origin))
        if d >= thr and prev_d < thr:
            frac = (thr - prev_d) / (d - prev_d)
            t_start = prev_t + frac * (t - prev_t)
            break
        prev_d, prev_t = d, t

    # crossing of the along-axis projection at the gate level
    times = [pt[0] for pt in trajectory]

    def find_crossing(target):
        for i in range(len(proj) - 1):
            t0, q0 = times[i], proj[i]
            t1, q1 = times[i + 1], proj[i + 1]
            if q0 == target:
                return t0
            if (q0 - target) * (q1 - target) < 0:
                t_fit = _fit_crossing(times, proj, i, target,
                                      half_win=fit_half_win, deg=fit_deg)
                if t_fit is not None:
                    return t_fit
                frac = (target - q0) / (q1 - q0)   # fallback: linear interpolation
                return t0 + frac * (t1 - t0)
        return None

    t_end = find_crossing(level)

    if t_end is None or t_end <= t_start:
        return None

    return {
        "start_time_ms": round(t_start * 1000, 3),
        "end_time_ms": round(t_end * 1000, 3),
        "elapsed_ms": round((t_end - t_start) * 1000, 3),
    }
