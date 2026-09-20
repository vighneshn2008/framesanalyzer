"""
box_tracker.py
--------------
Box-based timing for the rolling-body experiment.

Instead of sampling a colour / clicking a point on the object, you DRAW A BOX
around the object. A CSRT tracker follows the box frame by frame, and the time
is taken at the moment ANY part of the box (its leading edge) reaches the END
gate line.

How the timing works
  * You mark ONE gate: the END gate point E on the motion path.
  * The motion axis runs from the box's starting position towards E,
    u = (E - box_centre) / |E - box_centre|.
  * The END gate is the line through E perpendicular to that axis (so it
    works even when the camera is not perfectly square-on).
  * Timing STARTS automatically when the box's leading edge first moves more
    than `move_px` pixels from its resting position (release threshold) and
    STOPS when the leading edge crosses the END gate.
  * Crossing time is found with sub-frame interpolation, then refined by
    fitting  s(t) = c0 + c1*t + c2*t^2  (constant acceleration on the incline),
    which averages out tracker jitter. The fit is only used if it agrees with
    the raw interpolation; otherwise the raw value is used.

Tip: draw the box TIGHT around the object, at its resting position before
release. The leading edge of the box is what crosses the gate, so the box is
drawn once and tracked for the whole run.
"""

import cv2
import numpy as np

import perspective

MAX_TRACK_WIDTH = 1280      # frames wider than this are down-scaled for tracking only


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _create_tracker():
    """Best available tracker: CSRT (most accurate) -> KCF -> MIL."""
    for factory in ("TrackerCSRT_create", "TrackerKCF_create", "TrackerMIL_create"):
        for ns in (cv2, getattr(cv2, "legacy", None)):
            if ns is not None and hasattr(ns, factory):
                try:
                    return getattr(ns, factory)(), factory.replace("Tracker", "").replace("_create", "")
                except cv2.error:
                    continue
    raise RuntimeError("No OpenCV tracker available. Install: pip install opencv-contrib-python")


def gate_axis(box_xywh, end_point, H=None):
    """Motion axis for a single END gate, derived from the drawn box.

    The box is assumed to be at rest where it was first drawn. The motion axis
    runs from the box's centroid towards the END gate point; the END gate is
    the line through that point perpendicular to the axis.

    If a perspective homography `H` (image -> table plane) is given, the box's
    four corners, its centroid and the gate point are all rectified first, so
    the axis and the perpendicular gate line are computed in the DISTORTION-FREE
    plane of the table. This keeps the END gate physically perpendicular to the
    motion even when the camera is pointed at an angle.

    Returns (origin, u, level):
      origin = box centroid (image pixels, or rectified px if H is given)
      u      = unit vector along the motion axis (same space)
      level  = distance from the origin to the END gate along u (same space)
    """
    x, y, w, h = (float(v) for v in box_xywh)
    if H is None:
        origin = np.array([x + w / 2.0, y + h / 2.0], dtype=float)
        e = np.array(end_point, dtype=float)
    else:
        corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                           dtype=float)
        rect = perspective.transform_points(H, corners)
        origin = rect.mean(axis=0)
        e = perspective.transform_point(H, end_point)
    level = float(np.linalg.norm(e - origin))
    if level < 1e-6:
        raise ValueError("END gate point is on top of the box - move it clear of the object.")
    return origin, (e - origin) / level, level


def gate_line_points(point, u, H=None, length=4000.0):
    """
    Two image-space endpoints of the END gate line through `point`,
    perpendicular to the motion axis `u`.

    With `H` (perspective homography), the perpendicular direction is taken in
    the rectified table plane and the resulting line is mapped back to the
    image - so the drawn line is the TRUE physical gate line even when the
    camera is angled. Without H, the gate line is simply the pixel-space
    perpendicular (historical behaviour).
    """
    p = np.array(point, dtype=float)
    if H is None:
        n = np.array([-u[1], u[0]])
        return p + n * length, p - n * length
    er = perspective.transform_point(H, point)      # gate point in table plane
    n = np.array([-u[1], u[0]])                     # rectified perpendicular
    p1 = perspective.inverse_transform_points(H, [er + n * length])[0]
    p2 = perspective.inverse_transform_points(H, [er - n * length])[0]
    return p1, p2


def _draw_gate_line(img, point, u, color, H=None, length=4000.0):
    p1, p2 = gate_line_points(point, u, H=H, length=length)
    cv2.line(img, tuple(int(v) for v in p1), tuple(int(v) for v in p2), color, 1, cv2.LINE_AA)


# --------------------------------------------------------------------------
# Interactive steps
# --------------------------------------------------------------------------
def pick_frame(video_path):
    """Scrub through the video and choose the frame to start tracking from.
    (Object visible and still, BEFORE it starts moving.)
    Returns (frame_index, frame) or None."""
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release()
        return None
    win = "Pick start frame | slider or a/d = +-1, w/s = +-10 | Enter = OK | q = abort"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("frame", win, 0, max(n - 1, 1), lambda v: None)

    last_idx, frame, result = -1, None, None
    while True:
        idx = cv2.getTrackbarPos("frame", win)
        if idx != last_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, f = cap.read()
            if ret:
                frame = f
            last_idx = idx
        if frame is not None:
            disp = frame.copy()
            cv2.putText(disp, f"frame {idx}/{n - 1}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow(win, disp)
        key = cv2.waitKey(30) & 0xFF
        step = {ord('a'): -1, ord('d'): 1, ord('s'): -10, ord('w'): 10}.get(key)
        if step:
            cv2.setTrackbarPos("frame", win, int(np.clip(idx + step, 0, n - 1)))
        elif key in (13, 32) and frame is not None:
            result = (idx, frame.copy())
            break
        elif key in (ord('q'), 27):
            break
    cap.release()
    cv2.destroyAllWindows()
    return result


def select_box(frame):
    """Draw a box around the object. Returns (x, y, w, h) or None."""
    r = cv2.selectROI("Draw a TIGHT box around the object, then Enter/Space (c = cancel)",
                      frame, showCrosshair=False, fromCenter=False)
    cv2.destroyAllWindows()
    if r[2] < 3 or r[3] < 3:
        return None
    return tuple(int(v) for v in r)


def mark_gates(frame, box=None, H=None):
    """Click the END gate point, press 'c' to confirm, 'r' to reset, 'q' to abort.
    `H` optionally supplies the perspective homography used to draw the true
    (rectified) perpendicular gate line.
    Returns [(x, y)] or None."""
    pts = []
    win = "Click the END gate | c = confirm | r = reset | q = abort"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 1:
            pts.append((x, y))

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    result = None
    while True:
        disp = frame.copy()
        if box is not None:
            x, y, w, h = box
            cv2.rectangle(disp, (x, y), (x + w, y + h), (255, 200, 0), 2)
        for i, p in enumerate(pts):
            cv2.circle(disp, p, 4, (0, 255, 0), -1)
            cv2.putText(disp, "END", (p[0] + 10, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if len(pts) == 1 and box is not None:
            try:
                _, u, _ = gate_axis(box, pts[0], H=H)
                _draw_gate_line(disp, pts[0], u, (0, 255, 0), H=H)
            except ValueError:
                pass
        cv2.imshow(win, disp)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('r'):
            pts.clear()
        elif key == ord('c') and len(pts) == 1:
            result = list(pts)
            break
        elif key in (ord('q'), 27):
            break
    cv2.destroyAllWindows()
    return result


# --------------------------------------------------------------------------
# Tracking
# --------------------------------------------------------------------------
def track_box(video_path, start_idx, box, gates=None, show=True,
              frame_cb=None, stop_event=None, H=None, fps=None):
    """Track `box` (x, y, w, h) from frame `start_idx` to the end of the video.

    `H` optionally supplies the image -> table-plane homography used to draw
    the perspective-corrected gate line in the review window.

    `fps` optionally overrides the frame rate read from the video container
    (use this when the container FPS is wrong; None = use the video's own FPS).

    Returns dict: times_ms (N,), boxes (N,4) in ORIGINAL pixel coordinates,
                  fps, lost (tracker lost the object early), tracker (name)
    or None if the user aborted (preview 'q' or `stop_event` set).

    frame_cb(frame, frame_idx, box_xywh) is called for every tracked frame so a
    GUI can draw an embedded live preview (use show=False then).
    """
    cap = cv2.VideoCapture(video_path)
    if fps is None or fps <= 1:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not fps or fps <= 1:
        fps = 30.0

    frame = None
    for _ in range(start_idx + 1):          # sequential read = exact frame numbering
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError("Could not reach the chosen start frame.")

    scale = min(1.0, MAX_TRACK_WIDTH / frame.shape[1])

    def prep(f):
        return cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else f

    tracker, name = _create_tracker()
    tracker.init(prep(frame), tuple(int(round(v * scale)) for v in box))

    S, u, L = None, None, None
    if gates is not None:
        try:
            S, u, L = gate_axis(box, gates[0], H=H)
        except ValueError:
            pass

    idx = start_idx
    times = [idx / fps * 1000.0]
    boxes = [np.array(box, dtype=float)]
    lost = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        if stop_event is not None and stop_event.is_set():
            cap.release()
            return None
        ok, bb = tracker.update(prep(frame))
        if not ok:
            lost = True
            break
        b = np.array(bb, dtype=float) / scale
        boxes.append(b)
        times.append(idx / fps * 1000.0)
        if frame_cb is not None:
            frame_cb(frame, idx, b)

        if show:
            x, y, w, h = [int(round(v)) for v in b]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
            if gates is not None and u is not None:
                _draw_gate_line(frame, gates[0], u, (0, 255, 0), H=H)
            cv2.putText(frame, f"{name}  frame {idx}  (q = abort)", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return None

    cap.release()
    if show:
        cv2.destroyAllWindows()
    return {"times_ms": np.array(times), "boxes": np.array(boxes),
            "fps": fps, "lost": lost, "tracker": name}


# --------------------------------------------------------------------------
# Crossing-time computation
# --------------------------------------------------------------------------
def leading_edge(boxes, S, u, H=None):
    """Largest projection of the 4 box corners on the motion axis (rectified
    distance from S). With `H`, each corner is rectified through the table-plane
    homography first, so the projection is taken in the distortion-free plane."""
    x, y, w, h = boxes.T
    cx = np.stack([x, x + w, x, x + w], axis=1)
    cy = np.stack([y, y, y + h, y + h], axis=1)
    if H is not None:
        corners = np.stack([cx, cy], axis=-1)               # (N, 4, 2) image corners
        r = perspective.transform_points(H, corners.reshape(-1, 2)).reshape(-1, 4, 2)
        return ((r[..., 0] - S[0]) * u[0] + (r[..., 1] - S[1]) * u[1]).max(axis=1)
    return ((cx - S[0]) * u[0] + (cy - S[1]) * u[1]).max(axis=1)


def box_phase(box, at_rest_box, end_gate, move_px=2.0, H=None):
    """Phase of a tracked box relative to the timing:
    0 = still at rest (not released yet),
    1 = released (leading edge moved past the release threshold),
    2 = past the END gate (leading edge crossed the gate line).

    `H` rotates/rectifies the geometry to the table plane first, and converts
    the pixel `move_px` threshold into the rectified plane using the local
    pixel scale at the resting box."""
    S, u, level = gate_axis(at_rest_box, end_gate, H=H)
    lead = float(leading_edge(np.array([box], dtype=float), S, u, H=H)[0])
    if lead >= level:
        return 2
    mpp = perspective.local_mpp(H, at_rest_box[0] + at_rest_box[2] / 2.0,
                                at_rest_box[1] + at_rest_box[3] / 2.0)
    start_level = float(leading_edge(np.array([at_rest_box], dtype=float), S, u, H=H)[0])
    start_level += max(1.0, float(move_px)) * mpp
    if lead >= start_level:
        return 1
    return 0


def _first_crossing(t, s, level, sustain=3):
    """First time s reaches `level` (and stays there for `sustain` frames),
    linearly interpolated between frames. None if it never happens or if
    s is already past the level in the very first frame."""
    for i in range(len(s)):
        if s[i] >= level and np.all(s[i:i + sustain] >= level):
            if i == 0:
                return None
            s0, s1 = s[i - 1], s[i]
            return t[i - 1] + (level - s0) / (s1 - s0) * (t[i] - t[i - 1])
    return None


def _quad_crossing(t, s, level, near_ms):
    """Crossing of `level` from a quadratic fit; the root nearest `near_ms`."""
    t0 = t[0]
    tau = (t - t0) / 1000.0
    c2, c1, c0 = np.polyfit(tau, s, 2)
    rms = float(np.sqrt(np.mean((np.polyval([c2, c1, c0], tau) - s) ** 2)))
    if abs(c2) < 1e-9:
        roots = np.array([(level - c0) / c1]) if abs(c1) > 1e-9 else np.array([])
    else:
        roots = np.roots([c2, c1, c0 - level])
        roots = roots[np.isreal(roots)].real
    if len(roots) == 0:
        return None, rms
    root_ms = t0 + roots * 1000.0
    return float(root_ms[np.argmin(np.abs(root_ms - near_ms))]), rms


def compute_crossing(track, end_gate, move_px=2.0, H=None):
    """Time (ms) between the box first moving (release threshold) and its
    leading edge crossing the END gate. `end_gate` is the single END point.
    `H` optionally supplies the image -> table-plane homography so the gate
    line and all distances are computed in the perspective-free plane of the
    table.
    Returns dict or None if the END gate was not clearly crossed."""
    t = track["times_ms"]
    fps = track["fps"]
    boxes = track["boxes"]
    frame_ms = 1000.0 / fps

    S, u, level = gate_axis(boxes[0], end_gate, H=H)
    s = leading_edge(boxes, S, u, H=H)

    # motion onset: the object leaves its resting position
    mpp = perspective.local_mpp(H, boxes[0][0] + boxes[0][2] / 2.0,
                                boxes[0][1] + boxes[0][3] / 2.0)
    thr = max(1.0, float(move_px)) * mpp
    start_level = s[0] + thr

    t_end_raw = _first_crossing(t, s, level)
    if t_end_raw is None:
        return None

    t_start_raw = _first_crossing(t, s, start_level)
    if t_start_raw is None:
        return None  # the box never left its resting spot

    raw_elapsed = t_end_raw - t_start_raw
    if raw_elapsed <= 0:
        return None

    # quadratic refinement over the accelerating part of the run
    onset = max(int(np.searchsorted(t, t_start_raw)) - 2, 0)
    end_idx = int(np.searchsorted(t, t_end_raw))
    hi = min(len(t), end_idx + 3)
    method, fit_rms = "interpolated", None
    t_start, t_end = t_start_raw, t_end_raw
    if hi - onset >= 6:
        ts, ss = t[onset:hi], s[onset:hi]
        e_fit, rms_e = _quad_crossing(ts, ss, level, t_end_raw)
        s_fit, rms_s = _quad_crossing(ts, ss, start_level, t_start_raw)
        fit_rms = rms_e
        tol_px = max(1.5, 0.005 * level) * mpp
        if (e_fit is not None and s_fit is not None and rms_e < tol_px
                and abs(e_fit - t_end_raw) < 2 * frame_ms
                and abs(s_fit - t_start_raw) < 2 * frame_ms
                and e_fit > s_fit):
            t_start, t_end, method = s_fit, e_fit, "quadratic fit (a = const)"

    return {
        "elapsed_ms": round(t_end - t_start, 1),
        "start_time_ms": round(t_start, 1),
        "end_time_ms": round(t_end, 1),
        "raw_elapsed_ms": round(raw_elapsed, 1),
        "method": method,
        "start_mode": "motion onset (release threshold)",
        "fit_rms_px": None if fit_rms is None else round(fit_rms, 2),
        "gate_px": round(level, 1),
        "release_thr_px": round(thr, 1),
    }
