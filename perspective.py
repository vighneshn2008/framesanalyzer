"""
perspective.py
--------------
4-point homography rectification for the rolling-body experiment.

If the camera is not positioned perfectly perpendicular to the table plane,
lines that are physically straight-and-vertical (the END tape line and the
motion axis) appear slanted, and distances measured in pixels are distorted
along the incline. This module lets the user:

  1. click 4 points that form a rectangle of KNOWN physical size on the
     table plane (e.g. the table top, or a sheet of paper laid along the
     incline), then
  2. enter that rectangle's width and height in metres.

A homography H maps every image pixel onto that physical rectangle, so the
END gate line, the motion axis and all gate distances are afterwards computed
in a rectified, perspective-free plane. When the 4 corners are not marked
(H is None) the code falls back to plain pixel geometry exactly as before.
"""

import cv2
import numpy as np

_corner_labels = ["corner 1 (top-left)", "corner 2 (top-right)",
                  "corner 3 (bottom-right)", "corner 4 (bottom-left)"]


def mark_rectangle(frame, labels=_corner_labels):
    """
    Opens a window showing `frame`. User left-clicks 4 corners of a rectangle
    on the table plane, in order TL -> TR -> BR -> BL. Press 'c' to confirm,
    'r' to reset, 'q' to abort.

    Returns: list of 4 (x, y) image points in clockwise order, or None.
    """
    points = []
    clone = frame.copy()
    win = ("Mark the 4 corners of a rectangle on the table plane "
           "(a rectangle you can measure!).\n"
           "Order: " + " -> ".join(labels) + " | c=confirm r=reset q=abort")

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        disp = clone.copy()
        for i, p in enumerate(points):
            cv2.circle(disp, p, 8, (0, 0, 255), -1)
            cv2.putText(disp, labels[i].split(" ")[0], (p[0] + 12, p[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if len(points) == 4:
            cv2.polylines(disp, [np.array(points, dtype=np.int32)], True,
                          (255, 0, 0), 2)
        cv2.imshow(win, disp)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('r'):
            points = []
        elif key == ord('c') and len(points) == 4:
            cv2.destroyWindow(win)
            return list(points)
        elif key == ord('q'):
            cv2.destroyWindow(win)
            return None


def _order_corners(pts):
    """
    Reorder 4 arbitrary points into clockwise order starting from
    top-left: [TL, TR, BR, BL]. Returns an Nx2 float array.
    """
    pts = np.asarray(pts, dtype=float)
    s = pts.sum(axis=1)              # smallest ~= top-left, largest ~= bottom-right
    d = np.diff(pts, axis=1).ravel()  # x - y ; extremes are TR / BL
    idx = [int(np.argmin(s)), int(np.argmin(d)), int(np.argmax(s)), int(np.argmax(d))]
    return pts[idx]


def homography_from_rect(src_pts, width_m, height_m, px_per_m=1000.0):
    """
    Build the image -> table-plane homography.

    src_pts: 4 image points (any order) marking a rectangle on the table.
    width_m, height_m: the physical size (metres) of that rectangle.
    px_per_m: how many "rectified pixels" represent one metre. This is an
              arbitrary internal scale (it only affects the magnitude of the
              release gate thresholds / levels, which stay consistent within
              the rectified plane). Kept large so pixel-ish defaults such as
              a 2 px release threshold still make sense.

    Returns a 3x3 homography matrix H such that H dot [x, y, 1] ~ (x', y').
    """
    corners = _order_corners(src_pts)
    dst = np.array([
        [0.0, 0.0],
        [width_m * px_per_m, 0.0],
        [width_m * px_per_m, height_m * px_per_m],
        [0.0, height_m * px_per_m],
    ], dtype=np.float32)
    return cv2.getPerspectiveTransform(corners.astype(np.float32), dst)


def transform_points(H, pts):
    """
    Apply homography H to an (N, 2) point array; returns an (N, 2) array of
    rectified coordinates. Points are copied (input is never modified).
    """
    pts = np.asarray(pts, dtype=float)
    if H is None or pts.size == 0:
        return pts.copy()
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    ones = np.column_stack([pts, np.ones(pts.shape[0])])
    out = ones @ H.T
    out = out[:, :2] / out[:, 2:3]
    return out


def transform_point(H, pt):
    """Single-point version of transform_points."""
    return transform_points(H, np.asarray([pt], dtype=float))[0]


def inverse_transform_points(H, pts):
    """Map rectified points back to image pixels (inverse transform)."""
    if H is None:
        return np.asarray(pts, dtype=float).copy()
    return transform_points(np.linalg.inv(H), pts)


def local_mpp(H, x, y):
    """
    Approximate local scale factor near image pixel (x, y): the average number
    of RECTIFIED pixels spanned by a 1-image-pixel step in x and y. With the
    homography's arbitrary px_per_m rectified scale, this converts a threshold
    expressed in image pixels (e.g. a 2 px release motion) into rectified-pixel
    units so it can be compared directly against rectified gate levels.
    Returns 1.0 when H is None (rectified units == image pixels).
    """
    if H is None:
        return 1.0
    p0 = transform_point(H, (x, y))
    px = transform_point(H, (x + 1.0, y))
    py = transform_point(H, (x, y + 1.0))
    return float((np.linalg.norm(px - p0) + np.linalg.norm(py - p0)) * 0.5)