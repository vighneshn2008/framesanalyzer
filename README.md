# Rolling Body Moment of Inertia Analyzer

A Python/OpenCV tool to experimentally verify

```
a = g sin(θ) / (1 + I / (m R²))
```

by video-tracking objects rolling down an inclined table, timing them between
two marked points, and computing the experimental moment of inertia to
compare against the theoretical value.

## 1. Setup

```bash
pip install -r requirements.txt
python main.py          # GUI (recommended)
python main.py --cli    # text menu
```

The **box tracker** needs `opencv-contrib-python` (CSRT tracker); Pillow gives a faster GUI preview.

Works with Python 3.8+. Tested with OpenCV 4.x.

## 2. Physical setup tips (important for accuracy)

- **Camera position:** mount your phone/webcam on a tripod, aimed
  **perpendicular to the plane of motion** (i.e. looking straight at the side
  of the incline, not at an angle) to avoid perspective/parallax error in the
  timing.
- **Frame rate:** use the highest FPS your camera supports (many phones do
  120–240 fps "slow-mo"). The tool interpolates *between* frames for
  sub-frame accuracy, but a higher native FPS still gives cleaner tracking.
- **Lighting/contrast:** even, diffuse lighting with no moving shadows. To
  use "color" tracking, pick a 3D-print filament color that contrasts with
  the table.
- **Gates:** physically mark (e.g., tape) the END point/line on the table
  before recording, and measure the real distance from the object's release
  point to it with a ruler — you'll enter this distance (in meters) when
  configuring each solid.
- **Release:** let the object start from rest at a fixed backward position
  (the box tracker detects the exact moment it starts moving).

## 3. Using the tool

Run `python main.py`. You get a menu:

```
1. Configure a solid   -> name, mass (kg), radius (m), incline angle θ (deg),
                           gate distance (m), theoretical I (kg·m²)
2. Record a trial      -> pick a video (file or live webcam), mark the END
                            point, sample the object color (or draw a box) and
                            the tool reports the crossing time in milliseconds
3. View a solid's data
4. Generate & export comparison table (all 8 solids) -> CSV or XLSX
5. Exit
```

There are **8 solid slots** (as required for the experiment), each holding
**up to 3 trial times**. Data auto-saves to `rolling_bodies_data.json` in the
working directory, so you can quit and resume across sessions.

### Recording a trial — what happens under the hood

1. You supply a video (a pre-recorded file, or record fresh from a webcam).
2. You mark the **END** point (where timing stops) on the video.
3. You pick a tracking mode:
   - **Color mode** — click once on the object to sample its color; works on
     any surface with good contrast.
   - **Box tracker** — draw a box around the object; a CSRT tracker follows
     it frame by frame (most accurate).
4. The tool tracks the object, detects the exact moment it **starts moving**
   (release threshold, in pixels) and finds the **exact (sub-frame-interpolated)
   time** the object crosses the END gate. The difference is your crossing
   time, reported in milliseconds.
5. You choose to save it as Trial 1, 2, or 3 for that solid.

### Why interpolation instead of just counting frames?

Counting frames only gives you timing resolution equal to `1/fps` (e.g. 33 ms
at 30 fps). Instead, this tool tracks the object's continuous position each
frame and **linearly interpolates** between the two frames just before and
just after each gate is crossed to estimate the precise fractional-frame
crossing time. In a synthetic test (30 fps, ballistic motion), this reduced
timing error from ~33 ms down to **~0.1 ms**.

### The physics (`calculations.py`)

Given the average crossing time `t` over distance `d` (object starts from
rest):

```
a_exp = 2d / t²                                  (kinematics)
I_exp = m R² ( g sin(θ) / a_exp − 1 )             (from the incline formula)
```

`data_manager.py` computes this for every configured solid and produces a
table with trial times, average time, `a_exp`, `I_exp`, your entered
`I_theory`, and the **% error** between them — ready to export to Excel for
your report.

### Box tracker mode (in the GUI)

In *Record a trial -> step 3* pick **Box tracker**, then **Set up box & END gate...**:
choose the start frame, drag a tight box around the object at rest, click the END
gate on the path. A CSRT tracker follows the box; timing starts automatically when
the box's *leading edge* first moves more than the release threshold from its resting
spot, and stops when the leading edge crosses the END gate (perpendicular to the
release->END axis, so a slightly angled camera is fine), with sub-frame interpolation
refined by a constant-acceleration quadratic fit.
The review player shows the box, the gate line and a TIMING/before/after state.

### Perspective correction (optional)

If the camera can't be placed perfectly perpendicular to the table plane
(e.g. a phone on a tripod shooting at a slight angle), the END gate line and
distances along the incline are distorted. Before configuring the box, use
**Perspective -> Setup...**: click the 4 corners of a *measurable* rectangle on
the table plane (the table top, or a ruler/paper laid along the incline) in
order top-left -> top-right -> bottom-right -> bottom-left, press `c` to
confirm, and enter its physical width and height in metres. The gate line,
motion axis and all gate levels are then computed in a rectified,
perspective-free plane, and the END gate is drawn perpendicular to the motion
in that plane. **Perspective -> Clear** reverts to plain pixel geometry.

Other additions: outlier warning when saving a trial (>5 % from the other trials),
Std Dev / Spread columns in the results table, bounded preview queue.

## 4. Files

| File | Purpose |
|---|---|
| `main.py` | CLI menu tying everything together |
| `box_tracker.py` | Box-drawn CSRT tracking + leading-edge gate timing |
| `gui.py` | Tkinter GUI (all modes embedded) |
| `tracker.py` | OpenCV point-marking, object tracking, sub-frame crossing-time math |
| `perspective.py` | 4-point homography rectification (optional perspective correction) |
| `calculations.py` | Physics formulas (acceleration, experimental I, % error) |
| `data_manager.py` | JSON persistence for the 8 solids + Excel/CSV table export |
| `requirements.txt` | Python dependencies |

## 5. Suggested 8 solids for the experiment

Since you're 3D-printing objects with 100% infill, good mass-distribution
contrasts to design in Fusion 360 (all same outer radius, so the incline
angle/radius terms are controlled) include: solid cylinder, solid sphere,
thin-walled hollow cylinder (ring), thick-walled hollow cylinder, hollow
sphere (thin shell), a cylinder with a dense core + light shell (or vice
versa), a cylinder with an off-axis internal cavity (breaks I=symmetric
assumption — interesting extra credit), and a solid cone/other shape rolling
on its widest circular face. Enter each one's theoretical `I` (calculated
from its known geometry and PLA/ABS density) into the tool for comparison.
