"""
friction_gui.py
---------------
Companion GUI for measuring the coefficient of friction (mu) and air drag
of a square block of known mass sliding down an incline.

It reuses tracker.py (the same tracking / release-detection / END-gate
timing algorithm as the rolling-body app) and the physics helpers from
calculations.py / friction_slope.py, but is kept as a separate module.

Runnable standalone:
    python friction_gui.py

or opened from the "Configure solids" (home) screen of the main GUI via
the "Friction-slope analyser" button.
"""

import os
import queue
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, messagebox, filedialog

import cv2

try:
    import PIL.Image
    import PIL.ImageTk
    HAVE_PIL = True
except Exception:  # noqa: BLE001 - PIL optional; fall back to PPM
    HAVE_PIL = False

import calculations as calc
import tracker
import friction_slope as fs

MAX_W = 1000

PALETTE = {
    "bg": "#0E1428",
    "surface": "#1B2340",
    "surface_alt": "#151B32",
    "row_alt": "#232C4D",
    "text": "#E6EAF5",
    "text_muted": "#9AA6C7",
    "border": "#2F3B63",
    "primary": "#5EA0FF",
    "primary_dark": "#10162C",
    "accent": "#FFC24B",
    "header_grad_top": "#1B2A5B",
    "header_grad_bot": "#2E4A9C",
}


class FrictionSlopeApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Friction Slope Analyser")

        self.video_path = None
        self.gate_points = None
        self.color_range = None
        self.ui_mode = "idle"   # idle | mark | pick | track
        self.mark_mode = None
        self.pick_mode = None
        self.cv2_busy = False
        self.pending_callback = None
        self.cv2_q = queue.Queue(maxsize=2)
        self.cam = None
        self._track_first = None
        self._track_released = False

        base_family = "Segoe UI"
        if base_family not in tkfont.families():
            base_family = "Helvetica"
        self.font_base = tkfont.Font(family=base_family, size=10)
        self.font_bold = tkfont.Font(family=base_family, size=10, weight="bold")
        self.font_h1 = tkfont.Font(family=base_family, size=15, weight="bold")
        self.font_h2 = tkfont.Font(family=base_family, size=10, weight="bold")
        self.font_small = tkfont.Font(family=base_family, size=9)
        self.option_add("*Font", self.font_base)

        p = PALETTE
        style = ttk.Style(self.master)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", background=p["bg"], foreground=p["text"])
        style.configure("TFrame", background=p["bg"])
        style.configure("TLabel", background=p["bg"], foreground=p["text"])
        style.configure("Muted.TLabel", background=p["bg"],
                        foreground=p["text_muted"], font=self.font_small)
        style.configure("Primary.TButton", background=p["primary"])
        style.configure("Accent.TButton", background=p["primary"],
                        foreground="#FFFFFF")
        style.configure("Danger.TButton", background="#B3322F")
        style.configure("TLabelframe", background=p["surface"],
                        bordercolor=p["border"], relief="solid", borderwidth=1)
        style.configure("TLabelframe.Label", background=p["surface"],
                        foreground=p["accent"], font=self.font_h2)
        style.configure("TLabelframe.TLabel", background=p["surface"])
        style.configure("TEntry", fieldbackground="#FFFFFF",
                        foreground=p["primary_dark"])
        style.configure("TSpinbox", fieldbackground="#FFFFFF",
                        foreground=p["primary_dark"], arrowsize=14)
        style.configure("TCombobox", fieldbackground="#FFFFFF",
                        foreground=p["primary_dark"])

        self._build_ui()
        self.after(50, self._poll_queue)

    # ------------------------------------------------------------------
    def _build_ui(self):
        p = PALETTE

        header = tk.Frame(self.master, bg=p["header_grad_top"])
        header.pack(fill="x", side="top")
        inner = tk.Frame(header, bg=p["header_grad_top"])
        inner.pack(fill="x", padx=18, pady=(12, 12))
        tk.Label(inner, text="Friction Slope Analyser",
                 bg=p["header_grad_top"], fg="#FFFFFF",
                 font=self.font_h1, anchor="w").pack(anchor="w")
        tk.Label(inner,
                 text="mu / air drag of a sliding block - same tracking "
                      "algorithm as the rolling-body app",
                 bg=p["header_grad_top"], fg=p["accent"],
                 font=self.font_bold, anchor="w", wraplength=900).pack(
            anchor="w", pady=(4, 0))
        tk.Frame(self.master, bg=p["accent"], height=3).pack(fill="x", side="top")

        body = ttk.Frame(self.master, padding=(12, 10, 12, 10))
        body.pack(fill="both", expand=True)

        left = ttk.Frame(body)
        left.pack(side="left", fill="y", padx=(0, 12))
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        # ---- left: inputs ----
        g1 = ttk.LabelFrame(left, text="1. Block & surface", padding=8)
        g1.pack(fill="x", pady=(0, 8))
        self.param_vars = {
            "mass_kg": tk.StringVar(),
            "theta_deg": tk.StringVar(),
            "distance_m": tk.StringVar(),
            "area_m2": tk.StringVar(),
            "actual_ms": tk.StringVar(),
            "move_px": tk.StringVar(value="2.0"),
        }
        fields = [
            ("Block mass (kg)", "mass_kg", 18),
            ("Incline theta (deg)", "theta_deg", 18),
            ("Dist. release to END gate (m)", "distance_m", 18),
            ("Frontal area A (m2, optional)", "area_m2", 18),
        ]
        for r, (label, key, width) in enumerate(fields):
            ttk.Label(g1, text=label).grid(row=r, column=0, sticky="w", pady=3)
            ttk.Entry(g1, textvariable=self.param_vars[key], width=width).grid(
                row=r, column=1, sticky="we", pady=3, padx=8)
        ttk.Label(g1, text="Release threshold (px)").grid(
            row=len(fields), column=0, sticky="w", pady=3)
        ttk.Spinbox(g1, from_=0.0, to=50.0, increment=0.5, width=18,
                    textvariable=self.param_vars["move_px"]).grid(
            row=len(fields), column=1, sticky="we", pady=3, padx=8)
        g1.columnconfigure(1, weight=1)

        g2 = ttk.LabelFrame(left, text="2. Video & tracking", padding=8)
        g2.pack(fill="x", pady=(0, 8))
        ttk.Button(g2, text="Load video of the sliding block...",
                   style="Primary.TButton", command=self.browse_video).grid(
            row=0, column=0, sticky="w")
        self.file_label_var = tk.StringVar(value="No video loaded")
        ttk.Label(g2, textvariable=self.file_label_var, style="Muted.TLabel",
                  wraplength=280).grid(row=1, column=0, sticky="w", pady=4)
        self.mark_btn = ttk.Button(g2, text="Mark the END gate point",
                                   command=self.mark_gate)
        self.mark_btn.grid(row=2, column=0, sticky="w")
        self.pick_color_btn = ttk.Button(g2, text="Pick block color...",
                                         command=self.pick_color)
        self.pick_color_btn.grid(row=3, column=0, sticky="w", pady=(4, 0))
        self.mode_var = tk.StringVar(value="color")
        ttk.Radiobutton(g2, text="Color-based tracking", value="color",
                        variable=self.mode_var,
                        command=self._update_states).grid(
            row=4, column=0, sticky="w", pady=(6, 0))
        camrow = ttk.Frame(g2)
        camrow.grid(row=5, column=0, sticky="w", pady=(6, 0))
        ttk.Label(camrow, text="Webcam:").pack(side="left")
        self.cam_index_var = tk.StringVar(value="0")
        self.cam_cb = ttk.Combobox(camrow, textvariable=self.cam_index_var,
                                   values=tuple(str(i) for i in range(6)),
                                   width=4, state="readonly")
        self.cam_cb.pack(side="left", padx=(4, 8))
        self.webcam_btn = ttk.Button(camrow, text="Record with webcam...",
                                     style="Primary.TButton",
                                     command=self.start_webcam_record)
        self.webcam_btn.pack(side="left")

        g3 = ttk.LabelFrame(left, text="3. Measure time & friction", padding=8)
        g3.pack(fill="x", pady=(0, 8))
        self.run_btn = ttk.Button(g3, text="Measure time from video",
                                  style="Accent.TButton",
                                  command=self.run_measure)
        self.run_btn.pack(anchor="w")
        ttk.Label(g3, text="Actual measured time (ms):",
                  font=self.font_small).pack(anchor="w", pady=(8, 2))
        ttk.Entry(g3, textvariable=self.param_vars["actual_ms"],
                  width=18).pack(anchor="w")
        self.compute_btn = ttk.Button(g3, text="Compute mu / drag",
                                      command=self.compute_results)
        self.compute_btn.pack(anchor="w", pady=(6, 0))

        g4 = ttk.LabelFrame(left, text="Results", padding=8)
        g4.pack(fill="x")
        self.result_var = tk.StringVar(value="No measurement yet.")
        ttk.Label(g4, textvariable=self.result_var, foreground=p["accent"],
                  font=self.font_bold, justify="left", wraplength=280,
                  anchor="w").pack(anchor="w")

        # ---- right: embedded preview ----
        self.video_msg_var = tk.StringVar(
            value="Load a video and mark the END gate point; the block's "
                  "release-to-gate time is measured the same way as in the "
                  "main app, then mu / drag is computed from your inputs.")
        ttk.Label(right, textvariable=self.video_msg_var, wraplength=700,
                  style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        self.video_photo = None
        self._img_scale = 1.0
        self._disp_size = (0, 0)
        self.video_label = tk.Label(right, text="\n\nNo video loaded\n",
                                    anchor="center", justify="center",
                                    relief="flat", borderwidth=0,
                                    bg="#10162C", fg=p["text_muted"],
                                    font=self.font_base)
        self.video_label.grid(row=1, column=0, sticky="nsew", pady=(6, 6))
        self.video_label.bind("<Button-1>", self._on_preview_click)

        self.toolbars = {}
        tbk_mark = ttk.Frame(right)
        ttk.Button(tbk_mark, text="Reset gate",
                   command=self.on_mark_reset).pack(side="left")
        ttk.Button(tbk_mark, text="Cancel", style="Danger.TButton",
                   command=self.on_mark_cancel).pack(side="left", padx=6)
        self.toolbars["mark"] = tbk_mark
        tbk_cam = ttk.Frame(right)
        self.cam_rec_var = tk.StringVar(value="Start recording")
        self.cam_rec_btn = ttk.Button(tbk_cam, textvariable=self.cam_rec_var,
                                      style="Accent.TButton",
                                      command=self.on_cam_rec_toggle)
        self.cam_rec_btn.pack(side="left")
        ttk.Button(tbk_cam, text="Finish & use this video",
                   style="Primary.TButton",
                   command=self.on_cam_finish).pack(side="left", padx=6)
        ttk.Button(tbk_cam, text="Cancel", style="Danger.TButton",
                   command=self.on_cam_cancel).pack(side="left")
        self.toolbars["cam"] = tbk_cam

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(self.master, textvariable=self.status_var, anchor="w",
                  style="Muted.TLabel").pack(fill="x", padx=12, pady=(0, 8))

        self._update_states()

    # ------------------------------------------------------------------
    # Preview plumbing
    # ------------------------------------------------------------------
    def _to_photo(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if HAVE_PIL:
            return PIL.ImageTk.PhotoImage(PIL.Image.fromarray(rgb))
        ok, buf = cv2.imencode(".ppm", rgb)
        return tk.PhotoImage(data=buf.tobytes())

    def _show_bgr(self, frame, draw=None):
        if draw is not None:
            frame = frame.copy()
            draw(frame)
        h, w = frame.shape[:2]
        lw = self.video_label.winfo_width()
        maxw = min(MAX_W, lw) if lw > 200 else MAX_W
        if w > maxw:
            s = maxw / w
            disp = cv2.resize(frame, (maxw, int(h * s)))
        else:
            disp = frame
        self._img_scale = w / disp.shape[1]
        self._disp_size = (disp.shape[1], disp.shape[0])
        self.video_photo = self._to_photo(disp)
        self.video_label.configure(image=self.video_photo, text="")

    def _clear_preview(self, message):
        self.video_photo = None
        self.video_label.configure(image="", text=message, compound="center")

    def _on_preview_click(self, event):
        dw, dh = self._disp_size
        ow = self.video_label.winfo_width()
        oh = self.video_label.winfo_height()
        sx = int((event.x - (ow - dw) // 2) * self._img_scale)
        sy = int((event.y - (oh - dh) // 2) * self._img_scale)
        if self.ui_mode == "mark" and self.mark_mode:
            if not self.mark_mode["points"]:
                self.mark_mode["points"].append((sx, sy))
                self._confirm_gate(sx, sy)
        elif self.ui_mode == "pick" and self.pick_mode:
            self._sample_color(sx, sy)

    # ------------------------------------------------------------------
    def _show_toolbar(self, name):
        for tb in self.toolbars.values():
            tb.grid_remove()
        if name and name in self.toolbars:
            self.toolbars[name].grid(row=2, column=0, sticky="w", pady=(6, 0))

    # ------------------------------------------------------------------
    def browse_video(self):
        path = filedialog.askopenfilename(
            title="Choose video of the sliding block",
            filetypes=[("Video files", "*.avi *.mp4 *.mov *.mkv *.webm *.mjpg"),
                       ("All files", "*.*")])
        if not path:
            return
        self._load_existing_video(path)

    def _load_existing_video(self, path):
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            messagebox.showerror("Load video",
                                 "Could not read the first video frame.")
            return
        self.video_path = path
        self.gate_points = None
        self.color_range = None
        self.ui_mode = "idle"
        self._track_first = None
        self._track_released = False
        self._show_toolbar(None)
        self.file_label_var.set(f"Video: {os.path.basename(path)}")
        self.video_msg_var.set(
            f"Loaded {os.path.basename(path)}. Mark the END gate point on the "
            "preview (single click).")
        self._show_bgr(frame)
        self.status(f"Loaded video: {os.path.abspath(path)}")
        self._update_states()

    def mark_gate(self):
        if not self.video_path:
            messagebox.showinfo("Mark gate", "Load a video first.")
            return
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return
        self.ui_mode = "mark"
        self.mark_mode = {"frame": frame, "points": []}
        self.video_label.configure(cursor="crosshair")
        self._show_toolbar("mark")
        self.video_msg_var.set(
            "Click once on the END gate point in the preview (where the "
            "block's time is measured TO).")
        self._show_bgr(frame)

    def on_mark_reset(self):
        if self.mark_mode:
            self.mark_mode["points"] = []
            self.video_msg_var.set("Click again to place the END gate point.")
            self._show_bgr(self.mark_mode["frame"])

    def on_mark_cancel(self):
        self.ui_mode = "idle"
        self.mark_mode = None
        self.pick_mode = None
        self.video_label.configure(cursor="")
        self._show_toolbar(None)
        self.video_msg_var.set("Marking cancelled. No END gate set.")
        self._update_states()

    def _confirm_gate(self, x, y):
        frame = self.mark_mode["frame"] if self.mark_mode else None
        self.gate_points = [(x, y)]
        self.mark_mode = None
        self.ui_mode = "idle"
        self.video_label.configure(cursor="")
        self._show_toolbar(None)
        self.video_msg_var.set(
            f"END gate set at ({x}, {y}). Press 'Measure time from video'.")
        if frame is not None:
            def draw(f, gx=x, gy=y):
                cv2.circle(f, (gx, gy), 8, (0, 0, 255), -1)
                cv2.line(f, (gx, 0), (gx, f.shape[0]), (0, 0, 255), 1)
                cv2.putText(f, "END", (gx + 12, gy), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)
            self._show_bgr(frame, draw=draw)
        self._update_states()

    def pick_color(self):
        if not self.video_path:
            messagebox.showinfo("Pick color", "Load a video first.")
            return
        cap = cv2.VideoCapture(self.video_path)
        frame = None
        for _ in range(8):
            ret, frame = cap.read()
            if not ret:
                break
        cap.release()
        if frame is None:
            return
        self.ui_mode = "pick"
        self.pick_mode = {"frame": frame,
                          "hsv": cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)}
        self.color_range = None
        self.video_label.configure(cursor="crosshair")
        self.video_msg_var.set(
            "Click once on the block in the preview to sample its color.")
        self._show_bgr(frame)

    def _sample_color(self, x, y):
        pm = self.pick_mode
        h, w = pm["hsv"].shape[:2]
        box = 6
        x0, x1 = max(0, x - box), min(w, x + box)
        y0, y1 = max(0, y - box), min(h, y + box)
        patch = pm["hsv"][y0:y1, x0:x1].reshape(-1, 3)
        self.color_range = tracker.color_range_from_patch(patch)
        self.ui_mode = "idle"
        frame = pm["frame"]
        self.pick_mode = None
        self.video_label.configure(cursor="")
        self.video_msg_var.set(
            "Block color sampled. Run 'Measure time from video'.")

        def draw(f, sx=x, sy=y):
            cv2.circle(f, (sx, sy), box, (0, 255, 0), 2)
            cv2.circle(f, (sx, sy), box * 3, (0, 255, 0), 1)
        self._show_bgr(frame, draw=draw)
        self._update_states()

    # ------------------------------------------------------------------
    # Webcam recording
    # ------------------------------------------------------------------
    def start_webcam_record(self):
        try:
            idx = int(self.cam_index_var.get())
        except ValueError:
            idx = 0
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            messagebox.showerror("Webcam", f"Could not open webcam #{idx}.")
            return
        # Probe: some indices "open" but never deliver a frame (wrong index or
        # the camera is in use elsewhere) - that used to crash in the reader
        # loop. Check up-front instead.
        try:
            ret, probe = cap.read()
        except cv2.error:
            ret, probe = False, None
        if not ret or probe is None or probe.size == 0:
            cap.release()
            messagebox.showerror("Webcam",
                                 f"Webcam #{idx} opened but produced no frames.\n"
                                 "It may be in use by another app or is not a "
                                 "working camera.\n"
                                 "Try a different webcam index in the dropdown.")
            return
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0
        h, w = probe.shape[:2]
        out_path = os.path.join(os.getcwd(), "friction_webcam_take.avi")
        writer = self._make_writer(out_path, fps, w, h)
        if writer is None:
            cap.release()
            messagebox.showerror("Webcam", "Could not create a video writer.")
            return
        self.cam = {
            "cap": cap, "fps": fps, "w": w, "h": h,
            "q": queue.Queue(maxsize=1), "stop": threading.Event(),
            "rec": False, "rec_start": 0.0, "writer": writer,
            "out_path": out_path, "has_file": False, "failed": False,
        }
        self.ui_mode = "cam"
        self.cam_rec_var.set("Start recording")
        self.video_msg_var.set("Webcam live. Press 'Start recording' to "
                               "capture, then 'Finish & use this video'.")
        self._show_toolbar("cam")
        self._update_states()

        def loop():
            cam = self.cam
            try:
                while not cam["stop"].is_set():
                    try:
                        ret, frame = cam["cap"].read()
                    except cv2.error:
                        ret, frame = False, None
                    if not ret or frame is None:
                        cam["failed"] = True
                        break
                    if cam["rec"] and cam["writer"] is not None:
                        cam["writer"].write(frame)
                    elapsed = (time.monotonic() - cam["rec_start"]
                               if cam["rec"] else 0.0)
                    try:
                        cam["q"].put_nowait((frame, cam["rec"], elapsed))
                    except queue.Full:
                        pass
            finally:
                cam["cap"].release()

        self.cam["thread"] = threading.Thread(target=loop, daemon=True)
        self.cam["thread"].start()
        self.after(33, self._poll_cam)

    @staticmethod
    def _make_writer(path, fps, w, h):
        for codec in ("XVID", "MJPG", "mp4v"):
            writer = None
            try:
                writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec),
                                         fps, (w, h))
            except cv2.error:
                writer = None
            if writer is not None and writer.isOpened():
                return writer
            if writer is not None:
                writer.release()
        return None

    def on_cam_rec_toggle(self):
        cam = self.cam
        if cam is None:
            return
        if not cam["rec"]:
            cam["rec"] = True
            cam["rec_start"] = time.monotonic()
            cam["has_file"] = True
            self.cam_rec_var.set("Stop recording")
            self.video_msg_var.set("Recording... press 'Stop recording' to "
                                   "capture the video.")
        else:
            cam["rec"] = False
            self.cam_rec_var.set("Start recording")
            self.video_msg_var.set("Recording stopped. You can re-record, or "
                                   "press 'Finish & use this video'.")

    def _poll_cam(self):
        cam = self.cam
        if cam is None:
            self._show_toolbar(None)
            return
        if cam.get("failed"):
            self._clear_preview("Webcam stopped producing frames.")
            messagebox.showwarning(
                "Webcam",
                "The webcam stopped producing frames.\n"
                "It may be in use by another app or a wrong index was chosen.\n"
                "Click 'Record with webcam...' again to retry.")
            self._finish_cam(keep_file=False)
            self.status("Webcam failed to capture frames.")
            return
        try:
            frame, rec, elapsed = cam["q"].get_nowait()
        except queue.Empty:
            frame = None
        if frame is not None:
            def draw(f, rec=rec, elapsed=elapsed):
                self._draw_cam_overlay(f, rec, elapsed)
            self._show_bgr(frame, draw=draw)
        self.after(33, self._poll_cam)

    @staticmethod
    def _draw_cam_overlay(frame, rec, elapsed):
        h, w = frame.shape[:2]
        if rec:
            cv2.circle(frame, (20, h - 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (38, h - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.circle(frame, (20, h - 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, "LIVE", (38, h - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    def _finish_cam(self, keep_file):
        cam = self.cam
        if cam is None:
            return
        cam["stop"].set()
        cam["rec"] = False
        thread = cam.get("thread")
        if thread is not None:
            thread.join(timeout=3)
        writer = cam.get("writer")
        if writer is not None:
            writer.release()
            cam["writer"] = None
        out_path = cam["out_path"]
        self.cam = None
        self.ui_mode = "idle"
        self._show_toolbar(None)
        self._update_states()
        if not keep_file and os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass

    def on_cam_cancel(self):
        if self.cam is None:
            return
        self._finish_cam(keep_file=False)
        self._clear_preview("Webcam session cancelled - use a video file.")
        self.status("Webcam recording cancelled.")

    def on_cam_finish(self):
        cam = self.cam
        if cam is None:
            return
        if cam["rec"]:
            messagebox.showinfo("Still recording", "Press 'Stop recording' "
                                                   "first.")
            return
        if not cam["has_file"]:
            messagebox.showinfo("Nothing recorded", "Press 'Start recording' "
                                                    "before finishing.")
            return
        out = cam["out_path"]
        self._finish_cam(keep_file=True)
        if not os.path.exists(out) or os.path.getsize(out) < 1000:
            self._clear_preview("Recorded file was empty - please retry.")
            return
        self._load_existing_video(out)

    # ------------------------------------------------------------------
    # Background tracking
    # ------------------------------------------------------------------
    def _track_worker(self):
        return tracker.track_object(
            self.video_path, color_range=self.color_range, show=False,
            frame_cb=self._track_frame_cb)

    def _track_frame_cb(self, frame, t_sec, centroid, frame_idx):
        try:
            self.cv2_q.put_nowait(("frame", (frame, t_sec, centroid, frame_idx)))
        except queue.Full:
            pass

    def run_measure(self):
        if not (self.video_path and self.gate_points):
            messagebox.showinfo("Measure",
                                "Load a video and mark the END gate first.")
            return
        if self.color_range is None:
            messagebox.showinfo("Measure", "Pick the block color first.")
            return
        if self.cv2_busy or self.cam is not None:
            return
        if not self._read_params():
            return
        self.cv2_busy = True
        self.pending_callback = self.on_tracking_done
        self.ui_mode = "track"
        self._track_first = None
        self._track_released = False
        self.video_msg_var.set("Tracking the block... live preview below.")
        self._update_states()

        def run():
            try:
                result = self._track_worker()
                self.cv2_q.put(("done", result))
            except Exception as exc:  # noqa: BLE001
                self.cv2_q.put(("error", exc))

        threading.Thread(target=run, daemon=True).start()

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.cv2_q.get_nowait()
                if kind == "frame":
                    frame, t_sec, centroid, frame_idx = payload
                    if self.ui_mode == "track":
                        def draw(f, c=centroid, t=t_sec, idx=frame_idx):
                            if self.gate_points:
                                (ex, ey) = self.gate_points[0]
                                cv2.circle(f, (ex, ey), 8, (0, 0, 255), -1)
                                cv2.line(f, (ex, 0), (ex, f.shape[0]),
                                         (0, 0, 255), 1)
                            if c is not None:
                                colour = (0, 255, 0)
                                try:
                                    move_px = float(self.param_vars["move_px"].get())
                                except ValueError:
                                    move_px = 2.0
                                if self._track_first is None:
                                    self._track_first = (c[0], c[1])
                                if (self._track_released or
                                        abs(c[0] - self._track_first[0]) >= move_px or
                                        abs(c[1] - self._track_first[1]) >= move_px):
                                    self._track_released = True
                                    colour = (0, 255, 255)
                                cv2.circle(f, (int(c[0]), int(c[1])), 6,
                                           colour, -1)
                            cv2.putText(f, f"frame={idx} t={t:.3f}s", (12, 26),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                        (0, 255, 255), 2)
                        self._show_bgr(frame, draw=draw)
                    continue
                cb = self.pending_callback
                self.pending_callback = None
                self.cv2_busy = False
                self.ui_mode = "idle"
                self._update_states()
                if kind == "error":
                    self.status("Error: " + str(payload))
                    if cb:
                        cb("__comerror__: " + str(payload))
                elif cb:
                    cb(payload)
        except queue.Empty:
            pass
        self.after(50, self._poll_queue)

    # ------------------------------------------------------------------
    def _read_params(self):
        for key in ("mass_kg", "theta_deg", "distance_m"):
            if not self.param_vars[key].get().strip():
                messagebox.showinfo("Missing value", f"Enter {key}.")
                return False
        return True

    def compute_results(self, ms=None):
        try:
            mass = float(self.param_vars["mass_kg"].get())
            theta = float(self.param_vars["theta_deg"].get())
            dist = float(self.param_vars["distance_m"].get())
        except ValueError:
            messagebox.showerror("Compute",
                                 "Enter numeric mass, theta, distance.")
            return
        if ms is None:
            try:
                ms = float(self.param_vars["actual_ms"].get())
            except ValueError:
                messagebox.showerror("Compute",
                                     "Enter a valid time in milliseconds.")
                return
        if ms <= 0:
            messagebox.showerror("Compute", "Time must be positive.")
            return
        t_s = ms / 1000.0
        area = 0.0
        if self.param_vars["area_m2"].get().strip():
            try:
                area = float(self.param_vars["area_m2"].get())
            except ValueError:
                area = 0.0
        try:
            f = fs.compute_friction(mass, theta, dist, t_s)
            mu = f["mu_no_drag"]
            lines = [f"time         : {t_s * 1000:.1f} ms",
                     f"a = 2d/t^2   : {f['a_exp']:.4f} m/s^2",
                     f"v_avg        : {f['v_avg']:.3f} m/s",
                     f"mu (no drag) : {mu:.4f}"]
            if not f["will_slide"]:
                lines.append("WARNING: mu >= tan(theta); block would NOT slide!")
            d = fs.compute_drag_cd(mass, theta, dist, t_s, mu, area or None)
            lines.append(f"F_drag (avg) : {d['f_drag']:.4f} N")
            if d["cd"] is not None:
                lines.append(f"C_d          : {d['cd']:.4f}")
            t_pred = fs.predict_time(mu, theta, dist)
            if t_pred:
                lines.append(f"pred. time   : {t_pred * 1000:.1f} ms")
            self.result_var.set("\n".join(lines))
            self.status("mu / drag computed from the given time.")
        except (ValueError, ZeroDivisionError) as exc:
            messagebox.showerror("Compute", str(exc))

    def on_tracking_done(self, payload):
        if isinstance(payload, str) and payload.startswith("__comerror__"):
            messagebox.showerror("Measure", payload)
            return
        trajectory, fps = payload
        try:
            move_px = float(self.param_vars["move_px"].get())
        except ValueError:
            move_px = 2.0
        res = tracker.compute_crossing_time_ms(trajectory, self.gate_points,
                                               move_px=move_px)
        if res is None or not trajectory:
            self.video_msg_var.set(
                "No release-to-gate crossing detected. Re-mark the gate, "
                "improve lighting, or re-sample the block color.")
            self.status("Measurement failed.")
            self.result_var.set("No crossing detected from the video.")
            return
        ms = res["elapsed_ms"]
        self.param_vars["actual_ms"].set(f"{ms:.1f}")
        self.video_msg_var.set(
            f"Video time = {ms:.1f} ms ({fps:.1f} fps). Check the 'Actual "
            "measured time', then press 'Compute mu / drag'.")
        self.status("Video time measured. Press 'Compute mu / drag'.")
        self.compute_results(ms=ms)

    def _update_states(self):
        busy = self.cv2_busy
        cam_active = self.cam is not None
        have_video = bool(self.video_path)
        have_gate = bool(self.gate_points)
        have_color = bool(self.color_range)
        ready = have_video and have_gate and have_color and not busy and not cam_active
        self.mark_btn.configure(
            state="normal" if have_video and not busy and not cam_active
            else "disabled")
        self.pick_color_btn.configure(
            state="normal" if have_video and not busy and not cam_active
            else "disabled")
        self.webcam_btn.configure(
            state="normal" if not busy and not cam_active else "disabled")
        self.run_btn.configure(state="normal" if ready else "disabled")

    # ------------------------------------------------------------------
    def after(self, ms, fn):
        return self.master.after(ms, fn)

    def option_add(self, *args):
        return self.master.option_add(*args)

    def status(self, msg):
        self.status_var.set(msg)


def _fit_window(win, w=1100, h=700, minw=980, minh=600):
    sw = win.winfo_screenwidth()
    sh = win.winfo_screenheight()
    w = min(w, int(sw * 0.97))
    h = min(h, int(sh * 0.97))
    win.geometry(f"{w}x{h}+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 2)}")
    win.minsize(min(minw, sw), min(minh, sh))


def launch(parent=None):
    """Open the analyser. No parent -> standalone window."""
    if parent is None:
        root = tk.Tk()
        FrictionSlopeApp(root)
        _fit_window(root)
        root.mainloop()
        return root
    win = tk.Toplevel(parent)
    FrictionSlopeApp(win)
    _fit_window(win)
    return win


if __name__ == "__main__":
    launch()