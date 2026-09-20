"""
gui.py
------
Tkinter GUI for the Rolling Body Moment of Inertia Analyzer.

Run:  python gui.py

Tabs:
  * Configure solids  - name / mass / radius / theta / distance / I_theory
  * Record a trial    - webcam or existing video -> mark gates -> track ->
                        measure crossing time -> save as trial 1/2/3
  * Results & export  - live comparison table (Excel/CSV)

Camera / video interaction is EMBEDDED in the Tk window (no separate OpenCV
windows): live webcam recording, gate marking, color sampling and a live
tracking preview all render inside the GUI. After tracking finishes a
frame-by-frame review player opens with a corner timer overlay.

Long running work (tracking) runs in a background thread; frames and results
are marshalled back to the tkinter thread via a queue polled by `after`.
"""

import os
import queue
import threading
import time
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk, messagebox, filedialog, simpledialog

import cv2

try:
    import PIL.Image
    import PIL.ImageTk
    HAVE_PIL = True
except Exception:  # noqa: BLE001 - PIL optional; fall back to PPM/PNG
    HAVE_PIL = False

import numpy as np

import data_manager as dm
import tracker
import box_tracker
import perspective

MAX_W = 1000  # largest width shown in the embedded preview

# ----------------------------------------------------------------------
# Colour palette / theme constants
# ----------------------------------------------------------------------
PALETTE = {
    "bg": "#F4F6FB",          # app background
    "surface": "#FFFFFF",     # cards / panels
    "header_grad_top": "#1B2A5B",
    "header_grad_bot": "#2E4A9C",
    "primary": "#2E4A9C",     # main accent (deep indigo-blue)
    "primary_dark": "#1B2A5B",
    "accent": "#F2A93B",      # warm amber accent for primary actions
    "accent_dark": "#D68F1F",
    "success": "#1E8A5F",
    "danger": "#C0392B",
    "text": "#1C2333",
    "text_muted": "#5B6478",
    "border": "#D8DDEA",
    "row_alt": "#EEF1FA",
}


class RollingBodyApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Rolling Body Moment of Inertia Analyzer")
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w, h = min(1320, int(sw * 0.97)), min(860, int(sh * 0.97))
        self.geometry(f"{w}x{h}+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 2)}")
        self.minsize(min(1080, sw), min(720, sh))

        self.data = dm.load_data()

        self.cv2_q = queue.Queue(maxsize=8)   # bounded: no runaway memory
        self.pending_callback = None
        self.cv2_busy = False

        # per-session measurement state
        self.video_path = None
        self.gate_points = None
        self.color_range = None
        self.result = None
        self.slowmo_mult = 1.0     # video time divider for slow-motion footage
        self.trajectory = None
        self.timeline = {}      # frame_idx -> (cx, cy)
        self.crossing_text = None
        self.pending_color = None
        self._preview_enabled = True

        # perspective rectification (image -> table-plane homography)
        self.H = None            # 3x3 homography, or None (perspective off)
        self.persp_mode = None   # {"frame", "points"} during 4-corner marking
        self.persp_src = None    # the 4 image reference points that made H

        # live preview state
        self.video_photo = None
        self._img_scale = 1.0
        self._disp_size = (0, 0)
        self.ui_mode = "idle"   # idle | webcam | mark | pick | track | review
        self.mark_mode = None   # {"frame", "points"}
        self.pick_mode = None   # {"frame", "hsv"}
        self.cam = None         # webcam session dict
        self.review = None      # review player dict
        self._track_first = None
        self._track_released = False
        self._release_frame = None

        # box-tracker state
        self.box = None            # (x, y, w, h) on the start frame
        self.box_gates = None      # [END] point
        self.box_start = 0         # frame index tracking starts from
        self.box_setup = None      # interactive setup dict
        self.box_timeline = {}     # frame_idx -> box (after tracking)
        self.run_mode = "box"
        self.stop_event = threading.Event()

        self.configure(bg=PALETTE["bg"])
        self._setup_style()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(50, self._poll_cv2_queue)
        self.after(33, self._poll_webcam)
        self.refresh_all()

    # ------------------------------------------------------------------
    # Look & feel
    # ------------------------------------------------------------------
    def _setup_style(self):
        """Configure a cohesive, modern ttk theme for the whole app."""
        p = PALETTE
        base_family = "Segoe UI"
        if base_family not in tkfont.families():
            base_family = "Helvetica"

        self.font_base = tkfont.Font(family=base_family, size=10)
        self.font_bold = tkfont.Font(family=base_family, size=10, weight="bold")
        self.font_h1 = tkfont.Font(family=base_family, size=17, weight="bold")
        self.font_team = tkfont.Font(family=base_family, size=13, weight="bold")
        self.font_h2 = tkfont.Font(family=base_family, size=10, weight="bold")
        self.font_small = tkfont.Font(family=base_family, size=9)
        self.option_add("*Font", self.font_base)

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(".", background=p["bg"], foreground=p["text"],
                        font=self.font_base)
        style.configure("TFrame", background=p["bg"])
        style.configure("Surface.TFrame", background=p["surface"])
        style.configure("TLabel", background=p["bg"], foreground=p["text"])
        style.configure("Muted.TLabel", background=p["bg"],
                        foreground=p["text_muted"], font=self.font_small)
        style.configure("Status.TLabel", background=p["primary_dark"],
                        foreground="#FFFFFF", font=self.font_small,
                        padding=(10, 5))

        # LabelFrame ("card") look
        style.configure("TLabelframe", background=p["surface"],
                        bordercolor=p["border"], relief="solid",
                        borderwidth=1)
        style.configure("TLabelframe.Label", background=p["surface"],
                        foreground=p["primary_dark"], font=self.font_h2)
        style.configure("TLabelframe.TLabel", background=p["surface"])

        # Buttons
        style.configure("TButton", background=p["surface"],
                        foreground=p["text"], font=self.font_base,
                        padding=(10, 6), relief="flat", borderwidth=1,
                        bordercolor=p["border"])
        style.map("TButton",
                 background=[("active", p["row_alt"]), ("disabled", p["bg"])],
                 foreground=[("disabled", p["text_muted"])])

        style.configure("Accent.TButton", background=p["accent"],
                        foreground="#3A2200", font=self.font_bold,
                        padding=(12, 7), relief="flat", borderwidth=0)
        style.map("Accent.TButton",
                 background=[("active", p["accent_dark"]),
                             ("disabled", p["border"])],
                 foreground=[("disabled", p["text_muted"])])

        style.configure("Primary.TButton", background=p["primary"],
                        foreground="#FFFFFF", font=self.font_bold,
                        padding=(12, 7), relief="flat", borderwidth=0)
        style.map("Primary.TButton",
                 background=[("active", p["primary_dark"]),
                             ("disabled", p["border"])],
                 foreground=[("disabled", p["text_muted"])])

        style.configure("Danger.TButton", background=p["surface"],
                        foreground=p["danger"], font=self.font_base,
                        padding=(10, 6), relief="flat", borderwidth=1,
                        bordercolor=p["danger"])
        style.map("Danger.TButton",
                 background=[("active", "#FBEAE8")])

        # Notebook / tabs
        style.configure("TNotebook", background=p["bg"], borderwidth=0)
        style.configure("TNotebook.Tab", background=p["bg"],
                        foreground=p["text_muted"], font=self.font_h2,
                        padding=(16, 9), borderwidth=0)
        style.map("TNotebook.Tab",
                 background=[("selected", p["surface"])],
                 foreground=[("selected", p["primary_dark"])],
                 expand=[("selected", (1, 1, 1, 0))])

        # Entries / combobox
        style.configure("TEntry", fieldbackground=p["surface"],
                        foreground=p["text"], bordercolor=p["border"],
                        lightcolor=p["border"], darkcolor=p["border"],
                        padding=4)
        style.map("TEntry", bordercolor=[("focus", p["primary"])])
        style.configure("TCombobox", fieldbackground=p["surface"],
                        foreground=p["text"], bordercolor=p["border"],
                        padding=4)
        style.map("TCombobox", fieldbackground=[("readonly", p["surface"])],
                 bordercolor=[("focus", p["primary"])])

        style.configure("TCheckbutton", background=p["surface"],
                        foreground=p["text"])
        style.map("TCheckbutton", background=[("active", p["surface"])])
        style.configure("TRadiobutton", background=p["surface"],
                        foreground=p["text"])
        style.map("TRadiobutton", background=[("active", p["surface"])])

        # Treeview
        style.configure("Treeview", background=p["surface"],
                        fieldbackground=p["surface"], foreground=p["text"],
                        rowheight=26, borderwidth=0, font=self.font_base)
        style.configure("Treeview.Heading", background=p["primary"],
                        foreground="#FFFFFF", font=self.font_h2,
                        relief="flat", padding=(6, 6))
        style.map("Treeview.Heading", background=[("active", p["primary_dark"])])
        style.map("Treeview",
                 background=[("selected", p["primary"])],
                 foreground=[("selected", "#FFFFFF")])

        # Scrollbar
        style.configure("Vertical.TScrollbar", background=p["bg"],
                        troughcolor=p["bg"], bordercolor=p["bg"],
                        arrowcolor=p["primary"])
        style.configure("Horizontal.TScrollbar", background=p["bg"],
                        troughcolor=p["bg"], bordercolor=p["bg"],
                        arrowcolor=p["primary"])

        # Scale (review slider)
        style.configure("TScale", background=p["surface"],
                        troughcolor=p["border"])

    def _build_header(self, parent):
        """Course / experiment / group banner shown above the tabs."""
        p = PALETTE
        header = tk.Frame(parent, bg=p["header_grad_top"])
        header.pack(fill="x", side="top", pady=(0, 10))

        inner = tk.Frame(header, bg=p["header_grad_top"])
        inner.pack(fill="x", padx=18, pady=(12, 12))

        tk.Label(inner, text="BITS U104 \u2014 Foundations of Measurement and Experimentation",
                 bg=p["header_grad_top"], fg="#FFFFFF",
                 font=self.font_h1, anchor="w", justify="left").pack(
            anchor="w")

        tk.Label(inner,
                 text="Experimental Investigation and Calculation of the "
                      "Moment of Inertia of Rolling Body",
                 bg=p["header_grad_top"], fg=p["accent"],
                 font=self.font_bold, anchor="w", justify="left",
                 wraplength=1200).pack(anchor="w", pady=(4, 0))

        tk.Label(inner, text="Group L1 29A  \u2022  Vighnesh Nauso Shetye  &  Harshal Lule",
                 bg=p["header_grad_top"], fg="#C9D3F0",
                 font=self.font_team, anchor="w", justify="left").pack(
            anchor="w", pady=(6, 0))

        # thin accent underline
        tk.Frame(parent, bg=p["accent"], height=3).pack(fill="x", side="top")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        self.main = ttk.Frame(self, padding=0)
        self.main.pack(fill="both", expand=True)

        self._build_header(self.main)

        body = ttk.Frame(self.main, padding=(12, 0, 12, 10))
        body.pack(fill="both", expand=True)

        self.notebook = ttk.Notebook(body)
        self.notebook.pack(fill="both", expand=True)

        self.tab_config = ttk.Frame(self.notebook, padding=12)
        self.tab_record = ttk.Frame(self.notebook, padding=12)
        self.tab_results = ttk.Frame(self.notebook, padding=12)

        self.notebook.add(self.tab_config, text="  Configure solids  ")
        self.notebook.add(self.tab_record, text="  Record a trial  ")
        self.notebook.add(self.tab_results, text="  Results & export  ")

        self._build_config_tab(self.tab_config)
        self._build_record_tab(self.tab_record)
        self._build_results_tab(self.tab_results)

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(body, textvariable=self.status_var, anchor="w",
                 style="Status.TLabel").pack(fill="x", pady=(8, 0))

    def _build_config_tab(self, parent):
        left = ttk.Frame(parent)
        left.pack(side="left", fill="y", padx=(0, 12))
        right = ttk.LabelFrame(parent, text="Edit selected solid", padding=12)
        right.pack(side="left", fill="both", expand=True)

        ttk.Button(left, text="Friction-slope analyser...",
                   style="Primary.TButton", command=self.open_friction_gui).pack(
            anchor="w", pady=(0, 10))

        ttk.Label(left, text="Solids (click to select):",
                  font=self.font_h2).pack(anchor="w", pady=(0, 4))
        self.solid_tree = ttk.Treeview(left, columns=("slot", "name", "filled"),
                                       show="headings", height=8, selectmode="browse")
        for c, w, text, anchor in (("slot", 50, "Slot", "center"),
                                   ("name", 210, "Name", "w"),
                                   ("filled", 100, "Trials", "center")):
            self.solid_tree.heading(c, text=text)
            self.solid_tree.column(c, width=w, anchor=anchor)
        self.solid_tree.pack(anchor="w")
        self.solid_tree.bind("<<TreeviewSelect>>", self.on_solid_select)

        self.cfg_vars = {
            "name": tk.StringVar(),
            "mass_kg": tk.StringVar(),
            "radius_m": tk.StringVar(),
            "theta_deg": tk.StringVar(),
            "distance_m": tk.StringVar(),
            "I_theory": tk.StringVar(),
            "mu_friction": tk.StringVar(),
            "cd_drag": tk.StringVar(),
        }
        self.use_friction_var = tk.BooleanVar(value=False)
        self.use_drag_var = tk.BooleanVar(value=False)

        fields = [
            ("Name", "name", 34),
            ("Mass (kg)", "mass_kg", 16),
            ("Radius (m)", "radius_m", 16),
            ("Incline theta (deg)", "theta_deg", 16),
            ("Gate distance (m)", "distance_m", 16),
            ("I theoretical (kg.m^2)", "I_theory", 18),
        ]
        for r, (label, key, width) in enumerate(fields):
            ttk.Label(right, text=label).grid(row=r, column=0, sticky="w", pady=3)
            ttk.Entry(right, textvariable=self.cfg_vars[key], width=width).grid(
                row=r, column=1, sticky="we", pady=3, padx=8)

        # loss-model options (rows 6, 7)
        loss = ttk.LabelFrame(right, text="Loss corrections (for a fairer I_exp)",
                              padding=8)
        loss.grid(row=6, column=0, columnspan=2, sticky="we", pady=(10, 0))
        ttk.Checkbutton(loss, text="Rolling friction / resistance",
                        variable=self.use_friction_var).grid(row=0, column=0, sticky="w")
        ttk.Label(loss, text="mu =").grid(row=0, column=1, sticky="e", padx=(12, 2))
        ttk.Entry(loss, textvariable=self.cfg_vars["mu_friction"], width=8).grid(
            row=0, column=2, sticky="w")
        ttk.Label(loss, text="(dimensionless)", style="Muted.TLabel").grid(
            row=0, column=3, sticky="w", padx=(8, 0))

        ttk.Checkbutton(loss, text="Air drag",
                        variable=self.use_drag_var).grid(row=1, column=0, sticky="w")
        ttk.Label(loss, text="Cd =").grid(row=1, column=1, sticky="e", padx=(12, 2))
        ttk.Entry(loss, textvariable=self.cfg_vars["cd_drag"], width=8).grid(
            row=1, column=2, sticky="w")
        ttk.Label(loss, text="(sphere~0.47, A=\u03c0R\u00b2, \u03c1=1.225)",
                  style="Muted.TLabel").grid(row=1, column=3, sticky="w", padx=(8, 0))
        loss.columnconfigure(3, weight=1)

        right.columnconfigure(1, weight=1)

        btnrow = ttk.Frame(right)
        btnrow.grid(row=7, column=0, columnspan=2, sticky="w", pady=(14, 0))
        ttk.Button(btnrow, text="Save changes", style="Primary.TButton",
                   command=self.save_solid).pack(side="left")
        ttk.Button(btnrow, text="Clear trials", style="Danger.TButton",
                   command=self.clear_trials).pack(side="left", padx=8)

    def _build_record_tab(self, parent):
        parent.columnconfigure(0, weight=0, minsize=520)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)

        left = ttk.Frame(parent)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
        right = ttk.LabelFrame(parent, text="Camera / video preview", padding=8)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        # ---------------- left: control groups ----------------
        g1 = ttk.LabelFrame(left, text="1. Solid", padding=8)
        g1.pack(fill="x", pady=(0, 8))
        ttk.Label(g1, text="Solid:").pack(side="left")
        self.solid_cb = ttk.Combobox(g1, state="readonly", width=40)
        self.solid_cb.pack(side="left", padx=6)
        self.solid_cb.bind("<<ComboboxSelected>>",
                           lambda e: self._record_solid_changed())

        g2 = ttk.LabelFrame(left, text="2. Video source", padding=8)
        g2.pack(fill="x", pady=(0, 8))
        self.source_var = tk.StringVar(value="file")
        ttk.Radiobutton(g2, text="Existing video file", value="file",
                        variable=self.source_var,
                        command=self._update_step_states).grid(row=0, column=0, sticky="w")
        self.file_entry = ttk.Entry(g2, width=40)
        self.file_entry.grid(row=1, column=0, sticky="we", padx=(18, 6))
        ttk.Button(g2, text="Browse...", command=self.browse_video).grid(
            row=1, column=1)
        slowrow = ttk.Frame(g2)
        slowrow.grid(row=2, column=0, sticky="w", padx=(18, 0), pady=(4, 0))
        ttk.Label(slowrow, text="Slow-mo factor:").pack(side="left")
        self.slowmo_var = tk.StringVar(value="1")
        self.slowmo_spin = ttk.Spinbox(slowrow, from_=1.0, to=32.0, increment=0.5,
                                       width=6, textvariable=self.slowmo_var)
        self.slowmo_spin.pack(side="left", padx=(6, 6))
        ttk.Label(slowrow, text="\u00d7").pack(side="left")
        ttk.Label(slowrow, text="measured time \u00f7 factor = real time "
                                "(1 = normal, 2 = half speed, \u2026)",
                  style="Muted.TLabel").pack(side="left")
        fpsrow = ttk.Frame(g2)
        fpsrow.grid(row=3, column=0, sticky="w", padx=(18, 0), pady=(4, 0))
        ttk.Label(fpsrow, text="Video FPS:").pack(side="left")
        self.fps_var = tk.StringVar(value="")
        self.fps_spin = ttk.Spinbox(fpsrow, from_=1.0, to=1000.0, increment=1.0,
                                    width=6, textvariable=self.fps_var)
        self.fps_spin.pack(side="left", padx=(6, 6))
        ttk.Label(fpsrow, text="overrides the video container FPS "
                               "(leave empty = use reported FPS)",
                  style="Muted.TLabel").pack(side="left")
        ttk.Radiobutton(g2, text="Live webcam (records a new video first)",
                        value="webcam", variable=self.source_var,
                        command=self._update_step_states).grid(
            row=4, column=0, sticky="w", pady=(4, 0))
        camrow = ttk.Frame(g2)
        camrow.grid(row=5, column=0, sticky="w", padx=(18, 0))
        ttk.Label(camrow, text="Webcam:").pack(side="left")
        self.cam_index_var = tk.StringVar(value="0")
        self.cam_cb = ttk.Combobox(camrow, textvariable=self.cam_index_var,
                                   values=tuple(str(i) for i in range(6)),
                                   width=4, state="readonly")
        self.cam_cb.bind("<<ComboboxSelected>>",
                         lambda e: self._update_step_states())
        self.cam_cb.pack(side="left", padx=(4, 8))
        self.webcam_btn = ttk.Button(camrow, text="Record with webcam...",
                                     command=self.start_webcam_record)
        self.webcam_btn.pack(side="left")
        g2.columnconfigure(0, weight=1)

        g3 = ttk.LabelFrame(left, text="3. END gate & tracking mode", padding=8)
        g3.pack(fill="x", pady=(0, 8))
        self.mark_btn = ttk.Button(g3, text="Mark the END gate point",
                                   style="Primary.TButton",
                                   command=self.mark_gates)
        self.mark_btn.grid(row=0, column=0, sticky="w", padx=(0, 20))
        self.mode_var = tk.StringVar(value="box")
        ttk.Radiobutton(g3, text="Color-based", value="color",
                        variable=self.mode_var,
                        command=self._update_step_states).grid(row=0, column=1, sticky="w")
        self.pick_color_btn = ttk.Button(g3, text="Pick object color...",
                                         command=self.pick_color)
        self.pick_color_btn.grid(row=0, column=2, sticky="w", padx=10)
        ttk.Radiobutton(g3, text="Box tracker (draw a box)", value="box",
                        variable=self.mode_var,
                        command=self._update_step_states).grid(row=1, column=1, sticky="w")
        self.box_btn = ttk.Button(g3, text="Set up box & END gate...",
                                  style="Primary.TButton", command=self.setup_box)
        self.box_btn.grid(row=1, column=2, sticky="w", padx=10)
        sf = ttk.Frame(g3)
        sf.grid(row=2, column=1, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Label(sf, text="Start frame:").pack(side="left")
        self.box_start_var = tk.StringVar(value="0")
        self.box_start_spin = ttk.Spinbox(sf, from_=0, to=100000, width=7,
                                          textvariable=self.box_start_var,
                                          command=self._box_reload_frame)
        self.box_start_spin.pack(side="left", padx=4)
        self.box_start_spin.bind("<Return>", lambda e: self._box_reload_frame())
        self.box_info_var = tk.StringVar(value="")
        ttk.Label(g3, textvariable=self.box_info_var, style="Muted.TLabel").grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(4, 0))

        psp = ttk.Frame(g3)
        psp.grid(row=4, column=0, columnspan=3, sticky="w", pady=(8, 0))
        self.psp_btn = (
            ttk.Button(psp, text="Set perspective reference...",
                       command=self.setup_perspective),
            ttk.Button(psp, text="Clear perspective",
                       command=self.clear_perspective),
        )
        self.psp_btn[0].pack(side="left")
        self.psp_btn[1].pack(side="left", padx=6)
        self.persp_info_var = tk.StringVar(value="Perspective: off")
        ttk.Label(psp, textvariable=self.persp_info_var, style="Muted.TLabel").pack(
            side="left", padx=6)

        g4 = ttk.LabelFrame(left, text="4. Track & measure", padding=8)
        g4.pack(fill="x", pady=(0, 8))
        self.show_preview = tk.BooleanVar(value=True)
        ttk.Checkbutton(g4, text="Preview live frames while tracking",
                        variable=self.show_preview,
                        command=self._update_step_states).grid(
            row=0, column=0, sticky="w")
        self.run_btn = ttk.Button(g4, text="Track & measure crossing time",
                                  style="Accent.TButton",
                                  command=self.run_tracking)
        self.run_btn.grid(row=0, column=1, sticky="w", padx=14)
        self.release_thresh_var = tk.StringVar(value="2.0")
        ttk.Label(g4, text="Release threshold (px):").grid(
            row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Spinbox(g4, from_=0.0, to=50.0, increment=0.5, width=6,
                    textvariable=self.release_thresh_var).grid(
            row=1, column=1, sticky="w", padx=14, pady=(8, 0))
        self.result_var = tk.StringVar(value="No measurement yet.")
        ttk.Label(g4, textvariable=self.result_var, foreground=PALETTE["primary"],
                  font=self.font_bold).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))

        g5 = ttk.LabelFrame(left, text="5. Save result as a trial", padding=8)
        g5.pack(fill="x")
        self.trial_num_var = tk.StringVar(value="1")
        ttk.Label(g5, text="Trial:").pack(side="left")
        self.trial_cb = ttk.Combobox(g5, values=("1", "2", "3"),
                                     textvariable=self.trial_num_var,
                                     state="readonly", width=5)
        self.trial_cb.pack(side="left", padx=6)
        self.save_btn = ttk.Button(g5, text="Save trial", style="Accent.TButton",
                                   command=self.save_trial)
        self.save_btn.pack(side="left")

        # ---------------- right: embedded preview ----------------
        self.video_msg_var = tk.StringVar(
            value="Load a video or start the webcam; the live preview "
                  "will appear here.")
        ttk.Label(right, textvariable=self.video_msg_var, wraplength=700,
                  style="Muted.TLabel").grid(row=0, column=0, sticky="w")

        self.video_label = tk.Label(right, text="\n\nNo video loaded\n",
                                    anchor="center", justify="center",
                                    relief="flat", borderwidth=0,
                                    bg="#10162C", fg="#9AA6C7",
                                    font=self.font_base)
        self.video_label.grid(row=1, column=0, sticky="nsew", pady=(6, 6))
        self.video_label.bind("<Button-1>", self._on_preview_click)
        self.video_label.bind("<B1-Motion>", self._on_preview_drag)
        self.video_label.bind("<ButtonRelease-1>", self._on_preview_release)

        self.toolbars = {}
        self._build_cam_toolbar(right)
        self._build_mark_toolbar(right)
        self._build_review_toolbar(right)

    def _build_cam_toolbar(self, parent):
        tb = ttk.Frame(parent)
        self.cam_rec_var = tk.StringVar(value="Start recording")
        self.cam_rec_btn = ttk.Button(tb, textvariable=self.cam_rec_var,
                                      style="Accent.TButton",
                                      command=self.on_cam_rec_toggle)
        self.cam_rec_btn.pack(side="left")
        ttk.Button(tb, text="Finish & use this video", style="Primary.TButton",
                   command=self.on_cam_finish).pack(side="left", padx=6)
        ttk.Button(tb, text="Cancel", style="Danger.TButton",
                   command=self.on_cam_cancel).pack(side="left")
        self.cam_info_var = tk.StringVar(value="")
        ttk.Label(tb, textvariable=self.cam_info_var, style="Muted.TLabel").pack(
            side="left", padx=10)
        self.toolbars["cam"] = tb

    def _build_mark_toolbar(self, parent):
        tb = ttk.Frame(parent)
        ttk.Button(tb, text="Reset points",
                   command=self.on_mark_reset).pack(side="left")
        self.mark_confirm_btn = ttk.Button(tb, text="Confirm gates",
                                           style="Accent.TButton",
                                           command=self.on_mark_confirm)
        self.mark_confirm_btn.pack(side="left", padx=6)
        ttk.Button(tb, text="Cancel", style="Danger.TButton",
                   command=self.on_mark_cancel).pack(side="left")
        self.mark_info_var = tk.StringVar(value="")
        ttk.Label(tb, textvariable=self.mark_info_var, style="Muted.TLabel").pack(
            side="left", padx=10)
        self.toolbars["mark"] = tb

    def _build_review_toolbar(self, parent):
        tb = ttk.Frame(parent)
        ttk.Button(tb, text="|<\u23ee", width=3,
                   command=lambda: self.review_step(-10)).pack(side="left")
        ttk.Button(tb, text="<\u25c0", width=3,
                   command=lambda: self.review_step(-1)).pack(side="left", padx=2)
        self.review_play_var = tk.StringVar(value="\u25b6 Play")
        self.review_play_btn = ttk.Button(tb, textvariable=self.review_play_var,
                                          width=7, style="Accent.TButton",
                                          command=self.review_play_toggle)
        self.review_play_btn.pack(side="left", padx=2)
        ttk.Button(tb, text="\u25b6>", width=3,
                   command=lambda: self.review_step(1)).pack(side="left", padx=2)
        ttk.Button(tb, text="\u23ed>|", width=3,
                   command=lambda: self.review_step(10)).pack(side="left")
        self.review_slider_var = tk.IntVar(value=0)
        self.review_slider = ttk.Scale(tb, from_=0, to=100, orient="horizontal",
                                       variable=self.review_slider_var,
                                       command=self.on_review_scrub, length=320)
        self.review_slider.pack(side="left", padx=10)
        self.review_count_var = tk.StringVar(value="0/0")
        ttk.Label(tb, textvariable=self.review_count_var).pack(side="left", padx=4)
        ttk.Button(tb, text="\u2715 Close", style="Danger.TButton",
                   command=self.close_review).pack(side="left", padx=10)
        self.toolbars["review"] = tb

    def _build_results_tab(self, parent):
        ttk.Label(parent, text="Results table (click Refresh after adding trials):",
                  font=self.font_h2).pack(anchor="w")
        wrap = ttk.Frame(parent)
        wrap.pack(fill="both", expand=True, pady=(6, 6))
        self.results_tree = ttk.Treeview(wrap, show="headings")
        ys = ttk.Scrollbar(wrap, orient="vertical", command=self.results_tree.yview)
        xs = ttk.Scrollbar(wrap, orient="horizontal", command=self.results_tree.xview)
        self.results_tree.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        self.results_tree.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="we")
        wrap.rowconfigure(0, weight=1)
        wrap.columnconfigure(0, weight=1)

        btnrow = ttk.Frame(parent)
        btnrow.pack(fill="x")
        ttk.Button(btnrow, text="Refresh table",
                   command=self.refresh_results).pack(side="left")
        ttk.Button(btnrow, text="Export to Excel/CSV...",
                   style="Primary.TButton",
                   command=self.export_results).pack(side="left", padx=8)

    # ------------------------------------------------------------------
    # Solid list / configuration
    # ------------------------------------------------------------------
    def refresh_solid_list(self):
        labels = []
        for s in self.data["solids"]:
            filled = sum(1 for t in s["trials_ms"] if t is not None)
            name = s["name"] or "(unconfigured)"
            labels.append(f"Slot {s['slot']}: {name}  [{filled}/3]")
        self.solid_cb.configure(values=labels)

        self.solid_tree.delete(*self.solid_tree.get_children())
        self.solid_tree.tag_configure("odd", background=PALETTE["row_alt"])
        self.solid_tree.tag_configure("even", background=PALETTE["surface"])
        for i, s in enumerate(self.data["solids"]):
            filled = sum(1 for t in s["trials_ms"] if t is not None)
            self.solid_tree.insert("", "end", iid=str(s["slot"]),
                                   values=(s["slot"], s["name"] or "(unconfigured)",
                                           f"{filled}/3 trials"),
                                   tags=("odd" if i % 2 else "even",))
        if self.solid_tree.get_children():
            first = self.solid_tree.get_children()[0]
            self.solid_tree.selection_set(first)
            self._fill_cfg_form(int(first))
        self._update_step_states()

    def on_solid_select(self, event):
        sel = self.solid_tree.selection()
        if sel:
            self._fill_cfg_form(int(sel[0]))

    def open_friction_gui(self):
        try:
            import friction_gui
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Friction-slope analyser",
                                 "Could not load friction_gui:\n" + str(exc))
            return
        if getattr(self, "_friction_win", None) is not None:
            try:
                if self._friction_win.winfo_exists():
                    self._friction_win.lift()
                    return
            except tk.TclError:
                pass
        self._friction_win = friction_gui.launch(self)

    def _fill_cfg_form(self, slot):
        s = dm.get_solid(self.data, slot)
        self.cfg_vars["name"].set(s["name"] or "")
        for k in ("mass_kg", "radius_m", "theta_deg", "distance_m", "I_theory",
                  "mu_friction", "cd_drag"):
            v = s.get(k)
            self.cfg_vars[k].set("" if v is None else f"{v:g}")
        self.use_friction_var.set(bool(s.get("use_friction", False)))
        self.use_drag_var.set(bool(s.get("use_drag", False)))

    def save_solid(self):
        sel = self.solid_tree.selection()
        if not sel:
            messagebox.showinfo("Nothing selected", "Click a solid in the list first.")
            return
        s = dm.get_solid(self.data, int(sel[0]))
        s["name"] = self.cfg_vars["name"].get().strip() or None
        s["use_friction"] = bool(self.use_friction_var.get())
        s["use_drag"] = bool(self.use_drag_var.get())
        for k in ("mass_kg", "radius_m", "theta_deg", "distance_m", "I_theory"):
            text = self.cfg_vars[k].get().strip()
            if not text:
                s[k] = None
                continue
            try:
                s[k] = float(text)
            except ValueError:
                messagebox.showerror("Bad number",
                                     f"'{text}' is not a valid number for {k}.")
                return
        for k in ("mu_friction", "cd_drag"):
            text = self.cfg_vars[k].get().strip()
            s[k] = float(text) if text else 0.0
            if s[k] < 0:
                messagebox.showerror("Bad value", f"{k} must be >= 0.")
                return
        dm.save_data(self.data)
        self.refresh_all()
        self.status("Configuration saved.")

    def clear_trials(self):
        sel = self.solid_tree.selection()
        if not sel:
            return
        slot = int(sel[0])
        if not messagebox.askyesno("Clear trials",
                                   f"Clear all recorded trials for slot {slot}?"):
            return
        dm.get_solid(self.data, slot)["trials_ms"] = [None, None, None]
        dm.save_data(self.data)
        self.refresh_all()
        self.status(f"Trials for slot {slot} cleared.")

    # ------------------------------------------------------------------
    # Embedded preview plumbing
    # ------------------------------------------------------------------
    def _show_bgr(self, frame, draw=None):
        """Downscale to MAX_W, apply an optional overlay draw(frame), display."""
        if draw is not None:
            frame = frame.copy()
            draw(frame)
        h, w = frame.shape[:2]
        maxw = min(MAX_W, max(320, self.video_label.winfo_width()))
        if w > maxw:
            s = maxw / w
            disp = cv2.resize(frame, (maxw, int(h * s)))
        else:
            disp = frame
        self._img_scale = w / disp.shape[1]
        self._disp_size = (disp.shape[1], disp.shape[0])
        self.video_photo = self._to_photo(disp)
        self.video_label.configure(image=self.video_photo, text="")

    def _to_photo(self, bgr):
        """Turn a BGR frame into a tk photo quickly (PIL direct, else PPM/PNG)."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if HAVE_PIL:
            return PIL.ImageTk.PhotoImage(PIL.Image.fromarray(rgb))
        ok, buf = cv2.imencode(".ppm", rgb)
        if ok:
            return tk.PhotoImage(data=buf.tobytes())
        ok, buf = cv2.imencode(".png", rgb)
        return tk.PhotoImage(data=buf.tobytes())

    def _clear_preview(self, message):
        self.video_photo = None
        self.video_label.configure(image="", text=message, compound="center")

    def _event_xy(self, event):
        dw, dh = self._disp_size
        ow = self.video_label.winfo_width()
        oh = self.video_label.winfo_height()
        return (int((event.x - (ow - dw) // 2) * self._img_scale),
                int((event.y - (oh - dh) // 2) * self._img_scale))

    def _on_preview_click(self, event):
        sx, sy = self._event_xy(event)
        if self.ui_mode == "mark" and self.mark_mode:
            if len(self.mark_mode["points"]) < 1:
                self.mark_mode["points"].append((sx, sy))
                self._redraw_mark()
        elif self.ui_mode == "persp" and self.persp_mode:
            if len(self.persp_mode["points"]) < 4:
                self.persp_mode["points"].append((sx, sy))
                self._redraw_persp()
        elif self.ui_mode == "pick" and self.pick_mode:
            self._sample_color_click(sx, sy)
        elif self.ui_mode == "boxsetup" and self.box_setup:
            bs = self.box_setup
            if bs["stage"] == "box":
                bs["drag"] = (sx, sy)
                bs["drag_now"] = (sx, sy)
            elif len(bs["gates"]) < 1:
                bs["gates"].append((sx, sy))
                bs["stage"] = "done"
                self._redraw_boxsetup()

    def _on_preview_drag(self, event):
        bs = self.box_setup
        if self.ui_mode == "boxsetup" and bs and bs.get("drag"):
            bs["drag_now"] = self._event_xy(event)
            self._redraw_boxsetup()

    def _on_preview_release(self, event):
        bs = self.box_setup
        if self.ui_mode != "boxsetup" or not bs or not bs.get("drag"):
            return
        (x0, y0), (x1, y1) = bs["drag"], self._event_xy(event)
        bs["drag"] = bs["drag_now"] = None
        x, y, w, h = min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)
        if w >= 4 and h >= 4:
            bs["box"] = (x, y, w, h)
            bs["stage"] = "gate"
        self._redraw_boxsetup()

    def _show_toolbar(self, name):
        for tb in self.toolbars.values():
            tb.grid_remove()
        if name and name in self.toolbars:
            self.toolbars[name].grid(row=2, column=0, sticky="ew", pady=(6, 0))

    # ------------------------------------------------------------------
    # Optional: 4-point perspective correction
    # ------------------------------------------------------------------
    def setup_perspective(self):
        if not self.video_path:
            return
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            messagebox.showerror("Perspective",
                                 "Could not read the first video frame.")
            return
        self.close_review()
        self.ui_mode = "persp"
        self.persp_mode = {"frame": frame, "points": []}
        self.video_msg_var.set(
            "Click the 4 corners of a rectangle on the TABLE plane "
            "(e.g. the table top - a rectangle you can measure with a ruler).\n"
            "Order: top-left, top-right, bottom-right, bottom-left.")
        self.mark_info_var.set("0/4 corners placed")
        self.mark_confirm_btn.configure(text="Confirm & compute", state="disabled")
        self.video_label.configure(cursor="crosshair")
        self._show_toolbar("mark")
        self._redraw_persp()

    def _redraw_persp(self):
        pm = self.persp_mode
        if pm is None:
            return
        pts = pm["points"]
        labels = ["TL", "TR", "BR", "BL"]

        def draw(f):
            for i, p in enumerate(pts):
                cv2.circle(f, p, 8, (0, 0, 255), -1)
                cv2.putText(f, labels[i], (p[0] + 12, p[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if len(pts) == 4:
                cv2.polylines(f, [np.array(pts, dtype=np.int32)], True,
                              (255, 0, 0), 2)
                cv2.putText(f, "Confirm to compute the homography",
                            (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 0, 0), 2)

        self._show_bgr(pm["frame"], draw=draw)
        self.mark_info_var.set(f"{len(pts)}/4 corners placed")
        self.mark_confirm_btn.configure(state="normal" if len(pts) == 4 else "disabled")

    def _confirm_perspective(self):
        pm = self.persp_mode
        if not pm or len(pm["points"]) != 4:
            return
        try:
            w = simpledialog.askfloat(
                "Reference rectangle width",
                "Physical WIDTH of the marked rectangle (metres):",
                parent=self, minvalue=0.01, initialvalue=1.0)
            if not w:  # cancelled
                return
            h = simpledialog.askfloat(
                "Reference rectangle height",
                "Physical HEIGHT of the marked rectangle (metres):\n"
                "(along the direction that is 'up' in the table plane)",
                parent=self, minvalue=0.01, initialvalue=0.6)
            if not h:
                return
        except Exception:  # noqa: BLE001
            return
        try:
            self.H = perspective.homography_from_rect(pm["points"], w, h)
        except cv2.error as exc:
            self.H = None
            messagebox.showerror("Perspective",
                                 "Could not build the homography:\n" + str(exc))
            return
        self.persp_src = list(pm["points"])
        self.persp_mode = None
        self.ui_mode = "idle"
        self.video_label.configure(cursor="")
        self._show_toolbar(None)
        self.persp_info_var.set(f"Perspective: ON  ({w:g} x {h:g} m rectangle)")
        self.video_msg_var.set(
            "Perspective correction set: the END gate is now measured "
            "perpendicular to the motion in the table plane, so the gate line "
            "is correct even when the camera is angled. Mark the END gate / "
            "box as usual.")
        self.status("Perspective homography set (4 reference corners).")
        self._update_step_states()

    def clear_perspective(self):
        self.H = None
        self.persp_src = None
        self.persp_mode = None
        self.persp_info_var.set("Perspective: off")
        self.status("Perspective correction cleared.")
        self._update_step_states()

    # ------------------------------------------------------------------
    # STEP 2a: embedded gate marking
    # ------------------------------------------------------------------
    def mark_gates(self):
        if not self.video_path:
            return
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            messagebox.showerror("Mark gates", "Could not read the first video frame.")
            return
        self.close_review()
        self.ui_mode = "mark"
        self.mark_mode = {"frame": frame, "points": []}
        self.video_msg_var.set(
            "Click once on the END gate point in the preview (where the "
            "crossing time is measured to).")
        self.mark_info_var.set("0/1 point placed")
        self.mark_confirm_btn.configure(text="Confirm gate", state="disabled")
        self.video_label.configure(cursor="crosshair")
        self._show_toolbar("mark")
        self._redraw_mark()

    def _redraw_mark(self):
        mm = self.mark_mode
        pts = mm["points"]

        def draw(f):
            if pts:
                p = pts[0]
                cv2.circle(f, p, 8, (0, 0, 255), -1)
                cv2.putText(f, "END", (p[0] + 12, p[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        self._show_bgr(mm["frame"], draw=draw)
        self.mark_info_var.set(f"{len(pts)}/1 point placed")
        self.mark_confirm_btn.configure(state="normal" if len(pts) == 1 else "disabled")
        if len(pts) == 1:
            self.on_mark_confirm()

    def on_mark_reset(self):
        if self.ui_mode == "boxsetup":
            self.box_setup.update(box=None, gates=[], stage="box", drag=None)
            self._redraw_boxsetup()
            return
        if self.ui_mode == "persp":
            self.persp_mode["points"] = []
            self._redraw_persp()
            return
        if self.mark_mode:
            self.mark_mode["points"] = []
            self._redraw_mark()

    def on_mark_cancel(self):
        if self.ui_mode == "boxsetup":
            self.box_setup = None
            self.video_msg_var.set("Box setup cancelled.")
            self.status("Box setup cancelled.")
        elif self.ui_mode == "persp":
            self.persp_mode = None
            self.video_msg_var.set("Perspective setup cancelled.")
            self.status("Perspective setup cancelled.")
        elif self.ui_mode == "pick":
            self.pick_mode = None
            self.pending_color = None
            self.video_msg_var.set("Color sampling cancelled.")
            self.status("Color sampling cancelled.")
        else:
            self.mark_mode = None
            self.video_msg_var.set("Gate marking cancelled.")
            self.status("Gate marking cancelled.")
        self.ui_mode = "idle"
        self.video_label.configure(cursor="")
        self._show_toolbar(None)

    def on_mark_confirm(self):
        if self.ui_mode == "boxsetup":
            self._confirm_boxsetup()
            return
        if self.ui_mode == "persp":
            self._confirm_perspective()
            return
        if self.ui_mode == "pick":
            self.on_mark_confirm_shared()
            return
        if not self.mark_mode or len(self.mark_mode["points"]) != 1:
            return
        self.gate_points = list(self.mark_mode["points"])
        self.mark_mode = None
        self.ui_mode = "idle"
        self.video_label.configure(cursor="")
        self._show_toolbar(None)
        ex, ey = self.gate_points[0]
        self.video_msg_var.set(
            f"END gate set at ({ex}, {ey}). Timing starts when the object "
            "starts moving and stops at this gate.")
        self.status("END gate confirmed. Choose tracking mode and run step 4.")
        self._update_step_states()

    # ------------------------------------------------------------------
    # STEP 2c: embedded box + gate setup (box tracker mode)
    # ------------------------------------------------------------------
    @staticmethod
    def _read_frame_seq(path, idx):
        """Sequential read so the frame numbering matches box_tracker exactly."""
        cap = cv2.VideoCapture(path)
        frame = None
        for _ in range(idx + 1):
            ret, f = cap.read()
            if not ret:
                break
            frame = f
        cap.release()
        return frame

    def _box_start_idx(self):
        try:
            return max(0, int(float(self.box_start_var.get())))
        except ValueError:
            return 0

    def setup_box(self):
        if not self.video_path:
            return
        idx = self._box_start_idx()
        frame = self._read_frame_seq(self.video_path, idx)
        if frame is None:
            messagebox.showerror("Box setup", f"Could not read frame {idx}.")
            return
        self.close_review()
        self.ui_mode = "boxsetup"
        self.box_setup = {"frame": frame, "idx": idx, "stage": "box", "box": None,
                          "gates": [], "drag": None, "drag_now": None}
        self.mark_confirm_btn.configure(text="Confirm box & gate", state="disabled")
        self.video_label.configure(cursor="crosshair")
        self._show_toolbar("mark")
        self._redraw_boxsetup()

    def _box_reload_frame(self):
        """Start-frame spinbox changed: show that frame (keeps setup mode)."""
        if not self.video_path:
            return
        if self.ui_mode == "boxsetup" and self.box_setup:
            frame = self._read_frame_seq(self.video_path, self._box_start_idx())
            if frame is not None:
                self.box_setup.update(frame=frame, idx=self._box_start_idx())
                self._redraw_boxsetup()
        elif self.ui_mode in ("idle",):
            self.setup_box()

    def _redraw_boxsetup(self):
        bs = self.box_setup
        if bs is None:
            return
        msgs = {"box": "Drag a TIGHT box around the object at its resting position.",
                "gate": "Click the END gate on the path (where timing stops).",
                "done": "Ready - press 'Confirm box & gate'."}
        self.video_msg_var.set(f"[frame {bs['idx']}] " + msgs[bs["stage"]])

        def draw(f):
            box = bs["box"]
            if bs.get("drag") and bs.get("drag_now"):
                (x0, y0), (x1, y1) = bs["drag"], bs["drag_now"]
                box = (min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
            if box:
                x, y, w, h = box
                cv2.rectangle(f, (x, y), (x + w, y + h), (255, 200, 0), 2)
            g = bs["gates"]
            for i, p in enumerate(g):
                col = (0, 255, 0)
                cv2.circle(f, p, 6, col, -1)
                cv2.putText(f, "END", (p[0] + 10, p[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            if len(g) == 1 and box is not None:
                try:
                    _, u, _ = box_tracker.gate_axis(box, g[0], H=self.H)
                    box_tracker._draw_gate_line(f, g[0], u, (0, 255, 0), H=self.H)
                except ValueError:
                    pass
        self._show_bgr(bs["frame"], draw=draw)
        ok = bs["box"] is not None and len(bs["gates"]) == 1
        self.mark_confirm_btn.configure(state="normal" if ok else "disabled")
        self.mark_info_var.set(f"box: {'set' if bs['box'] else '-'}   END gate: {len(bs['gates'])}/1")

    def _confirm_boxsetup(self):
        bs = self.box_setup
        if not bs or bs["box"] is None or len(bs["gates"]) != 1:
            return
        try:
            box_tracker.gate_axis(bs["box"], bs["gates"][0], H=self.H)
        except ValueError as exc:
            messagebox.showerror("Gate", str(exc))
            return
        self.box, self.box_gates, self.box_start = bs["box"], list(bs["gates"]), bs["idx"]
        self.box_setup = None
        self.ui_mode = "idle"
        self.video_label.configure(cursor="")
        self._show_toolbar(None)
        _, _, level = box_tracker.gate_axis(self.box, self.box_gates[0], H=self.H)
        self.box_info_var.set(f"Box {self.box[2]}x{self.box[3]} px, gate distance {level:.0f} px, "
                              f"start frame {self.box_start}"
                              + ("  [perspective corrected]" if self.H is not None else ""))
        self.video_msg_var.set("Box & END gate set. Run step 4 to track and measure.")
        self.status("Box & END gate confirmed.")
        self._update_step_states()

    # ------------------------------------------------------------------
    # STEP 2b: embedded color sampling
    # ------------------------------------------------------------------
    def pick_color(self):
        if not self.video_path:
            return
        cap = cv2.VideoCapture(self.video_path)
        frame = None
        for _ in range(8):
            ret, frame = cap.read()
            if not ret:
                break
        cap.release()
        if frame is None:
            messagebox.showerror("Color sampling",
                                 "Could not read a frame for color sampling.")
            return
        self.close_review()
        self.ui_mode = "pick"
        self.pick_mode = {"frame": frame,
                          "hsv": cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)}
        self.pending_color = None
        self.video_msg_var.set(
            "Click once on the moving object in the preview to sample its color, "
            "then click 'Confirm color'.")
        self.mark_info_var.set("No sample yet")
        self.mark_confirm_btn.configure(text="Confirm color", state="disabled")
        self.video_label.configure(cursor="crosshair")
        self._show_toolbar("mark")
        self._show_bgr(self.pick_mode["frame"])

    def _sample_color_click(self, x, y):
        pm = self.pick_mode
        h, w = pm["hsv"].shape[:2]
        box = 6
        x0, x1 = max(0, x - box), min(w, x + box)
        y0, y1 = max(0, y - box), min(h, y + box)
        patch = pm["hsv"][y0:y1, x0:x1].reshape(-1, 3)
        lower, upper = tracker.color_range_from_patch(patch)
        self.pending_color = (lower, upper)

        def draw(f):
            cv2.circle(f, (x, y), box, (0, 255, 0), 2)
            cv2.circle(f, (x, y), box * 3, (0, 255, 0), 1)
        self._show_bgr(pm["frame"], draw=draw)
        self.mark_info_var.set(
            f"Sampled HSV {tuple(lower)} .. {tuple(upper)}")
        self.mark_confirm_btn.configure(state="normal")

    def on_mark_confirm_shared(self):
        if self.ui_mode == "pick" and self.pending_color is not None:
            self.color_range = self.pending_color
            self.pick_mode = None
            self.ui_mode = "idle"
            self.video_label.configure(cursor="")
            self._show_toolbar(None)
            self.video_msg_var.set(
                "Color range sampled. Keep color mode and run step 4.")
            self.status("Color range set.")
            self._update_step_states()

    # ------------------------------------------------------------------
    # STEP: embedded webcam recording
    # ------------------------------------------------------------------
    def start_webcam_record(self):
        s = self._record_solid()
        if s is None:
            messagebox.showinfo("Select a solid", "Pick a solid in step 1 first.")
            return
        try:
            cam_idx = int(self.cam_index_var.get())
        except ValueError:
            cam_idx = 0
        cap = cv2.VideoCapture(cam_idx)
        if not cap.isOpened():
            cap.release()
            messagebox.showerror("Webcam",
                                 f"Could not open webcam #{cam_idx}.")
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
            messagebox.showerror(
                "Webcam",
                f"Webcam #{cam_idx} opened but produced no frames.\n"
                "It may be in use by another app or is not a working camera.\n"
                "Try a different webcam index in the dropdown.")
            return
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0
        h, w = probe.shape[:2]
        safe = "".join(c if c.isalnum() else "_" for c in (s["name"] or "unknown"))
        out_path = f"video_slot{s['slot']}_{safe}.avi"

        writer, out_path = self._make_writer(out_path, fps, w, h)
        if writer is None:
            cap.release()
            messagebox.showerror("Webcam", f"Could not create video writer for {out_path}.")
            return

        self.cam = {
            "cap": cap, "fps": fps, "w": w, "h": h,
            "q": queue.Queue(maxsize=1),
            "stop": threading.Event(),
            "rec": False, "rec_start": 0.0, "writer": writer,
            "out_path": out_path, "has_file": False, "failed": False,
        }
        self.ui_mode = "webcam"
        self.cam_rec_var.set("Start recording")
        self.cam_info_var.set(f"{w}x{h} @~{fps:.0f} fps")
        self.video_msg_var.set(
            "Webcam live. Press 'Start recording' (●) to capture. "
            "Watch the red REC timer in the corner.")
        self._show_toolbar("cam")
        self._update_step_states()

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
                    if cam["rec"]:
                        elapsed = time.monotonic() - cam["rec_start"]
                    else:
                        elapsed = 0.0
                    try:
                        cam["q"].put_nowait((frame, cam["rec"], elapsed))
                    except queue.Full:
                        pass
            finally:
                cam["cap"].release()

        self.cam["thread"] = threading.Thread(target=loop, daemon=True)
        self.cam["thread"].start()
        self.after(33, self._poll_webcam)

    def _make_writer(self, path, fps, w, h):
        for codec in ("XVID", "MJPG", "mp4v"):
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = None
            try:
                writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
            except cv2.error:
                writer = None
            if writer is not None and writer.isOpened():
                return writer, path
            if writer is not None:
                writer.release()
        return None, path

    def on_cam_rec_toggle(self):
        cam = self.cam
        if cam is None:
            return
        if not cam["rec"]:
            cam["rec"] = True
            cam["rec_start"] = time.monotonic()
            cam["has_file"] = True
            self.cam_rec_var.set("Stop recording")
            self.video_msg_var.set("Recording... press 'Stop recording' to finish.")
        else:
            cam["rec"] = False
            self.cam_rec_var.set("Start recording")
            self.cam_rec_btn.configure(state="normal")
            self.video_msg_var.set(
                "Recording stopped. Press 'Start recording' to re-record "
                "(overwrites the last take), or 'Finish & use this video' to "
                "continue the experiment.")
        self._poll_webcam()

    def _finish_cam_session(self, keep_file):
        cam = self.cam
        if cam is None:
            return
        cam["stop"].set()
        cam["rec"] = False
        thread = cam.get("thread")
        if thread is not None:
            thread.join(timeout=3)
        if cam["writer"] is not None:
            cam["writer"].release()
            cam["writer"] = None
        if not keep_file and os.path.exists(cam["out_path"]):
            try:
                os.remove(cam["out_path"])
            except OSError:
                pass
        self.cam = None
        self.ui_mode = "idle"
        self._show_toolbar(None)
        self._update_step_states()

    def on_cam_finish(self):
        cam = self.cam
        if cam is None:
            return
        if cam["rec"]:
            messagebox.showinfo("Still recording",
                                "Press 'Stop recording' first.")
            return
        if not cam["has_file"]:
            messagebox.showinfo("Nothing recorded",
                                "Press 'Start recording' to capture a video first, "
                                "then stop it.")
            return
        out = cam["out_path"]
        self._finish_cam_session(keep_file=True)
        if not os.path.exists(out) or os.path.getsize(out) == 0:
            self._clear_preview("The recorded file is empty. Please retry.")
            self.video_msg_var.set("Recorded file was empty - please retry.")
            self.status("Empty recording discarded.")
            return
        self._set_video(out)
        self.video_msg_var.set("Webcam recording finished and loaded.")
        self.status("Webcam video ready. Mark the END gate point (step 3).")

    def on_cam_cancel(self):
        cam = self.cam
        if cam is None:
            return
        self._finish_cam_session(keep_file=False)
        self._clear_preview("Webcam session cancelled. Load a video or retry.")
        self.status("Webcam recording cancelled.")

    def _poll_webcam(self):
        cam = self.cam
        if cam is not None:
            if cam.get("failed"):
                self._clear_preview("Webcam stopped producing frames.")
                messagebox.showwarning(
                    "Webcam",
                    "The webcam stopped producing frames.\n"
                    "It may be in use by another app or a wrong index was chosen.\n"
                    "Click 'Record with webcam...' again to retry.")
                self._finish_cam_session(keep_file=False)
                self.status("Webcam failed to capture frames.")
                return
            try:
                payload = cam["q"].get_nowait()
            except queue.Empty:
                payload = None
            if payload is not None:
                frame, rec, elapsed = payload

                def draw(f, rec=rec, elapsed=elapsed):
                    self._draw_rec_overlay(f, rec, elapsed)
                self._show_bgr(frame, draw=draw)
            self.after(33, self._poll_webcam)

    def _draw_rec_overlay(self, frame, rec, elapsed):
        h, w = frame.shape[:2]
        if rec:
            cv2.circle(frame, (20, h - 30), 10, (0, 0, 255), -1)
            mm, ss, tt = int(elapsed // 60), int(elapsed % 60), int(elapsed * 10 % 10)
            t = f"REC  {mm:02d}:{ss:02d}.{tt}"
            cv2.putText(frame, t, (40, h - 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 0, 255), 2)
        else:
            cv2.circle(frame, (20, h - 30), 10, (0, 255, 0), -1)
            cv2.putText(frame, "LIVE", (40, h - 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)

    # ------------------------------------------------------------------
    # STEP 4: tracking (background thread) + embedded live preview
    # ------------------------------------------------------------------
    def run_tracking(self):
        box_mode = self.mode_var.get() == "box"
        if box_mode and not (self.video_path and self.box and self.box_gates):
            self.status("Set up the box and the END gate first.")
            return
        if not box_mode and not (self.video_path and self.gate_points):
            self.status("First load a video and mark the END gate point.")
            return
        self.run_mode = self.mode_var.get()
        self.stop_event.clear()
        self.box_timeline = {}
        if self.mode_var.get() == "color" and self.color_range is None:
            messagebox.showinfo("Sample color first",
                                "Click 'Pick object color...' before tracking.")
            return
        self.close_review()
        self.result = None
        self.trajectory = None
        self._preview_enabled = self.show_preview.get()
        self._track_first = None
        self._track_released = False
        self._release_frame = None
        self.ui_mode = "track"
        self.video_msg_var.set("Tracking... live preview below.")
        self.result_var.set("Processing video...")
        self._cv2_worker(self._track_worker, self.on_tracking_done)

    def _fps_override(self):
        """Optional manual FPS from the 'Video FPS' field (None = use video FPS)."""
        try:
            v = float(self.fps_var.get())
        except ValueError:
            return None
        return v if v and v > 1 else None

    def _track_worker(self):
        fps = self._fps_override()
        if self.run_mode == "box":
            return box_tracker.track_box(
                self.video_path, self.box_start, self.box, self.box_gates,
                show=False, frame_cb=self._box_frame_cb, stop_event=self.stop_event,
                H=self.H, fps=fps)
        return tracker.track_object(
            self.video_path,
            color_range=self.color_range,
            show=False,
            frame_cb=self._track_frame_cb,
            fps=fps)

    def _track_frame_cb(self, frame, t_sec, centroid, frame_idx):
        if not getattr(self, "_preview_enabled", True):
            return
        try:
            self.cv2_q.put_nowait(("frame", (frame, t_sec, centroid, frame_idx)))
        except queue.Full:
            pass

    def _box_frame_cb(self, frame, idx, b):
        if not getattr(self, "_preview_enabled", True):
            return
        try:
            self.cv2_q.put_nowait(("boxframe", (frame.copy(), idx, b)))
        except queue.Full:
            pass

    def _on_box_done(self, track):
        if track is None:
            self.ui_mode = "idle"
            self.result_var.set("Tracking aborted.")
            self.status("Tracking aborted.")
            return
        self.fps = track["fps"]
        self.box_timeline = {self.box_start + i: bb for i, bb in enumerate(track["boxes"])}
        self.timeline = {}
        try:
            move_px = float(self.release_thresh_var.get())
        except ValueError:
            move_px = 2.0
        res = box_tracker.compute_crossing(track, self.box_gates[0], move_px=move_px,
                                           H=self.H)
        if res is None:
            self.result = None
            self.crossing_text = None
            self.ui_mode = "idle"
            why = (" The tracker lost the object early - draw a tighter box or start "
                   "closer to the release.") if track["lost"] else ""
            self.video_msg_var.set("Box tracker: END gate not clearly crossed." + why)
            self.result_var.set("Measurement failed - no clear crossing.")
            self.status("Measurement failed.")
            return
        self.result = res
        scaled = self._apply_slowmo(res)
        fps = track["fps"]
        slow = f"  [\u00f7{self.slowmo_mult:g} slow-mo]" if scaled else ""
        self.crossing_text = (f"RELEASE {res['start_time_ms']:.1f} | END {res['end_time_ms']:.1f} | "
                              f"\u0394 {res['elapsed_ms']} ms ({res['method']}){slow}")
        warn = "  [tracker lost object after END]" if track["lost"] else ""
        self.result_var.set(
            f"Crossing time (RELEASE \u2192 END gate) = {res['elapsed_ms']} ms  "
            f"[{track['tracker']}, {fps:.1f} fps, {res['method']}]{warn}{slow}")
        self.status("Measurement complete. Review, then choose a trial number and Save.")
        self._start_review()
        self._update_step_states()

    def _find_release_frame(self, trajectory, move_px):
        """Frame index where the blob first leaves its resting spot."""
        if not trajectory:
            return None
        cx0, cy0 = trajectory[0][1], trajectory[0][2]
        for (_t, cx, cy, fidx) in trajectory:
            if (abs(cx - cx0) >= move_px or abs(cy - cy0) >= move_px):
                return fidx
        return trajectory[0][3]

    def on_tracking_done(self, payload):
        if isinstance(payload, str) and payload.startswith("__comerror__"):
            self.ui_mode = "idle"
            self.result_var.set("Tracking failed.")
            messagebox.showerror("Tracking", payload)
            return
        if self.run_mode == "box":
            self._on_box_done(payload)
            return
        trajectory, fps = payload
        self.trajectory = trajectory
        self.fps = fps
        self.timeline = {fidx: (cx, cy) for (_t, cx, cy, fidx) in trajectory}
        try:
            move_px = float(self.release_thresh_var.get())
        except ValueError:
            move_px = 2.0
        self._release_frame = self._find_release_frame(trajectory, move_px)
        res = tracker.compute_crossing_time_ms(trajectory, self.gate_points,
                                               move_px=move_px, H=self.H)
        if res is None:
            self.result = None
            self.crossing_text = None
            self.ui_mode = "idle"
            self.video_msg_var.set(
                "No clear crossing detected. Re-mark gates, improve lighting, "
                "or re-sample the object color.")
            self.result_var.set("Measurement failed - no clear crossing.")
            self.status("Measurement failed.")
            return
        self.result = res
        scaled = self._apply_slowmo(res)
        slow = f"  [\u00f7{self.slowmo_mult:g} slow-mo]" if scaled else ""
        self.crossing_text = (f"RELEASE {res['start_time_ms']:.1f} ms | "
                              f"END {res['end_time_ms']:.1f} ms | "
                              f"\u0394 {res['elapsed_ms']} ms ({fps:.1f} fps){slow}")
        self.result_var.set(
            f"Crossing time (release \u2192 END gate) = {res['elapsed_ms']} ms  "
            f"(release {res['start_time_ms']} ms, END {res['end_time_ms']} ms, "
            f"video ~{fps:.1f} fps){slow}")
        self.status("Measurement complete. Use the frame player below to review, "
                    "then pick a trial number and Save.")
        self._start_review()
        self._update_step_states()

    # ------------------------------------------------------------------
    # STEP 4b: frame-by-frame review player
    # ------------------------------------------------------------------
    def _start_review(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            messagebox.showerror("Review", "Could not open video for review.")
            return
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        self.review = {
            "cap": cap, "fps": self.fps, "total": total,
            "idx": 0, "playing": False,
        }
        self.ui_mode = "review"
        self.review_play_var.set("\u25b6 Play")
        self.review_slider.configure(to=max(0, total - 1))
        self.review_slider_var.set(0)
        self.video_msg_var.set("Frame-by-frame review: play, step, or drag the "
                               "slider. Time is shown in the corner.")
        self._show_toolbar("review")
        self._load_review_frame(0)

    def _load_review_frame(self, idx):
        rv = self.review
        if rv is None:
            return
        idx = max(0, min(idx, rv["total"] - 1))
        rv["cap"].set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = rv["cap"].read()
        if not ret:
            return
        self._show_bgr(frame, draw=lambda f: self._draw_review(f))
        rv["idx"] = idx
        self.review_slider_var.set(idx)
        self.review_count_var.set(f"{idx}/{rv['total']}")

    def _draw_color_gate(self, frame, origin=None):
        """Draw the END gate for color mode. With a homography set, draws the true
        (perspective-corrected) perpendicular gate line using `origin` as the
        motion-axis reference (the object at rest); otherwise a plain vertical
        line through the marked point."""
        if not self.gate_points:
            return
        (ex, ey) = self.gate_points[0]
        cv2.circle(frame, (ex, ey), 8, (0, 0, 255), -1)
        if self.H is None:
            cv2.line(frame, (ex, 0), (ex, frame.shape[0]), (0, 0, 255), 1)
            return
        if origin is None:
            return
        o = perspective.transform_point(self.H, origin)
        g = perspective.transform_point(self.H, (ex, ey))
        d = g - o
        level = float(np.linalg.norm(d))
        if level < 1e-6:
            return
        u = d / level
        p1, p2 = box_tracker.gate_line_points((ex, ey), u, H=self.H, length=level * 1.2)
        cv2.line(frame, tuple(int(v) for v in p1), tuple(int(v) for v in p2),
                 (0, 0, 255), 1, cv2.LINE_AA)

    def _draw_review(self, frame):
        h, w = frame.shape[:2]
        if self.run_mode == "box":
            self._draw_review_box(frame)
            return
        origin = self.trajectory[0][1:3] if self.trajectory else None
        self._draw_color_gate(frame, origin=origin)
        rv = self.review
        fidx = rv["idx"]
        if fidx in self.timeline:
            cx, cy = self.timeline[fidx]
            colour = (0, 255, 0)
            if self._release_frame is not None and fidx >= self._release_frame:
                colour = (0, 255, 255)
            cv2.circle(frame, (int(cx), int(cy)), 6, colour, -1)
            cv2.circle(frame, (int(cx), int(cy)), 10, colour, 1)
        # corner timer (top-right) + frame info (top-left)
        t_sec = (fidx / rv["fps"] / self.slowmo_mult) if rv["fps"] else 0.0
        timer = f"t = {t_sec:.3f} s"
        (tw, th), _ = cv2.getTextSize(timer, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
        cv2.putText(frame, timer, (w - tw - 18, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 0), 2)
        slow = f" (\u00f7{self.slowmo_mult:g} slow-mo)" if self.slowmo_mult > 1 else ""
        cv2.putText(frame, f"frame {fidx}/{rv['total']} ({rv['fps']:.1f} fps){slow}",
                    (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        if self.crossing_text:
            cv2.putText(frame, self.crossing_text, (12, h - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 120, 0), 2)

    def _draw_review_box(self, frame):
        h, w = frame.shape[:2]
        rv = self.review
        fidx = rv["idx"]
        try:
            _, u, _ = box_tracker.gate_axis(self.box, self.box_gates[0], H=self.H)
            box_tracker._draw_gate_line(frame, self.box_gates[0], u, (0, 255, 0), H=self.H)
        except ValueError:
            pass
        bb = self.box_timeline.get(fidx)
        t_ms = (fidx / rv["fps"] * 1000.0 / self.slowmo_mult) if rv["fps"] else 0.0
        if self.result:
            r = self.result
            col = (255, 200, 0) if t_ms < r["start_time_ms"] else (
                (255, 255, 0) if t_ms <= r["end_time_ms"] else (0, 255, 0))
            state = ("at rest" if t_ms < r["start_time_ms"] else
                     "RELEASED / TIMING" if t_ms <= r["end_time_ms"] else "END crossed")
        else:
            col, state = (255, 200, 0), ""
        if bb is not None:
            x, y, bw, bh = [int(round(v)) for v in bb]
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), col, 2)
        timer = f"t = {t_ms / 1000:.3f} s"
        (tw, _), _ = cv2.getTextSize(timer, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
        cv2.putText(frame, timer, (w - tw - 18, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 0), 2)
        slow = f" (\u00f7{self.slowmo_mult:g} slow-mo)" if self.slowmo_mult > 1 else ""
        cv2.putText(frame, f"frame {fidx}/{rv['total']} ({rv['fps']:.1f} fps){slow}", (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        if self.result:
            cv2.putText(frame, state, (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        col, 2)
            cv2.circle(frame, (w - 150, 30), 8, (255, 200, 0), 2)
            cv2.circle(frame, (w - 105, 30), 8, (255, 255, 0), 2)
            cv2.circle(frame, (w - 60, 30), 8, (0, 255, 0), 2)
            cv2.putText(frame, "rest", (w - 150, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)
            cv2.putText(frame, "run", (w - 105, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            cv2.putText(frame, "done", (w - 60, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        if self.crossing_text:
            cv2.putText(frame, self.crossing_text, (12, h - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 120, 0), 2)

    def review_play_toggle(self):
        rv = self.review
        if rv is None:
            return
        rv["playing"] = not rv["playing"]
        self.review_play_var.set("\u23f8 Pause" if rv["playing"] else "\u25b6 Play")
        if rv["playing"]:
            self._review_tick()
        else:
            self._load_review_frame(rv["idx"])

    def _review_tick(self):
        rv = self.review
        if rv is None or not rv["playing"]:
            return
        nxt = rv["idx"] + 1
        if nxt >= rv["total"]:
            rv["playing"] = False
            self.review_play_var.set("\u25b6 Play")
            self._load_review_frame(rv["idx"])
            return
        self._load_review_frame(nxt)
        interval = max(20, int(1000 / rv["fps"]) if rv["fps"] else 33)
        self.after(interval, self._review_tick)

    def review_step(self, delta):
        rv = self.review
        if rv is None or self.cv2_busy:
            return
        rv["playing"] = False
        self.review_play_var.set("\u25b6 Play")
        self._load_review_frame(rv["idx"] + delta)

    def on_review_scrub(self, value):
        rv = self.review
        if rv is None:
            return
        self._load_review_frame(int(round(float(value))))

    def close_review(self):
        if self.review is not None:
            rv = self.review
            rv["playing"] = False
            rv["cap"].release()
            self.review = None
            if self.ui_mode == "review":
                self.ui_mode = "idle"
                self._show_toolbar(None)

    # ------------------------------------------------------------------
    # STEP 5: save result
    # ------------------------------------------------------------------
    def _slowmo_mult(self):
        """Slow-motion factor from the 'Slow-mo factor' spinbox (>=1)."""
        try:
            v = float(self.slowmo_var.get())
        except ValueError:
            v = 1.0
        return v if v and v >= 1 else 1.0

    def _apply_slowmo(self, res):
        """
        Divide the measured video times by the slow-motion factor so the
        reported / saved times are REAL times (slow-mo playback stretches
        time; e.g. an 8x slow-mo video shows 8x the true elapsed time).
        Mutates `res` in place. Returns True if a correction was applied.
        """
        mult = self._slowmo_mult()
        self.slowmo_mult = mult
        if res is None or mult <= 1.0:
            return False
        for k in ("start_time_ms", "end_time_ms", "elapsed_ms", "raw_elapsed_ms"):
            if k in res and res[k] is not None:
                res[k] = round(res[k] / mult, 3)
        return True

    def save_trial(self):
        if self.result is None:
            messagebox.showinfo("Nothing to save", "Run a measurement first (step 4).")
            return
        s = self._record_solid()
        if s is None:
            messagebox.showinfo("Select a solid", "Pick a solid in step 1 first.")
            return
        num = int(self.trial_num_var.get())
        others = [t for i, t in enumerate(s["trials_ms"]) if t is not None and i != num - 1]
        if others:
            mean = sum(others) / len(others)
            dev = abs(self.result["elapsed_ms"] - mean) / mean * 100
            if dev > 5 and not messagebox.askyesno(
                    "Possible outlier",
                    f"This time differs {dev:.1f}% from the other trials "
                    f"(mean {mean:.1f} ms).\nSave anyway?"):
                return
        s["trials_ms"][num - 1] = self.result["elapsed_ms"]
        dm.save_data(self.data)
        filled = sum(1 for t in s["trials_ms"] if t is not None)
        self.result = None
        self.result_var.set(
            f"Trial {num} saved for slot {s['slot']} ({filled}/3 recorded).")
        self.refresh_solid_list()
        self.status(f"Saved trial {num} for slot {s['slot']}.")

    # ------------------------------------------------------------------
    # Misc helpers / state
    # ------------------------------------------------------------------
    def _record_solid(self):
        idx = self.solid_cb.current()
        if idx < 0:
            return None
        return self.data["solids"][idx]

    def _record_solid_changed(self):
        self.result = None
        self.result_var.set("No measurement yet.")
        self._update_step_states()

    def browse_video(self):
        path = filedialog.askopenfilename(
            title="Choose video",
            filetypes=[("Video files", "*.avi *.mp4 *.mov *.mkv *.webm *.mjpg"),
                       ("All files", "*.*")])
        if path:
            self._set_video(path)

    def _set_video(self, path):
        self.close_review()
        self.video_path = path
        self.gate_points = None
        self.color_range = None
        self.result = None
        self.trajectory = None
        self.box = self.box_gates = None
        self.box_timeline = {}
        self.box_info_var.set("")
        self.clear_perspective()
        self.file_entry.delete(0, "end")
        self.file_entry.insert(0, path)
        self.video_msg_var.set(
            f"Video loaded: {os.path.basename(path)}. "
            "Mark the END gate point (step 3).  "
            "Optional: set a perspective reference if the camera is angled.")
        self.result_var.set("No measurement yet.")
        self._clear_preview("v\nVideo loaded:\n" + os.path.basename(path))
        self.status(f"Loaded video: {os.path.abspath(path)}")
        self._update_step_states()

    def _update_step_states(self):
        source = self.source_var.get()
        mode = self.mode_var.get()
        busy = self.cv2_busy
        cam_active = self.cam is not None
        have_video = bool(self.video_path)
        have_gates = bool(self.gate_points)
        have_color = bool(self.color_range)

        self.solid_cb.configure(state="disabled" if busy or cam_active else "readonly")
        if cam_active:
            self.webcam_btn.configure(state="disabled")
            self.file_entry.configure(state="disabled")
        elif source == "webcam":
            self.webcam_btn.configure(state="disabled" if busy else "normal")
            self.file_entry.configure(state="disabled")
        else:
            self.webcam_btn.configure(state="disabled")
            self.file_entry.configure(state="normal")
        is_box = mode == "box"
        idle_ok = have_video and not busy and not cam_active
        for _d in (0, 1):
            self.psp_btn[_d].configure(
                state="normal" if idle_ok else "disabled")
        self.mark_btn.configure(state="normal" if (idle_ok and not is_box) else "disabled")
        self.box_btn.configure(state="normal" if (idle_ok and is_box) else "disabled")
        self.box_start_spin.configure(state="normal" if (idle_ok and is_box) else "disabled")
        self.pick_color_btn.configure(
            state="normal" if (have_video and mode == "color" and not busy
                               and not cam_active) else "disabled")
        if is_box:
            can_run = idle_ok and self.box is not None and bool(self.box_gates)
        else:
            can_run = (have_video and have_gates and have_color
                       and not busy and not cam_active)
        self.run_btn.configure(state="normal" if can_run else "disabled")
        self.save_btn.configure(state="normal" if self.result is not None else "disabled")

    def refresh_all(self):
        self.refresh_solid_list()
        self.refresh_results()

    def status(self, msg):
        self.status_var.set(msg)
        self._update_step_states()

    # ------------------------------------------------------------------
    # Background thread plumbing
    # ------------------------------------------------------------------
    def _cv2_worker(self, fn, callback):
        if self.cv2_busy:
            return
        self.cv2_busy = True
        self.pending_callback = callback
        self._update_step_states()

        def run():
            try:
                result = fn()
                self.cv2_q.put(("done", result))
            except Exception as exc:  # noqa: BLE001 - report to the GUI
                self.cv2_q.put(("error", exc))

        threading.Thread(target=run, daemon=True).start()

    def _poll_cv2_queue(self):
        try:
            while True:
                kind, payload = self.cv2_q.get_nowait()
                if kind == "frame":
                    frame, t_sec, centroid, frame_idx = payload
                    if self.ui_mode == "track":
                        def draw(f, c=centroid, t=t_sec, idx=frame_idx):
                            self._draw_color_gate(f, origin=self._track_first)
                            if c is not None:
                                colour = (0, 255, 0)
                                try:
                                    move_px = float(self.release_thresh_var.get())
                                except ValueError:
                                    move_px = 2.0
                                if self._track_first is None:
                                    self._track_first = (c[0], c[1])
                                if self._track_released:
                                    colour = (0, 255, 255)
                                elif (abs(c[0] - self._track_first[0]) >= move_px or
                                      abs(c[1] - self._track_first[1]) >= move_px):
                                    self._track_released = True
                                    colour = (0, 255, 255)
                                cv2.circle(f, (int(c[0]), int(c[1])), 6,
                                           colour, -1)
                            cv2.putText(f, f"frame={idx} t={t:.3f}s",
                                        (12, 26), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7, (0, 255, 0), 2)
                        self._show_bgr(frame, draw=draw)
                    continue
                if kind == "boxframe":
                    frame, idx, bb = payload
                    if self.ui_mode == "track":
                        def draw(f, idx=idx, bb=bb):
                            try:
                                move_px = float(self.release_thresh_var.get())
                            except ValueError:
                                move_px = 2.0
                            try:
                                phase = box_tracker.box_phase(
                                    bb, self.box, self.box_gates[0], move_px,
                                    H=self.H)
                            except ValueError:
                                phase = 0
                            colours = {0: (255, 200, 0),
                                       1: (255, 255, 0),
                                       2: (0, 255, 0)}
                            labels = {0: "at rest",
                                      1: "RELEASED",
                                      2: "END crossed"}
                            x, y, bw, bh = [int(round(v)) for v in bb]
                            cv2.rectangle(f, (x, y), (x + bw, y + bh),
                                          colours[phase], 2)
                            try:
                                _, u, _ = box_tracker.gate_axis(self.box, self.box_gates[0],
                                                                H=self.H)
                                box_tracker._draw_gate_line(f, self.box_gates[0], u,
                                                            (0, 255, 0), H=self.H)
                            except ValueError:
                                pass
                            cv2.putText(f, f"frame={idx}  {labels[phase]}", (12, 26),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                        colours[phase], 2)
                        self._show_bgr(frame, draw=draw)
                    continue
                cb = self.pending_callback
                self.pending_callback = None
                self.cv2_busy = False
                if kind == "error":
                    msg = str(payload)
                    if cb:
                        cb("__comerror__: " + msg)
                    else:
                        messagebox.showerror("Error", msg)
                elif cb:
                    cb(payload)
        except queue.Empty:
            pass
        self.after(50, self._poll_cv2_queue)

    # ------------------------------------------------------------------
    # Results & export
    # ------------------------------------------------------------------
    def refresh_results(self):
        df = dm.build_results_table(self.data)
        df = df.fillna("")
        cols = list(df.columns)
        self.results_tree.delete(*self.results_tree.get_children())
        self.results_tree.configure(columns=cols)
        for c in cols:
            self.results_tree.heading(c, text=c)
            self.results_tree.column(c, width=110, anchor="center")
        self.results_tree.tag_configure("odd", background=PALETTE["row_alt"])
        self.results_tree.tag_configure("even", background=PALETTE["surface"])
        for i, (_, row) in enumerate(df.iterrows()):
            self.results_tree.insert("", "end", values=tuple(row),
                                     tags=("odd" if i % 2 else "even",))

    def export_results(self):
        path = filedialog.asksaveasfilename(
            title="Export results",
            defaultextension=".xlsx",
            filetypes=[("Excel workbook", "*.xlsx"), ("CSV file", "*.csv")])
        if not path:
            return
        if not (path.lower().endswith(".xlsx") or path.lower().endswith(".csv")):
            path += ".xlsx"
        out, df = dm.export_table(self.data, path)
        messagebox.showinfo("Exported",
                            f"Saved {len(df)} rows to:\n{os.path.abspath(out)}")

    def on_close(self):
        self.stop_event.set()
        if self.cam is not None:
            self._finish_cam_session(keep_file=False)
        self.close_review()
        try:
            cv2.destroyAllWindows()
        except Exception:  # noqa: BLE001
            pass
        self.destroy()


if __name__ == "__main__":
    app = RollingBodyApp()
    app.mainloop()