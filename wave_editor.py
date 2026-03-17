"""wave_editor.py
----------------
Visual drag-based wave editor with modulation zone support.

Public API
----------
    editor.get_curve(n_samples)  →  (times, amplitudes)  both in [0, 1]
    editor.h_divisions.get()     →  current pitch rows
    editor.v_divisions.get()     →  current time columns
"""

import json
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional, Tuple

import numpy as np

try:
    import wave_synth as _synth
    _HAS_SYNTH = True
except ImportError:
    _HAS_SYNTH = False

try:
    from scipy.interpolate import CubicSpline
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# ── Palette ────────────────────────────────────────────────────────────────
_BG      = '#1e1e2e'
_BG2     = '#12121c'
_SURFACE = '#2a2a3c'
_OVERLAY = '#45475a'
_TEXT    = '#cdd6f4'
_SUBTEXT = '#a6adc8'
_MUTED   = '#585b70'
_BLUE    = '#89b4fa'
_GREEN   = '#a6e3a1'
_RED     = '#f38ba8'

# Zone fill / outline pairs
_ZONE_PALETTE = [
    ('#cba6f7', '#9a79c7'),   # mauve
    ('#fab387', '#c9855e'),   # peach
    ('#a6e3a1', '#78b574'),   # green
    ('#89dceb', '#5aacbb'),   # sky
    ('#f38ba8', '#c25e7a'),   # red
    ('#f9e2af', '#c8b17e'),   # yellow
]

# ── Layout ─────────────────────────────────────────────────────────────────
_CW      = 900
_CH      = 520
_MARGIN  = 38
_PR      = 7          # control-point radius
_EDGE    = 8          # zone-edge hit width (px each side)

# ── Shared constant ─────────────────────────────────────────────────────────
SCORE_BPM = 9600

# ── Scale vocabulary ────────────────────────────────────────────────────────
_KEYS  = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
_MODES = ['major', 'minor', 'dorian', 'phrygian', 'lydian', 'mixolydian', 'locrian']
_SOUNDS = [
    'piano_sound',
    'bell',
    'crystal_bowl',
    'rich_bell',
    'marimba',
    'soft_kalimba',
    'tonal_percussion',
    'vibraphone',
    'pad',
    'ethereal_pad',
    'shimmer',
    'magic_shimmer',
    'bright_crystalline',
    'breathy_flute',
]


class WaveEditor:
    """Interactive spline wave editor with modulation zones.

    Curve mode (default)
    --------------------
    - Click empty canvas  →  add control point
    - Drag point          →  move it
    - Right-click point   →  remove it

    Zone mode  (toggle "Edit zones" checkbox)
    ------------------------------------------
    - Click zone body     →  select zone (shows key/mode in panel)
    - Drag zone body      →  move zone
    - Drag zone edge      →  resize zone
    - Right-click zone    →  delete zone
    - "Add Zone" button   →  new full-width zone
    """

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Wave Editor")
        self.root.configure(bg=_BG)
        self.root.resizable(False, False)

        self.h_divisions = tk.IntVar(value=16)
        self.v_divisions = tk.IntVar(value=16)

        self._pts: list = []
        self._drag_pt: Optional[list] = None

        # Zones
        self._zones: list = []
        self._zone_drag: Optional[dict] = None
        self._sel_zone: Optional[dict] = None

        self._init_default_zone()
        self._build_ui()
        self._init_default_curve()
        self._redraw()

    # ── Default data ───────────────────────────────────────────────────────

    def _init_default_zone(self) -> None:
        self._zones = [{'x': float(_MARGIN), 'w': float(_CW - 2 * _MARGIN),
                        'key': 'C', 'mode': 'major', 'color_idx': 0}]

    def _init_default_curve(self) -> None:
        left  = _MARGIN + 10
        right = _CW - _MARGIN - 10
        mid   = _CH / 2
        amp   = (_CH - 2 * _MARGIN) * 0.38
        xs    = np.linspace(left, right, 7)
        ys    = mid - amp * np.sin(np.linspace(0, 3 * np.pi, 7))
        self._pts = [[float(x), float(y)] for x, y in zip(xs, ys)]

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # Canvas
        self.canvas = tk.Canvas(self.root, width=_CW, height=_CH,
                                bg=_BG2, highlightthickness=0, cursor='crosshair')
        self.canvas.grid(row=0, column=0, padx=(12, 6), pady=(12, 4))
        self.canvas.bind('<Button-1>',        self._on_click)
        self.canvas.bind('<B1-Motion>',       self._on_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)
        self.canvas.bind('<Button-3>',        self._on_right_click)
        self.canvas.bind('<Motion>',          self._on_motion)

        # Right panel
        panel = tk.Frame(self.root, bg=_BG, width=188)
        panel.grid(row=0, column=1, padx=(0, 12), pady=(12, 4), sticky='ns')
        panel.grid_propagate(False)

        def lbl(text, size=9, color=_TEXT, bold=False):
            return tk.Label(panel, text=text, bg=_BG, fg=color,
                            font=('Helvetica', size, 'bold' if bold else 'normal'))

        def sep():
            tk.Frame(panel, bg=_OVERLAY, height=1).pack(fill=tk.X, padx=10, pady=6)

        lbl("Wave Editor", 14, _TEXT, bold=True).pack(pady=(16, 4))
        lbl("drag points to sculpt the curve", 8, _MUTED).pack(pady=(0, 6))
        sep()

        # Pitch-row slider
        lbl("Pitch rows", 9, _SUBTEXT).pack(pady=(6, 0))
        self._h_lbl = lbl("16", 12, _BLUE)
        ttk.Scale(panel, from_=2, to=48, variable=self.h_divisions,
                  orient=tk.HORIZONTAL, command=self._on_grid_change,
                  ).pack(fill=tk.X, padx=16, pady=2)
        self._h_lbl.pack()
        sep()

        # Time-col slider
        lbl("Time cols", 9, _SUBTEXT).pack(pady=(4, 0))
        self._v_lbl = lbl("16", 12, _BLUE)
        ttk.Scale(panel, from_=2, to=32, variable=self.v_divisions,
                  orient=tk.HORIZONTAL, command=self._on_grid_change,
                  ).pack(fill=tk.X, padx=16, pady=2)
        self._v_lbl.pack()
        sep()

        # Tips
        for tip in ("click  →  add point", "drag   →  move", "right-click  →  remove"):
            lbl(tip, 8, _MUTED).pack(anchor='w', padx=16, pady=1)

        sep()

        # ── Zones section ─────────────────────────────────────────────────
        lbl("Zones", 9, _SUBTEXT, bold=True).pack(pady=(4, 4))

        zone_row = tk.Frame(panel, bg=_BG)
        zone_row.pack(fill=tk.X, padx=12, pady=(0, 4))
        self._zone_edit_var = tk.BooleanVar(value=False)
        tk.Checkbutton(zone_row, text="Edit", variable=self._zone_edit_var,
                       bg=_BG, fg=_TEXT, selectcolor=_SURFACE, activebackground=_BG,
                       font=('Helvetica', 9),
                       command=self._on_zone_mode_change).pack(side=tk.LEFT)
        ttk.Button(zone_row, text="Add Zone", command=self._add_zone).pack(side=tk.RIGHT)

        self._zone_sel_lbl = lbl("— select a zone —", 7, _MUTED)
        self._zone_sel_lbl.pack(pady=(0, 2))

        # Key row
        krow = tk.Frame(panel, bg=_BG)
        krow.pack(fill=tk.X, padx=12, pady=1)
        tk.Label(krow, text="key", bg=_BG, fg=_SUBTEXT,
                 font=('Helvetica', 8)).pack(side=tk.LEFT, padx=(0, 4))
        self._zone_key_var = tk.StringVar(value='C')
        tk.OptionMenu(krow, self._zone_key_var, 'C', *_KEYS,
                      command=self._on_zone_prop_change
                      ).pack(side=tk.RIGHT, fill=tk.X, expand=True)

        # Mode row
        mrow = tk.Frame(panel, bg=_BG)
        mrow.pack(fill=tk.X, padx=12, pady=1)
        tk.Label(mrow, text="mode", bg=_BG, fg=_SUBTEXT,
                 font=('Helvetica', 8)).pack(side=tk.LEFT, padx=(0, 4))
        self._zone_mode_var = tk.StringVar(value='major')
        tk.OptionMenu(mrow, self._zone_mode_var, 'major', *_MODES,
                      command=self._on_zone_prop_change
                      ).pack(side=tk.RIGHT, fill=tk.X, expand=True)

        # Spacer + reset
        tk.Frame(panel, bg=_BG).pack(expand=True)
        ttk.Button(panel, text="Reset curve", command=self._reset).pack(
            fill=tk.X, padx=14, pady=(0, 12))

        self._build_footer()

    # ── Footer ─────────────────────────────────────────────────────────────

    def _build_footer(self) -> None:
        foot = tk.Frame(self.root, bg=_SURFACE, pady=7)
        foot.grid(row=1, column=0, columnspan=2, sticky='ew', padx=12, pady=(0, 10))

        def lbl(text):
            return tk.Label(foot, text=text, bg=_SURFACE, fg=_SUBTEXT,
                            font=('Helvetica', 8))

        def entry(var, width):
            return tk.Entry(foot, textvariable=var, width=width,
                            bg=_BG2, fg=_TEXT, insertbackground=_TEXT,
                            relief=tk.FLAT, font=('Helvetica', 9))

        self._name_var = tk.StringVar(value='v1')
        lbl("name").pack(side=tk.LEFT, padx=(10, 2))
        entry(self._name_var, 6).pack(side=tk.LEFT, padx=(0, 8))

        self._sound_var = tk.StringVar(value='piano_sound')
        lbl("sound").pack(side=tk.LEFT, padx=(0, 2))
        ttk.Combobox(foot, textvariable=self._sound_var, values=_SOUNDS,
                     width=16, state='readonly',
                     font=('Helvetica', 9)).pack(side=tk.LEFT, padx=(0, 8))

        self._octave_var = tk.IntVar(value=0)
        lbl("octave").pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(foot, text='−', bg=_BG2, fg=_TEXT, relief=tk.FLAT,
                  font=('Helvetica', 9), width=1,
                  command=lambda: self._octave_var.set(max(-4, self._octave_var.get() - 1))
                  ).pack(side=tk.LEFT)
        tk.Label(foot, textvariable=self._octave_var, bg=_SURFACE, fg=_BLUE,
                 font=('Helvetica', 9), width=2).pack(side=tk.LEFT)
        tk.Button(foot, text='+', bg=_BG2, fg=_TEXT, relief=tk.FLAT,
                  font=('Helvetica', 9), width=1,
                  command=lambda: self._octave_var.set(min(4, self._octave_var.get() + 1))
                  ).pack(side=tk.LEFT, padx=(0, 16))

        tk.Frame(foot, bg=_OVERLAY, width=1).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        self._duration_var = tk.StringVar(value='30')
        lbl("duration (s)").pack(side=tk.LEFT, padx=(6, 2))
        entry(self._duration_var, 5).pack(side=tk.LEFT, padx=(0, 8))
        lbl(f"BPM = {SCORE_BPM}").pack(side=tk.LEFT, padx=(0, 16))

        tk.Frame(foot, bg=_OVERLAY, width=1).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        ttk.Button(foot, text="Save…", command=self._save).pack(side=tk.LEFT, padx=(6, 3))
        ttk.Button(foot, text="Load…", command=self._load).pack(side=tk.LEFT, padx=(0, 4))

        tk.Frame(foot, bg=_OVERLAY, width=1).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        ttk.Button(foot, text="▶ Play",     command=self._play).pack(side=tk.LEFT, padx=(6, 3))
        ttk.Button(foot, text="■ Stop",     command=self._stop).pack(side=tk.LEFT, padx=(0, 3))
        ttk.Button(foot, text="Save WAV…",  command=self._save_wav).pack(side=tk.LEFT, padx=(0, 8))

        self._status_var = tk.StringVar(value="")
        tk.Label(foot, textvariable=self._status_var, bg=_SURFACE, fg=_GREEN,
                 font=('Helvetica', 8)).pack(side=tk.LEFT, padx=(0, 8))

    # ── Zone logic ─────────────────────────────────────────────────────────

    def _add_zone(self) -> None:
        new = {'x': float(_MARGIN), 'w': float(_CW - 2 * _MARGIN),
               'key': 'C', 'mode': 'major',
               'color_idx': len(self._zones) % len(_ZONE_PALETTE)}
        self._zones.append(new)
        self._zone_edit_var.set(True)
        self._on_zone_mode_change()
        self._select_zone(new)
        self._redraw()

    def _select_zone(self, zone: Optional[dict]) -> None:
        self._sel_zone = zone
        if zone:
            key  = zone.get('key', 'C')
            mode = zone.get('mode', 'major')
            self._zone_key_var.set(key)
            self._zone_mode_var.set(mode)
            self._zone_sel_lbl.config(text=f"{key} {mode}")
        else:
            self._zone_sel_lbl.config(text="— select a zone —")

    def _on_zone_mode_change(self) -> None:
        if not self._zone_edit_var.get():
            self._select_zone(None)
        cursor = 'fleur' if self._zone_edit_var.get() else 'crosshair'
        self.canvas.config(cursor=cursor)
        self._redraw()

    def _on_zone_prop_change(self, _=None) -> None:
        if self._sel_zone is not None:
            self._sel_zone['key']  = self._zone_key_var.get()
            self._sel_zone['mode'] = self._zone_mode_var.get()
            self._zone_sel_lbl.config(
                text=f"{self._sel_zone['key']} {self._sel_zone['mode']}")
            self._redraw()

    def _hit_zone(self, x: float, y: float) -> Tuple[Optional[dict], str]:
        """Return (zone, mode) where mode ∈ {'left','right','body'}, or (None,'')."""
        if not (_MARGIN <= y <= _CH - _MARGIN):
            return None, ''
        for zone in reversed(self._zones):
            zx, zw = zone['x'], zone['w']
            if abs(x - zx) <= _EDGE:
                return zone, 'left'
            if abs(x - (zx + zw)) <= _EDGE:
                return zone, 'right'
            if zx < x < zx + zw:
                return zone, 'body'
        return None, ''

    def _zones_to_scale_regions(self) -> list:
        gl = float(_MARGIN)
        gw = float(_CW - 2 * _MARGIN)
        vd = self.v_divisions.get()
        out = []
        for zone in self._zones:
            cs = (zone['x'] - gl) / gw * vd
            ce = (zone['x'] + zone['w'] - gl) / gw * vd
            cs = max(0.0, min(float(vd), cs))
            ce = max(0.0, min(float(vd), ce))
            if ce > cs:
                out.append({'col_start': round(cs, 4), 'col_end': round(ce, 4),
                            'key': zone.get('key', 'C'), 'mode': zone.get('mode', 'major')})
        return out

    # ── Mouse events ───────────────────────────────────────────────────────

    def _on_click(self, ev: tk.Event) -> None:
        # Zone mode
        if self._zone_edit_var.get():
            zone, mode = self._hit_zone(ev.x, ev.y)
            if zone:
                self._zone_drag = {'zone': zone, 'mode': mode, 'ox': ev.x,
                                   'oz_x': zone['x'], 'oz_w': zone['w']}
                self._select_zone(zone)
            else:
                self._select_zone(None)
            self._redraw()
            return

        # Curve mode
        hit = self._hit(ev.x, ev.y)
        if hit:
            self._drag_pt = hit
            return
        nx = float(ev.x)
        existing = {p[0] for p in self._pts}
        while nx in existing:
            nx += 0.5
        new_pt = [nx, float(self._clamp_y(ev.y))]
        self._pts.append(new_pt)
        self._sort()
        self._drag_pt = new_pt
        self._redraw()

    def _on_drag(self, ev: tk.Event) -> None:
        if self._zone_drag is not None:
            dx   = ev.x - self._zone_drag['ox']
            z    = self._zone_drag['zone']
            mode = self._zone_drag['mode']
            gl, gr = float(_MARGIN), float(_CW - _MARGIN)
            min_w  = 24.0
            if mode == 'move':
                z['x'] = max(gl, min(gr - z['w'], self._zone_drag['oz_x'] + dx))
            elif mode == 'left':
                orig_r = self._zone_drag['oz_x'] + self._zone_drag['oz_w']
                new_x  = max(gl, min(orig_r - min_w, self._zone_drag['oz_x'] + dx))
                z['w'] = orig_r - new_x
                z['x'] = new_x
            elif mode == 'right':
                z['w'] = max(min_w, min(gr - z['x'], self._zone_drag['oz_w'] + dx))
            self._redraw()
            return

        if self._drag_pt is None:
            return
        self._drag_pt[0] = float(max(1, min(_CW - 1, ev.x)))
        self._drag_pt[1] = float(self._clamp_y(ev.y))
        self._sort()
        self._redraw()

    def _on_release(self, _ev: tk.Event) -> None:
        self._zone_drag = None
        self._drag_pt   = None

    def _on_right_click(self, ev: tk.Event) -> None:
        if self._zone_edit_var.get():
            zone, _ = self._hit_zone(ev.x, ev.y)
            if zone and len(self._zones) > 1:
                self._zones.remove(zone)
                if self._sel_zone is zone:
                    self._select_zone(None)
                self._redraw()
            return
        hit = self._hit(ev.x, ev.y)
        if hit and len(self._pts) > 2:
            self._pts.remove(hit)
            self._redraw()

    def _on_motion(self, ev: tk.Event) -> None:
        if not self._zone_edit_var.get():
            return
        _, mode = self._hit_zone(ev.x, ev.y)
        if mode in ('left', 'right'):
            self.canvas.config(cursor='sb_h_double_arrow')
        elif mode == 'body':
            self.canvas.config(cursor='fleur')
        else:
            self.canvas.config(cursor='fleur')

    def _on_grid_change(self, _val=None) -> None:
        self._h_lbl.config(text=str(self.h_divisions.get()))
        self._v_lbl.config(text=str(self.v_divisions.get()))
        self._redraw()

    @staticmethod
    def _clamp_y(y: float) -> float:
        return max(1.0, min(float(_CH - 1), y))

    def _hit(self, x: float, y: float, r: float = _PR + 6) -> Optional[list]:
        for pt in self._pts:
            if (pt[0] - x) ** 2 + (pt[1] - y) ** 2 <= r * r:
                return pt
        return None

    def _sort(self) -> None:
        self._pts.sort(key=lambda p: p[0])

    def _reset(self) -> None:
        self._pts = []
        self._drag_pt = None
        self._init_default_curve()
        self._redraw()

    # ── Drawing ────────────────────────────────────────────────────────────

    def _redraw(self) -> None:
        self.canvas.delete('all')
        self._draw_grid()
        self._draw_zones()
        self._draw_curve()
        self._draw_points()

    def _draw_zones(self) -> None:
        edit_mode = self._zone_edit_var.get()
        TAB_H     = 18   # height of solid header tab in px

        for zone in self._zones:
            zx, zw   = zone['x'], zone['w']
            ci       = zone.get('color_idx', 0) % len(_ZONE_PALETTE)
            fill, outline = _ZONE_PALETTE[ci]
            selected = (zone is self._sel_zone)
            top      = _MARGIN
            bot      = _CH - _MARGIN

            # Transparent body — dashed outline only
            self.canvas.create_rectangle(
                zx, top, zx + zw, bot,
                fill='', outline=_TEXT if selected else outline,
                width=2 if selected else 1,
                dash=(6, 3) if not selected else (),
            )

            # Solid colour header tab at the top
            self.canvas.create_rectangle(
                zx, top, zx + zw, top + TAB_H,
                fill=fill, outline='',
            )

            # Label inside the tab
            label = f"{zone.get('key','?')} {zone.get('mode','?')}"
            self.canvas.create_text(
                zx + zw / 2, top + TAB_H / 2,
                text=label, fill=_BG2,
                font=('Helvetica', 8, 'bold'),
            )

            # Resize handles (edit mode only)
            if edit_mode:
                for ex in (zx, zx + zw):
                    self.canvas.create_rectangle(
                        ex - _EDGE, top + TAB_H + 4,
                        ex + _EDGE, bot - 4,
                        fill=_OVERLAY, outline=outline, width=1,
                    )

    def _draw_grid(self) -> None:
        hd   = self.h_divisions.get()
        vd   = self.v_divisions.get()
        l, r = _MARGIN, _CW - _MARGIN
        t, b = _MARGIN, _CH - _MARGIN

        # Vertical lines (time cols)
        for i in range(vd + 1):
            x     = l + (r - l) * i / vd
            color = _OVERLAY if (i == 0 or i == vd) else _SURFACE
            self.canvas.create_line(x, t, x, b, fill=color, width=1)
            if 0 < i < vd:
                self.canvas.create_text(x, b + 14, text=str(i),
                                        fill=_MUTED, font=('Helvetica', 7))

        # Horizontal lines (pitch-row boundaries)
        for i in range(hd + 1):
            y     = t + (b - t) * i / hd
            color = _OVERLAY if (i == 0 or i == hd) else _SURFACE
            self.canvas.create_line(l, y, r, y, fill=color, width=1)

        # Row labels at BAND CENTRES (not at boundary lines)
        for i in range(hd):
            y_mid = t + (b - t) * (i + 0.5) / hd
            row   = hd - i          # row 1 = bottom band, row hd = top band
            self.canvas.create_text(l - 14, y_mid, text=str(row),
                                    fill=_MUTED, font=('Helvetica', 7))

        self.canvas.create_rectangle(l, t, r, b, outline=_OVERLAY, width=1)

    def _draw_curve(self) -> None:
        if len(self._pts) < 2:
            return
        pts = sorted(self._pts, key=lambda p: p[0])
        xs  = np.array([p[0] for p in pts])
        ys  = np.array([p[1] for p in pts])
        _, idx = np.unique(xs, return_index=True)
        xs, ys = xs[idx], ys[idx]
        if len(xs) < 2:
            return
        n = max(800, len(xs) * 100)
        if _HAS_SCIPY and len(xs) >= 3:
            cs     = CubicSpline(xs, ys)
            x_fine = np.linspace(xs[0], xs[-1], n)
            y_fine = cs(x_fine)
        else:
            x_fine = np.linspace(xs[0], xs[-1], n)
            y_fine = np.interp(x_fine, xs, ys)
        coords = []
        for x, y in zip(x_fine.tolist(), y_fine.tolist()):
            coords.extend((x, y))
        self.canvas.create_line(*coords, fill=_BLUE, width=2, smooth=False)

    def _draw_points(self) -> None:
        for pt in self._pts:
            x, y  = pt
            color = _RED if (pt is self._drag_pt) else _GREEN
            self.canvas.create_oval(x - _PR, y - _PR, x + _PR, y + _PR,
                                    fill=color, outline=_TEXT, width=1)

    # ── Public API ─────────────────────────────────────────────────────────

    def get_curve(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Return (times, amplitudes) both in [0, 1]."""
        if len(self._pts) < 2:
            return np.array([]), np.array([])
        pts = sorted(self._pts, key=lambda p: p[0])
        xs  = np.array([p[0] for p in pts])
        ys  = np.array([p[1] for p in pts])
        _, idx = np.unique(xs, return_index=True)
        xs, ys = xs[idx], ys[idx]
        if len(xs) < 2:
            return np.array([]), np.array([])
        if _HAS_SCIPY and len(xs) >= 3:
            cs     = CubicSpline(xs, ys)
            x_fine = np.linspace(xs[0], xs[-1], n_samples)
            y_fine = cs(x_fine)
        else:
            x_fine = np.linspace(xs[0], xs[-1], n_samples)
            y_fine = np.interp(x_fine, xs, ys)
        grid_top = float(_MARGIN)
        grid_bot = float(_CH - _MARGIN)
        t   = (x_fine - x_fine.min()) / (x_fine.max() - x_fine.min())
        amp = 1.0 - (y_fine - grid_top) / (grid_bot - grid_top)
        return t, np.clip(amp, 0.0, 1.0)

    # ── File I/O ───────────────────────────────────────────────────────────

    def _voice_dict(self) -> dict:
        return {
            'name':           self._name_var.get(),
            'sound_func':     self._sound_var.get(),
            'octave_shift':   int(self._octave_var.get()),
            'h_divisions':    self.h_divisions.get(),
            'v_divisions':    self.v_divisions.get(),
            'control_points': [p[:] for p in self._pts],
            'scale_regions':  self._zones_to_scale_regions(),
            'zones':          [dict(z) for z in self._zones],
        }

    def _wave_dict(self) -> dict:
        """Return the current editor state as a wave JSON dict (without saving to disk)."""
        return {
            'format_version': 1,
            'bpm': SCORE_BPM,
            'duration_seconds': float(self._duration_var.get() or 30),
            'voices': [self._voice_dict()],
        }

    def _play(self) -> None:
        if not _HAS_SYNTH:
            messagebox.showerror("Missing module",
                                 "wave_synth.py not found next to wave_editor.py.")
            return
        self._status_var.set("Rendering…")
        self.root.update_idletasks()

        def _worker():
            try:
                _synth.play(self._wave_dict(), blocking=True)
                self.root.after(0, lambda: self._status_var.set("Done"))
            except RuntimeError as e:
                msg = str(e)
                self.root.after(0, lambda: self._status_var.set("No audio device"))
                self.root.after(0, lambda: messagebox.showwarning("Playback", msg))

        threading.Thread(target=_worker, daemon=True).start()
        self._status_var.set("Playing…")

    def _stop(self) -> None:
        if _HAS_SYNTH:
            _synth.stop()
            self._status_var.set("Stopped")

    def _save_wav(self) -> None:
        if not _HAS_SYNTH:
            messagebox.showerror("Missing module",
                                 "wave_synth.py not found next to wave_editor.py.")
            return
        self.root.update()
        self.root.lift()
        path = filedialog.asksaveasfilename(
            title='Save WAV', defaultextension='.wav',
            filetypes=[('WAV audio', '*.wav'), ('All files', '*.*')],
        )
        if not path:
            return
        self._status_var.set("Rendering…")
        self.root.update_idletasks()

        def _worker():
            try:
                _synth.render(self._wave_dict(), output_wav_path=path)
                self.root.after(0, lambda: self._status_var.set(
                    f"Saved {Path(path).name}"))
            except Exception as e:
                self.root.after(0, lambda: self._status_var.set("Error"))
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))

        threading.Thread(target=_worker, daemon=True).start()

    def _save(self) -> None:
        self.root.update()
        self.root.lift()
        path = filedialog.asksaveasfilename(
            title='Save wave', defaultextension='.json',
            filetypes=[('JSON', '*.json'), ('All files', '*.*')],
        )
        if not path:
            return
        data = {'format_version': 1, 'bpm': SCORE_BPM,
                'duration_seconds': float(self._duration_var.get() or 30),
                'voices': [self._voice_dict()]}
        Path(path).write_text(json.dumps(data, indent=2))
        self.root.title(f"Wave Editor — {Path(path).name}")

    def _load(self) -> None:
        self.root.update()
        self.root.lift()
        path = filedialog.askopenfilename(
            title='Load wave',
            filetypes=[('JSON', '*.json'), ('All files', '*.*')],
        )
        if not path:
            return
        data = json.loads(Path(path).read_text())
        self._duration_var.set(str(data.get('duration_seconds', 30)))
        if data.get('voices'):
            v = data['voices'][0]
            self._name_var.set(v.get('name', 'v1'))
            self._sound_var.set(v.get('sound_func', 'piano_sound'))
            self._octave_var.set(v.get('octave_shift', 0))
            self.h_divisions.set(v.get('h_divisions', 8))
            self.v_divisions.set(v.get('v_divisions', 16))
            self._pts     = [list(p) for p in v.get('control_points', [])]
            self._drag_pt = None
            self._h_lbl.config(text=str(self.h_divisions.get()))
            self._v_lbl.config(text=str(self.v_divisions.get()))
            raw_zones = v.get('zones', [])
            if raw_zones:
                self._zones = [dict(z) for z in raw_zones]
            else:
                self._init_default_zone()
            self._select_zone(None)
        self.root.title(f"Wave Editor — {Path(path).name}")
        self._redraw()


# ── Launcher ───────────────────────────────────────────────────────────────

def launch() -> None:
    root = tk.Tk()
    WaveEditor(root)
    root.mainloop()


if __name__ == '__main__':
    launch()
