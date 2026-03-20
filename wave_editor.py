"""wave_editor.py
----------------
Multi-voice wave editor with full-piano diatonic grid, zoom/pan timeline.

Coordinate system
-----------------
Control points are stored as [t_frac, amp] where:
  t_frac ∈ [0, 1]  —  fraction of total piece duration
  amp    ∈ [0, 1]  —  pitch (0 = lowest row / bottom, 1 = highest row / top)

Zones are stored as {t_start, t_end} piece-fractions, not canvas pixels.
This format is read directly by wave_synth.py (format_version 2).
"""

import json
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

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

_ZONE_PALETTE = [
    ('#cba6f7', '#9a79c7'),
    ('#fab387', '#c9855e'),
    ('#a6e3a1', '#78b574'),
    ('#89dceb', '#5aacbb'),
    ('#f38ba8', '#c25e7a'),
    ('#f9e2af', '#c8b17e'),
]

_VOICE_COLORS = [
    '#89b4fa', '#a6e3a1', '#fab387',
    '#cba6f7', '#89dceb', '#f38ba8', '#f9e2af',
]

# ── Layout constants ────────────────────────────────────────────────────────
N_ROWS    = 49          # fixed diatonic rows — full piano range (~7 octaves × 7 steps)
_CELL_H   = 11          # pixels per row
_MARGIN_L = 42          # left margin (room for row labels)
_MARGIN_T = 38          # top margin
_MARGIN_B = 24          # bottom margin (time labels)
_PANEL_W  = 215         # right panel width
_PR       = 7           # control-point dot radius
_EDGE     = 8           # zone-edge hit width px

_CANVAS_H = _MARGIN_T + N_ROWS * _CELL_H + _MARGIN_B   # ~601 px
_CANVAS_W = 1200        # initial canvas width (resizes with window)

SCORE_BPM = 9600

# ── Scale vocabulary ────────────────────────────────────────────────────────
_KEYS  = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
_MODES = ['major', 'minor', 'dorian', 'phrygian', 'lydian', 'mixolydian', 'locrian']
_SOUNDS = [
    # Sampled instruments (require download_vsco.py / download_salamander.py)
    'xylophone', 'organ', 'chimes', 'flute', 'violin',
    'sampled_piano',
    # Concert percussion (pitch-shifted per grid row)
    'snare', 'bass_drum', 'timpani', 'resonant_tom',
    # Algorithmic synths
    'piano_sound', 'bell', 'crystal_bowl', 'rich_bell',
    'marimba', 'soft_kalimba', 'tonal_percussion', 'vibraphone',
    'pad', 'ethereal_pad', 'shimmer', 'magic_shimmer',
    'bright_crystalline', 'breathy_flute',
]

# Row that maps to C4 (center anchor, matching wave_synth._CENTER_MIDI logic)
_CENTER_ROW = (N_ROWS + 1) // 2   # = 25

# Labels shown on left axis — every 7 rows (one diatonic octave), centred
_AXIS_LABELS = {}   # populated after N_ROWS is known
for _r in range(1, N_ROWS + 1):
    _off = _r - _CENTER_ROW          # semitone-ish offset from C4 row
    _octave_step = _off // 7         # approximate octave relative to C4
    _deg_in_oct  = _off % 7
    if _deg_in_oct == 0 or _r in (1, N_ROWS):
        _oct_num = 4 + _octave_step
        _AXIS_LABELS[_r] = f"C{_oct_num}"


class WaveEditor:
    """Multi-voice wave editor."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Wave Editor")
        self.root.configure(bg=_BG)
        self.root.resizable(True, False)

        # ── Timeline view state (in seconds) ──────────────────────────────
        self._total_dur  = 30.0
        self._view_start = 0.0
        self._view_end   = 30.0

        # ── Voices ────────────────────────────────────────────────────────
        self._voices: list = [self._new_voice('v1', 0)]
        self._cur_v:  int  = 0

        # ── Interaction state ─────────────────────────────────────────────
        self._drag_pt    = None   # [t_frac, amp] currently being dragged
        self._zone_drag  = None   # dict with drag context
        self._sel_zone   = None   # currently selected zone dict
        self._zone_edit_var = tk.BooleanVar(value=False)

        # Guard against recursive prop-change callbacks
        self._loading_props = False

        # ── Reverb ────────────────────────────────────────────────────────
        self._reverb_preset_var = tk.StringVar(value='Dry')
        self._reverb_wet_var    = tk.DoubleVar(value=0.0)

        # Playhead
        self._playhead_sec: Optional[float] = None

        self._build_ui()
        self._update_voice_list()
        self._update_zoom_label()
        self.root.after(50, self._redraw)   # after geometry is settled

    # ── Voice factory ───────────────────────────────────────────────────────

    def _new_voice(self, name: str, color_idx: int,
                   start_t: float = 0.05) -> dict:
        return {
            'name':           name,
            'sound_func':     'sampled_piano',
            'octave_shift':   0,
            'color_idx':      color_idx % len(_VOICE_COLORS),
            'control_points': [[start_t, 0.5], [0.95, 0.5]],
            'zones': [{'t_start': 0.0, 't_end': 1.0,
                       'key': 'C', 'mode': 'major', 'color_idx': 0}],
        }

    def _cur_voice(self) -> dict:
        return self._voices[self._cur_v]

    # ── Coordinate transforms ───────────────────────────────────────────────

    def _content_w(self) -> float:
        """Drawable width from left margin to right canvas edge."""
        cw = self.canvas.winfo_width()
        if cw <= 1:
            cw = _CANVAS_W
        return max(100.0, float(cw - _MARGIN_L - 4))

    def _content_h(self) -> float:
        return float(N_ROWS * _CELL_H)

    def _t_to_x(self, t_frac: float) -> float:
        """Piece-fraction [0,1] → canvas x pixel."""
        span  = self._view_end - self._view_start
        t_sec = t_frac * self._total_dur
        return _MARGIN_L + (t_sec - self._view_start) / span * self._content_w()

    def _x_to_t(self, x_px: float) -> float:
        """Canvas x pixel → piece-fraction [0,1]."""
        span  = self._view_end - self._view_start
        t_sec = self._view_start + (x_px - _MARGIN_L) / self._content_w() * span
        return t_sec / self._total_dur

    def _amp_to_y(self, amp: float) -> float:
        """amp [0,1] → canvas y pixel (0=bottom row, 1=top row)."""
        return _MARGIN_T + (1.0 - amp) * self._content_h()

    def _y_to_amp(self, y_px: float) -> float:
        return 1.0 - (y_px - _MARGIN_T) / self._content_h()

    @staticmethod
    def _clamp_t(t: float) -> float:
        return max(0.0, min(1.0, t))

    @staticmethod
    def _clamp_amp(a: float) -> float:
        return max(0.0, min(1.0, a))

    # ── UI construction ─────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # Left: canvas + scrollbar
        c_frame = tk.Frame(self.root, bg=_BG2)
        c_frame.grid(row=0, column=0, padx=(10, 4), pady=(10, 4), sticky='nsew')
        self.root.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            c_frame, height=_CANVAS_H, bg=_BG2,
            highlightthickness=0, cursor='crosshair',
        )
        self.canvas.grid(row=0, column=0, sticky='ew')
        c_frame.columnconfigure(0, weight=1)

        self._hscroll = tk.Scrollbar(c_frame, orient=tk.HORIZONTAL,
                                     command=self._on_hscroll)
        self._hscroll.grid(row=1, column=0, sticky='ew')

        self.canvas.bind('<Button-1>',        self._on_click)
        self.canvas.bind('<B1-Motion>',       self._on_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)
        self.canvas.bind('<Button-3>',        self._on_right_click)
        self.canvas.bind('<Motion>',          self._on_motion)
        self.canvas.bind('<Configure>',       lambda _e: self._redraw())
        self.canvas.bind('<MouseWheel>',      self._on_wheel)
        self.canvas.bind('<Button-4>',        self._on_wheel)
        self.canvas.bind('<Button-5>',        self._on_wheel)

        # Right: panel
        panel = tk.Frame(self.root, bg=_BG, width=_PANEL_W)
        panel.grid(row=0, column=1, padx=(0, 10), pady=(10, 4), sticky='ns')
        panel.grid_propagate(False)
        self._build_panel(panel)

        # Footer
        self._build_footer()

    def _build_panel(self, panel: tk.Frame) -> None:
        def lbl(text, size=9, color=_TEXT, bold=False):
            return tk.Label(panel, text=text, bg=_BG, fg=color,
                            font=('Helvetica', size, 'bold' if bold else 'normal'))

        def sep():
            tk.Frame(panel, bg=_OVERLAY, height=1).pack(fill=tk.X, padx=8, pady=5)

        lbl("Wave Editor", 13, _TEXT, bold=True).pack(pady=(14, 2))
        lbl(f"{N_ROWS} diatonic rows · full piano range", 7, _MUTED).pack()
        sep()

        # ── Timeline controls ─────────────────────────────────────────────
        lbl("Timeline", 9, _SUBTEXT, bold=True).pack(pady=(4, 2))

        zoom_row = tk.Frame(panel, bg=_BG)
        zoom_row.pack(fill=tk.X, padx=10, pady=2)
        tk.Button(zoom_row, text='−', bg=_BG2, fg=_TEXT, relief=tk.FLAT,
                  font=('Helvetica', 11), width=2,
                  command=self._zoom_out).pack(side=tk.LEFT)
        tk.Button(zoom_row, text='+', bg=_BG2, fg=_TEXT, relief=tk.FLAT,
                  font=('Helvetica', 11), width=2,
                  command=self._zoom_in).pack(side=tk.LEFT, padx=(2, 8))
        self._zoom_lbl = tk.Label(zoom_row, text="30s view", bg=_BG, fg=_BLUE,
                                  font=('Helvetica', 8))
        self._zoom_lbl.pack(side=tk.LEFT)

        pan_row = tk.Frame(panel, bg=_BG)
        pan_row.pack(fill=tk.X, padx=10, pady=(0, 2))
        tk.Button(pan_row, text='◀', bg=_BG2, fg=_TEXT, relief=tk.FLAT,
                  font=('Helvetica', 9), width=2,
                  command=self._pan_left).pack(side=tk.LEFT)
        tk.Button(pan_row, text='▶', bg=_BG2, fg=_TEXT, relief=tk.FLAT,
                  font=('Helvetica', 9), width=2,
                  command=self._pan_right).pack(side=tk.LEFT, padx=(2, 0))

        sep()

        # ── Voices ────────────────────────────────────────────────────────
        lbl("Voices", 9, _SUBTEXT, bold=True).pack(pady=(4, 2))

        vbtn = tk.Frame(panel, bg=_BG)
        vbtn.pack(fill=tk.X, padx=8, pady=(0, 4))
        ttk.Button(vbtn, text="Add", command=self._add_voice).pack(side=tk.LEFT)
        ttk.Button(vbtn, text="Delete", command=self._del_voice).pack(side=tk.LEFT, padx=(4, 0))

        lb_frame = tk.Frame(panel, bg=_BG2, bd=0)
        lb_frame.pack(fill=tk.X, padx=8, pady=(0, 4))
        self._voice_lb = tk.Listbox(
            lb_frame, height=4, bg=_BG2, fg=_TEXT,
            selectbackground=_OVERLAY, selectforeground=_TEXT,
            font=('Helvetica', 9), relief=tk.FLAT, exportselection=False,
        )
        self._voice_lb.pack(fill=tk.X)
        self._voice_lb.bind('<<ListboxSelect>>', self._on_voice_select)

        # Voice property editors
        pf = tk.Frame(panel, bg=_BG)
        pf.pack(fill=tk.X, padx=8)

        r1 = tk.Frame(pf, bg=_BG)
        r1.pack(fill=tk.X, pady=1)
        tk.Label(r1, text='name', bg=_BG, fg=_SUBTEXT,
                 font=('Helvetica', 8)).pack(side=tk.LEFT)
        self._v_name_var = tk.StringVar(value='v1')
        tk.Entry(r1, textvariable=self._v_name_var, width=8,
                 bg=_BG2, fg=_TEXT, insertbackground=_TEXT,
                 relief=tk.FLAT, font=('Helvetica', 9)).pack(side=tk.RIGHT)
        self._v_name_var.trace_add('write', self._on_voice_prop_change)

        r2 = tk.Frame(pf, bg=_BG)
        r2.pack(fill=tk.X, pady=1)
        tk.Label(r2, text='sound', bg=_BG, fg=_SUBTEXT,
                 font=('Helvetica', 8)).pack(side=tk.LEFT)
        self._v_sound_var = tk.StringVar(value='sampled_piano')
        ttk.Combobox(r2, textvariable=self._v_sound_var, values=_SOUNDS,
                     width=14, state='readonly',
                     font=('Helvetica', 8)).pack(side=tk.RIGHT)
        self._v_sound_var.trace_add('write', self._on_voice_prop_change)

        r3 = tk.Frame(pf, bg=_BG)
        r3.pack(fill=tk.X, pady=1)
        tk.Label(r3, text='octave', bg=_BG, fg=_SUBTEXT,
                 font=('Helvetica', 8)).pack(side=tk.LEFT)
        self._v_oct_var = tk.IntVar(value=0)
        oct_inner = tk.Frame(r3, bg=_BG)
        oct_inner.pack(side=tk.RIGHT)
        tk.Button(oct_inner, text='−', bg=_BG2, fg=_TEXT, relief=tk.FLAT,
                  font=('Helvetica', 9), width=1,
                  command=lambda: self._v_oct_var.set(
                      max(-4, self._v_oct_var.get() - 1))).pack(side=tk.LEFT)
        tk.Label(oct_inner, textvariable=self._v_oct_var, bg=_BG, fg=_BLUE,
                 font=('Helvetica', 9), width=2).pack(side=tk.LEFT)
        tk.Button(oct_inner, text='+', bg=_BG2, fg=_TEXT, relief=tk.FLAT,
                  font=('Helvetica', 9), width=1,
                  command=lambda: self._v_oct_var.set(
                      min(4, self._v_oct_var.get() + 1))).pack(side=tk.LEFT)
        self._v_oct_var.trace_add('write', self._on_voice_prop_change)

        sep()

        # ── Zones ─────────────────────────────────────────────────────────
        lbl("Zones", 9, _SUBTEXT, bold=True).pack(pady=(4, 2))

        zone_row = tk.Frame(panel, bg=_BG)
        zone_row.pack(fill=tk.X, padx=8, pady=(0, 4))
        tk.Checkbutton(zone_row, text="Edit zones", variable=self._zone_edit_var,
                       bg=_BG, fg=_TEXT, selectcolor=_SURFACE, activebackground=_BG,
                       font=('Helvetica', 9),
                       command=self._on_zone_mode_change).pack(side=tk.LEFT)
        ttk.Button(zone_row, text="Add", command=self._add_zone).pack(side=tk.RIGHT)

        self._zone_sel_lbl = tk.Label(panel, text="— select a zone —",
                                      bg=_BG, fg=_MUTED, font=('Helvetica', 7))
        self._zone_sel_lbl.pack()

        krow = tk.Frame(panel, bg=_BG)
        krow.pack(fill=tk.X, padx=8, pady=1)
        tk.Label(krow, text="key", bg=_BG, fg=_SUBTEXT,
                 font=('Helvetica', 8)).pack(side=tk.LEFT)
        self._zone_key_var = tk.StringVar(value='C')
        tk.OptionMenu(krow, self._zone_key_var, 'C', *_KEYS,
                      command=self._on_zone_prop_change).pack(
                          side=tk.RIGHT, fill=tk.X, expand=True)

        mrow = tk.Frame(panel, bg=_BG)
        mrow.pack(fill=tk.X, padx=8, pady=1)
        tk.Label(mrow, text="mode", bg=_BG, fg=_SUBTEXT,
                 font=('Helvetica', 8)).pack(side=tk.LEFT)
        self._zone_mode_var = tk.StringVar(value='major')
        tk.OptionMenu(mrow, self._zone_mode_var, 'major', *_MODES,
                      command=self._on_zone_prop_change).pack(
                          side=tk.RIGHT, fill=tk.X, expand=True)

        sep()

        # ── Reverb ────────────────────────────────────────────────────────
        lbl("Reverb", 9, _SUBTEXT, bold=True).pack(pady=(4, 2))

        _preset_names = list(
            _synth.REVERB_PRESETS.keys() if _HAS_SYNTH else
            ['Dry', 'Small Room', 'Studio', 'Concert Hall', 'Cathedral', 'Cave']
        )

        rrow = tk.Frame(panel, bg=_BG)
        rrow.pack(fill=tk.X, padx=8, pady=1)
        tk.Label(rrow, text='space', bg=_BG, fg=_SUBTEXT,
                 font=('Helvetica', 8)).pack(side=tk.LEFT)
        self._reverb_cb = ttk.Combobox(
            rrow, textvariable=self._reverb_preset_var,
            values=_preset_names, width=13, state='readonly',
            font=('Helvetica', 8))
        self._reverb_cb.pack(side=tk.RIGHT)
        self._reverb_preset_var.trace_add('write', self._on_reverb_preset_change)

        wrow = tk.Frame(panel, bg=_BG)
        wrow.pack(fill=tk.X, padx=8, pady=(1, 4))
        tk.Label(wrow, text='amount', bg=_BG, fg=_SUBTEXT,
                 font=('Helvetica', 8)).pack(side=tk.LEFT)
        self._reverb_wet_lbl = tk.Label(wrow, text='0%', bg=_BG, fg=_BLUE,
                                        font=('Helvetica', 8), width=4)
        self._reverb_wet_lbl.pack(side=tk.RIGHT)
        tk.Scale(
            wrow, variable=self._reverb_wet_var,
            from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL,
            bg=_BG, fg=_TEXT, troughcolor=_BG2,
            highlightthickness=0, sliderlength=14, showvalue=False,
            command=self._on_reverb_wet_change,
        ).pack(side=tk.RIGHT, fill=tk.X, expand=True)

        # Spacer + reset
        tk.Frame(panel, bg=_BG).pack(expand=True)
        ttk.Button(panel, text="Reset voice curve", command=self._reset).pack(
            fill=tk.X, padx=12, pady=(0, 10))

    def _build_footer(self) -> None:
        foot = tk.Frame(self.root, bg=_SURFACE, pady=7)
        foot.grid(row=1, column=0, columnspan=2, sticky='ew', padx=10, pady=(0, 10))

        def lbl(text):
            return tk.Label(foot, text=text, bg=_SURFACE, fg=_SUBTEXT,
                            font=('Helvetica', 8))

        def entry(var, width):
            return tk.Entry(foot, textvariable=var, width=width,
                            bg=_BG2, fg=_TEXT, insertbackground=_TEXT,
                            relief=tk.FLAT, font=('Helvetica', 9))

        self._duration_var = tk.StringVar(value='30')
        lbl("duration (s)").pack(side=tk.LEFT, padx=(10, 2))
        dur_e = entry(self._duration_var, 6)
        dur_e.pack(side=tk.LEFT, padx=(0, 4))
        dur_e.bind('<Return>',   self._on_duration_change)
        dur_e.bind('<FocusOut>', self._on_duration_change)

        tk.Frame(foot, bg=_OVERLAY, width=1).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        ttk.Button(foot, text="Save…",    command=self._save).pack(side=tk.LEFT, padx=(6, 3))
        ttk.Button(foot, text="Load…",    command=self._load).pack(side=tk.LEFT, padx=(0, 4))

        tk.Frame(foot, bg=_OVERLAY, width=1).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        ttk.Button(foot, text="▶ Play",    command=self._play).pack(side=tk.LEFT, padx=(6, 3))
        ttk.Button(foot, text="■ Stop",    command=self._stop).pack(side=tk.LEFT, padx=(0, 3))
        ttk.Button(foot, text="Save WAV…", command=self._save_wav).pack(side=tk.LEFT, padx=(0, 8))

        self._status_var = tk.StringVar(value="")
        tk.Label(foot, textvariable=self._status_var, bg=_SURFACE, fg=_GREEN,
                 font=('Helvetica', 8)).pack(side=tk.LEFT, padx=(0, 8))

    # ── Voice management ────────────────────────────────────────────────────

    def _update_voice_list(self) -> None:
        self._voice_lb.delete(0, tk.END)
        for v in self._voices:
            self._voice_lb.insert(tk.END, f"  {v['name']}")
        self._voice_lb.selection_clear(0, tk.END)
        self._voice_lb.selection_set(self._cur_v)
        self._load_voice_props()

    def _load_voice_props(self) -> None:
        self._loading_props = True
        v = self._cur_voice()
        self._v_name_var.set(v['name'])
        self._v_sound_var.set(v.get('sound_func', 'sampled_piano'))
        self._v_oct_var.set(v.get('octave_shift', 0))
        self._loading_props = False

    def _on_voice_select(self, _ev=None) -> None:
        sel = self._voice_lb.curselection()
        if sel:
            self._cur_v = int(sel[0])
            self._sel_zone = None
            self._load_voice_props()
            self._redraw()

    def _on_voice_prop_change(self, *_) -> None:
        if self._loading_props:
            return
        v = self._cur_voice()
        v['name']       = self._v_name_var.get()
        v['sound_func'] = self._v_sound_var.get()
        try:
            v['octave_shift'] = int(self._v_oct_var.get())
        except (tk.TclError, ValueError):
            pass
        self._update_voice_list()
        self._redraw()

    def _add_voice(self) -> None:
        idx     = len(self._voices)
        start_t = min(p[0] for p in self._cur_voice()['control_points'])
        self._voices.append(self._new_voice(f'v{idx + 1}', idx, start_t))
        self._cur_v = len(self._voices) - 1
        self._sel_zone = None
        self._update_voice_list()
        self._redraw()

    def _del_voice(self) -> None:
        if len(self._voices) <= 1:
            return
        self._voices.pop(self._cur_v)
        self._cur_v = min(self._cur_v, len(self._voices) - 1)
        self._sel_zone = None
        self._update_voice_list()
        self._redraw()

    # ── Zoom / Pan ──────────────────────────────────────────────────────────

    def _update_zoom_label(self) -> None:
        span = self._view_end - self._view_start
        self._zoom_lbl.config(text=f"{span:.1f}s view")

    def _update_scrollbar(self) -> None:
        first = self._view_start / max(self._total_dur, 1e-6)
        last  = self._view_end   / max(self._total_dur, 1e-6)
        self._hscroll.set(first, last)

    def _clamp_view(self) -> None:
        """Keep view window inside [0, total_dur]."""
        span = self._view_end - self._view_start
        self._view_start = max(0.0, min(self._total_dur - span, self._view_start))
        self._view_end   = self._view_start + span
        if self._view_end > self._total_dur:
            self._view_end   = self._total_dur
            self._view_start = max(0.0, self._total_dur - span)

    def _zoom_in(self) -> None:
        mid  = (self._view_start + self._view_end) / 2
        span = max(1.0, (self._view_end - self._view_start) * 0.6)
        self._view_start = mid - span / 2
        self._view_end   = mid + span / 2
        self._clamp_view()
        self._update_zoom_label()
        self._redraw()

    def _zoom_out(self) -> None:
        mid  = (self._view_start + self._view_end) / 2
        span = min(self._total_dur, (self._view_end - self._view_start) / 0.6)
        self._view_start = mid - span / 2
        self._view_end   = mid + span / 2
        self._clamp_view()
        self._update_zoom_label()
        self._redraw()

    def _pan_left(self) -> None:
        delta = (self._view_end - self._view_start) * 0.25
        self._view_start -= delta
        self._view_end   -= delta
        self._clamp_view()
        self._redraw()

    def _pan_right(self) -> None:
        delta = (self._view_end - self._view_start) * 0.25
        self._view_start += delta
        self._view_end   += delta
        self._clamp_view()
        self._redraw()

    def _on_wheel(self, ev: tk.Event) -> None:
        """Zoom centred on the mouse cursor."""
        direction = (-1 if ev.delta > 0 else 1) if hasattr(ev, 'delta') and ev.delta != 0 \
                    else (-1 if ev.num == 4 else 1)
        factor   = 0.8 if direction < 0 else 1 / 0.8
        old_span = self._view_end - self._view_start
        new_span = max(1.0, min(self._total_dur, old_span * factor))
        # Keep the time under the mouse fixed
        t_mouse  = self._x_to_t(ev.x) * self._total_dur
        ratio    = (t_mouse - self._view_start) / max(old_span, 1e-9)
        self._view_start = t_mouse - ratio * new_span
        self._view_end   = self._view_start + new_span
        self._clamp_view()
        self._update_zoom_label()
        self._redraw()

    def _on_hscroll(self, *args) -> None:
        span = self._view_end - self._view_start
        if args[0] == 'moveto':
            self._view_start = float(args[1]) * self._total_dur
            self._view_end   = self._view_start + span
        elif args[0] == 'scroll':
            delta = int(args[1]) * span * 0.1
            self._view_start += delta
            self._view_end   += delta
        self._clamp_view()
        self._update_zoom_label()
        self._redraw()

    # ── Zone logic ──────────────────────────────────────────────────────────

    @property
    def _zones(self) -> list:
        return self._cur_voice()['zones']

    def _add_zone(self) -> None:
        new = {'t_start': 0.0, 't_end': 1.0,
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
            self._zone_key_var.set(zone.get('key', 'C'))
            self._zone_mode_var.set(zone.get('mode', 'major'))
            self._zone_sel_lbl.config(
                text=f"{zone.get('key', '?')} {zone.get('mode', '?')}")
        else:
            self._zone_sel_lbl.config(text="— select a zone —")

    def _on_zone_mode_change(self) -> None:
        if not self._zone_edit_var.get():
            self._select_zone(None)
        self.canvas.config(cursor='fleur' if self._zone_edit_var.get() else 'crosshair')
        self._redraw()

    def _on_zone_prop_change(self, _=None) -> None:
        if self._sel_zone is not None:
            self._sel_zone['key']  = self._zone_key_var.get()
            self._sel_zone['mode'] = self._zone_mode_var.get()
            self._zone_sel_lbl.config(
                text=f"{self._sel_zone['key']} {self._sel_zone['mode']}")
            self._redraw()

    def _hit_zone(self, x: float, y: float):
        top = _MARGIN_T
        bot = _MARGIN_T + N_ROWS * _CELL_H
        if not (top <= y <= bot):
            return None, ''
        for zone in reversed(self._zones):
            zx1 = self._t_to_x(zone['t_start'])
            zx2 = self._t_to_x(zone['t_end'])
            if abs(x - zx1) <= _EDGE:
                return zone, 'left'
            if abs(x - zx2) <= _EDGE:
                return zone, 'right'
            if zx1 < x < zx2:
                return zone, 'body'
        return None, ''

    def _zones_to_scale_regions(self, voice: dict) -> list:
        out = []
        for z in voice['zones']:
            ts = max(0.0, min(1.0, z['t_start']))
            te = max(0.0, min(1.0, z['t_end']))
            if te > ts:
                out.append({'t_start': round(ts, 6), 't_end': round(te, 6),
                            'key': z.get('key', 'C'), 'mode': z.get('mode', 'major')})
        return out

    # ── Mouse events ────────────────────────────────────────────────────────

    def _on_click(self, ev: tk.Event) -> None:
        if self._zone_edit_var.get():
            zone, mode = self._hit_zone(ev.x, ev.y)
            if zone:
                self._zone_drag = {
                    'zone': zone, 'mode': mode, 'ox': ev.x,
                    'oz_ts': zone['t_start'], 'oz_te': zone['t_end'],
                }
                self._select_zone(zone)
            else:
                self._select_zone(None)
            self._redraw()
            return

        v   = self._cur_voice()
        hit = self._hit_pt(ev.x, ev.y, v['control_points'])
        if hit:
            self._drag_pt = hit
            return

        t_frac = self._clamp_t(self._x_to_t(ev.x))
        amp    = self._clamp_amp(self._y_to_amp(ev.y))
        existing = {p[0] for p in v['control_points']}
        while t_frac in existing:
            t_frac += 1e-5
        new_pt = [t_frac, amp]
        v['control_points'].append(new_pt)
        v['control_points'].sort(key=lambda p: p[0])
        self._drag_pt = new_pt
        self._redraw()

    def _on_drag(self, ev: tk.Event) -> None:
        if self._zone_drag is not None:
            dz   = self._zone_drag
            zone = dz['zone']
            mode = dz['mode']
            dt   = self._x_to_t(ev.x) - self._x_to_t(dz['ox'])
            min_frac = max(0.01, 2.0 / max(self._total_dur, 1.0))
            if mode == 'body':
                w  = dz['oz_te'] - dz['oz_ts']
                ts = max(0.0, min(1.0 - w, dz['oz_ts'] + dt))
                zone['t_start'] = ts
                zone['t_end']   = ts + w
            elif mode == 'left':
                zone['t_start'] = max(0.0, min(dz['oz_te'] - min_frac,
                                               dz['oz_ts'] + dt))
            elif mode == 'right':
                zone['t_end'] = max(dz['oz_ts'] + min_frac,
                                    min(1.0, dz['oz_te'] + dt))
            self._redraw()
            return

        if self._drag_pt is None:
            return
        self._drag_pt[0] = self._clamp_t(self._x_to_t(ev.x))
        self._drag_pt[1] = self._clamp_amp(self._y_to_amp(ev.y))
        self._cur_voice()['control_points'].sort(key=lambda p: p[0])
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
        v   = self._cur_voice()
        hit = self._hit_pt(ev.x, ev.y, v['control_points'])
        if hit and len(v['control_points']) > 2:
            v['control_points'].remove(hit)
            self._redraw()

    def _on_motion(self, ev: tk.Event) -> None:
        if not self._zone_edit_var.get():
            return
        _, mode = self._hit_zone(ev.x, ev.y)
        self.canvas.config(
            cursor='sb_h_double_arrow' if mode in ('left', 'right') else 'fleur')

    def _on_duration_change(self, _ev=None) -> None:
        try:
            new_dur = float(self._duration_var.get())
            if new_dur > 0:
                self._total_dur  = new_dur
                self._view_end   = min(self._view_end, new_dur)
                if self._view_start >= self._view_end:
                    self._view_start = 0.0
                    self._view_end   = new_dur
                self._update_zoom_label()
                self._redraw()
        except ValueError:
            pass

    # ── Reverb callbacks ────────────────────────────────────────────────────

    def _on_reverb_preset_change(self, *_) -> None:
        """When the preset changes, update the wet slider to the preset default."""
        if not _HAS_SYNTH:
            return
        preset = self._reverb_preset_var.get()
        params = _synth.REVERB_PRESETS.get(preset, {})
        wet    = float(params.get('wet', 0.0))
        self._reverb_wet_var.set(wet)
        self._reverb_wet_lbl.config(text=f'{int(wet * 100)}%')

    def _on_reverb_wet_change(self, value) -> None:
        wet = float(value)
        self._reverb_wet_lbl.config(text=f'{int(wet * 100)}%')

    # ── Hit testing ─────────────────────────────────────────────────────────

    def _hit_pt(self, x: float, y: float, pts: list, r: float = _PR + 6) -> Optional[list]:
        for pt in pts:
            px = self._t_to_x(pt[0])
            py = self._amp_to_y(pt[1])
            if (px - x) ** 2 + (py - y) ** 2 <= r * r:
                return pt
        return None

    # ── Reset ────────────────────────────────────────────────────────────────

    def _reset(self) -> None:
        v  = self._cur_voice()
        n  = 7
        xs = np.linspace(0.05, 0.95, n)
        ys = 0.5 + 0.3 * np.sin(np.linspace(0, 3 * np.pi, n))
        v['control_points'] = [[float(x), float(y)] for x, y in zip(xs, ys)]
        self._drag_pt = None
        self._redraw()

    # ── Drawing ──────────────────────────────────────────────────────────────

    def _redraw(self) -> None:
        self.canvas.delete('all')
        self._draw_grid()
        # Inactive voices first, active voice on top
        for i, v in enumerate(self._voices):
            if i != self._cur_v:
                self._draw_voice(v, active=False)
        self._draw_voice(self._cur_voice(), active=True)
        self._draw_zones(self._cur_voice())
        self._update_scrollbar()
        self._draw_playhead()   # always on top of everything

    def _right_x(self) -> float:
        cw = self.canvas.winfo_width()
        if cw <= 1:
            cw = _CANVAS_W
        return float(cw - 4)

    def _draw_grid(self) -> None:
        l  = float(_MARGIN_L)
        r  = self._right_x()
        t  = float(_MARGIN_T)
        b  = float(_MARGIN_T + N_ROWS * _CELL_H)

        # ── Horizontal pitch-row lines ────────────────────────────────────
        for i in range(N_ROWS + 1):
            row   = N_ROWS - i          # row 1 = bottom, N_ROWS = top
            y     = t + i * _CELL_H
            # Highlight centre row (C4) and every 7 rows (octave boundary)
            off   = row - _CENTER_ROW
            is_c4 = (row == _CENTER_ROW)
            is_oct = (off % 7 == 0)
            if is_c4:
                color, width = _BLUE, 1
            elif is_oct:
                color, width = _OVERLAY, 1
            else:
                color, width = _SURFACE, 1
            self.canvas.create_line(l, y, r, y, fill=color, width=width)

        # ── Left-axis labels ──────────────────────────────────────────────
        for row, label in _AXIS_LABELS.items():
            y_mid = t + (N_ROWS - row + 0.5) * _CELL_H
            color = _BLUE if row == _CENTER_ROW else _MUTED
            self.canvas.create_text(l - 4, y_mid, text=label,
                                    fill=color, font=('Helvetica', 7), anchor='e')

        # ── Vertical time lines ───────────────────────────────────────────
        view_span = self._view_end - self._view_start
        # Pick a human-readable step
        step = 1.0
        for s in [0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300]:
            if 5 <= view_span / s <= 25:
                step = s
                break

        import math
        t_cur = math.ceil(self._view_start / step) * step
        while t_cur <= self._view_end + step * 0.001:
            x = self._t_to_x(t_cur / self._total_dur)
            if l <= x <= r:
                is_origin = abs(t_cur) < 0.001
                col = _OVERLAY if is_origin else _SURFACE
                self.canvas.create_line(x, t, x, b, fill=col, width=1)
                lbl = f"{t_cur:.2g}s"
                self.canvas.create_text(x, b + 10, text=lbl,
                                        fill=_MUTED, font=('Helvetica', 7))
            t_cur = round(t_cur + step, 9)
            if t_cur > self._view_end + view_span:
                break  # safety

        # ── Shade region outside piece ────────────────────────────────────
        piece_x0 = self._t_to_x(0.0)
        piece_x1 = self._t_to_x(1.0)
        for shade_x1, shade_x2 in [(l, piece_x0), (piece_x1, r)]:
            if shade_x1 < shade_x2:
                self.canvas.create_rectangle(
                    shade_x1, t, shade_x2, b, fill=_BG, outline='', stipple='gray25')

        # ── Border ────────────────────────────────────────────────────────
        self.canvas.create_rectangle(l, t, r, b, outline=_OVERLAY, width=1)

    def _draw_zones(self, voice: dict) -> None:
        edit_mode = self._zone_edit_var.get()
        TAB_H = 16
        t = float(_MARGIN_T)
        b = float(_MARGIN_T + N_ROWS * _CELL_H)
        r = self._right_x()

        for zone in voice['zones']:
            zx1 = self._t_to_x(zone['t_start'])
            zx2 = self._t_to_x(zone['t_end'])
            # Clip to visible canvas
            zx1c = max(float(_MARGIN_L), zx1)
            zx2c = min(r, zx2)
            if zx2c <= zx1c:
                continue

            ci   = zone.get('color_idx', 0) % len(_ZONE_PALETTE)
            fill, outline = _ZONE_PALETTE[ci]
            selected = (zone is self._sel_zone)

            self.canvas.create_rectangle(
                zx1c, t, zx2c, b,
                fill='', outline=_TEXT if selected else outline,
                width=2 if selected else 1,
                dash=(6, 3) if not selected else (),
            )
            self.canvas.create_rectangle(
                zx1c, t, zx2c, t + TAB_H,
                fill=fill, outline='',
            )
            label = f"{zone.get('key', '?')} {zone.get('mode', '?')}"
            self.canvas.create_text(
                (zx1c + zx2c) / 2, t + TAB_H / 2,
                text=label, fill=_BG2, font=('Helvetica', 8, 'bold'),
            )

            if edit_mode:
                for ex in (zx1, zx2):
                    if float(_MARGIN_L) <= ex <= r:
                        self.canvas.create_rectangle(
                            ex - _EDGE, t + TAB_H + 4, ex + _EDGE, b - 4,
                            fill=_OVERLAY, outline=outline, width=1,
                        )

    def _draw_voice(self, voice: dict, active: bool = True) -> None:
        ci    = voice.get('color_idx', 0) % len(_VOICE_COLORS)
        color = _VOICE_COLORS[ci] if active else _MUTED
        width = 2 if active else 1

        pts = voice['control_points']
        if len(pts) < 2:
            return

        sorted_pts = sorted(pts, key=lambda p: p[0])
        xs = np.array([p[0] for p in sorted_pts])
        ys = np.array([p[1] for p in sorted_pts])
        _, idx = np.unique(xs, return_index=True)
        xs, ys = xs[idx], ys[idx]
        if len(xs) < 2:
            return

        n = max(500, len(xs) * 80)
        x_fine = np.linspace(xs[0], xs[-1], n)
        if _HAS_SCIPY and len(xs) >= 3:
            y_fine = CubicSpline(xs, ys)(x_fine)
        else:
            y_fine = np.interp(x_fine, xs, ys)

        r = self._right_x()
        coords: list = []

        for tf, amp in zip(x_fine.tolist(), y_fine.tolist()):
            cx = self._t_to_x(tf)
            cy = self._amp_to_y(amp)
            if float(_MARGIN_L) <= cx <= r:
                coords.extend((cx, cy))
            else:
                if len(coords) >= 4:
                    self.canvas.create_line(*coords, fill=color,
                                            width=width, smooth=False)
                coords = []

        if len(coords) >= 4:
            self.canvas.create_line(*coords, fill=color, width=width, smooth=False)

        # Control points (active voice only)
        if active:
            for pt in pts:
                px = self._t_to_x(pt[0])
                py = self._amp_to_y(pt[1])
                if float(_MARGIN_L) <= px <= r:
                    dot_color = _RED if (pt is self._drag_pt) else _GREEN
                    self.canvas.create_oval(px - _PR, py - _PR, px + _PR, py + _PR,
                                            fill=dot_color, outline=_TEXT, width=1)

    # ── Wave dict ───────────────────────────────────────────────────────────

    def _wave_dict(self) -> dict:
        voices = []
        for v in self._voices:
            voices.append({
                'name':           v['name'],
                'sound_func':     v.get('sound_func', 'sampled_piano'),
                'octave_shift':   int(v.get('octave_shift', 0)),
                'h_divisions':    N_ROWS,
                'control_points': [p[:] for p in v['control_points']],
                'scale_regions':  self._zones_to_scale_regions(v),
            })
        preset  = self._reverb_preset_var.get()
        rv_base = (_synth.REVERB_PRESETS.get(preset, {}) if _HAS_SYNTH else {})
        return {
            'format_version':   2,
            'bpm':              SCORE_BPM,
            'duration_seconds': self._total_dur,
            'reverb': {
                'preset':    preset,
                'room_size': float(rv_base.get('room_size', 0.0)),
                'damping':   float(rv_base.get('damping',   0.5)),
                'wet':       float(self._reverb_wet_var.get()),
            },
            'voices':           voices,
        }

    # ── Playback ────────────────────────────────────────────────────────────

    def _play(self) -> None:
        if not _HAS_SYNTH:
            messagebox.showerror("Missing module", "wave_synth.py not found.")
            return
        self._status_var.set("Rendering…")
        self.root.update_idletasks()

        def _worker():
            try:
                _synth.play(self._wave_dict(), blocking=False)
                self.root.after(0, self._start_playhead_poll)
                _synth.wait()
                self.root.after(0, self._on_playback_done)
            except RuntimeError as e:
                msg = str(e)
                self.root.after(0, lambda: self._status_var.set("No audio device"))
                self.root.after(0, lambda: messagebox.showwarning("Playback", msg))

        threading.Thread(target=_worker, daemon=True).start()
        self._status_var.set("Playing…")

    def _stop(self) -> None:
        if _HAS_SYNTH:
            _synth.stop()
        self._playhead_sec = None
        self._draw_playhead()
        self._status_var.set("Stopped")

    # ── Playhead ─────────────────────────────────────────────────────────────

    def _start_playhead_poll(self) -> None:
        self._status_var.set("Playing…")
        self._poll_playhead()

    def _poll_playhead(self) -> None:
        """Called every ~16 ms while audio is playing; updates only the playhead."""
        t = _synth.get_play_time() if _HAS_SYNTH else None
        if t is not None:
            self._playhead_sec = t
            self._draw_playhead()
            self.root.after(16, self._poll_playhead)
        else:
            self._playhead_sec = None
            self._draw_playhead()

    def _on_playback_done(self) -> None:
        self._playhead_sec = None
        self._draw_playhead()
        self._status_var.set("Done")

    def _draw_playhead(self) -> None:
        """Delete and redraw only the playhead line (no full canvas redraw)."""
        self.canvas.delete('playhead')
        if self._playhead_sec is None:
            return
        t_frac = self._playhead_sec / max(self._total_dur, 1e-6)
        x = self._t_to_x(t_frac)
        t = float(_MARGIN_T)
        b = float(_MARGIN_T + N_ROWS * _CELL_H)
        r = self._right_x()
        if float(_MARGIN_L) <= x <= r:
            self.canvas.create_line(x, t, x, b,
                                    fill='white', width=1,
                                    tags='playhead')
            self.canvas.create_text(x + 3, t - 10,
                                    text=f"{self._playhead_sec:.1f}s",
                                    fill=_TEXT, font=('Helvetica', 7),
                                    anchor='w', tags='playhead')

    def _save_wav(self) -> None:
        if not _HAS_SYNTH:
            messagebox.showerror("Missing module", "wave_synth.py not found.")
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

    # ── File I/O ────────────────────────────────────────────────────────────

    def _save(self) -> None:
        self.root.update()
        self.root.lift()
        path = filedialog.asksaveasfilename(
            title='Save wave', defaultextension='.json',
            filetypes=[('JSON', '*.json'), ('All files', '*.*')],
        )
        if not path:
            return
        preset  = self._reverb_preset_var.get()
        rv_base = (_synth.REVERB_PRESETS.get(preset, {}) if _HAS_SYNTH else {})
        data = {
            'format_version':   2,
            'bpm':              SCORE_BPM,
            'duration_seconds': self._total_dur,
            'view_start':       self._view_start,
            'view_end':         self._view_end,
            'reverb': {
                'preset':    preset,
                'room_size': float(rv_base.get('room_size', 0.0)),
                'damping':   float(rv_base.get('damping',   0.5)),
                'wet':       float(self._reverb_wet_var.get()),
            },
            'voices': [
                {
                    'name':           v['name'],
                    'sound_func':     v.get('sound_func', 'sampled_piano'),
                    'octave_shift':   int(v.get('octave_shift', 0)),
                    'color_idx':      v.get('color_idx', 0),
                    'control_points': [p[:] for p in v['control_points']],
                    'zones':          [dict(z) for z in v['zones']],
                }
                for v in self._voices
            ],
        }
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
        data    = json.loads(Path(path).read_text())
        version = data.get('format_version', 1)

        self._total_dur = float(data.get('duration_seconds', 30))
        self._duration_var.set(str(self._total_dur))

        rev = data.get('reverb', {})
        self._reverb_preset_var.set(rev.get('preset', 'Dry'))
        self._reverb_wet_var.set(float(rev.get('wet', 0.0)))
        self._reverb_wet_lbl.config(text=f"{int(float(rev.get('wet', 0.0)) * 100)}%")

        if version >= 2:
            self._view_start = float(data.get('view_start', 0))
            self._view_end   = float(data.get('view_end', self._total_dur))
            self._voices = []
            for i, rv in enumerate(data.get('voices', [])):
                v = {
                    'name':           rv.get('name', f'v{i + 1}'),
                    'sound_func':     rv.get('sound_func', 'sampled_piano'),
                    'octave_shift':   int(rv.get('octave_shift', 0)),
                    'color_idx':      rv.get('color_idx', i % len(_VOICE_COLORS)),
                    'control_points': [list(p) for p in rv.get('control_points', [])],
                    'zones':          [dict(z) for z in rv.get('zones', [])],
                }
                if not v['control_points']:
                    v = self._new_voice(v['name'], v['color_idx'])
                self._voices.append(v)

        else:
            # Legacy format v1: single voice, pixel-space coordinates
            self._view_start = 0.0
            self._view_end   = self._total_dur
            OLD_M, OLD_CW, OLD_CH = 38, 900, 520
            self._voices = []
            for i, rv in enumerate(data.get('voices', [])):
                raw_pts = sorted(rv.get('control_points', []), key=lambda p: p[0])
                new_pts = []
                if raw_pts:
                    xs = [p[0] for p in raw_pts]
                    x0, x1 = min(xs), max(xs)
                    for p in raw_pts:
                        tf  = (p[0] - x0) / (x1 - x0) if x1 > x0 else 0.5
                        amp = 1.0 - (p[1] - OLD_M) / (OLD_CH - 2 * OLD_M)
                        new_pts.append([float(tf), float(max(0.0, min(1.0, amp)))])

                new_zones = []
                for z in rv.get('zones', []):
                    ts = (z['x'] - OLD_M) / (OLD_CW - 2 * OLD_M)
                    te = (z['x'] + z['w'] - OLD_M) / (OLD_CW - 2 * OLD_M)
                    new_zones.append({
                        't_start':   float(max(0.0, min(1.0, ts))),
                        't_end':     float(max(0.0, min(1.0, te))),
                        'key':       z.get('key', 'C'),
                        'mode':      z.get('mode', 'major'),
                        'color_idx': z.get('color_idx', 0),
                    })
                if not new_zones:
                    new_zones = [{'t_start': 0.0, 't_end': 1.0,
                                  'key': 'C', 'mode': 'major', 'color_idx': 0}]

                v = self._new_voice(rv.get('name', f'v{i + 1}'), i)
                v['sound_func']   = rv.get('sound_func', 'sampled_piano')
                v['octave_shift'] = int(rv.get('octave_shift', 0))
                if new_pts:
                    v['control_points'] = new_pts
                v['zones'] = new_zones
                self._voices.append(v)

        if not self._voices:
            self._voices = [self._new_voice('v1', 0)]

        self._cur_v    = 0
        self._sel_zone = None
        self._update_voice_list()
        self._update_zoom_label()
        self.root.title(f"Wave Editor — {Path(path).name}")
        self._redraw()


# ── Launcher ────────────────────────────────────────────────────────────────

def launch() -> None:
    root = tk.Tk()
    WaveEditor(root)
    root.mainloop()


if __name__ == '__main__':
    launch()
