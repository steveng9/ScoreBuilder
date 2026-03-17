"""wave_synth.py
----------------
Direct audio rendering from .wave.json files.

Computes spline-crossing times as floating-point seconds, giving
sample-accurate timing with no integer-tick quantisation.  Each crossing
event triggers a synthesised note that decays naturally, so notes overlap
and ring like a real instrument.

Modulation logic
----------------
Every key zone uses the same centre-anchored row→degree formula:

    row_map[row] = d_center + (row - center_row)

where d_center is the scale degree nearest to _CENTER_MIDI (C4) in the
current zone's key/mode, and center_row = (h_divisions + 1) // 2.

This guarantees pitch-monotonic grid lines across any key change, provided
the same scale TYPE (e.g., all major, all minor) is used across zones.
Common tones near C4 naturally stay at the same row because the anchor
pitch is fixed regardless of key.  Works for any interval set: diatonic,
pentatonic, chromatic, 31-TET, JI, etc.

CLI
---
    python wave_synth.py <wave.json>             # play
    python wave_synth.py <wave.json> output.wav  # save WAV
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional

try:
    from scipy.interpolate import CubicSpline
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    import sounddevice as sd
    _HAS_SD = True
except ImportError:
    _HAS_SD = False

# ── Audio constants ────────────────────────────────────────────────────────
FS             = 44100    # sample rate
NOTE_DURATION  = 2.5      # seconds each triggered note is synthesised for
MAX_HOLD_SEC   = 1.5      # re-attack same note if held longer than this (s)

# ── Canvas geometry (must match wave_editor.py) ────────────────────────────
_CW     = 900
_CH     = 520
_MARGIN = 38

# ── Pitch vocabulary ───────────────────────────────────────────────────────
_KEY_SEMITONE: dict = {
    'C': 0, 'C#': 1, 'D': 2, 'Eb': 3, 'E': 4, 'F': 5,
    'F#': 6, 'G': 7, 'Ab': 8, 'A': 9, 'Bb': 10, 'B': 11,
}

_MODE_INTERVALS: dict = {
    'major':            [0, 2, 4, 5, 7, 9, 11],
    'minor':            [0, 2, 3, 5, 7, 8, 10],
    'dorian':           [0, 2, 3, 5, 7, 9, 10],
    'phrygian':         [0, 1, 3, 5, 7, 8, 10],
    'lydian':           [0, 2, 4, 6, 7, 9, 11],
    'mixolydian':       [0, 2, 4, 5, 7, 9, 10],
    'locrian':          [0, 1, 3, 5, 6, 8, 10],
    'pentatonic_major': [0, 2, 4, 7, 9],
    'pentatonic_minor': [0, 3, 5, 7, 10],
    'chromatic':        list(range(12)),
}

_TONIC_MIDI: dict = {k: v + 24 for k, v in _KEY_SEMITONE.items()}
_CENTER_MIDI: int = 60   # C4 — grid-centre anchor pitch


# ── Synthesisers ───────────────────────────────────────────────────────────

def _piano(freq: float, duration: float, fs: int = FS) -> np.ndarray:
    """Additive piano: harmonic series with individually-decaying partials."""
    n = int(duration * fs)
    t = np.arange(n, dtype=np.float32) / fs
    harmonics = [1,    2,    3,    4,    5,    6,    7,    8   ]
    amps      = [1.00, 0.50, 0.30, 0.18, 0.10, 0.06, 0.03, 0.02]
    decays    = [1.80, 3.50, 5.00, 7.00, 9.00, 11.0, 13.0, 16.0]
    sig = np.zeros(n, dtype=np.float32)
    for h, a, d in zip(harmonics, amps, decays):
        sig += a * np.exp(-d * t) * np.sin(2 * np.pi * freq * h * t)
    atk = min(int(0.008 * fs), n)
    sig[:atk] *= np.linspace(0, 1, atk, dtype=np.float32)
    return sig


def _bell(freq: float, duration: float, fs: int = FS) -> np.ndarray:
    """Crystal-bowl bell: inharmonic partials, long sustain."""
    n = int(duration * fs)
    t = np.arange(n, dtype=np.float32) / fs
    ratios = [1.000, 2.756, 5.404, 8.933, 13.34, 20.00]
    amps   = [1.000, 0.600, 0.300, 0.150, 0.075, 0.040]
    decays = [0.300, 0.600, 1.200, 2.400, 4.000, 6.500]
    sig = np.zeros(n, dtype=np.float32)
    for r, a, d in zip(ratios, amps, decays):
        sig += a * np.exp(-d * t) * np.sin(2 * np.pi * freq * r * t)
    atk = min(int(0.003 * fs), n)
    sig[:atk] *= np.linspace(0, 1, atk, dtype=np.float32)
    return sig


def _marimba(freq: float, duration: float, fs: int = FS) -> np.ndarray:
    """Woody marimba/kalimba: few inharmonic partials, fast decay."""
    n = int(duration * fs)
    t = np.arange(n, dtype=np.float32) / fs
    ratios = [1.0, 4.0,  10.0 ]
    amps   = [1.0, 0.35,  0.12]
    decays = [4.0, 8.0,  16.0 ]
    sig = np.zeros(n, dtype=np.float32)
    for r, a, d in zip(ratios, amps, decays):
        sig += a * np.exp(-d * t) * np.sin(2 * np.pi * freq * r * t)
    atk = min(int(0.004 * fs), n)
    sig[:atk] *= np.linspace(0, 1, atk, dtype=np.float32)
    return sig


def _pad(freq: float, duration: float, fs: int = FS) -> np.ndarray:
    """Ethereal pad: slow attack, slight vibrato, detuned chorus."""
    n = int(duration * fs)
    t = np.arange(n, dtype=np.float32) / fs
    vib = (1 + 0.003 * np.sin(2 * np.pi * 5.5 * t)).astype(np.float32)
    detunes = [0.0, 0.003, -0.003]
    sig = sum(np.sin(2 * np.pi * freq * (1 + d) * vib * t) for d in detunes)
    sig = (sig / len(detunes)).astype(np.float32)
    # Slow attack + global exponential fade
    env = np.ones(n, dtype=np.float32)
    atk = min(int(0.15 * fs), n)
    env[:atk] = np.linspace(0, 1, atk, dtype=np.float32)
    sig *= env * np.exp(-0.4 * t)
    return sig


def _shimmer(freq: float, duration: float, fs: int = FS) -> np.ndarray:
    """Bright shimmer: multiple octave-spaced harmonics with tremolo."""
    n = int(duration * fs)
    t = np.arange(n, dtype=np.float32) / fs
    trem = (1 + 0.15 * np.sin(2 * np.pi * 7.0 * t)).astype(np.float32)
    harmonics = [1, 2, 4, 8]
    amps      = [1.0, 0.5, 0.25, 0.12]
    decays    = [1.5, 2.5, 4.0,  6.0 ]
    sig = np.zeros(n, dtype=np.float32)
    for h, a, d in zip(harmonics, amps, decays):
        sig += a * np.exp(-d * t) * np.sin(2 * np.pi * freq * h * t)
    sig *= trem
    atk = min(int(0.005 * fs), n)
    sig[:atk] *= np.linspace(0, 1, atk, dtype=np.float32)
    return sig


_SYNTH_MAP: dict = {
    'piano_sound':      _piano,
    'piano':            _piano,
    'crystal_bowl':     _bell,
    'crystal_bowl_with_pop': _bell,
    'rich_bell':        _bell,
    'soft_bell':        _bell,
    'bell':             _bell,
    'tubular_bell':     _bell,
    'marimba':          _marimba,
    'soft_marimba':     _marimba,
    'soft_kalimba':     _marimba,
    'tonal_percussion': _marimba,
    'vibraphone':       _marimba,
    'pad':              _pad,
    'ethereal_pad':     _pad,
    'layered_chorus':   _pad,
    'shimmer':          _shimmer,
    'magic_shimmer':    _shimmer,
    'bright_crystalline': _shimmer,
}

def _get_synth(name: str):
    return _SYNTH_MAP.get(name, _piano)


# ── Pitch helpers ──────────────────────────────────────────────────────────

def _degree_to_midi(degree: int, tonic_midi: int, intervals: list) -> int:
    """Convert scale degree to absolute MIDI note number.

    degree=1 → tonic, degree=8 → tonic+octave, degree=0 → leading tone below.
    Negative degrees extend below the tonic using Python floor-division.
    """
    i         = degree - 1
    scale_len = len(intervals)
    oct_steps = i // scale_len
    step      = i % scale_len
    return tonic_midi + 12 * oct_steps + intervals[step]


def _nearest_degree(ref_midi: float, tonic_midi: int, intervals: list) -> int:
    """Return the scale degree whose pitch is nearest to ref_midi."""
    best_deg, best_diff = 1, float('inf')
    for d in range(-28, 36):
        diff = abs(_degree_to_midi(d, tonic_midi, intervals) - ref_midi)
        if diff < best_diff:
            best_diff, best_deg = diff, d
    return best_deg


def _midi_to_freq(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def _build_row_degree_map(h_divisions: int, tonic_midi: int,
                          intervals: list) -> dict:
    """Centre-anchored row→degree map for one key zone.

    The centre row maps to the scale degree nearest to _CENTER_MIDI (C4).
    Every row above/below is the next/previous scale degree.  Using this
    formula identically for every zone (not just the first) gives
    pitch-monotonic grids across key changes: higher row always = higher
    pitch, for any scale type with any root.
    """
    center_row = (h_divisions + 1) // 2
    d_center   = _nearest_degree(_CENTER_MIDI, tonic_midi, intervals)
    return {row: d_center + (row - center_row)
            for row in range(1, h_divisions + 1)}


# ── Zone helpers ───────────────────────────────────────────────────────────

def _active_region(scale_regions: list, norm_t: float,
                   v_divisions: int) -> Optional[dict]:
    """Return the active scale region at norm_t (same policy as wave_translator)."""
    if not scale_regions:
        return None
    col = norm_t * v_divisions
    for r in reversed(scale_regions):
        if r['col_start'] <= col < r['col_end']:
            return r
    # Gap: return left-neighbour zone, or first zone if before all zones
    sorted_r = sorted(scale_regions, key=lambda r: r['col_start'])
    prev = None
    for r in sorted_r:
        if r['col_start'] <= col:
            prev = r
        else:
            break
    return prev if prev is not None else sorted_r[0]


# ── Curve and crossing detection ───────────────────────────────────────────

def _build_curve(control_points: list, n: int = 100_000):
    pts = sorted(control_points, key=lambda p: p[0])
    xs  = np.array([p[0] for p in pts], dtype=float)
    ys  = np.array([p[1] for p in pts], dtype=float)
    _, idx = np.unique(xs, return_index=True)
    xs, ys = xs[idx], ys[idx]
    if len(xs) < 2:
        return np.array([]), np.array([])
    if _HAS_SCIPY and len(xs) >= 3:
        x_fine = np.linspace(xs[0], xs[-1], n)
        y_fine = CubicSpline(xs, ys)(x_fine)
    else:
        x_fine = np.linspace(xs[0], xs[-1], n)
        y_fine = np.interp(x_fine, xs, ys)
    grid_top = float(_MARGIN)
    grid_bot = float(_CH - _MARGIN)
    t   = (x_fine - x_fine.min()) / (x_fine.max() - x_fine.min())
    amp = 1.0 - (y_fine - grid_top) / (grid_bot - grid_top)
    return t, np.clip(amp, 0.0, 1.0)


def _crossing_events(control_points: list, h_divisions: int,
                     dur_sec: float, n: int = 100_000) -> list:
    """Return list of (time_sec: float, row: int) crossing events.

    Each entry is the moment the spline curve enters a new grid row.
    Times are floating-point seconds — no tick quantisation.
    The final entry is (dur_sec, None) as a sentinel.
    """
    t, amp = _build_curve(control_points, n)
    if t.size == 0:
        return [(0.0, 1), (dur_sec, None)]

    rows = (np.floor(amp * h_divisions).astype(int) + 1).clip(1, h_divisions)
    events: list = [(0.0, int(rows[0]))]

    for i in range(len(rows) - 1):
        r0, r1 = int(rows[i]), int(rows[i + 1])
        if r0 == r1:
            continue
        a0, a1 = float(amp[i]), float(amp[i + 1])
        t0, t1 = float(t[i]),   float(t[i + 1])
        step    = 1 if r1 > r0 else -1

        for r_new in range(r0 + step, r1 + step, step):
            bdry = min(r_new, r_new - step) / h_divisions
            frac = (bdry - a0) / (a1 - a0) if abs(a1 - a0) > 1e-12 else 0.5
            frac  = max(0.0, min(1.0, frac))
            t_sec = (t0 + frac * (t1 - t0)) * dur_sec
            events.append((t_sec, r_new))

    events.append((dur_sec, None))

    # Deduplicate: multiple crossings at the same instant → keep the last row
    deduped: list = []
    for entry in events:
        if deduped and abs(deduped[-1][0] - entry[0]) < 1e-9:
            deduped[-1] = list(entry)
        else:
            deduped.append(list(entry))
    return deduped


def _insert_reattacks(events: list, max_hold: float) -> list:
    """Insert synthetic re-attack events for rows held longer than max_hold seconds.

    Without this, a note that decays while the wave stays on one row
    goes silent mid-piece.  Re-attacks keep the sound alive.
    """
    out: list = []
    for i, (t_start, row) in enumerate(events[:-1]):
        out.append((t_start, row))
        if row is None:
            continue
        t_end = events[i + 1][0]
        pos   = t_start + max_hold
        while pos < t_end - 0.02:
            out.append((pos, row))
            pos += max_hold
    out.append(events[-1])   # sentinel
    return out


# ── Main render ────────────────────────────────────────────────────────────

def render(wave_data, output_wav_path: Optional[str] = None,
           fs: int = FS) -> np.ndarray:
    """Render a wave dict (or path to .json) to a normalised audio buffer.

    Parameters
    ----------
    wave_data : dict | str | Path
        Parsed wave JSON dict, or path to a .json file.
    output_wav_path : str | Path, optional
        If provided, write a 32-bit float WAV to this path.
    fs : int
        Sample rate in Hz.

    Returns
    -------
    np.ndarray  shape (n_samples,), dtype float32, peak ≤ 0.9
    """
    if not isinstance(wave_data, dict):
        wave_data = json.loads(Path(wave_data).read_text())

    dur_sec = float(wave_data['duration_seconds'])
    voices  = wave_data['voices']
    # Extra tail so the last notes can ring out naturally
    n_total = int(dur_sec * fs) + int(NOTE_DURATION * fs)
    mix     = np.zeros(n_total, dtype=np.float64)

    for v in voices:
        h_div      = int(v['h_divisions'])
        v_div      = int(v.get('v_divisions', 16))
        regions    = v.get('scale_regions', [])
        oct_shift  = int(v.get('octave_shift', 0))
        synth_func = _get_synth(v.get('sound_func', 'piano_sound'))

        events = _crossing_events(v['control_points'], h_div, dur_sec)
        events = _insert_reattacks(events, MAX_HOLD_SEC)

        for i, (t_start, row) in enumerate(events[:-1]):
            if row is None:
                continue

            # Determine active key zone
            norm_t  = t_start / dur_sec if dur_sec > 0 else 0.0
            region  = _active_region(regions, norm_t, v_div)
            if region is None:
                continue

            key       = region.get('key', 'C')
            mode      = region.get('mode', 'major')
            intervals = _MODE_INTERVALS.get(mode, _MODE_INTERVALS['major'])
            tonic     = _TONIC_MIDI.get(key, 24)

            # Centre-anchored row → degree → MIDI → Hz
            row_map = _build_row_degree_map(h_div, tonic, intervals)
            degree  = row_map[row]
            midi    = _degree_to_midi(degree, tonic, intervals) + 12 * oct_shift
            freq    = _midi_to_freq(float(midi))

            if freq <= 20.0 or freq > 20000.0 or not np.isfinite(freq):
                continue

            # Synthesise note
            note = synth_func(freq, NOTE_DURATION, fs)

            # Mix into buffer at the exact floating-point start time
            start = int(t_start * fs)
            end   = start + len(note)
            if start >= n_total:
                continue
            if end > n_total:
                note = note[:n_total - start]
                end  = n_total
            mix[start:end] += note

    # Normalise to 0.9 peak
    peak = np.max(np.abs(mix))
    if peak > 1e-8:
        mix *= 0.9 / peak

    audio = mix.astype(np.float32)

    if output_wav_path:
        import scipy.io.wavfile as wf
        wf.write(str(output_wav_path), fs, audio)
        print(f"Wrote {output_wav_path}")

    return audio


def play(wave_data, blocking: bool = True) -> None:
    """Render and play through speakers.

    Requires the `sounddevice` package:  pip install sounddevice
    """
    if not _HAS_SD:
        raise RuntimeError(
            "sounddevice is not installed.\n"
            "Install it with:  pip install sounddevice\n"
            "Or save a WAV file instead."
        )
    audio = render(wave_data)
    sd.play(audio, FS)
    if blocking:
        sd.wait()


def stop() -> None:
    """Stop any currently playing audio."""
    if _HAS_SD:
        sd.stop()


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python wave_synth.py <wave.json> [output.wav]")
        sys.exit(1)
    src = sys.argv[1]
    if len(sys.argv) >= 3:
        render(src, sys.argv[2])
    else:
        try:
            print("Rendering…")
            play(src, blocking=True)
        except RuntimeError as e:
            print(e)
            out = Path(src).with_suffix('.wav')
            render(src, str(out))
