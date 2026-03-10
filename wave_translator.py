"""wave_translator.py
--------------------
Translates a saved .wave.json file into a .txt score file.

Constants
---------
SCORE_BPM        : int  — eighth-note tick rate (not a musical tempo).
                          Set high (e.g. 960) for fine time resolution.
                          Change here to tune precision globally.
TICKS_PER_MEASURE: int  — score-file measure length in eighth notes.
                          Purely for readability; does not affect timing.

CLI usage
---------
    python wave_translator.py input.wave.json [output.txt]
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from scipy.interpolate import CubicSpline
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# ── Tuneable constants ─────────────────────────────────────────────────────
SCORE_BPM         = 960   # tick rate; raise for finer resolution
TICKS_PER_MEASURE = 99999 # eighth notes per measure bar.
                          # Setting this larger than the total piece length
                          # produces zero internal | barlines, eliminating
                          # the re-attack chop at every bar boundary.
                          # Lower it (e.g. 32) only if you want visible bars.
MAX_NOTE_TICKS    = 1     # max eighth-note duration per note event.
                          # Piano sound buffers are ~0.56s; at BPM=960 that's
                          # ~18 ticks.  Keep this at or below 16 to stay safe.

# ── Canvas geometry — must stay in sync with wave_editor.py ───────────────
_CW     = 900
_CH     = 520
_MARGIN = 38


# ── Curve reconstruction ───────────────────────────────────────────────────

def _build_curve(control_points: list, n: int = 10_000):
    """Reconstruct normalised (t, amp) from saved canvas control points."""
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


# ── Timing helpers ─────────────────────────────────────────────────────────

def _to_tick(t_sec: float, bpm: int) -> int:
    """Absolute time in seconds → eighth-note tick index."""
    return round(t_sec * bpm / 30.0)


# ── Scale region lookup ────────────────────────────────────────────────────

def _active_region(scale_regions: list, norm_t: float,
                   v_divisions: int) -> Optional[dict]:
    col = norm_t * v_divisions
    for r in scale_regions:
        if r['col_start'] <= col < r['col_end']:
            return r
    return None


# ── Crossing detection ─────────────────────────────────────────────────────

def _crossings(t: np.ndarray, amp: np.ndarray,
               h_divisions: int, scale_regions: list,
               v_divisions: int, bpm: int,
               dur_sec: float) -> list:
    """Return note events: list of {tick, duration, row, scale}.

    A new event is emitted whenever the curve moves to a different pitch row
    OR enters a different scale region.  Row 1 = bottom, row h_divisions = top.
    """
    rows = (np.floor(amp * h_divisions).astype(int) + 1).clip(1, h_divisions)

    events    = []
    prev_row   = int(rows[0])
    prev_tick  = _to_tick(float(t[0]) * dur_sec, bpm)
    prev_scale = _active_region(scale_regions, float(t[0]), v_divisions)

    for i in range(1, len(rows)):
        cur_row   = int(rows[i])
        cur_scale = _active_region(scale_regions, float(t[i]), v_divisions)

        if cur_row != prev_row or cur_scale != prev_scale:
            cur_tick = _to_tick(float(t[i]) * dur_sec, bpm)
            dur      = cur_tick - prev_tick
            if dur > 0:
                events.append(dict(tick=prev_tick, duration=dur,
                                   row=prev_row, scale=prev_scale))
            prev_row, prev_tick, prev_scale = cur_row, cur_tick, cur_scale

    # Final event
    final_tick = _to_tick(dur_sec, bpm)
    dur = final_tick - prev_tick
    if dur > 0:
        events.append(dict(tick=prev_tick, duration=dur,
                           row=prev_row, scale=prev_scale))
    return events


# ── Degree mapping ─────────────────────────────────────────────────────────

def _row_to_degree(row: int, scale: Optional[dict]) -> str:
    """Map a grid row (1-indexed from bottom) to a score degree token.

    Currently: row number == scale degree (1..h_divisions).
    When scale regions carry richer data (accidentals, modal offsets, etc.)
    this function is the single place to extend that logic.
    """
    return str(row)


# ── Jitter filter ─────────────────────────────────────────────────────────

def _merge_consecutive(events: list) -> list:
    """Merge adjacent events with the same (row, scale) pair.

    Floating-point noise near a row boundary can flip a sample back and forth
    between two rows, producing spurious zero- or one-tick events.  Merging
    same-row neighbours collapses them before the note-length limiter runs.
    """
    if not events:
        return events
    merged = [dict(events[0])]
    for ev in events[1:]:
        if ev['row'] == merged[-1]['row'] and ev['scale'] == merged[-1]['scale']:
            merged[-1]['duration'] += ev['duration']
        else:
            merged.append(dict(ev))
    return merged


# ── Note-length limiter ────────────────────────────────────────────────────

def _split_long_notes(events: list, max_ticks: int) -> list:
    """Break any event longer than max_ticks into repeated same-pitch events.

    Each sub-event gets its own correct tick value so that section filtering
    in translate() assigns sub-events to the right key section.
    Re-attacks at BPM=960 are ~31ms each — inaudible as gaps for piano.
    """
    out = []
    for ev in events:
        remaining = ev['duration']
        tick      = ev['tick']
        while remaining > max_ticks:
            out.append(dict(ev, tick=tick, duration=max_ticks))
            tick      += max_ticks
            remaining -= max_ticks
        if remaining > 0:
            out.append(dict(ev, tick=tick, duration=remaining))
    return out


# ── Score line formatter ───────────────────────────────────────────────────

def _format_voice(events: list, total_ticks: int,
                  ticks_per_measure: int) -> str:
    """Render a voice's event list as a score notation line.

    Notes that cross a measure boundary are split and re-notated in the next
    measure with the same degree (re-attack).  At SCORE_BPM=960 an eighth note
    is ~31ms, so re-attacks at measure boundaries are musically imperceptible.
    """
    if not events:
        return f"r.{total_ticks}"

    measures:     list  = []
    cur_measure:  list  = []
    ticks_in_bar: int   = 0

    # Work on a copy so we can mutate durations during split
    queue = [dict(e) for e in events]
    qi    = 0

    while qi < len(queue):
        ev      = queue[qi]
        space   = ticks_per_measure - ticks_in_bar
        take    = min(ev['duration'], space)
        token   = _row_to_degree(ev['row'], ev['scale'])
        cur_measure.append(f"{token}.{take}-")
        ticks_in_bar  += take
        ev['duration'] -= take

        if ev['duration'] > 0:
            # Note straddles bar boundary — close bar, carry remainder forward
            measures.append(', '.join(cur_measure))
            cur_measure  = []
            ticks_in_bar = 0
            # Don't advance qi; ev still has remaining duration
        else:
            qi += 1

        if ticks_in_bar == ticks_per_measure:
            measures.append(', '.join(cur_measure))
            cur_measure  = []
            ticks_in_bar = 0

    if cur_measure:
        measures.append(', '.join(cur_measure))

    return ' | '.join(measures)


# ── Section boundary helpers ───────────────────────────────────────────────

def _section_boundaries(voices: list, bpm: int,
                        dur_sec: float, total_ticks: int) -> list:
    """Return sorted tick positions at which any voice's scale region starts/ends."""
    pts = {0, total_ticks}
    for v in voices:
        vd = v.get('v_divisions', 16)
        for r in v.get('scale_regions', []):
            pts.add(_to_tick(r['col_start'] / vd * dur_sec, bpm))
            pts.add(_to_tick(r['col_end']   / vd * dur_sec, bpm))
    return sorted(pts)


def _key_at_tick(voices: list, tick: int, total_ticks: int) -> str:
    """Key string for the section starting at *tick*, driven by voice[0]."""
    if not voices:
        return 'C major'
    v  = voices[0]
    vd = v.get('v_divisions', 16)
    norm_t = tick / total_ticks if total_ticks > 0 else 0.0
    r = _active_region(v.get('scale_regions', []), norm_t, vd)
    return f"{r['key']} {r['mode']}" if r else 'C major'


# ── Public API ─────────────────────────────────────────────────────────────

def translate(wave_path: str, score_path: Optional[str] = None) -> str:
    """Load *wave_path* (.wave.json) and produce a score .txt string."""
    data        = json.loads(Path(wave_path).read_text())
    bpm         = int(data.get('bpm', SCORE_BPM))
    dur_sec     = float(data['duration_seconds'])
    voices      = data['voices']
    tpm         = int(data.get('ticks_per_measure', TICKS_PER_MEASURE))
    max_ticks   = int(data.get('max_note_ticks', MAX_NOTE_TICKS))
    total_ticks = _to_tick(dur_sec, bpm)

    lines = []

    # ── Voice declarations ────────────────────────────────────────────────
    for v in voices:
        lines.append(f"voice: {v['name']}, @{v['sound_func']}, {v['octave_shift']}")
    lines.append('')

    # ── Build all events upfront ──────────────────────────────────────────
    all_events: list = []
    for v in voices:
        t, amp = _build_curve(v['control_points'])
        if t.size == 0:
            all_events.append([])
            continue
        evs = _crossings(t, amp,
                         h_divisions   = v['h_divisions'],
                         scale_regions = v.get('scale_regions', []),
                         v_divisions   = v['v_divisions'],
                         bpm           = bpm,
                         dur_sec       = dur_sec)
        evs = _merge_consecutive(evs)
        evs = _split_long_notes(evs, max_ticks)
        all_events.append(evs)

    # ── Emit one score section per scale-region boundary ─────────────────
    boundaries = _section_boundaries(voices, bpm, dur_sec, total_ticks)

    for sec_start, sec_end in zip(boundaries, boundaries[1:]):
        sec_ticks = sec_end - sec_start
        if sec_ticks <= 0:
            continue

        lines.append(f"qtr_note = {bpm}")
        lines.append(_key_at_tick(voices, sec_start, total_ticks))
        lines.append('')

        for evs in all_events:
            sec_evs = [e for e in evs if sec_start <= e['tick'] < sec_end]
            if not sec_evs:
                lines.append(f"r.{sec_ticks}")
            else:
                line = _format_voice(sec_evs, sec_ticks, min(tpm, sec_ticks))
                # Pad any gap so the section duration is exact
                ev_sum = sum(e['duration'] for e in sec_evs)
                if ev_sum < sec_ticks:
                    line += f", r.{sec_ticks - ev_sum}"
                lines.append(line)
        lines.append('')

    score = '\n'.join(lines).rstrip('\n') + '\n'

    if score_path:
        Path(score_path).write_text(score)
        print(f"Wrote {score_path}")

    return score


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python wave_translator.py <input.wave.json> [output.txt]")
        sys.exit(1)
    result = translate(sys.argv[1], sys.argv[2] if len(sys.argv) >= 3 else None)
    if len(sys.argv) < 3:
        print(result)
