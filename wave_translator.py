"""wave_translator.py
--------------------
Translates a saved .wave.json file into a .txt score file.

Constants
---------
SCORE_BPM        : int  — eighth-note tick rate (not a musical tempo).
                          Set high (e.g. 9600) for fine time resolution.
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
SCORE_BPM         = 9600   # tick rate; raise for finer resolution
TICKS_PER_MEASURE = 99999  # eighth notes per measure bar.
                           # Setting this larger than the total piece length
                           # produces zero internal | barlines, eliminating
                           # the re-attack chop at every bar boundary.
                           # Lower it (e.g. 32) only if you want visible bars.
MAX_NOTE_TICKS    = 250    # max eighth-note duration per note event (re-attack).
                           # At BPM=9600: 250 ticks ≈ 0.78 s — safe for piano.

# ── Canvas geometry — must stay in sync with wave_editor.py ───────────────
_CW     = 900
_CH     = 520
_MARGIN = 38

# ── Pitch vocabulary ────────────────────────────────────────────────────────
# Semitone offset (0–11) for each key name.
_KEY_SEMITONE: dict = {
    'C': 0, 'C#': 1, 'D': 2, 'Eb': 3, 'E': 4, 'F': 5,
    'F#': 6, 'G': 7, 'Ab': 8, 'A': 9, 'Bb': 10, 'B': 11,
}

# Diatonic semitone steps (from tonic) for each mode.
_MODE_INTERVALS: dict = {
    'major':      [0, 2, 4, 5, 7, 9, 11],
    'minor':      [0, 2, 3, 5, 7, 8, 10],
    'dorian':     [0, 2, 3, 5, 7, 9, 10],
    'phrygian':   [0, 1, 3, 5, 7, 8, 10],
    'lydian':     [0, 2, 4, 6, 7, 9, 11],
    'mixolydian': [0, 2, 4, 5, 7, 9, 10],
    'locrian':    [0, 1, 3, 5, 6, 8, 10],
}

# MIDI number of degree-1 (tonic) in MATLAB's octave-1 convention.
# Standard MIDI: C4=60, so C1=24, C#1=25, …
_TONIC_MIDI: dict = {k: v + 24 for k, v in _KEY_SEMITONE.items()}

# MIDI anchor for grid centering: middle row of the grid targets this pitch.
_CENTER_MIDI: int = 60   # C4


# ── Curve reconstruction ───────────────────────────────────────────────────

def _build_curve(control_points: list, n: int = 100_000):
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


# ── Scale region helpers ────────────────────────────────────────────────────

def _same_musical_key(a: Optional[dict], b: Optional[dict]) -> bool:
    """True iff both scale dicts share the same key and mode (or both are None)."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return a.get('key') == b.get('key') and a.get('mode') == b.get('mode')


def _active_region(scale_regions: list, norm_t: float,
                   v_divisions: int) -> Optional[dict]:
    """Return the scale region active at norm_t.

    Overlap policy : last zone in the list wins (consistent with the editor's
                     visual stack order — later zones drawn on top).
    Gap policy     : when norm_t falls outside all zones, extend the nearest
                     preceding zone (or the first zone if before all zones).
                     This prevents silent fallback to raw row numbers / C major.
    """
    if not scale_regions:
        return None
    col = norm_t * v_divisions
    # Last-added zone wins for overlaps (iterate in reverse)
    for r in reversed(scale_regions):
        if r['col_start'] <= col < r['col_end']:
            return r
    # Gap: find the zone whose col_start is closest and <= col (left neighbour)
    sorted_r = sorted(scale_regions, key=lambda r: r['col_start'])
    prev = None
    for r in sorted_r:
        if r['col_start'] <= col:
            prev = r
        else:
            break
    return prev if prev is not None else sorted_r[0]


# ── Degree / pitch helpers ──────────────────────────────────────────────────

def _degree_to_midi(degree: int, tonic_midi: int, intervals: list) -> int:
    """Convert a scale degree to an absolute MIDI note number.

    Uses Python floor-division so negative degrees extend correctly below tonic.
    degree=1 → tonic_midi, degree=8 → tonic+octave, degree=0 → leading tone below.
    """
    i         = degree - 1
    scale_len = len(intervals)
    oct_steps = i // scale_len   # floor: negative when below tonic
    step      = i % scale_len    # always non-negative (Python semantics)
    return tonic_midi + 12 * oct_steps + intervals[step]


def _nearest_degree(ref_midi: float, tonic_midi: int, intervals: list) -> int:
    """Return the scale degree whose pitch is nearest to ref_midi."""
    best_deg  = 1
    best_diff = float('inf')
    for d in range(-28, 36):   # ±4 octaves from tonic — safe for any grid
        diff = abs(_degree_to_midi(d, tonic_midi, intervals) - ref_midi)
        if diff < best_diff:
            best_diff = diff
            best_deg  = d
    return best_deg


def _build_row_degree_maps(sections_info: list, h_divisions: int) -> list:
    """Build a per-section row→degree-string dict with pitch-continuous remapping.

    Design
    ------
    Grid rows are scale steps — row 1 is the lowest step, row h_divisions the
    highest.  This makes rhythm uniform: a straight wave crossing h_divisions
    rows at constant speed produces evenly-spaced note events regardless of the
    scale's interval structure.

    First section
        The middle row is anchored to _CENTER_MIDI (≈ C4).  Rows above/below
        the centre are consecutive scale degrees above/below that anchor.

    Subsequent sections
        Each row carries a "pitch context" (the MIDI pitch it produced in the
        previous section).  The new degree is the one in the new key whose pitch
        is nearest to that context.  For maximally-distant keys (e.g. C↔F#)
        the remap is at most ½ semitone — completely smooth modulation.

    Args
        sections_info : list of {'key': str, 'mode': str}, one per section
        h_divisions   : number of grid rows

    Returns
        List of {row(int): degree_str}, parallel to sections_info.
    """
    maps: list = []
    pitch_context: dict = {}   # row → MIDI pitch produced in the previous section

    for sec_idx, sec in enumerate(sections_info):
        key        = sec.get('key', 'C')
        mode       = sec.get('mode', 'major')
        intervals  = _MODE_INTERVALS.get(mode, _MODE_INTERVALS['major'])
        tonic_midi = _TONIC_MIDI.get(key, 24)
        row_map: dict = {}

        if sec_idx == 0:
            # Centre the grid on _CENTER_MIDI
            center_row = (h_divisions + 1) // 2
            d_center   = _nearest_degree(_CENTER_MIDI, tonic_midi, intervals)
            for row in range(1, h_divisions + 1):
                deg = d_center + (row - center_row)
                row_map[row] = str(deg)
                pitch_context[row] = float(_degree_to_midi(deg, tonic_midi, intervals))
        else:
            for row in range(1, h_divisions + 1):
                ref = pitch_context.get(row, float(_CENTER_MIDI))
                deg = _nearest_degree(ref, tonic_midi, intervals)
                row_map[row] = str(deg)
                pitch_context[row] = float(_degree_to_midi(deg, tonic_midi, intervals))

        maps.append(row_map)

    return maps


# ── Event builder (crossing-based) ─────────────────────────────────────────

def _build_events(control_points: list,
                  h_divisions: int, scale_regions: list,
                  v_divisions: int, bpm: int, dur_sec: float,
                  section_maps: list) -> list:
    """Build note events by detecting grid-row crossings in the spline curve.

    Uses the high-resolution curve (100 k samples) to find the precise time
    each grid-row boundary is crossed, then rounds to the nearest tick.
    This gives sub-tick accuracy (crossing position is interpolated to within
    ~0.14 ms for a 14-second piece) rather than tick-level granularity.

    Events only split when the scale degree changes — key/mode zone boundaries
    do NOT trigger a new event, so a note rings through a modulation zone
    change without re-attack.

    section_maps : sorted list of (tick_start: int, row_degree_map: dict)
                   where row_degree_map maps row(int) → degree_str.

    Returns list of {tick, duration, degree, scale}.
    """
    t, amp = _build_curve(control_points, n=100_000)
    if t.size == 0:
        return []

    total_ticks = _to_tick(dur_sec, bpm)
    if total_ticks <= 0:
        return []

    rows = (np.floor(amp * h_divisions).astype(int) + 1).clip(1, h_divisions)

    # ── Find all grid-row crossings ─────────────────────────────────────────
    # Each entry: (tick, row_number_after_crossing)
    crossings: list = [(0, int(rows[0]))]

    for i in range(len(rows) - 1):
        r0, r1 = int(rows[i]), int(rows[i + 1])
        if r0 == r1:
            continue

        a0, a1 = float(amp[i]), float(amp[i + 1])
        t0, t1 = float(t[i]),   float(t[i + 1])
        step   = 1 if r1 > r0 else -1

        for r_new in range(r0 + step, r1 + step, step):
            # Boundary amplitude for entering row r_new:
            #   going up (+1): floor(amp*h_div) crosses r_new-1, so amp = (r_new-1)/h_div
            #   going down (-1): floor(amp*h_div) crosses r_new, so amp = r_new/h_div
            # Equivalently: min(r_new, r_new - step) / h_divisions
            bdry = min(r_new, r_new - step) / h_divisions
            if abs(a1 - a0) > 1e-12:
                frac = (bdry - a0) / (a1 - a0)
            else:
                frac = 0.5
            frac    = max(0.0, min(1.0, frac))
            t_cross = t0 + frac * (t1 - t0)
            tick    = max(0, min(total_ticks - 1,
                                 round(t_cross * total_ticks)))
            crossings.append((tick, r_new))

    crossings.append((total_ticks, None))   # end sentinel

    # ── De-duplicate: multiple crossings at the same tick → keep last row ──
    deduped: list = []
    for entry in crossings:
        if deduped and deduped[-1][0] == entry[0]:
            deduped[-1] = entry   # later crossing wins
        else:
            deduped.append(list(entry))

    # ── Build raw events from crossing intervals ────────────────────────────
    raw_evs: list = []
    sm_idx  = 0

    for i in range(len(deduped) - 1):
        tick, row   = deduped[i]
        next_tick   = deduped[i + 1][0]
        if tick >= next_tick:
            continue
        while (sm_idx + 1 < len(section_maps)
               and section_maps[sm_idx + 1][0] <= tick):
            sm_idx += 1
        degree = section_maps[sm_idx][1].get(row, str(row))
        scale  = _active_region(scale_regions, tick / total_ticks, v_divisions)
        raw_evs.append(dict(tick=tick, duration=next_tick - tick,
                            degree=degree, scale=scale))

    # ── RLE by degree: merge consecutive same-degree events ─────────────────
    # This handles waves that bounce near a row boundary without re-attacking.
    # Key/mode zone boundaries are intentionally NOT a split condition here —
    # notes ring through modulation zones without re-attack.
    events: list = []
    for ev in raw_evs:
        if events and events[-1]['degree'] == ev['degree']:
            events[-1]['duration'] += ev['duration']
        else:
            events.append(dict(ev))

    return events


# ── Note-length limiter ────────────────────────────────────────────────────

def _split_long_notes(events: list, max_ticks: int) -> list:
    """Break any event longer than max_ticks into repeated same-pitch events.

    Each sub-event gets its own correct tick value so that section filtering
    in translate() assigns sub-events to the right key section.
    Re-attacks refresh the piano envelope before the sound fully decays.
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


# ── Section boundary helpers ───────────────────────────────────────────────

def _section_boundaries(voices: list, bpm: int,
                        dur_sec: float, total_ticks: int) -> list:
    """Return sorted list of (tick_start, (key, mode)) for each section.

    Key and mode are recorded directly at zone-edge detection time, not
    re-evaluated at rounded tick positions.  This avoids floating-point
    rounding errors where tick/total_ticks can fall just outside the new
    zone's col_start, returning the wrong key.

    Uses only voices[0]'s scale regions to determine the global key/mode
    (all voices in a section share the same key signature).
    """
    if not voices:
        return [(0, ('C', 'major'))]

    v       = voices[0]
    vd      = v.get('v_divisions', 16)
    regions = v.get('scale_regions', [])

    sections: dict = {}   # tick → (key, mode)

    candidates = sorted({r['col_start'] / vd for r in regions} |
                         {r['col_end']   / vd for r in regions})
    prev_km = None
    for norm_t in candidates:
        r  = _active_region(regions, norm_t, vd)
        km = (r.get('key', 'C'), r.get('mode', 'major')) if r else ('C', 'major')
        if km != prev_km:
            tick = max(0, min(total_ticks, _to_tick(norm_t * dur_sec, bpm)))
            if tick not in sections:
                sections[tick] = km   # store km AT DETECTION TIME
            prev_km = km

    if 0 not in sections:
        r = _active_region(regions, 0.0, vd)
        sections[0] = (r.get('key', 'C'), r.get('mode', 'major')) if r else ('C', 'major')

    return sorted(sections.items())   # [(tick, (key, mode)), ...]


def _key_at_tick(voices: list, tick: int, total_ticks: int) -> str:
    """Key string (e.g. 'F# major') for the section starting at *tick*.
    (Utility; not called by translate() — use _section_boundaries instead.)
    """
    key, mode = _key_mode_at_tick(voices, tick, total_ticks)
    return f"{key} {mode}"


def _key_mode_at_tick(voices: list, tick: int, total_ticks: int):
    """Return (key, mode) tuple for the section starting at *tick*.
    (Utility; not called by translate() — use _section_boundaries instead.)
    """
    if not voices:
        return 'C', 'major'
    v      = voices[0]
    vd     = v.get('v_divisions', 16)
    norm_t = tick / total_ticks if total_ticks > 0 else 0.0
    r      = _active_region(v.get('scale_regions', []), norm_t, vd)
    if r:
        return r.get('key', 'C'), r.get('mode', 'major')
    return 'C', 'major'


# ── Public API ─────────────────────────────────────────────────────────────

def translate(wave_path: str, score_path: Optional[str] = None) -> str:
    """Load *wave_path* (.wave.json) and produce a score .txt string."""
    data        = json.loads(Path(wave_path).read_text())
    bpm         = int(data.get('bpm', SCORE_BPM))
    dur_sec     = float(data['duration_seconds'])
    voices      = data['voices']
    max_ticks   = int(data.get('max_note_ticks', MAX_NOTE_TICKS))
    total_ticks = _to_tick(dur_sec, bpm)

    lines = []

    # ── Voice declarations ────────────────────────────────────────────────
    for v in voices:
        lines.append(f"voice: {v['name']}, @{v['sound_func']}, {v['octave_shift']}")
    lines.append('')

    # ── Section boundaries (key/mode stored alongside each tick) ─────────
    # Using the stored km avoids re-evaluation rounding errors.
    sec_starts    = _section_boundaries(voices, bpm, dur_sec, total_ticks)
    boundaries    = [t for t, _ in sec_starts] + [total_ticks]
    sections_info = [{'key': km[0], 'mode': km[1]} for _, km in sec_starts]

    # ── Build all voice events upfront ────────────────────────────────────
    all_events: list = []
    for v in voices:
        row_maps     = _build_row_degree_maps(sections_info, v['h_divisions'])
        section_maps = [(t, row_maps[i]) for i, (t, _) in enumerate(sec_starts)]
        evs = _build_events(v['control_points'],
                            h_divisions   = v['h_divisions'],
                            scale_regions = v.get('scale_regions', []),
                            v_divisions   = v['v_divisions'],
                            bpm           = bpm,
                            dur_sec       = dur_sec,
                            section_maps  = section_maps)
        evs = _split_long_notes(evs, max_ticks)
        # Re-evaluate each event's zone at its actual tick.
        # _split_long_notes sub-events copy the original scale, which may now
        # fall in a different zone after the re-attack tick advances past a boundary.
        v_vd      = v.get('v_divisions', 16)
        v_regions = v.get('scale_regions', [])
        for ev in evs:
            ev['scale'] = _active_region(v_regions, ev['tick'] / total_ticks, v_vd)
        all_events.append(evs)

    # ── Derive score section boundaries from voice 0's key transitions ────
    # A score section starts at the tick of the first event that plays in a
    # new key — NOT at the GUI zone boundary.  This means the section header
    # appears exactly where the music first sounds in the new key, with no
    # artificial clips or leading rests at zone edges.
    score_sections: list = []   # [(tick, key, mode)]
    if all_events:
        prev_km = None
        for ev in all_events[0]:
            scale = ev.get('scale')
            km = (scale.get('key', 'C'), scale.get('mode', 'major')) if scale else ('C', 'major')
            if km != prev_km:
                score_sections.append((ev['tick'], km[0], km[1]))
                prev_km = km
    if not score_sections:
        score_sections = [(0, 'C', 'major')]

    sec_ends = [s[0] for s in score_sections[1:]] + [total_ticks]

    # ── Emit one score section per key-transition ──────────────────────────
    for (sec_start, sec_key, sec_mode), sec_end in zip(score_sections, sec_ends):
        sec_ticks = sec_end - sec_start
        if sec_ticks <= 0:
            continue

        lines.append(f"qtr_note = {bpm}")
        lines.append(f"{sec_key} {sec_mode}")
        lines.append('')

        for evs in all_events:
            sec_evs = [e for e in evs if sec_start <= e['tick'] < sec_end]

            if not sec_evs:
                lines.append(f"r.{sec_ticks}")
            else:
                tokens: list = []
                ev_sum: int  = 0

                # For the primary voice, sec_evs[0]['tick'] == sec_start always
                # (sections are derived from its events).  Other voices may have
                # a small gap if their last note is still ringing; handle it.
                lead = sec_evs[0]['tick'] - sec_start
                if lead > 0:
                    tokens.append(f"r.{lead}")
                    ev_sum += lead

                for e in sec_evs:
                    tokens.append(f"{e['degree']}.{e['duration']}")
                    ev_sum += e['duration']

                if ev_sum < sec_ticks:
                    tokens.append(f"r.{sec_ticks - ev_sum}")

                lines.append(', '.join(tokens))

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
