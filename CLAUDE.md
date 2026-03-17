# CLAUDE.md — Microtonal Music Workspace

## What this repo is

A unified workspace for composing and synthesizing microtonal music. It contains:

- **Python tools** (repo root) — `wave_editor.py` / `wave_translator.py` for drawing spline-based melodic curves and converting them to score `.txt` files.
- **MATLAB library** (`matlab/`) — the `+microtonal` synthesis and notation engine that parses `.txt` scores and renders audio.

The `.txt` score file format is the contract between the Python side and the MATLAB side. Preserve it exactly.

---

## Repo layout

```
ScoreBuilder/
├── wave_editor.py          — Tkinter GUI for drawing spline waves with modulation zones
├── wave_translator.py      — converts .wave.json → .txt score file
├── main.py                 — launches wave_editor
├── wave*.json              — saved wave editor sessions
├── wave*.txt               — translated score files (output of wave_translator)
└── matlab/
    ├── +microtonal/        — MATLAB package: scales, audio, notation, rhythm
    │   ├── +scales/        — TET and JI scale generation
    │   ├── +audio/         — audio buffer mixing and playback
    │   ├── +notation/      — score text parsing and synthesis pipeline
    │   └── +rhythm/        — stochastic rhythm generators
    ├── sounds/             — 29 synthesis functions (@piano_sound, @crystal_bowl, etc.)
    ├── scores/             — .txt score files (input to MATLAB renderer)
    ├── audio_files/        — rendered .wav output files
    ├── composition_scripts/ — standalone algorithmic composition scripts
    ├── API_REFERENCE.md    — full MATLAB API docs
    └── README.md           — MATLAB library description
```

---

## Running MATLAB

**Always `cd` into `matlab/` before running MATLAB scripts**, so the `+microtonal` package is on the path:

```matlab
% from matlab/
buf = microtonal.notation.notation_to_audio('scores/MyScore.txt');
microtonal.audio.play('my_score', buf);
```

Score files live in `matlab/scores/`. Audio output goes to `matlab/audio_files/`.

Key MATLAB files to read when debugging audio or score-parsing issues:

- `matlab/+microtonal/+notation/parse_notation.m`   — score text → section/voice structs
- `matlab/+microtonal/+notation/notation_to_audio.m` — section structs → audio pipeline
- `matlab/+microtonal/+audio/build_audio_buffer.m`  — note arrays → waveform buffer
- `matlab/sounds/piano_sound.m`                     — main synthesis function

---

## Score File Format (the full protocol)

### File structure

```
Title
(Year)
transcribed/composed by Author

voice: <name>, @<sound_func>, <octave_shift>
voice: <name>, @<sound_func>, <octave_shift>
[tuning: <name>]   ← optional

qtr_note = <BPM>
<Key> <major|minor>

<voice 1 line>  |  <more measures>
<voice 2 line>  |  <more measures>
...

<voice 1 continued>  |  ...
<voice 2 continued>  |  ...

qtr_note = <new BPM>
<new Key> <major|minor>

...
```

### Voice declarations (required, before first `qtr_note`)

```
voice: soprano, @piano_sound, 0
voice: alto,    @piano_sound, -1
```

- `@piano_sound` references a MATLAB function in `matlab/sounds/`
- `octave_shift`: integer, shifts all notes in this voice up/down by N octaves
- Available sound functions: see `matlab/API_REFERENCE.md` for the full table

### Tuning (optional, in preamble)

```
tuning: tet              ← default, 12-TET
tuning: ji               ← auto-maps major→major_5limit, minor→minor_5limit
tuning: pythagorean_major
tuning: major_7limit
```

### Tempo and key sections

```
qtr_note = 120
C major

qtr_note = 72
A minor
```

Tempo/key can change mid-score. Each change starts a new section.

### Key signature variants

```
C major
F# minor
Ab major
Ab [1,3,5,6,8,10,12]       ← custom 1-indexed TET semitone positions
Ab [1/1,9/8,5/4,4/3,3/2,5/3,15/8]   ← explicit JI ratios
```

### Note syntax: `<degree>[s|f].<duration>`

- `degree`: integer scale degree. `1` = tonic, `2` = supertonic, etc.
  - Negative values go below tonic: `-1` = leading tone below, `-2` = 7th below, etc.
  - `0` = leading tone (one step below tonic in scale)
- `s` / `f`: optional sharp / flat accidental (e.g. `4s.2`, `7f.1`)
- `duration`: in **eighth notes** (1=eighth, 2=quarter, 3=dotted quarter, 4=half, 6=dotted half, 8=whole); any positive integer is valid
- `-` at end: sustain to measure bar (e.g. `5.1-`) — **avoid in wave_translator output** (stretches to full section with TICKS_PER_MEASURE=99999)

Examples:
```
3.2      = 3rd degree, quarter note (2 eighths)
7f.1     = flat 7th, eighth note
-2.4     = 2nd degree below tonic, half note
1.1-     = tonic, eighth note, sustained until | barline
```

### Rests: `r.<duration>`

```
r.2      = quarter rest
r.4      = half rest
r.8      = whole rest
```

### Measure and block structure

- Notes within a measure separated by `,`
- Measures separated by `|`
- Each line = one voice
- Blank line = block separator (voices still continue; blank lines just group notation visually)
- `#` = comment line (ignored by parser, acts as block separator)
- All voices within a measure **must sum to the same duration**

### Example score snippet

```
voice: v1, @piano_sound, 0
voice: v2, @piano_sound, -1

qtr_note = 60
C major

# measure 1 and 2
8.2, 7.1, 8.1, 6.3, 5.1, 4.2, | 3.2, 2.1, 3.4
3.2, 2.1, 3.4, 2.3,            | -1.3, -1.1, -1.4
```

---

## MATLAB Architecture

### Package structure (`matlab/+microtonal/`)

- **`+scales/`** — `tet_scales()` for any N-TET, `get_mode(tet, mode_name)` for predefined steps (12/19/31/53-TET), `ratio_scale()` for JI, `get_ji_scales()`, `note_to_freq()`, `cents()`, `compare_tunings()`
- **`+audio/`** — `build_audio_buffer()` mixes note arrays into a waveform with 20ms fade envelopes; `play()` saves WAV and plays back
- **`+notation/`** — `parse_notation()` reads `.txt` → section/voice structs; `notation_to_audio()` is the end-to-end pipeline; `format_score()` validates/aligns score files; `generate_notation()` generates random compositions
- **`+rhythm/`** — `stochastic_rhythm()` generates onset times/durations via: uniform, poisson, euclidean, fibonacci, accelerando, lcm

### Sound functions (`matlab/sounds/`)

Signature: `sound = func_name(freq, fs, duration)`. Referenced in scores as `@func_name`.

Key sounds: `@piano_sound`, `@crystal_bowl`, `@rich_bell`, `@ethereal_pad`, `@soft_kalimba`, `@tubular_bell`, `@wind_chime`, `@exotic_gamelan`, `@vibraphone`, `@shakuhachi`, `@breathy_flute`, and 18 more. Full table in `matlab/API_REFERENCE.md`.

### Adding a new TET system

Create `matlab/+microtonal/+scales/get_<N>tet_modes.m` returning a struct with mode names as fields and step arrays as values. `get_mode()` will auto-discover it.

---

## Python Tools

### wave_editor.py

Tkinter GUI for drawing a spline-based melodic curve over a time-pitched grid. Features:
- Modulation zones: assign different key/mode/scale regions to time ranges
- Multiple voices with configurable sound function and octave shift
- Saves state to `.wave.json`

### wave_translator.py

Converts `.wave.json` → `.txt` score file using crossing-based pitch detection.

Key constants (tune at top of file):
- `SCORE_BPM = 9600` — eighth-note tick rate (≈3 ms/tick); high value = fine time resolution
- `TICKS_PER_MEASURE = 99999` — effectively one measure per section (no barlines)
- `MAX_NOTE_TICKS = 250` — max duration before re-attack (≈0.78s at BPM=9600)

CLI usage:
```bash
python wave_translator.py input.wave.json matlab/scores/output.txt
```

Or from Python:
```python
from wave_translator import translate
translate('wave10.json', 'matlab/scores/wave10.txt')
```

---

## Reference scores

In `matlab/scores/`:

- `Xenakis_SixChansons.txt` — 6-voice piano, C# major; demonstrates voice layout and negative degrees
- `Stravinsky_sonataDuet.txt` — 4-voice F major; demonstrates `s`/`f` accidentals, sustain `-`, aligned columns
- `sonata_in_24.txt` — WIP 4-voice sonata cycling through 24 keys; good example of comments/blank lines structuring a long score

---

## Python work to build

The Python side is nascent. Potential directions:

- Classes: `Score`, `Voice`, `Section`, `Measure`, `Note`, `Rest`
- Builder/DSL for constructing scores programmatically
- Algorithmic composition tools (chord progressions, voice leading, counterpoint, serialist generators)
- Validator checking measure-length consistency before writing `.txt`
- Output: write `.txt` to `matlab/scores/`, ready for MATLAB renderer
