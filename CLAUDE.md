# CLAUDE.md — ScoreBuilder

## What this repo is

This is a **Python** project for composing and generating `.txt` score files in the microtonal text notation protocol. Those `.txt` files are then executed in a separate MATLAB repository (`~/Documents/music/MATLAB/`) which synthesizes and plays the audio.

This repo does **not** do any audio synthesis. Its sole job is to produce well-formed `.txt` score files.

---

## The companion MATLAB repo

Location: `/Users/stevengolob/Documents/music/MATLAB/`

To render a score from MATLAB:
```matlab
buf = microtonal.notation.notation_to_audio('scores/MyScore.txt');
microtonal.audio.play('my_score', buf);
```

Score files go in `~/Documents/music/MATLAB/scores/`.

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

- `@piano_sound` references a MATLAB function in `~/Documents/music/MATLAB/sounds/`
- `octave_shift`: integer, shifts all notes in this voice up/down by N octaves
- Available sound functions: `piano_sound` (most commonly used), others in `sounds/`

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
- `duration`: in **eighth notes** (1=eighth, 2=quarter, 3=dotted quarter, 4=half, 6=dotted half, 8=whole)
- `-` at end: sustain to measure bar (e.g. `5.1-`)

Examples:
```
3.2      = 3rd degree, quarter note (2 eighths)
7f.1     = flat 7th, eighth note
-2.4     = 2nd degree below tonic, half note
1.1-     = tonic, eighth note duration, sustained until | barline
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

## What to build in Python

The goal is a Python library/toolkit that **generates these `.txt` score files** algorithmically or interactively. Ideas (to be decided with the user):

- Classes representing `Score`, `Voice`, `Section`, `Measure`, `Note`, `Rest`
- A builder/DSL for constructing scores programmatically
- Algorithmic composition tools (chord progressions, voice leading, counterpoint rules, serialist generators, etc.)
- A validator that checks measure-length consistency before writing to file
- Output: write a `.txt` file in the exact format above, ready to drop into `~/Documents/music/MATLAB/scores/`

The score format is the **contract** between this Python project and the MATLAB renderer. Preserve it exactly.

---

## Example scores (for reference)

Three reference scores live in the MATLAB repo at `~/Documents/music/MATLAB/scores/`:

- `Xenakis_SixChansons.txt` — 6-voice piano piece, C# major, demonstrates voice layout and negative degrees
- `Stravinsky_sonataDuet.txt` — 4-voice, F major, demonstrates `s`/`f` accidentals, sustain `-`, and `|` measure barlines with aligned columns
- `sonata_in_24.txt` — work-in-progress 4-voice sonata cycling through 24 keys; mostly commented-out plan sections — good example of how comments and blank lines structure a long score

---

## Where to pick up

The user has not yet written any Python code in this repo. The next step is to **design the Python architecture** for score generation and start implementing it. Start by asking the user what kind of composition workflow they have in mind, then build accordingly.
