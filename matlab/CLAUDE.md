# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Algorithmic microtonal music library for MATLAB. Provides scale generation (TET and just intonation), audio synthesis, a text notation system for composing multi-voice pieces, and algorithmic rhythm generation.

## Running Code

This is a MATLAB project. All code runs in the MATLAB environment. Add the project root to MATLAB's path so the `+microtonal` package is accessible. There is no build system, test suite, or linter.

To play a composition from a text score:
```matlab
buf = microtonal.notation.notation_to_audio('scores/MyScore.txt');
microtonal.audio.play('my_score', buf);
```

Audio files are saved to `audio_files/` as WAV at 44100 Hz sample rate.

## Architecture

### Package Structure (`+microtonal/`)

MATLAB's `+` package convention is used throughout. All library code lives under `+microtonal/` with subpackages:

- **`+scales/`** — Scale and tuning systems. `tet_scales()` generates frequencies for any N-TET system. `get_mode(tet, mode_name)` retrieves predefined mode step patterns for 12, 19, 31, and 53-TET (each in its own `get_<N>tet_modes.m`). `ratio_scale()` handles just intonation. `note_to_freq()` converts note names to Hz.
- **`+audio/`** — `build_audio_buffer()` mixes note arrays into a single audio buffer with fade envelopes. `play()` saves to WAV and plays back.
- **`+notation/`** — Text notation system. `parse_notation()` reads `.txt` score files into structured section/voice data. `notation_to_audio()` is the end-to-end pipeline (parse → scale lookup → synthesis). `format_score()` validates and aligns score files.
- **`+rhythm/`** — `stochastic_rhythm()` generates rhythmic patterns via multiple algorithms (uniform, poisson, euclidean, fibonacci, accelerando, lcm).

### Sound Functions (`sounds/`)

Each `.m` file in `sounds/` is a standalone synthesis function with signature `sound = func_name(freq, fs, duration)`. These are referenced by name in score files (e.g., `@piano_sound`) and resolved via `str2func()`.

### Composition Scripts (`composition_scripts/`)

Standalone scripts that use the library to generate music. These are not part of the library API.

### Text Notation System (`scores/`)

Scores are `.txt` files parsed by `microtonal.notation.parse_notation()`. Key format details:

- Voice declarations: `voice: <name>, @<sound_func>, <octave_shift>`
- Tempo/key sections: `qtr_note = <BPM>` followed by `<Key> <major|minor>`
- Notes: `<degree>[s|f].<duration_in_eighths>` (e.g., `3.2` = 3rd degree for a quarter note, `7f.1` = flat 7th for an eighth)
- Rests: `r.<duration>`, measures separated by `|`, voices on parallel lines
- Blank lines separate notation blocks; `#` for comments

### Adding a New TET System

Create `+microtonal/+scales/get_<N>tet_modes.m` returning a struct with mode names as fields and step arrays as values. `get_mode()` will auto-discover it.
