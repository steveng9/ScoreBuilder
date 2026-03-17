# Microtonal Library — API Reference

## Scales (`microtonal.scales.*`)

### `tet_scales(root_freq, tet, mode_steps, num_octaves)`

Generate frequencies for any N-TET (N-Tone Equal Temperament) scale.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `root_freq` | double | — | Root frequency in Hz |
| `tet` | int | — | Equal divisions per octave (12, 19, 31, 53, any) |
| `mode_steps` | array | — | Scale degrees within the TET (e.g., `[0,2,4,5,7,9,11]`) |
| `num_octaves` | int | 1 | Number of octaves to generate |

**Returns:** `frequencies` — array of Hz values spanning the requested octaves.

```matlab
% 12-TET major scale, 2 octaves from middle C
freqs = microtonal.scales.tet_scales(261.63, 12, [0,2,4,5,7,9,11], 2);

% Full chromatic scale
chromatic = microtonal.scales.tet_scales(440, 12, 0:11, 1);

% 19-TET major scale
steps_19 = microtonal.scales.get_mode(19, 'major');
freqs_19 = microtonal.scales.tet_scales(261.63, 19, steps_19, 1);

% 31-TET quarter-tone scale
steps_31 = microtonal.scales.get_mode(31, 'quarter_tone');
freqs_31 = microtonal.scales.tet_scales(261.63, 31, steps_31, 3);
```

---

### `get_mode(tet, mode_name)`

Retrieve predefined mode step patterns for a given TET system.

| Parameter | Type | Description |
|-----------|------|-------------|
| `tet` | int | TET system (12, 19, 31, 53, or custom) |
| `mode_name` | string | Mode name (see tables below) |

**Returns:** `mode_steps` — array of step indices within the TET.

For custom TET values, create `+microtonal/+scales/get_<N>tet_modes.m` and it will be auto-discovered.

#### 12-TET Modes

| Mode | Steps |
|------|-------|
| `'major'` | `[0, 2, 4, 5, 7, 9, 11]` |
| `'minor'` | `[0, 2, 3, 5, 7, 8, 10]` |
| `'harmonic_minor'` | `[0, 2, 3, 5, 7, 8, 11]` |
| `'melodic_minor'` | `[0, 2, 3, 5, 7, 9, 11]` |
| `'dorian'` | `[0, 2, 3, 5, 7, 9, 10]` |
| `'phrygian'` | `[0, 1, 3, 5, 7, 8, 10]` |
| `'lydian'` | `[0, 2, 4, 6, 7, 9, 11]` |
| `'mixolydian'` | `[0, 2, 4, 5, 7, 9, 10]` |
| `'locrian'` | `[0, 1, 3, 5, 6, 8, 10]` |
| `'pentatonic_major'` | `[0, 2, 4, 7, 9]` |
| `'pentatonic_minor'` | `[0, 3, 5, 7, 10]` |
| `'chromatic'` | `[0:11]` |
| `'whole_tone'` | `[0, 2, 4, 6, 8, 10]` |
| `'blues'` | `[0, 3, 5, 6, 7, 10]` |
| `'diminished'` | `[0, 2, 3, 5, 6, 8, 9, 11]` |
| `'bebop_major'` | `[0, 2, 4, 5, 7, 8, 9, 11]` |
| `'bebop_dominant'` | `[0, 2, 4, 5, 7, 9, 10, 11]` |

#### 19-TET Modes

`'major'`, `'minor'`, `'exotic1'`, `'chromatic'`

#### 31-TET Modes

`'major'`, `'minor'`, `'harmonic'`, `'quarter_tone'`, `'chromatic'`, `'custom1'`

#### 53-TET Modes

`'pythagorean_major'`, `'just_major'`, `'chromatic'`

```matlab
steps = microtonal.scales.get_mode(12, 'dorian');
steps = microtonal.scales.get_mode(31, 'harmonic');
steps = microtonal.scales.get_mode(53, 'just_major');
```

---

### `ratio_scale(root_freq, ratios, num_octaves)`

Generate frequencies from frequency ratios (for just intonation and custom tunings).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `root_freq` | double | — | Root frequency in Hz |
| `ratios` | array | — | Frequency ratios relative to root (each >= 1 and < 2) |
| `num_octaves` | int | 1 | Number of octaves to generate |

**Returns:** `frequencies` — array of Hz values.

```matlab
% 5-limit JI major scale
ji_major = microtonal.scales.ratio_scale(261.63, [1, 9/8, 5/4, 4/3, 3/2, 5/3, 15/8], 2);

% Harmonic series from 110 Hz
harmonics = microtonal.scales.ratio_scale(110, (1:8)/1, 1);

% Use predefined JI scales
ji = microtonal.scales.get_ji_scales();
freqs = microtonal.scales.ratio_scale(261.63, ji.minor_7limit, 2);
```

---

### `get_ji_scales()`

Returns a struct of predefined just intonation ratio arrays for use with `ratio_scale()`.

**Returns:** `scales` — struct with the following fields:

#### 5-Limit (primes 2, 3, 5)

| Scale | Ratios |
|-------|--------|
| `major_5limit` | `[1, 9/8, 5/4, 4/3, 3/2, 5/3, 15/8]` |
| `minor_5limit` | `[1, 9/8, 6/5, 4/3, 3/2, 8/5, 9/5]` |

#### Pythagorean (primes 2, 3 only)

| Scale | Ratios |
|-------|--------|
| `pythagorean_major` | `[1, 9/8, 81/64, 4/3, 3/2, 27/16, 243/128]` |
| `pythagorean_minor` | `[1, 9/8, 32/27, 4/3, 3/2, 128/81, 16/9]` |

#### 7-Limit (primes 2, 3, 5, 7)

| Scale | Ratios |
|-------|--------|
| `major_7limit` | `[1, 9/8, 5/4, 4/3, 3/2, 5/3, 7/4]` |
| `minor_7limit` | `[1, 9/8, 7/6, 4/3, 3/2, 8/5, 7/4]` |

#### 11-Limit and Beyond

| Scale | Ratios |
|-------|--------|
| `scale_11limit` | `[1, 9/8, 5/4, 11/8, 3/2, 13/8, 7/4]` |

#### Harmonic Series Segments

| Scale | Description |
|-------|-------------|
| `harmonic_8_16` | Partials 8–16 (one octave of harmonic series) |
| `harmonic_4_8` | Partials 4–8 |
| `harmonic_6_12` | Partials 6–12 |

#### Historical / World

| Scale | Description |
|-------|-------------|
| `arabic_alfarabi` | Al-Farabi's Arabic scale (10th century) |
| `slendro_approx` | Approximation of Javanese Slendro (pentatonic) |

#### Experimental

| Scale | Description |
|-------|-------------|
| `undertone_8_16` | Subharmonic series (inverted intervals) |
| `drone` | Pure intervals only — `[1, 5/4, 4/3, 3/2, 5/3]` |

```matlab
ji = microtonal.scales.get_ji_scales();
freqs = microtonal.scales.ratio_scale(261.63, ji.arabic_alfarabi, 2);
freqs = microtonal.scales.ratio_scale(130, ji.harmonic_8_16, 1);
```

---

### `note_to_freq(note_name)`

Convert a standard note name to frequency in Hz (12-TET, A4 = 440).

| Parameter | Type | Description |
|-----------|------|-------------|
| `note_name` | string | Note name: `<Letter>[#|b]<Octave>` (e.g., `'C4'`, `'F#3'`, `'Bb2'`) |

**Returns:** `freq` — frequency in Hz.

```matlab
microtonal.scales.note_to_freq('A4')    % => 440.00
microtonal.scales.note_to_freq('C4')    % => 261.63
microtonal.scales.note_to_freq('F#3')   % => 185.00
```

---

### `cents(ratio)`

Convert a frequency ratio to cents (1200 cents = 1 octave).

| Parameter | Type | Description |
|-----------|------|-------------|
| `ratio` | double/array | Frequency ratio(s) |

**Returns:** `c` — interval size in cents.

```matlab
microtonal.scales.cents(2)            % => 1200   (octave)
microtonal.scales.cents(3/2)          % => 701.96 (perfect fifth)
microtonal.scales.cents(5/4)          % => 386.31 (just major third)
microtonal.scales.cents(2^(7/12))     % => 700    (12-TET fifth)
```

---

### `compare_tunings(ji_ratios, tet)`

Print a comparison table of JI ratios vs. their nearest TET approximations.

| Parameter | Type | Description |
|-----------|------|-------------|
| `ji_ratios` | array | Just intonation frequency ratios |
| `tet` | int | TET system to compare against |

**Output:** Console table showing ratio, JI cents, nearest TET step, TET cents, and deviation.

```matlab
ji = microtonal.scales.get_ji_scales();
microtonal.scales.compare_tunings(ji.major_5limit, 12);   % How well does 12-TET approximate JI?
microtonal.scales.compare_tunings(ji.major_5limit, 31);   % 31-TET is much closer
microtonal.scales.compare_tunings(ji.major_5limit, 53);   % 53-TET is extremely close
```

---

## Audio (`microtonal.audio.*`)

### `build_audio_buffer(notes, start_times, durations, sound_func, fs)`

Mix an array of notes into a single audio buffer.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `notes` | array | — | Frequencies in Hz |
| `start_times` | array | — | Note onset times in seconds |
| `durations` | array | — | Note durations in seconds |
| `sound_func` | function handle | — | Synthesis function (e.g., `@piano_sound`) |
| `fs` | int | 44100 | Sample rate in Hz |

**Returns:** `audio_buffer` — normalized mono audio signal (peak 0.9).

Each note is synthesized via `sound_func(freq, fs, duration)`, given a 20ms fade envelope, and mixed into the buffer at the specified time.

```matlab
notes = [261.63, 329.63, 392.00, 523.25];
times = [0, 0.5, 1.0, 1.5];
durs  = [0.4, 0.4, 0.4, 0.8];
buf = microtonal.audio.build_audio_buffer(notes, times, durs, @piano_sound);
```

---

### `play(name, audio_buffer, fs)`

Save audio to WAV and play it.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | — | Filename (no extension) — saved to `audio_files/<name>.wav` |
| `audio_buffer` | array | — | Audio data |
| `fs` | int | 44100 | Sample rate |

```matlab
microtonal.audio.play('my_piece', buf);
microtonal.audio.play('my_piece', buf, 48000);  % custom sample rate
```

---

## Rhythm (`microtonal.rhythm.*`)

### `stochastic_rhythm(n_notes, tempo, method)`

Generate rhythmic patterns using various algorithmic methods.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_notes` | int | — | Number of notes |
| `tempo` | double | 120 | Quarter note BPM |
| `method` | string | `'uniform'` | Algorithm (see below) |

**Returns:** `[start_times, durations]` — onset times and durations in seconds.

#### Methods

| Method | Description |
|--------|-------------|
| `'uniform'` | Evenly spaced at beat intervals |
| `'poisson'` | Random exponentially-distributed spacing (naturalistic) |
| `'euclidean'` | Maximally even distribution via Bjorklund's algorithm |
| `'fibonacci'` | Durations follow Fibonacci sequence ratios |
| `'accelerando'` | Gradually accelerating (2x beat down to 0.25x beat) |
| `'lcm'` | Polyrhythmic — superimposed streams at rates 1, 1.5, 2, 3 |

```matlab
[t, d] = microtonal.rhythm.stochastic_rhythm(16, 120, 'uniform');
[t, d] = microtonal.rhythm.stochastic_rhythm(20, 90, 'poisson');
[t, d] = microtonal.rhythm.stochastic_rhythm(12, 100, 'euclidean');
[t, d] = microtonal.rhythm.stochastic_rhythm(10, 80, 'fibonacci');
[t, d] = microtonal.rhythm.stochastic_rhythm(24, 120, 'accelerando');
[t, d] = microtonal.rhythm.stochastic_rhythm(16, 100, 'lcm');
```

---

## Notation (`microtonal.notation.*`)

### `notation_to_audio(filename, tet)`

End-to-end pipeline: parse a text score and synthesize audio.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filename` | string | — | Path to `.txt` score file |
| `tet` | int | 12 | TET system to use |

**Returns:** `audio_buffer` — synthesized audio ready for `play()`.

```matlab
buf = microtonal.notation.notation_to_audio('scores/Xenakis_SixChansons.txt');
buf = microtonal.notation.notation_to_audio('scores/MyPiece.txt', 31);
microtonal.audio.play('my_piece', buf);
```

---

### `parse_notation(filename)`

Parse a `.txt` score file into structured data.

| Parameter | Type | Description |
|-----------|------|-------------|
| `filename` | string | Path to `.txt` score file |

**Returns:**
- `sections` — struct array (one per tempo/key section):
  - `.tonic` — tonic note with octave (e.g., `'C#1'`)
  - `.mode` — mode name (e.g., `'major'`)
  - `.eighth_note_duration` — seconds per eighth note
  - `.voices` — cell array of voices → cell array of measures → Nx3 matrix `[degree, duration, accidental]`
- `voice_info` — struct array:
  - `.name` — voice name
  - `.sound_func` — synthesis function name
  - `.octave_shift` — octave offset

---

### `format_score(filename)`

Validate and reformat a score file with aligned measure bars.

```matlab
microtonal.notation.format_score('scores/MyPiece.txt');
```

---

### `generate_notation(filename, num_voices, num_measures, key, tempo, standard_measure_length)`

Generate a random composition in text notation format.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filename` | string | — | Output `.txt` file path |
| `num_voices` | int | 4 | Number of voices |
| `num_measures` | int | 12 | Number of measures |
| `key` | string | `'C major'` | Key signature |
| `tempo` | int | 60 | Quarter note BPM |
| `standard_measure_length` | double | 6 | Eighth notes per measure |

```matlab
microtonal.notation.generate_notation('scores/random_piece.txt');
microtonal.notation.generate_notation('scores/fast.txt', 3, 8, 'D minor', 120, 8);
```

---

## Text Score Format

```
Title
(Year)
transcribed by Author

voice: <name>, @<sound_func>, <octave_shift>
voice: <name>, @<sound_func>, <octave_shift>

qtr_note = <BPM>
<Key> <major|minor>

<voice 1 notes> | <voice 1 notes> | ...
<voice 2 notes> | <voice 2 notes> | ...
```

**Notes:** `<degree>[s|f].<duration_in_eighths>` — e.g., `3.2` (3rd degree, quarter note), `7f.1` (flat 7th, eighth note)

**Rests:** `r.<duration>` — e.g., `r.4` (half-note rest)

**Degrees:** 1 = tonic, 2 = supertonic, ...; 0 = leading tone below tonic; negative values go lower

**Accidentals:** `s` = sharp (+1 chromatic step), `f` = flat (-1 chromatic step)

**Measures:** separated by `|`; blank lines separate notation blocks; `#` starts a comment

---

## Sound Functions (`sounds/`)

All share the signature: `sound = func_name(freq, fs, duration)`

| Function | Character |
|----------|-----------|
| `@piano_sound` | Percussive piano with slow decay |
| `@crystal_bowl` | Ethereal, haunting; slow decay with detuned harmonics |
| `@crystal_bowl_with_pop` | Crystal bowl with softer mallet impact |
| `@rich_bell` | Complex bell with many harmonics |
| `@ethereal_pad` | Very slow attack; chorus with detuned oscillators |
| `@soft_kalimba` | Thumb piano; gentle with strong fundamental |
| `@tubular_bell` | Bright tubular bells |
| `@magic_shimmer` | Vibrato/tremolo shimmer |
| `@wind_chime` | Metallic with inharmonic ratios |
| `@water_drop` | Resonant liquid sound |
| `@exotic_steel_drum` | Caribbean bright with metallic shimmer |
| `@layered_chorus` | Multiple detuned voices, thick and pure |
| `@pure_tubey` | Hollow/tubular with minimal overtones |
| `@evolving_timbre` | Harmonics change over time |
| `@deep_resonant` | Warm, rich fundamentals with subtle vibrato |
| `@bright_crystalline` | High harmonics, sparkle |
| `@kind_music_box` | Delicate, nostalgic |
| `@tonal_percussion` | Marimba/xylophone; clear fundamental, fast decay |
| `@soft_marimba` | Warm, woody tone |
| `@inharmonic_complex` | Non-integer harmonic ratios; metallic beating |
| `@vibraphone` | Metallic with inharmonic overtones and vibrato |
| `@shakuhachi` | Japanese flute with breath noise |
| `@whispy_whistle` | Pure tone with slight vibrato and air noise |
| `@breathy_flute` | Flute/pan flute with breath noise |
| `@ocarina_sound` | Hollow woody tone with texture noise |
| `@exotic_gamelan` | Inharmonic metallic with beating shimmer |

---

## Cookbook

### Play a JI drone

```matlab
ji = microtonal.scales.get_ji_scales();
freqs = microtonal.scales.ratio_scale(130.81, ji.drone, 1);
times = zeros(1, length(freqs));       % all start at once
durs  = ones(1, length(freqs)) * 8;    % 8-second drone
buf = microtonal.audio.build_audio_buffer(freqs, times, durs, @deep_resonant);
microtonal.audio.play('ji_drone', buf);
```

### Algorithmic composition with Euclidean rhythm

```matlab
steps = microtonal.scales.get_mode(31, 'harmonic');
freqs = microtonal.scales.tet_scales(261.63, 31, steps, 2);
[t, d] = microtonal.rhythm.stochastic_rhythm(16, 100, 'euclidean');
notes = freqs(randi(length(freqs), 1, 16));
buf = microtonal.audio.build_audio_buffer(notes, t, d, @exotic_gamelan);
microtonal.audio.play('euclidean_31tet', buf);
```

### Compare how well different TET systems approximate JI

```matlab
ji = microtonal.scales.get_ji_scales();
microtonal.scales.compare_tunings(ji.major_5limit, 12);
microtonal.scales.compare_tunings(ji.major_5limit, 19);
microtonal.scales.compare_tunings(ji.major_5limit, 31);
microtonal.scales.compare_tunings(ji.major_5limit, 53);
```

### Build a custom TET system

Create `+microtonal/+scales/get_17tet_modes.m`:

```matlab
function modes = get_17tet_modes()
    modes.major = [0, 3, 6, 7, 10, 13, 16];
    modes.minor = [0, 3, 4, 7, 10, 11, 14];
end
```

Then use it:

```matlab
steps = microtonal.scales.get_mode(17, 'major');
freqs = microtonal.scales.tet_scales(261.63, 17, steps, 2);
```
