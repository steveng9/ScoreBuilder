% QUICK_REFERENCE Usage examples for all microtonal library functions
% Run sections individually (highlight + Ctrl+Enter) to test each function.
%
% Package structure:
%   microtonal.scales.*    — tet_scales, ratio_scale, get_mode, get_ji_scales,
%                            cents, compare_tunings, note_to_freq
%   microtonal.audio.*     — build_audio_buffer, play
%   microtonal.notation.*  — parse_notation, notation_to_audio, format_score
%   microtonal.rhythm.*    — stochastic_rhythm

%% ========================================================================
%  SCALES: microtonal.scales.*
%  ========================================================================

%% tet_scales — Generate frequencies for a TET scale
%  tet_scales(root_freq, tet, mode_steps, num_octaves)

major_steps = microtonal.scales.get_mode(12, 'major');
freqs = microtonal.scales.tet_scales(261.63, 12, major_steps, 2);
disp('12-TET major scale (2 octaves):'); disp(freqs);

% Chromatic scale (all 12 notes)
chromatic = microtonal.scales.tet_scales(440, 12, 0:11, 1);
disp('12-TET chromatic:'); disp(chromatic);

% 19-TET major scale
steps_19 = microtonal.scales.get_mode(19, 'major');
freqs_19 = microtonal.scales.tet_scales(261.63, 19, steps_19, 1);
disp('19-TET major:'); disp(freqs_19);

%% get_mode — Get scale degree steps for a given TET and mode name
%  Supported TET: 12, 19, 31, 53

disp(microtonal.scales.get_mode(12, 'major'));
disp(microtonal.scales.get_mode(12, 'dorian'));
disp(microtonal.scales.get_mode(31, 'major'));

%% cents — Convert a frequency ratio to cents

disp(microtonal.scales.cents(2));       % 1200 (octave)
disp(microtonal.scales.cents(3/2));     % 701.96 (perfect fifth)
disp(microtonal.scales.cents(5/4));     % 386.31 (just major third)
disp(microtonal.scales.cents(2^(7/12))); % 700 (12-TET fifth)

%% ========================================================================
%  JUST INTONATION: microtonal.scales.*
%  ========================================================================

%% get_ji_scales — Get a struct of named JI ratio arrays

ji = microtonal.scales.get_ji_scales();
disp('Available JI scales:'); disp(fieldnames(ji));

%% ratio_scale — Generate frequencies from JI ratios

ji_major = microtonal.scales.ratio_scale(261.63, ji.major_5limit, 2);
disp('5-limit JI major (2 octaves):'); disp(ji_major);

harmonics = microtonal.scales.ratio_scale(110, (1:8)/1, 1);
disp('Harmonic series from 110 Hz:'); disp(harmonics);

%% compare_tunings — Compare JI ratios against a TET system

ji = microtonal.scales.get_ji_scales();
microtonal.scales.compare_tunings(ji.major_5limit, 12);
microtonal.scales.compare_tunings(ji.major_5limit, 31);

%% ========================================================================
%  NOTATION: microtonal.notation.*
%  ========================================================================

%% parse_notation — Parse a .txt score into sections
%  [sections, voice_info] = parse_notation(filename)

% [sections, voice_info] = microtonal.notation.parse_notation('scores/Xenakis_SixChansons.txt');

%% notation_to_audio — Render a .txt score to audio
%  audio_buffer = notation_to_audio(filename, tet)

% buf = microtonal.notation.notation_to_audio('scores/Xenakis_SixChansons.txt');
% microtonal.audio.play('test_notation', buf);

%% format_score — Validate and align a score file

% microtonal.notation.format_score('scores/Xenakis_SixChansons.txt');

%% ========================================================================
%  RHYTHM: microtonal.rhythm.*
%  ========================================================================

%% stochastic_rhythm — Generate rhythmic patterns
%  Methods: 'uniform', 'poisson', 'euclidean', 'fibonacci', 'accelerando', 'lcm'

[t, d] = microtonal.rhythm.stochastic_rhythm(16, 120, 'uniform');
disp('Uniform rhythm start times:'); disp(t);

[t, d] = microtonal.rhythm.stochastic_rhythm(12, 100, 'euclidean');
disp('Euclidean rhythm start times:'); disp(t);

[t, d] = microtonal.rhythm.stochastic_rhythm(10, 90, 'fibonacci');
disp('Fibonacci rhythm durations:'); disp(d);

%% ========================================================================
%  AUDIO: microtonal.audio.*
%  ========================================================================

%% play — Save and play an audio buffer
%  play(name, audio_buffer, fs)

% fs = 44100; t = 0:1/fs:1;
% buf = sin(2*pi*440*t);
% microtonal.audio.play('test_sine', buf, fs);
