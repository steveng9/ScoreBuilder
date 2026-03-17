addpath("sounds/");

fs = 44100;
note_dur    = 0.45;
note_gap    = 0.04;
pair_gap    = 0.5;
section_gap = 0.8;
interval_dur = 0.8;   % longer notes for isolated intervals
triad_dur    = 2.0;

root = microtonal.scales.note_to_freq('C4');
sound_func = @crystal_bowl_with_pop;

ji = microtonal.scales.get_ji_scales();

%% Build scales
maj12 = microtonal.scales.tet_scales(root, 12, ...
    microtonal.scales.get_mode(12,'major'), 1);
min12 = microtonal.scales.tet_scales(root, 12, ...
    microtonal.scales.get_mode(12,'minor'), 1);
chr12 = microtonal.scales.tet_scales(root, 12, ...
    microtonal.scales.get_mode(12,'chromatic'), 1);

majJI = microtonal.scales.ratio_scale(root, ji.major_5limit, 1);
minJI = microtonal.scales.ratio_scale(root, ji.minor_5limit, 1);

%% Helpers
render_melody = @(freqs) microtonal.audio.build_audio_buffer( ...
    freqs, ...
    (0:length(freqs)-1) * (note_dur + note_gap), ...
    ones(size(freqs)) * note_dur, ...
    sound_func, fs);

render_chord = @(freqs) microtonal.audio.build_audio_buffer( ...
    freqs, ...
    zeros(size(freqs)), ...
    ones(size(freqs)) * triad_dur, ...
    sound_func, fs);

render_interval = @(freqs) microtonal.audio.build_audio_buffer( ...
    freqs, ...
    [0, interval_dur + note_gap], ...
    [interval_dur, interval_dur], ...
    sound_func, fs);

silence = @(dur) zeros(1, round(fs * dur));

play_section = @(buf) playblocking(audioplayer(buf, fs));

%% =============================================
%  FULL DEMO
%% =============================================
full_buf = [];

% --- Major scale ---
fprintf('\n=== SCALES ===\n');
fprintf('  12-TET Major scale...\n');
section = [render_melody(maj12), silence(pair_gap)];
play_section(section);
full_buf = [full_buf, section];

fprintf('  Just Intonation Major scale...\n');
section = [render_melody(majJI), silence(section_gap)];
play_section(section);
full_buf = [full_buf, section];

% --- Minor scale ---
fprintf('  12-TET Minor scale...\n');
section = [render_melody(min12), silence(pair_gap)];
play_section(section);
full_buf = [full_buf, section];

fprintf('  Just Intonation Minor scale...\n');
section = [render_melody(minJI), silence(section_gap)];
play_section(section);
full_buf = [full_buf, section];

% --- Chromatic scale ---
fprintf('  12-TET Chromatic scale...\n');
section = [render_melody(chr12), silence(section_gap)];
play_section(section);
full_buf = [full_buf, section];

%% --- Isolated intervals ---
fprintf('\n=== INTERVALS (root + interval, ET then JI) ===\n');

intervals = {
    'Major 2nd',   2^(2/12),  9/8,   'ratio 9/8';
    'Minor 3rd',   2^(3/12),  6/5,   'ratio 6/5';
    'Major 3rd',   2^(4/12),  5/4,   'ratio 5/4';
    'Perfect 4th', 2^(5/12),  4/3,   'ratio 4/3';
    'Perfect 5th', 2^(7/12),  3/2,   'ratio 3/2';
    'Major 6th',   2^(9/12),  5/3,   'ratio 5/3';
    'Major 7th',   2^(11/12), 15/8,  'ratio 15/8';
};

for k = 1:size(intervals, 1)
    name    = intervals{k, 1};
    et_rat  = intervals{k, 2};
    ji_rat  = intervals{k, 3};
    ji_desc = intervals{k, 4};

    et_cents = 1200 * log2(et_rat);
    ji_cents = 1200 * log2(ji_rat);
    diff_cents = et_cents - ji_cents;

    fprintf('  %s — ET: %.1f cents, JI: %.1f cents (%s), diff: %+.1f cents\n', ...
        name, et_cents, ji_cents, ji_desc, diff_cents);

    % Play root→interval for ET, then for JI
    fprintf('    12-TET...\n');
    section = [render_interval([root, root * et_rat]), silence(pair_gap)];
    play_section(section);
    full_buf = [full_buf, section];

    fprintf('    Just Intonation...\n');
    section = [render_interval([root, root * ji_rat]), silence(section_gap)];
    play_section(section);
    full_buf = [full_buf, section];
end

%% --- Triads ---
fprintf('\n=== TRIADS (simultaneous) ===\n');

fprintf('  12-TET Major triad...\n');
triad12_maj = root * [1, 2^(4/12), 2^(7/12)];
section = [render_chord(triad12_maj), silence(pair_gap)];
play_section(section);
full_buf = [full_buf, section];

fprintf('  JI Major triad (1 : 5/4 : 3/2)...\n');
triadJI_maj = root * [1, 5/4, 3/2];
section = [render_chord(triadJI_maj), silence(section_gap)];
play_section(section);
full_buf = [full_buf, section];

fprintf('  12-TET Minor triad...\n');
triad12_min = root * [1, 2^(3/12), 2^(7/12)];
section = [render_chord(triad12_min), silence(pair_gap)];
play_section(section);
full_buf = [full_buf, section];

fprintf('  JI Minor triad (1 : 6/5 : 3/2)...\n');
triadJI_min = root * [1, 6/5, 3/2];
section = [render_chord(triadJI_min), silence(pair_gap)];
play_section(section);
full_buf = [full_buf, section];

%% Save the full demo
fprintf('\nDone! Saving to audio_files/tuning_comparison.wav\n');
full_buf = full_buf / max(abs(full_buf)) * 0.9;
audiowrite('audio_files/tuning_comparison.wav', full_buf, fs);
