% Golden ratio scale generator
addpath('sounds');

% Parameters
N = 4;  % Number of notes
c3 = microtonal.scales.note_to_freq('C3');  % Starting frequency
% golden_ratio = (1 + sqrt(5)) / 2;  % φ ≈ 1.618
golden_ratio = 1.618
% golden_ratio = 1.682

% Generate notes
notes = zeros(1, N);
for i = 1:N
    notes(i) = c3 * golden_ratio^(i-1);
end

% Display the frequencies
fprintf('Golden Ratio Scale (%d notes):\n', N);
for i = 1:N
    fprintf('Note %d: %.2f Hz (C3 × φ^%d)\n', i, notes(i), i-1);
end

% Playback parameters
num_loops = 10;  % How many times to cycle through
note_interval = 0.4;  % Time between note starts (seconds) - adjust for tempo
note_duration = 3;  % How long each note lasts (seconds)

% Generate the full sequence
total_notes = N * num_loops;
note_sequence = repmat(notes, 1, num_loops);
start_times = (0:total_notes-1) * note_interval;
durations = ones(1, total_notes) * note_duration;

% Build audio using your existing function
sound_func = @crystal_bowl_with_pop;
audio_buffer = microtonal.audio.build_audio_buffer(note_sequence, start_times, durations, sound_func);

% Play
sound(audio_buffer, 44100);
fprintf('\nPlaying %d loops...\n', num_loops);