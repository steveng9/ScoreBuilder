% Golden ratio scale generator with modulation and gentle cutoffs

addpath('sounds');

% Parameters
N = 3;  % Number of notes in the pattern
c3 = microtonal.scales.note_to_freq('C3');  % Starting frequency
golden_ratio = (1 + sqrt(5)) / 2;  % φ ≈ 1.618

% Define your modulation sequence (shift amounts in semitones)
modulation_sequence = [0, 0, 2, 0, 2, 0, -1, -1, -1, -4, -4, -4, -4, -4, -4, 0, 0, 0, 0, 0, -4, 0, 0];  % Start, up 2 semitones, up 4, up 5, up 7

% Playback parameters
note_interval = 0.25;  % Time between note starts (seconds)
fadeout_time = 0.2;  % Gentle fadeout duration (seconds)
final_ringout = 5;  % How long to let final notes ring (seconds)

% Calculate durations
section_length = N * note_interval;  % Total time for one modulation section

% Generate the pattern for each modulation
note_sequence = [];
start_times = [];
durations = [];
current_time = 0;

for mod_idx = 1:length(modulation_sequence)
    % Calculate base frequency for this modulation
    base_freq = c3 * 2^(modulation_sequence(mod_idx)/12);
    
    % Generate golden ratio notes from this base
    for i = 1:N
        note_freq = base_freq * golden_ratio^(i-1);
        note_sequence = [note_sequence, note_freq];
        start_times = [start_times, current_time];
        
        % Check if this is the last modulation
        if mod_idx == length(modulation_sequence)
            % Let final notes ring out
            note_dur = final_ringout;
        else
            % Calculate how long until the next modulation starts
            time_until_next_section = section_length - (i - 1) * note_interval;
            note_dur = time_until_next_section + fadeout_time;
        end
        
        durations = [durations, note_dur];
        current_time = current_time + note_interval;
    end
    
    fprintf('Modulation %d: Base = %.2f Hz (%+d semitones from C3)\n', ...
        mod_idx, base_freq, modulation_sequence(mod_idx));
end

% Build audio
sound_func = @crystal_bowl_with_pop;
audio_buffer = microtonal.audio.build_audio_buffer(note_sequence, start_times, durations, sound_func);

% Play
sound(audio_buffer, 44100);
fprintf('\nPlaying %d modulations with %d notes each...\n', length(modulation_sequence), N);
fprintf('Each section is %.2f seconds long\n', section_length);
fprintf('Final notes ring for %.2f seconds\n', final_ringout);