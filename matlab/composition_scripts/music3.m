% Add the folder to MATLAB's path
addpath('sounds');

% Create a mix of overlapping bells
fs = 44100;
total_duration = 5;  % Total composition length
audio_buffer = zeros(1, total_duration * fs);

% Times when each note should start (in seconds)
start_times = [0, 0.2, .45, 1.5, 2.0, 2.5];
notes = [440, 523, 659, 523, 440, 392];  % A, C, E, C, A, G

for i = 1:length(notes)
    bell_sound = tubular_bell(notes(i), fs, 2.0);
    start_sample = round(start_times(i) * fs) + 1;
    end_sample = min(start_sample + length(bell_sound) - 1, length(audio_buffer));
    
    % Add the bell sound to the buffer (overlapping)
    audio_buffer(start_sample:end_sample) = audio_buffer(start_sample:end_sample) + ...
        bell_sound(1:(end_sample - start_sample + 1));
end

% Normalize to prevent clipping
audio_buffer = audio_buffer / max(abs(audio_buffer)) * 0.9;

% Save and play
audiowrite('overlapping_bells.wav', audio_buffer, fs);
sound(audio_buffer, fs);