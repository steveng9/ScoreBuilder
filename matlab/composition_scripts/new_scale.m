% Add the folder to MATLAB's path
addpath('sounds');

% Create a mix of overlapping bells
fs = 44100;
total_duration = 20;  % Total composition length
audio_buffer = zeros(1, total_duration * fs);







% pattern

n = 10;
resultArray = zeros(1, n);

for i = 1:n
    i = int16(i);
    if mod(i, 2) == 1
        resultArray(i) = idivide(i, 2) + 1;
    else
        resultArray(i) = idivide(i, 2) + 2;
    end
end

% Display the result
disp(resultArray);
pattern = resultArray;


% Times when each note should start (in seconds)
start_times = 0:.5:4.9;
disp(start_times);
scale = microtonal.scales.tet_scales(microtonal.scales.note_to_freq("c4"), 31, microtonal.scales.get_mode(31, 'quarter_tone'), 2);

notes = scale(pattern);
disp(notes);

start_times = [start_times, 2, 2];
disp(start_times);
notes = [notes, microtonal.scales.note_to_freq("a3"), microtonal.scales.note_to_freq("b3")];
disp(notes);


for i = 1:length(notes)
    bell_sound = tubular_bell(notes(i), fs, 8.0);
    start_sample = round(start_times(i) * fs) + 1;
    end_sample = min(start_sample + length(bell_sound) - 1, length(audio_buffer));
    
    % Add the bell sound to the buffer (overlapping)
    audio_buffer(start_sample:end_sample) = audio_buffer(start_sample:end_sample) + ...
        bell_sound(1:(end_sample - start_sample + 1));
end

% Normalize to prevent clipping
audio_buffer = audio_buffer / max(abs(audio_buffer)) * 0.9;

% Save and play 
audiowrite('new_scale2.wav', audio_buffer, fs);
sound(audio_buffer, fs);
