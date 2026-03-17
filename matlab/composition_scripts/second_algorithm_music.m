
addpath('sounds');

note_durs = 5;
scale = microtonal.scales.tet_scales(microtonal.scales.note_to_freq("A3"), 31, microtonal.scales.get_mode(31, 'major'), 3);
tone = @pure_tubey;


% initialArray = [5, 2, 1, 3];
% initialArray = [1,6,5,3];
% initialArray = [1,6,5,3,4];
% initialArray = [6,5,6,3,2,3];
initialArray = [6,5,3];
n = 5; % modulus value
M = 1;
meter = .37;

% Initialize an empty array to store results
resultArray = [];

% Loop to generate the repeated array with decremented values
for i = 0:(n*M)-1
    % Calculate the new value and apply modulo
    newValue = mod(initialArray - i, n);
    % Append to the result array
    resultArray = [resultArray, newValue]; % Store each row
end






start_times = (0:length(resultArray)-1) * meter;

notes1 = scale(mod(resultArray, n) + 1);
notes2 = scale(mod(resultArray + 3, n) + 3);
notes3 = scale(mod(resultArray + 5, n) + 13);

% notes = [notes1, notes2, notes3];
notes = [notes1, notes2];
start_times = [start_times, start_times];
% start_times = [start_times, start_times, start_times];
durations = note_durs * ones(1, length(notes));


audio_buffer = microtonal.audio.build_audio_buffer(notes, start_times, durations, tone);

microtonal.audio.play("first_walk", audio_buffer);














