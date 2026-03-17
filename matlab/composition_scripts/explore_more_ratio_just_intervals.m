% Pure ratio interval exploration

addpath('sounds');

% Parameters
fs = 44100;
sound_func = @crystal_bowl_with_pop;
base_freq = microtonal.scales.note_to_freq('C3');  % Tonic frequency

%% PART 1: Explore specific intervals
% 
% fprintf('=== PART 1: INTERVAL EXPLORATION ===\n\n');
% 
% % Define intervals to explore
% intervals = {
%     {7/4, 'Harmonic 7th (compare to minor 7 and major 6)'}
%     {7/6, '7/6'}
%     {11/6, '11/6'}
%     {8/7, '8/7'}
%     {9/7, '9/7'}
%     {10/7, '10/7'}
%     {11/7, '11/7'}
%     {12/7, '12/7'}
%     {13/7, '13/7'}
%     {13/10, '13/10'}
%     {17/10, '17/10'}
% };
% 
% % Play each interval: melodic then harmonic
% for i = 1:length(intervals)
%     ratio = intervals{i}{1};
%     description = intervals{i}{2};
% 
%     fprintf('Playing %s (ratio: %.4f)\n', description, ratio);
% 
%     % Calculate frequencies
%     freq1 = base_freq;
%     freq2 = base_freq * ratio;
% 
%     % Melodic (one after another)
%     fprintf('  Melodic...\n');
%     audio_melodic = microtonal.audio.build_audio_buffer(...
%         [freq1, freq2], [0, 1.5], [1.2, 2], sound_func);
%     sound(audio_melodic, fs);
%     pause(length(audio_melodic)/fs + 0.5);
% 
%     % Harmonic (together)
%     fprintf('  Harmonic...\n');
%     audio_harmonic = microtonal.audio.build_audio_buffer(...
%         [freq1, freq2], [0, 0], [3, 3], sound_func);
%     sound(audio_harmonic, fs);
%     pause(length(audio_harmonic)/fs + 0.5);
% 
%     fprintf('\n');
% end

%% PART 2: Scale with all ratios over X (linear spacing!)
X = 10;
fprintf('=== PART 2: LINEAR SCALE (ratios n/%d) ===\n\n', X);

% Generate scale: X/X, (X+1)/X, (X+2)/X, ..., 2X/X
scale_ratios = (X:X*2) / X;
scale_freqs = base_freq * scale_ratios;

fprintf('Scale frequencies:\n');
for i = 1:length(scale_ratios)
    fprintf('  %d/%d = %.2f Hz\n', i+(X-1), X, scale_freqs(i));
end
fprintf('\n');

% Play the scale ascending
fprintf('Playing ascending scale...\n');
note_times = (0:length(scale_freqs)-1) * 0.6;
note_durs = ones(1, length(scale_freqs)) * 1.2;
note_durs(end) = 3;  % Let last note ring

audio_scale = microtonal.audio.build_audio_buffer(scale_freqs, note_times, note_durs, sound_func);
sound(audio_scale, fs);
pause(length(audio_scale)/fs + 0.5);

% Play some random melodies with this scale
fprintf('Playing random melodies with linear scale...\n\n');

num_melodies = 3;
notes_per_melody = 18;

for melody_num = 1:num_melodies
    fprintf('Melody %d:\n', melody_num);
    
    % Generate random note sequence
    note_indices = randi([1, length(scale_freqs)], 1, notes_per_melody);
    melody_freqs = scale_freqs(note_indices);
    
    % Create rhythm
    melody_times = (0:notes_per_melody-1) * 0.5;
    melody_durs = ones(1, notes_per_melody) * 1.5;
    melody_durs(end) = 3;  % Final note rings out
    
    % Build and play
    audio_melody = microtonal.audio.build_audio_buffer(melody_freqs, melody_times, melody_durs, sound_func);
    sound(audio_melody, fs);
    pause(length(audio_melody)/fs + 0.5);
end

fprintf('\n=== EXPLORATION COMPLETE ===\n');
fprintf('Notice how the linear scale (n/10) has uneven spacing,\n');
fprintf('unlike the exponential spacing of equal temperament!\n');