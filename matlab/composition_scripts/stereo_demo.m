% Stereo panning demonstration

addpath('sounds');

% Parameters
fs = 44100;
sound_func = @crystal_bowl_with_pop;
c3 = microtonal.scales.note_to_freq('C3');

% Three notes
notes = [c3, c3 * 5/4, c3 * 3/2];  % C, E, G (major triad)

fprintf('=== STEREO PANNING DEMO ===\n\n');

%% Example 1: Hard left, center, hard right
fprintf('Example 1: Three notes panned left, center, right\n');

% Build mono audio for each note
note1 = microtonal.audio.build_audio_buffer(notes(1), 0, 3, sound_func);
note2 = microtonal.audio.build_audio_buffer(notes(2), 0, 3, sound_func);
note3 = microtonal.audio.build_audio_buffer(notes(3), 0, 3, sound_func);

% Make sure all are same length
max_len = max([length(note1), length(note2), length(note3)]);
note1 = [note1, zeros(1, max_len - length(note1))];
note2 = [note2, zeros(1, max_len - length(note2))];
note3 = [note3, zeros(1, max_len - length(note3))];

% Create stereo: [left_channel; right_channel]
stereo_audio = zeros(2, max_len);

% Note 1: Hard LEFT (100% left, 0% right)
stereo_audio(1, :) = stereo_audio(1, :) + note1;
stereo_audio(2, :) = stereo_audio(2, :) + 0 * note1;

% Note 2: CENTER (50% left, 50% right)
stereo_audio(1, :) = stereo_audio(1, :) + note2;
stereo_audio(2, :) = stereo_audio(2, :) + note2;

% Note 3: Hard RIGHT (0% left, 100% right)
stereo_audio(1, :) = stereo_audio(1, :) + 0 * note3;
stereo_audio(2, :) = stereo_audio(2, :) + note3;

% Normalize and play
stereo_audio = stereo_audio' / max(abs(stereo_audio(:))) * 0.9;
sound(stereo_audio, fs);
pause(max_len/fs + 1);

%% Example 2: Ascending scale that pans from left to right
fprintf('Example 2: Scale panning from left to right\n');

scale = c3 * [1, 9/8, 5/4, 4/3, 3/2, 5/3, 15/8, 2];  % Just intonation major
num_notes = length(scale);

% Build audio
mono_audio = microtonal.audio.build_audio_buffer(scale, (0:num_notes-1)*0.7, ones(1,num_notes)*1.5, sound_func);

% Create stereo with smooth panning
stereo_panning = zeros(2, length(mono_audio));

% Calculate pan position for each sample
for i = 1:length(mono_audio)
    % Pan goes from 0 (left) to 1 (right) over the duration
    pan = i / length(mono_audio);
    
    % Apply equal-power panning
    left_gain = cos(pan * pi/2);
    right_gain = sin(pan * pi/2);
    
    stereo_panning(1, i) = mono_audio(i) * left_gain;
    stereo_panning(2, i) = mono_audio(i) * right_gain;
end

stereo_panning = stereo_panning';
sound(stereo_panning, fs);
pause(length(stereo_panning)/fs + 1);

%% Example 3: Bouncing ball effect (ping-pong between speakers)
fprintf('Example 3: Notes bouncing left-right-left-right\n');

bounce_notes = [c3, c3*3/2, c3*2, c3*3/2];  % C, G, C, G
pans = [0, 1, 0, 1];  % 0=left, 1=right

stereo_bounce = zeros(2, round(fs * 5));

for i = 1:length(bounce_notes)
    note_audio = microtonal.audio.build_audio_buffer(bounce_notes(i), 0, 1.5, sound_func);
    start_sample = round((i-1) * 1.0 * fs) + 1;
    end_sample = min(start_sample + length(note_audio) - 1, size(stereo_bounce, 2));
    
    % Pan: 0=left, 1=right
    left_gain = cos(pans(i) * pi/2);
    right_gain = sin(pans(i) * pi/2);
    
    stereo_bounce(1, start_sample:end_sample) = stereo_bounce(1, start_sample:end_sample) + ...
        note_audio(1:(end_sample-start_sample+1)) * left_gain;
    stereo_bounce(2, start_sample:end_sample) = stereo_bounce(2, start_sample:end_sample) + ...
        note_audio(1:(end_sample-start_sample+1)) * right_gain;
end

stereo_bounce = stereo_bounce' / max(abs(stereo_bounce(:))) * 0.9;
sound(stereo_bounce, fs);

fprintf('\n=== DEMO COMPLETE ===\n');
fprintf('Key concepts:\n');
fprintf('- Stereo is [left; right] or 2-row matrix\n');
fprintf('- Pan = 0 means left, pan = 1 means right\n');
fprintf('- Use cos/sin for equal-power panning\n');
fprintf('- Transpose to (samples x 2) before sound()\n');