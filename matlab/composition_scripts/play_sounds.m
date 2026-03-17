
addpath('sounds');
fs = 44100;

% inorder_5tet = 1:4;
% inorder_7tet = 1:6;
% inorder_12tet = 1:11;
% inorder_19tet = 1:18;
% inorder_31tet = 1:30;
% inorder_37tet = 1:36;


% dur = .5;
scale_tet = 5;
sounds = {@tonal_percussion, @vibraphone, @breathy_flute, @shakuhachi, @tubular_bell, @whispy_whistle, @wind_chime}

for i = 1:length(sounds)
    sound_func = sounds{i};

    num_octaves = 3;
    % scale = microtonal.scales.tet_scales(microtonal.scales.note_to_freq("c3"), scale_tet, 0:(scale_tet-1), num_octaves);
    scale = microtonal.scales.tet_scales(microtonal.scales.note_to_freq("c3"), scale_tet, 0:(scale_tet-1), num_octaves);
    start_times = linspace(0, 4, scale_tet*num_octaves+1);
    % sep = .4;
    dur = (start_times(2) - start_times(1))*6;
    % start_times = 0:sep:i*sep;
    disp(start_times);

    audio_buffer = microtonal.audio.build_audio_buffer(scale, start_times, dur * ones(1, scale_tet*num_octaves+1), sound_func);
    
        
    % Save and play 
    audiowrite('scales.wav', audio_buffer, fs);
    % sound(audio_buffer, fs);

    [y, Fs] = audioread('scales.wav');
    player = audioplayer(y, Fs);
    play(player);
    while isplaying(player)
        pause(0.1);
    end
    disp('Audio playback finished.');

    
end














