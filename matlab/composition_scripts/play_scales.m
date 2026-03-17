
addpath('sounds');

dur = 4;

for i = [5, 7, 12, 19, 31, 37]
    scale = microtonal.scales.tet_scales(microtonal.scales.note_to_freq("c4"), i, 0:(i-1), 1);
    start_times = linspace(0, 8, i+1);

    audio_buffer = microtonal.audio.build_audio_buffer(scale, start_times, dur * ones(1, i+1), @tubular_bell);

    microtonal.audio.play("scales", audio_buffer);
end
