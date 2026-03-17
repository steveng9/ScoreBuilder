
addpath('sounds');

note_durs = 6;
tet = 12;
scale = microtonal.scales.tet_scales(microtonal.scales.note_to_freq("A3"), tet, microtonal.scales.get_mode(tet, 'major'), 3);
tone = @crystal_bowl_with_pop;

L = 20;
S = 8;
notes = [];
start_times = [];
for i = 1:S
    rhythm = exp((S-i)/S*2);
    disp(rhythm);
    s = 0:rhythm:L;
    start_times = [start_times, s];
    notes = [notes, repmat(scale(i), 1, length(s))];
end

durations = note_durs * ones(1, length(notes));


audio_buffer = microtonal.audio.build_audio_buffer(notes, start_times, durations, tone);

microtonal.audio.play("LCM_4", audio_buffer);














