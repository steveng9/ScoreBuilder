
addpath('sounds');

note_durs = 6;
scale = microtonal.scales.tet_scales(microtonal.scales.note_to_freq("f3"), 12, microtonal.scales.get_mode(12, 'dorian'), 3);
tone = @pure_tubey;


steps1 = [1,2,1,2,1,1,2,1,1,1,2,1,3,4,5];
notes1 = scale(steps1);
start_times1 = (0:length(steps1)-1) *.5;

steps2 = [13, 15,16, 17, 13, 15, 16, 19, 13, 14, 16];
start_times2 = (0:length(steps2)-1) * 2/3;
notes2 = scale(steps2);

notes = [notes1, notes2];
start_times = [start_times1, start_times2];
durations = note_durs * ones(1, length(notes));


audio_buffer = microtonal.audio.build_audio_buffer(notes, start_times, durations, tone);

microtonal.audio.play("random_dorian", audio_buffer);














