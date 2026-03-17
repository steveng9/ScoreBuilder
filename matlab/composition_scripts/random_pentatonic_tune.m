
addpath('sounds');

dur = 6;
num_notes = 50;
track_length = 6;

scale = microtonal.scales.tet_scales(microtonal.scales.note_to_freq("c2"), 12, microtonal.scales.get_mode(12, 'dorian'), 4);

start_times = [0, track_length * rand(1, num_notes-1)];
notes = datasample(scale, num_notes);
durations = dur * ones(1, num_notes);
audio_buffer = microtonal.audio.build_audio_buffer(notes, start_times, durations, @pure_tubey);

microtonal.audio.play("random_dorian", audio_buffer);














