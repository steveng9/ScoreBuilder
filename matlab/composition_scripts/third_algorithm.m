
addpath('sounds');

note_durs = 6;
tet = 12;
scale = microtonal.scales.tet_scales(microtonal.scales.note_to_freq("A3"), tet, microtonal.scales.get_mode(tet, 'major'), 3);
tone = @magic_shimmer;

L = 32;
s1 = 0:.5:L;
s2 = 0:.6:L;
s3 = 0:.7:L;
s4 = 0:.8:L;
s5 = 0:.9:L;
s6 = 0:1.0:L;
s7 = 0:1.2:L;
s8 = 0:1.4:L;

n1 = repmat(scale(8), 1, length(s1));
n2 = repmat(scale(7), 1, length(s2));
n3 = repmat(scale(6), 1, length(s3));
n4 = repmat(scale(5), 1, length(s4));
n5 = repmat(scale(4), 1, length(s5));
n6 = repmat(scale(3), 1, length(s6));
n7 = repmat(scale(2), 1, length(s7));
n8 = repmat(scale(1), 1, length(s8));





notes = [n1, n2, n3, n4, n5, n6, n7, n8];
start_times = [s1, s2, s3, s4, s5, s6, s7, s8];
durations = note_durs * ones(1, length(notes));


audio_buffer = microtonal.audio.build_audio_buffer(notes, start_times, durations, tone);

microtonal.audio.play("LCM", audio_buffer);














