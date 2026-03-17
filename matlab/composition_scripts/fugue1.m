
addpath('sounds');

tet = 12;
scale = microtonal.scales.tet_scales(microtonal.scales.note_to_freq("E2"), tet, microtonal.scales.get_mode(tet, 'major'), 5);
tone = @ocarina_sound;

pattern = [6,4,3,6,3,2,3];
rhythm = [1,1.5,3,1,1,1.5,3];
% pattern = [5,4,1,3,2,3,4];
% rhythm = [1,1,1,1,2,1,2];



%alto
alto_pattern = [mod7(pattern)+7, mod7(pattern+5)+7, mod7(pattern+3)+7, mod7(pattern-pattern(length(pattern))+4)+7];
alto_rhythms = [rhythm*1.5, rhythm*1.5, rhythm*3, rhythm*1.5];
alto_start_times = cumsum(alto_rhythms) - alto_rhythms;

%bass
bass_pattern = [mod7(pattern-pattern(1)), mod7(pattern+3), mod7(pattern-pattern(length(pattern)))];
bass_rhythms = [rhythm*3, rhythm*2.5, rhythm*2];
bass_start_times = cumsum(bass_rhythms) - bass_rhythms + alto_rhythms(1);

%soprano
soprano_pattern = [mod7(pattern+1)+14, mod7(pattern+2)+14, mod7(pattern-2)+14, mod7(pattern(4:7)-2)+14];
soprano_rhythms = [rhythm*1, rhythm*1.5, rhythm*1.5, rhythm(4:7)*1.5];
soprano_start_times = cumsum(soprano_rhythms) - soprano_rhythms + bass_start_times(5); 
soprano_start_times((length(pattern)*3+1):length(soprano_start_times)) = soprano_start_times((length(pattern)*3+1):length(soprano_start_times))+4; % add just a little space before last phrase

%tenor
tenor_pattern = [mod7(pattern)+3, mod7(pattern-pattern(length(pattern)))+2];
tenor_rhythms = [rhythm*2, rhythm*2];
tenor_start_times = cumsum(tenor_rhythms) - tenor_rhythms + soprano_start_times(9);



durations = [bass_rhythms, tenor_rhythms, alto_rhythms, soprano_rhythms] * 1.05;
notes = [scale(bass_pattern), scale(tenor_pattern), scale(alto_pattern), scale(soprano_pattern)];
start_times = [bass_start_times, tenor_start_times, alto_start_times, soprano_start_times];

multiplier = .29;
durations = durations * multiplier;
start_times = start_times * multiplier;
audio_buffer = microtonal.audio.build_audio_buffer(notes, start_times, durations, tone);

microtonal.audio.play("fugue2", audio_buffer);







function m = mod7(arr)
    m = mod(arr, 7) + 1
end






