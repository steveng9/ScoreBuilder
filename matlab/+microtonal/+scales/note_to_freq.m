
% ========================================================================
% CONVENIENCE FUNCTION: Convert note name to frequency (12-TET only)
% ========================================================================
function freq = note_to_freq(note_name)
    % NOTE_TO_FREQ Converts standard note names to frequencies (12-TET)
    % Examples: 'C4' -> 261.63, 'A4' -> 440, 'F#3' -> 185.00
    
    note_name = char(note_name);
    note_letter = upper(note_name(1));
    
    % Check for sharp or flat
    idx = 2;
    accidental = 0;
    if length(note_name) >= 2
        if note_name(2) == '#'
            accidental = 1;
            idx = 3;
        elseif note_name(2) == 'b'
            accidental = -1;
            idx = 3;
        end
    end
    
    octave = str2double(note_name(idx:end));
    
    note_map = containers.Map(...
        {'C', 'D', 'E', 'F', 'G', 'A', 'B'}, ...
        [0, 2, 4, 5, 7, 9, 11]);
    
    semitones_from_c = note_map(note_letter) + accidental;
    semitones_from_a4 = semitones_from_c + (octave - 4) * 12 - 9;
    freq = 440 * (2^(semitones_from_a4/12));
end
