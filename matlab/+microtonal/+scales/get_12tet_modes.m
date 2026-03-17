
function modes = get_12tet_modes()
    % GET_12TET_MODES Returns standard modes for 12-tone equal temperament
    % Each mode is defined as steps within the 12-tone octave
    
    modes = struct();
    
    % Standard Western modes
    modes.major = [0, 2, 4, 5, 7, 9, 11];
    modes.minor = [0, 2, 3, 5, 7, 8, 10];  % Natural minor
    modes.harmonic_minor = [0, 2, 3, 5, 7, 8, 11];
    modes.melodic_minor = [0, 2, 3, 5, 7, 9, 11];
    
    % Church modes
    modes.dorian = [0, 2, 3, 5, 7, 9, 10];
    modes.phrygian = [0, 1, 3, 5, 7, 8, 10];
    modes.lydian = [0, 2, 4, 6, 7, 9, 11];
    modes.mixolydian = [0, 2, 4, 5, 7, 9, 10];
    modes.locrian = [0, 1, 3, 5, 6, 8, 10];
    
    % Pentatonic
    modes.pentatonic_major = [0, 2, 4, 7, 9];
    modes.pentatonic_minor = [0, 3, 5, 7, 10];
    
    % Other scales
    modes.chromatic = 0:11;  % All 12 tones
    modes.whole_tone = [0, 2, 4, 6, 8, 10];
    modes.blues = [0, 3, 5, 6, 7, 10];
    modes.diminished = [0, 2, 3, 5, 6, 8, 9, 11];  % Half-whole diminished
    
    % Jazz scales
    modes.bebop_major = [0, 2, 4, 5, 7, 8, 9, 11];
    modes.bebop_dominant = [0, 2, 4, 5, 7, 9, 10, 11];
end
