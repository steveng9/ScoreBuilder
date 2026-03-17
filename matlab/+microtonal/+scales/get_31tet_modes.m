

function modes = get_31tet_modes()
    % GET_31TET_MODES Example modes for 31-tone equal temperament
    % 31-TET provides excellent approximations of 5-limit just intonation
    % and allows for very pure thirds and sixths
    
    modes = struct();
    
    % Major-like scale in 31-TET (better approximation of just intonation)
    modes.major = [0, 5, 10, 13, 18, 23, 28];
    
    % Minor-like scale in 31-TET
    modes.minor = [0, 5, 8, 13, 18, 21, 26];
    
    % Harmonic series approximation in 31-TET
    modes.harmonic = [0, 5, 10, 13, 17, 20, 23, 26, 28];
    
    % Quarter-tone scale (using the extra resolution)
    modes.quarter_tone = [0, 2, 5, 7, 10, 13, 15, 18, 20, 23, 26, 28];
    
    % All 31 tones
    modes.chromatic = 0:30;
    
    % Example custom mode (you'll add your own here!)
    modes.custom1 = [0, 3, 7, 11, 15, 19, 23, 27];
end
