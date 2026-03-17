
function modes = get_19tet_modes()
    % GET_19TET_MODES Example modes for 19-tone equal temperament
    % 19-TET is interesting because it has good approximations of both
    % just intonation intervals and provides additional harmonic possibilities
    
    modes = struct();
    
    % Major-like scale in 19-TET (approximating 12-TET major)
    modes.major = [0, 3, 6, 8, 11, 14, 17];
    
    % Minor-like scale in 19-TET
    modes.minor = [0, 3, 5, 8, 11, 13, 16];
    
    % Exotic 19-TET scale
    modes.exotic1 = [0, 2, 5, 8, 10, 13, 16];
    
    % All 19 tones
    modes.chromatic = 0:18;
end