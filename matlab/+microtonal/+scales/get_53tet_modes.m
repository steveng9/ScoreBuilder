
function modes = get_53tet_modes()
    % GET_53TET_MODES Example modes for 53-tone equal temperament
    % 53-TET is remarkable for its accuracy in approximating Pythagorean tuning
    % and provides extremely precise intervals
    
    modes = struct();
    
    % Pythagorean major scale in 53-TET
    modes.pythagorean_major = [0, 9, 18, 22, 31, 40, 49];
    
    % Pure thirds major (5-limit just intonation approximation)
    modes.just_major = [0, 9, 17, 22, 31, 39, 48];
    
    % All 53 tones
    modes.chromatic = 0:52;
end
