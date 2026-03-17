function frequencies = ratio_scale(root_freq, ratios, num_octaves)
    % RATIO_SCALE Generate frequencies from rational interval ratios
    %
    % Unlike tet_scales which divides the octave into equal logarithmic steps,
    % this function uses exact frequency ratios for just intonation and other
    % ratio-based tuning systems.
    %
    % Inputs:
    %   root_freq: root frequency in Hz (e.g., 261.63 for C4)
    %   ratios: array of frequency ratios relative to root (e.g., [1, 9/8, 5/4, ...])
    %           ratios should be >= 1 and < 2 (within one octave), starting with 1
    %   num_octaves: number of octaves to generate (default: 1)
    %
    % Output:
    %   frequencies: array of frequencies spanning the requested octaves
    %
    % Examples:
    %   % 5-limit just intonation major scale
    %   ji_major = microtonal.ratio_scale(261.63, [1, 9/8, 5/4, 4/3, 3/2, 5/3, 15/8], 2);
    %
    %   % Harmonic series (partials 1-8)
    %   harmonics = microtonal.ratio_scale(110, (1:8)/1, 1);
    %
    %   % Pythagorean tuning
    %   pyth = microtonal.ratio_scale(440, [1, 9/8, 81/64, 4/3, 3/2, 27/16, 243/128], 1);
    %
    %   % Custom ratio experiment
    %   custom = microtonal.ratio_scale(220, [1, 7/6, 5/4, 11/8, 3/2, 7/4], 2);

    if nargin < 3
        num_octaves = 1;
    end

    frequencies = [];
    for octave = 0:(num_octaves - 1)
        octave_multiplier = 2^octave;
        frequencies = [frequencies, root_freq * ratios * octave_multiplier];
    end

    % Add the final root note of the last octave
    frequencies = [frequencies, root_freq * (2^num_octaves)];
end
