function frequencies = tet_scales(root_freq, tet, mode_steps, num_octaves)
    % TET_SCALES Generate frequencies for a scale in any TET system
    %
    % Inputs:
    %   root_freq: Root frequency in Hz (e.g., 440 for A4, 261.63 for C4)
    %   tet: Number of equal divisions per octave (e.g., 12, 19, 31, 53)
    %   mode_steps: Array of scale degrees within the TET (e.g., [0,2,4,5,7,9,11] for major)
    %   num_octaves: Number of octaves to generate (default: 1)
    %
    % Output:
    %   frequencies: Array of frequencies for the scale
    %
    % Examples:
    %   major_scale = microtonal.tet_scales(261.63, 12, [0,2,4,5,7,9,11], 2);
    %   custom_31tet = microtonal.tet_scales(440, 31, [0,5,10,13,18,23,28], 1);

    if nargin < 4
        num_octaves = 1;
    end

    % Build all step values across octaves (vectorized)
    octaves = 0:(num_octaves - 1);
    steps = bsxfun(@plus, mode_steps(:), octaves * tet);
    steps = steps(:)';

    % Compute frequencies from steps
    frequencies = root_freq * (2 .^ (steps / tet));

    % Add the final root note of the last octave
    frequencies = [frequencies, root_freq * (2^num_octaves)];
end
