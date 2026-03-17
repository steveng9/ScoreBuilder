function mode_steps = get_mode(tet, mode_name)
    % GET_MODE Retrieve mode definition for a given TET system
    %
    % Inputs:
    %   tet: Number of equal divisions per octave (12, 19, 31, 53, or any custom)
    %   mode_name: Name of the mode (e.g., 'major', 'minor', 'dorian')
    %
    % For supported TET systems (12, 19, 31, 53), looks up predefined modes.
    % For other TET values, looks for a function named get_<tet>tet_modes
    % in the +microtonal package (e.g., +microtonal/get_17tet_modes.m).
    %
    % Examples:
    %   steps = microtonal.get_mode(12, 'major');
    %   steps = microtonal.get_mode(31, 'harmonic');

    switch tet
        case 12
            modes = microtonal.scales.get_12tet_modes();
        case 19
            modes = microtonal.scales.get_19tet_modes();
        case 31
            modes = microtonal.scales.get_31tet_modes();
        case 53
            modes = microtonal.scales.get_53tet_modes();
        otherwise
            % Try to find a custom mode function: microtonal.scales.get_<tet>tet_modes
            func_name = sprintf('microtonal.scales.get_%dtet_modes', tet);
            if exist(func_name, 'file') || exist(func_name, 'builtin')
                modes = feval(func_name);
            else
                error(['No mode definitions available for %d-TET.\n' ...
                       'Create +microtonal/+scales/get_%dtet_modes.m to define your own.'], tet, tet);
            end
    end

    if ~isfield(modes, mode_name)
        available = strjoin(fieldnames(modes), ', ');
        error('Mode "%s" not defined for %d-TET. Available: %s', mode_name, tet, available);
    end

    mode_steps = modes.(mode_name);

    % Validate steps are within range
    if any(mode_steps < 0) || any(mode_steps >= tet)
        error('Mode steps must be in range [0, %d) for %d-TET', tet, tet);
    end
end
