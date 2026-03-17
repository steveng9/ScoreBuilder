function key_map = default_key_map()
    % DEFAULT_KEY_MAP Returns default QWERTY-to-scale-degree mapping
    %
    % Maps 3 rows of the keyboard to sequential scale degrees,
    % spanning roughly 2.5 octaves. Bottom row (Z-/) is the lowest,
    % home row (A-;) is the middle, top row (Q-P) is the highest.
    %
    % Output:
    %   key_map: containers.Map mapping character -> scale degree index
    %
    % The mapping assumes the scale has 7-10 notes per octave.
    % For scales with more or fewer notes, you may want to create
    % a custom key_map.

    key_map = containers.Map();

    % Bottom row: degrees 1-10 (lowest octave)
    bottom_keys = 'zxcvbnm,.';
    for i = 1:length(bottom_keys)
        key_map(bottom_keys(i)) = i;
    end

    % Home row: degrees 10-19 (middle octave)
    home_keys = 'asdfghjkl';
    for i = 1:length(home_keys)
        key_map(home_keys(i)) = i + 9;
    end

    % Top row: degrees 19-28 (upper octave)
    top_keys = 'qwertyuiop';
    for i = 1:length(top_keys)
        key_map(top_keys(i)) = i + 18;
    end
end
