function c = cents(ratio)
    % CENTS Convert a frequency ratio to cents
    %
    % One octave (ratio 2) = 1200 cents
    % One 12-TET semitone = 100 cents
    %
    % Input:
    %   ratio: frequency ratio (scalar or array), e.g. 3/2, 5/4
    %
    % Output:
    %   c: interval size in cents
    %
    % Examples:
    %   microtonal.cents(2)       % => 1200 (octave)
    %   microtonal.cents(3/2)     % => 701.96 (perfect fifth)
    %   microtonal.cents(5/4)     % => 386.31 (just major third)
    %   microtonal.cents(2^(7/12)) % => 700 (12-TET fifth)

    c = 1200 * log2(ratio);
end
