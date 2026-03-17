function scales = get_ji_scales()
    % GET_JI_SCALES Predefined just intonation scale collections
    %
    % Returns a struct of named ratio arrays for use with ratio_scale().
    %
    % Example:
    %   ji = microtonal.get_ji_scales();
    %   freqs = microtonal.ratio_scale(261.63, ji.major_5limit, 2);
    %
    % See also: microtonal.ratio_scale, microtonal.cents, microtonal.compare_tunings

    scales = struct();

    % === 5-LIMIT (using primes 2, 3, 5) ===

    % Major scale (Ptolemy's intense diatonic / syntonic diatonic)
    scales.major_5limit = [1, 9/8, 5/4, 4/3, 3/2, 5/3, 15/8];

    % Natural minor
    scales.minor_5limit = [1, 9/8, 6/5, 4/3, 3/2, 8/5, 9/5];

    % === PYTHAGOREAN (using only primes 2, 3) ===

    % Major scale built from pure fifths (3/2)
    scales.pythagorean_major = [1, 9/8, 81/64, 4/3, 3/2, 27/16, 243/128];

    % Minor scale from pure fifths
    scales.pythagorean_minor = [1, 9/8, 32/27, 4/3, 3/2, 128/81, 16/9];

    % === 7-LIMIT (using primes 2, 3, 5, 7) ===

    % Septimal / blues-flavored major
    scales.major_7limit = [1, 9/8, 5/4, 4/3, 3/2, 5/3, 7/4];

    % Septimal minor (with harmonic 7th and subminor 3rd)
    scales.minor_7limit = [1, 9/8, 7/6, 4/3, 3/2, 8/5, 7/4];

    % === 11-LIMIT (using primes up to 11) ===

    % Scale featuring 11th harmonic intervals
    scales.scale_11limit = [1, 9/8, 5/4, 11/8, 3/2, 13/8, 7/4];

    % === HARMONIC SERIES SEGMENTS ===

    % Partials 8-16 (one octave of the harmonic series)
    scales.harmonic_8_16 = (8:16) / 8;

    % Partials 4-8
    scales.harmonic_4_8 = (4:8) / 4;

    % Partials 6-12 (starting from the 6th partial)
    scales.harmonic_6_12 = (6:12) / 6;

    % === HISTORICAL / WORLD SCALES ===

    % Al-Farabi's Arabic scale (10th century)
    scales.arabic_alfarabi = [1, 9/8, 81/64, 4/3, 3/2, 27/16, 16/9];

    % Slendro-like (approximation of Javanese pentatonic)
    scales.slendro_approx = [1, 8/7, 21/16, 3/2, 12/7];

    % === EXPERIMENTAL ===

    % Undertone series (subharmonics) — intervals inverted
    scales.undertone_8_16 = 16 ./ (16:-1:8) ;

    % Pure intervals only (no seconds) — great for drones/chords
    scales.drone = [1, 5/4, 4/3, 3/2, 5/3];
end
