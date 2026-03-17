function piano = piano_sound(freq, fs, dur)
% PIANO_SOUND Physically-informed piano synthesis
%
% Models three detuned "unison strings" per note (as on a real grand piano),
% each with inharmonic partial frequencies:
%
%   f_p = p * f_string * sqrt(1 + B * p^2)
%
% where B is the inharmonicity coefficient, which grows with frequency
% (shorter, stiffer strings in the treble have higher B). The slow beating
% between the three strings (~2-4 Hz) produces the characteristic warmth
% that purely harmonic additive synthesis lacks.
%
%   freq  - fundamental frequency in Hz
%   fs    - sample rate in Hz
%   dur   - duration in seconds

    n = round(dur * fs);
    t = (0:n-1) / fs;

    % --- String inharmonicity coefficient ---
    % B grows with freq: treble strings are shorter and stiffer
    B = 0.00025 * (freq / 440)^1.5;
    B = min(B, 0.008);   % clamp — avoid extreme distortion at very high notes

    % --- Three unison strings per note ---
    % Spread of ±1.8 cents → ~2–4 Hz beating in the middle register
    string_detune  = [-1.8,  0.0,  +1.8];   % cents
    string_d_scale = [ 0.95, 1.00,   1.05];  % each string decays slightly differently

    % --- Partial amplitude envelope ---
    % Empirical piano spectrum: strong low harmonics, falling off above 4th
    num_harmonics = 8;
    harm_amps = [1.00, 0.55, 0.28, 0.16, 0.10, 0.07, 0.05, 0.03];

    % --- Decay times ---
    % Base decay: lower notes sustain much longer than high notes
    base_decay = 0.25 + (440 / max(freq, 50)) * 0.22;
    % Higher harmonics decay faster (lose overtone brightness first)
    harm_decay_add = (0:num_harmonics-1) * 0.08;

    % --- Build tone from all strings and harmonics ---
    piano = zeros(1, n);
    for s = 1:3
        f_string = freq * 2^(string_detune(s) / 1200);
        d_string = base_decay * string_d_scale(s);
        for p = 1:num_harmonics
            % Inharmonic frequency: sharper than integer multiples
            f_p = p * f_string * sqrt(1 + B * p^2);
            if f_p >= fs / 2, continue; end   % skip if above Nyquist
            decay_p = d_string + harm_decay_add(p);
            % Divide by 3 so total level matches single-string designs
            piano = piano + (harm_amps(p) / 3) * sin(2*pi*f_p*t) .* exp(-decay_p*t);
        end
    end

    % --- Hammer strike noise (the physical "thunk" at key press) ---
    strike_n = round(0.005 * fs);
    if n >= strike_n
        strike = randn(1, strike_n) * 0.12 .* exp(-600 * (0:strike_n-1) / fs);
        piano(1:strike_n) = piano(1:strike_n) + strike;
    end

    % Smooth cosine attack (3 ms — instant but click-free)
    att = min(round(0.003 * fs), n);
    piano(1:att) = piano(1:att) .* (1 - cos(pi * (0:att-1) / att)) / 2;

    % Fade tail
    fade = min(round(0.02 * fs), n);
    piano(end-fade+1:end) = piano(end-fade+1:end) .* linspace(1, 0, fade);

    % Normalize
    pk = max(abs(piano));
    if pk > 0, piano = piano / pk; end
end
