function sound_out = soft_bell(freq, fs, dur)
% SOFT_BELL Soft percussive bell — celesta / music-box timbre
%
% Uses the physical resonance modes of a free-free metal bar rather than
% integer harmonics. The resulting inharmonic partial series gives a true
% bell quality instead of the "synthy" character of additive sine synthesis.
%
% Partial ratios (free-free bar flexural modes): 1.000, 2.756, 5.404, 8.933
% A fixed cents-spread is applied to each partial to add warmth via beating.
%
%   freq  - fundamental frequency in Hz
%   fs    - sample rate in Hz
%   dur   - duration in seconds

    n = round(dur * fs);
    t = (0:n-1) / fs;

    % --- Inharmonic partial series ---
    % Ratios from free-free bar physics (glockenspiel / celesta / music box)
    ratios  = [1.000,  2.756,  5.404,  8.933];
    % Fixed detuning in cents — deterministic, same timbre every call
    detune  = [0.00,  +0.60,  -0.35,  +0.75];
    % Relative amplitudes (strong fundamental, falling off quickly)
    amps    = [1.000,  0.450,  0.200,  0.080];
    % Decay rate multipliers (higher partials fade first)
    d_mult  = [1.000,  1.500,  2.300,  3.800];

    % Base decay: lower notes ring longer (like real bells)
    base_decay = 1.0 + 120 / max(freq, 80);

    % Build tone from inharmonic partials
    sound_out = zeros(1, n);
    for k = 1:length(ratios)
        f_k = freq * ratios(k) * 2^(detune(k) / 1200);
        if f_k >= fs / 2, continue; end   % skip partials above Nyquist
        sound_out = sound_out + amps(k) * sin(2*pi*f_k*t) .* exp(-base_decay * d_mult(k) * t);
    end

    % Smooth cosine attack (4 ms — click-free onset)
    att = min(round(0.004 * fs), n);
    sound_out(1:att) = sound_out(1:att) .* (1 - cos(pi * (0:att-1) / att)) / 2;

    % Fade tail
    fade = min(round(0.015 * fs), n);
    sound_out(end-fade+1:end) = sound_out(end-fade+1:end) .* linspace(1, 0, fade);

    % Normalize
    pk = max(abs(sound_out));
    if pk > 0, sound_out = sound_out / pk; end
end
