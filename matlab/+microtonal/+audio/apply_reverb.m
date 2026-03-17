function out = apply_reverb(audio_in, fs, rt60, wet)
% APPLY_REVERB Apply synthetic convolution reverb to an audio buffer
%
% Generates a room impulse response entirely in MATLAB (no IR file needed)
% and convolves it with the input via FFT. The output is slightly longer
% than the input so the reverb tail decays naturally after the last note.
%
% The IR has three components:
%   1. Pre-delay  — models the time sound takes to reach the first wall
%   2. Early reflections — a handful of discrete echoes with Fibonacci spacing
%   3. Diffuse late tail — two filtered noise streams (broadband + low-pass)
%      with independent exponential decay; high frequencies fade faster,
%      mimicking air absorption in a real room.
%
% Usage:
%   buf = microtonal.audio.apply_reverb(buf);
%   buf = microtonal.audio.apply_reverb(buf, 44100, 1.8, 0.30);
%   microtonal.audio.play('piece', buf);
%
% Inputs:
%   audio_in  - input audio row vector
%   fs        - sample rate in Hz          (default 44100)
%   rt60      - reverb decay time in s     (default 1.3 — medium room)
%   wet       - wet/dry mix 0..1           (default 0.28)
%
% Output:
%   out  - mixed audio, length = length(audio_in) + length(IR) - 1

    if nargin < 2, fs   = 44100; end
    if nargin < 3, rt60 = 1.3;   end
    if nargin < 4, wet  = 0.28;  end

    ir = make_ir(fs, rt60);

    % FFT convolution — O(N log N) instead of O(N^2)
    n_fft   = 2^nextpow2(length(audio_in) + length(ir) - 1);
    wet_sig = real(ifft(fft(audio_in, n_fft) .* fft(ir, n_fft)));
    out_len = length(audio_in) + length(ir) - 1;
    wet_sig = wet_sig(1:out_len);

    % Scale wet signal to the same RMS level as the dry signal.
    % This keeps the reverb from pumping up or down the perceived volume.
    rms_dry = sqrt(mean(audio_in.^2));
    rms_wet = sqrt(mean(wet_sig.^2));
    if rms_wet > 1e-10
        wet_sig = wet_sig * (rms_dry / rms_wet);
    end

    % Pad dry signal to match wet length, then mix
    dry_padded = [audio_in, zeros(1, out_len - length(audio_in))];
    out = (1 - wet) * dry_padded + wet * wet_sig;

    % Normalize to 0.9 peak
    pk = max(abs(out));
    if pk > 0, out = out / pk * 0.9; end
end


function ir = make_ir(fs, rt60)
% MAKE_IR Build a synthetic room impulse response
%
% Uses a deterministic random seed so the same IR is produced on every call.
% Restore the caller's RNG state afterward to avoid side effects.

    rng_state = rng();
    rng(7, 'twister');   % fixed seed — same IR on every call

    pre_s   = 0.013;     % 13 ms pre-delay (models ~4 m to nearest wall)
    ir_len  = round((rt60 + pre_s + 0.06) * fs);
    ir      = zeros(1, ir_len);
    pre     = round(pre_s * fs);

    % Direct sound
    ir(pre + 1) = 1.0;

    % --- Early reflections ---
    % Fibonacci-ish spacing avoids the "flutter echo" of regular delays.
    % Seven echoes covering 10–63 ms after the direct sound.
    er_ms   = [10,   14,   19,   26,   35,   47,   63];
    er_gain = [0.70, 0.62, 0.55, 0.47, 0.40, 0.33, 0.27];
    for i = 1:length(er_ms)
        samp = pre + round(er_ms(i) / 1000 * fs);
        if samp <= ir_len
            ir(samp) = ir(samp) + er_gain(i);
        end
    end

    % --- Diffuse late reverb tail ---
    % Two noise streams: full-band and low-pass filtered.
    % The low-pass stream represents lows that ring longer in real rooms.
    % Both decay exponentially; the low-pass one uses a longer RT60.
    late_start = pre + round(0.065 * fs);   % tail blends in ~65 ms in
    late_len   = ir_len - late_start;

    if late_len > 1
        t_late = (0:late_len-1) / fs;

        % Full-band noise stream
        n_full = randn(1, late_len);
        d_full = exp(-6.908 * t_late / rt60);

        % Low-pass noise stream: simple first-order IIR (~3.5 kHz cutoff at 44.1 kHz)
        % lows persist ~60% longer than the RT60 (air absorbs highs faster)
        n_low  = filter(0.15, [1, -0.85], randn(1, late_len));
        d_low  = exp(-6.908 * t_late / (rt60 * 1.6));

        tail = n_full .* d_full + 0.6 * n_low .* d_low;
        ir(late_start+1:end) = tail * 0.10;
    end

    % Normalize IR peak to 1 (RMS matching in caller handles final level)
    pk = max(abs(ir));
    if pk > 0, ir = ir / pk; end

    rng(rng_state);   % restore caller's RNG state
end
