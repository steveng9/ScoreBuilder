
% Type 2: Breathy flute / pan flute
function flute = breathy_flute(freq, fs, dur)
    t = 0:1/fs:dur;
    
    % Attack envelope
    attack_time = 0.05;  % Slower attack for breath sound
    attack_samples = round(attack_time * fs);
    attack_env = ones(size(t));
    attack_env(1:attack_samples) = linspace(0, 1, attack_samples);
    
    % Fundamental + odd harmonics (flute-like)
    flute = sin(2*pi*freq*t) + ...
            0.3*sin(2*pi*freq*3*t) + ...
            0.1*sin(2*pi*freq*5*t);
    
    % Add breath noise (filtered white noise)
    noise = randn(size(t)) * 0.15;
    % Low-pass filter the noise
    noise = filter(ones(1,20)/20, 1, noise);
    
    % Combine tone and noise
    flute = flute + noise;
    
    % Gentle decay envelope
    decay_env = exp(-0.5*t);
    
    flute = flute .* attack_env .* decay_env;
    flute = flute / max(abs(flute));
end