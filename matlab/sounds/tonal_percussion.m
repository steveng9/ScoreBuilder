function perc = tonal_percussion(freq, fs, dur)
    t = 0:1/fs:dur;
    
    % Instant attack (percussive)
    attack_samples = round(0.001 * fs);
    attack_env = ones(size(t));
    if attack_samples > 0
        attack_env(1:attack_samples) = linspace(0, 1, attack_samples);
    end
    
    % Marimba/xylophone-like sound
    % Has a clear fundamental but also strong harmonics
    % Key: relatively fast decay but not too fast
    perc = sin(2*pi*freq*t) .* exp(-3*t) + ...
           0.5*sin(2*pi*freq*2*t) .* exp(-4*t) + ...
           0.3*sin(2*pi*freq*3*t) .* exp(-5*t) + ...
           0.2*sin(2*pi*freq*4*t) .* exp(-6*t) + ...
           0.1*sin(2*pi*freq*5*t) .* exp(-7*t);
    
    % Add a bit of noise for the "mallet strike"
    strike_dur = 0.002;
    strike_samples = round(strike_dur * fs);
    strike = randn(1, strike_samples) * 0.15 .* exp(-400*(0:strike_samples-1)/fs);
    
    if length(perc) >= strike_samples
        perc(1:strike_samples) = perc(1:strike_samples) + strike;
    end
    
    perc = perc .* attack_env;
    
    % Fade out
    fade_samples = min(round(0.02 * fs), length(perc));
    fade_out = ones(size(perc));
    fade_out(end-fade_samples+1:end) = linspace(1, 0, fade_samples);
    perc = perc .* fade_out;
    
    perc = perc / max(abs(perc));
end