function vibe = vibraphone(freq, fs, dur)
    t = 0:1/fs:dur;
    
    % Very quick attack
    attack_samples = round(0.0005 * fs);
    attack_env = ones(size(t));
    if attack_samples > 0
        attack_env(1:attack_samples) = linspace(0, 1, attack_samples);
    end
    
    % Vibraphone has metallic overtones and slower decay
    % Also has slight vibrato
    vibrato_rate = 6;  % Hz
    vibrato_depth = 0.005;
    vibrato = 1 + vibrato_depth * sin(2*pi*vibrato_rate*t);
    
    vibe = sin(2*pi*freq*vibrato.*t) .* exp(-2*t) + ...
           0.6*sin(2*pi*freq*2.76*t) .* exp(-2.5*t) + ...  % Inharmonic
           0.4*sin(2*pi*freq*5.40*t) .* exp(-3*t) + ...
           0.2*sin(2*pi*freq*8.93*t) .* exp(-3.5*t);
    
    % Metallic attack
    strike_dur = 0.001;
    strike_samples = round(strike_dur * fs);
    strike = randn(1, strike_samples) * 0.1 .* exp(-600*(0:strike_samples-1)/fs);
    
    if length(vibe) >= strike_samples
        vibe(1:strike_samples) = vibe(1:strike_samples) + strike;
    end
    
    vibe = vibe .* attack_env;
    
    % Fade out
    fade_samples = min(round(0.02 * fs), length(vibe));
    fade_out = ones(size(vibe));
    fade_out(end-fade_samples+1:end) = linspace(1, 0, fade_samples);
    vibe = vibe .* fade_out;
    
    vibe = vibe / max(abs(vibe));
end