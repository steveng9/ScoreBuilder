
function sound_out = new_tone(freq, fs, dur)
    t = 0:1/fs:dur;
    
    % Slightly faster attack for more percussive feel
    attack_samples = round(0.001 * fs);  % 3ms (was 5ms)
    attack_env = ones(size(t));
    if attack_samples > 0
        attack_env(1:attack_samples) = (1 - cos(pi*linspace(0,1,attack_samples)))/2;
    end
    
    % Pure fundamental with just a hint of overtones (tubey/hollow sound)
    sound_out = sin(2*pi*freq*t) .* exp(-2.0*t) + ...
                0.3*sin(2*pi*freq*2*t) .* exp(-3*t) + ...
                0.15*sin(2*pi*freq*3.1*t) .* exp(-3.5*t);
                0.08*sin(2*pi*freq*2.6*t) .* exp(-4.5*t);
    
    sound_out = sound_out .* attack_env;
    
    % Fade out
    fade_samples = min(round(0.02 * fs), length(sound_out));
    fade_out = ones(size(sound_out));
    fade_out(end-fade_samples+1:end) = linspace(1, 0, fade_samples);
    sound_out = sound_out .* fade_out;
    
    sound_out = sound_out / max(abs(sound_out));
end