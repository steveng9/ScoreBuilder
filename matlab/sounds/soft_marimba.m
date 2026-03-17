% 1. Soft marimba (warm, gentle)
function sound_out = soft_marimba(freq, fs, dur)
    t = 0:1/fs:dur;
    
    % Gentle attack
    attack_samples = round(0.008 * fs);
    attack_env = ones(size(t));
    if attack_samples > 0
        attack_env(1:attack_samples) = (1 - cos(pi*linspace(0,1,attack_samples)))/2;
    end
    
    % Warm, woody tone with slow decay
    sound_out = sin(2*pi*freq*t) .* exp(-1.8*t) + ...
                0.4*sin(2*pi*freq*2*t) .* exp(-2.2*t) + ...
                0.2*sin(2*pi*freq*3*t) .* exp(-2.6*t) + ...
                0.1*sin(2*pi*freq*4*t) .* exp(-3*t);
    
    sound_out = sound_out .* attack_env;
    
    % Fade out
    fade_samples = min(round(0.02 * fs), length(sound_out));
    fade_out = ones(size(sound_out));
    fade_out(end-fade_samples+1:end) = linspace(1, 0, fade_samples);
    sound_out = sound_out .* fade_out;
    
    sound_out = sound_out / max(abs(sound_out));
end
