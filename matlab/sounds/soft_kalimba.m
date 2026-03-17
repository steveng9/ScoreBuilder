
% 5. Soft kalimba (thumb piano, gentle)
function sound_out = soft_kalimba(freq, fs, dur)
    t = 0:1/fs:dur;
    
    % Medium-soft attack
    attack_samples = round(0.005 * fs);
    attack_env = ones(size(t));
    if attack_samples > 0
        attack_env(1:attack_samples) = (1 - cos(pi*linspace(0,1,attack_samples)))/2;
    end
    
    % Kalimba has strong fundamental and selected harmonics
    sound_out = sin(2*pi*freq*t) .* exp(-2*t) + ...
                0.4*sin(2*pi*freq*3*t) .* exp(-2.5*t) + ...
                0.2*sin(2*pi*freq*5*t) .* exp(-3*t);
    
    sound_out = sound_out .* attack_env;
    
    fade_samples = min(round(0.02 * fs), length(sound_out));
    fade_out = ones(size(sound_out));
    fade_out(end-fade_samples+1:end) = linspace(1, 0, fade_samples);
    sound_out = sound_out .* fade_out;
    
    sound_out = sound_out / max(abs(sound_out));
end