
% 4. Unusual water drop (resonant, liquid)
function sound_out = water_drop(freq, fs, dur)
    t = 0:1/fs:dur;
    
    % Very fast attack (like a droplet impact)
    attack_samples = round(0.002 * fs);
    attack_env = ones(size(t));
    if attack_samples > 0
        attack_env(1:attack_samples) = linspace(0, 1, attack_samples).^0.3;
    end
    
    % Unusual frequency ratios for liquid resonance
    sound_out = sin(2*pi*freq*t) .* exp(-3*t) + ...
                0.5*sin(2*pi*freq*1.414*t) .* exp(-3.5*t) + ...
                0.3*sin(2*pi*freq*2.236*t) .* exp(-4*t);
    
    % Add brief splash noise
    splash_dur = 0.01;
    splash_samples = round(splash_dur * fs);
    if length(sound_out) >= splash_samples
        splash = randn(1, splash_samples) * 0.15 .* exp(-200*(0:splash_samples-1)/fs);
        sound_out(1:splash_samples) = sound_out(1:splash_samples) + splash;
    end
    
    sound_out = sound_out .* attack_env;
    
    fade_samples = min(round(0.02 * fs), length(sound_out));
    fade_out = ones(size(sound_out));
    fade_out(end-fade_samples+1:end) = linspace(1, 0, fade_samples);
    sound_out = sound_out .* fade_out;
    
    sound_out = sound_out / max(abs(sound_out));
end
