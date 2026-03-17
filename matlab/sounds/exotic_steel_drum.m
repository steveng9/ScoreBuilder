
% 6. Exotic steel drum (Caribbean, bright)
function sound_out = exotic_steel_drum(freq, fs, dur)
    t = 0:1/fs:dur;
    
    % Quick attack
    attack_samples = round(0.004 * fs);
    attack_env = ones(size(t));
    if attack_samples > 0
        attack_env(1:attack_samples) = linspace(0, 1, attack_samples).^0.4;
    end
    
    % Steel drum has unusual harmonic structure
    sound_out = sin(2*pi*freq*t) .* exp(-1.3*t) + ...
                0.6*sin(2*pi*freq*1.5*t) .* exp(-1.6*t) + ...
                0.4*sin(2*pi*freq*2.3*t) .* exp(-1.9*t) + ...
                0.3*sin(2*pi*freq*3.8*t) .* exp(-2.2*t) + ...
                0.2*sin(2*pi*freq*5.1*t) .* exp(-2.5*t);
    
    % Add metallic shimmer
    shimmer = 1 + 0.08 * sin(2*pi*9*t);
    sound_out = sound_out .* shimmer;
    
    sound_out = sound_out .* attack_env;
    
    fade_samples = min(round(0.02 * fs), length(sound_out));
    fade_out = ones(size(sound_out));
    fade_out(end-fade_samples+1:end) = linspace(1, 0, fade_samples);
    sound_out = sound_out .* fade_out;
    
    sound_out = sound_out / max(abs(sound_out));
end
