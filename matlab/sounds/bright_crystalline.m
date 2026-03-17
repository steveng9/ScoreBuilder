
% 3. Bright crystalline (high harmonics, pure sparkle)
function sound_out = bright_crystalline(freq, fs, dur)
    t = 0:1/fs:dur;
    
    attack_samples = round(0.003 * fs);
    attack_env = ones(size(t));
    if attack_samples > 0
        attack_env(1:attack_samples) = linspace(0, 1, attack_samples).^0.5;
    end
    
    % Emphasize higher harmonics for brightness
    sound_out = 0.7*sin(2*pi*freq*t) .* exp(-1.5*t) + ...
                0.6*sin(2*pi*freq*2*t) .* exp(-1.8*t) + ...
                0.5*sin(2*pi*freq*3*t) .* exp(-2.0*t) + ...
                0.5*sin(2*pi*freq*4*t) .* exp(-2.2*t) + ...
                0.4*sin(2*pi*freq*5*t) .* exp(-2.4*t) + ...
                0.4*sin(2*pi*freq*6*t) .* exp(-2.6*t) + ...
                0.3*sin(2*pi*freq*7*t) .* exp(-2.8*t) + ...
                0.3*sin(2*pi*freq*8*t) .* exp(-3.0*t) + ...
                0.2*sin(2*pi*freq*9*t) .* exp(-3.2*t) + ...
                0.2*sin(2*pi*freq*10*t) .* exp(-3.4*t) + ...
                0.15*sin(2*pi*freq*11*t) .* exp(-3.6*t) + ...
                0.15*sin(2*pi*freq*12*t) .* exp(-3.8*t);
    
    % Gentle shimmer
    shimmer = 1 + 0.06 * sin(2*pi*8*t);
    sound_out = sound_out .* shimmer;
    
    sound_out = sound_out .* attack_env;
    
    fade_samples = min(round(0.02 * fs), length(sound_out));
    fade_out = ones(size(sound_out));
    fade_out(end-fade_samples+1:end) = linspace(1, 0, fade_samples);
    sound_out = sound_out .* fade_out;
    
    sound_out = sound_out / max(abs(sound_out));
end
