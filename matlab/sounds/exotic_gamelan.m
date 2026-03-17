
% 3. Exotic gamelan (metallic, shimmering, Indonesian)
function sound_out = exotic_gamelan(freq, fs, dur)
    t = 0:1/fs:dur;
    
    % Quick attack
    attack_samples = round(0.003 * fs);
    attack_env = ones(size(t));
    if attack_samples > 0
        attack_env(1:attack_samples) = linspace(0, 1, attack_samples).^0.5;
    end
    
    % Inharmonic partials (characteristic of gamelan)
    sound_out = sin(2*pi*freq*t) .* exp(-1.5*t) + ...
                0.6*sin(2*pi*freq*2.1*t) .* exp(-1.8*t) + ...
                0.4*sin(2*pi*freq*3.3*t) .* exp(-2.1*t) + ...
                0.3*sin(2*pi*freq*4.7*t) .* exp(-2.4*t) + ...
                0.2*sin(2*pi*freq*5.9*t) .* exp(-2.7*t);
    
    % Add beating for shimmer
    beating = 1 + 0.1 * sin(2*pi*7*t);
    sound_out = sound_out .* beating;
    
    sound_out = sound_out .* attack_env;
    
    fade_samples = min(round(0.02 * fs), length(sound_out));
    fade_out = ones(size(sound_out));
    fade_out(end-fade_samples+1:end) = linspace(1, 0, fade_samples);
    sound_out = sound_out .* fade_out;
    
    sound_out = sound_out / max(abs(sound_out));
end
