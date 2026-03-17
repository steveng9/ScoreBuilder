
% Type 4: Ocarina / ceramic flute
function ocarina = ocarina_sound(freq, fs, dur)
    t = 0:1/fs:dur;
    
    % Medium attack
    attack_time = 0.03;
    attack_samples = round(attack_time * fs);
    attack_env = ones(size(t));
    attack_env(1:attack_samples) = linspace(0, 1, attack_samples);
    
    % Hollow, woody tone
    ocarina = sin(2*pi*freq*t) + ...
              0.4*sin(2*pi*freq*2*t) + ...
              0.2*sin(2*pi*freq*3*t) + ...
              0.1*sin(2*pi*freq*4*t);
    
    % Subtle noise for texture
    texture = randn(size(t)) * 0.05;
    texture = filter(ones(1,15)/15, 1, texture);
    
    ocarina = ocarina + texture;
    
    % Medium decay
    decay_env = exp(-0.8*t);
    
    ocarina = ocarina .* attack_env .* decay_env;
    ocarina = ocarina / max(abs(ocarina));
end
