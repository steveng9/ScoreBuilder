
% Type 3: Whispy / airy whistle
function whistle = whispy_whistle(freq, fs, dur)
    t = 0:1/fs:dur;
    
    % Very gentle attack
    attack_time = 0.08;
    attack_samples = round(attack_time * fs);
    attack_env = ones(size(t));
    attack_env(1:attack_samples) = linspace(0, 1, attack_samples).^2;  % Smooth curve
    
    % Pure tone with slight vibrato
    vibrato_rate = 5;  % Hz
    vibrato_depth = 0.01;  % 1% frequency variation
    vibrato = 1 + vibrato_depth * sin(2*pi*vibrato_rate*t);
    
    whistle = sin(2*pi*freq*vibrato.*t) + ...
              0.2*sin(2*pi*freq*2*vibrato.*t);
    
    % Add air/breath noise
    air_noise = randn(size(t)) * 0.08;
    air_noise = filter(ones(1,30)/30, 1, air_noise);
    
    whistle = whistle + air_noise;
    
    % Slow decay
    decay_env = exp(-0.3*t);
    
    whistle = whistle .* attack_env .* decay_env;
    whistle = whistle / max(abs(abs(whistle)) + 1e-10);  % Avoid divide by zero
end
