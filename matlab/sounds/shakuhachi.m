
% Type 6: Shakuhachi (Japanese bamboo flute)
function shaku = shakuhachi(freq, fs, dur)
    t = 0:1/fs:dur;
    
    % Quick but smooth attack
    attack_time = 0.04;
    attack_samples = round(attack_time * fs);
    attack_env = ones(size(t));
    attack_env(1:attack_samples) = (1 - cos(pi*linspace(0,1,attack_samples)))/2;
    
    % Fundamental with strong second harmonic
    shaku = sin(2*pi*freq*t) + ...
            0.5*sin(2*pi*freq*2*t) + ...
            0.2*sin(2*pi*freq*3*t);
    
    % Significant breath noise (characteristic of shakuhachi)
    breath = randn(size(t)) * 0.2;
    breath = filter(ones(1,25)/25, 1, breath);
    
    shaku = shaku + breath;
    
    % Natural decay
    decay_env = exp(-0.6*t);
    
    shaku = shaku .* attack_env .* decay_env;
    shaku = shaku / max(abs(shaku));
end
