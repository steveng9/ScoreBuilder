
% Type 5: Ethereal / floating pad
function pad = ethereal_pad(freq, fs, dur)
    t = 0:1/fs:dur;
    
    % Very slow attack
    attack_time = 0.15;
    attack_samples = round(attack_time * fs);
    attack_env = ones(size(t));
    attack_env(1:attack_samples) = linspace(0, 1, attack_samples).^3;
    
    % Multiple detuned oscillators for chorus effect
    pad = sin(2*pi*freq*t) + ...
          0.7*sin(2*pi*freq*1.01*t) + ...
          0.7*sin(2*pi*freq*0.99*t) + ...
          0.3*sin(2*pi*freq*2*t);
    
    % Breathy noise
    breath = randn(size(t)) * 0.1;
    breath = filter(ones(1,40)/40, 1, breath);
    
    pad = pad + breath;
    
    % Very slow decay
    decay_env = exp(-0.2*t);
    
    pad = pad .* attack_env .* decay_env;
    pad = pad / max(abs(pad));
end
