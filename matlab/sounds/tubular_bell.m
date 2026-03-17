% 
% % Type 1: Bright tubular bells
% function bell = tubular_bell(freq, fs, dur)
%     t = 0:1/fs:dur;
%     bell = sin(2*pi*freq*t) .* exp(-1*t) + ...
%            0.5*sin(2*pi*freq*2*t) .* exp(-1.5*t) + ...
%            0.3*sin(2*pi*freq*3*t) .* exp(-2*t);
%     bell = bell / max(abs(bell));
% end
% 

% Type 1: Bright tubular bells (NO CLICKING)
function bell = tubular_bell(freq, fs, dur)
    t = 0:1/fs:dur;
    
    % Add attack envelope to prevent clicking (fade in over first 10ms)
    attack_time = 0.001;  % 10 milliseconds
    attack_samples = round(attack_time * fs);
    attack_env = ones(size(t));
    attack_env(1:attack_samples) = linspace(0, 1, attack_samples);
    
    bell = sin(2*pi*freq*t) .* exp(-1*t) + ...
           0.5*sin(2*pi*freq*2*t) .* exp(-1.5*t) + ...
           0.3*sin(2*pi*freq*3*t) .* exp(-2*t);
    
    % Apply attack envelope
    bell = bell .* attack_env;
    bell = bell / max(abs(bell));
end