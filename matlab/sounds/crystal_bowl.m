
% 7. Unusual crystal bowl (ethereal, haunting)
function sound_out = crystal_bowl(freq, fs, dur)
    t = 0:1/fs:dur;
    
    % Very slow attack for bowed effect
    attack_samples = round(0.02 * fs);
    attack_env = ones(size(t));
    if attack_samples > 0
        attack_env(1:attack_samples) = (1 - cos(pi*linspace(0,1,attack_samples)))/2;
    end
    
    % Very pure with slow decay and detuned harmonics
    sound_out = sin(2*pi*freq*t) .* exp(-0.8*t) + ...
                0.5*sin(2*pi*freq*2.01*t) .* exp(-1*t) + ...
                0.3*sin(2*pi*freq*3.02*t) .* exp(-1.2*t) + ...
                0.2*sin(2*pi*freq*4.03*t) .* exp(-1.4*t);
    
    % Add slow pulsing
    pulse = 1 + 0.12 * sin(2*pi*4*t);
    sound_out = sound_out .* pulse;
    
    sound_out = sound_out .* attack_env;
    
    fade_samples = min(round(0.02 * fs), length(sound_out));
    fade_out = ones(size(sound_out));
    fade_out(end-fade_samples+1:end) = linspace(1, 0, fade_samples);
    sound_out = sound_out .* fade_out;
    
    sound_out = sound_out / max(abs(sound_out));
end