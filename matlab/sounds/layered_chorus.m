
% 5. Layered chorus (multiple detuned versions, thick but pure)
function sound_out = layered_chorus(freq, fs, dur)
    t = 0:1/fs:dur;
    
    attack_samples = round(0.007 * fs);
    attack_env = ones(size(t));
    if attack_samples > 0
        attack_env(1:attack_samples) = (1 - cos(pi*linspace(0,1,attack_samples)))/2;
    end
    
    % Multiple slightly detuned voices
    detune = [1.0, 0.995, 1.005, 0.998, 1.002];
    sound_out = zeros(size(t));
    
    for d = 1:length(detune)
        f = freq * detune(d);
        voice = sin(2*pi*f*t) .* exp(-1.5*t) + ...
                0.5*sin(2*pi*f*2*t) .* exp(-1.8*t) + ...
                0.3*sin(2*pi*f*3*t) .* exp(-2.1*t) + ...
                0.2*sin(2*pi*f*4*t) .* exp(-2.4*t) + ...
                0.15*sin(2*pi*f*5*t) .* exp(-2.7*t);
        sound_out = sound_out + voice / length(detune);
    end
    
    sound_out = sound_out .* attack_env;
    
    fade_samples = min(round(0.02 * fs), length(sound_out));
    fade_out = ones(size(sound_out));
    fade_out(end-fade_samples+1:end) = linspace(1, 0, fade_samples);
    sound_out = sound_out .* fade_out;
    
    sound_out = sound_out / max(abs(sound_out));
end
