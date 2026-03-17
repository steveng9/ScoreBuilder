
% 2. Deep resonant (complex low harmonics, warm)
function sound_out = deep_resonant(freq, fs, dur)
    t = 0:1/fs:dur;
    
    attack_samples = round(0.006 * fs);
    attack_env = ones(size(t));
    if attack_samples > 0
        attack_env(1:attack_samples) = (1 - cos(pi*linspace(0,1,attack_samples)))/2;
    end
    
    % Rich fundamental with slower decay
    sound_out = 1.0*sin(2*pi*freq*t) .* exp(-0.8*t) + ...
                0.8*sin(2*pi*freq*2*t) .* exp(-1.0*t) + ...
                0.6*sin(2*pi*freq*3*t) .* exp(-1.2*t) + ...
                0.5*sin(2*pi*freq*4*t) .* exp(-1.4*t) + ...
                0.4*sin(2*pi*freq*5*t) .* exp(-1.6*t) + ...
                0.3*sin(2*pi*freq*6*t) .* exp(-1.8*t) + ...
                0.25*sin(2*pi*freq*7*t) .* exp(-2.0*t) + ...
                0.2*sin(2*pi*freq*8*t) .* exp(-2.2*t);
    
    % Add subtle slow vibrato for warmth
    vibrato = 1 + 0.008 * sin(2*pi*4*t);
    sound_out = sound_out .* vibrato;
    
    sound_out = sound_out .* attack_env;
    
    fade_samples = min(round(0.02 * fs), length(sound_out));
    fade_out = ones(size(sound_out));
    fade_out(end-fade_samples+1:end) = linspace(1, 0, fade_samples);
    sound_out = sound_out .* fade_out;
    
    sound_out = sound_out / max(abs(sound_out));
end
