
% 4. Inharmonic complex (non-integer ratios, exotic but pure)
function sound_out = inharmonic_complex(freq, fs, dur)
    t = 0:1/fs:dur;
    
    attack_samples = round(0.005 * fs);
    attack_env = ones(size(t));
    if attack_samples > 0
        attack_env(1:attack_samples) = (1 - cos(pi*linspace(0,1,attack_samples)))/2;
    end
    
    % Non-harmonic partials (like metallic instruments)
    sound_out = 1.0*sin(2*pi*freq*1.0*t) .* exp(-1.3*t) + ...
                0.7*sin(2*pi*freq*2.13*t) .* exp(-1.6*t) + ...
                0.5*sin(2*pi*freq*3.41*t) .* exp(-1.9*t) + ...
                0.4*sin(2*pi*freq*4.77*t) .* exp(-2.2*t) + ...
                0.3*sin(2*pi*freq*6.21*t) .* exp(-2.5*t) + ...
                0.25*sin(2*pi*freq*7.59*t) .* exp(-2.8*t) + ...
                0.2*sin(2*pi*freq*9.13*t) .* exp(-3.1*t);
    
    % Add beating between close frequencies
    beating = 1 + 0.1 * sin(2*pi*6*t);
    sound_out = sound_out .* beating;
    
    sound_out = sound_out .* attack_env;
    
    fade_samples = min(round(0.02 * fs), length(sound_out));
    fade_out = ones(size(sound_out));
    fade_out(end-fade_samples+1:end) = linspace(1, 0, fade_samples);
    sound_out = sound_out .* fade_out;
    
    sound_out = sound_out / max(abs(sound_out));
end