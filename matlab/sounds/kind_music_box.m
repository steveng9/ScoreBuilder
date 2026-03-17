
% 2. Kind music box (delicate, nostalgic)
function sound_out = kind_music_box(freq, fs, dur)
    t = 0:1/fs:dur;
    
    % Very gentle attack
    attack_samples = round(0.01 * fs);
    attack_env = ones(size(t));
    if attack_samples > 0
        attack_env(1:attack_samples) = (1 - cos(pi*linspace(0,1,attack_samples)))/2;
    end
    
    % Sweet, pure tone with gentle harmonics
    sound_out = sin(2*pi*freq*t) .* exp(-1.2*t) + ...
                0.3*sin(2*pi*freq*2*t) .* exp(-1.5*t) + ...
                0.15*sin(2*pi*freq*4*t) .* exp(-1.8*t);
    
    % Add slight vibrato for warmth
    vibrato = 1 + 0.01 * sin(2*pi*5*t);
    sound_out = sound_out .* vibrato;
    
    sound_out = sound_out .* attack_env;
    
    fade_samples = min(round(0.02 * fs), length(sound_out));
    fade_out = ones(size(sound_out));
    fade_out(end-fade_samples+1:end) = linspace(1, 0, fade_samples);
    sound_out = sound_out .* fade_out;
    
    sound_out = sound_out / max(abs(sound_out));
end