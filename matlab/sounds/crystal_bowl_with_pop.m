function sound_out = crystal_bowl_with_pop(freq, fs, dur)
    t = 0:1/fs:dur;
    
    % Medium attack - noticeable but gentle
    attack_samples = round(0.001 * fs);  % 8ms
    attack_env = ones(size(t));
    if attack_samples > 0
        % Smooth but quicker curve
        attack_env(1:attack_samples) = (1 - cos(pi*linspace(0, 1, attack_samples)))/2;
    end
    
    % Very pure with slow decay and detuned harmonics (crystal bowl)
    sound_out = sin(2*pi*freq*t) .* exp(-0.8*t) + ...
                0.5*sin(2*pi*freq*2.01*t) .* exp(-1*t) + ...
                0.3*sin(2*pi*freq*3.02*t) .* exp(-1.2*t) + ...
                0.2*sin(2*pi*freq*4.03*t) .* exp(-1.4*t);
    
    % Soft mallet impact - more present but still gentle
    thud_dur = 0.06;  % 15ms
    thud_samples = round(thud_dur * fs);
    if length(sound_out) >= thud_samples
        % Low frequency thump 
        thud_freq = freq * .9;  % Low but not too low
        thud = sin(2*pi*thud_freq*(0:thud_samples-1)/fs) .* exp(-100*(0:thud_samples-1)/fs) * 0.7;
        
        % Soft contact noise with less filtering (more present)
        soft_noise = randn(1, thud_samples) * 0.06;
        soft_noise = filter(ones(1,25)/25, 1, soft_noise);  % Less filtering
        soft_noise = soft_noise .* exp(-70*(0:thud_samples-1)/fs);
        
        sound_out(1:thud_samples) = sound_out(1:thud_samples) + thud + soft_noise;
    end
    
    % Add slow pulsing for shimmer
    pulse = 1 + 0.12 * sin(2*pi*4*t);
    sound_out = sound_out .* pulse;
    
    sound_out = sound_out .* attack_env;
    
    % Fade out
    fade_samples = min(round(0.02 * fs), length(sound_out));
    fade_out = ones(size(sound_out));
    fade_out(end-fade_samples+1:end) = linspace(1, 0, fade_samples);
    sound_out = sound_out .* fade_out;
    
    sound_out = sound_out / max(abs(sound_out));
end