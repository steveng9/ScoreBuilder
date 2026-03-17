function shimmer = magic_shimmer(freq, fs, dur)
    t = 0:1/fs:dur;
    
    % Gentle attack
    attack_samples = round(0.008 * fs);
    attack_env = ones(size(t));
    if attack_samples > 0
        attack_env(1:attack_samples) = (1 - cos(pi*linspace(0,1,attack_samples)))/2;
    end
    
    % Gentle vibrato/tremolo for shimmer
    % More shimmer on lower frequencies
    shimmer_rate = 7 + (440/freq) * 2;  % Slower shimmer for lower notes
    shimmer_depth = 0.2 + (440/freq) * 0.05;  % More depth for lower notes
    shimmer_mod = 1 + shimmer_depth * sin(2*pi*shimmer_rate*t);
    
    % Clear tone with high harmonics for sparkle (but not metallic)
    shimmer = sin(2*pi*freq*t) .* exp(-1.5*t) + ...
              0.4*sin(2*pi*freq*2*t) .* exp(-2*t) + ...
              0.25*sin(2*pi*freq*3*t) .* exp(-2.5*t) + ...
              0.15*sin(2*pi*freq*4*t) .* exp(-3*t) + ...
              0.1*sin(2*pi*freq*5*t) .* exp(-3.5*t);
    
    % Apply shimmer modulation
    shimmer = shimmer .* shimmer_mod;
    shimmer = shimmer .* attack_env;
    
    % Fade out
    fade_samples = min(round(0.02 * fs), length(shimmer));
    fade_out = ones(size(shimmer));
    fade_out(end-fade_samples+1:end) = linspace(1, 0, fade_samples);
    shimmer = shimmer .* fade_out;
    
    shimmer = shimmer / max(abs(shimmer));
end
