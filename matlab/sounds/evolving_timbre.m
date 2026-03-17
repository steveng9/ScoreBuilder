
% 6. Evolving timbre (harmonics change over time)
function sound_out = evolving_timbre(freq, fs, dur)
    t = 0:1/fs:dur;
    
    attack_samples = round(0.005 * fs);
    attack_env = ones(size(t));
    if attack_samples > 0
        attack_env(1:attack_samples) = (1 - cos(pi*linspace(0,1,attack_samples)))/2;
    end
    
    % Each harmonic has different envelope shape
    h1 = sin(2*pi*freq*t) .* exp(-1.0*t);
    h2 = sin(2*pi*freq*2*t) .* exp(-1.3*t) .* (1 + 0.5*exp(-3*t));  % Starts strong
    h3 = sin(2*pi*freq*3*t) .* exp(-1.6*t);
    h4 = sin(2*pi*freq*4*t) .* exp(-1.9*t) .* (0.5 + 0.5*exp(-5*t));
    h5 = sin(2*pi*freq*5*t) .* exp(-2.2*t);
    h6 = sin(2*pi*freq*6*t) .* exp(-2.5*t) .* (1 - 0.3*exp(-2*t));  % Grows in
    h7 = sin(2*pi*freq*7*t) .* exp(-2.8*t);
    h8 = sin(2*pi*freq*8*t) .* exp(-3.1*t);
    
    sound_out = h1 + 0.6*h2 + 0.5*h3 + 0.4*h4 + 0.3*h5 + 0.25*h6 + 0.2*h7 + 0.15*h8;
    
    sound_out = sound_out .* attack_env;
    
    fade_samples = min(round(0.02 * fs), length(sound_out));
    fade_out = ones(size(sound_out));
    fade_out(end-fade_samples+1:end) = linspace(1, 0, fade_samples);
    sound_out = sound_out .* fade_out;
    
    sound_out = sound_out / max(abs(sound_out));
end