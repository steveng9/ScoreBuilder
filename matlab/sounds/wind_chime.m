
% Type 2: Metallic wind chimes
function chime = wind_chime(freq, fs, dur)
    t = 0:1/fs:dur;
    chime = sin(2*pi*freq*t) .* exp(-3*t) + ...
            0.4*sin(2*pi*freq*2.7*t) .* exp(-3.5*t) + ...
            0.3*sin(2*pi*freq*4.1*t) .* exp(-4*t) + ...
            0.2*sin(2*pi*freq*5.9*t) .* exp(-4.5*t);
    chime = chime / max(abs(chime));
end