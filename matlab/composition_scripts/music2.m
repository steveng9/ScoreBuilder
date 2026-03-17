% Parameters
fs = 44100;  % Sample rate
duration = 2;  % Duration in seconds

% Function to create a bell sound
function bell = create_bell(freq, fs, dur)
    t = 0:1/fs:dur;
    
    % Bell has multiple partials with inharmonic ratios
    % and exponential decay
    bell = 0.5 * sin(2*pi*freq*t) .* exp(-1.5*t) + ...
           0.3 * sin(2*pi*freq*2.4*t) .* exp(-2*t) + ...
           0.2 * sin(2*pi*freq*3.8*t) .* exp(-2.5*t) + ...
           0.1 * sin(2*pi*freq*5.2*t) .* exp(-3*t);
    
    % Normalize
    bell = bell / max(abs(bell));
end

% Create different types of chimes to try:

% Type 1: Bright tubular bells
function bell = tubular_bell(freq, fs, dur)
    t = 0:1/fs:dur;
    bell = sin(2*pi*freq*t) .* exp(-1*t) + ...
           0.5*sin(2*pi*freq*2*t) .* exp(-1.5*t) + ...
           0.3*sin(2*pi*freq*3*t) .* exp(-2*t);
    bell = bell / max(abs(bell));
end

% Type 2: Metallic wind chimes
function chime = wind_chime(freq, fs, dur)
    t = 0:1/fs:dur;
    chime = sin(2*pi*freq*t) .* exp(-3*t) + ...
            0.4*sin(2*pi*freq*2.7*t) .* exp(-3.5*t) + ...
            0.3*sin(2*pi*freq*4.1*t) .* exp(-4*t) + ...
            0.2*sin(2*pi*freq*5.9*t) .* exp(-4.5*t);
    chime = chime / max(abs(chime));
end

% Type 3: Deep church bell
function bell = church_bell(freq, fs, dur)
    t = 0:1/fs:dur;
    bell = sin(2*pi*freq*t) .* exp(-0.5*t) + ...
           0.6*sin(2*pi*freq*1.5*t) .* exp(-0.8*t) + ...
           0.4*sin(2*pi*freq*2.5*t) .* exp(-1*t) + ...
           0.2*sin(2*pi*freq*3.5*t) .* exp(-1.5*t);
    bell = bell / max(abs(bell));
end

% Example: Create a simple melody with different chime types
notes = [261.63, 293.66, 329.63, 349.23, 392.00];  % C, D, E, F, G

% Try Type 1 - Tubular Bells
composition1 = [];
for i = 1:length(notes)
    bell_sound = tubular_bell(notes(i), fs, 1.5);
    composition1 = [composition1, bell_sound];
end
audiowrite('tubular_bells.wav', composition1, fs);
sound(composition1, fs);
pause(length(composition1)/fs + 0.5);

% Try Type 2 - Wind Chimes
composition2 = [];
for i = 1:length(notes)
    chime_sound = wind_chime(notes(i), fs, 1.2);
    composition2 = [composition2, chime_sound];
end
audiowrite('wind_chimes.wav', composition2, fs);
sound(composition2, fs);
pause(length(composition2)/fs + 0.5);

% Try Type 3 - Church Bells
composition3 = [];
for i = 1:length(notes)
    bell_sound = church_bell(notes(i), fs, 2.0);
    composition3 = [composition3, bell_sound];
end
audiowrite('church_bells.wav', composition3, fs);
sound(composition3, fs);