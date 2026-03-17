% Create a bell-like sound using additive synthesis
fs = 44100;  % Sample rate
duration = 2;  % seconds
t = 0:1/fs:duration;

% Bell has multiple harmonics with exponential decay
frequencies = [440, 880, 1320, 1760];  % Fundamental + harmonics
amplitudes = [1, 0.6, 0.4, 0.2];
decay_rates = [1, 1.5, 2, 2.5];

bell_sound = zeros(size(t));
for i = 1:length(frequencies)
    envelope = amplitudes(i) * exp(-decay_rates(i) * t);
    harmonic = envelope .* sin(2*pi*frequencies(i)*t);
    bell_sound = bell_sound + harmonic;
end

% Play the sound
sound(bell_sound, fs);