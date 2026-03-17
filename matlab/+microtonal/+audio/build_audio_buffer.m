function audio_buffer = build_audio_buffer(notes, start_times, durations, sound_func, fs)
    % BUILD_AUDIO_BUFFER Synthesize a composition from note arrays
    %
    % Inputs:
    %   notes: array of frequencies in Hz
    %   start_times: array of note start times in seconds
    %   durations: array of note durations in seconds
    %   sound_func: handle to a sound synthesis function (e.g., @piano_sound)
    %   fs: sample rate in Hz (default: 44100)

    if nargin < 5
        fs = 44100;
    end

    end_times = (start_times * fs) + (durations * fs);
    audio_buffer = zeros(1, round(max(end_times)) + 1);

    for i = 1:length(notes)
        note_sound = sound_func(notes(i), fs, durations(i));

        % Add fade-out to prevent clipping at the end of each note
        fade_samples = min(round(0.02 * fs), length(note_sound));  % 20ms fade
        fade_out = ones(size(note_sound));
        fade_out(end-fade_samples+1:end) = linspace(1, 0, fade_samples);
        note_sound = note_sound .* fade_out;

        start_sample = round(start_times(i) * fs) + 1;
        end_sample = start_sample + length(note_sound) - 1;

        % Add the note sound to the buffer (overlapping)
        audio_buffer(start_sample:end_sample) = audio_buffer(start_sample:end_sample) + note_sound;
    end

    % Normalize to prevent clipping
    audio_buffer = audio_buffer / max(abs(audio_buffer)) * 0.9;
end