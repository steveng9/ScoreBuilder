function play(name, audio_buffer, fs)
    % PLAY Save and play an audio buffer
    %
    % Inputs:
    %   name: filename (without extension) to save under audio_files/
    %   audio_buffer: audio data to play
    %   fs: sample rate in Hz (default: 44100)

    if nargin < 3
        fs = 44100;
    end

    audiowrite("audio_files/" + name + '.wav', audio_buffer, fs);

    player = audioplayer(audio_buffer, fs);
    playblocking(player);
    disp('Audio playback finished.');
end