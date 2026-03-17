function generate_notation(filename, num_voices, num_measures, key, tempo, standard_measure_length)
    % GENERATE_NOTATION Generate a random composition in text notation format
    %
    % Inputs:
    %   filename: output .txt file path
    %   num_voices: number of voices (default: 4)
    %   num_measures: number of measures (default: 12)
    %   key: key signature string (default: 'C major')
    %   tempo: quarter note BPM (default: 60)
    %   standard_measure_length: eighth notes per measure (default: 6)
    %
    % Example:
    %   microtonal.generate_notation('scores/random1.txt');
    %   microtonal.generate_notation('scores/fast.txt', 3, 8, 'D minor', 120, 8);

    if nargin < 2, num_voices = 4; end
    if nargin < 3, num_measures = 12; end
    if nargin < 4, key = 'C major'; end
    if nargin < 5, tempo = 60; end
    if nargin < 6, standard_measure_length = 6; end

    rng('shuffle');

    % Build file header
    notation = sprintf('Random Composition\n(2026)\ntranscribed by Algorithm\n\n');
    notation = sprintf('%sqtr_note = %d\n%s\n[\n', notation, tempo, key);

    for m = 1:num_measures
        % Vary measure length occasionally
        if rand() < 0.7
            measure_length = standard_measure_length;
        else
            measure_length = randi([4, 8]);
        end

        for v = 1:num_voices
            notation = sprintf('%s    [', notation);

            remaining = measure_length;
            first_note = true;

            while remaining > 0
                if ~first_note
                    notation = sprintf('%s, ', notation);
                end
                first_note = false;

                if rand() < 0.15 && remaining >= 2
                    % Rest
                    duration = min(randi([1, 3]), remaining);
                    notation = sprintf('%sr.%d', notation, duration);
                    remaining = remaining - duration;
                else
                    % Note
                    if rand() < 0.8
                        degree = randi([1, 7]);
                    else
                        degree = randi([-2, 10]);
                    end

                    accidental = '';
                    if rand() < 0.1
                        if rand() < 0.5
                            accidental = 's';
                        else
                            accidental = 'f';
                        end
                    end

                    max_dur = min(4, remaining);
                    duration = randi([1, max_dur]);

                    notation = sprintf('%s%d%s.%d', notation, degree, accidental, duration);
                    remaining = remaining - duration;
                end
            end

            if v < num_voices
                notation = sprintf('%s],\n', notation);
            else
                notation = sprintf('%s]\n', notation);
            end
        end
    end

    notation = sprintf('%s]\n', notation);

    % Save to file
    fid = fopen(filename, 'w');
    if fid == -1
        error('Could not write to file: %s', filename);
    end
    fprintf(fid, '%s', notation);
    fclose(fid);

    fprintf('Saved random composition to %s\n', filename);
end
