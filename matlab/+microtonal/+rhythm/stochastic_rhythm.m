function [start_times, durations] = stochastic_rhythm(n_notes, tempo, method)
    % STOCHASTIC_RHYTHM Generate rhythmic patterns algorithmically
    %
    % Inputs:
    %   n_notes: number of notes to generate
    %   tempo: quarter note BPM (default: 120)
    %   method: rhythm generation method (default: 'uniform')
    %     'uniform'    - evenly spaced notes
    %     'poisson'    - Poisson process (random, naturalistic spacing)
    %     'euclidean'  - Euclidean rhythm (maximally even distribution)
    %     'fibonacci'  - durations follow Fibonacci ratios
    %     'accelerando' - gradually accelerating tempo
    %     'lcm'        - polyrhythmic pattern based on LCM relationships
    %
    % Outputs:
    %   start_times: array of note onset times in seconds
    %   durations: array of note durations in seconds
    %
    % Examples:
    %   [t, d] = microtonal.stochastic_rhythm(16, 120, 'uniform');
    %   [t, d] = microtonal.stochastic_rhythm(20, 90, 'poisson');
    %   [t, d] = microtonal.stochastic_rhythm(12, 100, 'euclidean');

    if nargin < 2, tempo = 120; end
    if nargin < 3, method = 'uniform'; end

    beat_duration = 60 / tempo;  % quarter note in seconds

    switch lower(method)
        case 'uniform'
            % Evenly spaced
            start_times = (0:n_notes-1) * beat_duration;
            durations = ones(1, n_notes) * beat_duration;

        case 'poisson'
            % Poisson process — exponentially distributed inter-onset intervals
            % Mean interval is one beat
            intervals = -beat_duration * log(rand(1, n_notes));
            start_times = [0, cumsum(intervals(1:end-1))];
            durations = intervals * 1.2;  % Slightly overlapping

        case 'euclidean'
            % Euclidean rhythm: distribute n_notes as evenly as possible
            % across a grid of 2*n_notes slots
            num_slots = n_notes * 2;
            slot_duration = beat_duration / 2;
            pattern = euclidean_pattern(n_notes, num_slots);
            onsets = find(pattern) - 1;
            start_times = onsets * slot_duration;
            % Duration extends to next onset
            durations = zeros(1, length(onsets));
            for i = 1:length(onsets)-1
                durations(i) = (onsets(i+1) - onsets(i)) * slot_duration;
            end
            durations(end) = (num_slots - onsets(end)) * slot_duration;

        case 'fibonacci'
            % Durations follow Fibonacci sequence ratios
            fib = [1, 1];
            while length(fib) < n_notes
                fib(end+1) = fib(end) + fib(end-1);
            end
            fib = fib(1:n_notes);
            % Normalize so average duration is one beat
            fib_normalized = fib / mean(fib) * beat_duration;
            durations = fib_normalized;
            start_times = [0, cumsum(durations(1:end-1))];

        case 'accelerando'
            % Gradually accelerating (shorter intervals over time)
            % Starts at 2x beat duration, ends at 0.25x
            ratios = linspace(2, 0.25, n_notes);
            intervals = ratios * beat_duration;
            durations = intervals * 1.1;
            start_times = [0, cumsum(intervals(1:end-1))];

        case 'lcm'
            % Polyrhythmic: superimpose rhythmic streams at different rates
            % Creates a pattern where notes cluster at LCM points
            rates = [1, 1.5, 2, 3];  % Different subdivision rates
            all_times = [];
            for r = 1:length(rates)
                stream_interval = beat_duration / rates(r);
                stream_times = 0:stream_interval:(n_notes * beat_duration / 2);
                all_times = [all_times, stream_times];
            end
            all_times = unique(sort(all_times));
            % Take first n_notes
            if length(all_times) > n_notes
                all_times = all_times(1:n_notes);
            end
            start_times = all_times;
            n_actual = length(start_times);
            durations = ones(1, n_actual) * beat_duration;

        otherwise
            error('Unknown rhythm method: %s. Use: uniform, poisson, euclidean, fibonacci, accelerando, lcm', method);
    end
end


function pattern = euclidean_pattern(pulses, steps)
    % Bjorklund's algorithm for Euclidean rhythm generation
    if pulses >= steps
        pattern = ones(1, steps);
        return;
    end
    if pulses == 0
        pattern = zeros(1, steps);
        return;
    end

    pattern = zeros(1, steps);
    bucket = 0;
    for i = 1:steps
        bucket = bucket + pulses;
        if bucket >= steps
            bucket = bucket - steps;
            pattern(i) = 1;
        end
    end
end
