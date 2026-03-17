function audio_buffer = notation_to_audio(filename, tet)
    % NOTATION_TO_AUDIO Parse a text notation file and generate audio
    %
    % Sound functions are specified per-voice in the score file via
    % voice: meta lines (e.g., "voice: soprano, @piano_sound, +1").
    % Tempo and key can change between sections within the score.
    %
    % Inputs:
    %   filename: path to .txt notation file
    %   tet: (optional) TET system to use (default: 12)
    %
    % Output:
    %   audio_buffer: synthesized audio ready for microtonal.play()
    %
    % Examples:
    %   buf = microtonal.notation_to_audio('scores/Xenakis_SixChansons.txt');
    %   buf = microtonal.notation_to_audio('my_piece.txt', 31);
    %   microtonal.play('my_piece', buf);

    if nargin < 2
        tet = 12;
    end

    [sections, voice_info, metadata] = microtonal.notation.parse_notation(filename);

    % Resolve sound functions from voice: meta lines
    num_voices = length(voice_info);
    sound_funcs = cell(1, num_voices);
    for v = 1:num_voices
        if ~isempty(voice_info(v).sound_func)
            sound_funcs{v} = str2func(voice_info(v).sound_func);
            fprintf('Voice %d (%s): using @%s\n', v, voice_info(v).name, voice_info(v).sound_func);
        else
            error('Voice %d (%s) has no sound function. Add a voice: line to the score.', ...
                v, voice_info(v).name);
        end
    end

    octave_shifts = [voice_info.octave_shift];

    % Generate audio across all sections
    all_notes = [];
    all_times = [];
    all_durations = [];
    all_sound_idxs = [];  % index into sound_funcs
    current_time = 0;

    NUM_SCALE_OCTAVES = 6;
    BASE_SCALE_INDEX = 22;

    for s = 1:length(sections)
        sec = sections(s);

        % Build scales for this section's key
        base_freq = microtonal.scales.note_to_freq(sec.tonic);

        % --- Scale resolution: named mode or custom bracket notation ------
        % Resolves scale_pattern (TET step offsets or JI ratios) and use_ratios.
        if ~isempty(sec.scale_steps)
            % Custom scale written inline in the key line, e.g. "Ab [1,3,5,6,8,10,12]"
            scale_pattern = sec.scale_steps;
            use_ratios = strcmp(sec.mode, 'custom_ji');
        elseif strcmpi(metadata.tuning, 'tet')
            scale_pattern = microtonal.scales.get_mode(tet, sec.mode);
            use_ratios = false;
        else
            % JI named mode — resolve scale name from tuning: line
            ji_scales = microtonal.scales.get_ji_scales();
            if strcmpi(metadata.tuning, 'ji')
                ji_mode_map = struct('major', 'major_5limit', 'minor', 'minor_5limit');
                if ~isfield(ji_mode_map, sec.mode)
                    error('No default JI mapping for mode "%s". Use a specific name (e.g. tuning: pythagorean_major).', sec.mode);
                end
                ji_name = ji_mode_map.(sec.mode);
            else
                ji_name = metadata.tuning;
            end
            if ~isfield(ji_scales, ji_name)
                error('Unknown JI scale "%s". Available: %s', ji_name, strjoin(fieldnames(ji_scales), ', '));
            end
            scale_pattern = ji_scales.(ji_name);
            use_ratios = true;
        end

        % --- Frequency array construction: TET steps or JI ratios --------
        if use_ratios
            ji_chromatic = [1, 16/15, 9/8, 6/5, 5/4, 4/3, 45/32, 3/2, 8/5, 5/3, 9/5, 15/8];
            chromatic_scale = microtonal.scales.ratio_scale(base_freq, ji_chromatic, NUM_SCALE_OCTAVES);
            diatonic_scale  = microtonal.scales.ratio_scale(base_freq, scale_pattern, NUM_SCALE_OCTAVES);
        else
            chromatic_scale = microtonal.scales.tet_scales(base_freq, tet, 0:(tet-1), NUM_SCALE_OCTAVES);
            diatonic_scale  = microtonal.scales.tet_scales(base_freq, tet, scale_pattern, NUM_SCALE_OCTAVES);
        end

        eighth_note_duration = sec.eighth_note_duration;
        num_measures = length(sec.voices{1});

        for m = 1:num_measures
            measure_start_time = current_time;

            % Pre-compute measure bounds (needed for sustain duration extension)
            measure_duration = sum(sec.voices{1}{m}(:, 2), 'omitnan') * eighth_note_duration;
            measure_end_time = measure_start_time + measure_duration;

            for v = 1:num_voices
                measure_notes = sec.voices{v}{m};
                voice_time = measure_start_time;

                for n = 1:size(measure_notes, 1)
                    degree = measure_notes(n, 1);
                    duration_eighths = measure_notes(n, 2);
                    accidental = measure_notes(n, 3);
                    sustain = size(measure_notes, 2) >= 4 && measure_notes(n, 4);
                    notated_duration_sec = duration_eighths * eighth_note_duration;

                    if ~isnan(degree)
                        if sustain
                            audio_duration_sec = measure_end_time - voice_time;
                        else
                            audio_duration_sec = notated_duration_sec;
                        end

                        freq = calculate_frequency_with_accidental(...
                            degree, accidental, octave_shifts(v), ...
                            chromatic_scale, diatonic_scale, scale_pattern, ...
                            BASE_SCALE_INDEX, tet);

                        all_notes = [all_notes, freq];
                        all_times = [all_times, voice_time];
                        all_durations = [all_durations, audio_duration_sec];
                        all_sound_idxs = [all_sound_idxs, v];
                    end
                    voice_time = voice_time + notated_duration_sec;
                end
            end

            current_time = current_time + measure_duration;
        end
    end

    fprintf('Generating audio with %d notes across %d sections...\n', length(all_notes), length(sections));

    % Check if all voices use the same sound function
    unique_funcs = unique(all_sound_idxs);
    all_same = length(unique_funcs) == 1;

    if all_same
        audio_buffer = microtonal.audio.build_audio_buffer(all_notes, all_times, all_durations, sound_funcs{unique_funcs(1)});
    else
        % Multiple sound functions — build buffer manually
        fs = 44100;
        total_duration = max(all_times + all_durations);
        audio_buffer = zeros(1, round(total_duration * fs));

        fade_len = round(0.02 * fs);  % 20ms fade-in/out to prevent clicks
        for i = 1:length(all_notes)
            note_sound = sound_funcs{all_sound_idxs(i)}(all_notes(i), fs, all_durations(i));

            % Apply fade-in and fade-out envelope
            n_samp = length(note_sound);
            fl = min(fade_len, floor(n_samp / 2));
            envelope = ones(1, n_samp);
            envelope(1:fl) = linspace(0, 1, fl);
            envelope(end-fl+1:end) = linspace(1, 0, fl);
            note_sound = note_sound .* envelope;

            start_sample = round(all_times(i) * fs) + 1;
            end_sample = start_sample + n_samp - 1;

            if end_sample > length(audio_buffer)
                audio_buffer(end+1:end_sample) = 0;
            end
            audio_buffer(start_sample:end_sample) = audio_buffer(start_sample:end_sample) + note_sound;
        end

        % Normalize
        peak = max(abs(audio_buffer));
        if peak > 0
            audio_buffer = audio_buffer / peak * 0.9;
        end
    end
end


function freq = calculate_frequency_with_accidental(degree, accidental, octave_shift, ...
                                                     chromatic_scale, diatonic_scale, ...
                                                     scale_pattern, base_index, tet)
    num_scale_degrees = length(scale_pattern);
    diatonic_index = base_index + round(degree) - 1 + (octave_shift * num_scale_degrees);

    if diatonic_index < 1
        error("Not enough notes in scale to go low enough for what's notated");
    end
    if diatonic_index > length(diatonic_scale)
        error("Not enough notes in scale to go high enough for what's notated");
    end

    natural_freq = diatonic_scale(diatonic_index);

    if accidental == 0
        freq = natural_freq;
        return;
    end

    [~, chromatic_idx] = min(abs(chromatic_scale - natural_freq));
    chromatic_idx = chromatic_idx + accidental;

    if chromatic_idx < 1 || chromatic_idx > length(chromatic_scale)
        error("Accidental moves note outside available range");
    end

    freq = chromatic_scale(chromatic_idx);
end
