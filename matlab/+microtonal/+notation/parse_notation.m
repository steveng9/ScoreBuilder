function [sections, voice_info, metadata] = parse_notation(filename)
    % PARSE_NOTATION Parse a text notation file into sections of voice/note data
    %
    % Reads a .txt file written in the microtonal text notation protocol
    % and returns structured section and voice data ready for audio generation.
    %
    % Input:
    %   filename: path to a .txt notation file
    %
    % Outputs:
    %   sections: struct array, one per tempo/key section, with fields:
    %       .tonic:               tonic note name with octave (e.g., 'C#1')
    %       .mode:                mode name string (e.g., 'major', 'minor')
    %       .eighth_note_duration: duration of one eighth note in seconds
    %       .voices:              cell array of voices, each a cell array of
    %                             measures, each an Nx3 matrix [degree, dur, accidental]
    %   voice_info: struct array with fields:
    %       .name:         voice name (e.g., 'soprano')
    %       .sound_func:   sound function name (e.g., 'crystal_bowl_with_pop')
    %       .octave_shift: octave offset (e.g., +1, 0, -1)
    %
    % =====================================================================
    % TEXT NOTATION PROTOCOL
    % =====================================================================
    %
    % File format:
    %   Title
    %   (Year)
    %   transcribed by Author
    %
    %   voice: <name>, @<sound_func>, <octave_shift>
    %   voice: <name>, @<sound_func>, <octave_shift>
    %   ...
    %
    %   qtr_note = <BPM>
    %   <Key> <major|minor>
    %
    %   <voice 1 measures>  |  <voice 1 measures>  |  ...
    %   <voice 2 measures>  |  <voice 2 measures>  |  ...
    %   ...
    %
    %   <voice 1 continued>  |  ...
    %   <voice 2 continued>  |  ...
    %
    %   qtr_note = <new BPM>
    %   <new Key> <major|minor>
    %
    %   <voice 1 measures>  |  ...
    %   <voice 2 measures>  |  ...
    %
    % Tempo and key are specified before each group of notation lines.
    % They can change between groups to allow modulation and tempo changes.
    %
    % Note format: <degree>[s|f].<duration>
    %   degree:   scale degree (1=tonic, 2=supertonic, etc.)
    %             negative values go below tonic (e.g., -2)
    %             0 = leading tone below tonic
    %   s/f:      optional sharp or flat accidental
    %   duration: in eighth notes (1=eighth, 2=quarter, 4=half, etc.)
    %
    % Rests: r.<duration>
    %
    % Voice meta lines (required, before first tempo/key):
    %   voice: soprano, @crystal_bowl_with_pop, +1
    %   voice: alto, @piano_sound, 0
    %   One line per voice, specifying name, sound function, and octave shift.
    %
    % Structure:
    %   - Each line within a block is one voice
    %   - Measures are separated by | (pipe)
    %   - Blocks of voices are separated by blank lines
    %   - Multiple blocks under the same tempo/key are concatenated
    %   - All voices within a measure must sum to the same duration
    %
    % Example:
    %   3.2, 3.1, 3.1, | 3.3, 2.1  = two measures of voice 1
    %   1.2, 0.2,      | 1.2, 0.4  = two measures of voice 2
    %
    %   7f.1             = degree 7 flat for 1 eighth note
    %   r.4              = rest for 4 eighth notes (half note)
    %   -2.4             = 2 degrees below tonic for 4 eighth notes
    %
    % Lines starting with # are comments.
    % =====================================================================

    REST_MARKER = NaN;
    metadata.tuning = 'tet';  % default; overridden by 'tuning: <name>' in score

    fid = fopen(filename, 'r');
    if fid == -1
        error('Could not open file: %s', filename);
    end

    file_content = fread(fid, '*char')';
    fclose(fid);

    % Strip comment lines (lines beginning with optional whitespace then #).
    % The content is erased but the newline stays, so each comment becomes a
    % blank line and block separators are preserved.
    file_content = regexprep(file_content, '(?m)^\s*#[^\r\n]*', '');
    lines = regexp(file_content, '\r?\n', 'split');

    % Parse voice: and tuning: meta lines (scan from beginning, stop at first qtr_note)
    voice_info = [];
    first_voice_line = -1;
    last_voice_line = -1;
    for i = 1:length(lines)
        trimmed = strtrim(lines{i});
        if ~isempty(regexp(trimmed, 'qtr_note\s*=\s*\d+', 'once'))
            break;  % preamble ends at first tempo line
        end
        if startsWith(trimmed, 'voice:')
            vi = parse_voice_meta(trimmed);
            if isempty(voice_info)
                voice_info = vi;
                first_voice_line = i;
            else
                voice_info(end+1) = vi;
            end
            last_voice_line = i;
            fprintf('Voice: %s, @%s, octave %+d\n', vi.name, vi.sound_func, vi.octave_shift);
        elseif startsWith(trimmed, 'tuning:')
            metadata.tuning = strtrim(trimmed(8:end));
            fprintf('Tuning: %s\n', metadata.tuning);
        end
    end

    if isempty(voice_info)
        error('No voice: lines found in score. Add voice: lines before the first tempo/key.');
    end

    num_voices = length(voice_info);

    % Parse sections: each section starts with qtr_note + key, followed by notation blocks
    % Scan from after the last voice: line
    sections = [];
    idx = last_voice_line + 1;

    while idx <= length(lines)
        % Skip blank lines
        while idx <= length(lines) && isempty(strtrim(lines{idx}))
            idx = idx + 1;
        end
        if idx > length(lines)
            break;
        end

        % Look for tempo line
        trimmed = strtrim(lines{idx});
        tempo_match = regexp(trimmed, 'qtr_note\s*=\s*(\d+)', 'tokens');
        if isempty(tempo_match)
            % Not a tempo line — might be notation that continues (shouldn't happen)
            idx = idx + 1;
            continue;
        end

        qtr_note_bpm = str2double(tempo_match{1}{1});
        eighth_note_duration = 60 / (qtr_note_bpm * 2);
        idx = idx + 1;

        % Skip blank lines before key
        while idx <= length(lines) && isempty(strtrim(lines{idx}))
            idx = idx + 1;
        end

        % Parse key line
        key_text = strtrim(lines{idx});
        [tonic, mode, scale_steps] = parse_key_signature(key_text);
        idx = idx + 1;

        fprintf('\nSection: %s, quarter = %d BPM\n', key_text, qtr_note_bpm);

        % Collect notation blocks until we hit another tempo line or end of file
        notation_blocks = {};
        current_block = {};

        while idx <= length(lines)
            trimmed = strtrim(lines{idx});

            % Check if this is the start of a new section (tempo line)
            if ~isempty(regexp(trimmed, 'qtr_note\s*=\s*\d+', 'once'))
                % Save current block and stop
                if ~isempty(current_block)
                    notation_blocks{end+1} = current_block;
                end
                break;
            end

            if isempty(trimmed)
                if ~isempty(current_block)
                    notation_blocks{end+1} = current_block;
                    current_block = {};
                end
            else
                current_block{end+1} = trimmed;
            end
            idx = idx + 1;
        end

        % Don't forget trailing block at end of file
        if ~isempty(current_block)
            notation_blocks{end+1} = current_block;
        end

        if isempty(notation_blocks)
            continue;
        end

        % Parse notation blocks into voice measures
        section_voices = parse_notation_blocks(notation_blocks, num_voices, REST_MARKER);

        % Build section struct
        sec.tonic = tonic;
        sec.mode = mode;
        sec.scale_steps = scale_steps;  % [] for named modes; array for custom scale
        sec.eighth_note_duration = eighth_note_duration;
        sec.voices = section_voices;

        num_section_measures = length(section_voices{1});
        fprintf('  %d measures\n', num_section_measures);

        if isempty(sections)
            sections = sec;
        else
            sections(end+1) = sec;
        end
    end

    fprintf('\nParsed %d sections, %d voices\n', length(sections), num_voices);

    % Validate measure lengths within each section
    for s = 1:length(sections)
        validate_measure_lengths(sections(s).voices, s);
    end
end


function section_voices = parse_notation_blocks(blocks, num_voices, REST_MARKER)
    % Parse notation blocks into voice cell arrays of measures

    all_voices_per_block = cell(length(blocks), 1);

    for block_idx = 1:length(blocks)
        block_lines = blocks{block_idx};
        block_num_voices = length(block_lines);

        if block_num_voices ~= num_voices
            error('Notation block has %d lines but expected %d voices', ...
                block_num_voices, num_voices);
        end

        voices_this_block = cell(num_voices, 1);

        for v = 1:num_voices
            voice_text = block_lines{v};
            measure_texts = strsplit(voice_text, '|');
            voice_measures = cell(1, length(measure_texts));

            for m = 1:length(measure_texts)
                measure_content = strtrim(measure_texts{m});
                if isempty(measure_content)
                    continue;
                end

                if measure_content(end) == ','
                    measure_content = measure_content(1:end-1);
                end

                notes_text = strsplit(measure_content, ',');
                notes_text = strtrim(notes_text);

                measure_notes = [];
                for n = 1:length(notes_text)
                    note_str = notes_text{n};
                    if isempty(note_str)
                        continue;
                    end

                    if startsWith(note_str, 'r')
                        parts = strsplit(note_str, '.');
                        if length(parts) == 2
                            duration = str2double(parts{2});
                            measure_notes = [measure_notes; REST_MARKER, duration, 0, 0];
                        end
                    else
                        % Check for sustain marker '-' at end of token
                        sustain = 0;
                        if endsWith(note_str, '-')
                            sustain = 1;
                            note_str = note_str(1:end-1);
                        end

                        accidental = 0;
                        clean_note = note_str;

                        if contains(note_str, 's')
                            accidental = 1;
                            clean_note = strrep(note_str, 's', '');
                        elseif contains(note_str, 'f')
                            accidental = -1;
                            clean_note = strrep(note_str, 'f', '');
                        end

                        parts = strsplit(clean_note, '.');
                        if length(parts) == 2
                            degree = str2double(parts{1});
                            duration = str2double(parts{2});
                            measure_notes = [measure_notes; degree, duration, accidental, sustain];
                        end
                    end
                end

                voice_measures{m} = measure_notes;
            end

            voice_measures = voice_measures(~cellfun(@isempty, voice_measures));
            voices_this_block{v} = voice_measures;
        end

        all_voices_per_block{block_idx} = voices_this_block;
    end

    % Concatenate blocks into continuous voice arrays
    section_voices = cell(num_voices, 1);
    for v = 1:num_voices
        all_measures = {};
        for block_idx = 1:length(all_voices_per_block)
            all_measures = [all_measures, all_voices_per_block{block_idx}{v}];
        end
        section_voices{v} = all_measures;
    end
end


function vi = parse_voice_meta(line)
    % Parse a line like: voice: soprano, @crystal_bowl_with_pop, +1
    content = strtrim(line(7:end));  % strip 'voice:'
    parts = strsplit(content, ',');

    vi.name = strtrim(parts{1});

    if length(parts) >= 2
        func_str = strtrim(parts{2});
        if func_str(1) == '@'
            func_str = func_str(2:end);
        end
        vi.sound_func = func_str;
    else
        vi.sound_func = '';
    end

    if length(parts) >= 3
        vi.octave_shift = str2double(strtrim(parts{3}));
    else
        vi.octave_shift = 0;
    end
end


function [tonic, mode, scale_steps] = parse_key_signature(key_text)
    parts = strsplit(key_text);
    tonic_str = parts{1};
    tonic = [tonic_str, '1'];
    scale_steps = [];

    % Check for inline scale definition, e.g.:
    %   Ab [1,3,5,6,8,10,12]        — 1-indexed TET semitone positions
    %   Ab [1/1,9/8,5/4,4/3,3/2,5/3,15/8]  — explicit JI ratios
    bracket = regexp(key_text, '\[([^\]]+)\]', 'tokens', 'once');
    if ~isempty(bracket)
        content = strtrim(bracket{1});
        if contains(content, '/')
            % Ratio notation — parse each token as numerator/denominator
            mode = 'custom_ji';
            tokens = strsplit(content, ',');
            scale_steps = zeros(1, numel(tokens));
            for k = 1:numel(tokens)
                frac = strsplit(strtrim(tokens{k}), '/');
                if numel(frac) == 2
                    scale_steps(k) = str2double(frac{1}) / str2double(frac{2});
                else
                    scale_steps(k) = str2double(frac{1});
                end
            end
        else
            % Integer notation — 1-indexed semitone positions, convert to 0-indexed
            mode = 'custom';
            nums = str2double(strsplit(content, {',', ' '}));
            scale_steps = nums(~isnan(nums)) - 1;
        end
    elseif contains(lower(key_text), 'major')
        mode = 'major';
    elseif contains(lower(key_text), 'minor')
        mode = 'minor';
    else
        mode = 'major';
    end
end


function validate_measure_lengths(voices, section_idx)
    num_measures = length(voices{1});
    num_voices = length(voices);

    for v = 1:num_voices
        if length(voices{v}) ~= num_measures
            error('Section %d: Voice %d has %d measures, but voice 1 has %d measures', ...
                section_idx, v, length(voices{v}), num_measures);
        end
    end

    for m = 1:num_measures
        total_duration = sum(voices{1}{m}(:, 2), 'omitnan');
        for v = 2:num_voices
            voice_duration = sum(voices{v}{m}(:, 2), 'omitnan');
            if abs(voice_duration - total_duration) > 0.01
                error('Section %d, Measure %d: Voice 1 has %.1f eighths, Voice %d has %.1f eighths', ...
                    section_idx, m, total_duration, v, voice_duration);
            end
        end
    end
end
