function format_score(filename)
    % FORMAT_SCORE Validate and format a text notation score for readability
    %
    % Reads a score file, validates it, and rewrites it with aligned pipes
    % so that measure boundaries line up across all voices.
    %
    % The original file structure (comments, blank lines, spacing) is
    % preserved exactly. Only notation lines (note/rest data) are replaced
    % with their pipe-aligned equivalents.
    %
    % Input:
    %   filename: path to a .txt notation file
    %
    % Example:
    %   microtonal.format_score('scores/Xenakis_SixChansons.txt');

    fid = fopen(filename, 'r');
    if fid == -1
        error('Could not open file: %s', filename);
    end
    file_content = fread(fid, '*char')';
    fclose(fid);

    original_lines = regexp(file_content, '\r?\n', 'split');

    % Build a comment-stripped copy for parsing.
    % Comment lines become blank so they act as block separators without
    % affecting structure detection.
    parse_lines = original_lines;
    for i = 1:length(parse_lines)
        if startsWith(strtrim(parse_lines{i}), '#')
            parse_lines{i} = '';
        end
    end

    % Scan forward past the header to find voice: declarations
    voice_meta_lines = {};
    idx = 1;
    while idx <= length(parse_lines)
        trimmed = strtrim(parse_lines{idx});
        if startsWith(trimmed, 'voice:') || startsWith(trimmed, 'qtr_note')
            break;
        end
        idx = idx + 1;
    end

    % Collect voice: lines (ignore blank / tuning lines here — only need count)
    while idx <= length(parse_lines)
        trimmed = strtrim(parse_lines{idx});
        if startsWith(trimmed, 'voice:')
            voice_meta_lines{end+1} = trimmed;
            idx = idx + 1;
        elseif isempty(trimmed) || startsWith(trimmed, 'tuning:')
            idx = idx + 1;
        else
            break;
        end
    end

    num_voices = length(voice_meta_lines);
    if num_voices == 0
        error('No voice: lines found in score');
    end

    fprintf('Score has %d voices\n', num_voices);

    % line_replacements maps original line index (1-based) -> formatted content.
    % Only notation lines will have entries here; everything else is written
    % back verbatim so all whitespace and comment spacing is preserved.
    line_replacements = containers.Map('KeyType', 'int32', 'ValueType', 'char');

    % Parse sections using parse_lines, collecting notation blocks and
    % tracking original line indices so we can do targeted replacement.
    while idx <= length(parse_lines)
        % Skip blank lines
        while idx <= length(parse_lines) && isempty(strtrim(parse_lines{idx}))
            idx = idx + 1;
        end
        if idx > length(parse_lines)
            break;
        end

        % Expect tempo line
        trimmed = strtrim(parse_lines{idx});
        tempo_match = regexp(trimmed, 'qtr_note\s*=\s*\d+', 'once');
        if isempty(tempo_match)
            idx = idx + 1;
            continue;
        end
        tempo_line = trimmed;
        idx = idx + 1;

        % Skip blank lines before key
        while idx <= length(parse_lines) && isempty(strtrim(parse_lines{idx}))
            idx = idx + 1;
        end

        % Key line
        key_line = strtrim(parse_lines{idx});
        idx = idx + 1;

        % Collect notation blocks: each is a contiguous run of num_voices
        % non-blank lines, separated by blank lines or comment lines (which
        % parse_lines has already blanked out).
        notation_blocks = {};
        current_block_lines   = {};
        current_block_indices = [];

        while idx <= length(parse_lines)
            trimmed = strtrim(parse_lines{idx});

            % Stop at the start of the next section
            if ~isempty(regexp(trimmed, 'qtr_note\s*=\s*\d+', 'once'))
                if ~isempty(current_block_lines)
                    notation_blocks{end+1} = make_block(current_block_lines, current_block_indices);
                    current_block_lines   = {};
                    current_block_indices = [];
                end
                break;
            end

            if isempty(trimmed)
                % Blank line (or original comment line): flush any open block
                if ~isempty(current_block_lines)
                    notation_blocks{end+1} = make_block(current_block_lines, current_block_indices);
                    current_block_lines   = {};
                    current_block_indices = [];
                end
            else
                current_block_lines{end+1}   = trimmed;
                current_block_indices(end+1) = idx;
            end

            idx = idx + 1;
        end

        % Flush final block
        if ~isempty(current_block_lines)
            notation_blocks{end+1} = make_block(current_block_lines, current_block_indices);
        end

        % Validate and format each notation block
        total_measures = 0;

        for b = 1:length(notation_blocks)
            block = notation_blocks{b};
            block_lines = block.lines;

            if length(block_lines) ~= num_voices
                error('Block has %d lines but expected %d voices', length(block_lines), num_voices);
            end

            % Split each voice line into measures
            voice_measures        = cell(num_voices, 1);
            num_measures_in_block = -1;

            for v = 1:num_voices
                raw_measures = strsplit(block_lines{v}, '|');
                cleaned = {};
                for m = 1:length(raw_measures)
                    txt = strtrim(raw_measures{m});
                    if isempty(txt), continue; end
                    if txt(end) == ','
                        txt = strtrim(txt(1:end-1));
                    end
                    cleaned{end+1} = txt;
                end
                voice_measures{v} = cleaned;

                if num_measures_in_block == -1
                    num_measures_in_block = length(cleaned);
                elseif length(cleaned) ~= num_measures_in_block
                    error('Voice %d has %d measures but voice 1 has %d in block', ...
                        v, length(cleaned), num_measures_in_block);
                end
            end

            total_measures = total_measures + num_measures_in_block;

            % Find max width for each measure column
            col_widths = zeros(1, num_measures_in_block);
            for m = 1:num_measures_in_block
                for v = 1:num_voices
                    w = length(voice_measures{v}{m});
                    if w > col_widths(m)
                        col_widths(m) = w;
                    end
                end
            end

            % Build formatted lines and record replacements by original index
            for v = 1:num_voices
                parts = cell(1, num_measures_in_block);
                for m = 1:num_measures_in_block
                    txt = voice_measures{v}{m};
                    if m < num_measures_in_block
                        parts{m} = sprintf('%-*s', col_widths(m) + 1, [txt, ',']);
                    else
                        parts{m} = txt;
                    end
                end
                line_replacements(int32(block.line_indices(v))) = strjoin(parts, ' | ');
            end
        end

        fprintf('Section "%s" at %s: %d measures\n', key_line, tempo_line, total_measures);

        % Validate durations across voices
        validate_block_durations(notation_blocks, num_voices);
    end

    % Write back: original lines verbatim except notation lines, which get
    % their pipe-aligned replacements. All comments, blank lines, and any
    % other spacing the user has added are preserved unchanged.
    fid = fopen(filename, 'w');
    if fid == -1
        error('Could not write to file: %s', filename);
    end

    for i = 1:length(original_lines)
        key = int32(i);
        if isKey(line_replacements, key)
            fprintf(fid, '%s\n', line_replacements(key));
        else
            fprintf(fid, '%s\n', original_lines{i});
        end
    end

    fclose(fid);
    fprintf('Formatted and saved: %s\n', filename);
end


function blk = make_block(lines, line_indices)
    blk.lines        = lines;
    blk.line_indices = line_indices;
end


function validate_block_durations(blocks, num_voices)
    measure_idx = 0;
    for b = 1:length(blocks)
        block = blocks{b}.lines;
        raw_measures_v1 = strsplit(block{1}, '|');

        for m = 1:length(raw_measures_v1)
            txt = strtrim(raw_measures_v1{m});
            if isempty(txt), continue; end
            measure_idx = measure_idx + 1;

            dur1 = sum_measure_duration(txt);

            for v = 2:num_voices
                v_measures = strsplit(block{v}, '|');
                if m > length(v_measures), continue; end
                v_txt = strtrim(v_measures{m});
                if isempty(v_txt), continue; end

                dur_v = sum_measure_duration(v_txt);
                if abs(dur_v - dur1) > 0.01
                    fprintf('WARNING: Measure %d - voice 1 has %.1f eighths, voice %d has %.1f eighths\n', ...
                        measure_idx, dur1, v, dur_v);
                end
            end
        end
    end
end


function dur = sum_measure_duration(measure_text)
    if measure_text(end) == ','
        measure_text = measure_text(1:end-1);
    end
    tokens = strsplit(measure_text, ',');
    dur = 0;
    for i = 1:length(tokens)
        tok = strtrim(tokens{i});
        if isempty(tok), continue; end
        parts = strsplit(tok, '.');
        if length(parts) == 2
            dur = dur + str2double(parts{2});
        end
    end
end
