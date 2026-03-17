% HeathersSong.m — Generate and play "Heather's Song"
% 5-voice crystal bowl arrangement in D major
% Chord progression from Joni Mitchell's "Both Sides Now"
% with extended chords (sus4, add9, 7sus4) and algorithmic melody

rng(7);  % Seed for reproducibility. Change for different versions.

%% Chord definitions (scale degrees in D major)
% D major: 1=D, 2=E, 3=F#, 4=G, 5=A, 6=B, 7=C#
%
% Asus4   = A, D, E       → 5, 1, 2
% Gadd9   = G, B, D, A    → 4, 6, 1, 5
% A7sus4  = A, D, E, G    → 5, 1, 2, 4
% D       = D, F#, A      → 1, 3, 5
% D9      = D, F#, A, E   → 1, 3, 5, 2  (Dadd9)

CT.Asus4  = [5, 1, 2];
CT.Gadd9  = [4, 6, 1, 5];
CT.A7sus4 = [5, 1, 2, 4];
CT.D      = [1, 3, 5];
CT.D9     = [1, 3, 5, 2];

% 4-voice harmony voicings: [alto(0), tenor(0), bari(-1), bass(-2)]
% Chosen for smooth voice leading between adjacent chords
%
%           alto  tenor  bari  bass     Actual notes:
HV.Asus4  = [2,    1,     5,    5];  %  E4   D4   A3   A2
HV.Gadd9  = [5,    1,     6,    4];  %  A4   D4   B3   G2
HV.A7sus4 = [2,    1,     5,    4];  %  E4   D4   A3   G2
HV.D      = [3,    1,     5,    1];  %  F#4  D4   A3   D2
HV.D9     = [3,    2,     5,    1];  %  F#4  E4   A3   D2

%% Chord progression
% Verse:  Asus4 Gadd9 A7sus4 D D | Gadd9 D Asus4 Gadd9 A7sus4 D |
%         Asus4 Gadd9 A7sus4 | Asus4 Gadd9 A7sus4
verse = {'Asus4','Gadd9','A7sus4','D','D', ...
         'Gadd9','D','Asus4','Gadd9','A7sus4','D', ...
         'Asus4','Gadd9','A7sus4', ...
         'Asus4','Gadd9','A7sus4'};

% Chorus: D A7sus4 Gadd9 D9 D | Gadd9 D Gadd9 D |
%         Asus4 Gadd9 A7sus4 D | Asus4 Gadd9 D A7sus4
chorus = {'D','A7sus4','Gadd9','D9','D', ...
          'Gadd9','D','Gadd9','D', ...
          'Asus4','Gadd9','A7sus4','D', ...
          'Asus4','Gadd9','D','A7sus4'};

verse_groups = [5, 6, 3, 3];    % measures per ledger line
chorus_groups = [5, 4, 4, 4];

% Structure: V V C V C C
section_order = {'V','V','C','V','C','C'};

% Build full progression and ledger groupings
progression = {};
line_groups = [];
section_type = [];  % 0=verse, 1=chorus
for s = 1:length(section_order)
    if section_order{s} == 'V'
        progression = [progression, verse];
        line_groups = [line_groups, verse_groups];
        section_type = [section_type, zeros(1, length(verse))];
    else
        progression = [progression, chorus];
        line_groups = [line_groups, chorus_groups];
        section_type = [section_type, ones(1, length(chorus))];
    end
end
total_measures = length(progression);

% Pre-compute section boundaries (for melody shaping)
section_ends = cumsum(cellfun(@(s) ternary(s=='V', length(verse), length(chorus)), section_order));
section_starts = [1, section_ends(1:end-1)+1];

%% Generate all voice measures
num_voices = 5;
all_measures = cell(num_voices, total_measures);
mel_deg = 3;  % melody starts on F#

for m = 1:total_measures
    chord = progression{m};
    tones = CT.(chord);
    harm = HV.(chord);
    is_chorus = section_type(m);

    % Find position within current section
    si = find(m >= section_starts, 1, 'last');
    pos_in_sec = (m - section_starts(si)) / (section_ends(si) - section_starts(si));
    near_end = pos_in_sec > 0.8;
    is_first = (m == section_starts(si));
    is_last_measure = (m == total_measures);

    % --- Harmony voices (v2-v5) ---
    for hv = 1:4
        base = harm(hv);
        r = rand();
        if near_end || r < 0.45
            % Whole note (more common near phrase endings)
            all_measures{hv+1, m} = sprintf('%d.8', base);
        elseif r < 0.62
            % Two halves: move between chord tones
            alt = pick(tones);
            all_measures{hv+1, m} = sprintf('%d.4, %d.4', base, alt);
        elseif r < 0.75
            % Dotted half + quarter to another tone
            alt = pick(tones);
            all_measures{hv+1, m} = sprintf('%d.6, %d.2', base, alt);
        elseif r < 0.87
            % Quarter approach note + dotted half (step into chord tone)
            approach = base + pick([-1, 1]);
            all_measures{hv+1, m} = sprintf('%d.2, %d.6', approach, base);
        else
            % Gentle rearticulation: half, rest, half
            all_measures{hv+1, m} = sprintf('%d.3, r.2, %d.3', base, base);
        end
    end

    % --- Melody voice (v1) ---
    energy = 0.4 + 0.25 * is_chorus + 0.15 * pos_in_sec;
    [mel_str, mel_deg] = gen_melody(mel_deg, tones, energy, ...
                                     is_first, near_end, is_last_measure);
    all_measures{1, m} = mel_str;
end

%% Build score text
score = '';
score = [score, 'Heather''s Song' newline];
score = [score, '(2026)' newline];
score = [score, 'composed by Steven Golob' newline];
score = [score, newline];
score = [score, 'voice: melody, @ocarina_sound, 1' newline];
score = [score, 'voice: alto, @crystal_bowl_with_pop, 0' newline];
score = [score, 'voice: tenor, @crystal_bowl_with_pop, 0' newline];
score = [score, 'voice: baritone, @crystal_bowl_with_pop, -1' newline];
score = [score, 'voice: bass, @crystal_bowl_with_pop, -2' newline];
score = [score, newline];
score = [score, 'qtr_note = 130' newline];
score = [score, 'D major' newline];

% Write measures grouped into ledger lines
m_idx = 1;
for g = 1:length(line_groups)
    n = line_groups(g);
    score = [score, newline];
    for v = 1:num_voices
        line = '';
        for mi = 0:(n-1)
            if mi > 0
                line = [line, ' | '];
            end
            line = [line, all_measures{v, m_idx + mi}];
            if mi < n - 1
                line = [line, ','];
            end
        end
        score = [score, line newline];
    end
    m_idx = m_idx + n;
end

%% Save, format, and play
score_file = 'scores/HeathersSong.txt';
fid = fopen(score_file, 'w');
fprintf(fid, '%s', score);
fclose(fid);
fprintf('Score saved to %s\n', score_file);

microtonal.notation.format_score(score_file);
buf = microtonal.notation.notation_to_audio(score_file);
microtonal.audio.play('HeathersSong', buf);


%% ===== Local Functions =====

function item = pick(arr)
    item = arr(randi(length(arr)));
end

function r = ternary(cond, a, b)
    if cond, r = a; else, r = b; end
end

function [str, end_deg] = gen_melody(prev_deg, chord_tones, energy, ...
                                      is_section_start, near_end, is_last)
    % Generate one measure (8 eighth notes) of melody
    %
    % energy: 0-1, controls rhythmic activity
    % is_section_start: true on first measure of a section
    % near_end: true in last ~20% of a section
    % is_last: true on final measure of the piece

    % --- Choose rhythm pattern (must sum to 8) ---
    calm     = {[8], [4,4], [6,2], [2,6]};
    moderate = {[4,4], [2,2,4], [4,2,2], [3,3,2], [2,4,2]};
    active   = {[2,2,2,2], [1,1,2,4], [4,1,1,2], [2,1,1,2,2], ...
                [3,1,2,2], [1,3,2,2], [2,2,1,1,2]};

    if is_last
        durs = [8];  % resolve on whole note
    elseif is_section_start && rand() < 0.6
        % Sections often start with a rest + pickup
        pickup_patterns = {[4,4], [4,2,2], [6,2], [2,2,4]};
        durs = pickup_patterns{randi(length(pickup_patterns))};
    elseif near_end
        durs = calm{randi(length(calm))};
    else
        r = rand();
        if r < 0.25 - 0.15*energy
            durs = calm{randi(length(calm))};
        elseif r < 0.65
            durs = moderate{randi(length(moderate))};
        else
            durs = active{randi(length(active))};
        end
    end

    % --- Generate notes ---
    deg = prev_deg;
    parts = {};
    rest_placed = false;

    for i = 1:length(durs)
        % Rest logic: sections can start with rest, otherwise occasional
        if is_section_start && i == 1 && rand() < 0.5
            parts{end+1} = sprintf('r.%d', durs(i));
            rest_placed = true;
            continue;
        end
        if ~rest_placed && i > 1 && i < length(durs) && rand() < 0.10
            parts{end+1} = sprintf('r.%d', durs(i));
            rest_placed = true;
            continue;
        end

        % Pick the note
        if i == 1 && ~rest_placed
            % First sounding note: favor chord tones
            if rand() < 0.8
                deg = nearest_tone(deg, chord_tones);
            else
                deg = deg + pick([-1, 0, 1]);
            end
        elseif is_last
            % Final note: land on tonic or 5th
            deg = pick([1, 5]);
        else
            action = rand();
            if action < 0.28
                % Stepwise motion
                deg = deg + pick([-2, -1, 1, 2]);
            elseif action < 0.50
                % To a chord tone
                deg = pick(chord_tones);
            elseif action < 0.65
                % Neighbor tone
                deg = deg + pick([-1, 1]);
            elseif action < 0.78
                % Leap (3rd or 4th)
                deg = deg + pick([-4, -3, 3, 4]);
            elseif action < 0.88
                % Hold previous note
                % deg stays same
            else
                % Leap to a distant chord tone for drama
                ct = pick(chord_tones);
                if abs(ct - deg) < 3
                    ct = ct + pick([-7, 7]);  % octave displacement
                end
                deg = ct;
            end
        end

        % Clamp: melody at octave +1, range ~B4 to D6
        deg = max(-1, min(8, deg));
        parts{end+1} = sprintf('%d.%d', deg, durs(i));
    end

    str = strjoin(parts, ', ');
    end_deg = deg;
end

function nearest = nearest_tone(deg, tones)
    [~, idx] = min(abs(tones - deg));
    nearest = tones(idx);
end
