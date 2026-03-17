function compare_tunings(ji_ratios, tet)
    % COMPARE_TUNINGS Show cents deviation between JI ratios and TET approximations
    %
    % Displays a table comparing each just intonation interval to its
    % nearest approximation in the given TET system.
    %
    % Inputs:
    %   ji_ratios: array of JI frequency ratios (e.g., [1, 9/8, 5/4, 4/3, 3/2, 5/3, 15/8])
    %   tet: TET system to compare against (e.g., 12, 19, 31, 53)
    %
    % Examples:
    %   ji = microtonal.get_ji_scales();
    %   microtonal.compare_tunings(ji.major_5limit, 12);
    %   microtonal.compare_tunings(ji.major_5limit, 31);
    %   microtonal.compare_tunings(ji.major_7limit, 53);

    fprintf('\n=== JI vs %d-TET Comparison ===\n\n', tet);
    fprintf('%-12s  %-10s  %-10s  %-10s  %-10s\n', ...
        'Ratio', 'JI (cents)', 'TET step', 'TET (cents)', 'Deviation');
    fprintf('%s\n', repmat('-', 1, 58));

    total_deviation = 0;

    for i = 1:length(ji_ratios)
        ratio = ji_ratios(i);
        ji_cents = microtonal.scales.cents(ratio);

        % Find nearest TET step
        tet_step = round(ji_cents / (1200 / tet));
        tet_cents = tet_step * (1200 / tet);
        deviation = ji_cents - tet_cents;

        % Format ratio as fraction if possible
        [num, den] = rat(ratio, 1e-6);
        if den == 1
            ratio_str = sprintf('%d', num);
        else
            ratio_str = sprintf('%d/%d', num, den);
        end

        fprintf('%-12s  %10.2f  %10d  %10.2f  %+10.2f\n', ...
            ratio_str, ji_cents, tet_step, tet_cents, deviation);

        total_deviation = total_deviation + abs(deviation);
    end

    avg_deviation = total_deviation / length(ji_ratios);
    max_dev = 0;
    for i = 1:length(ji_ratios)
        ji_c = microtonal.scales.cents(ji_ratios(i));
        tet_step = round(ji_c / (1200 / tet));
        tet_c = tet_step * (1200 / tet);
        dev = abs(ji_c - tet_c);
        if dev > max_dev
            max_dev = dev;
        end
    end

    fprintf('%s\n', repmat('-', 1, 58));
    fprintf('Average deviation: %.2f cents\n', avg_deviation);
    fprintf('Maximum deviation: %.2f cents\n', max_dev);
    fprintf('(Just noticeable difference is ~5-6 cents)\n\n');
end
