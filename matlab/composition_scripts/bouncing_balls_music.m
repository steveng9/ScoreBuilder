% ========================================================================
% BOUNCING BALLS MUSICAL SIMULATION (Project JDM Style)
% PRE-GENERATES simulation, then plays back smoothly with synchronized audio
% ========================================================================

function bouncing_balls_music()

    addpath('sounds');
    % Configuration
    n_balls = 14;  % Number of balls (scale degrees)
    box_size = 26;  % Size of the square arena
    simulation_time = 200;  % Seconds to run
    dt = 0.01;  % Time step for physics
    fps = 30;  % Frames per second for playback
    max_vel = 1.5;
    
    % Audio setup
    fs = 44100;
    sound_func = @crystal_bowl_with_pop;  % Change to @magic_shimmer or your other sounds
    
    % Get the musical scale
    tonic = microtonal.scales.note_to_freq("c2");

    scale = microtonal.scales.tet_scales(tonic, 19, 0:4, 3);
    % modes = microtonal.scales.get_12tet_modes();
    % scale = microtonal.scales.tet_scales(tonic, 12, modes.major, 2);
    % modes = microtonal.scales.get_19tet_modes();
    % scale = microtonal.scales.tet_scales(tonic, 19, modes.chromatic, 2);
    % modes = microtonal.scales.get_19tet_modes();
    % scale = microtonal.scales.tet_scales(tonic, 19, modes.exotic1, 2);
    % modes = microtonal.scales.get_31tet_modes();
    % scale = microtonal.scales.tet_scales(tonic, 31, modes.quarter_tone, 2);

    
    % Select n notes from the scale (evenly spaced through 2 octaves)
    note_indices = round(linspace(1, length(scale), n_balls));
    notes = scale(note_indices);
    
    % Initialize balls
    balls = initialize_balls(n_balls, box_size, notes, max_vel);
    
    fprintf('Pre-generating simulation...\n');
    
    % PRE-GENERATE: Run physics and record all states
    num_steps = round(simulation_time / dt);
    frames_per_step = round(dt * fps);
    ball_history = zeros(n_balls, 8, num_steps);
    collision_events = {};  % Store collision times and ball indices
    wall_collision_events = {};  % Store wall collision times
    
    for step = 1:num_steps
        % Update physics
        [balls, collisions, wall_collisions] = update_physics(balls, box_size, dt);
        ball_history(:, :, step) = balls;
        
        % Record ball-ball collision events with timestamp
        if ~isempty(collisions)
            for c = 1:size(collisions, 1)
                event.time = (step - 1) * dt;
                event.ball1 = collisions(c, 1);
                event.ball2 = collisions(c, 2);
                event.freq1 = balls(collisions(c, 1), 7);
                event.freq2 = balls(collisions(c, 2), 7);
                collision_events{end+1} = event;
            end
        end
        
        % Record wall collision events
        if ~isempty(wall_collisions)
            for w = 1:length(wall_collisions)
                wall_event.time = (step - 1) * dt;
                wall_event.ball = wall_collisions(w);
                wall_event.freq = balls(wall_collisions(w), 7);
                wall_collision_events{end+1} = wall_event;
            end
        end
        
        % Progress indicator
        if mod(step, 1000) == 0
            fprintf('  Progress: %.1f%%\n', (step/num_steps)*100);
        end
    end
    
    fprintf('Generating audio...\n');
    
    % PRE-GENERATE AUDIO: Use your build_audio_buffer function
    collision_notes = [];
    collision_times = [];
    collision_durations = [];
    
    % Add ball-ball collision sounds (two notes)
    for i = 1:length(collision_events)
        event = collision_events{i};
        
        % Add both notes from the collision
        collision_notes = [collision_notes, event.freq1, event.freq2];
        collision_times = [collision_times, event.time, event.time];
        collision_durations = [collision_durations, 5, 5];
    end
    
    for i = 1:length(wall_collision_events)
        event = wall_collision_events{i};
        
        collision_notes = [collision_notes, event.freq];
        collision_times = [collision_times, event.time];
        collision_durations = [collision_durations, 5];
    end
    
    % Use your build_audio_buffer function (make sure it's in your path)
    if ~isempty(collision_notes)
        audio_track = microtonal.audio.build_audio_buffer(collision_notes, collision_times, collision_durations, sound_func);
    else
        audio_track = zeros(1, round(simulation_time * fs));
    end
    
    fprintf('Total ball-ball collisions: %d\n', length(collision_events));
    fprintf('Total wall collisions: %d\n', length(wall_collision_events));
    fprintf('Starting playback...\n\n');
    
    % PLAYBACK: Show animation synchronized with audio
    fig = figure('Position', [100, 100, 800, 800]);
    ax = axes('Position', [0.1, 0.1, 0.8, 0.8]);
    hold on;
    axis equal;
    axis([0 box_size 0 box_size]);
    title('Bouncing Balls Music Simulation');
    grid on;
    
    % Color map for balls - soft blues, lavenders, with a hint of chartreuse
    colors = [
        0.7, 0.8, 1.0;    % Soft light blue
        0.6, 0.7, 0.95;   % Periwinkle
        0.75, 0.7, 0.9;   % Lavender
        0.65, 0.75, 1.0;  % Sky blue
        0.8, 0.75, 0.95;  % Pale lavender
        0.7, 0.85, 0.9;   % Powder blue
        0.75, 0.9, 0.7;   % Soft chartreuse
        0.65, 0.8, 0.85;  % Aqua
    ];
    % Repeat colors if more balls than colors
    if n_balls > size(colors, 1)
        colors = repmat(colors, ceil(n_balls/size(colors,1)), 1);
    end
    colors = colors(1:n_balls, :);
    
    % Draw initial balls
    ball_handles = gobjects(n_balls, 1);
    initial_balls = ball_history(:, :, 1);
    for i = 1:n_balls
        ball_handles(i) = draw_ball(initial_balls(i, :), colors(i, :));
    end
    
    % Start audio playback (non-blocking)
    player = audioplayer(audio_track, fs);
    play(player);
    
    % Animate synchronized with audio
    start_time = tic;
    frame = 1;
    step_per_frame = max(1, round(1 / (fps * dt)));
    
    collision_flash = zeros(n_balls, 1);  % Timer for collision flash effect
    
    while toc(start_time) < simulation_time && ishandle(fig)
        current_time = toc(start_time);
        target_step = round(current_time / dt);
        target_step = min(target_step, num_steps);
        
        if target_step >= 1
            current_balls = ball_history(:, :, target_step);
            
            % Check for collisions at this time to trigger visual flash
            for i = 1:length(collision_events)
                event = collision_events{i};
                if abs(event.time - current_time) < dt
                    collision_flash(event.ball1) = 10;  % Flash for 10 frames
                    collision_flash(event.ball2) = 10;
                end
            end
            
            % Update ball positions and visual effects
            for i = 1:n_balls
                update_ball_position(ball_handles(i), current_balls(i, :));
                
                % Flash effect on collision
                if collision_flash(i) > 0
                    set(ball_handles(i), 'EdgeColor', 'r', 'LineWidth', 3);
                    collision_flash(i) = collision_flash(i) - 1;
                else
                    set(ball_handles(i), 'EdgeColor', 'k', 'LineWidth', 1);
                end
            end
            
            drawnow limitrate;
        end
        
        % Frame rate limiting
        pause(1/fps - toc(start_time) + frame/fps);
        frame = frame + 1;
    end
    
    fprintf('Playback complete!\n');
end

function balls = initialize_balls(n, box_size, notes, max_vel)
    % Initialize n balls with random positions and velocities
    % balls: [x, y, vx, vy, radius, mass, note_freq, color_index]
    
    balls = zeros(n, 8);
    
    for i = 1:n
        % Random position (with margin from edges)
        margin = 1;
        balls(i, 1) = margin + rand() * (box_size - 2*margin);  % x
        balls(i, 2) = margin + rand() * (box_size - 2*margin);  % y
        
        % Random velocity
        balls(i, 3) = (rand() - 0.5) * 2 * max_vel;  % vx
        balls(i, 4) = (rand() - 0.5) * 2 * max_vel;  % vy
        
        % Size inversely proportional to pitch (lower = bigger, higher = smaller)
        pitch_factor = notes(end) / notes(i);  % Ratio to lowest note (inverted)
        balls(i, 5) = 0.2 + 0.25 * pitch_factor;  % radius (lower notes = bigger)
        
        % Mass proportional to area (π*r²)
        balls(i, 6) = pi * balls(i, 5)^2;  % mass
        
        % Note frequency
        balls(i, 7) = notes(i);
        
        % Color index
        balls(i, 8) = i;
    end
    
    % Check for initial overlaps and separate
    for i = 1:n
        for j = i+1:n
            while check_overlap(balls(i, :), balls(j, :))
                balls(j, 1) = 1 + rand() * (box_size - 2);
                balls(j, 2) = 1 + rand() * (box_size - 2);
            end
        end
    end
end

function overlap = check_overlap(ball1, ball2)
    dx = ball1(1) - ball2(1);
    dy = ball1(2) - ball2(2);
    dist = sqrt(dx^2 + dy^2);
    overlap = dist < (ball1(5) + ball2(5));
end

function [balls, collisions, wall_collisions] = update_physics(balls, box_size, dt)
    n = size(balls, 1);
    collisions = [];
    wall_collisions = [];
    
    % Update positions
    balls(:, 1) = balls(:, 1) + balls(:, 3) * dt;  % x += vx * dt
    balls(:, 2) = balls(:, 2) + balls(:, 4) * dt;  % y += vy * dt
    
    % Wall collisions
    for i = 1:n
        wall_hit = false;
        
        % Left/Right walls
        if balls(i, 1) - balls(i, 5) < 0
            balls(i, 1) = balls(i, 5);
            balls(i, 3) = -balls(i, 3);
            wall_hit = true;
        elseif balls(i, 1) + balls(i, 5) > box_size
            balls(i, 1) = box_size - balls(i, 5);
            balls(i, 3) = -balls(i, 3);
            wall_hit = true;
        end
        
        % Top/Bottom walls
        if balls(i, 2) - balls(i, 5) < 0
            balls(i, 2) = balls(i, 5);
            balls(i, 4) = -balls(i, 4);
            wall_hit = true;
        elseif balls(i, 2) + balls(i, 5) > box_size
            balls(i, 2) = box_size - balls(i, 5);
            balls(i, 4) = -balls(i, 4);
            wall_hit = true;
        end
        
        % Record wall collision
        if wall_hit
            wall_collisions = [wall_collisions, i];
        end
    end
    
    % Ball-ball collisions (elastic collision with conservation of momentum)
    for i = 1:n
        for j = i+1:n
            dx = balls(j, 1) - balls(i, 1);
            dy = balls(j, 2) - balls(i, 2);
            dist = sqrt(dx^2 + dy^2);
            
            % Check for collision
            if dist < (balls(i, 5) + balls(j, 5)) && dist > 0
                % Record collision
                collisions = [collisions; i, j];
                
                % Normal vector
                nx = dx / dist;
                ny = dy / dist;
                
                % Relative velocity
                dvx = balls(i, 3) - balls(j, 3);
                dvy = balls(i, 4) - balls(j, 4);
                
                % Relative velocity in collision normal direction
                dvn = dvx * nx + dvy * ny;
                
                % Don't process if velocities are separating
                if dvn > 0
                    % Masses
                    m1 = balls(i, 6);
                    m2 = balls(j, 6);
                    
                    % Impulse scalar (elastic collision formula)
                    impulse = 2 * dvn / (m1 + m2);
                    
                    % Update velocities
                    balls(i, 3) = balls(i, 3) - impulse * m2 * nx;
                    balls(i, 4) = balls(i, 4) - impulse * m2 * ny;
                    balls(j, 3) = balls(j, 3) + impulse * m1 * nx;
                    balls(j, 4) = balls(j, 4) + impulse * m1 * ny;
                    
                    % Separate balls to prevent overlap
                    overlap = (balls(i, 5) + balls(j, 5)) - dist;
                    separation = overlap / 2 + 0.01;
                    balls(i, 1) = balls(i, 1) - separation * nx;
                    balls(i, 2) = balls(i, 2) - separation * ny;
                    balls(j, 1) = balls(j, 1) + separation * nx;
                    balls(j, 2) = balls(j, 2) + separation * ny;
                end
            end
        end
    end
end

function h = draw_ball(ball, color)
    % Draw a ball as a filled circle
    theta = linspace(0, 2*pi, 50);
    x = ball(1) + ball(5) * cos(theta);
    y = ball(2) + ball(5) * sin(theta);
    h = fill(x, y, color, 'EdgeColor', 'k', 'LineWidth', 1);
end

function update_ball_position(handle, ball)
    % Update the position of a ball graphic
    theta = linspace(0, 2*pi, 50);
    x = ball(1) + ball(5) * cos(theta);
    y = ball(2) + ball(5) * sin(theta);
    set(handle, 'XData', x, 'YData', y);
end