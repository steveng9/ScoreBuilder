addpath("sounds/");

% score_file = "scores/Xenakis_SixChansons.txt";
% score_file = "scores/HeathersSong.txt";
% score_file = "scores/first_text_music.txt";
% score_file = "scores/modulation_trial.txt";
% score_file = "scores/gathering_grounds.txt";
% score_file = "scores/serial_generated.txt";
% score_file = "scores/Stravinsky_sonataDuet.txt";
% score_file = "scores/split_complementary.txt";
% score_file = "scores/sonata_in_24.txt";
% score_file = "scores/houses_harmony.txt";
score_file = "scores/wave1.txt";

[~, score_name, ~] = fileparts(score_file);

microtonal.notation.format_score(score_file);
buf = microtonal.notation.notation_to_audio(score_file);
% buf = microtonal.audio.apply_reverb(buf);
microtonal.audio.play(score_name, buf);