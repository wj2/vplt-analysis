load bootsy-11thrun-05-05-2015/dim_pref_look_nolever-bootsy-05-05-2015.mat;
fold = 'bootsy-11thrun-05-05-2015/';
imlg = strcat(fold, 'dim_pref_look_nolever-bootsy-05-05-2015_imglog.txt');

chop = 500;

novRightTrials = [591, 686];
novRightSides = ones(1, length(novRightTrials))*2;
novLeftTrials = [823, 535];
novLeftSides = ones(1, length(novRightTrials))*1;
trials = [novRightTrials, novLeftTrials];
sides = [novRightSides, novLeftSides];
randinds = randperm(length(trials));
trials = trials(randinds);
sides = sides(randinds);
esigs = cell(length(trials), 1);
imgons = zeros(length(trials));
for i = 1:length(trials)
    [eye, imgon] = getEyeDat(bhv, trials(i));
    esigs{i} = eye(chop:end, :);
    imgons(i) = imgon - chop;
end
disp(trials);
plotEye(esigs, sides, 4, 4, 3, 0, imgons);
    