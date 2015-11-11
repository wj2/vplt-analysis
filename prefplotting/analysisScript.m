%% analysis pref_looking data
stimOffset = 3;
stimDeg = 5.5;
fixThr = 5;

% datafolder = 'bootsy-2ndrun-04-22-2015/';
% matname = 'bootsy-2nd-04-22-2015.mat';

% datafolder = 'bootsy-3rdrun-04-23-2015/';
% matname = 'bootsy-3rd-04-23-2015.mat';

datafolder = 'bootsy-4thrun-04-24-2015/';
matname = 'bootsy-4th-04-24-2015.mat';

% datafolder = 'bootsy-1strun-4-20-15/';
% matname = 'dim_look-bootsy-04-20-15.mat';
load(strcat(datafolder, matname));

% bhvpath = strcat(datafolder, basename, '.mat');
% impath = strcat(datafolder, basename, '_imglog.txt');
relevantConds = [7, 8, 9, 10];

fixOccCode = 8;
trialStartCode = 9;
imgOnCode = 191;
imgOffCode = 192;
trialEndCode = 18;

bhv = bootbhv;
famVNov = [];
novOnly = [];
famOnly = [];
dirLook = [];
fixords = {};
for i = 1:length(bhv.ConditionNumber)
    cond = bhv.ConditionNumber(i);
    starts = bhv.CodeTimes{i}(bhv.CodeNumbers{i} == trialStartCode);
    startTrial = starts(end);
    fixOcc = bhv.CodeTimes{i}(bhv.CodeNumbers{i} == fixOccCode);
    imgOn = bhv.CodeTimes{i}(bhv.CodeNumbers{i} == imgOnCode);
    imgOff = bhv.CodeTimes{i}(bhv.CodeNumbers{i} == imgOffCode);
    endts = bhv.CodeTimes{i}(bhv.CodeNumbers{i} == trialEndCode);
    endTrial = endts(end);
    eyedat = bhv.AnalogData{i}.EyeSignal;
    imlook = eyedat(imgOn - fixOcc+300:imgOn-fixOcc+1000, :);
    % size(eyedat, 1)
    % endTrial
    % startTrial
    % endTrial - startTrial
    % assert(size(eyedat, 1) == endTrial - startTrial);
    if cond == 7 % fam on left, nov on right
        novt = boxEyeData(imlook, stimDeg, ...
            stimDeg, stimOffset, 0);
        famt = boxEyeData(imlook, stimDeg, ...
            stimDeg, -stimOffset, 0);
        [famfix, novfix, fo] = imgFixes(eyedat, stimDeg, stimDeg, ...
            stimOffset, 0, fixThr);
        fixords = [fixords, fo];
        famVNov = [famVNov; size(famt, 1), size(novt, 1)];
        dirLook = [dirLook; size(famt, 1), size(novt, 1)];
    elseif cond == 10 % nov on left, fam on right
        novt = boxEyeData(imlook, stimDeg, ...
            stimDeg, -stimOffset, 0);
        famt = boxEyeData(imlook, stimDeg, ...
            stimDeg, stimOffset, 0);
        [novfix, famfix, fo] = imgFixes(eyedat, stimDeg, stimDeg, ...
            stimOffset, 0, fixThr);
        fixords = [fixords, ~fo];
        famVNov = [famVNov; size(famt, 1), size(novt, 1)];
        dirLook = [dirLook; size(novt, 1), size(famt, 1)];
    elseif cond == 8
        nov1 = boxEyeData(imlook, stimDeg, ...
            stimDeg, -stimOffset, 0);
        nov2 = boxEyeData(imlook, stimDeg, ...
            stimDeg, stimOffset, 0);
        novOnly = [novOnly; size(nov1, 1); size(nov2, 1)];
        dirLook = [dirLook; size(nov1, 1), size(nov2, 1)];
    elseif cond == 9
        fam1 = boxEyeData(imlook, stimDeg, ...
            stimDeg, -stimOffset, 0);
        fam2 = boxEyeData(imlook, stimDeg, ...
            stimDeg, stimOffset, 0);
        famOnly = [famOnly; size(fam1, 1); size(fam2, 1)];
        dirLook = [dirLook; size(fam1, 1), size(fam2, 1)];
    end
end

figure; hold on;
size(famVNov)
mfamnov = mean(famVNov);
sfamnov = std(famVNov);
[~, p] = ttest2(famVNov(:, 1), famVNov(:, 2));
bar(0:1, mfamnov);
ax = gca;
ax.XTick = [0, 1];
ax.XTickLabel = {'familiar', 'novel'};
title(sprintf('%.5f', p));
errorbar(0:1, mfamnov, sfamnov, '+');
hold off;

figure; hold on;
difffamnov = famVNov(:, 1) - famVNov(:, 2);
bar(mean(difffamnov));
[~, p] = ttest(difffamnov);
title(sprintf('fam - nov; p = %.5f', p));
errorbar(mean(difffamnov), std(difffamnov));
hold off;

figure; hold on;
histogram(difffamnov, 40);
title('fam - nov');
hold off;

figure; hold on;
dirdiffLook = dirLook(:, 1) - dirLook(:, 2);
histogram(dirdiffLook, 40);
[~, p] = ttest(dirdiffLook);
title(sprintf('left - right; p=%.3f', p));
hold off;

% figure; hold on;
% blockedfn = reshape([difffamnov; nan; nan; nan; nan; nan], ...
%     20, (size(difffamnov, 1) + 5) / 20);
% plot(mean(blockedfn));
% hold off;

figure; hold on;
monlys = [mean(famOnly), mean(novOnly)];
sonlys = [std(famOnly), std(novOnly)];
[~, p] = ttest2(famOnly, novOnly);
tt = sprintf('total time on fam-fam (left) and nov-nov (right); p = %.4f',...
    p);
title(tt);
bar(0:1, monlys);
errorbar(0:1, monlys, sonlys, '+');
hold off;