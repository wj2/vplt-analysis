% make bar plots
winbeg = 200;
winend = 1500;

% within session
load bootsy-15thrun-05-11-2015/dim_pref_look_nolever-bootsy-05-11-2015.mat;
fold = 'bootsy-15thrun-05-11-2015/';
imlg = strcat(fold, 'dim_pref_look_nolever-bootsy-05-11-2015_imglog.txt');

wiL = getRelevTrials(imlg, 9, 'diffspec', 1, ...
    'greaterthan', true, 'leftcond', @(x) x == 0);
wiR = getRelevTrials(imlg, 9, 'diffspec', -1, ...
    'lessthan', true, 'rightcond', @(x) x == 0);
wiSession = getPrefBars(bhv, winbeg, winend, wiL, wiR);

figure; hold on;
[~, l1] = plotPrefEvolSide(bhv, 400, 50, 5000, wiL, wiR, 1);
% title('within session');

% originally familiar
load bootsy-11thrun-05-05-2015/dim_pref_look_nolever-bootsy-05-05-2015.mat;
fold = 'bootsy-11thrun-05-05-2015/';
imlg = strcat(fold, 'dim_pref_look_nolever-bootsy-05-05-2015_imglog.txt');

ltL = getRelevTrials(imlg, 10);
ltR = getRelevTrials(imlg, 7);

ltSession = getPrefBars(bhv, winbeg, winend, ltL, ltR);
[~, l2] = plotPrefEvolSide(bhv, 400, 50, 5000, ltL, ltR, .9);
% title('between sessions');
legend([l1, l2], {'within session', 'between sessions'});
print(gcf, '-dpng', '-r300', 'nov-fam-timecourse.png');
hold off;

figure; hold on;
ylabel('proportion of time spent on image');
barheights = [wiSession(1:2), ltSession(1:2)];
errbars = [wiSession(3:4), ltSession(3:4)];
% bar([0, 1, 3, 4], barheights);
fambars = [wiSession(1), ltSession(1)];
novbars = [wiSession(2), ltSession(2)];
famerrs = [wiSession(3), ltSession(3)];
noverrs = [wiSession(4), ltSession(4)];
famcolor = [.5, .5, 1];
novcolor = [1, .5, .5];
errorbar([0, 3], fambars, famerrs, '+', 'Color', famcolor);
errorbar([1, 4], novbars, noverrs, '+', 'Color', novcolor);
bar([0, 3], fambars, .3, 'FaceColor', famcolor);
bar([1, 4], novbars, .3, 'FaceColor', novcolor);
ax = gca;
ax.XTick = [0, 1, 3, 4];
ax.YTick = [0, .3, .6];
ax.XTickLabel = {'2-5 views', '1st view', '~10,000 views', '1st view'};
% ax.XTickLabelRotation = 45;
ax.FontSize = 13.5;
title(sprintf('within session (left) and across sessions (right) novelty bias'));
print(gcf, '-dpng', '-r300', 'nov-fam-bars.png');
hold off;

