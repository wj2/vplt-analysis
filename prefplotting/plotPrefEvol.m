
function [ alldat, preflook ] = plotPrefEvol( bhv, winsize, winstep, ...
    tottime )

winends = winsize:winstep:tottime;
alldat = cell(length(winends), 1);
preflook = zeros(length(winends), 3);
for i = 1:length(winends)
    d = avgLooking(bhv, winends(i) - winsize + 1, winends(i));
    comp = d.famvnov;
    proport = mean(comp(:, 2) - comp(:, 1)) / winsize;
    sterrprop = (std(comp(:, 2) - comp(:, 1)) / winsize) ... 
        / sqrt(size(comp, 1));
    [~, p] = ttest(comp(:, 2) - comp(:, 1));
    preflook(i, :) = [proport, sterrprop, p];
    alldat{i} = d;
end

figure; hold on;
winmarks = winends - winsize / 2;
plot(winmarks, preflook(:, 1));
errorbar(winmarks, preflook(:, 1), preflook(:, 2));
sigs = winmarks(preflook(:, 3) < .001);
plot(sigs, ones(size(sigs)), '+');
sigs
[~, bind] = max(preflook(:, 1));
maxend = winends(bind);
maxstart = winends(bind) - winsize + 1;
tt = sprintf('winsize = %.1f; best window = [%i, %i]', ...
    winsize, maxstart, maxend);
title(tt);
xlabel('middle of window post image onset (ms)');
ylabel('(novel - familiar) / winsize');
hold off; 
end

