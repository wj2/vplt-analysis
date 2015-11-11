function [ preflook, bwlook, l ] = plotPrefEvolSide( bhv, winsize, ...
    winstep, tottime, trialsLeft, trialsRight, hei, varargin )

parser = inputParser;
parser.KeepUnmatched = true;
parser.addRequired('bhv');
parser.addRequired('winsize');
parser.addRequired('winstep');
parser.addRequired('tottime');
parser.addRequired('trialsLeft');
parser.addRequired('trialsRight');
parser.addRequired('hei');
parser.addParameter('bigwind', 0);
parser.addParameter('lookthr', 5);

parser.parse(bhv, winsize, winstep, tottime, trialsLeft, trialsRight, ...
    hei, varargin{:});
bhv = parser.Results.bhv;
winsize = parser.Results.winsize;
winstep = parser.Results.winstep;
tottime = parser.Results.tottime;
trialsLeft = parser.Results.trialsLeft;
trialsRight = parser.Results.trialsRight;
hei = parser.Results.hei;
bigwind = parser.Results.bigwind;
additPars = parser.Unmatched;
lookthr = parser.Results.lookthr;

winends = winsize:winstep:tottime;
preflook = zeros(length(winends), 3);
for i = 1:length(winends)
    [lN1, lF1] = avgLookingSide(bhv, winends(i) - winsize + 1, ...
        winends(i), trialsLeft);
    [lF2, lN2] = avgLookingSide(bhv, winends(i) - winsize + 1, ...
        winends(i), trialsRight);
    comp = [lF1, lN1; lF2, lN2];
    prefind = comp(:, 2) - comp(:, 1);
    if i == 1
        allfind = prefind;
    else
        allfind = [allfind, prefind];
    end
    % proport = nanmean(prefind) / winsize;
    % sterrprop = (nanstd(prefind) / winsize) ... 
    %     / sqrt(sum(~isnan(prefind)));
    % [~, p] = ttest(comp(:, 2) - comp(:, 1));
    % preflook(i, :) = [proport, sterrprop, p];
end
allfind(all(abs(allfind) < lookthr, 2), :) = nan;
proports = nanmean(allfind, 1) / winsize;
sterrprop = (nanstd(allfind, 1) / winsize) ./ sqrt(sum(~isnan(allfind), 1));
[~, p] = ttest(allfind);
preflook = [proports', sterrprop', p'];
size(preflook)

bwlook = zeros(1, 3);
[lN1, lF1] = avgLookingSide(bhv, 1, ...
    bigwind, trialsLeft);
[lF2, lN2] = avgLookingSide(bhv, 1, ...
    bigwind, trialsRight);
comp = [lF1, lN1; lF2, lN2];
prefind = comp(:, 2) - comp(:, 1);
prefind(all(abs(prefind) < lookthr, 2), : ) = nan;
bwlook(1, 1) = nanmean(prefind);
bwlook(1, 2) = (nanstd(prefind) / sqrt(sum(~isnan(prefind))));
[~, p] = ttest(prefind);
bwlook(1, 3) = p;
p
sqrt(sum(~isnan(prefind)))

winmarks = winends - winsize / 2;
l = plot(winmarks, preflook(:, 1), additPars);
additPars.Color = get(l, 'Color');
errPars = additPars;
errPars.Marker = '+';
errorbar(winmarks, preflook(:, 1), preflook(:, 2), ...
    'Marker', errPars.Marker, 'Visible', errPars.Visible,...
    'Color', errPars.Color);
sigs = winmarks(preflook(:, 3) < .05);
plot(sigs, ones(size(sigs))*hei, '+', additPars);
[~, bind] = max(preflook(:, 1));
maxend = winends(bind);
maxstart = winends(bind) - winsize + 1;
tt = sprintf('winsize = %.1f', ...
    winsize);
title(tt);
xlabel('middle of window post image onset (ms)');
ylabel('(novel - familiar) / winsize');
end

