function [ bws ] = makeEvolPlot( bhvseq, imglogseq, varargin )

parser = inputParser;
parser.addRequired('bhvseq');
parser.addRequired('imglogseq');
parser.addParameter('nncond', 9);
parser.addParameter('incond', 7);
parser.addParameter('nicond', 10);
parser.addParameter('nfcond', 10);
parser.addParameter('fncond', 7);
parser.addParameter('includeSuperFam', false);
parser.addParameter('barPlot', true);
parser.addParameter('bigWindowSize', 2000);
parser.addParameter('winsize', 500);
parser.addParameter('winstep', 50);
parser.addParameter('tottime', 5000);
parser.addParameter('reductionFactor', 15);
parser.addParameter('numPieces', 5);
parser.addParameter('titleText', '');

parser.parse(bhvseq, imglogseq, varargin{:});

nncond = parser.Results.nncond;
incond = parser.Results.incond;
nicond = parser.Results.nicond;

nfcond = parser.Results.nfcond;
fncond = parser.Results.fncond;

includeSuperFam = parser.Results.includeSuperFam;

barplot = parser.Results.barPlot;
bwSize = parser.Results.bigWindowSize;

bhvs = loadBhvSeq(bhvseq);
hei = [.475, .475, .5, .5, .525];
winsize = parser.Results.winsize;
winstep = parser.Results.winstep;
tottime = parser.Results.tottime;
redfac = parser.Results.reductionFactor;
reps = parser.Results.numPieces;
start = 1;

titletext = parser.Results.titleText;

figure; hold on;
if includeSuperFam
    labels = cell(reps+1, 1);
    labelhandles = zeros(reps+1, 1);
    bws = zeros(reps+1, 3);
    lines = cell(reps+1, 1);
else
    labels = cell(reps-1, 1);
    labelhandles = zeros(reps, 1);
    bws = zeros(reps, 3);
    lines = cell(reps, 1);
end

% visibs = {'on', 'off', 'off', 'on', 'off', 'off'};
visibs = {'on', 'off', 'on', 'off', 'on'};
linecols = [.5, 0, 0; 0, 0, 0; 0, .5, 0; 0, 0, 0; 0, 0, .5];

for i = start:reps
    novcond = @(x) x == 0;
    novcondIntFam = @(x) ones(size(x));
    famcond = @(x) (floor(x/redfac) == i) & (x > 0);
    famcondIntFam = @(x) (floor(x/redfac) == i - 1);
    % [lnts, lnbhvs] = getRelevTrials(imglogseq, nncond, ...
    %     'rightcond', famcond, 'leftcond', novcond, 'multlogs', true);
    % [rnts, rnbhvs] = getRelevTrials(imglogseq, nncond, ...
    %     'rightcond', novcond, 'leftcond', famcond, 'multlogs', true);
    [lnts, lnbhvs] = getRelevTrials(imglogseq, nicond, ...
        'rightcond', famcondIntFam, 'leftcond', novcondIntFam, ...
        'multlogs', true);
    [rnts, rnbhvs] = getRelevTrials(imglogseq, incond, ...
        'rightcond', novcondIntFam, 'leftcond', famcondIntFam, ...
        'multlogs', true);
    tsleft = [lnts, lnbhvs];
    tsright = [rnts, rnbhvs];
    begin = redfac*(i-1);
    ending = redfac*i - 1;
    if begin == 0
        begin = begin + 1;
    end
    [~, bw, l] = plotPrefEvolSide( bhvs, winsize, ...
        winstep, tottime, tsleft, tsright, hei(i), 'bigwind', bwSize, ...
        'Visible', visibs{i}, 'Color', linecols(i, :));
    % labels{i} = sprintf('%d - %d views (n = %d)', begin, ...
    %     ending, size(tsright, 1) + size(tsleft, 1));
    % labels{i} = sprintf('%d views (n = %d)', i, ...
    %     size(tsright, 1) + size(tsleft, 1));
    labels{i} = sprintf('%d - %d', begin, ending);
    labelhandles(i) = l;
    % set(l, 'Visible', 'off');
    lines{i} = l;
    bw
    bws(i, :) = bw;
end

% get novel v super familiar
sfcol = [0, .5, 0];
if includeSuperFam
    [lnts, lnbhvs] = getRelevTrials(imglogseq, nfcond, 'multlogs', true);
    [rnts, rnbhvs] = getRelevTrials(imglogseq, fncond, 'multlogs', true);
    tsleft = [lnts, lnbhvs];
    tsright = [rnts, rnbhvs];
    [~, bw, l] = plotPrefEvolSide( bhvs, winsize, ...
        winstep, tottime, tsleft, tsright, hei(end), 'bigwind', bwSize, ...
        'Visible', visibs{end}, 'Color', linecols(end, :));
    % labels{end} = sprintf('~10,000 views (n = %d)', ...
    %     size(tsright, 1) + size(tsleft, 1));
    labels{end} = sprintf('~10,000');
    labelhandles(end) = l;
    lines{end} = l;
    bws(end, :) = bw;
end
% legend(labelhandles, labels);
ax = gca;
ax.FontSize = 15;
ax.YTick = [0, .2, .4];
ax.XTick = [0, 1000, 2000, 3000, 4000, 5000];
ylim([-.1, .55]);
xlabel('time (ms)');
ylabel('magnitude of novelty bias');
title('');
hold off;

barcolor = [.34, .34, .34];
% if barplot
%     figure; hold on;
%     xs = 1:size(labelhandles, 1);
%     bar(xs, bws(:, 1), 'FaceColor', barcolor);
%     errorbar(xs, bws(:, 1), bws(:, 2), '+', 'Color', barcolor);
%     title('across session');
%     ylabel('novel - familiar looking time (ms)');
%     ax = gca;
%     ax.XTick = xs;
%     ax.XTickLabelRotation = 45;
%     ax.XTickLabel = labels;
% end

if barplot
    xs = 1:size(labelhandles, 1);
    figure; hold on;
    for i = 1:size(labelhandles, 1)
        l = lines{i};
        if strcmp(l.Visible, 'on')
            col = l.Color;
        else
            col = barcolor;
        end
        bar(i, bws(i, 1), 'FaceColor', col);
        errorbar(i, bws(i, 1), bws(i, 2), '+', 'Color', col);
    end
    title(titletext);
    ylabel('novel - familiar looking time (ms)');
    xlabel('views');
    ax = gca;
    ax.FontSize = 15;
    ax.XTick = xs;
    ax.XTickLabelRotation = 45;
    ax.XTickLabel = labels;
    ax.YTick = [0, 200, 400, 600];
end

end

