function [ tnums, bhvnums ] = getRelevTrials( imglog, condnum, varargin )
parser = inputParser;
parser.addRequired('imglog');
parser.addRequired('condnum');
parser.addParameter('diffspec', 1000);
parser.addParameter('greaterthan', false);
parser.addParameter('lessthan', false);
parser.addParameter('leftcond', @(x) ones(size(x)));
parser.addParameter('rightcond', @(x) ones(size(x)));
parser.addParameter('multlogs', false);
parser.parse(imglog, condnum, varargin{:});
imglog = parser.Results.imglog;
condnum = parser.Results.condnum;
diffspec = parser.Results.diffspec;
greaterthan = parser.Results.greaterthan;
lessthan = parser.Results.lessthan;
leftcond = parser.Results.leftcond;
rightcond = parser.Results.rightcond;
multlogs = parser.Results.multlogs;

assert(~(greaterthan & lessthan));

if multlogs
    imlg = mergeLogs(imglog);
    allbhvnums = imlg{8};
else
    imlg = readImgLog(imglog);
end

trials = imlg{1};
conds = imlg{4}(1:2:end) == condnum;
reps = imlg{5};
repsleft = leftcond(reps(1:2:end));
repsright = rightcond(reps(2:2:end));
usereps = repsleft & repsright;
dreps = diff(reps);
if diffspec < 999;
    if greaterthan
        repcond = dreps(1:2:end) >= diffspec;
    elseif lessthan
        repcond = dreps(1:2:end) <= diffspec;
    else
        repcond = dreps(1:2:end) == diffspec;
    end
else
    repcond = ones(size(dreps(1:2:end)));
end
repcond = repcond & usereps;
allcond = conds & repcond;
halftrials = trials(1:2:end);
tnums = halftrials(allcond);
if multlogs
    halfbhvnums = allbhvnums(1:2:end);
    bhvnums = halfbhvnums(allcond);
else
    bhvnums = [];
end
end
