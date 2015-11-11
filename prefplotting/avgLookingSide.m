function [ lookL, lookR ] = avgLookingSide( bhv, startWin, endWin, trials )
windstruc.startwin = startWin;
windstruc.endwin = endWin;

stimOffset = 3;
stimDeg = 5.5;

fixOccCode = 8;
trialStartCode = 9;
imgOnCode = 191;
imgOffCode = 192;
trialEndCode = 18;

lookL = zeros(length(trials), 1);
lookR = zeros(length(trials), 1);
for i = 1:size(trials, 1)
    if size(trials, 2) == 2
        usebhv = bhv{trials(i, 2)};
        tNum = trials(i, 1);
    else
        usebhv = bhv;
        tNum = trials(i);
    end
    % cond = bhv.ConditionNumber(tNum);
    % starts = bhv.CodeTimes{tNum}(bhv.CodeNumbers{tNum} == trialStartCode);
    % startTrial = starts(end);
    fixOcc = usebhv.CodeTimes{tNum}(usebhv.CodeNumbers{tNum} == fixOccCode);
    imgOn = usebhv.CodeTimes{tNum}(usebhv.CodeNumbers{tNum} == imgOnCode);

    % imgOff = bhv.CodeTimes{tNum}(bhv.CodeNumbers{tNum} == imgOffCode);
    % endts = bhv.CodeTimes{tNum}(bhv.CodeNumbers{tNum} == trialEndCode);
    % endTrial = endts(end);
    eyedat = usebhv.AnalogData{tNum}.EyeSignal;
    if ((usebhv.TrialError(tNum) ~= 0) ...
            || (imgOn - fixOcc + endWin > size(eyedat, 1)))
        lookL(i) = nan;
        lookR(i) = nan;
    else
        imlook = eyedat(imgOn - fixOcc+startWin:imgOn-fixOcc+endWin, :);
        lr = boxEyeData(imlook, stimDeg, ...
            stimDeg, stimOffset, 0);
        ll = boxEyeData(imlook, stimDeg, ...
            stimDeg, -stimOffset, 0);
        lookL(i) = size(ll, 1);
        lookR(i) = size(lr, 1);
    end
end
end

