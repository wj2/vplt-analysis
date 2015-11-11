function [ windstruc ] = avgLooking( bhv, startWin, endWin )

windstruc = struct();
windstruc.startwin = startWin;
windstruc.endwin = endWin;

stimOffset = 3;
stimDeg = 5.5;

fixOccCode = 8;
trialStartCode = 9;
imgOnCode = 191;
imgOffCode = 192;
trialEndCode = 18;

famVNov = [];
novOnly = [];
famOnly = [];
dirLook = [];
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
    imlook = eyedat(imgOn - fixOcc+startWin:imgOn-fixOcc+endWin, :);
    % size(eyedat, 1)
    % endTrial
    % startTrial
    % endTrial - startTrial
    % assert(size(eyedat, 1) == endTrial - startTrial);
    if cond == 7 % 7; 12; 14 fam on left, nov on right
        novt = boxEyeData(imlook, stimDeg, ...
            stimDeg, stimOffset, 0);
        famt = boxEyeData(imlook, stimDeg, ...
            stimDeg, -stimOffset, 0);
        famVNov = [famVNov; size(famt, 1), size(novt, 1)];
        dirLook = [dirLook; size(famt, 1), size(novt, 1)];
    elseif cond == 10 % 10; 13; 15  % nov on left, fam on right
        novt = boxEyeData(imlook, stimDeg, ...
            stimDeg, -stimOffset, 0);
        famt = boxEyeData(imlook, stimDeg, ...
            stimDeg, stimOffset, 0);
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

windstruc.famonly = famOnly;
windstruc.novonly = novOnly;
windstruc.dirlook = dirLook;
windstruc.famvnov = famVNov;

end

