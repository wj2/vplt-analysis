function [ lfxs, rfxs, fixord ] = imgFixes( eyesig, wid, hei, xof, yof,...
    thresh)
fixes = detectFix(eyesig, thresh);
lfxs = boxEyeData(fixes, wid, hei, -xof, yof);
rfxs = boxEyeData(fixes, wid, hei, xof, yof);
fixord = [zeros(size(lfxs, 1), 1); ones(size(rfxs, 1), 1)];
[~, fis] = sort([lfxs(:, 5); rfxs(:, 5)]);
fixord = fixord(fis);
end

