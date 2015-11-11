function [ ms ] = getPrefBars( bhv, winbeg, winend, tl, tr )

[lN1, lF1] = avgLookingSide(bhv, winbeg, winend, tl);
[lF2, lN2] = avgLookingSide(bhv, winbeg, winend, tr);
fam = [lF1; lF2];
fam = fam ./ (winend - winbeg);
nov = [lN1; lN2];
nov = nov ./ (winend - winbeg);

famMean = mean(fam);
novMean = mean(nov);
famSEM = std(fam) / sqrt(size(fam, 1) - 1);
novSEM = std(nov) / sqrt(size(nov, 1) - 1);
[~, p] = ttest2(fam, nov);
ms = [famMean, novMean, famSEM, novSEM, p];
end

