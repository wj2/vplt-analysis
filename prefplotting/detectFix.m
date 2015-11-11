function [ fixes ] = detectFix( eyesig, thresh )
deg = 1;
filtspan = 3;
deye = abs(diff(eyesig, deg, 1));
deye = [smooth(deye(:, 1), filtspan), smooth(deye(:, 2), filtspan)];
md = mean(deye);
sd = std(deye);
xlocs = conv(eyesig(:, 1), ones(deg+1, 1) / (deg + 1), 'valid');
ylocs = conv(eyesig(:, 2), ones(deg+1, 1) / (deg + 1), 'valid');
fixing = (deye(:, 1) <= md(1) + sd(1) * thresh) ...
    & (deye(:, 2) <= md(2) + sd(2) * thresh);
regends = diff(fixing);
starts = find(regends == 1);
ends = find(regends == -1);

numfixes = max(size(starts, 1), size(ends, 1));
fixes = zeros(numfixes, 6);
if numfixes > 0
    if ends(1) < starts(1) % then the first region is a fixation
        starts = [1; starts];
    end
    if ends(end) < starts(end)
        ends = [ends; length(xlocs)];
    end
end
for i = 1:length(starts)
    xavg = mean(xlocs(starts(i):ends(i)));
    yavg = mean(ylocs(starts(i):ends(i)));
    xstd = std(xlocs(starts(i):ends(i)));
    ystd = std(ylocs(starts(i):ends(i)));
    fixes(i, :) = [xavg, yavg, xstd, ystd, starts(i), ends(i)];
end
end

