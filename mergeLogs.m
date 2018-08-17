function [ bigl ] = mergeLogs( imglgs )

namePos = 7;
timesSeenPos = 5;

    function [ log ] = appendEntry(log, adlg, ind)
        for k = 1:length(adlg)
            log{k} = cat(1, log{k}, adlg{k}(ind));
        end
    end

for i = 1:length(imglgs)
    log = readImgLog(imglgs{i});
    log = cat(2, log, ones(size(log{1}, 1), 1)*i);
    if i == 1
        bigl = log;
    else
        for j = 1:size(log{1}, 1)
            nm = log{namePos}{j};
            seents = sum(strcmp(bigl{namePos}, nm));
            log{timesSeenPos}(j) = seents; 
            bigl = appendEntry(bigl, log, j);
        end
    end
end
end

