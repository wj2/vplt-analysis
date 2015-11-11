function [ bhvseq ] = loadBhvSeq( bhvpaths )
bhvseq = cell(length(bhvpaths), 1);
for i = 1:length(bhvpaths)
    btemp = load(bhvpaths{i});
    bhvseq{i} = btemp.bhv;
end

