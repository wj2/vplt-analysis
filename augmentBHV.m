function [ augdats ] = augmentBHV( dats, imglogs )

biglog = mergeLogs(imglogs);
augdats = cell(length(dats), 1);
for i = 1:length(dats)
    if ~isempty(dats{i})
        augdats{i} = struct();
        diffbase = fields(dats{i});
        augdats{i}.BHV = dats{i}.(diffbase{1}).BHV;
        augdats{i}.NEURO = dats{i}.(diffbase{1}).NEURO;
        rellog = biglog{end} == i;
        limgs = rellog & (biglog{2} == 1);
        rimgs = rellog & (biglog{2} == 2);
        vplttris = biglog{1}(rellog);
        vplttris = vplttris(1:2:end);
        trials = size(augdats{i}.BHV.TrialNumber, 1);
        augdats{i}.BHV.leftimg = cell(trials, 1);
        augdats{i}.BHV.leftviews = nan(trials, 1);
        augdats{i}.BHV.rightimg = cell(trials, 1);
        augdats{i}.BHV.rightviews = nan(trials, 1);
        
        augdats{i}.BHV.leftimg(vplttris) = biglog{7}(limgs);
        augdats{i}.BHV.leftviews(vplttris) = biglog{5}(limgs);
        augdats{i}.BHV.rightimg(vplttris) = biglog{7}(rimgs);
        augdats{i}.BHV.rightviews(vplttris) = biglog{5}(rimgs);
    end
end
end

