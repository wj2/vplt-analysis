function [ dats ] = loadAllData( datlist )
dats = cell(length(datlist), 1);
for i = 1:length(datlist)
    if ~isempty(datlist{i})
        dats{i} = load(datlist{i});
    end
end
end

