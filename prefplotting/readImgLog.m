function [ readlog ] = readImgLog( imglog )
timeseen = 5;
f = fopen(imglog);
try     
    readlog = textscan(f, '%f\t%f\t%f\t%f\t%s\t%s');
    pseudoseen = zeros(size(readlog{1}));
    assert(size(readlog{end}, 1) > 1);
    readlog = [readlog(1:timeseen-1) {pseudoseen} readlog(timeseen:end)];
catch 
    fclose(f);
    f = fopen(imglog);
    readlog = textscan(f, '%f\t%f\t%f\t%f\t%f\t%s\t%s');
end
fclose(f);
end

