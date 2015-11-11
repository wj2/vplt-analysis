function [ inbox, inds ] = boxEyeData( data, widdeg, heideg, xoff, yoff )

xcol = data(:, 1);
ycol = data(:, 2);
inds = (xcol <= xoff + widdeg / 2) & (xcol >= xoff - widdeg / 2) ...
    & (ycol <= yoff + heideg / 2) & (ycol >= yoff - heideg / 2);
inbox = data(inds, :);
end

