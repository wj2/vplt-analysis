function [  ] = plotEye( eyesigs, novsides, wid, hei, x, y, imgons )

neicolor = [.6, .6, .6];
famcolor = [.5, .5, 1];
novcolor = [1, .5, .5];
video = VideoWriter('pref-looking-video.avi', 'Motion JPEG AVI');
video.FrameRate = 70;
video.Quality = 100;
open(video);
f = figure('Position', [0, 0, 512, 512]); hold on;
whitebg(f, [0, 0, 0]);
mr = max([wid, hei]);
axis square;
xlim([-x-1.2*mr, x+1.2*mr]);
ylim([-x-1.2*mr, x+1.2*mr]);
axis off;

itilen = 1.5;
for z = 1:size(eyesigs, 1)
    novDummy = plot(0, 0, 'Color', novcolor, 'Visible', 'off');
    famDummy = plot(1, 1, 'Color', famcolor, 'Visible', 'off');
    legend([novDummy, famDummy], {'novel', 'familiar'});
    legend boxoff;
    % do ITI
    for dummy = 1:round(itilen*video.FrameRate)
        drawnow;
        writeVideo(video, getframe);
    end
    eyesig = eyesigs{z};
    [~, linds] = boxEyeData(eyesig, wid, hei, -x, y);
    [~, rinds] = boxEyeData(eyesig, wid, hei, x, y);
    
    neither = ~(linds | rinds);
    dnei = diff([0; neither]);
    neiStarts = find(dnei == 1);
    labelNei = ones(size(neiStarts))*3;
    neiEnds = dnei == -1;
    
    dlinds = diff([0; linds]);
    drinds = diff([0; rinds]);
    dlStarts = find(dlinds == 1);
    labelLef = ones(size(dlStarts));
    dlEnds = dlinds == -1;
    
    drStarts = find(drinds == 1);
    labelRig = ones(size(drStarts));
    drEnds = drinds == -1;
    
    imgon = imgons(z);
    sNov = novsides(z);
    colorlabs = cell(3, 1);
    colorlabs{3} = neicolor;
    colorlabs{sNov} = novcolor;
    if sNov == 1
        sFam = 2;
        colorlabs{sFam} = famcolor;
        labelRig = labelRig*sFam;
        labelLef = labelLef*sNov;
    elseif sNov == 2
        sFam = 1;
        colorlabs{sFam} = famcolor;
        labelRig = labelRig*sNov;
        labelLef = labelLef*sFam;
    end
    
    allabs = [labelNei; labelLef; labelRig];
    allstars = [neiStarts; dlStarts; drStarts];
    [sortstarts, sinds] = sort(allstars);
    slabels = allabs(sinds);
    
    rR = rectangle('Position', [x-(wid/2), y-(hei/2), wid, hei], ...
        'EdgeColor', colorlabs{2}, 'Visible', 'off');
    rL = rectangle('Position', [-x-(wid/2), y-(hei/2), wid, hei], ...
        'EdgeColor', colorlabs{1}, 'Visible', 'off');
    
    lines = cell(length(sortstarts), 1);
    k = 0;
    txt = text(-x-1*mr, x+1*mr, sprintf('%.2fs', k / 1000), ...
        'Color', neicolor, 'FontSize', 15);
    sortstarts = [sortstarts; size(eyesig, 1)];
    for i = 1:length(sortstarts) - 1
        lines{i} = animatedline('Color', colorlabs{slabels(i)});
        for j = sortstarts(i):sortstarts(i+1)
            k = k + 1;
            if k == imgon
                set(rR, 'Visible', 'on');
                set(rL, 'Visible', 'on');
            end
            txt.String = sprintf('%.2fs', k / 1000);
            addpoints(lines{i}, eyesig(j, 1), eyesig(j, 2));
            if mod(k, 10) == 0
                drawnow;
                writeVideo(video, getframe);
            end
        end
        if i + 1 < size(sortstarts, 1)
            addpoints(lines{i}, eyesig(j+1, 1), eyesig(j+1, 2));
        end
    end
    cla;
end
close(video);
end

