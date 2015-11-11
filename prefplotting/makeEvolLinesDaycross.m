function [ ] = makeEvolLinesDaycross( bws )

figure; hold on;
rang = 1:size(bws, 1);
colors = [.5, 0, 0; .8, .8, .8; 0, .5, 0; .2, .2, .2; 0, 0, .5];
linewid = 3;
for i = 1:size(bws, 2);
    errorbar(rang, bws(:, i, 1), bws(:, i, 2), 'Color', colors(i, :),...
        'linewidth', linewid);
end
xlabel('days of viewing (~55 views/day)');
ylabel('novel - familiar looking time (ms)');
ax = gca;
ax.FontSize = 15;
ax.XTick = rang;
ax.YTick = [0, 200, 400, 600];
legend('1-2 views', '3-5 views', '6-8 views', '9-11 views', '12-14 views',...
    'Location','NorthWest');
legend BOXOFF
end

