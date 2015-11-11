function [ ] = makeEvolLinesEnvisioned( bws )

figure; hold on;
for i = 1:size(bws, 1)
    errorbar(1:5, bws(i, :, 1), bws(i, :, 2))
end
legend('day 1', 'day 2', 'day 3', 'day4');
end

