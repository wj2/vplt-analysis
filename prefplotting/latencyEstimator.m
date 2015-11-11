function [ expectedLat, stdLat ] = latencyEstimator( spiketimes, bgwin, ...
    intwin, step )
%% 2.1a
% This first method is derived from section 2.1.2 of Pawlas et al. in
% Neural Computation (2010). We use the first 80ms of data that we have to
% estimate the background firing rate. Typically we would use actual
% background firing. This method is fairly involved, but essentially we
% estimate the distribution of first spikes after some time point in
% background firing due to noise (because we have no background here, we
% use the firing rate observed in the first 80ms of all trials and assume
% background firing is Poisson, from which we easily obtain this
% distribution), then we observe the distribution of first spike times when
% the first spike seen may be either evoked or noise (we can obtain the 
% empirical cdf from the trials we have). The comparison of these two 
% distributions gives us the distribution of first evoked spike times.

    function [ out ] = funcOverTimeWindow( spiketimes, begin, ending, func )
        makeCut = @(x) length(x((begin <= x) & (x < ending))) / (ending - begin);
        earlySpikes = cellfun(makeCut, spiketimes);
        out = func(earlySpikes);
    end

    function [ out ] = evokedSTCumDistrib( spiketimes, t, lambda )
        out = 1 - (1 - empiricalFSCumDistrib(spiketimes, t))*exp(lambda*t);
    end

    function [ accumulate ] = eulerIntegrate( func, begin, ending, step )
        accumulate = 0;
        for t = begin:step:ending
            accumulate = accumulate + step*func(t);
        end
    end

    function [ prob ] = empiricalFSCumDistrib( spiketimes, t )
        spikeBefore = @(x) any(x < t) | isempty(x);
        successes = sum(cellfun(spikeBefore, spiketimes));
        prob = successes / length(spiketimes);
    end

% get all spikes occuring within the first 80ms
lambda = funcOverTimeWindow(spiketimes, bgwin(1), bgwin(2), @mean);
anonExpectFunc = @(a) 1 - evokedSTCumDistrib(spiketimes, a, lambda);
expectedLatencySP = eulerIntegrate(anonExpectFunc, intwin(1), intwin(2),...
    step);
anonSMFunc = @(a) 2*a*(1 - evokedSTCumDistrib(spiketimes, a, lambda));
smLatency = eulerIntegrate(anonSMFunc, intwin(1), intwin(2), step);
stdLatencySP = sqrt(smLatency - expectedLatencySP*expectedLatencySP);

expectedLat = expectedLatencySP;
stdLat = stdLatencySP;
end

