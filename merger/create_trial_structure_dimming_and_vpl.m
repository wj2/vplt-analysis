function x = create_trial_structure_dimming_and_vpl(data, task_codes, extract_LFP, bhv_offset)

% This function converts the data structure from MonkeyLogic into a
% structure where each entry consists of a single trial. The structures
% contains the spike times, bar up time, stim onset times and motion category
% information.
% Nicolas Masse, 2013

if ~exist('task_codes','var') || isempty(task_codes)
    [task_codes, task_code_descriptions] = define_task_codes;
end

if ~exist('extract_LFP','var') || isempty(extract_LFP)
    % define default extract_LFP
    extract_LFP = 0;
end

if ~exist('bhv_offset','var') || isempty(bhv_offset)
% only to be used if there was a mismatch between the number of trials in
% the BHV and the number of trials extracted in the NEURO structure
    bhv_offset = 0;
end

[trial_starts_ms, trial_ends_ms, trial_start_ind, trial_end_ind] = get_trial_starts_and_ends(data, task_codes, task_code_descriptions);

% make sure the correct number of trials has been extracted
% if length(trial_starts_ms) ~= length(trial_ends_ms) | ...
%     (length(trial_starts_ms) ~= length(data.BHV.TrialError))
%     error('Incorrect number of trials has been extracted')
% end

% add dimming trials to structure
x = extract_trial_data(data, trial_starts_ms, trial_ends_ms, extract_LFP, task_codes, task_code_descriptions, bhv_offset,trial_start_ind, trial_end_ind);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [trial_starts_ms, trial_ends_ms,trial_start_ind,trial_end_ind] = get_trial_starts_and_ends(data, task_codes, task_code_descriptions, trial_type_codes)

% determine number of trial types from BHV data
num_trials = length(data.BHV.TrialError);

% determine relevant code numbers
ind_ISI_start = find(strcmp(task_code_descriptions,'ISI_start'));
ind_ISI_end = find(strcmp(task_code_descriptions,'ISI_end'));
ind_trial_start = find(strcmp(task_code_descriptions,'trial_start'));

ISI_start_code = task_codes(ind_ISI_start);
ISI_end_code = task_codes(ind_ISI_end);

% find the start, end of each trial 
trial_start_ind = find(data.NEURO.CodeNumbers(1:end-2) == ISI_end_code & ...
    data.NEURO.CodeNumbers(2:end-1) == ISI_end_code & data.NEURO.CodeNumbers(3:end) == ISI_end_code);
    
trial_end_ind = [];

for i = 1:length(trial_start_ind)
    trial_end_current  = find(data.NEURO.CodeNumbers(trial_start_ind(i):end) ...
        == ISI_start_code , 1,'first');
    trial_end_ind = [trial_end_ind trial_end_current+trial_start_ind(i)-1];
end
        
trial_starts_ms = data.NEURO.CodeTimes(trial_start_ind);
trial_ends_ms = data.NEURO.CodeTimes(trial_end_ind);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x = extract_trial_data(data, trial_starts_ms, trial_ends_ms, extract_LFP, ...
    task_codes, task_code_descriptions, bhv_offset, trial_start_ind, trial_end_ind)


BHV = data.BHV;
% only to be used if there was a mismatch between the number of trials in
% the BHV and the number of trials extracted in the NEURO structure
% for Wahwah Feb 20, bhv_offset = 5
if ~exist('bhv_offset','var') || isempty(bhv_offset)
    bhv_offset = 0;
end

x = [];
count = 0;
minImgOnCode = 56;
maxImgOnCode = 180;
imgOffCode = 55;
for i = 1:length(data.BHV.TrialNumber)
    bhvgo = BHV.CodeTimes{i}(1);
    usei = i + bhv_offset;
    count = count + 1;  
    x(count).trial_type = BHV.ConditionNumber(usei);
    x(count).TrialError = BHV.TrialError(usei);
    trial_inds = find(data.NEURO.CodeTimes >= trial_starts_ms(i) & ...
    data.NEURO.CodeTimes <= trial_ends_ms(i));
    x(count).block_number = BHV.BlockNumber(usei);
    x(count).code_numbers = data.NEURO.CodeNumbers(trial_start_ind(i):trial_end_ind(i)+2);
    x(count).code_times = data.NEURO.CodeTimes(trial_start_ind(i):trial_end_ind(i)+2) - trial_starts_ms(i);
    x(count).trial_starts = trial_starts_ms(i);
    x(count).datafile = data.BHV.DataFileName;
    x(count).datanum = data.BHV.DataNum;
    useOnCodes = (minImgOnCode <= data.BHV.CodeNumbers{usei} & ...
        data.BHV.CodeNumbers{usei} <= maxImgOnCode);
    useOffCodes = data.BHV.CodeNumbers{usei} == imgOffCode;
    x(count).centimgon = data.BHV.CodeTimes{usei}(useOnCodes) - bhvgo;
    x(count).centimgoff = data.BHV.CodeTimes{usei}(useOffCodes) - bhvgo;
    x(count).trialnum = data.BHV.TrialNumber(i);
    
    x(count).leftimg = BHV.leftimg{i};
    x(count).leftviews = BHV.leftviews(i);
    x(count).rightimg = BHV.rightimg{i};
    x(count).rightviews = BHV.rightviews(i);
        
    for j = 1:length(task_codes)
        fieldname = task_code_descriptions{j};
        code = task_codes(j);
        code_ind = find(data.NEURO.CodeNumbers(trial_inds) == code, 1, 'first');
        code_time = data.NEURO.CodeTimes(trial_inds(code_ind)) - trial_starts_ms(i);
        x(count).(fieldname) = code_time;
    end
    x(count).eyepos = BHV.AnalogData{i}.EyeSignal(bhvgo:end, :);
    
    % extract spike times
    if ~isempty(data.NEURO.Neuron)
        spike_signals = fieldnames(data.NEURO.Neuron);
        x(count).spike_times = cell(1, length(spike_signals));
        for j = 1:length(spike_signals)
            spikes = data.NEURO.Neuron.(spike_signals{j});
            spike_inds = find(spikes >= trial_starts_ms(i) - 1000 & ... % collect spikes from 0 ms before trial start
                spikes <= trial_ends_ms(i)+1000); % collect spikes up until 1000 ms after trial end
            x(count).spike_times{j} = spikes(spike_inds) - trial_starts_ms(i);
        end
    else
        warning('No spikes in data structure!')
        x(count).spike_times = {};
    end
    
    % extract LFPs
    if extract_LFP
        if ~isempty(data.NEURO.LFP)
            LFP_signals = fieldnames(data.NEURO.LFP);
            for j = 1:length(LFP_signals)
                % extract LFPs from 0ms before trial start to 2000ms
                % after trial end
                LFP = data.NEURO.LFP.(LFP_signals{j})(trial_starts_ms(i)-1000:trial_ends_ms(i)+1000);
                x(count).LFP{j} = LFP';
            end
        else
            x(count).LFP = [];
        end
    end

    % get the direction and start times of the dot stimuli
    task_cond_num = BHV.ConditionNumber(i+bhv_offset);
    x(count).task_cond_num = task_cond_num;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function start_times = get_stimulus_start_times(data, trial_inds, stim_start_codes)

% these codes denote the start of task objects
% stim_start_codes = 23:2:27;
code_numbers = data.NEURO.CodeNumbers(trial_inds);
code_times = data.NEURO.CodeTimes(trial_inds);
[~, start_inds] = intersect(code_numbers, stim_start_codes);
start_times = code_times(start_inds) - code_times(1);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x = extract_trial_data_mem_saccade(x, data, trial_starts_ms, trial_ends_ms, trial_type, extract_LFP)

% trial_type = 1 are memory saccade trials
trial_starts_ms = trial_starts_ms(trial_type == 1);
trial_ends_ms = trial_ends_ms(trial_type == 1);
count = length(x);

if isfield(data, 'BHV');
    BHV = data.BHV;
elseif isfield(data, 'BHV1');
    BHV = data.BHV1;
else
    error('No corresponding BHV strcuture found');
end

if length(trial_starts_ms) > length(BHV.ConditionNumber)
    warning('There is a mismatch between the number of trials starts from data.NEURO and the number of trials inthe BHV structure.')
    warning(['Only usig the first ',num2str(length(BHV.ConditionNumber)),' trials'])
    trial_starts_ms = trial_starts_ms(1:length(BHV.ConditionNumber));
end

for i = 1:length(trial_starts_ms)
       
    trial_inds = find(data.NEURO.CodeTimes >= trial_starts_ms(i) & ...
        data.NEURO.CodeTimes <= trial_ends_ms(i));
    trial_times = data.NEURO.CodeTimes(trial_inds);
    
    
    % extract fixation point off time
    fixation_off_ind = find(data.NEURO.CodeNumbers(trial_inds) == 36, 1, 'first');
    reward_ind = find(data.NEURO.CodeNumbers(trial_inds) == 96, 1, 'first');
    sac_target_on_ind = find(data.NEURO.CodeNumbers(trial_inds) == 25, 1, 'first');
    
    count = count + 1;
    x(count).trial_type = 'memory_saccade';
    x(count).fixation_off = data.NEURO.CodeTimes(trial_inds(fixation_off_ind)) - trial_starts_ms(i);
    x(count).reward_time = data.NEURO.CodeTimes(trial_inds(reward_ind)) - trial_starts_ms(i);
    x(count).condition_number = BHV.ConditionNumber(i);
    x(count).sac_target_on = data.NEURO.CodeTimes(trial_inds(sac_target_on_ind)) - trial_starts_ms(i);
    
    % extract spike times
    if ~isempty(data.NEURO.Neuron)
        spike_signals = fieldnames(data.NEURO.Neuron);
        x(count).spike_times = cell(1, length(spike_signals));
        for j = 1:length(spike_signals)
            spikes = data.NEURO.Neuron.(spike_signals{j});
            % find spike times from 0ms before trial start to 2000ms after
            % trial start
            spike_inds = find(spikes >= trial_starts_ms(i) - 0 & ...
                spikes <= trial_ends_ms(i) + 2000);
            x(count).spike_times{j} = spikes(spike_inds) - trial_starts_ms(i);
        end
    end
    
     % extract LFPs
    if extract_LFP
        if ~isempty(data.NEURO.LFP)
            LFP_signals = fieldnames(data.NEURO.LFP);
            for j = 1:length(LFP_signals)
                % extract LFPs from 0ms before trial start to 2000ms after
                % trial start
                LFP = data.NEURO.LFP.(LFP_signals{j})(trial_starts_ms(i):trial_ends_ms(i)+2000);
                x(count).LFP{j} = LFP';
            end
        else
            x(count).LFP = [];
        end
    end
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [task_codes, task_code_descriptions] = define_task_codes

i = 1;
task_codes(i) = 35;
task_code_descriptions{i} = 'fixation_on';

i = i + 1;
task_codes(i) = 36;
task_code_descriptions{i} = 'fixation_off';

i = i + 1;
task_codes(i) = 4;
task_code_descriptions{i} = 'lever_release';

i = i + 1;
task_codes(i) = 7;
task_code_descriptions{i} = 'lever_hold';

i = i + 1;
task_codes(i) = 48;
task_code_descriptions{i} = 'reward_time';

i = i + 1;
task_codes(i) = 18;
task_code_descriptions{i} = 'ISI_start';

i = i + 1;
task_codes(i) = 9;
task_code_descriptions{i} = 'ISI_end';

i = i + 1;
task_codes(i) = 8;
task_code_descriptions{i} = 'fixation_acquired';

i = i + 1;
task_codes(i) = 191;
task_code_descriptions{i} = 'left_img_on';

i = i + 1;
task_codes(i) = 192;
task_code_descriptions{i} = 'left_img_off';

i = i + 1;
task_codes(i) = 195;
task_code_descriptions{i} = 'right_img_on';

i = i + 1;
task_codes(i) = 196;
task_code_descriptions{i} = 'right_img_off';

