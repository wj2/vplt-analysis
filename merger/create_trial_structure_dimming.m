function x = create_trial_structure_dimming(data, task_codes, extract_LFP, bhv_offset)

% This function converts the data structure from MonkeyLogic into a
% structure where each entry consists of a single trial. The structures
% contains the spike times, bar up time, stim onset times and motion category
% information.
% Krithika Mohan 2015

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
x = extract_trial_data(data, trial_starts_ms, trial_ends_ms, extract_LFP, task_codes, task_code_descriptions, bhv_offset, trial_start_ind, trial_end_ind);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [trial_starts_ms, trial_ends_ms, trial_start_ind, trial_end_ind] = get_trial_starts_and_ends(data, task_codes, task_code_descriptions, trial_type_codes)

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
    task_codes, task_code_descriptions, bhv_offset,trial_start_ind,trial_end_ind)


BHV = data.BHV;
% only to be used if there was a mismatch between the number of trials in
% the BHV and the number of trials extracted in the NEURO structure
% for Wahwah Feb 20, bhv_offset = 5
if ~exist('bhv_offset','var') || isempty(bhv_offset)
    bhv_offset = 0;
end

x = [];
count = 0;

for i = 1:length(trial_starts_ms)
        
    count = count + 1;  
    x(count).trial_type = 'dimming';
    x(count).TrialError = BHV.TrialError(i+bhv_offset);
    trial_inds = find(data.NEURO.CodeTimes >= trial_starts_ms(i) & ...
    data.NEURO.CodeTimes <= trial_ends_ms(i));
    x(count).block_number = BHV.BlockNumber(i+bhv_offset);
    x(count).code_numbers = data.NEURO.CodeNumbers(trial_start_ind(i):trial_end_ind(i)+2);
    x(count).code_times = data.NEURO.CodeTimes(trial_start_ind(i):trial_end_ind(i)+2);
    x(count).trial_starts = trial_starts_ms;
    x(count).image_nos = data.BHV.UserVars(1,count).img_index;
    
    for j = 1:length(task_codes)
        fieldname = task_code_descriptions{j};
        code = task_codes(j);
        code_ind = find(data.NEURO.CodeNumbers(trial_inds) == code, 1, 'first');
        code_time = data.NEURO.CodeTimes(trial_inds(code_ind)) - trial_starts_ms(i);
        x(count).(fieldname) = code_time;
    end
    
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

% i = i + 1;
% task_codes(i) = 23;
% task_code_descriptions{i} = 'image_on';
% 
% i = i + 1;
% task_codes(i) = 24;
% task_code_descriptions{i} = 'image_off';
% 
% i = i + 1;
% task_codes(i) = 25;
% task_code_descriptions{i} = 'dim_image_on';
% 
% i = i + 1;
% task_codes(i) = 26;
% task_code_descriptions{i} = 'dim_image_off';

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