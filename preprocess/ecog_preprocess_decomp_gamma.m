% this code decompose NBG power and BBG power in a band pass method using hilbert transform.
% 
% May 2020 -- Ye Li, Department of Neuroscience, BCM.

%% start/tidy
clear all
close all

%% subject information
sbj_name     = input('Subject code: ','s');
project_name = input('Project name: ', 's');
bn           = {input('Block number: ','s')}; % block names, can loop for multiple
ref_type     = input('CAR or BP ref: ','s');   % common average vs bipolar

for ii = 1:length(bn)

    % block name
    block_name = bn{ii};

    % master file
    load(sprintf('~/Documents/MATLAB/ECoG/neuralData/originalData/%s/master_%s_%s_%s.mat',sbj_name,project_name,sbj_name,block_name));
    
    % event file (used to get the global mean of spectral data - wiping out the time window before and after the real task)
    load(sprintf('%s/task_events_%s_%s_%s.mat',master_vars.result_dir, project_name, sbj_name, block_name));

    % good channels (for CAR)
    elecs = setxor([1:master_vars.nchan],[master_vars.badchan, master_vars.refchan, master_vars.epichan]);;

    % NBG band info
    f_lo_NBG = 20;
    f_hi_NBG = 60; 
    freq_NBG = [f_lo_NBG; f_hi_NBG]';
    
    % BBG band info
    f_lo_BBG = 70:10:140;
    f_hi_BBG = 80:10:150;
    freq_BBG = [f_lo_BBG; f_hi_BBG]';

    % real task window
    onset_1st_trial   = round(events_info.all_trial_onset(1)/master_vars.compress);
    offset_last_trial = round(events_info.all_trial_offset(end)/master_vars.compress);

    for ci = elecs
        band = [];

        %% NBG band pass
        % get NBG power
        signal_NBG    = band_pass_hilbert(master_vars, ci, freq_NBG); % 1 x signal length

        % normalize the entire power series as % of global mean
        norm_base_NBG = mean(signal_NBG(onset_1st_trial:offset_last_trial));
        NBG_power     = 100*(signal_NBG/norm_base_NBG) - 100; % make it percentage change

        %% BBG band pass
        % get power for each 10Hz wide band (70-80Hz, 80-90Hz, ... 140-150Hz)
        signal_BBG    = band_pass_hilbert(master_vars, ci, freq_BBG); % 8 x signal length

        % normalize each band using global mean
        norm_base_BBG = mean(signal_BBG(:,onset_1st_trial:offset_last_trial),2);
        signal_BBG_normed = bsxfun(@rdivide, signal_BBG, norm_base_NBG);
        BBG_power     = mean(100*signal_BBG_normed - 100,1);

        % store results
        band.elec = ci;
        band.block_name = block_name;
        band.NBG_power = NBG_power;
        band.BBG_power = BBG_power;
        
        % save results
        save(sprintf('%s/TBand_decomp_%s_%s_%s_%.d',master_vars.Spec_dir, ref_type, master_vars.sbj_name, block_name, ci),'band');
        
        disp(['Channel ' num2str(ci) ' has been processed!'])

    end
end



