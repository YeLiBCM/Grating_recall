% this code decompose NBG power and BBG power in a band pass method using hilbert transform.
%
% May 2020 -- Ye Li, Department of Neuroscience, BCM.

%% start/tidy
clear all
close all

%% subject information
sbj_name     = input('Subject code: ','s');
project_name = input('Project name: ', 's');
bn           = {input('Block number: ','s')};  
ref_type     = input('CAR or BP ref: ','s');   % common average vs bipolar
norm_type    = input('Normalization base [blk/ISI]: ','s');  % use testing phase of the whole block 
                                                             % or the last 1 sec of ISI to normalize the data

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
        
    % epoch window for normalization            
    switch norm_type
       case 'ISI'
          tf_srate     = round(master_vars.ecog_srate/master_vars.compress);
          norm_start   = floor(2.5*tf_srate);
          norm_end     = ceil(1.5*tf_srate);
          test_trigger = round(events_info.test_tone_onset/master_vars.compress);
       case 'blk'
          norm_start = round(events_info.test_tone_onset(1)/master_vars.compress);
          norm_end   = round(events_info.all_trial_offset(end)/master_vars.compress);
    end   
    
    for ci = elecs
        band = [];
        
        % Hilbert band pass
        % band pass NBG
        [signal_NBG, raw_sig_NBG] = band_pass_hilbert(master_vars, ci, freq_NBG); % 1 x signal length
        
        % band pass BBG
        [signal_BBG, raw_sig_BBG] = band_pass_hilbert(master_vars, ci, freq_BBG); % 8 x signal length
        
        %% uncomment the following lines to check band pass performance
        %figure(1)
        %[p_NBG, f_NBG] = pwelch(raw_sig_NBG, 2000, 1000, 2000, 1000);
        %plot(f_NBG, p_NBG)
        %hold on
        %
        %for i = 1:8
        %    [p_BBG, f_BBG] = pwelch(raw_sig_BBG(i,:), 2000, 1000, 2000, 1000);
        %    plot(f_BBG, p_BBG)
        %    hold on
        %end
        %
        %xlim([2,200])
        %xlabel('Frequency(Hz)')
        %ylabel('Amplitude')
        
        % get normalization base
        switch norm_type
           case 'ISI'
              norm_series_NBG = [];
              norm_series_BBG = [];
              
              for i = 1:length(test_trigger)
              ISI_epoch_NBG = signal_NBG(1,test_trigger-norm_start:test_trigger-norm_end);
              ISI_epoch_BBG = signal_BBG(:,test_trigger-norm_start:test_trigger-norm_end);
        
              norm_series_NBG = [norm_series_NBG ISI_epoch_NBG];
              norm_series_BBG = [norm_series_BBG ISI_epoch_BBG];
              end
              
              norm_base_NBG = mean(norm_series_NBG);
              norm_base_BBG = mean(norm_series_BBG,2);
              
           case 'blk'
              norm_base_NBG = mean(signal_NBG(norm_start:norm_end));
              norm_base_BBG = mean(signal_BBG(:,norm_start:norm_end),2);
        end
        
        %% NBG power extraction
        signal_NBG_normed = signal_NBG/norm_base_NBG;
        NBG_power         = 100*signal_NBG_normed - 100;
        
        %% BBG power extraction
        signal_BBG_normed = bsxfun(@rdivide, signal_BBG, norm_base_BBG);
        BBG_power         = mean(signal_BBG_normed.*100 - 100, 1);
        
        % store results
        band.elec = ci;
        band.block_name = block_name;
        band.NBG_power = NBG_power;
        band.BBG_power = BBG_power;
        
        % save results
        switch norm_type
           case 'ISI'
              save(sprintf('%s/TBand_decomp_%s_%s_%s_%.d',master_vars.Spec_dir, ref_type, master_vars.sbj_name, block_name, ci),'band');
           case 'blk'
              save(sprintf('%s/TBand_decomp_BLK_%s_%s_%s_%.d',master_vars.Spec_dir, ref_type, master_vars.sbj_name, block_name, ci),'band');
        end
        
        disp(['Channel ' num2str(ci) ' has been processed!'])
        
    end
end



