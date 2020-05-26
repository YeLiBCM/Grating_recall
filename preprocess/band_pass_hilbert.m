function decomp_signal = band_pass_hilbert(master_vars,ci,freq)
% band_pass_hilbert.m band pass signal with hilbert transform
% channel_filt.m will be used
% Input: 
%       master_vars = master file
%       ci          = electrode index
%       freq        = frequency ranges that you want to band pass (col1: f_lo, col2: f_high)
% Output:
%       decomp_signal = band-passed power (not amplitude)
%
% May, 2020 -- Ye Li, Department of Neuroscience, BCM
    % useful variables
    block_name = master_vars.block_num;
    srate_raw  = round(master_vars.ecog_srate);
    srate_comp = round(srate_raw/master_vars.compress);

    % load analogTraces (using CAR)
    load(sprintf('%s/CARiEEG%s_%.2d.mat',master_vars.CAR_dir,block_name, ci));

    % down sample signal
    signal = double(analogTraces);
    clear analogTraces
    decomp_input = decimate(signal, master_vars.compress, 'FIR');

    %% filter data within a freq range and extract power via hilbert transform
    srate = round(srate_comp); % sample rate

    % initialize amplitude
    tmp_amplitude = zeros(length(freq(:,1)),length(decomp_input));
    
    for fi = 1:length(freq(:,1))

        % filter out signal within a frequency band
        tmp_bandpass = channel_filt(decomp_input, srate, freq(fi,2), freq(fi,1), []);
        
        % hilbert transform, obtain analytic signal
        tmp_bandpass_analytic = hilbert(tmp_bandpass);

        % get the power
        tmp_power_hilbert = abs(tmp_bandpass_analytic).^2;
        
        % add to tmp_amplitude
        tmp_amplitude(fi,:) = tmp_power_hilbert;
    end
    
    decomp_signal = single(tmp_amplitude);
end