import numpy as np
from scipy import stats
from ECoGBasic import SpectralBasic

class SpectralBurst(SpectralBasic):
    
    def __init__(self, sbj_name, blk_name, task_name):
        
        SpectralBasic.__init__(self,sbj_name, blk_name, task_name)

    def filter_burst(self,arr,dur):
        '''
        This function filtered a series of signal by a required duration
        
        Args:
        arr: a binary vector with 1 indicates signal and 0 indicates no signal
        dur: required duration
        
        Returns:
        arr_new = a new binary vector that wiped out signals lasting shorter than the required duration
        '''
        
        # initialize count
        count = 0
        
        for i in range(len(arr)):
            
            # if (the i-element of arr) = 1, add count
            if arr[i] == 1:
                count += 1
                
                # if the next element is 0, or if i reaches the last element, compare the count with dur
                if (i < (len(arr)-1) and arr[i+1] == 0) or (i == len(arr)-1):
                    
                    # delect the signals if the length is shorter than the required dur
                    if count < dur:
                        arr[i-count+1 : i+1] = 0
                    
                    # reset count
                    count = 0
        
        arr_new = arr
        
        return arr_new
    
    def freq_burst(self, ci):
        '''
        A standard version of frequency-by-frequency burst detection:
        Trial mean of the last 1 sec of ISI (the first 1000 points in time series) was used to calculate threshold.
        
        Args:
        ci: channel number of electrode
        
        Returns:
        freq_burst: array of burst events (trials, frequencies, time points)
        '''
        
        # extract epoch data (trials x frequencies x time points)
        spectral_mx, blk_trial, freq_vect = self.extract_spectral_data(ci, extract_type = 'freq')
        
        ntrial, nfreq, epoch_len = spectral_mx.shape[0], spectral_mx.shape[1], spectral_mx.shape[2]
        all_burst = []
        
        for i in range(ntrial):
            spec_v = spectral_mx[i,:,:] # (frequencies x time points)
            
            mn_freq_amp = np.mean(spec_v[:,0:1000], axis = 1)
            sd_freq_amp = np.std(spec_v[:,0:1000], axis = 1)
                
            # calculated the threshold based on mean and std
            threshold = sd_freq_amp*2 + mn_freq_amp
            
            for f in range(nfreq):
                
                cut_thr = (spec_v[f,:] > threshold[f]).astype(int)
                dur     = np.round(3*(1/freq_vect[0,f])*1000)
            
                # wipe out burst lasts no longer than 3 cycles
                freq_burst = self.filter_burst(cut_thr,dur)
                    
                # append result
                all_burst = np.append(all_burst,freq_burst)
        
        freq_burst = all_burst.reshape([ntrial, nfreq, epoch_len])
        
        return freq_burst
    
    def gamma_burst(self, ci, gamma_type = 'average'):
        '''
        A standard version of gamma burst detection:
        Trial mean of the last 1 sec of ISI (the first 1000 points in time series) was used to calculate threshold.
        
        Args:
        ci: channel number of electrode
        gamma_type: the method used to get gamma activity
                    = 'average': get the gamma activity by averaging the amplitude within a frequency band
                    = 'decomp': get tha gamma activity by extracting band pass result made by Hilbert transform
        
        Returns:
        gamma_burst: array of burst events (trials, 2, time points)
        '''
        NBG_epoch, BBG_epoch = self.get_gamma(ci, gamma_type)
        
        # get the threshold
        mn_NBG_amp    = np.mean(NBG_epoch[:,0:1000], axis = 1)
        std_NBG_amp   = np.std(NBG_epoch[:,0:1000], axis = 1)
        threshold_NBG = 2*mn_NBG_amp + std_NBG_amp
                
        mn_BBG_amp    = np.mean(BBG_epoch[:,0:1000], axis = 1)
        std_BBG_amp   = np.std(BBG_epoch[:,0:1000], axis = 1)
        threshold_BBG = 2*mn_BBG_amp + std_BBG_amp
        
        # get duration requirement
        if gamma_type == 'average':
            dur_NBG = np.round(3*(1/20)*1000)
            dur_BBG = np.round(3*(1/70)*1000)
        elif gamma_type == 'decomp':
            dur_NBG = np.round(3*(1/40)*1000)
            dur_BBG = np.round(3*(1/110)*1000)
        else:
            print('Wrong gamma_type!')
        
        # apply detection
        ntrial = NBG_epoch.shape[0]
        all_NBG_burst = []
        all_BBG_burst = []
        
        for i in range(ntrial):
            
            binary_NBG = (NBG_epoch[i,:] > threshold_NBG[i]).astype(int)
            binary_BBG = (BBG_epoch[i,:] > threshold_BBG[i]).astype(int)
            
            NBG_burst  = self.filter_burst(binary_NBG, dur_NBG)
            BBG_burst  = self.filter_burst(binary_BBG, dur_BBG)
            
            # append burst
            all_NBG_burst = np.append(all_NBG_burst, NBG_burst)
            all_BBG_burst = np.append(all_BBG_burst, BBG_burst)
            
        NBG_burst_epoch = all_NBG_burst.reshape([ntrial, NBG_epoch.shape[1]])
        BBG_burst_epoch = all_BBG_burst.reshape([ntrial, BBG_epoch.shape[1]])
        
        return NBG_burst_epoch, BBG_burst_epoch
    
    def gamma_burst_explore():
        
        pass
    
    def gamma_burst_global():
        
        pass