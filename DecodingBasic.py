import os
import scipy.io as sio
import numpy as np
import mat73
from functools import reduce

class DataBasic():
    ''' Get spectral data for decoding
    
    Attributes:
       sbj_name: code of subject, e.g. 'YBN'
       blk_name: list of block number, e.g. ['036', '038']
       data_usage: 'train' or 'generalize'
       home_dir: path of directory
       master_dict: a dictionary of master_vars
       event_dict: a dictionary of events_info
    
    Methods:
       elec_record: extract good vis channel or good resp channel
       extract_spectral_data: extract epoch of spectral data of a given task (all blocks will be concatenated together)
       get_gamma: extract NBG activity and BBG activity of epoch    
    '''
    def __init__(self, sbj_name, blk_name, task_name):
        self.sbj_name = sbj_name
        self.blk_name = blk_name
        self.task_name = task_name
        self.home_dir = os.path.expanduser('~/Documents/MATLAB/ECoG')
        
        self.master_dict = {}
        self.event_dict  = {}
        
        # master_dict & event_dict
        for i in range(len(self.blk_name)):
            block_name = self.blk_name[i]
            master_path = self.home_dir + '/neuralData/originalData/{}/master_{}_{}_{}.mat'.format(
                                          self.sbj_name,self.task_name,self.sbj_name,block_name)
            master_file = sio.loadmat(master_path)    
            self.master_dict[block_name] = master_file['master_vars']
            
            # extract all the event_info of a given task and store in event_dict
            event_path  = self.home_dir + '/Results/{}/{}/{}/task_events_{}_{}_{}.mat'.format(
                                          self.task_name,self.sbj_name,block_name,self.task_name,self.sbj_name,block_name)
            event_file  = sio.loadmat(event_path)
            self.event_dict[block_name] = event_file['events_info']
            
    def elec_record(self, elec_type = 'vis'):
        '''
        extract electrodes (visual cortex or visually responsive) that are good in all following tasks:
        1) vis_tone_only
        2) vis_contrast_recall
        3) vis_contrast_only
        4) vis_contrast_gamma
        
        Args:
        elec_type : good electrodes type, 
                   'vis' = good electrodes within visual cortex,
                   'resp' = visual responsive electrodes
        Returns:
        elec_list : a list of good electrodes with the tpye requested
        '''
        vis_elec = {
            'YBN' : np.array([97, 98, 99, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,\
                              114, 115, 116, 117, 118, 119, 120, 121, 122, 124, 125, 126, 127])
            }
        
        resp_elec = {
            'YBN': np.array([97, 98, 99, 101, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,\
                             117, 118, 119, 120, 121, 122, 124, 125, 126, 127])
            }
        if elec_type == 'vis':
            elec_list = vis_elec[self.sbj_name]
        elif elec_type == 'resp':
            elec_list = resp_elec[self.sbj_name]
        else:
            print('Wrong electrode type!')
            
        return elec_list
    
    def extract_training_data(self, ci):
        '''
        Extract spectral data of all trials in a training task.
        Training tasks include:
        1) study phase of vis_contrast_recall
        2) stimulus window of vis_contrast_only
        3) vis_contrast_gamma if collected
        
        Args:
        ci: channel number of the electrode
        
                    
        Returns:
        spectral_mx: spectral data of all block of a given task (trial number x frequencies x time points)
        blk_trial: a list of trial numbers of blocks of a given task
        freq_vect: a numpy array of frequency points
        '''
        blk_trial = []
        trial_index = 0
        
        for iblk in range(len(self.blk_name)):
            block_name = self.blk_name[iblk]
            
            # extract master and event file back
            master_vars = self.master_dict[block_name]
            events_info = self.event_dict[block_name]
            
            # sample rate - tf amplitude data is downsampled
            ecog_srate = self.master_dict[block_name]['ecog_srate'][0][0][0][0]
            compress   = self.master_dict[block_name]['compress'][0][0][0][0]            
            tf_srate  = round(ecog_srate/compress)
            
            # define epoch window
            epoch_prestim  = np.floor(1.0*tf_srate) 
            epoch_poststim = np.ceil(1.0*tf_srate)
            epoch_norm     = np.floor(0.5*tf_srate)
            
            # trigger
            if self.task_name == 'vis_contrast_gamma':
                trigger = np.round(events_info['trial_onset'][0][0]/compress)
            elif self.task_name == 'vis_contrast_recall':
                trigger = np.round(events_info['study_tone_trial_onset'][0][0]/compress)
            elif self.task_name == 'vis_contrast_only':
                trigger = np.round(events_info['test_tone_onset'][0][0]/compress)
            else:
                print('Wrong task!')
                
            # load spectral data
            Spec_dir = os.path.expanduser(master_vars['Spec_dir'][0][0][0])
            TF_path = Spec_dir + '/TF_decomp_CAR_{}_{}_{}.mat'.format(self.sbj_name,block_name,str(ci))
            
            try:
                TF_amp = mat73.loadmat(TF_path)['band']['tf_data']['amplitude'].T
            except:
                TF_amp = sio.loadmat(TF_path)['band']['tf_data'].item()['amplitude'].item()
            
            # store trial number information for each block
            ntrial = trigger.shape[1]
            blk_trial.append(ntrial)
            
            # contrast information              
            if self.task_name == 'vis_contrast_recall':
                blk_contrast_info = events_info['trial_contrast'].item().reshape([ntrial*2,])
                add_contrast = blk_contrast_info[0:30]
            else:
                blk_contrast_info = events_info['trial_contrast'].item().reshape([ntrial,])
                add_contrast = blk_contrast_info
                
            if iblk == 0:
                contrast_info = add_contrast
            else:
                contrast_info = np.append(contrast_info, add_contrast)
            
            # extract epoch
            for iepoch in range(ntrial):
                # extract epoch
                epoch_start = (trigger[0][iepoch] - epoch_prestim - 1).astype(int)
                epoch_end   = (trigger[0][iepoch] + epoch_poststim).astype(int)
                tmp_epoch = TF_amp[:,epoch_start:epoch_end]
                
                # define normalization base (pre-tone 0.5 sec)
                norm_start = (epoch_prestim - epoch_norm - 1).astype(int)
                norm_end   = (epoch_prestim).astype(int)
                norm_vect  = np.mean(tmp_epoch[:,norm_start:norm_end],axis = 1)
                
                # normalization
                norm_vect  = norm_vect.reshape([len(norm_vect),1])
                tmp_normed = tmp_epoch/norm_vect
        
                # append data
                tmp_normed = np.expand_dims(tmp_normed, axis = 0)
                if trial_index == 0:
                    spectral_mx = tmp_normed
                else:
                    spectral_mx = np.append(spectral_mx, tmp_normed, axis = 0)
                    
                trial_index += 1
              
        if contrast_info[0] < 10:
            contrast_info = (100*contrast_info).astype(int)  
            
        return spectral_mx, contrast_info, blk_trial
    
    def extract_testing_data(self, ci):
        '''
        Extract spectral data of all trials in a testing task.
        Testing tasks include:
        1) test phase of vis_contrast_recall
        2) vis_tone_only (as control condition)
        
        Args:
        ci: channel number of the electrode
        
                    
        Returns:
        spectral_mx: spectral data of all block of a given task (trial number x frequencies x time points)
        blk_trial: a list of trial numbers of blocks of a given task
        freq_vect: a numpy array of frequency points
        
        '''
        
        blk_trial = []
        trial_index = 0
        
        for iblk in range(len(self.blk_name)):
            block_name = self.blk_name[iblk]
            
            # extract master and event file back
            master_vars = self.master_dict[block_name]
            events_info = self.event_dict[block_name]
            
            # sample rate - tf amplitude data is downsampled
            ecog_srate = self.master_dict[block_name]['ecog_srate'][0][0][0][0]
            compress   = self.master_dict[block_name]['compress'][0][0][0][0]            
            tf_srate  = round(ecog_srate/compress)
            
            # define epoch window
            epoch_prestim  = np.floor(1.0*tf_srate) 
            epoch_poststim = np.ceil(5.0*tf_srate)
            epoch_norm     = np.floor(0.5*tf_srate)
            
            # trigger
            trigger = np.round(events_info['test_tone_onset'][0][0]/compress)
            
            # load spectral data
            Spec_dir = os.path.expanduser(master_vars['Spec_dir'][0][0][0])
            TF_path = Spec_dir + '/TF_decomp_CAR_{}_{}_{}.mat'.format(self.sbj_name,block_name,str(ci))
            TF_amp = mat73.loadmat(TF_path)['band']['tf_data']['amplitude'].T
            
            # store trial number information for each block
            ntrial = trigger.shape[1]
            blk_trial.append(ntrial)
            
            # contrast information              
            if self.task_name == 'vis_contrast_recall':
                blk_contrast_info = events_info['trial_contrast'].item().reshape([ntrial*2,])
                add_contrast = blk_contrast_info[30:]
            else:
                blk_contrast_info = events_info['trial_contrast'].item().reshape([ntrial,])
                add_contrast = blk_contrast_info
                
            if iblk == 0:
                contrast_info = add_contrast
            else:
                contrast_info = np.append(contrast_info, add_contrast)
            
            # extract epoch
            for iepoch in range(ntrial):
                # extract epoch
                epoch_start = (trigger[0][iepoch] - epoch_prestim - 1).astype(int)
                epoch_end   = (trigger[0][iepoch] + epoch_poststim).astype(int)
                tmp_epoch = TF_amp[:,epoch_start:epoch_end]
                
                # define normalization base (pre-tone 0.5 sec)
                norm_start = (epoch_prestim - epoch_norm - 1).astype(int)
                norm_end   = (epoch_prestim).astype(int)
                norm_vect  = np.mean(tmp_epoch[:,norm_start:norm_end],axis = 1)
                
                # normalization
                norm_vect  = norm_vect.reshape([len(norm_vect),1])
                tmp_normed = tmp_epoch/norm_vect
        
                # append data
                tmp_normed = np.expand_dims(tmp_normed, axis = 0)
                if trial_index == 0:
                    spectral_mx = tmp_normed
                else:
                    spectral_mx = np.append(spectral_mx, tmp_normed, axis = 0)
                    
                trial_index += 1
                
        if contrast_info[0] < 10:
            contrast_info = (100*contrast_info).astype(int)  
            
        return spectral_mx, contrast_info, blk_trial
    
    def get_band_value(self,band,spect_mx,ci):
        '''
        Get mean band value (e.g. mean NBG).
        
        Args:
        band: list contains start and end frequency of a band, e.g. [20,60] for NBG.
        spect_mx: numpy array of spectral data, trial number x freq points x time points
        
        Returns:
        band_val: array of band mean values, 1 x time points
        '''
        
        if spect_mx.shape[1] == 200:
            freq_vect = np.arange(2,202,1)
        elif spect_mx.shape[1] == 101:
            freq_vect = np.arange(2,203,2)
        else:
            print('Wrong frequency resolution!')
        
        band_start = np.where(freq_vect == band[0])[0][0]
        band_end   = np.where(freq_vect == band[1])[0][0]
        
        band_val = np.mean(spect_mx[:,band_start:band_end,:],axis=1)
        
        return band_val