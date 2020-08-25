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
                              114, 115, 116, 117, 118, 119, 120, 121, 122, 124, 125, 126, 127]),
            'YBI' : np.array([105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,\
                             120, 121, 122, 123, 124, 126, 127, 128])
            }
        
        resp_elec = {
            'YBN': np.array([97, 98, 99, 101, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,\
                             117, 118, 119, 120, 121, 122, 124, 125, 126, 127]),
            'YBI': np.array([105, 113, 114, 115, 116, 117])
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
        
        band_val = np.mean(spect_mx[:,band_start:band_end+1,:],axis=1)
        
        return band_val
    
    def get_NBG_spect(self,spect_mx,ci,point_num = 5):
        '''
        get values within NBG
        point_num == 5: get 5 points (20Hz, 30Hz, 40Hz, 50Hz, 60Hz)
        point_num == 11: get 11 points (20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60 Hz) 
        '''
        
        if spect_mx.shape[1] == 200:
            freq_vect = np.arange(2,202,1)
        elif spect_mx.shape[1] == 101:
            freq_vect = np.arange(2,203,2)
        else:
            print('Wrong frequency resolution!')
        
        if point_num == 5:
            NBG_1 = np.where(freq_vect == 20)[0][0]
            NBG_2 = np.where(freq_vect == 30)[0][0]
            NBG_3 = np.where(freq_vect == 40)[0][0]
            NBG_4 = np.where(freq_vect == 50)[0][0]
            NBG_5 = np.where(freq_vect == 60)[0][0]
        
            NBG_ind = np.array([NBG_1,NBG_2,NBG_3,NBG_4,NBG_5])
        elif point_num == 11:
            NBG_1 = np.where(freq_vect == 20)[0][0]
            NBG_2 = np.where(freq_vect == 24)[0][0]
            NBG_3 = np.where(freq_vect == 28)[0][0]
            NBG_4 = np.where(freq_vect == 32)[0][0]
            NBG_5 = np.where(freq_vect == 36)[0][0]
            NBG_6 = np.where(freq_vect == 40)[0][0]
            NBG_7 = np.where(freq_vect == 44)[0][0]
            NBG_8 = np.where(freq_vect == 48)[0][0]
            NBG_9 = np.where(freq_vect == 52)[0][0]
            NBG_10 = np.where(freq_vect == 56)[0][0]
            NBG_11 = np.where(freq_vect == 60)[0][0]
            
            NBG_ind = np.array([NBG_1,NBG_2,NBG_3,NBG_4,NBG_5,NBG_6,NBG_7,NBG_8,NBG_9,NBG_10,NBG_11])
        else:
            print('Wrong type!')
            
        spect_NBG = spect_mx[:,NBG_ind,:]
        
        return spect_NBG
    
    def get_BBG_spect(self,spect_mx,ci,point_num = 9):
        '''
        get values within NBG
        point_num == 9: get 9 points (70Hz, 80Hz, 90Hz, 100Hz, 110Hz, 120Hz, 130Hz, 140Hz, 150Hz)
        point_num == 21: get 21 points (4Hz interval) 
        '''
        
        if spect_mx.shape[1] == 200:
            freq_vect = np.arange(2,202,1)
        elif spect_mx.shape[1] == 101:
            freq_vect = np.arange(2,203,2)
        else:
            print('Wrong frequency resolution!')
        
        if point_num == 9:
            BBG_1 = np.where(freq_vect == 70)[0][0]
            BBG_2 = np.where(freq_vect == 80)[0][0]
            BBG_3 = np.where(freq_vect == 90)[0][0]
            BBG_4 = np.where(freq_vect == 100)[0][0]
            BBG_5 = np.where(freq_vect == 110)[0][0]
            BBG_6 = np.where(freq_vect == 120)[0][0]
            BBG_7 = np.where(freq_vect == 130)[0][0]
            BBG_8 = np.where(freq_vect == 140)[0][0]
            BBG_9 = np.where(freq_vect == 150)[0][0]
        
            BBG_ind = np.array([BBG_1,BBG_2,BBG_3,BBG_4,BBG_5,BBG_6,BBG_7,BBG_8,BBG_9])
        elif point_num == 21:
            BBG_1  = np.where(freq_vect == 70)[0][0]
            BBG_2  = np.where(freq_vect == 74)[0][0]
            BBG_3  = np.where(freq_vect == 78)[0][0]
            BBG_4  = np.where(freq_vect == 82)[0][0]
            BBG_5  = np.where(freq_vect == 86)[0][0]
            BBG_6  = np.where(freq_vect == 90)[0][0]
            BBG_7  = np.where(freq_vect == 94)[0][0]
            BBG_8  = np.where(freq_vect == 98)[0][0]
            BBG_9  = np.where(freq_vect == 102)[0][0]
            BBG_10 = np.where(freq_vect == 106)[0][0]
            BBG_11 = np.where(freq_vect == 110)[0][0]
            BBG_12 = np.where(freq_vect == 114)[0][0]
            BBG_13 = np.where(freq_vect == 118)[0][0]
            BBG_14 = np.where(freq_vect == 122)[0][0]
            BBG_15 = np.where(freq_vect == 126)[0][0]
            BBG_16 = np.where(freq_vect == 130)[0][0]
            BBG_17 = np.where(freq_vect == 134)[0][0]
            BBG_18 = np.where(freq_vect == 138)[0][0]
            BBG_19 = np.where(freq_vect == 142)[0][0]
            BBG_20 = np.where(freq_vect == 146)[0][0]
            BBG_21 = np.where(freq_vect == 150)[0][0]
            
            BBG_ind = np.array([BBG_1,BBG_2,BBG_3,BBG_4,BBG_5,BBG_6,BBG_7,BBG_8,BBG_9,BBG_10,\
                                BBG_11,BBG_12,BBG_13,BBG_14,BBG_15,BBG_16,BBG_17,BBG_18,BBG_19,BBG_20,BBG_21])
        else:
            print('Wrong type!')
            
        spect_BBG = spect_mx[:,BBG_ind,:]
        
        return spect_BBG