import os
import scipy.io as sio
import mat73
import numpy as np
from functools import reduce

class SpectralBasic():
    ''' Basic process of ECoG spectral data.
    
    Attributes:
       sbj_name: code of subject, e.g. 'YBN'
       blk_name: list of block number, e.g. ['036', '038']
       task_name: 'vis_tone_only' or 'vis_contrast_recall'
       home_dir: path of directory
       master_dict: a dictionary of master_vars
       event_dict: a dictionary of events_info
    
    Methods:
       select_channel: update master_dict and event_dict, extract good channel or good visual channel
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
    
    def select_channel(self, elec_type = 'vis'):
        '''
        1) make master dictionary and event dictionary for a given task and save to self
        2) extract good channels or good channels within visual cortex
        
        Args:
        elec_type : good electrodes type, 'vis' = good electrodes within visual cortex
        
        Returns:
        elec_list : a list of good electrodes with the tpye requested
        '''
        
        # visual electrodes dictionary
        vis_elec = {'YBE': np.concatenate((np.arange(35,61),np.asarray([62]),np.arange(64,68),np.arange(70,83)),axis=0),
                    'YBG': np.concatenate((np.arange(36,40),np.arange(41,47),np.arange(48,59),np.arange(91,99),np.asarray([107]),np.arange(109,115)),axis=0),
                    'YBI': np.arange(105,129),
                    'YBJ': np.arange(59,91),
                    'YBN': np.arange(97,129),
                    'YCP': np.arange(65,87)}
        
        # extract bad electrodes list
        elec_del    = []
        
        for i in range(len(self.blk_name)):
            block_name = self.blk_name[i]
            
            # extract all the master_vars of a given task and store in master_dict
            master_path = self.home_dir + '/neuralData/originalData/{}/master_{}_{}_{}.mat'.format(
                                          self.sbj_name,self.task_name,self.sbj_name,block_name)
            master_file = sio.loadmat(master_path)    
            self.master_dict[block_name] = master_file['master_vars']
            
            # extract all the event_info of a given task and store in event_dict
            event_path  = self.home_dir + '/Results/{}/{}/{}/task_events_{}_{}_{}.mat'.format(
                                          self.task_name,self.sbj_name,block_name,self.task_name,self.sbj_name,block_name)
            event_file  = sio.loadmat(event_path)
            self.event_dict[block_name] = event_file['events_info']
            
            # number of recorded channels
            nchan = self.master_dict[block_name]['nchan'][0][0][0][0]
            
            if self.master_dict[block_name]['badchan'][0][0].size != 0:
                badchan = self.master_dict[block_name]['badchan'][0][0][0]
            else:
                badchan = []
    
            if self.master_dict[block_name]['refchan'][0][0].size != 0:
                refchan = self.master_dict[block_name]['refchan'][0][0][0]
            else:
                refchan = []
        
            if self.master_dict[block_name]['epichan'][0][0].size != 0:
                epichan = self.master_dict[block_name]['epichan'][0][0][0]
            else:
                epichan = []
    
            elec_del = np.append(elec_del,badchan)
            elec_del = np.append(elec_del,refchan)
            elec_del = np.append(elec_del,epichan)
        
        
        elec_all = np.arange(1,nchan)
        elec_good = np.setxor1d(elec_all,elec_del)
        
        
        subject_name = self.sbj_name
        
        # get good channels in the visual cortex
        elec_vis = reduce(np.intersect1d, (vis_elec[subject_name], elec_good))
        
        if elec_type == 'vis':
            elec_list = elec_vis
        elif elec_type == 'all':
            elec_list = elec_good
        else:
            print('Wrong electrode type!')
            
        return elec_list
    
    def extract_spectral_data(self, ci, extract_type = 'freq'):
        '''
        Extract spectral data of all trials in a given task
        
        Args:
        ci: channel number of the electrode
        extract_type: the spectral_data you want to extract
                      = 'freq': decomposed by frequency-wise, 2Hz resolution
                      = 'band': decomposed by band-pass, Hilbert transform
        Returns:
        spectral_mx: spectral data of all block of a given task (trial number x frequencies x time points)
        blk_trial: a list of trial numbers of blocks of a given task
        freq_vect: a numpy array of frequency points
        '''
        
        blk_trial   = [] # trial number for each block
        trial_index = 0  # trial index for all blocks
        
        for iblk in range(len(self.blk_name)):
            
            block_name = self.blk_name[iblk]
            
             # extract master and event file back
            master_vars = self.master_dict[block_name]
            events_info = self.event_dict[block_name]
            
            # define epoch window
            # sample rate - tf amplitude data is downsampled
            ecog_srate = self.master_dict[block_name]['ecog_srate'][0][0][0][0]
            compress   = self.master_dict[block_name]['compress'][0][0][0][0]
            
            tf_srate  = round(ecog_srate/compress)
            
            #test_epoch_prestim  = np.floor(1.0*tf_srate) #old version
            # new version: include ISI
            test_epoch_prestim  = np.floor(2.5*tf_srate)
            test_epoch_poststim = np.ceil(5.0*tf_srate)
            test_epoch_norm     = np.floor(0.5*tf_srate)
            
            # define trigger
            test_trigger = np.round(events_info['test_tone_onset'][0][0]/compress)
                       
            # load spectral data
            Spec_dir = os.path.expanduser(master_vars['Spec_dir'][0][0][0])
            
            if extract_type == 'freq':                
                # extract TF amplitude
                TF_path = Spec_dir + '/TF_decomp_CAR_{}_{}_{}.mat'.format(self.sbj_name,block_name,str(ci))
                TF_data = mat73.loadmat(TF_path)['band']
                TF_amp  = TF_data['tf_data']['amplitude'].T
                
                # frequency information
                freq_vect = TF_data['freq'].astype(int)
                
            elif extract_type == 'band':
                # frequency information
                freq_vect = np.array([[20, 60], [70, 150]])
                
                # extract band amplitude
                Band_path = Spec_dir + '/TBand_decomp_CAR_{}_{}_{}.mat'.format(self.sbj_name,block_name,str(ci))
                Band_data = sio.loadmat(Band_path)['band']
                NBG_power = Band_data['NBG_power'][0][0]
                BBG_power = Band_data['BBG_power'][0][0]
                
                TF_amp = np.append(NBG_power, BBG_power, axis = 0)
                
            else:
                print('Wrong extract type!')
            
            # append trial number for each block to 'blk_trial'
            blk_trial.append(test_trigger.shape[1])
            
            # extract epoch
            for iepoch in range(test_trigger.shape[1]):
                
                # extract epoch
                epoch_start = (test_trigger[0][iepoch]-test_epoch_prestim-1).astype(int)
                epoch_end   = (test_trigger[0][iepoch]+test_epoch_poststim).astype(int)
                tmp_epoch = TF_amp[:,epoch_start:epoch_end]
                
                # define normalization base (pre-tone 0.5 sec)
                norm_start = (test_epoch_prestim-test_epoch_norm-1).astype(int)
                norm_end   = (test_epoch_prestim).astype(int)
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
                    
        return spectral_mx, blk_trial, freq_vect
    
    def get_gamma(self, ci, compute_method = 'average'):
        '''
        Extract gamma activities (NBG and BBG) array by the method indicated. 
        
        Args:
        ci: channel number of the electrode
        compute_method: method to extract NBG and BBG
                        = 'average': get the gamma activity by averaging the amplitude within a frequency band
                        = 'decomp': get tha gamma activity by extracting band pass result 
                                    made by Hilbert transform (done in MATLAB)
        Returns:        
        NBG_epoch: array of NBG amplitude (trials, time points)
        BBG_epoch: array of BBG amplitude (trials, time points)
        '''
        
        if compute_method == 'average':
            spectral_mx, blk_trial, freq_vect = self.extract_spectral_data(ci, extract_type = 'freq')
            
            # get the averaged value of a given frequency band
            NBG_start = np.where(freq_vect == 20)[1][0]
            NBG_end   = np.where(freq_vect == 60)[1][0] + 1                
            NBG_epoch = np.mean(spectral_mx[:,NBG_start:NBG_end,:], axis = 1)
                
            BBG_start = np.where(freq_vect == 70)[1][0]
            BBG_end   = np.where(freq_vect == 150)[1][0] + 1                
            BBG_epoch = np.mean(spectral_mx[:,BBG_start:BBG_end,:], axis = 1)
                
        elif compute_method == 'decomp':
            spectral_mx, blk_trial, freq_vect = self.extract_spectral_data(ci, extract_type = 'band')
            
            NBG_epoch = spectral_mx[:,0,:]
            BBG_epoch = spectral_mx[:,1,:]
        else:
            print('Wrong compute type!')
        
        return NBG_epoch, BBG_epoch   
