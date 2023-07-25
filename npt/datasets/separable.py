import os, scipy.io 
from operator import itemgetter

import pandas as pd
import numpy as np

from npt.datasets.base import BaseDataset

class SeparableDataset(BaseDataset):

    def __init__(self, c):
        super().__init__(
            fixed_test_set_index=None)
        self.c = c

        self.num_target_cols = []
        self.is_data_loaded = False
        if c.data_set=='separable':
            self.tmp_file_names = ['separable.npy']
        elif c.data_set=='separable1':
            self.tmp_file_names = ['separable1.npy']
        self.ad = True

    def load(self):

        data = np.load(os.path.join(self.c.data_path, self.tmp_file_names[0]))

        self.data_table  = data[:,:-1]
        self.target = data[:,-1]

        self.norm_samples = self.data_table [self.target == 0]  # 1800 norm
        self.anom_samples = self.data_table [self.target == 1]  # 200 anom
        
        self.norm_samples = np.c_[self.norm_samples, 
                                  np.zeros(self.norm_samples.shape[0])]
        
        self.anom_to_keep = np.c_[self.anom_samples[:100],
                                  np.ones(self.anom_samples[:100].shape[0])]
        self.anom_for_contamination = np.c_[self.anom_samples[100:],
                                  np.ones(self.anom_samples[100:].shape[0])]

        if self.c.anomalies_in_inference and self.c.share_contamination>0.:
            self.num_anom_inference = compute_num_anom(self.c.share_contamination,
                                                 0.5 * len(self.norm_samples))
            max_share = len(self.anom_for_contamination) + (len(self.anom_for_contamination)+
                                                            0.5*len( self.norm_samples))
            err = f'share of anomalies is too high, has to be less or equal to {max_share}'
            assert self.num_anom_inference <= len(self.anom_for_contamination), err
            
            if self.num_anom_inference<100:
                self.anom_samples_inference = self.anom_for_contamination[:self.num_anom_inference]

                self.data_table = np.concatenate((self.anom_samples_inference, 
                                                self.norm_samples,
                                                self.anom_to_keep),
                                                axis=0)
            else:
                self.data_table = np.concatenate((self.anom_for_contamination, 
                                self.norm_samples,
                                self.anom_to_keep),
                                axis=0)

            self.ratio = 100.0 * (0.5*len(self.norm_samples)) / ((0.5*len(self.norm_samples)) +
                                                                  len(self.anom_to_keep))

        else:
            self.data_table = np.concatenate((self.norm_samples,
                                  self.anom_to_keep),
                                 axis=0)
        
            self.ratio = 100.0 * (0.5*len(self.norm_samples)) / ((0.5*len(self.norm_samples)) + 
                                                                 len(self.anom_to_keep))

        self.N, self.D = self.data_table.shape
        self.cat_target_cols = [self.D - 1]  # Anomaly Detection

        #here no categorical features
        self.cat_features = []
        self.num_features = list(range(0, self.D-1)) ##only numerical
        self.missing_matrix = np.zeros((self.N, self.D), dtype=np.bool_)

        self.num_normal = len(self.norm_samples)
        self.is_data_loaded = True

def compute_num_anom(contamination_share, num_norm):
    number_anom = (contamination_share/(1 - contamination_share)) * num_norm
    return int(number_anom)