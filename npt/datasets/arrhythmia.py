import os, scipy.io 
from operator import itemgetter

import pandas as pd
import numpy as np

from npt.datasets.base import BaseDataset

class ArrhythmiaDataset(BaseDataset):

    '''
    https://archive.ics.uci.edu/static/public/5/arrhythmia.zip
    '''
    def __init__(self, c):
        super().__init__(
            fixed_test_set_index=None)
        self.c = c

        self.num_target_cols = []
        self.is_data_loaded = False
        self.tmp_file_names = ['arrhythmia.mat']

        self.ad = True
        self.fixed_test_set_index = None

    def load(self):

        data = scipy.io.loadmat(os.path.join(self.c.data_path, 'arrhythmia.mat'))
        self.data_table  = data['X']  
        self.target = ((data['y']).astype(np.int32)).reshape(-1)

        self.norm_samples = self.data_table [self.target == 0]
        self.anom_samples = self.data_table [self.target == 1]

        self.norm_samples = np.c_[self.norm_samples, 
                                  np.zeros(self.norm_samples.shape[0])]

        self.anom_samples = np.c_[self.anom_samples, 
                                  np.ones(self.anom_samples.shape[0])]
        
        self.ratio = (100.0 * (0.5*len(self.norm_samples)) /
                     ((0.5*len(self.norm_samples)) + len(self.anom_samples)))

        self.data_table = np.concatenate((self.norm_samples, self.anom_samples),
                                         axis=0)

        self.N, self.D = self.data_table.shape

        self.cat_features = [1, 21, 22, 23, 24, 25, 26]
        self.num_features = [x for x in list(range(0, self.D-1)) if x not in self.cat_features]
        
        if self.c.exp_cat_as_num_features:
            print('Considering categorical features as numerical features')
            self.num_features = list(range(0, self.D-1))
            self.cat_features = []
        
        if not self.c.exp_keep_categorical_features:
            print('Removing categorical features')
            self.data_table = self.data_table[:, self.num_features + [self.D-1]]
            self.D = self.data_table.shape[1]
            
        print(f'There are {self.D-1} feature in total, with {len(self.num_features)}'
              f'continuous features and {len(self.cat_features)} categorical features')
        
        self.missing_matrix = np.zeros((self.N, self.D), dtype=np.bool_)

        self.num_normal = len(self.norm_samples)
        self.cat_target_cols = [self.D - 1]  # Anomaly Detection
        self.is_data_loaded = True
        

