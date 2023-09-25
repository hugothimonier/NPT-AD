import os
from operator import itemgetter

import pandas as pd
import numpy as np

from npt.datasets.base import BaseDataset

CAT_FEATURES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
               13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
               23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
               33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
               43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
               53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
               63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
               73, 74, 75, 76, 77, 78, 79, 80, 81, 82,
               83, 84, 85, 86, 87, 88, 89, 90, 91, 92,
               93, 94, 95, 96, 97, 98, 99, 100, 101, 102,
               103, 104, 105, 106, 107, 108, 109, 110, 
               111, 112, 113, 114, 115, 116, 117, 118, 
               119, 120, 121, 122, 123, 124, 125, 126, 
               127, 128, 129, 130, 131, 132, 133, 134, 
               135, 136, 137, 138, 139, 140, 141, 144, 
               145, 146, 147, 148, 149, 150, 151, 152,
               154, 155, 156, 157, 195]

class BackdoorDataset(BaseDataset):

    '''
    https://www.kaggle.com/code/ramantalwar00/eda-metadata-shap-multi-class-split-unsw-nb15
    
    '''
    
        
    def __init__(self, c):
        super().__init__(
            fixed_test_set_index=None)
        self.c = c

        self.num_target_cols = []
        self.is_data_loaded = False
        self.tmp_file_names = ['3_backdoor.npz']

        self.ad = True
        
    def load(self):
        
        filename = os.path.join(self.c.data_path, self.tmp_file_names[0])
        data = np.load(filename, allow_pickle=True)
        self.data_table  = data['X']
        self.target = ((data['y']).astype(np.int32)).reshape(-1)
        
        self.norm_samples = self.data_table [self.target == 0]
        self.anom_samples = self.data_table [self.target == 1]

        self.norm_samples = np.c_[self.norm_samples, 
                            np.zeros(self.norm_samples.shape[0])]
        self.anom_samples = np.c_[self.anom_samples, 
                                  np.ones(self.anom_samples.shape[0])]

        self.ratio = (100.0 * (0.5*len(self.norm_samples)) / ((0.5*len(self.norm_samples)) +
                                                             len(self.anom_samples)))
        self.data_table = np.concatenate((self.norm_samples, self.anom_samples),
                                         axis=0)
        self.N, self.D = self.data_table.shape
        self.cat_features = CAT_FEATURES
        self.num_features = [ele for ele in range(self.D) if ele not in self.cat_features]
        
        if self.c.exp_cat_as_num_features:
            print('Considering categorical features as numerical features')
            self.num_features = list(range(0, self.D-1))
            self.cat_features = []
        
        if not self.c.exp_keep_categorical_features:
            print('Removing categorical features')
            self.data_table = self.data_table[:, self.num_features + [self.D-1]]
            self.num_features = list(range(0, self.D-1))
            self.cat_features = []
            self.D = self.data_table.shape[1]
            
        print(f'There are {self.D-1} feature in total, with {len(self.num_features)}'
              f'continuous features and {len(self.cat_features)} categorical features')
        
        self.cat_target_cols = [self.D - 1]
        self.missing_matrix = np.zeros((self.N, self.D), dtype=np.bool_)
        self.num_normal = len(self.norm_samples)
        self.is_data_loaded = True