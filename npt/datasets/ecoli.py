import os
from scipy.io import arff
from operator import itemgetter

import pandas as pd
import numpy as np

from npt.datasets.base import BaseDataset

class EcoliDataset(BaseDataset):

    '''
    https://archive.ics.uci.edu/ml/datasets/ecoli
    '''

    def __init__(self, c):
        super().__init__(
            fixed_test_set_index=None)
        self.c = c

        self.num_target_cols = []
        self.is_data_loaded = False
        self.tmp_file_names = ['ecoli.data']

        self.ad = True

    def load(self):
        
        filename = os.path.join(self.c.data_path, self.tmp_file_names[0])
        data = pd.read_csv(filename, header=None, sep='\s+')
        self.anom_samples = data[data[8].isin(['omL','imL','imS'])].iloc[:,:-1]
        self.norm_samples = data[~data[8].isin(['omL','imL','imS'])].iloc[:,:-1]

        self.anom_samples = self.anom_samples.drop(8, axis=1)
        self.norm_samples = self.norm_samples.drop(8, axis=1)

        self.norm_samples = np.c_[self.norm_samples, 
                            np.zeros(self.norm_samples.shape[0])]
        self.anom_samples = np.c_[self.anom_samples, 
                                  np.ones(self.anom_samples.shape[0])]

        self.ratio = (100.0 * (0.5*len(self.norm_samples)) / ((0.5*len(self.norm_samples)) +
                                                             len(self.anom_samples)))
        self.data_table = np.concatenate((self.norm_samples, self.anom_samples),
                                         axis=0)
        self.N, self.D = self.data_table.shape
        self.cat_target_cols = [self.D - 1]
        self.cat_features = [0]
        self.num_features = list(range(1, self.D-1))

        self.missing_matrix = np.zeros((self.N, self.D), dtype=np.bool_)
        self.num_normal = len(self.norm_samples)
        self.is_data_loaded = True