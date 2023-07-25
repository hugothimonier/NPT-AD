import os, scipy.io 
from operator import itemgetter

import pandas as pd
import numpy as np

from npt.datasets.base import BaseDataset

class SpeechDataset(BaseDataset):

    '''
    http://odds.cs.stonybrook.edu/speech-dataset/
    '''

    def __init__(self, c):
        super().__init__(
            fixed_test_set_index=None)
        self.c = c

        self.num_target_cols = []
        self.is_data_loaded = False
        self.tmp_file_names = ['speech.mat']

        self.ad = True

    def load(self):

        data = scipy.io.loadmat(os.path.join(self.c.data_path, self.tmp_file_names[0]))
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
        self.cat_target_cols = [self.D - 1]
        self.cat_features = []
        self.num_features = list(range(0, self.D-1))

        self.missing_matrix = np.zeros((self.N, self.D), dtype=np.bool_)
        self.num_normal = len(self.norm_samples)
        self.is_data_loaded = True