import os
from scipy.io import arff
from operator import itemgetter

import pandas as pd
import numpy as np

from npt.datasets.base import BaseDataset

class MulcrossDataset(BaseDataset):

    '''
    https://www.openml.org/search?type=data&sort=runs&id=40897&status=active
    '''

    def __init__(self, c):
        super().__init__(
            fixed_test_set_index=None)
        self.c = c

        self.num_target_cols = []
        self.is_data_loaded = False
        self.tmp_file_names = ['mulcross.arff']

        self.ad = True

    def load(self):

        data, _ = arff.loadarff(os.path.join(self.c.data_path, 
                                self.tmp_file_names[0]))
        data = pd.DataFrame(data)
        self.data_table = pd.get_dummies(data.iloc[:,:-1]).to_numpy()
        classes = data.iloc[:, -1].values

        self.norm_samples = self.data_table [classes == b'Normal']
        self.anom_samples = self.data_table [classes == b'Anomaly']

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