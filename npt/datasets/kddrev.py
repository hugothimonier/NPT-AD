import os

import pandas as pd
import numpy as np

from npt.datasets.base import BaseDataset

class KddRevDataset(BaseDataset):

    def __init__(self, c):
        super().__init__(
            fixed_test_set_index=None)
        self.c = c

        self.num_target_cols = []
        self.is_data_loaded = False
        self.tmp_file_names = ['kddcup.data_10_percent.gz']

        self.ad = True

    def load(self):

        names_file = os.path.join(self.c.data_path, 'kdd_names.csv')
        data_file = os.path.join(self.c.data_path, 'kddcup.data_10_percent.gz')

        df_colnames = pd.read_csv(names_file, skiprows=1, sep=':', names=['f_names', 'f_types'])
        df_colnames.loc[df_colnames.shape[0]] = ['status', ' symbolic.']
        df = pd.read_csv(data_file, header=None, names=df_colnames['f_names'].values)
        df_symbolic = df_colnames[df_colnames['f_types'].str.contains('symbolic.')]
        df_continuous = df_colnames[df_colnames['f_types'].str.contains('continuous.')]
        
        if not self.c.exp_keep_categorical_features:
            print('Removing categorical features')
            to_remove = list(df_symbolic['f_names'])
            to_remove.remove('status') ##keep target
            df.drop(to_remove, axis=1, inplace=True)

        df_keys = df.keys()
        self.N, self.D = df.shape
        
        self.num_features = []
        for cont in df_continuous['f_names']:
            self.num_features.append(df_keys.get_loc(cont))
            
        self.cat_features = []
        if self.c.exp_keep_categorical_features:
            for cat in df_symbolic['f_names']:
                self.cat_features.append(df_keys.get_loc(cat))
                
            if self.c.exp_cat_as_num_features:
                print('Considering categorical features as numerical features')
                self.num_features = list(range(self.D - 1))
                self.cat_features = []
            
        print('There are {} features in total, with {} categorical'
              'and {} numerical features.'.format(len(df.columns)-1,
                                                 len(self.cat_features),
                                                 len(self.num_features)))

        self.target = np.where(df['status'] == 'normal.', 1, 0)
        self.data_table = df.iloc[:, :-1].to_numpy()
        self.anom_samples = self.data_table[self.target == 0]
        self.norm_samples = self.data_table[self.target == 1]

        np.random.seed(self.c.np_seed)
        rp = np.random.permutation(len(self.anom_samples))
        rp_cut = rp[:24319]
        self.anom_samples = self.anom_samples[rp_cut]

        self.norm_samples = np.c_[self.norm_samples, 
                                  np.zeros(self.norm_samples.shape[0])]
        self.anom_samples = np.c_[self.anom_samples, 
                                  np.ones(self.anom_samples.shape[0])]
        
        self.ratio = (100.0 * (0.5*len(self.norm_samples)) / 
        ((0.5*len(self.norm_samples)) + len(self.anom_samples)))

        self.data_table = np.concatenate((self.norm_samples, self.anom_samples),
                                        axis=0)


        self.N, self.D = self.data_table.shape
        self.cat_target_cols = [self.D - 1]  # Anomaly Detection

        print('There are {} features in total, with {} categorical'
        ' and {} numerical features.'.format(self.D-1,
                                            len(self.cat_features),
                                            len(self.num_features),
                                            ))


        self.missing_matrix = np.zeros((self.N, self.D), dtype=np.bool_)
        self.is_data_loaded = True
        self.tmp_file_or_dir_names = ['kddrev']
        self.num_normal = len(self.norm_samples)
        self.is_data_loaded = True
