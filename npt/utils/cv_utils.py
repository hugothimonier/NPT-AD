"""Cross-validation utils."""

from collections import Counter
from enum import IntEnum

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from random import Random

class DatasetMode(IntEnum):
    """Used in batching."""
    TRAIN = 0
    VAL = 1
    TEST = 2


DATASET_MODE_TO_ENUM = {
    'train': DatasetMode.TRAIN,
    'val': DatasetMode.VAL,
    'test': DatasetMode.TEST
}

DATASET_ENUM_TO_MODE = {
    DatasetMode.TRAIN: 'train',
    DatasetMode.VAL: 'val',
    DatasetMode.TEST: 'test'
}

def get_class_reg_train_val_test_splits_ad(label_rows, c,
                                           num_normal:int=None,
                                           num_anom_inference:int=0):

    N = len(label_rows)
    # as by construction anomalies are the last rows of label rows,
    # from label_row[num_normal:N] are anomalies
    if not c.anomalies_in_inference:
        train_indices, val_indices = train_test_split(np.arange(num_normal), test_size=0.5,
                                        random_state=c.np_seed, shuffle=True)
    else:
        assert num_anom_inference > 0, ('One cannot set anomalies_in_inference to True'
                                        ' and not specify the number of anomalies in inference.')
        train_indices, val_indices = train_test_split(np.arange(num_anom_inference,
                                                                num_normal), test_size=0.5,
                                                    random_state=c.np_seed, shuffle=True)
        train_indices = np.concatenate((np.arange(num_anom_inference), train_indices),
                                                axis=0)

    if c.exp_contamination_share_train == 0:
        #add anomalies to val and test (which will be the same)
        test_indices = np.concatenate((val_indices, np.arange(num_normal, N)),
                                                axis=0)
        val_indices = test_indices.copy()
    else:
        anom = np.arange(num_normal, N)
        num_contamination = round(c.exp_contamination_share_train * len(anom))
        anom_indices = np.arange(len(anom))
        
        train_anom_indices = np.random.choice(anom, size=num_contamination,
                                                replace=False)
        val_anom_indices = np.delete(anom_indices, train_anom_indices)
        
        train_anom = anom[train_anom_indices]
        val_anom = anom[val_anom_indices]
        
        train_indices = np.concatenate((train_indices, train_anom), axis=0)
        test_indices = np.concatenate((val_indices, val_anom),
                                                axis=0)
        val_indices = test_indices.copy()
        

    Random(c.torch_seed).shuffle(test_indices)
    Random((c.torch_seed) * 2).shuffle(val_indices)

    return train_indices, val_indices, test_indices

def get_n_cv_splits(c):
    if not c.ad:
        return int(1 / c.exp_test_perc)  # Rounds down
    else:
        return c.exp_n_runs
