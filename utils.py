"""Util functions."""

import numpy as np
from constants import Constants

np.random.seed(0)




def generate_synthetic_data(model_type='MLP', num_samples=50000):
    if model_type == 'MLP':
        features = np.random.random([num_samples, Constants._MLP_NUM_FEATURES])
    elif model_type == 'TCN':
        features = np.random.random(([num_samples, Constants._TCN_LENGTH, Constants._TCN_NUM_CHANNELS]))
    else:
        raise ValueError('Wrong model type.')

    targets = np.random.random([num_samples, Constants._NUM_TARGETS])

    return features, targets
