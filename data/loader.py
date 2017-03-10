
# coding: utf-8

# In[ ]:

import numpy as np

def load_data(file):
    return np.load(file)

def load_samples():
    return load_data('data.npy')

