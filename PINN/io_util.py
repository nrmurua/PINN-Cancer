import os
import numpy as np
import torch
import glob

def load_data(path, device='cpu'):
    files = glob.glob(path + '*.npy')
    
    data = {}

    for file in files:
        data[file[-5]] = torch.FloatTensor(np.array(np.load(file))).to(device) 

    return data

