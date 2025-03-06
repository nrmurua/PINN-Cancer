import os
import numpy as np
import torch
import glob

def load_data(train_path, full_path, device='cpu'):
    t_files = glob.glob(train_path)
    f_files = glob.glob(full_path)
    
    train_data = []
    full_data = []

    for file in t_files:
        data = np.load(file)
        train_data.append(data)

    for file in f_files:
        data = np.load(file)
        full_data.append(data)

    return (
        torch.FloatTensor(np.array(train_data)).to(device),
        torch.FloatTensor(np.array(full_data)).to(device)
    )

