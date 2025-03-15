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

def save_model(model, dir_path, file):
    os.makedirs(dir_path, exist_ok=True)
    model_path = os.path.join(dir_path, file)
    torch.save(model.state_dict(), model_path)

    return model_path

def load_model(model, path):
    model.load_state_dict(torch.load(path))  
    
def print_metrics(metrics):
    print(f"Metrics of the trained model: \n")

    for label, metric in metrics.items():
        print(f"{label} Variable Metrics: \n")
        for metric_name, value in metric.items():
            print(f"  {metric_name}: {value:.6f}")
        print()



def load_ODE_data(path, data_case, samples_index, device='cpu'):
    data_train = {}
    labels = ['N', 'T', 'I']

    for label in labels:
        file = f'{label}_{data_case}.npy'
        file_path = os.path.join(path, file)
        data_train[label] = torch.FloatTensor(np.array(np.load(file_path))[samples_index]).to(device)

    return data_train