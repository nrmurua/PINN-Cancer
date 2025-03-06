import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error
from plots import compare_data_prediction

def evaluate_model(model, t, x, data, dataset_name='Training'):
    with torch.no_grad():
        pred = model(t, x)

    pred_np = pred.cpu().numpy()
    data_np = data.cpu().numpy()

    metrics = {}
    labels = ['N', 'T', 'I']

    for i, label in enumerate(labels):
        mse = mean_squared_error(data_np[:, i], pred_np[:, i])
        mae = mean_absolute_error(data_np[:, i], pred_np[:, i])
        rmse = np.sqrt(mse)
        r2 = 1 - (np.sum((data_np[:, i] - pred_np[:, i])**2) / np.sum((data_np[:, i] - np.mean(data_np[:, i]))**2))

        metrics[label] = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R2 _Score': r2
        }

        print(f"\nModelo Evaluation for {dataset_name} Dataset:")
        for label, metric in metrics.items():
            print(f"{label} variable Metrics:")
            for metric_name, value in metric.items():
                print(f"{metric_name}: {value:.6f}")

        return metrics