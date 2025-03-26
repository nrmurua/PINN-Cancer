import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os

class EvaluatorODE():
    def __init__(self, full_t, data_full, data_train, device='cpu'):
        super(EvaluatorODE, self).__init__()
        
        self.t = full_t

        self.data_train = data_train
        self.full_data = data_full
        
        self.device = device

    def evaluate(self, model, ploting=True, save_path=None):
        with torch.no_grad():
            sol = model(self.t)

        metrics = {
            'N': {},
            'T': {},
            'I': {}
        }

        labels = ['N', 'T', 'I']

        sol_dict = {}

        for i, label in enumerate(labels):
            print(label)
            metrics[label]['MSE'] = torch.mean((self.full_data[label] - sol[:,i]) ** 2)
            metrics[label]['MAE'] = torch.mean(torch.abs(self.full_data[label] - sol[:,i]))
            metrics[label]['RMSE'] = torch.sqrt(metrics[label]['MSE'])
            metrics[label]['R2_Score'] = (1 - (torch.sum((self.full_data[label] - sol[:, i])**2) / 
                             torch.sum((self.full_data[label] - torch.mean(self.full_data[label]))**2)))

            sol_dict[label] = sol[:,i].cpu().numpy()

        print(sol_dict)

        if ploting:
            self.plot_comparison(sol_dict, save_path)

        return metrics, sol_dict
    
    def plot_comparison(self, sol, save_path=None):
        plt.figure(figsize=(15,10))
        labels = ['N', 'T', 'I']

        last_train_time = self.data_train['t'][-1].cpu().numpy()

        for i, label in enumerate(labels):
            plt.subplot(3, 1, i+1)
            plt.plot(
                self.data_train['t'].cpu().numpy(),
                self.data_train[label].cpu().numpy(),
                'o',
                color='red',
                label='Training Data',
                alpha=0.7,
                markersize=4
            )
            plt.plot(self.t.cpu().numpy(), self.full_data[label].cpu().numpy(), '-', label='Full Data', alpha=0.4)
            plt.plot(self.t.cpu().numpy(), sol[label], 'b-', label='PINN  Prediction')

            plt.axvline(x=last_train_time, color='red', linestyle=':', label='Training/Prediction Boundary')

            plt.title(f'{labels[i]} Dynamics')
            plt.xlabel('Time')
            plt.ylabel(labels[i])
            plt.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()

        plt.show()
        plt.close()


    def save_metrics(self, model, metrics, save_path):
        metrics_path = os.path.join(save_path, "metrics.txt")
        with open(metrics_path, "w") as f:
            f.write(f"Best loss: {model.best_loss:.6f}\n")
            f.write(f"Best epoch: {model.best_epoch:.6f}\n\n")
            f.write("\nFull Data Metrics:\n")

            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")

            f.write("\nLearned Parameters:\n")
            params = model.get_eq_params()
            for name, param in params.items():
                f.write(f"{name}: {param.data.cpu().numpy()}\n")