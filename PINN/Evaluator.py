import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error

class Evaluator():
    def __init__(self, t_domain, x_domain, data_train, data_full, device='cpu'):
        super(Evaluator, self).__init__()

        full_t = torch.linspace(0, t_domain[0], t_domain[1], device=device)
        x_samples = int((x_domain[1] - x_domain[0])/x_domain[2] + 1)
        full_x = torch.linspace(x_domain[0], x_domain[1], x_samples, device=device)

        self.full_grid = torch.cartesian_prod(full_t, full_x)
        self.t = full_t.shape[0]
        self.x = full_x.shape[0]
        self.data_train = data_train

        self.full_data = {
            'N': data_full['N'].view(-1),
            'T': data_full['T'].view(-1),
            'I': data_full['I'].view(-1)
        }

        self.device = device

    def evaluate(self, model):
        with torch.no_grad():
            sol = model(self.full_grid)

        metrics = {
            'N': {},
            'T': {},
            'I': {}
        }

        labels = ['N', 'T', 'I']

        sol_matrix = {}

        for i, label in enumerate(labels):
            metrics[label]['MSE'] = torch.mean((self.full_data[label] - sol[:,i]) ** 2)
            metrics[label]['MAE'] = torch.mean(torch.abs(self.full_data[label] - sol[:,i]))
            metrics[label]['RMSE'] = torch.sqrt(metrics[label]['MSE'])
            metrics[label]['R2_Score'] = (1 - (torch.sum((self.full_data[label] - sol[:, i])**2) / 
                             torch.sum((self.full_data[label] - torch.mean(self.full_data[label]))**2)))

            sol_matrix[label] = sol[:,i].view(self.t, self.x).cpu().numpy()

        return metrics, sol_matrix