import torch
import torch.nn as nn
import os

class AdaptiveLossWeights(nn.Module):
    def __init__(self, init_weights, device):
        super(AdaptiveLossWeights, self).__init__()

        self.const_weights = torch.tensor([
            init_weights['params'],
            init_weights['init']
        ], device=device)

        self.device = device


    def forward(self, losses_dict):
        data_loss = losses_dict['L_data']
        physics_loss = losses_dict['L_physics']

        eps = 1e-8

        log_losses = torch.log(torch.stack([data_loss, physics_loss]) + eps)
        weights = torch.softmax(log_losses, dim=0)

        # Alternativa (Método 2): inverso proporcional (opcional)
        # losses = torch.stack([data_loss, physics_loss])
        # weights = losses / losses.sum()

        # Pérdida total
        total_loss = (
            weights[0] * data_loss 
            + weights[1] * physics_loss 
            + self.const_weights[0] * losses_dict['L_params']
            + self.const_weights[1] * losses_dict['L_init']
        )

        output = {
            'data': weights[0].item(), 
            'physics': weights[1].item(), 
            'params': self.const_weights[0].item(), 
            'init': self.const_weights[1].item()
        }

        return total_loss, output
    
    