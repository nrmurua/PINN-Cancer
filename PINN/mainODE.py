from PINN_ODE import PINN_ODE as pinn
from EvaluatorODE import EvaluatorODE as ev_ODE
from debug_functions import *

from io_util import load_ODE_data, save_model
from plots import plot, plot2D, plot_training_losses

import time 
import torch 
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#############################
##### Debugging options ##### 
#############################

debug_mode = False
debug_printable = False
debug_plot = False
debug_train_printable = True
disable_train = False


if __name__ == "__main__":  
    ########################################
    ##### PINN Architecture parameters ##### 
    ########################################

    nn_arch = {
        'neurons': 25,
        'layers': 5    
    }
    

    ############################################
    ##### Time domain params               #####
    #####                                  #####
    ##### time_params = [ <limit_time>,    #####
    #####                 <full_samples> ] #####
    #####                                  #####
    ############################################

    time_params = [120.02, 6001]

    full_t = torch.linspace(0, time_params[0], time_params[1], device=device).unsqueeze(-1)

    ###############################################
    ##### Data parameters for the file reader #####
    ###############################################

    data_case = 0
    data_noise = '0.030'

    n_points = 60
    jump = 50

    ##################################
    ##### Time and Space indices  ####
    ##################################

    samples_index = torch.linspace(0, jump*(n_points-1), n_points).int()    
    data_t = torch.linspace(0, ((n_points-1)*jump)*0.02, n_points).unsqueeze(-1).to(device)

    ################################
    ##### Load data from files  ####
    ################################

    
    path = f'./data_ODE/noisy_data/noise_{data_noise}/'

    data_train, data_full = load_ODE_data(path, data_case, samples_index, device)
    data_train['t'] = data_t

    if debug_printable:
        print('Data Train: \n')
        print(data_train)
        print('\n')

        print('Data Full: \n')
        print(data_train)
        print('\n\n')

    ################################
    ##### Model initialization  ####
    ################################

    physics_time_params = [120.02, 6001]

    physics_training_domain = torch.linspace(0, physics_time_params[0], physics_time_params[1], device=device).unsqueeze(-1)
    # print(physics_training_domain)

    loss_weights = {
        'physics': 1,
        'data': 0.01,
        'params': 1,
        'init': 1
    }

    model = pinn(data_train, physics_training_domain, nn_arch, loss_weights, device)

    if debug_printable:
        test_show_model_states(model)

    if debug_mode:
        test_forward(model, printable=debug_printable)

    train_params = {
        'pretrain_epochs': 10000,
        'train_epochs': 10000,
        'pretrain_lr': 1e-3,
        'train_lr': 1e-2,
        'patience': 1000,
        'target_loss': 1e-5,
        'batch_size': 6001
    }

    ################################
    #####    Model training     ####
    ################################

    if debug_mode:
        test_data_loss(model, printable=debug_printable)
        test_physics_loss(model, printable=debug_printable)
        test_parameter_range_regularization(model, printable=debug_printable)
        test_initial_condition_loss(model, printable=debug_printable)
    
    if not disable_train:
        start_time = time.time()
        model.train(train_params, printable=debug_train_printable)
        end_time = time.time()
        train_time = end_time - start_time

    dir_path = f'results/case_{data_case}/noise_{data_noise}/{nn_arch["layers"]}_{nn_arch["neurons"]}_{jump}_{n_points}/'
    file = 'model.pth'

    model_path = save_model(model, dir_path, file)

    if debug_mode:
        test_load_model(model, model_path)
        
    ##################################
    #####    Model evaluation     ####
    ##################################

    evaluator = ev_ODE(full_t, data_full, data_train, device)

    metrics, sol = evaluator.evaluate(model, ploting=True, save_path=dir_path)
    
    
    evaluator.save_metrics(model, metrics, save_path=dir_path)
    #plot_training_losses(losses, save_path=dir_path)
    
    
    

