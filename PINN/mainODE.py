from PINN_ODE import PINN_ODE as pinn
from Evaluator import Evaluator as ev
from debug_functions import *

from io_util import load_ODE_data, save_model, load_model, print_metrics
from plots import plot, plot2D

import time 
import torch 
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#############################
##### Debugging options ##### 
#############################

debug_mode = True
debug_printable = True
debug_plot = False
disable_train = False

if __name__ == "__main__":  
    ########################################
    ##### PINN Architecture parameters ##### 
    ########################################

    nn_arch = {
        'neurons': 100,
        'layers': 5    }
    

    ############################################
    ##### Time domain params               #####
    #####                                  #####
    ##### time_params = [ <limit_time>,    #####
    #####                 <full_samples> ] #####
    #####                                  #####
    ############################################

    time_params = [120, 6001]

    full_t = torch.linspace(0, time_params[0], time_params[1], device=device)

    ###############################################
    ##### Data parameters for the file reader #####
    ###############################################

    data_case = 0
    data_noise = 0.003

    n_points = 20
    jump = 100

    ##################################
    ##### Time and Space indices  ####
    ##################################

    samples_index = torch.linspace(0, jump*(n_points-1), n_points).int()    
    data_t = torch.linspace(0, ((n_points-1)*jump)*0.02, n_points).unsqueeze(-1).to(device)

    ################################
    ##### Load data from files  ####
    ################################

    
    path = f'./data_ODE/noisy_data/noise_{data_noise}/'

    data_train = load_ODE_data(path, data_case, samples_index, device)
    data_train['t'] = data_t

    if debug_printable:
        print(data_train)

    ################################
    ##### Model initialization  ####
    ################################

    physics_time_params = [120, 2000]

    physics_training_domain = torch.linspace(0, physics_time_params[0], physics_time_params[1]).unsqueeze(-1).to(device)

    model = pinn(data_train, physics_training_domain, nn_arch, device)

    if debug_printable:
        test_show_model_states(model)

    if debug_mode:
        test_forward(model, printable=debug_printable)

    train_params = {
        'epochs': 10000,
        'init_lr': 1e-2,
        'patience': 2000,
        'save_model_path': ''
    }

    loss_weights = {
        'physics': 10,
        'data': 0.01,
        'params': 1,
        'init': 1
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
        model.train(train_params, loss_weights, printable=debug_printable)
        end_time = time.time()
        train_time = end_time - start_time

    # dir_path = f'results/case_{data_case}/noise_{data_noise}/{nn_arch["layers"]}_{nn_arch["neurons"]}_{jump}_{n_points}/'
    # file = 'model.pth'

    # model_path = save_model(model, dir_path, file)

    # if debug_mode:
    #     test_load_model(model, model_path)
        
    # ##################################
    # #####    Model evaluation     ####
    # ##################################
    

    # data_full = load_data(full_path, device)
    # evaluator = ev(time_params, space_params, data_train, data_full, device)

    # metrics, sol = evaluator.evaluate(model)
    # print_metrics(metrics)
    

