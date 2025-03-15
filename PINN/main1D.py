from PINN1D import PINN1D as pinn
from Evaluator import Evaluator as ev
from debug_functions import *

from io_util import load_data, save_model, load_model, print_metrics
from plots import plot, plot2D
from data_util import sec_to_matrix, matrix_to_sec

import time 
import torch 
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#############################
##### Debugging options ##### 
#############################

debug_mode = True
debug_printable = False
debug_plot = False
disable_train = True

if __name__ == "__main__":  
    ########################################
    ##### PINN Architecture parameters ##### 
    ########################################

    nn_arch = {
        'neurons': 5,
        'layers': 3
    }
    

    ############################################
    ##### Time and Space domains params    #####
    #####                                  #####
    ##### time_params = [ <limit_time>,    #####
    #####                 <full_samples> ] #####
    #####                                  #####
    ##### space_params = [ <x1>,           #####
    #####                  <x2>,           #####
    #####                  <dx> ]          #####
    #####                                  #####
    ############################################

    time_params = [60, 3001]
    space_params = [-5, 5, 0.01] 

    full_t = torch.linspace(0, time_params[0], time_params[1], device=device)
    x_samples = int((space_params[1] - space_params[0])/space_params[2] + 1)
    full_x = torch.linspace(space_params[0], space_params[1], x_samples, device=device)


    ###############################################
    ##### Data parameters for the file reader #####
    ###############################################

    data_case = 0
    data_noise = 0.003

    n_points = 10
    jump = 25

    ##################################
    ##### Time and Space indices  ####
    ##################################
    
    data_x = torch.linspace(space_params[0], space_params[1], int((space_params[1]-space_params[0])/space_params[2]) + 1).to(device)
    data_t = torch.linspace(0, n_points-2*(jump*0.02), n_points).to(device)

    ################################
    ##### Load data from files  ####
    ################################

    full_path = f'./noisy_data/case_{data_case}/noise_{data_noise}/'
    train_path =  full_path + f'samples/{n_points}_{jump}/'

    if debug_mode:
        test_load(full_path, train_path, device, printable=debug_printable)

    data_train = load_data(train_path, device)
    data_train['x'] = data_x
    data_train['t'] = data_t

    if debug_plot:
        test_plot2D(data_train, data_x, data_t)

    ################################
    ##### Model initialization  ####
    ################################

    physics_time_params = [60, 15]
    physics_space_params = [-5, 5, 1]

    physics_training_domain = {
        't': torch.linspace(0, physics_time_params[0], physics_time_params[1], device=device),
        'x': torch.linspace(physics_space_params[0], physics_space_params[1], int((physics_space_params[1] - physics_space_params[0])/physics_space_params[2] + 1), device=device)
    }

    model = pinn(data_train, physics_training_domain, nn_arch, device)

    if debug_printable:
        test_show_model_states(model)

    if debug_mode:
        test_forward(model, printable=debug_printable)

    train_params = {
        'epochs': 1000,
        'init_lr': 1e-3,
        'patience': 2000,
        'save_model_path': ''
    }

    loss_weights = {
        'physics': 1,
        'data': 0.01,
        'params': 1,
        'init': 0.1
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

    dir_path = f'results/case_{data_case}/noise_{data_noise}/{nn_arch["layers"]}_{nn_arch["neurons"]}_{jump}_{n_points}/'
    file = 'model.pth'

    model_path = save_model(model, dir_path, file)

    if debug_mode:
        test_load_model(model, model_path)
        
    ##################################
    #####    Model evaluation     ####
    ##################################
    

    data_full = load_data(full_path, device)
    evaluator = ev(time_params, space_params, data_train, data_full, device)

    metrics, sol = evaluator.evaluate(model)
    print_metrics(metrics)
    

