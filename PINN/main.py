from PINN import PINN1D as pinn
from debug_functions import test_with_init_forward, test_plot2D, test_load, test_show_model_states
from debug_functions import test_data_loss
from io_util import load_data
from plots import plot, plot2D

import torch 
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    #############################
    ##### Debugging options ##### 
    #############################

    debug_mode = True
    printable = True

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

    ###########################
    ##### Data parameters #####
    ###########################

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
        test_load(full_path, train_path, device, printable)

    data_train = load_data(train_path, device)
    data_train['x'] = data_x
    data_train['t'] = data_t

    data_full = load_data(full_path, device)

    data_init = {
        'N': data_full['N'][0,:],
        'T': data_full['T'][0,:],
        'I': data_full['I'][0,:],
    }

    if debug_mode:
        test_plot2D(data_train, data_x, data_t)

    ################################
    ##### Model initialization  ####
    ################################

    model = pinn(data_train, nn_arch, time_params, space_params, device)

    if printable:
        test_show_model_states(model)

    if debug_mode:
        test_with_init_forward(model, printable)

    train_params = {
        'epochs': 100000,
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
        test_data_loss(model, printable)

    #model.train(data_train, train_params, loss_weights, device)

    # Evaluar formato del output del modelo para el calculo de las perdidas 