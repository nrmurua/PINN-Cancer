from PINN import PINN1D
from debug_functions import test_with_init_forward, test_plot2D, test_load, test_show_model_states
from io_util import load_data
from plots import plot, plot2D

import torch 
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_model(data_init, n_layers, n_neurons, time_params, space_params):
    model = PINN1D(data_init, n_layers, n_neurons, time_params, space_params, device)

    return model    

if __name__ == "__main__":

    #############################
    ##### Debugging options ##### 
    #############################

    debug_mode = True
    printable = True

    ########################################
    ##### PINN Architecture parameters ##### 
    ########################################

    neurons = 5
    layers = 3

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

    data_x = torch.linspace(space_params[0], space_params[1], int((space_params[1]-space_params[0])/space_params[2]) + 1)
    data_t = torch.linspace(0, n_points-2*(jump*0.02), n_points)

    ################################
    ##### Load data from files  ####
    ################################

    full_path = f'./noisy_data/case_{data_case}/noise_{data_noise}/'
    train_path =  full_path + f'samples/{n_points}_{jump}/'

    if debug_mode:
        test_load(full_path, train_path, device, printable)

    data_train = load_data(train_path, device)
    data_full = load_data(full_path, device)

    data_init = {
        'N': data_full['N'][0,:],
        'T': data_full['T'][0,:],
        'I': data_full['I'][0,:],
    }

    print(data_init)

    if debug_mode:
        test_plot2D(data_train, data_x, data_t)

    model = init_model(data_init, layers, neurons, time_params, space_params)

    if debug_mode:
        test_show_model_states(model)
        test_with_init_forward(model, printable=True)
   
