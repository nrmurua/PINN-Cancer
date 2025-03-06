from PINN import PINN1D
from debug_functions import test_with_init_forward, test_plot2D
from io_util import load_data
from plots import plot, plot2D

import torch 
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_model(N_init, T_init, I_init, n_layers, n_neurons, time_params, space_params):
    initial_condition = [N_init, T_init, I_init]
    model = PINN1D(initial_condition, n_layers, n_neurons, time_params, space_params)

    return model    

if __name__ == "__main__":
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
    train_path =  f'./noisy_data/case_{data_case}/noise_{data_noise}/samples/{n_points}_{jump}/'

    N_train, N_full = load_data(train_path + 'N.npy', full_path + 'N.npy', device)
    T_train, T_full = load_data(train_path + 'T.npy', full_path + 'T.npy', device)
    I_train, I_full = load_data(train_path + 'I.npy', full_path + 'I.npy', device)

    test_plot2D(N_train, T_train, I_train, data_x, data_t)

    N_init = N_train[0,0,:]
    T_init = T_train[0,0,:]
    I_init = I_train[0,0,:]

    model = init_model(N_init, T_init, I_init, layers, neurons, time_params, space_params)

    test_with_init_forward(model, printable=False)
   
