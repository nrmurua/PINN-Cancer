from PINN import PINN1D
from debug_functions import test_with_init_forward
from io_util import load_data
import torch 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_model(N_init, T_init, I_init, n_layers, n_neurons, time_params, space_params):
    initial_condition = [N_init, T_init, I_init]
    model = PINN1D(initial_condition, n_layers, n_neurons, time_params, space_params)

    return model    

if __name__ == "__main__":
    neurons = 5
    layers = 3

    time_params = [60, 3001]
    space_params = [-2, 2, 0.01] 

    ##### This need to be changed to the real loading functions ####

    N_train, N_full = load_data('./data/noisy_data/noise_0.030/N_0.npy', train_size=data_time, subsample_rate=sample_jump)
    T_train, T_full = load_data('./data/noisy_data/noise_0.030/T_0.npy', train_size=data_time, subsample_rate=sample_jump)
    I_train, I_full = load_data('./data/noisy_data/noise_0.030/I_0.npy', train_size=data_time, subsample_rate=sample_jump)


    # space_size = int((space_params[1] - space_params[0])/space_params[2])

    # N_init = N_train[]
    # T_init = 
    # I_init = 


    ##### This need to be changed to the real loading functions ####

    # model = init_model(N_init, T_init, I_init, layers, neurons, time_params, space_params)

    # test_with_init_forward(model, printable=True)
   
