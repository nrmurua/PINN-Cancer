from plots import plot2D
from io_util import load_data
from PINN import PINN1D


def test_init_model(N_init, T_init, I_init, layers, neurons, time_params, space_params):
    try:
        print('Testing model initialization')

        initial_condition = [N_init, T_init, I_init]
        model = PINN1D(initial_condition, layers, neurons, time_params, space_params)

        print('PINN Model initialized without errors')
        print('Ending test_init_model \n \n')
    
    except Exception as e:
        print(f"An error occurred: {e}")


def test_show_model_states(model):
    try:
        print('Showing PINN states for debug')
    
        model.show_model_states()
    
        print('Model state shown correctly')
        print('Ending test_show_model_states \n \n')
    
    except Exception as e:
        print(f'An error occurred: {e}')


def test_with_init_forward(model, printable=False):
    try:
        print('Testing forward function')

        output_test = model.init_forward()
        if printable:
            print(output_test)
        
        print("PINN is working")
        print('Ending test_with_init_forward')

    except Exception as e:
        print(f"An error occurred: {e}")


def test_plot2D(data, x, t):
    try:
        print('Testing plotting in 2D')

        N = data['N']
        T = data['T']
        I = data['I']

        N_plot = N.squeeze(0).cpu().numpy()
        T_plot = T.squeeze(0).cpu().numpy()
        I_plot = I.squeeze(0).cpu().numpy()

        plot2D(N_plot, x, t)
        plot2D(T_plot, x, t)
        plot2D(I_plot, x, t)
        
        print('Plots 2D are being generated correctly')
        print('Ending test_plot2D \n \n ')
    
    except Exception as e:
        print(f"An error occurred: {e}")


def test_load(train_path, full_path, device='cpu', printable=False):
    try:
        print('Testing data load function')

        data_train = load_data(train_path, device)
        data_full = load_data(full_path, device)

        print('Data loaded correctly')

        if printable:
            print('N_train: \n')
            print(data_train['N'])
            print('N_full: \n')
            print(data_full['N'])

            print('T_train: \n')
            print(data_train['T'])
            print('T_full: \n')
            print(data_full['T'])

            print('I_train: \n')
            print(data_train['I'])
            print('I_full: \n')
            print(data_full['I'])

            print('Data printed correctly')

        print('Ending test_load \n \n')
        
    except Exception as e:
        print(f"An error occurred: {e}")


