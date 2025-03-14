from plots import plot2D
from io_util import load_data, save_model, load_model
from PINN1D import PINN1D


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


def test_forward(model, printable=False):
    try:
        print('Testing forward function')

        output_test = model.forward(model.data_points)
        if printable:
            print(output_test)
        
        print("PINN is working")
        print('Ending test_forward')

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

            print('Data printed correctly \n')

        print('Ending test_load \n \n')
        
    except Exception as e:
        print(f"An error occurred: {e} \n \n \n")


def test_data_loss(model, printable):
    try:
        print('Testing the data loss function from the PINN model \n')

        data_loss = model.data_loss()

        print('Data loss computed correctly \n')

        if printable:
            print(data_loss)

            print('Data loss printed correctly \n')

        print('Ending test_data_loss \n \n')

    except Exception as e:
        print(f'An error has occurred: {e} \n \n \n')


def test_physics_loss(model, printable):
    try:
        print('Testing the physics loss function from the PINN model \n')

        physics_loss = model.physics_loss()

        print('Physics loss computed correctly \n')

        if printable:
            print(physics_loss)

            print('Physics loss printed correctly \n')

        print('Ending test_physics_loss \n \n')

    except Exception as e:
        print(f'An error has occurred: {e} \n \n \n')

def test_parameter_range_regularization(model, printable):
    try:
        print('Testing the Parameter Range loss function from the PINN model \n')

        param_loss = model.parameter_range_regularization()

        print('Parameter Range loss computed correctly \n')

        if printable:
            print(param_loss)

            print('Parameter Range loss printed correctly \n')

        print('Ending test_parameter_range_regularization \n \n')

    except Exception as e:
        print(f'An error has occurred: {e} \n \n \n')

def test_initial_condition_loss(model, printable):
    try:
        print('Testing the Initial Condition loss function from the PINN model \n')

        init_loss = model.initial_condition_loss()

        print('Initial Condition loss computed correctly \n')

        if printable:
            print(init_loss)

            print('Initial Condition loss printed correctly \n')

        print('Ending test_initial_condition_loss \n \n')

    except Exception as e:
        print(f'An error has occurred: {e} \n \n \n')

def test_model_training(model, train_params, loss_weights, printable):
    try:
        print('Starting model training \n')

        model.train(train_params, loss_weights, printable)

        print('Execution of training loops without errors')
        print('Ending test_model_training \n \n')

    except Exception as e:
        print(f'An error has occurres: {e} \n \n \n')

def test_save_model(model):
    try:
        print('Starting model training \n')

        model.evaluate_model()

        print('Execution of training loops without errors')
        print('Ending test_model_training \n \n')

    except Exception as e:
        print(f'An error has occurres: {e} \n \n \n')

def test_model_evaluation(model):
    try:
        print('Starting model training \n')

        model.evaluate_model()

        print('Execution of training loops without errors')
        print('Ending test_model_training \n \n')

    except Exception as e:
        print(f'An error has occurres: {e} \n \n \n')


def test_load_model(model, model_path):
    try:
        print('Testing model loading \n')

        load_model(model, model_path)

        print('Model loaded without errors')
        print('Ending test_load_model \n \n')

    except Exception as e:
        print(f'An error has occurres: {e} \n \n \n')