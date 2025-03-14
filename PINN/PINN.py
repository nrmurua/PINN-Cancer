import torch
import torch.nn as nn
import torch.optim as optim
import copy

class PINN1D(nn.Module):
    def __init__(self, data_train, physics_train_domain, nn_arch, time_domain, space_domain, device="cpu"):
        #########################################################################################################################################################
        #                                                                                                                                                       #
        #   Input:                                                                                                                                              #
        #       - initial_condition = [<t0>, <x>]                                                                                                               #
        #       - num_hidden_layers = int                                                                                                                       #
        #       - num_neurons = int                                                                                                                             #
        #       - time_domain = [<full_time_limit (>0)>, <number of samples>]                                                                                                  #
        #       - space_domain = [<x1>, <x2>, <dx>]                                                                                                             #
        #                                                                                                                                                       #
        #   Initialization Tasks:                                                                                                                               #
        #       - Initial setup of PINN architecture.                                                                                                           #
        #       - Define time and space domains.                                                                                                                #
        #       - Random and always positive parameters.                                                                                                        #
        #                                                                                                                                                       #
        #   Class Functions:                                                                                                                                    #
        #       - forward: Pass the input through the model, returning the output.                                                                              #
        #       - physics_Loss: Compute the Physics Loss with automatic differentiation.                                                                        #
        #       - get_eq_params: return a dictionary with all the parameters.                                                                            #
        #       - initial_condition_loss: compares the output of the model on time = 0 with the known noisy initial condition, returning a loss component.      #
        #       - parameter_range_regularization: enforces the restriction of parameters in the range [p0, p1] with p0<p1. Returns an extra loss component.     #
        #                                                                                                                                                       #
        #########################################################################################################################################################

        super(PINN1D, self).__init__()

        # Initialization of training data
        
        self.data_N = data_train['N']
        self.data_T = data_train['T']
        self.data_I = data_train['I']

        self.data_points = torch.cartesian_prod(data_train['t'], data_train['x'])

        # Initialization of the Physics points

        self.physics_train_domain = {
            't': physics_train_domain['t'],
            'x': physics_train_domain['x']
        }
        self.physics_points = torch.cartesian_prod(physics_train_domain['t'], physics_train_domain['x']).requires_grad_(True)

        # Initial setup of the PINN architecture 

        layers = []
        input_size = 2
        output_size = 3

        n_neurons = nn_arch['neurons']
        n_layers = nn_arch['layers']

        layers.append(nn.Linear(input_size, n_neurons))
        layers.append(nn.SiLU())

        for _ in range(n_layers):
            layers.append(nn.Linear(n_neurons, n_neurons))
            layers.append(nn.SiLU())

        layers.append(nn.Linear(n_neurons, output_size))

        self.solution_network = nn.Sequential(*layers).to(device)

        # Device hardware

        self.device = device

        # Initialization of the full time and space domains

        self.t_domain = torch.linspace(0, time_domain[0], time_domain[1], device=device)

        x_samples = int((space_domain[1] - space_domain[0])/space_domain[2] + 1)

        self.x_domain = torch.linspace(space_domain[0], space_domain[1], x_samples, device=device)

        # Range of the parameters for regularization

        self.min_param_value = 0.001
        self.max_param_value = 1.1

        # Initialization of parameters in logarithmic form

        self.eq_params = {
            'Ng_params': nn.Parameter(torch.randn(2, device=self.device) * 0.01),
            'Tg_params': nn.Parameter(torch.randn(2, device=self.device) * 0.01),
            'c1': nn.Parameter(torch.randn(1, device=self.device) * 0.01),
            'c2': nn.Parameter(torch.randn(1, device=self.device) * 0.01),
            'c3': nn.Parameter(torch.randn(1, device=self.device) * 0.01),
            'c4': nn.Parameter(torch.randn(1, device=self.device) * 0.01),
            'd1': nn.Parameter(torch.randn(1, device=self.device) * 0.01),
            's': nn.Parameter(torch.randn(1, device=self.device) * 0.01),
            'rho': nn.Parameter(torch.randn(1, device=self.device) * 0.01),
            'alpha': nn.Parameter(torch.randn(1, device=self.device) * 0.01),
            'Dn': nn.Parameter(torch.randn(1, device=self.device) * 0.01),
            'Dt': nn.Parameter(torch.randn(1, device=self.device) * 0.01),
            'Di': nn.Parameter(torch.randn(1, device=self.device) * 0.01)
        }

    # Properties for the recovery of parameters with the exponential function

    @property
    def Ng_params(self):
        return torch.exp(self.eq_params['Ng_params'])
        
    @property
    def Tg_params(self):
        return torch.exp(self.eq_params['Tg_params'])
                             
    @property
    def c1(self):
        return torch.exp(self.eq_params['c1'])
                             
    @property
    def c2(self):
        return torch.exp(self.eq_params['c2'])
        
    @property
    def c3(self):
        return torch.exp(self.eq_params['c3'])
                             
    @property
    def c4(self):
        return torch.exp(self.eq_params['c4'])                            

    @property
    def d1(self):
        return torch.exp(self.eq_params['d1'])
                             
    @property
    def s(self):
        return torch.exp(self.eq_params['s'])
        
    @property
    def rho(self):
        return torch.exp(self.eq_params['rho'])
        
    @property
    def alpha(self):
        return torch.exp(self.eq_params['alpha'])

    @property
    def Dn(self):
        return torch.exp(self.eq_params['Dn'])
        
    @property
    def Dt(self):
        return torch.exp(self.eq_params['Dt'])
        
    @property
    def Di(self):
        return torch.exp(self.eq_params['Di'])


    # Pass the input through the PINN model

    def forward(self, input_points):
        solution = self.solution_network(input_points)
        
        return solution
       

    def diff(self, U):

        grad_U = torch.autograd.grad(outputs=U, inputs=self.physics_points, grad_outputs=torch.ones_like(U), create_graph=True)[0]

        dU_t = grad_U[:,0]
        dU_x = grad_U[:,1]
        
        del grad_U

        grad2_U = torch.autograd.grad(outputs=dU_x, inputs=self.physics_points, grad_outputs=torch.ones_like(dU_x), create_graph=True)[0]

        dU_xx = grad2_U[:,1]

        del grad2_U
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        return dU_t, dU_xx

    # Compute the Physics_Loss with automatic differentiation

    def physics_loss(self):
        solution = self.forward(self.physics_points)

        N = solution[:,0]
        T = solution[:,1]
        I = solution[:,2]  

        def g(x, params):
            return params[0] * (1 - x/params[1])

        dN_t, dN_xx = self.diff(N)
        f_N = dN_t - (self.Dn * dN_xx + N * g(N, self.Ng_params) - self.c1 * N * T)
        del dN_t, dN_xx

        dT_t, dT_xx = self.diff(T)
        f_T = dT_t - (self.Dt * dT_xx + T * g(T, self.Tg_params) - self.c2 * T * N - self.c3 * T * I)
        del dT_t, dT_xx

        dI_t, dI_xx = self.diff(N)
        f_I = dI_t - (self.Di * dI_xx + self.s + (self.rho * I * T / (self.alpha + T)) - self.c4 * I * T - self.d1 * I)
        del dI_t, dI_xx

        return torch.mean(f_N**2 + f_T**2 + f_I**2)


    def data_loss(self):
        predicted = self.forward(self.data_points)
        
        N_loss = (predicted[:,0] - self.data_N.reshape(-1))**2
        T_loss = (predicted[:,1] - self.data_T.reshape(-1))**2
        I_loss = (predicted[:,2] - self.data_I.reshape(-1))**2

        return torch.mean(N_loss + T_loss + I_loss)


    # Return the parameters for the equations 

    def get_eq_params(self):
        return {
            'Ng_params':self.Ng_params,
            'Tg_params':self.Tg_params,
            'c1':self.c1,
            'c2':self.c2,
            'c3':self.c3,
            'c4':self.c4,
            'd1':self.d1,
            's':self.s,
            'rho':self.rho,
            'alpha':self.alpha,
            'Dn':self.Dn,
            'Dt':self.Dt,
            'Di':self.Di,
        }


    # Computes the initial condition loss

    def initial_condition_loss(self):
        initial_pred = self.forward(self.data_points[:1001])

        N_loss = initial_pred[:,0] - self.data_N[0,:]
        T_loss = initial_pred[:,1] - self.data_T[0,:]
        I_loss = initial_pred[:,2] - self.data_I[0,:]

        return torch.mean((N_loss**2 + T_loss**2 + I_loss**2))


    # Computes the parameter regularization loss

    def parameter_range_regularization(self):
        params = self.get_eq_params()

        total_penalty = 0.0

        for name, param in params.items():
            below_min_penalty = torch.sum(torch.relu(self.min_param_value - param)**2)
            above_max_penalty = torch.sum(torch.relu(param - self.max_param_value)**2)
            total_penalty += below_min_penalty + above_max_penalty

        return total_penalty        
    
    # Shows te state of the PINN parameters

    def show_model_states(self):
        print('Time Domain: \n')
        print(self.t_domain)
        print()

        print('Space Domain: \n')
        print(self.x_domain)
        print()

        print('Data_train: \n')
        print('N_train: ')
        print(self.data_N)

        print('T_train: ')
        print(self.data_T) 
        
        print('I_train: ')
        print(self.data_I)

        print('data_points: ')
        print(self.data_points) 
        print('\n')

        print('Equation parameters: \n')
        print(self.get_eq_params())
        print()
        
        print('Parameter range: \n')
        print('Min: ')
        print(self.min_param_value)
        print('Max: ')
        print(self.max_param_value)
        print()

        print('Device: \n')
        print(self.device)
        print()

        print('Printing model.parameters(): \n\n')

        for name, param in self.named_parameters():
            print(f"Nombre: {name}")
            print(f"Forma: {param.shape}")
            print(f"Valores: {param.data}\n")




    # Training loop for the PINN model

    def train(self, training_params, loss_weights, printable):
        print('Starting model Training')

        optimizer = optim.Adam(self.parameters(), lr=training_params['init_lr'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode = 'min',
            factor = 0.5,
            patience = 500,
            min_lr = 1e-5
        )

        best_loss = float('inf')
        losses = []
        count = 0

        best_state = {
            'eq': copy.deepcopy(self.eq_params),
            'sd': copy.deepcopy(self.state_dict())
        }

        for epoch in range(training_params['epochs']):
            optimizer.zero_grad()

            L_data = self.data_loss() * loss_weights['data']
            L_physics = self.physics_loss() * loss_weights['physics']
            L_param = self.parameter_range_regularization() * loss_weights['params']
            L_init = self.initial_condition_loss() * loss_weights['init']

            total_loss = L_physics + L_data + L_param + L_init

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step(total_loss)

            losses.append(total_loss.item())

            if total_loss < best_loss:
                best_loss = total_loss.item()
                count = 0
                best_state['eq'] = copy.deepcopy(self.eq_params)
                best_state['sd'] = copy.deepcopy(self.state_dict())
                last_saved_epoch = epoch
                last_saved_loss = best_loss
            else:
                count += 1

            if printable:
                if epoch % 100 == 0:
                    print(f'Epoch {epoch}: '
                        f'Total Loss = {total_loss.item():.9f}, '
                        f'Physics Loss = {L_physics.item():.9f}, '
                        f'Data Loss = {L_data.item():.9f}, '
                        f'Initial Condition Loss = {L_init.item():.9f}, '
                        f'Parameter Reg Loss = {L_param.item():.9f}, '
                        f'LR = {scheduler.get_last_lr()[0]:.6f}',
                    )
                    print(f'Last saved loss: {last_saved_loss:.9f}, '
                          f'Last saved epoch: {last_saved_epoch}\n')

            if count >= training_params['patience']:
                print(f'Early stopping at epoch {epoch}. Out of patience')
                print('Returning model state to last saved: \n')
                print(f'loss: {last_saved_loss:.9f}, '
                      f'epoch: {last_saved_epoch:.9f} \n')
                
                self.load_state_dict(best_state['sd'])
                self.eq_params = best_state['eq']

                break

            if best_loss < 1e-5:
                print(f'Early stopping at epoch {epoch}. Good enough')
                print(f'Reached Loss: {last_saved_loss:.9f}')
                break
        
        print('Model has been trained without errors :)')

    def evaluate_model(self, dataset='Training'):
        with torch.no_grad():
            if dataset == 'Training':
                pred = self(self.data_points)
            


        ### Complete the evaluation of the model in the 

