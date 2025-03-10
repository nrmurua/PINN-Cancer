import torch
import torch.nn as nn
import torch.optim as optim


class PINN1D(nn.Module):
    def __init__(self, data_train, nn_arch, time_domain, space_domain, device="cpu"):
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

        # Initial setup of the PINN architecture 
        self.data_N = data_train['N'].reshape(-1)
        self.data_T = data_train['T'].reshape(-1)
        self.data_I = data_train['I'].reshape(-1)

        self.input_points = torch.cartesian_prod(data_train['t'], data_train['x'])

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

        self.log_Ng_params = nn.Parameter(torch.randn(2, device=self.device) * 0.01)
        self.log_Tg_params = nn.Parameter(torch.randn(2, device=self.device) * 0.01)
        
        self.log_c1 = nn.Parameter(torch.randn(1, device=self.device) * 0.01)
        self.log_c2 = nn.Parameter(torch.randn(1, device=self.device) * 0.01)
        self.log_c3 = nn.Parameter(torch.randn(1, device=self.device) * 0.01)
        self.log_c4 = nn.Parameter(torch.randn(1, device=self.device) * 0.01)

        self.log_d1 = nn.Parameter(torch.randn(1, device=self.device) * 0.01)
        self.log_s = nn.Parameter(torch.randn(1, device=self.device) * 0.01)
        self.log_rho = nn.Parameter(torch.randn(1, device=self.device) * 0.01)
        self.log_alpha = nn.Parameter(torch.randn(1, device=self.device) * 0.01)

        self.log_Dn = nn.Parameter(torch.randn(1, device=self.device) * 0.01)
        self.log_Dt = nn.Parameter(torch.randn(1, device=self.device) * 0.01)
        self.log_Di = nn.Parameter(torch.randn(1, device=self.device) * 0.01)

    # Properties for the recovery of parameters with the exponential function

    @property
    def Ng_params(self):
        return torch.exp(self.log_Ng_params)
        
    @property
    def Tg_params(self):
        return torch.exp(self.log_Tg_params)
                             
    @property
    def c1(self):
        return torch.exp(self.log_c1)
                             
    @property
    def c2(self):
        return torch.exp(self.log_c2)
        
    @property
    def c3(self):
        return torch.exp(self.log_c3)
                             
    @property
    def c4(self):
        return torch.exp(self.log_c4)                            

    @property
    def d1(self):
        return torch.exp(self.log_d1)
                             
    @property
    def s(self):
        return torch.exp(self.log_s)
        
    @property
    def rho(self):
        return torch.exp(self.log_rho)
        
    @property
    def alpha(self):
        return torch.exp(self.log_alpha)

    @property
    def Dn(self):
        return torch.exp(self.log_Dn)
        
    @property
    def Dt(self):
        return torch.exp(self.log_Dt)
        
    @property
    def Di(self):
        return torch.exp(self.log_Di)


    # Pass the input through the PINN model

    def forward(self, input_points):
        solution = self.solution_network(input_points)
        
        return solution
        

    # Test initial condition Forward

    def init_forward(self):
        print(self.input_points[:1001])
        sol = self.solution_network(self.input_points[:1001])

        return sol


    # Compute the Physics_Loss with automatic differentiation

    def physics_loss(self, t, x):
        t_grad = t.clone().requires_grad_(True)
        x_grad = x.clone().requires_grad_(True)
        solution = self.forward(t_grad)
        N, T, I = solution[:, :, 0], solution[:, :, 1], solution[:, :, 2]

        # Autoderivacion

        # N temporal

        dN_dt = torch.autograd.grad(N, t_grad, grad_outputs=torch.ones_like(N), create_graph=True)[0]
            
        # N Espacial

        dN_dx = torch.autograd.grad(N, x_grad, grad_outputs=torch.ones_like(N), create_graph=True)[0]
        dN_dx2 = torch.autograd.grad(dN_dx, x_grad, grad_outputs=torch.ones_like(N), create_graph=True)[0]


        # T temporal

        dT_dt = torch.autograd.grad(T, t_grad, grad_outputs=torch.ones_like(T), create_graph=True)[0]
            
        # T Espacial

        dT_dx = torch.autograd.grad(T, x_grad, grad_outputs=torch.ones_like(T), create_graph=True)[0]
        dT_dx2 = torch.autograd.grad(dT_dx, x_grad, grad_outputs=torch.ones_like(T), create_graph=True)[0]


        # I temporal

        dI_dt = torch.autograd.grad(I, t_grad, grad_outputs=torch.ones_like(I), create_graph=True)[0]

        # I Espacial

        dI_dx = torch.autograd.grad(I, x_grad, grad_outputs=torch.ones_like(I), create_graph=True)[0]
        dI_dx2 = torch.autograd.grad(dI_dx, x_grad, grad_outputs=torch.ones_like(I), create_graph=True)[0]

        def g(x, params):
            return params[0] * (1 - x/params[1])
            
            
        f_N = dN_dt - (self.Dn*dN_dx2 + N*g(N, self.Ng_params) - self.c1*N*T)
        f_T = dT_dt - (self.Dt*dT_dx2 + T*g(T, self.Tg_params) - self.c2*N*T - self.c3*T*I)
        f_I = dI_dt - (self.Di*dI_dx2 + self.s + (self.rho*I*T/(self.alpha + T)) - self.c4*I*T - self.d1*I)

        return torch.mean(f_N**2 + f_T**2 + f_I**2)

    def data_loss(self):
        predicted = self.forward(self.input_points)
        
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

    def initial_condition_loss(self, x):
        initial_cond = torch.stack(torch.zeros(1, device=self.device), x)
        initial_pred = self.forward(initial_cond)

        return torch.mean((initial_pred - self.initial_condition)**2)


    # Computes the parameter regularization loss

    def parameter_range_regularization(self):
        params = self.get_positive_params

        total_penalty = 0.0

        for name, param in params.items():
            below_min_penalty = torch.sum(torch.relu(self.min_param_value - param)**2)
            above_max_penalty = torch.sum(torch.relu(params - self.max_param_value)**2)
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

        print('input_points: ')
        print(self.input_points) 
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

    def train(self, data_train, training_params, loss_weights, device):
        optimizer = optim.Adam(self.parameters(), lr=training_params['init_lr'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode = 'min',
            factor = 0.5,
            patience = 500,
            min_lr = 1e-5
        )

        for epoch in range(training_params['epochs']):
            optimizer.zero_grad()

            L_data = self.data_loss(data_train) * loss_weights['data']

            #### implementar las funciones de perdida