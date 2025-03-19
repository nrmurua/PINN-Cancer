import torch
import torch.nn as nn
import torch.optim as optim
import copy

class PINN_ODE(nn.Module):
    def __init__(self, data_train, physics_train_domain, nn_arch, device="cpu"):
        super(PINN_ODE, self).__init__()

        # Initialization of training data
        
        self.data_train = torch.column_stack([data_train['N'], data_train['T'], data_train['I']]).to(device)
        #print(data_train)
        self.data_points = data_train['t']

        # Initialization of the Physics points

        self.physics_train_domain = physics_train_domain

        # Initial setup of the PINN architecture 

        layers = []
        input_size = 1
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

        # Range of the parameters for regularization

        self.min_param_value = 0.001
        self.max_param_value = 1.1

        # Initialization of parameters in logarithmic form

        self.log_Ng_params = nn.Parameter(torch.randn(2, device=device) * 0.01)
        self.log_Tg_params = nn.Parameter(torch.randn(2, device=device) * 0.01)
        self.log_c1 = nn.Parameter(torch.randn(1, device=device) * 0.01)
        self.log_c2 = nn.Parameter(torch.randn(1, device=device) * 0.01)
        self.log_c3 = nn.Parameter(torch.randn(1, device=device) * 0.01)
        self.log_c4 = nn.Parameter(torch.randn(1, device=device) * 0.01)
        self.log_d1 = nn.Parameter(torch.randn(1, device=device) * 0.01)
        self.log_s = nn.Parameter(torch.randn(1, device=device) * 0.01)
        self.log_rho = nn.Parameter(torch.randn(1, device=device) * 0.01)
        self.log_alpha = nn.Parameter(torch.randn(1, device=device) * 0.01)
    
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


    def forward(self, input_points):
        solution = self.solution_network(input_points)

        return solution
    

    def data_loss(self):
        t = self.data_points
        pred = self.forward(t)

        return torch.mean((pred - self.data_train)**2)


    def physics_loss(self):
        t = self.physics_train_domain.requires_grad_(True)
        sol = self.forward(t)

        N = sol[:,0]    
        T = sol[:,1]
        I = sol[:,2]  

        def g(x, params):
            return params[0] * (1 - x/params[1])

        dN_t = torch.autograd.grad(outputs=N, inputs=t, grad_outputs=torch.ones_like(N), create_graph=True)[0]
        dT_t = torch.autograd.grad(outputs=T, inputs=t, grad_outputs=torch.ones_like(T), create_graph=True)[0]
        dI_t = torch.autograd.grad(outputs=I, inputs=t, grad_outputs=torch.ones_like(I), create_graph=True)[0]

        f_N = dN_t - (N * g(N, self.Ng_params) - self.c1 * N * T)
        f_T = dT_t - (T * g(T, self.Tg_params) - self.c2 * T * N - self.c3 * T * I)
        f_I = dI_t - (self.s + ((self.rho * I * T) / (self.alpha + T)) - self.c4 * I * T - self.d1 * I)

        return torch.mean(f_N**2 + f_T**2 + f_I**2)


    def initial_condition_loss(self):
        initial_pred = self.forward(self.data_points[0])
        return torch.mean((initial_pred - self.data_train[0,:])**2)

    def parameter_range_regularization(self):
        params = self.get_eq_params()

        total_penalty = 0.0

        for name, param in params.items():
            below_min_penalty = torch.sum(torch.relu(self.min_param_value - param)**2)
            above_max_penalty = torch.sum(torch.relu(param - self.max_param_value)**2)
            total_penalty += below_min_penalty + above_max_penalty

        return total_penalty

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
            'alpha':self.alpha
        }
    
    def set_eq_params(self, params):
        self.log_Ng_params = torch.log(params['Ng_params'])
        self.log_Tg_params = torch.log(params['Tg_params'])
        self.log_c1 = torch.log(params['c1'])
        self.log_c2 = torch.log(params['c2'])
        self.log_c3 = torch.log(params['c3'])
        self.log_c4 = torch.log(params['c4'])
        self.log_d1 = torch.log(params['d1'])
        self.log_s = torch.log(params['s'])
        self.log_rho = torch.log(params['rho'])
        self.log_alpha = torch.log(params['alpha'])


    def train(self, train_params, loss_weights, printable='False'):
        
        print('Starting model Training')

        optimizer = optim.Adam(self.parameters(), lr=train_params['init_lr'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode = 'min',
            factor = 0.5,
            patience = 250,
            min_lr = 1e-5
        )

        print(self.physics_train_domain)

        best_loss = float('inf')
        losses = []
        count = 0

        best_state = {
            'eq': self.get_eq_params(),
            'sd': self.state_dict()
        }

        for epoch in range(train_params['epochs']):
            optimizer.zero_grad()
            
            L_data = self.data_loss() * loss_weights['data']
            L_physics = self.physics_loss() * loss_weights['physics']
            L_param = self.parameter_range_regularization() * loss_weights['params']
            L_init = self.initial_condition_loss() * loss_weights['init']

            total_loss = L_physics + L_data 

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step(total_loss)

            losses.append(total_loss.item())

            if total_loss < best_loss:
                best_loss = total_loss.item()
                count = 0
                best_state['eq'] = self.get_eq_params()
                best_state['sd'] = self.state_dict()
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

            if count >= train_params['patience']:
                print(f'Early stopping at epoch {epoch}. Out of patience')
                break

            if best_loss < 1e-5:
                print(f'Early stopping at epoch {epoch}. Good enough')
                print(f'Reached Loss: {last_saved_loss:.9f}')
                break
        
        print('Model has been trained without errors :)')

        print('Setting the model state to the best found config: \n')
        
        print(f'loss: {last_saved_loss:.9f}, '
              f'epoch: {last_saved_epoch:.9f} \n')
        
        self.load_state_dict(best_state['sd'])
        self.set_eq_params(best_state['eq']) 


    def show_model_states(self):
        print('Data_train: \n')
        print('N_train: ')
        print(self.data_train)

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