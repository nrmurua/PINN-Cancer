import torch
import torch.nn as nn
import torch.optim as optim
import copy
from AdaptiveWeights import AdaptiveLossWeights as alw

class PINN_ODE(nn.Module):
    def __init__(self, data_train, physics_train_domain, nn_arch, loss_weights, device="cpu"):
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
        layers.append(nn.Sigmoid())

        for _ in range(n_layers):
            layers.append(nn.Linear(n_neurons, n_neurons))
            layers.append(nn.Sigmoid())

        layers.append(nn.Linear(n_neurons, output_size))

        self.solution_network = nn.Sequential(*layers).to(device)

        # Device hardware

        self.device = device

        # Relevant metrics

        self.best_loss = float('inf')
        self.best_epoch = 0

        # Adaptive Loss Weights

        self.loss_weights = loss_weights
        self.adaptive_losses = alw(loss_weights, device=device)

        # Range of the parameters for regularization

        self.min_param_value = 0.001
        self.max_param_value = 1.1

        # Initialization of equations parameters in logarithmic form

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
    
    # Recover loss parameters

    @property
    def physics_loss_weight(self):
        return torch.exp(self.log_physics_loss)
    
    @property
    def data_loss_weight(self):
        return torch.exp(self.log_data_loss)
    
    @property
    def params_loss_weight(self):
        return torch.exp(self.log_params_loss)
    
    @property
    def init_loss_weight(self):
        return torch.exp(self.log_init_loss)


    # Recover equation parameters

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


    def physics_loss(self, batch_size=None, delta_t=1e-2):
        if batch_size is not None:
            # Muestreo aleatorio sin reemplazo de puntos físicos
            idx = torch.randperm(self.physics_train_domain.shape[0])[:batch_size]
            t = self.physics_train_domain[idx]
        else:
            # Si no quieres batching, usa todos los puntos
            t = self.physics_train_domain

        # Obtención de la solución para los valores de t
        sol = self.forward(t)

        # Extracción de las variables de la solución
        N = sol[:, 0]
        T = sol[:, 1]
        I = sol[:, 2]

        # Función g para la dinámica
        def g(x, params):
            return params[0] * (1 - x/params[1])

        sol_future = self.forward(t + delta_t)

        # Euler implícito: Derivada de N
        N_future = sol_future[:, 0]  # Solución en el paso futuro
        dN_t = (N_future - N) / delta_t
        f_N = dN_t - (N * g(N, self.Ng_params) - self.c1 * N * T)
        
        # Euler implícito: Derivada de T
        T_future = sol_future[:, 1]  # Solución en el paso futuro
        dT_t = (T_future - T) / delta_t
        f_T = dT_t - (T * g(T, self.Tg_params) - self.c2 * T * N - self.c3 * T * I)

        # Euler implícito: Derivada de I
        I_future = sol_future[:, 2]  # Solución en el paso futuro
        dI_t = (I_future - I) / delta_t
        f_I = dI_t - (self.s + ((self.rho * I * T) / (self.alpha + T)) - self.c4 * I * T - self.d1 * I)

        # Pérdida física: Suma de los términos de la ecuación de dinámica
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
        self.log_Ng_params = nn.Parameter(torch.log(params['Ng_params']))
        self.log_Tg_params = nn.Parameter(torch.log(params['Tg_params']))
        self.log_c1 = nn.Parameter(torch.log(params['c1']))
        self.log_c2 = nn.Parameter(torch.log(params['c2']))
        self.log_c3 = nn.Parameter(torch.log(params['c3']))
        self.log_c4 = nn.Parameter(torch.log(params['c4']))
        self.log_d1 = nn.Parameter(torch.log(params['d1']))
        self.log_s = nn.Parameter(torch.log(params['s']))
        self.log_rho = nn.Parameter(torch.log(params['rho']))
        self.log_alpha = nn.Parameter(torch.log(params['alpha']))


    def train(self, train_params, printable='False'):
        
        print('Starting model Training')

        optimizer_pretrain = optim.Adam(self.parameters(), lr=train_params['pretrain_lr'])
        scheduler_pretrain = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_pretrain,
            mode = 'min',
            factor = 0.9,
            patience = 500,
            min_lr = 1e-5
        )

        best_loss = float('inf')
        losses_pretrain = []
        count = 0

        best_state = {}

        for epoch in range(train_params['pretrain_epochs']):
            optimizer_pretrain.zero_grad()

            losses_dict = {
                'L_physics': self.physics_loss(batch_size=train_params['batch_size']), 
                'L_init': self.initial_condition_loss()
            }    
            
            total_loss = (losses_dict['L_physics'] * self.loss_weights['physics']
                            + losses_dict['L_init'] * self.loss_weights['init'])

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

            optimizer_pretrain.step()

            scheduler_pretrain.step(total_loss)

            losses_pretrain.append(total_loss.item())

            if total_loss < best_loss:
                best_loss = total_loss.item()
                count = 0
                best_state['eq'] = self.get_eq_params()
                best_state['sd'] = self.state_dict()
                best_state['lw'] = self.adaptive_losses.state_dict()
                last_saved_epoch = epoch
                last_saved_loss = best_loss
            else:
                count += 1

            if printable:
                if epoch % 10 == 0:
                    print(f'Epoch {epoch}: '
                        f'Total Loss = {total_loss.item():.9f}, '
                        f'Physics Loss = {losses_dict["L_physics"]:.9f}, '
                        f'Initial Condition Loss = {losses_dict["L_init"]:.9f}, '
                    )
                    print(f'LR = {scheduler_pretrain.get_last_lr()[0]:.6f}')
                    print(f'Physics_loss = {self.loss_weights["physics"]:.6f}, '
                          f'Init_loss = {self.loss_weights["init"]:.6f}'
                    )
                    print(f'Last saved loss: {last_saved_loss:.9f}, '
                          f'Last saved epoch: {last_saved_epoch}\n')

            if count >= train_params['patience']:
                print(f'Early stopping at epoch {epoch}. Out of patience')
                break

            if best_loss < train_params['target_loss']:
                print(f'Early stopping at epoch {epoch}. Good enough')
                print(f'Reached Loss: {last_saved_loss:.9f}')
                break
        
        print('Model has been pre-trained without errors :)')

        print('Setting the model state to the best found config: \n')

        print('Starting Training with data\n\n')


        ## Training the model with data

        optimizer_train = optim.Adam(self.parameters(), lr=train_params['train_lr'])
        scheduler_train = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_train,
            mode = 'min',
            factor = 0.9,
            patience = 500,
            min_lr = 1e-5
        )

        losses_train = []
        count = 0

        for epoch in range(train_params['train_epochs']):
            optimizer_train.zero_grad()

            losses_dict = {
                'L_data': self.data_loss(),
                'L_physics': self.physics_loss(batch_size=train_params['batch_size']), 
                'L_params': self.parameter_range_regularization(),
                'L_init': self.initial_condition_loss()
            }    
            
            total_loss = (losses_dict['L_physics'] * self.loss_weights['physics']
                            + losses_dict['L_data'] * self.loss_weights['data']
                            + losses_dict['L_params'] * self.loss_weights['params']
                            + losses_dict['L_init'] * self.loss_weights['init'])

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

            optimizer_train.step()

            scheduler_train.step(total_loss)

            losses_train.append(total_loss.item())

            if total_loss < self.best_loss:
                self.best_loss = total_loss.item()
                self.best_epoch = epoch

                count = 0
                
                best_state['eq'] = self.get_eq_params()
                best_state['sd'] = self.state_dict()
                best_state['lw'] = self.adaptive_losses.state_dict()
            
            else:
                count += 1

            if printable:
                if epoch % 10 == 0:
                    print(f'Epoch {epoch}: '
                        f'Total Loss = {total_loss.item():.9f}, '
                        f'Physics Loss = {losses_dict["L_physics"]:.9f}, '
                        f'Data Loss = {losses_dict["L_data"]:.9f}, '
                        f'Initial Condition Loss = {losses_dict["L_init"]:.9f}, '
                        f'Parameter Reg Loss = {losses_dict["L_params"]:.9f}, '
                    )
                    print(f'LR = {scheduler_train.get_last_lr()[0]:.6f}')
                    print(f'Physics_loss = {self.loss_weights["physics"]:.6f}, '
                          f'Data_loss = {self.loss_weights["data"]:.6f}, '
                          f'Params_loss = {self.loss_weights["params"]:.6f}, '
                          f'Init_loss = {self.loss_weights["init"]:.6f}'
                    )
                    print(f'Last saved loss: {self.best_loss:.9f}, '
                          f'Last saved epoch: {self.best_epoch}\n')

            if count >= train_params['patience']:
                print(f'Early stopping at epoch {epoch}. Out of patience')
                break

            if self.best_loss < train_params['target_loss']:
                print(f'Early stopping at epoch {epoch}. Good enough')
                print(f'Reached Loss: {self.best_loss:.9f}')
                break

        print('Model has been trained without errors :)')

        print('Setting the model state to the best found config: \n')
        
        print(f'loss: {self.best_loss:.9f}, '
              f'epoch: {self.best_epoch:.9f} \n')
        
        self.set_eq_params(best_state['eq'])
        self.load_state_dict(best_state['sd'])
        self.adaptive_losses.load_state_dict(best_state['lw'])

        return losses_train



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