import numpy as np
from splitting_method import splitting_method
from plots import plot, plot2D

def initial_conditions0(x, x0, sigma):
    N_0 = 1 - 0.2 * np.exp(-(x - x0)**2 / sigma**2)  
    T_0 = 0.5 * np.exp(-(x - x0)**2 / sigma**2)      
    I_0 = 0.1 * np.exp(-(x - x0)**2 / sigma**2)

    parameters = {
        'b': [1.0, 0.81],       # Carrying Capacity
        'c': [1, 0.4, 0.5, 1],  # Competition Term
        'd': [0.2],             # Death Rate
        'r': [1.1, 1],          # Growth Rate
        's': 0.33,              # Immune Source Rate
        'alpha': 0.3,           # Immune Threshold Rate
        'rho': 0.2,             # Immune Response Rate
        'D': [0, 0.001, 0.001]  # Diffusion Coefficients
    }

    return N_0, T_0, I_0, parameters

def initial_conditions1(x, x0, sigma):
    # Gradual initial conditions to avoid instability
    N_0 = 1.1 - 0.4 * np.exp(-(x - x0)**2 / sigma**2)
    T_0 = 0.89 * np.exp(-(x - x0)**2 / sigma**2)      
    I_0 = 0.1 * np.exp(-(x - x0)**2 / sigma**2)

    parameters = {
        'b': [0.87, 1.14],                            # Carrying Capacity (lower to prevent overgrowth)
        'c': [0.62, 0.43, 0.23, 0.57],                  # Competition Term (lower values for more stable interaction)
        'd': [0.57],                                 # Death Rate (reasonable for cell survival)
        'r': [0.44, 0.42],                            # Growth Rate (reduced for stability)
        's': 0.09,                                   # Immune Source Rate (lower for controlled immune presence)
        'alpha': 0.15,                              # Immune Threshold Rate (lower to reduce threshold behavior)
        'rho': 0.13,                                 # Immune Response Rate (controlled immune response)
        'D': [0, 0.00003, 0.000001]                        # Diffusion Coefficients (moderate diffusion)
    }

    return N_0, T_0, I_0, parameters

def initial_conditions2(x, x0_1, x0_2, sigma):
    N_0 = 1 - 0.3 * (np.exp(-(x - x0_1)**2 / sigma**2) + np.exp(-(x - x0_2)**2 / sigma**2))  
    T_0 = 0.5 * (np.exp(-(x - x0_1)**2 / sigma**2) + np.exp(-(x - x0_2)**2 / sigma**2))     
    I_0 = 0.3 * (np.exp(-(x - x0_1)**2 / sigma**2) + np.exp(-(x - x0_2)**2 / sigma**2))

    parameters = {
        'b': [1.37, 1.08],                            # Carrying Capacity (lower to prevent overgrowth)
        'c': [0.31, 0.13, 0.17, 0.29],                  # Competition Term (lower values for more stable interaction)
        'd': [0.26],                                 # Death Rate (reasonable for cell survival)
        'r': [0.21, 0.25],                            # Growth Rate (reduced for stability)
        's': 0.07,                                   # Immune Source Rate (lower for controlled immune presence)
        'alpha': 0.07,                              # Immune Threshold Rate (lower to reduce threshold behavior)
        'rho': 0.08,                                 # Immune Response Rate (controlled immune response)
        'D': [0, 0.0007, 0.0005]                        # Diffusion Coefficients (moderate diffusion)
    }

    return N_0, T_0, I_0, parameters

def sim1d(cond=0):
    
    #############################################
    ###                                       ###
    ###             Define Domain             ###
    ###                                       ###
    #############################################

    dx = 0.01
    x = np.array([-5 + i * dx for i in range(int((10) / dx) + 1)])

    dt = 0.02
    t = np.array([0 + i * dt for i in range(int((60) / dt) + 1)])

    
    if cond == 0:
        # Set up initial conditions and time span
        x0 = 0
        sigma = 2.5
        N_0, T_0, I_0, parameters = initial_conditions0(x, x0, sigma)
    elif cond == 1:
        x0 = 0
        sigma = 1.5
        N_0, T_0, I_0, parameters = initial_conditions1(x,x0, sigma)
    elif cond == 2:
        x0 = 2
        sigma = 2
        N_0, T_0, I_0, parameters = initial_conditions2(x, -x0, x0, sigma)


    plot(N_0, T_0, I_0, x)

    # Solve the system using the splitting method
    N, T, I = splitting_method(parameters, N_0, T_0, I_0, t, x, dt, dx)

    np.save(f'./data/1D/case_{cond}/N.npy', N)
    np.save(f'./data/1D/case_{cond}/T.npy', T)
    np.save(f'./data/1D/case_{cond}/I.npy', I)

    plot(N[-1], T[-1], I[-1], x)

    plot2D(N,x,t,'Normal')
    plot2D(T,x,t,'Tumoral')
    plot2D(I,x,t,'Inmune')
    

