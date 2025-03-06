import numpy as np

def splitting_method(parameters, N_0, T_0, I_0, t, x, dt, dx,):
    N = np.zeros((len(t), len(x)), dtype=np.float64)
    T = np.zeros((len(t), len(x)), dtype=np.float64)
    I = np.zeros((len(t), len(x)), dtype=np.float64)

    N[0] = N_0
    T[0] = T_0
    I[0] = I_0

    #print(N)
    #print(T)
    #print(I)

    temp_N = np.zeros(len(x), dtype=np.float64)
    temp_T = np.zeros(len(x), dtype=np.float64)
    temp_I = np.zeros(len(x), dtype=np.float64)

    temp_N = np.clip(temp_N, 0, None)
    temp_T = np.clip(temp_T, 0, None)
    temp_I = np.clip(temp_I, 0, None)

    b = parameters['b']
    c = parameters['c']
    r = parameters['r']
    s = parameters['s']
    alpha = parameters['alpha']
    rho = parameters['rho']
    d = parameters['d']
    D = parameters['D']

    for i in range(1, len(t)):
        print(t[i])

        ## Compute step 1

        N_half = N[i-1] + dt*(r[0]*N[i-1]*(1.0-b[0]*N[i-1]) - c[0]*T[i-1]*N[i-1]) 
        T_half = T[i-1] + dt*(r[1]*T[i-1]*(1.0-b[1]*T[i-1]) - c[1]*T[i-1]*I[i-1] - c[2]*T[i-1]*N[i-1]) 
        I_half = I[i-1] + dt*(s + (rho*I[i-1]*T[i-1])/(alpha + T[i-1]) - c[3]*T[i-1]*I[i-1] - d[0]*I[i-1]) 
        
        ## Compute step 2
        
        N[i][1:-1] = N_half[1:-1] + (D[0] * dt / (dx**2)) * (N_half[2:] - 2*N_half[1:-1] + N_half[:-2])
        T[i][1:-1] = T_half[1:-1] + (D[1] * dt / (dx**2)) * (T_half[2:] - 2*T_half[1:-1] + T_half[:-2])
        I[i][1:-1] = I_half[1:-1] + (D[2] * dt / (dx**2)) * (I_half[2:] - 2*I_half[1:-1] + I_half[:-2])
        
        ## Spatial Boundaries ##

        N[i][0] = N[i][1]
        T[i][0] = T[i][1]
        I[i][0] = I[i][1]
        
        N[i][-1] = N[i][-2]
        T[i][-1] = T[i][-2]
        I[i][-1] = I[i][-2]
        
        #print(temp_N)
        #print(temp_T)
        #print(temp_I)
        
    return N, T, I



        

        
        