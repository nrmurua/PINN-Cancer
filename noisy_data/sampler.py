import numpy as np
import os 

def save_sampled_steps(cond, n, m):
    # Cargar los datos originales
    N = np.load(f'./noisy_data/case_{cond}/noise_0.003/N.npy')
    T = np.load(f'./noisy_data/case_{cond}/noise_0.003/T.npy')
    I = np.load(f'./noisy_data/case_{cond}/noise_0.003/I.npy')

    # Generar los índices de los tiempos a extraer
    indices = np.arange(0, n * m, m)
    indices = indices[indices < N.shape[0]]  # Asegurar que no se salga del rango

    print(N.shape)

    # Extraer los pasos de tiempo seleccionados
    N_subset = N[:][indices]
    T_subset = T[:][indices]
    I_subset = I[:][indices]

    print(N_subset.shape)

    # Definir el directorio de salida
    save_dir = f'./noisy_data/case_{cond}/noise_0.003/samples/{n}_{m}/'
    
    # Crear la carpeta si no existe
    os.makedirs(save_dir, exist_ok=True)

    # Guardar los nuevos archivos
    np.save(os.path.join(save_dir, 'N.npy'), N_subset)
    np.save(os.path.join(save_dir, 'T.npy'), T_subset)
    np.save(os.path.join(save_dir, 'I.npy'), I_subset)

    print(f'Se han guardado {len(indices)} muestras espaciadas cada {m} tiempos para la condición {cond}.')

# Definir la condición, el número de muestras y el intervalo
cond = 2  # Ajusta esto según la condición
n = 10   # Número de muestras deseadas
m = 50   # Intervalo entre muestras

save_sampled_steps(cond, n, m)