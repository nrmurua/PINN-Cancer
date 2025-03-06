import numpy as np
from plot_util import plot, plot2D, plot_and_save, plot2D_and_save
import os

cases = [0,1,2]   # Ajusta esto según la condición deseada
samples = [10, 20, 30]
jump = [25, 50]

for c in cases:
    for s in samples:
        for j in jump:
            file_dir = f'./noisy_data/case_{c}/noise_0.030/samples/{s}_{j}/'
            

            N = np.load(os.path.join(file_dir, 'N.npy'))
            T = np.load(os.path.join(file_dir, 'T.npy'))
            I = np.load(os.path.join(file_dir, 'I.npy'))

            # Crear el eje espacial
            x = np.linspace(0, 1, N.shape[1])  
            t = np.linspace(0, 1, N.shape[0])  

            # Graficar condición inicial (primer tiempo)
            plot_and_save(N[0], T[0], I[0], x, save_dir=file_dir, filename='init.png', plot_label='Initial case')

            # Graficar el último tiempo
            plot_and_save(N[-1], T[-1], I[-1], x, save_dir=file_dir, filename='last_known.png', plot_label='Last known case')

            # Graficar la evolución temporal de cada variable
            plot2D_and_save(N, x, t, name='Normal Cells', save_dir=file_dir, filename='N', plot_label='Evolution of Normal Cells')
            plot2D_and_save(T, x, t, name='Tumor Cells', save_dir=file_dir, filename='T', plot_label='Evolution of Tumor Cells')
            plot2D_and_save(I, x, t, name='Immune Cells', save_dir=file_dir, filename='I', plot_label='Evolution of Immune Cells')