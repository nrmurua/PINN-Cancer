import torch
import numpy as np
import matplotlib.pyplot as plt
            

# Definición de funciones de ploteo
def plot(N, T, I, x):
    plt.figure(figsize=(4, 8))

    # Graficar cada línea
    plt.plot(x, N, label='Normal Cells', color='blue')
    plt.plot(x, T, label='Tumor Cells', color='red')
    plt.plot(x, I, label='Immune Cells', color='green')

    # Personalización del gráfico
    plt.title('Absence of Drug', fontsize=16)
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)

    # Línea horizontal de referencia en y=0
    plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

    # Ajustar límites del eje y
    plt.ylim(0, 2)

    # Mostrar gráfico
    plt.show()

def plot2D(M, x, y, name='M'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if x.device != 'cpu':
        x = x.cpu()

    if y.device != 'cpu':
        y = y.cpu()

    X, Y = np.meshgrid(x, y)

    surf = ax.plot_surface(X, Y, M, cmap=plt.cm.cividis)

    ax.set_xlabel('x', labelpad=20)
    ax.set_ylabel('y', labelpad=20)
    ax.set_zlabel(name, labelpad=20)

    fig.colorbar(surf, shrink=0.5, aspect=8)

    plt.show()


def plot_training_losses(losses):
    i=0
