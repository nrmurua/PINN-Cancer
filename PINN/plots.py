import torch
import numpy as np
import matplotlib.pyplot as plt
import os            

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


def plot_training_losses(losses, log_scale=True, save_dir=None):
    loss_path = os.path.join(save_dir, "losses.png")

    plt.figure(figsize=(10, 6))
    
    plt.plot(range(len(losses)), losses, label='Total Loss', color='blue', linewidth=1.5)
    
    if log_scale:
        plt.yscale('log')
    
    plt.title('Training Losses over Epochs', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (Log Scale)' if log_scale else 'Loss', fontsize=12)
    plt.legend(fontsize=10)
    
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if loss_path:
        plt.savefig(loss_path, dpi=300)
        print(f"Training loss plot saved to {loss_path}")
    else:
        plt.show()
    
    plt.close()