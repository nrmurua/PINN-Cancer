import torch
import numpy as np
import matplotlib.pyplot as plt

def compare_data_prediction(model, t_train, x_train, train_data, full_data, 
                            time_domain, space_domain, device, save_path=None):
    plt.figure(figsize=(15,10))
    labels = ['N', 'T', 'I']

    full_t = torch.linspace(0, time_domain[0], time_domain[1], device=device)
    full_x = torch.linspace(space_domain[0], space_domain[1], space_domain[2], device=device)

    full_domain =torch.stack(full_t, full_x)

    with torch.no_grad():
        predicted_full = model(full_domain)
        predicted_full = predicted_full.view(time_domain[1], space_domain[2], -1)

    last_train_time = t_train[-1].cpu().numpy()

    plot_label = ['Normal Cells']

    for i in range(3):
        fig = plt.figure(figsize=(15, 5))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        
        X, Y = np.meshgrid(full_x.cpu().numpy(), full_t.cpu().numpy())
        
        # Plot real values
        surf1 = ax1.plot_surface(X, Y, train_data[i], cmap=plt.cm.cividis)
        ax1.set_xlabel('x', labelpad=8)
        ax1.set_ylabel('t', labelpad=8)
        ax1.set_title('Real Values', fontsize=16)
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=8)
        
        # Plot predicted values
        surf2 = ax2.plot_surface(X, Y, predicted_full[i], cmap=plt.cm.cividis)
        ax2.set_xlabel('x', labelpad=8)
        ax2.set_ylabel('t', labelpad=8)
        ax2.set_title('Predicted Values', fontsize=16)
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=8)
        
        plt.show()
        
        if save_path:
            plt.savefig(f"{save_path}_{labels[i]}.png")
        plt.show()
            

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
