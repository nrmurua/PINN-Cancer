import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

    X, Y = np.meshgrid(x, y)

    surf = ax.plot_surface(X, Y, M, cmap=plt.cm.cividis)

    ax.set_xlabel('x', labelpad=20)
    ax.set_ylabel('y', labelpad=20)
    ax.set_zlabel(name, labelpad=20)

    fig.colorbar(surf, shrink=0.5, aspect=8)

    plt.show()

def plot_and_save(N, T, I, x, save_dir, filename="plot.png", plot_label="Initial case"):
    """Genera y guarda un gráfico de las tres variables en el directorio especificado."""
    os.makedirs(save_dir, exist_ok=True)  # Crear la carpeta si no existe

    plt.figure(figsize=(8, 8))

    # Plot each line
    plt.plot(x, N, label='Normal Cells', color='blue')
    plt.plot(x, T, label='Tumor Cells', color='red')
    plt.plot(x, I, label='Immune Cells', color='green')

    # Customize the plot
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)

    # Add a horizontal line at y=0 for reference
    plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

    # Adjust the y-axis limits to better show the functions
    plt.ylim(0, 2)

    plt.title(plot_label, fontsize=16)

    # Save the plot
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Cierra la figura para liberar memoria

    print(f"Gráfico guardado en: {save_path}")

def plot2D_and_save(M, x, y, name, save_dir, filename="plot2D.png", plot_label=''):
    """Genera y guarda una gráfica 3D de M en el directorio especificado."""
    os.makedirs(save_dir, exist_ok=True)  # Crear la carpeta si no existe

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    X, Y = np.meshgrid(x, y)

    surf = ax.plot_surface(X, Y, M, cmap=plt.cm.cividis)

    ax.set_xlabel('x', labelpad=10)
    ax.set_ylabel('t', labelpad=10)

    fig.colorbar(surf, shrink=0.5, aspect=8)

    plt.title(plot_label, fontsize=16)

    # Save the plot
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=800, bbox_inches='tight')
    plt.close()  # Cierra la figura para liberar memoria

    print(f"Gráfico 3D guardado en: {save_path}")