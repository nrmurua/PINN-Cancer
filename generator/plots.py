from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

def plot(N, T, I, x):
    plt.figure(figsize=(4, 8))

    # Plot each line
    plt.plot(x, N, label='Normal Cells', color='blue')
    plt.plot(x, T, label='Tumor Cells', color='red')
    plt.plot(x, I, label='Immune Cells', color='green')

    # Customize the plot
    plt.title('Absence of Drug', fontsize=16)
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)

    # Add a horizontal line at y=0 for reference
    plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

    # Adjust the y-axis limits to better show the functions
    plt.ylim(0, 2)

    # Show the plot
    plt.show()

def plot2D(M, x, y, name='M'):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    X, Y = np.meshgrid(x, y)

    surf = ax.plot_surface(X, Y, M, cmap = plt.cm.cividis)

    ax.set_xlabel('x', labelpad=20)
    ax.set_ylabel('y', labelpad=20)
    ax.set_zlabel('z', labelpad=20)

    fig.colorbar(surf, shrink=0.5, aspect=8)

    plt.show()