import os
import matplotlib.pyplot as plt
import numpy as np

def process_and_plot_npy_file(file_path):
    """Lee un archivo .npy y genera una gráfica independiente."""
    try:
        # Cargar los datos desde el archivo .npy
        data = np.load(file_path)
        
        if data.ndim != 1:
            raise ValueError(f"Los datos no son unidimensionales en el archivo: {file_path}")
        
        # Crear eje X como índice
        x = np.arange(len(data)*0.02)
        
        # Generar la gráfica
        plt.figure()
        plt.plot(x, data, label=os.path.basename(file_path))
        plt.xlabel("Índice")
        plt.ylabel("Valores")
        plt.title(f"")
        plt.legend()
        plt.grid(True)

        # Guardar la gráfica como archivo PNG
        output_path = file_path.replace(".npy", ".png")  # Cambia la extensión del archivo
        plt.savefig(output_path)
        plt.close()
        print(f"Gráfica guardada en: {output_path}")
    except Exception as e:
        print(f"Error al procesar {file_path}: {e}")

def main():
    # Obtener el directorio actual
    root_dir = os.getcwd()

    # Recorrer los subdirectorios y buscar archivos que coincidan con el patrón
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.startswith(("I_", "N_", "T_")) and file.endswith(".npy"):
                file_path = os.path.join(subdir, file)
                process_and_plot_npy_file(file_path)

if __name__ == "__main__":
    main()