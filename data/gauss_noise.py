import os
import glob
import numpy as np

def add_noise_to_data(file_pattern, noise_level=0.03, output_dir="output"):
    """
    Add Gaussian noise to data files and save the noisy data to a subdirectory 
    named after the noise level.

    Args:
        file_pattern (str): Glob pattern to match the input files (e.g., './data/*.npy').
        noise_level (float): Standard deviation of the Gaussian noise to add.
        output_dir (str): Directory where noisy data will be saved.
    """
    # Create a subdirectory for this noise level
    noise_folder = os.path.join(output_dir, f"noise_{noise_level:.3f}")
    os.makedirs(noise_folder, exist_ok=True)

    # Process each file
    files = glob.glob(file_pattern)
    for file in files:
        # Load data from the file
        data = np.load(file)

        # Add Gaussian noise
        noisy_data = data + np.random.normal(0, noise_level, size=data.shape)

        # Save the noisy data to a new file in the noise level folder
        file_name = os.path.basename(file)  # Extract file name without the path
        output_file = os.path.join(noise_folder, file_name)
        np.save(output_file, noisy_data)
        print(f"Saved noisy data to {output_file}")

# Ejemplo de uso
if __name__ == "__main__":
    file_pattern = "./data/1D/case_2/*.npy"  
    noise_levels = [0.030, 0.003]  

    # Carpeta de salida principal
    output_dir = "./noisy_data/case_2"

    # Generar datos ruidosos para cada nivel de ruido
    for noise_level in noise_levels:
        print(f"Procesando datos con ruido {noise_level}")
        add_noise_to_data(file_pattern, noise_level=noise_level, output_dir=output_dir)
