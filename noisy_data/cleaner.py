import os

def delete_wrong_pngs(root_dir):
    """Elimina todos los archivos que terminan en .png.png dentro de root_dir y sus subdirectorios."""
    for folder, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(",png.png"):
                file_path = os.path.join(folder, file)
                os.remove(file_path)
                print(f"Eliminado: {file_path}")


delete_wrong_pngs("./noisy_data")