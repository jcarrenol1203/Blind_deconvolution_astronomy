import os
import numpy as np
import torch
from torch.utils.data import Dataset

class AstronomyDataset(Dataset):
    """
    Dataset personalizado para cargar las parejas de galaxias 
    (Sucia/Observada -> Limpia/Target) para el entrenamiento de la U-Net.
    """
    def __init__(self, data_dir, transform=None):
        """
        data_dir: Ruta a la carpeta donde guardaremos las imágenes simualdas.
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Listamos todos los archivos de galaxias sucias como referencia
        # Asumiendo que guardaremos archivos numerados tipo: 'dirty_0001.npy' y 'clean_0001.npy'
        self.filenames = [f for f in os.listdir(data_dir) if f.startswith('dirty_')]
        self.filenames.sort() # Los ordenamos para que coincidan perfectamente

    def __len__(self):
        """Devuelve el número total de imágenes en el dataset."""
        return len(self.filenames)

    def __getitem__(self, idx):
        """Carga y devuelve un par (imagen_sucia, imagen_limpia) dado un índice."""
        # 1. Obtener los nombres de los archivos correspondientes
        dirty_name = self.filenames[idx]
        clean_name = dirty_name.replace('dirty_', 'clean_')
        
        # 2. Construir las rutas completas
        dirty_path = os.path.join(self.data_dir, dirty_name)
        clean_path = os.path.join(self.data_dir, clean_name)
        
        # 3. Cargar las matrices desde el disco duro (.npy de NumPy)
        dirty_img = np.load(dirty_path).astype(np.float32)
        clean_img = np.load(clean_path).astype(np.float32)
        
        # 4. Convertir a Tensores de PyTorch (añadiendo la dimensión del canal: [Canal, Alto, Ancho])
        # Como son imágenes astronómicas en escala de grises, el canal es 1.
        dirty_tensor = torch.from_numpy(dirty_img).unsqueeze(0)
        clean_tensor = torch.from_numpy(clean_img).unsqueeze(0)
        
        # Aplicar transformaciones o aumentos de datos si se requieren (ej. rotaciones en caliente)
        if self.transform:
            dirty_tensor = self.transform(dirty_tensor)
            clean_tensor = self.transform(clean_tensor)
            
        return dirty_tensor, clean_tensor