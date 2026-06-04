import torch
from torch.utils.data import Dataset
import galsim
import numpy as np

# Importamos las funciones que ustedes mismos escribieron en los otros archivos
from src.psf_generator import generate_combined_psf
from src.image_simulator import generate_pair

class OnlineAstronomyDataset(Dataset):
    """
    Dataset de PyTorch que genera pares de imágenes de galaxias en tiempo real
    utilizando GalSim y los simuladores del proyecto (On-the-fly).
    """
    # CLAVE: Dejamos catalog_file=None y catalog_dir=None por defecto,
    # pero le agregamos sample="23.5" para apuntar a lo que ya tienes descargado.
    def __init__(self, catalog_file=None, catalog_dir=None, sample="23.5", pixel_scale=0.03):
        """
        Inicializa el catálogo COSMOS de GalSim usando la descarga oficial del sistema.
        """
        self.pixel_scale = pixel_scale
        
        # Al pasarle sample="23.5", GalSim buscará exactamente el archivo de 4.2GB
        # que bajaste en tu .venv automáticamente sin importar las rutas de carpetas.
        self.catalog = galsim.COSMOSCatalog(
            file_name=catalog_file,
            dir=catalog_dir,
            sample=sample
        )
        
        # Guardamos cuántas galaxias reales tiene el catálogo disponible
        self.total_galaxies = self.catalog.nobjects

    def __len__(self):
        """Devuelve el total de galaxias disponibles en el catálogo COSMOS."""
        return self.total_galaxies

    def __getitem__(self, idx):
        """
        Genera dinámicamente un par (Imagen_Observada, Imagen_Limpia) 
        para el índice solicitado en cada iteración del entrenamiento.
        """
        # 1. Generamos una PSF óptica+atmosférica aleatoria usando tu psf_generator
        psf_array, _ = generate_combined_psf(image_size=48, pixel_scale=self.pixel_scale)
        
        # 2. Llamamos a tu función maestra de image_simulator para hacer la física
        x_t_array, x_o_array = generate_pair(
            catalog=self.catalog,
            index=idx,
            psf_array=psf_array,
            psf_scale=self.pixel_scale,
            sigma_noise=0.02 # Usa el ruido aleatorio configurado
        )
        
        # 3. Convertimos a tensores de PyTorch con float32
        x_o_tensor = torch.from_numpy(x_o_array).float().unsqueeze(0)
        x_t_tensor = torch.from_numpy(x_t_array).float().unsqueeze(0)
        
        # 4. Devolvemos (Input_de_la_red, Target_esperado)
        return x_o_tensor, x_t_tensor