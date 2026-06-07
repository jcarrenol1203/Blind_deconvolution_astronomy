import torch
from torch.utils.data import Dataset
import galsim
import numpy as np

from psf_generator import generate_combined_psf
from image_simulator import generate_pair

# ------------------------------------------------------------
# Dataset de PyTorch que genera pares (x_o, x_t) on-the-fly.
#
# INPUT __init__:
#   index_pool   (np.ndarray): índices del catálogo COSMOS
#                asignados a este split (train o val).
#                Ejemplo: array([14823, 7341, 52011, ...])
#   catalog_file (str): nombre del archivo .fits
#   catalog_dir  (str): directorio donde está el catálogo
#   pixel_scale  (float): arcsec/pixel
# ------------------------------------------------------------
class OnlineAstronomyDataset(Dataset):

    def __init__(self, index_pool, catalog_file=None,
                 catalog_dir=None, pixel_scale=0.03):

        self.pixel_scale = pixel_scale
        self.index_pool  = index_pool   # array de 5000 o 1000 índices reales

        self.catalog = galsim.COSMOSCatalog( #se carga el catálogo COSMOS de GalSim, que contiene las galaxias reales del Hubble
            file_name=catalog_file,
            dir=catalog_dir,
        )

    def __len__(self):
        # Le dice al DataLoader hasta qué idx puede pedir: 0 a 4999
        return len(self.index_pool)

    # ------------------------------------------------------------
    # El DataLoader llama esta función con idx en [0, len-1].
    # idx es la posición dentro del split, NO el índice de COSMOS.
    #
    # INPUT:  idx (int) posición en index_pool, rango [0, 4999]
    # OUTPUT: x_o_tensor [1,48,48], x_t_tensor [1,48,48]
    # ------------------------------------------------------------
    def __getitem__(self, idx):

        # Traducción: posición local → índice real del catálogo COSMOS
        # Ejemplo: idx=37 → index_pool[37] = 14823 → galaxia 14823
        cosmos_idx = int(self.index_pool[idx])

        psf_array, _ = generate_combined_psf(
            image_size=48, pixel_scale=self.pixel_scale
        )

        x_t_array, x_o_array = generate_pair(
            catalog    = self.catalog,
            index      = cosmos_idx,  # entero, índice real de GalSim
            psf_array  = psf_array,
            psf_scale  = self.pixel_scale,
            sigma_noise= None         # muestrea U(0,1) según el paper
        )

        x_o_tensor = torch.from_numpy(x_o_array).float().unsqueeze(0)
        x_t_tensor = torch.from_numpy(x_t_array).float().unsqueeze(0)

        return x_o_tensor, x_t_tensor
    
"""if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    CATALOG_FILE = "real_galaxy_catalog_23.5.fits"
    CATALOG_DIR  = "data/COSMOS_23.5_training_sample"

    test_indices = np.array([0, 5, 120, 333, 1000, 2048, 7391, 14823, 30001, 50000])
    print(f"index_pool de prueba: {test_indices}")
    print(f"Tamaño esperado del dataset: {len(test_indices)}")
    print("-" * 50)

    dataset = OnlineAstronomyDataset(
        index_pool   = test_indices,
        catalog_file = CATALOG_FILE,
        catalog_dir  = CATALOG_DIR
    )

    print(f"dataset.__len__() = {len(dataset)}  ← debe ser 10, no 54000")
    print("-" * 50)

    # Visualizar los 10 pares: cada fila es una galaxia
    # columna izquierda = x_o (degradada), columna derecha = x_t (ground truth)
    fig, axes = plt.subplots(nrows=len(test_indices), ncols=2,
                             figsize=(5, 3 * len(test_indices)))

    for pos in range(len(test_indices)):
        x_o, x_t = dataset[pos]   # tensores [1, 48, 48]

        cosmos_idx = int(test_indices[pos])
        print(f"pos={pos} → cosmos_idx={cosmos_idx} | "
              f"x_o rango=[{x_o.min():.3f},{x_o.max():.3f}] | "
              f"x_t rango=[{x_t.min():.3f},{x_t.max():.3f}]")

        # .squeeze(0): elimina la dimensión de canal [1,48,48] → [48,48]
        # para que matplotlib pueda mostrar la imagen en escala de grises
        img_xo = x_o.squeeze(0).numpy()
        img_xt = x_t.squeeze(0).numpy()

        axes[pos, 0].imshow(img_xo, cmap='inferno', vmin=-1, vmax=1)
        axes[pos, 0].set_title(f"x_o  (cosmos idx={cosmos_idx})", fontsize=8)
        axes[pos, 0].axis('off')

        axes[pos, 1].imshow(img_xt, cmap='inferno', vmin=-1, vmax=1)
        axes[pos, 1].set_title(f"x_t  (cosmos idx={cosmos_idx})", fontsize=8)
        axes[pos, 1].axis('off')

    plt.suptitle("Pares generados por OnlineAstronomyDataset"
                 "Izquierda: observada (x_o) | Derecha: ground truth (x_t)",
                 fontsize=10, y=1.01)
    plt.tight_layout()
    plt.savefig("test_dataset_pairs.png", bbox_inches='tight', dpi=150)
    print("Figura guardada en test_dataset_pairs.png")
    plt.show()"""