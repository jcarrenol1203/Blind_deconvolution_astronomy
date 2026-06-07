import torch
from torch.utils.data import Dataset
import galsim
import numpy as np

from psf_generator import generate_combined_psf
from image_simulator import generate_pair

# ------------------------------------------------------------
# Dataset de PyTorch que genera pares (x_o, x_t) on-the-fly.
# ------------------------------------------------------------
class OnlineAstronomyDataset(Dataset):

    def __init__(self, index_pool, catalog_file=None,
                 catalog_dir=None, pixel_scale=0.03):

        self.pixel_scale = pixel_scale
        self.index_pool  = index_pool   # array de índices reales del catálogo

        self.catalog = galsim.COSMOSCatalog(
            file_name=catalog_file,
            dir=catalog_dir,
        )

    def __len__(self):
        return len(self.index_pool)

    def __getitem__(self, idx):
        # Traducción: posición local → índice real del catálogo COSMOS
        cosmos_idx = int(self.index_pool[idx])

        # Generar PSF combinada (Atmósfera + Óptica)
        psf_array, _ = generate_combined_psf(
            image_size=48, pixel_scale=self.pixel_scale
        )

        # 🚀 CORREGIDO: Se elimina 'sigma_noise=None' para que use internamente
        # la distribución aleatoria acotada U(0.01, 0.15) de image_simulator.py
        x_t_array, x_o_array = generate_pair(
            catalog    = self.catalog,
            index      = cosmos_idx,  
            psf_array  = psf_array,
            psf_scale  = self.pixel_scale
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