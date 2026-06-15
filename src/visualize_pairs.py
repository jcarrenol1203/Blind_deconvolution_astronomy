# visualize_pairs.ipynb  — pegar en un notebook o correr como script
# Muestra 20 pares (x_o, x_t) generados por el pipeline corregido

import sys
sys.path.insert(0, '/home/luifer/BlindDeconvolutionAstronomy/src')

import numpy as np
import matplotlib.pyplot as plt
import galsim

from psf_generator   import generate_combined_psf
from image_simulator import generate_pair

# --- Configuración ---
CATALOG_FILE = "real_galaxy_catalog_23.5.fits"
CATALOG_DIR  =  "/home/jacl/github/Blind_deconvolution_astronomy/.venv/lib/python3.11/site-packages/galsim/share/COSMOS_23.5_training_sample"
N_IMAGES     = 20
SEED         = 42

# --- Cargar catálogo ---
catalog = galsim.COSMOSCatalog(file_name=CATALOG_FILE, dir=CATALOG_DIR)
print(f"Catálogo cargado: {catalog.nobjects} galaxias disponibles")

# --- Índices de prueba reproducibles ---
rng     = np.random.default_rng(SEED)
indices = rng.integers(0, catalog.nobjects, size=N_IMAGES)
print(f"Índices a visualizar: {indices}")

# --- Generar pares y visualizar ---
fig, axes = plt.subplots(nrows=N_IMAGES, ncols=4,
                         figsize=(9, 3 * N_IMAGES))

axes[0, 0].set_title("x_o\n(observada + ruido)", fontsize=10)
axes[0, 1].set_title("x_t\n(ground truth)", fontsize=10)
axes[0, 2].set_title("x_t\n(Optical PSF)", fontsize=10)
axes[0, 3].set_title("PSF generada", fontsize=10)

for i, idx in enumerate(indices):
    # Generar PSF con los parámetros corregidos
    psf_array, psf_obj, optical, psf_params = generate_combined_psf(
        image_size=192, pixel_scale=0.03
    )

    # Generar par con sigma_noise=None → muestrea U(0,1)
    x_t, x_o = generate_pair(
        catalog    = catalog,
        index      = int(idx),
        psf_obj  = psf_obj,
        sigma_noise= None
    )

    # --- Columna 1: x_o degradada ---
    axes[i, 0].imshow(x_o, cmap='inferno', vmin=-1, vmax=1)
    axes[i, 0].set_ylabel(f"idx={idx}", fontsize=7)
    axes[i, 0].axis('off')

    # --- Columna 2: x_t ground truth ---
    axes[i, 1].imshow(x_t, cmap='inferno', vmin=-1, vmax=1)
    axes[i, 1].axis('off')
    #Optical PSF
    psf_opti = (optical - optical.min())
    psf_opti = psf_opti / psf_opti.max()
    axes[i, 2].imshow(psf_opti, cmap='viridis')
    axes[i, 2].set_title(
        f"fwhm={psf_params['fwhm_atmos']:.3f}\"",
        fontsize=6
    )
    axes[i, 2].axis('off')
    # --- Columna 4: PSF generada ---
    # Normalización propia para ver la forma de la PSF claramente
    psf_norm = (psf_array - psf_array.min())
    psf_norm = psf_norm / psf_norm.max()
    axes[i, 3].imshow(psf_norm, cmap='viridis')
    axes[i, 3].set_title(
        f"fwhm={psf_params['fwhm_atmos']:.3f}\"",
        fontsize=6
    )
    axes[i, 3].axis('off')

    print(f"[{i+1:2d}/20] idx={idx:5d} | "
          f"x_o=[{x_o.min():.3f},{x_o.max():.3f}] | "
          f"x_t=[{x_t.min():.3f},{x_t.max():.3f}] | "
          f"fwhm={psf_params['fwhm_atmos']:.3f}\"")

plt.suptitle("Pipeline corregido — 20 pares generados\n"
             "Observada | Ground Truth | PSF",
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig("pipeline_check.png", bbox_inches='tight', dpi=150)
print("\n✅ Figura guardada en pipeline_check.png")