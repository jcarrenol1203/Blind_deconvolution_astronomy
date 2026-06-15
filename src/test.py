import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import galsim
#from skimage.metrics import peak_signal_noise_ratio as psnr
#from skimage.metrics import structural_similarity as ssim


from model import UNet48
from dataset import OnlineAstronomyDataset

# ------------------------------------------------------------
# Configuración — mismos valores que train.py para garantizar
# que val_idx use exactamente los mismos índices del entrenamiento
# ------------------------------------------------------------
CATALOG_FILE  = "real_galaxy_catalog_23.5.fits"
CATALOG_DIR   = "/home/luifer/BlindDeconvolutionAstronomy/data/COSMOS_23.5_training_sample"
CHECKPOINT    = "best_model.pth"
N_TRAIN       = 5000
N_VAL         = 1000
SEED          = 42 #semilla usada en train.py para separar el dataset
N_IMAGES      = 20   # imágenes a visualizar

#Generar indices de train.py con misma semilla para obtener val_idx

def index_split(total_galaxies, n_train, n_val, seed=42):
    assert n_train + n_val <= total_galaxies, "No hay suficientes galaxias para el split."
    rng = np.random.default_rng(seed)
    indices = rng.permutation(total_galaxies)#Mezcla aleatoria de los indices de las galaxias

    #Se asignan indices
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    return train_idx, val_idx

#Cargar el modelo entrenado previamente con pesos guardados en checkpoint_path
def load_model (checkpoint_path, device):
    model= UNet48(in_channels=1, out_channels=1).to(device)
    #cargamos el checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    #Revisar si el checkpoint es un diccionario o solamente un diccionario plano con los pesos
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        print(f"   Checkpoint de época {checkpoint['epoch']+1} | "
              f"val_loss={checkpoint['val_loss']:.5f}")
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model

#Función de evaluación y plot de resultados
def evaluate_model():
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔍 Evaluando en: {device}")
    #cargar modelo
    model= load_model(checkpoint_path=CHECKPOINT, device= device)
    print(f"Modelo cargado desde {CHECKPOINT}")

    #Cargamos el catálogo y creamos la separación de indices de validación
    total_galaxies= galsim.COSMOSCatalog(file_name=CATALOG_FILE, dir=CATALOG_DIR).nobjects
    _ , val_idx = index_split(total_galaxies, N_TRAIN, N_VAL,seed=SEED)

    #Generamos dataset de validación
    val_dataset= OnlineAstronomyDataset(index_pool=val_idx[:N_IMAGES], catalog_file=CATALOG_FILE, catalog_dir=CATALOG_DIR)

    #Creamos los subplots
    fig, axes = plt.subplots(N_IMAGES, 3, figsize= (9,3*N_IMAGES))

    axes[0, 0].set_title("x_o\n(observada degradada)", fontsize=10)
    axes[0, 1].set_title("x_pred\n(reconstruida por UNet)", fontsize=10)
    axes[0, 2].set_title("x_t\n(ground truth)", fontsize=10)    

    #Ciclo principal de generación
    for i in range(N_IMAGES):
        x_o , x_t = val_dataset[i]
        with torch.no_grad():
            x_pred = model(x_o.unsqueeze(0).to(device)) #unsqueeze agrega dimensión de batch esperada por el modelo

        #Convertimos a numpy para graficar
        img_xo= x_o.squeeze().numpy()
        img_xt= x_t.squeeze().numpy()
        img_pred= x_pred.squeeze().cpu().numpy()

        axes[i, 0].imshow(img_xo,   cmap='inferno')
        axes[i, 1].imshow(img_pred, cmap='inferno')
        axes[i, 2].imshow(img_xt,   cmap='inferno')

        axes[i, 0].set_ylabel(f"Galaxy idx={val_idx[i]}", fontsize=7)
        for j in range(3):
            axes[i, j].axis('off')
    
    plt.suptitle("UNet48 — Blind Deconvolution\nObservada | Reconstruida | Ground Truth",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig("evaluation_results.png", bbox_inches='tight', dpi=150)
    print("📊 Figura guardada en evaluation_results.png")
    plt.show()

if __name__ == "__main__":
    evaluate_model()
        






