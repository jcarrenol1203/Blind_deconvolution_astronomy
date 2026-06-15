import os
import sys
import numpy as np
import torch
import galsim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── INYECTOR DE RUTAS (Arregla el ModuleNotFoundError: No module named 'src') ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Ruta de src/
ROOT_DIR = os.path.normpath(os.path.join(BASE_DIR, "..")) # Ruta raíz del proyecto
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
# ───────────────────────────────────────────────────────────────────────────────

from dataset import OnlineAstronomyDataset
from model import UNet48  

def psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 2.0) -> float:
    mse = np.mean((pred - target) ** 2)
    if mse == 0.0:
        return float('inf')
    return 10.0 * np.log10((data_range ** 2) / mse)

def evaluate_model(model, dataset, device, n_samples=5):
    """Evalúa exactamente n_samples aleatorios y calcula las métricas PSNR."""
    model.eval()
    psnr_initial_list = []
    psnr_final_list = []
    results_data = []

    indices = np.random.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    indices.sort()

    print(f"  Procesando {len(indices)} imágenes del catálogo COSMOS...")

    with torch.no_grad():
        for count, idx in enumerate(indices):
            x_o, x_t = dataset[int(idx)]
            inp = x_o.unsqueeze(0).to(device)
            
            refined_tensor = model(inp)

            x_o_np = x_o.squeeze().numpy()
            x_t_np = x_t.squeeze().numpy()
            refined_np = refined_tensor.squeeze().cpu().numpy()

            p_initial = psnr(x_o_np, x_t_np)
            p_final = psnr(refined_np, x_t_np)
            
            psnr_initial_list.append(p_initial)
            psnr_final_list.append(p_final)

            # 🛠️ Quitamos el cálculo del residual que ya no usaremos
            results_data.append((x_o_np, refined_np, x_t_np))
            
            print(f"    [{count + 1}/{len(indices)}] Muestra #{idx} | PSNR Entrada: {p_initial:.2f} dB | PSNR IA: {p_final:.2f} dB")

    return results_data, psnr_initial_list, psnr_final_list

def build_notebook_figure(results_data, psnr_initial, psnr_final):
    """Genera y retorna el objeto Figure adaptado visualmente sin columna residual."""
    n = len(results_data)
    if n == 0:
        return None
        
    # Reducimos el ancho de la figura de 14 a 11 ya que ahora son 3 columnas
    fig = plt.figure(figsize=(11, 3.5 * n + 1.2), facecolor='#0d0d0d')

    col_labels = [
        "Observada  x̄ₒ (Entrada)", 
        "Salida IA  x̂ (Predicción)", 
        "Ground Truth  x_t (Objetivo)"
    ]
    col_colors = ['#4a9eff', '#a78bfa', '#34d399']

    fig.suptitle(
        f"Evaluación de Deconvolución Ciega\n"
        f"PSNR Inicial Promedio: {np.mean(psnr_initial):.2f} dB   ➔   "
        f"PSNR Final Promedio: {np.mean(psnr_final):.2f} dB",
        fontsize=13, color='white', y=0.98, fontweight='bold', fontfamily='monospace'
    )

    # 🛠️ GridSpec configurado ahora a 3 columnas en vez de 4
    gs = gridspec.GridSpec(n, 3, figure=fig, hspace=0.35, wspace=0.1)

    for row, ((x_o, refined, x_t), p_i, p_f) in enumerate(zip(results_data, psnr_initial, psnr_final)):
        images = [x_o, refined, x_t]
        
        vmins  = [None, None, -1]
        vmaxs  = [None, None,  1]
        cmaps  = ['inferno', 'inferno', 'inferno']

        for col, (img, vmin, vmax, cm) in enumerate(zip(images, vmins, vmaxs, cmaps)):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(img, cmap=cm, vmin=vmin, vmax=vmax, interpolation='nearest')
            ax.axis('off')

            if row == 0:
                ax.set_title(col_labels[col], color=col_colors[col],
                             fontsize=10, pad=8, fontfamily='monospace', fontweight='bold')

            if col == 0:
                ax.text(0.5, -0.08, f"PSNR: {p_i:.2f} dB", transform=ax.transAxes, 
                        ha='center', fontsize=9, color='#4a9eff', fontfamily='monospace')
            elif col == 1:
                ganancia = p_f - p_i
                signo = "+" if ganancia >= 0 else ""
                ax.text(0.5, -0.08, f"{p_f:.2f} dB ({signo}{ganancia:.2f} dB)", transform=ax.transAxes, 
                        ha='center', fontsize=9, color='#a78bfa', fontfamily='monospace', fontweight='bold')

            if col == 0:
                ax.text(-0.15, 0.5, f"Galaxia {row+1}", transform=ax.transAxes, 
                        va='center', ha='center', fontsize=9, color='#888', rotation=90, fontfamily='monospace')

    return fig

def run_notebook_evaluation(model_filename="best_model.pth", n_samples=5):
    """Función maestra que garantiza el uso de un Test Set puro que la IA jamás ha visto."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_path = os.path.normpath(os.path.join(BASE_DIR, model_filename))
    if not os.path.exists(model_path):
        model_path = os.path.normpath(os.path.join(ROOT_DIR, "src", model_filename))
        if not os.path.exists(model_path):
            model_path = os.path.normpath(os.path.join(ROOT_DIR, model_filename))
            
    if not os.path.exists(model_path):
        print(f"❌ Error: No se encontró el archivo de pesos ({model_filename}).")
        return None

    model = UNet48(in_channels=1, out_channels=1).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    catalog_dir_final = "/home/jacl/github/Blind_deconvolution_astronomy/.venv/lib/python3.11/site-packages/galsim/share/COSMOS_23.5_training_sample"
    
    TOTAL_GALAXIES = galsim.COSMOSCatalog(file_name="real_galaxy_catalog_23.5.fits", dir=catalog_dir_final).nobjects
    
    rng = np.random.default_rng(seed=42)
    indices_permutados = rng.permutation(TOTAL_GALAXIES)
    
    # Test set puro aislado de train y validación
    pool_evaluacion = indices_permutados[6000:6500]
    
    dataset = OnlineAstronomyDataset(
        index_pool=pool_evaluacion, 
        catalog_file="real_galaxy_catalog_23.5.fits",
        catalog_dir=catalog_dir_final, 
        pixel_scale=0.03
    )
    
    results_data, psnr_initial, psnr_final = evaluate_model(model, dataset, device, n_samples=n_samples)
    fig = build_notebook_figure(results_data, psnr_initial, psnr_final)
    return fig

if __name__ == "__main__":
    print("⏳ Iniciando evaluación del modelo con tus rutas locales (Versión 3 Columnas)...")
    figura = run_notebook_evaluation(model_filename="best_model.pth", n_samples=5)
    if figura is not None:
        figura.savefig("evaluacion_unet48_resultados.png", bbox_inches='tight', dpi=150)
        print("\n✅ ¡Evaluación completada con éxito! Imagen guardada en 'evaluacion_unet48_resultados.png'")