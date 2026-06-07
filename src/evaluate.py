"""
evaluate.py — Evaluación para la arquitectura UNet48
Calcula métricas PSNR para 5 muestras aleatorias y genera la figura.
Adaptado para integrarse perfectamente de forma visual en Jupyter Notebooks.
"""

import os
import sys
import numpy as np
import torch
import galsim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ── INYECTOR DE RUTAS (Arregla el ModuleNotFoundError: No module named 'src') ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Ruta de src/
ROOT_DIR = os.path.normpath(os.path.join(BASE_DIR, "..")) # Ruta raíz del proyecto
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
# ───────────────────────────────────────────────────────────────────────────────

from dataset import OnlineAstronomyDataset
from model import UNet48  # 🚀 CORREGIDO: Nombre alineado con model.py

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

            residual = np.abs(refined_np - x_t_np)
            results_data.append((x_o_np, refined_np, x_t_np, residual))
            
            print(f"    [{count + 1}/{len(indices)}] Muestra #{idx} | PSNR Entrada: {p_initial:.2f} dB | PSNR IA: {p_final:.2f} dB")

    return results_data, psnr_initial_list, psnr_final_list

def build_notebook_figure(results_data, psnr_initial, psnr_final):
    """Genera y retorna el objeto Figure adaptado visualmente."""
    n = len(results_data)
    if n == 0:
        return None
        
    fig = plt.figure(figsize=(14, 3.5 * n + 1.2), facecolor='#0d0d0d')

    col_labels = [
        "Observada  x̄ₒ (Entrada)", 
        "Salida IA  x̂ (Predicción)", 
        "Ground Truth  x_t (Objetivo)", 
        "Residual Final  |x̂ − xₜ|"
    ]
    col_colors = ['#4a9eff', '#a78bfa', '#34d399', '#f97316']

    fig.suptitle(
        f"Evaluación de Deconvolución Ciega (UNet48)\n"
        f"PSNR Inicial Promedio: {np.mean(psnr_initial):.2f} dB   ➔   "
        f"PSNR Final Promedio (IA): {np.mean(psnr_final):.2f} dB",
        fontsize=13, color='white', y=0.98, fontweight='bold', fontfamily='monospace'
    )

    gs = gridspec.GridSpec(n, 4, figure=fig, hspace=0.35, wspace=0.1)

    for row, ((x_o, refined, x_t, residual), p_i, p_f) in enumerate(zip(results_data, psnr_initial, psnr_final)):
        images = [x_o, refined, x_t, residual]
        
        # 🚀 CORREGIDO: Ajuste dinámico de contraste (None) para la imagen observada y la IA.
        # De esta manera, matplotlib autoajusta los límites evitando saturar visualmente el ruido de fondo.
        vmins  = [None, None, -1, 0]
        vmaxs  = [None, None,  1, max(float(residual.max()), 1e-7)]
        cmaps  = ['inferno', 'inferno', 'inferno', 'hot']

        for col, (img, vmin, vmax, cm) in enumerate(zip(images, vmins, vmaxs, cmaps)):
            ax = fig.add_subplot(gs[row, col])
            im = ax.imshow(img, cmap=cm, vmin=vmin, vmax=vmax, interpolation='nearest')
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

            if col == 3:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="8%", pad=0.04)
                cb = plt.colorbar(im, cax=cax)
                cb.ax.tick_params(labelsize=7, colors='#888')
                cb.outline.set_edgecolor('#333')

    return fig

def run_notebook_evaluation(model_filename="best_model.pth", n_samples=5):
    """Función maestra modificada para ejecutar 5 muestras desde Jupyter Notebook."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_path = os.path.normpath(os.path.join(BASE_DIR, model_filename))
    if not os.path.exists(model_path):
        model_path = os.path.normpath(os.path.join(ROOT_DIR, "src", model_filename))
        if not os.path.exists(model_path):
            model_path = os.path.normpath(os.path.join(ROOT_DIR, model_filename))
            
    if not os.path.exists(model_path):
        print(f"❌ Error: No se encontró el archivo de pesos ({model_filename}).")
        return None

    # Instanciamos la clase correcta del modelo UNet48
    model = UNet48(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Ruta absoluta del entorno virtual al catálogo COSMOS
    catalog_dir_final = "/home/jacl/github/Blind_deconvolution_astronomy/.venv/lib/python3.11/site-packages/galsim/share/COSMOS_23.5_training_sample"
    
    # Generamos un index_pool de validación dinámico rápido (ej: 500 primeras galaxias)
    pool_evaluacion = np.arange(0, 500)
    
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
    pass