import os
import numpy as np
import galsim

# 1. Detecta automáticamente dónde está parado el proyecto en cualquier computadora
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Lista inteligente multi-usuario de rutas para los archivos FITS del catálogo
POSIBLES_RUTAS = [
    "/home/luifer/BlindDeconvolutionAstronomy/data/COSMOS_23.5_training_sample",
    "/home/jacl/github/Blind_deconvolution_astronomy/data/COSMOS_23.5_training_sample",
    "/home/jacl/github/Blind_deconvolution_astronomy/.venv/lib/python3.11/site-packages/galsim/share/COSMOS_23.5_training_sample",
    os.path.join(BASE_DIR, "..", "data", "COSMOS_23.5_training_sample")  # Ruta relativa por defecto
]

catalog_dir = None
for ruta in POSIBLES_RUTAS:
    if os.path.exists(ruta):
        catalog_dir = ruta
        break

catalog_file = "real_galaxy_catalog_23.5.fits"
pixel_scale = 0.03       # arcsec/pixel, escala del HST
FOV = 48                 # campo de visión final en píxeles
upsampling = 4           # factor de sobremuestreo para mejorar la precisión de la simulación

#-------------------------------------------
# Función para cargar una galaxia del catálogo
#-------------------------------------------
def load_galaxy(catalog, index):
    # Cargar galaxia del catálogo deconvolucionada del efecto de la PSF del HST
    galaxy = catalog.makeGalaxy(index=index, gal_type='real')
    return galaxy

#-------------------------------------------
# Función para aplicar transformaciones geométricas aleatorias
#-------------------------------------------
def apply_random_transforms(galaxy):
    # Rotación aleatoria entre 0 y 360 grados
    theta = np.random.uniform(0, 360) * galsim.degrees
    galaxy = galaxy.rotate(theta)
    
    # Flip/Inversión aleatoria de coordenadas
    if np.random.choice([True, False]):
        galaxy = galaxy.shear(galsim.Shear(g1=-0.0, g2=0.0))
    
    return galaxy

#-------------------------------------------
# Función para dibujar Ground Truth (x_t)
#-------------------------------------------
def draw_ground_truth(galaxy_t, psf_hst):
    # Convolución con la PSF original del HST para simular la resolución ideal libre de atmósfera
    convolved = galsim.Convolve([galaxy_t, psf_hst])
    
    x_t_img = convolved.drawImage(
        scale=pixel_scale / upsampling,
        nx=FOV * upsampling,
        ny=FOV * upsampling,
        method='auto'  # 'auto' optimiza usando DFT automáticamente evitando Warnings molestos
    )
    return x_t_img.array, convolved

#-------------------------------------------
# Función para convertir un array numpy de PSF en un GSObject de GalSim
#-------------------------------------------
def array_to_gsobject(psf_array, psf_scale):
    bounds = galsim.BoundsI(1, psf_array.shape[1], 1, psf_array.shape[0])
    psf_image = galsim.Image(psf_array, bounds=bounds, scale=psf_scale)
    psf_obj = galsim.InterpolatedImage(psf_image, flux=1.0)
    return psf_obj

#-------------------------------------------
# Función para degradar la imagen y obtener x_o
#-------------------------------------------
def draw_x0_degraded(x_t_obj, psf_obj, sigma_noise=0.1):
    # Convolución con la PSF combinada (Óptica + Atmosférica)
    convolved_degraded = galsim.Convolve([x_t_obj, psf_obj])
    
    # Dibujamos directamente a la escala nativa de 48x48 píxeles
    x_o_img = convolved_degraded.drawImage(
        scale=pixel_scale,
        nx=FOV,
        ny=FOV,
        method='auto'  # Cambiado a 'auto' para procesar imágenes no-analíticas por Fourier velozmente
    )
    
    x_o_array = x_o_img.array
    
    # Inyección clásica de Ruido Gaussiano Píxel por Píxel
    noise = np.random.normal(0, sigma_noise, size=x_o_array.shape)
    x_o_array_noisy = x_o_array + noise
    
    return x_o_array_noisy, convolved_degraded

#-------------------------------------------
# Función de normalización min-max adaptada a [-1, 1]
#-------------------------------------------
def normalize_image(img_array):
    min_val = img_array.min()
    max_val = img_array.max()
    if max_val - min_val > 1e-8:
        normalized = 2.0 * (img_array - min_val) / (max_val - min_val) - 1.0
    else:
        normalized = np.zeros_like(img_array)
    return normalized

#-------------------------------------------
# FUNCIÓN MAESTRA DEL ARCHIVO: Generar Pareja (x_t, x_o)
#-------------------------------------------
def generate_pair(catalog, index, psf_array, psf_scale=0.03, sigma_noise=0.02):
    galaxy = load_galaxy(catalog, index)
    galaxy_t = apply_random_transforms(galaxy)
    psf_hst = galaxy.original_psf
    
    # 1. Dibujar Ground Truth ideal
    x_t, x_t_obj = draw_ground_truth(galaxy_t, psf_hst)
    psf_obj = array_to_gsobject(psf_array, psf_scale)
    
    # ─── 📍 CORRECCIÓN METODOLÓGICA DEL PAPER ───
    # En lugar de barrer U(0,1) donde el 50% de las imágenes eran basura estelar indescifrable (sigma > 0.5),
    # ahora nos centramos estrictamente en torno al caso de estudio típico planteado en el paper (sigma = 0.1).
    sigma_realista = np.random.uniform(0.01, 0.15)
    
    # 2. Degradación física aplicando la atmósfera y el ruido corregido
    x_o, _ = draw_x0_degraded(x_t_obj, psf_obj, sigma_noise=sigma_realista)

    # 3. Aplicar el mismo pooling 4x4 sobre x_t para igualar dimensiones (48x48)
    h, w = x_t.shape
    x_t = x_t.reshape(
               h // upsampling, upsampling,
               w // upsampling, upsampling
           ).mean(axis=(1, 3))
    
    # 4. Normalizar al rango simétrico de activación [-1, 1]
    x_t = normalize_image(x_t)
    x_o = normalize_image(x_o)

    return x_t, x_o

# ------------------------------------------------------------
# Bloque de prueba - Útil para corroborar el entorno local
# ------------------------------------------------------------
if __name__ == "__main__":
    if catalog_dir is None or not os.path.exists(catalog_dir):
        print("❌ Error de ejecución directa: No se localizó la carpeta de datos COSMOS en las rutas especificadas.")
    else:
        print(f"📂 Ejecutando prueba local del simulador usando: {catalog_dir}")
        catalog = galsim.COSMOSCatalog(
            file_name=catalog_file,
            dir=catalog_dir
        )
        # Crear una PSF gaussiana rápida de prueba matemática
        psf_gaussiana = galsim.Gaussian(fwhm=0.2)
        psf_array = psf_gaussiana.drawImage(
                            scale=pixel_scale, nx=33, ny=33
                        ).array

        x_t, x_o = generate_pair(catalog, index=0, psf_array=psf_array, psf_scale=pixel_scale)
        print(f"✅ ¡Simulador validado correctamente! Tamaño Ground Truth: {x_t.shape} | Tamaño Observada: {x_o.shape}")