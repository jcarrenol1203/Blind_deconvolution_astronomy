import numpy as np
import galsim

#Constantes de configuración del pipeline
catalog_file = "real_galaxy_catalog_23.5.fits"   # Archivo de catálogo de galaxias reales
catalog_dir = "/home/luifer/BlinDeconvolutionAstronomy/data/COSMOS_23.5_training_sample"  # Directorio donde se encuentra el catálogo
pixel_scale= 0.03 #arcsec/pixel, escala del HST
FOV = 48 #campo vision final en pixeles
upsampling= 4 #factor de sobremuestreo para mejorar la precisión de la simulación

#-------------------------------------------
#Función para cargar una galaxia del catálogo
#Utiliza galsim.COSMOSCatalog para acceder al catálogo y
#utiliza galsim.RealGalaxy para obtener GSObject representando
#la galaxia desconvolucionada del efecto de la PSF del HST

#INPUT: catalog, index (int, indice dentro del catalogo)
#OUTPUT: GSObject de la galaxia lista para transformar
#-------------------------------------------

def load_galaxy(catalog, index):
    #Cargar galaxia del catálogo
    #NOTA: Se usa makeGalaxy con gal_type='real' para obtener la galaxia desconvolucionada
    #makeGalaxy es el metodo recomendado para usar con la clase COSMOSCatalog, ya que maneja internamente la deconvolución y otros detalles
    galaxy= catalog.makeGalaxy(index=index, gal_type='real')
    return galaxy

#-------------------------------------------
#Función para aplicar transformaciones geometricas aleatorias
#Estas transformaciones sirven para simular variabilidad morfologica
#debido a efectos de lente gravitacional y orientaciones aleatorias de las galaxias
#
#INPUT: galaxy (galsim.GSObject)
#OUTPUT: galaxy transformada (galsim.GSObject), parametros usados (dict)
#-------------------------------------------

def apply_random_transforms(galaxy):
    #Rotación: solo multipolos de 90° para evitar aliasing
    angle= np.random.choice([0, 90, 180, 270])
    galaxy = galaxy.rotate(angle * galsim.degrees)

    #Traslación: offset aleatorio en x e y, en unidades de pixel
    #Guiandonos del paper, U(-1, 1)
    dx= np.random.uniform(-1, 1)
    dy= np.random.uniform(-1, 1)
    galaxy = galaxy.shift(dx * pixel_scale, dy * pixel_scale)

    #Magnificación: U(1, 1.1)
    scale = np.random.uniform(1, 1.1)
    galaxy = galaxy.magnify(scale)

    # Shear: magnitud según pdf(x)=x en [0.01, 0.05]
    # Ángulo de posición: 2π * N(0,1)
    min_shear, max_shear = 0.01, 0.05
    shear_mag = np.sqrt(np.random.uniform(min_shear**2, max_shear**2))
    shear_angle = 2 * np.pi * np.random.normal(0, 1)
    g1 = shear_mag * np.cos(2 * shear_angle)
    g2 = shear_mag * np.sin(2 * shear_angle)
    galaxy = galaxy.shear(g1=g1, g2=g2)

    params = {
        'angle_deg' : angle,
        'dx_pix'    : dx,
        'dy_pix'    : dy,
        'scale'        : scale,
        'shear_mag' : shear_mag,
        'shear_angle': shear_angle
    }
    return galaxy, params

#-------------------------------------------
#Función para dibujar el gound truth X_t convolucionando el GSObject con la PSF del HST
#La convolución con PSF_HST actua como filtro de pasabajas antes del muestreo, respetando 
#el teorema de muestreo de Nyquist.

#INPUT: galaxy_transformada (galsim.GSObject), PSF_HST (galsim.GSObject)
#OUTPUT: x_t (np.ndarray), imagen ground truth de (FOV*Upsampling, FOV*Upsampling) 192 x 192 pixeles.
#-------------------------------------------

def draw_ground_truth(galaxy, PSF_HST):
    #convolucionar con la PSF del HST
    image_obj= galsim.Convolve([galaxy, PSF_HST])

    #dibujar la imagen con sobremuestreo
    #se usa drawImagen, metodo que muestre el GSObject en una malla de pixeles
    image = image_obj.drawImage(scale=pixel_scale, nx=FOV*upsampling, ny=FOV*upsampling)

    # .array para extraer el array numpy de la imagen de galsim
    return image.array , image_obj
    
#-------------------------------------------
#Función para convertir un array numpy a un objeto GSObject de galsim
#Necesario para convolucionar la PSF_new generada por psf_generator.py
#INPUT:psf_array (np.ndarray), pixel_scale (float)
#OUTPUT: psf_gsobject (galsim.GSObject)
#-------------------------------------------    

def array_to_gsobject(psf_array, pixel_scale):
    #Normalizar PSF para que la suma sea 1, asegurando que conserve la energía
    psf_array= psf_array / psf_array.sum()

    #galsim.Image envuelve el array np en un objeto GalSim
    #indicando el pixel scale para mantener las unidades correctas
    psf_image= galsim.Image(psf_array.astype(np.float64), scale=pixel_scale)

    #Crear GSObject con el metodo galsim.InterpolatedImage, que permite usar la imagen como una función continua
    psf_obj = galsim.InterpolatedImage(psf_image)

    return psf_obj

#-------------------------------------------
#Función para degradar imagen observada, y generar x_o aplicando tres procesos:
#1. Convolución con PSF_new (galsim.GSObject, óptica + atmosfera)
#2. Average pooling 4x4 para reducir resolución a 48x48 pixeles
#3. Agregar ruido gaussiano blanco aditivo
#
#INPUT: x_t (galsim.GSObject), PSF_new (galsim.GSObject), noise_std (float)~U(0,1)
#OUTPUT: x_o (np.ndarray), noise_std (float, para debugging)
#-------------------------------------------

def draw_x0_degraded(x_t_obj, psf_obj, noise_std =None):

    if noise_std is None:
        noise_std = np.random.uniform(0, 1)
    #Convolucion con PSF_new
    degraded_image= galsim.Convolve([x_t_obj, psf_obj])

    degraded_array = degraded_image.drawImage(scale=pixel_scale, nx=FOV*upsampling, ny=FOV*upsampling).array

    # Average pooling 4x4 para reducir la resolución a 48x48
    h, w = degraded_array.shape
    x_o  = degraded_array.reshape(
               h // upsampling, upsampling,
               w // upsampling, upsampling
           ).mean(axis=(1, 3))

    # Agregar ruido gaussiano blanco aditivo
    if noise_std > 0:
        x_o = x_o + np.random.normal(scale=noise_std, size=x_o.shape)

    return x_o, noise_std

#-------------------------------------------
#Función para normalizar imágenes a rango [-1,1]
# INPUT:
#   image (np.ndarray): array de cualquier rango
#
# OUTPUT:
#   image_norm (np.ndarray): array normalizado en [-1, 1]
# ------------------------------------------------------------
def normalize_image(image):
    x_min = image.min()
    x_max = image.max()

    # Evitar división por cero si la imagen es constante
    if x_max == x_min:
        return np.zeros_like(image)

    return 2.0 * (image - x_min) / (x_max - x_min) - 1.0


# ------------------------------------------------------------
# Función principal del simulador. Genera un par (x_t, x_o)
# completo ejecutando todo el pipeline de forward modelling.
# Esta es la función que llamará dataset.py en cada iteración.
#
# INPUT:
#   catalog    (galsim.COSMOSCatalog): catálogo ya cargado
#   index      (int): índice de la galaxia
#   psf_array  (np.ndarray): PSF generada por psf_generator.py
#   psf_scale  (float): pixel scale de la PSF en arcsec/pixel
#   sigma_noise(float o None): si None se muestrea de U(0,1)
#
# OUTPUT:
#   x_t (np.ndarray): ground truth normalizado 192x192
#   x_o (np.ndarray): imagen observada normalizada 48x48
# ------------------------------------------------------------
def generate_pair(catalog, index, psf_array, psf_scale, sigma_noise=None):

    galaxy           = load_galaxy(catalog, index)
    galaxy_t, _      = apply_random_transforms(galaxy)
    psf_hst          = galaxy.original_psf
    x_t, x_t_obj     = draw_ground_truth(galaxy_t, psf_hst)
    psf_obj          = array_to_gsobject(psf_array, psf_scale)
    x_o, _           = draw_x0_degraded(x_t_obj, psf_obj, sigma_noise)

    # Aplicar el mismo pooling 4x4 sobre x_t para que ambas
    # imágenes tengan el mismo tamaño 48x48 al entrar a la red
    h, w = x_t.shape
    x_t  = x_t.reshape(
               h // upsampling, upsampling,
               w // upsampling, upsampling
           ).mean(axis=(1, 3))
    
    x_t              = normalize_image(x_t)
    x_o              = normalize_image(x_o)

    return x_t, x_o



# ------------------------------------------------------------
# Bloque de prueba - eliminar antes de usar en entrenamiento
# ------------------------------------------------------------
if __name__ == "__main__":

    catalog = galsim.COSMOSCatalog(
        file_name=catalog_file,
        dir=catalog_dir
    )
    # PSF placeholder de prueba
    psf_gaussiana = galsim.Gaussian(fwhm=0.2)
    psf_array     = psf_gaussiana.drawImage(
                        scale=pixel_scale, nx=33, ny=33
                    ).array

    x_t, x_o = generate_pair(
        catalog   = catalog,
        index     = 0,
        psf_array = psf_array,
        psf_scale = pixel_scale,
    )

    print(f"x_t → shape: {x_t.shape} | rango: [{x_t.min():.4f}, {x_t.max():.4f}]")
    print(f"x_o → shape: {x_o.shape} | rango: [{x_o.min():.4f}, {x_o.max():.4f}]")

    