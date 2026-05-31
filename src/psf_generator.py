import galsim
import numpy as np

def generate_atmospheric_psf(fwhm_max=0.4):
    """Simula la atmósfera usando el modelo Kolmogorov del paper (Sección 4.2)."""
    # El paper menciona un FWHM máximo por defecto de 0.4 arcosegundos
    fwhm = np.random.uniform(0.0, fwhm_max)
    atmospheric_psf = galsim.Kolmogorov(fwhm=fwhm)
    return atmospheric_psf, fwhm

def generate_optical_psf(lam=800.0, diam=2.4, jmax=11, aberration_scale=0.05):
    """Simula las aberraciones ópticas de Zernike (Sección 4.2)."""
    zernike_coefs = np.zeros(jmax + 1)
    # Genera aberraciones aleatorias (coma, astigmatismo, etc.)
    for j in range(4, jmax + 1):
        zernike_coefs[j] = np.random.normal(0, aberration_scale)
        
    optical_psf = galsim.OpticalPSF(
        lam=lam, 
        diam=diam, 
        aberrations=zernike_coefs,
        obscuration=0.33, # Obstrucción central típica de telescopios como el Hubble
    )
    return optical_psf, zernike_coefs

def generate_combined_psf(image_size=48, pixel_scale=0.03):
    """Convoluciona óptica y atmósfera para la imagen de 48x48 píxeles."""
    atmos, fwhm = generate_atmospheric_psf()
    optical, zernikes = generate_optical_psf()
    
    # Convolución física exacta descrita en el artículo
    final_psf_model = galsim.Convolve([atmos, optical])
    
    # Se dibuja usando la escala de placa del Hubble (0.03 arcsec/pixel)
    psf_image = galsim.ImageF(image_size, image_size, scale=pixel_scale)
    final_psf_model.drawImage(image=psf_image)
    
    return psf_image.array, {
        "fwhm_atmos": fwhm,
        "zernikes": zernikes
    }