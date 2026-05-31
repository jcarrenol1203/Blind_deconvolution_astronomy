import galsim
import numpy as np

def generate_atmospheric_psf(fwhm_range=(0.5, 1.2)):
    """Simula la turbulencia de la atmósfera (seeing) usando el modelo de Kolmogorov."""
    fwhm = np.random.uniform(fwhm_range[0], fwhm_range[1])
    atmospheric_psf = galsim.Kolmogorov(fwhm=fwhm)
    return atmospheric_psf, fwhm

def generate_optical_psf(lam=800.0, diam=2.4, jmax=11, aberration_scale=0.05):
    """Simula las aberraciones ópticas del telescopio usando polinomios de Zernike."""
    zernike_coefs = np.zeros(jmax + 1)
    for j in range(4, jmax + 1):
        zernike_coefs[j] = np.random.normal(0, aberration_scale)
        
    optical_psf = galsim.OpticalPSF(
        lam=lam, 
        diam=diam, 
        aberrations=zernike_coefs,
        obscuration=0.33,
    )
    return optical_psf, zernike_coefs

def generate_combined_psf(image_size=48, pixel_scale=0.03):
    """Convoluciona la atmósfera y la óptica para entregar la matriz de píxeles final."""
    atmos, fwhm = generate_atmospheric_psf()
    optical, zernikes = generate_optical_psf()
    
    final_psf_model = galsim.Convolve([atmos, optical])
    
    psf_image = galsim.ImageF(image_size, image_size, scale=pixel_scale)
    final_psf_model.drawImage(image=psf_image)
    
    return psf_image.array, {
        "fwhm_atmos": fwhm,
        "zernikes": zernikes
    }