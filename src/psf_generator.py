import galsim
import numpy as np

def generate_atmospheric_psf(fwhm_max=0.4, atmos_e=0.01):
    """Simula la atmósfera usando el modelo Kolmogorov del paper (Sección 4.2) y agregando elipticidad a la PSF."""
    # El paper menciona un FWHM máximo por defecto de 0.4 arcosegundos
    fwhm = fwhm_max * np.random.uniform(0, 1) #Full Width at Half Maximum (FWHM) aleatorio entre 0 y 0.4 arcsec
    atmospheric_psf = galsim.Kolmogorov(fwhm=fwhm) #Físicamente mide como se dispersa la luz debido a la turbulencia atmosférica, lo que afecta la calidad de la imagen astronómica.
    # Magnitud del shear eliptico: atmos_e x U[1,2]
    ellipticity = atmos_e * np.random.uniform(1,2)
    #Shear angle: 2pi*N(0,1)
    shear_angle= 2* np.pi * np.random.normal(0,1)
    # Descomposición en componentes g1, g2 del shear reducido
    # g1 = e·cos(2θ): componente horizontal/vertical
    # g2 = e·sin(2θ): componente diagonal
    # El factor 2 en el ángulo refleja la simetría de spin-2 del shear
    g1 = ellipticity * np.cos(2 * shear_angle)
    g2 = ellipticity * np.sin(2 * shear_angle)
    atmospheric_psf = atmospheric_psf.shear(g1=g1, g2=g2)

    return atmospheric_psf, fwhm

def generate_optical_psf(lam=814.0, jmax=11, opt_obs_min =0.1, opt_obs_width= 0.4): #800 nm es una longitud de onda típica para observaciones ópticas, diam=2.4 m es el diámetro del telescopio Hubble, jmax=11 incluye términos de Zernike hasta el orden 11, y aberration_scale controla la magnitud de las aberraciones aleatorias.
    """Simula las aberraciones ópticas de Zernike (Sección 4.2)."""
    zernike_coefs = np.zeros(jmax + 1) #Inicializa los coeficientes de Zernike a cero, lo que corresponde a una óptica perfecta sin aberraciones. Luego, se generan aberraciones aleatorias para los términos de Zernike.
    #Se define obscuration segun el paper 
    obscuration = opt_obs_min + (opt_obs_width * np.random.uniform(0,1))
    #Proceso para calcular lambda/D, con lambda Fijo, variamos D que es el parametro que pide GalSim
    lam_ov_d = 0.017 + (0.007* np.random.uniform(0,1))
    diam= (lam * 1e-9) / (lam_ov_d * (np.pi/648000))
    
    # Genera aberraciones aleatorias (coma, astigmatismo, etc.)
    for j in range(4, jmax + 1): #Los primeros 4 términos corresponden a desplazamientos y enfoque, que no se consideran aberraciones en este contexto. 
        if j==4: #defocusing
            zernike_coefs[j] = np.random.normal(0, 0.10)
        else: #Resto de aberraciones
            zernike_coefs[j] = np.random.normal(0, 0.07)
        
    optical_psf = galsim.OpticalPSF(
        lam=lam, 
        diam=diam, 
        aberrations=zernike_coefs,
        obscuration= obscuration, # Obstrucción central típica de telescopios como el Hubble
    )
    return optical_psf, zernike_coefs

def generate_combined_psf(image_size=192, pixel_scale=0.03): #0.03 es la relación de arcosegundos por píxel para el Hubble, lo que significa que cada píxel representa 0.03 arcosegundos en el cielo.
    """Convoluciona óptica y atmósfera para la imagen de 192x192 píxeles."""
    atmos, fwhm = generate_atmospheric_psf()
    psf_optical, zernikes = generate_optical_psf()
    
    # Convolución física exacta descrita en el artículo
    final_psf_model = galsim.Convolve([atmos, psf_optical]) #Convoluciona la PSF atmosférica y óptica para obtener la PSF combinada que simula tanto los efectos de la atmósfera como las aberraciones ópticas del telescopio.
    
    optical = galsim.ImageF(image_size, image_size, scale=pixel_scale) 
    psf_optical.drawImage(image=optical)
    
    # Se dibuja usando la escala de placa del Hubble (0.03 arcsec/pixel)
    psf_image = galsim.ImageF(image_size, image_size, scale=pixel_scale) 
    final_psf_model.drawImage(image=psf_image)
    
    return psf_image.array, final_psf_model,optical.array,{
        "fwhm_atmos": fwhm,
        "zernikes": zernikes
    } 