import numpy as np
from scipy.constants import pi

def calcular_concentracion(Q, u, H, sigma_y, sigma_z, x, y, z):
    """Calcula la concentración de contaminante en un punto (x, y, z) usando el modelo gaussiano."""
    termino_1 = Q / (2 * pi * u * sigma_y * sigma_z)
    termino_2 = np.exp(-y**2 / (2 * sigma_y**2))
    termino_3 = np.exp(-(z-H)**2 / (2 * sigma_z**2)) + np.exp(-(z+H)**2 / (2 * sigma_z**2))
    C = termino_1 * termino_2 * termino_3
    return C

def calcular_sigma(x, estabilidad):
    """Calcula los coeficientes de dispersión sigma_y y sigma_z según la estabilidad atmosférica."""
    # Estos valores se deben ajustar para que seaan mas puntuales
    if estabilidad == 'A':  # Muy inestable
        sigma_y = 0.4 * x
        sigma_z = 0.5 * x
    elif estabilidad == 'D':  # Neutro
        sigma_y = 0.1 * x
        sigma_z = 0.05 * x
    else:  # Estable
        sigma_y = 0.05 * x
        sigma_z = 0.01 * x
    return sigma_y, sigma_z