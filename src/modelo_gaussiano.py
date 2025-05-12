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
    x = np.array(x)
    estabilidad = estabilidad.upper()

    if estabilidad == 'A':  # Muy inestable
        sigma_y = 0.22 * x * (1 + 0.0001 * x) ** (-0.5)
        sigma_z = 0.20 * x
    elif estabilidad == 'B':
        sigma_y = 0.16 * x * (1 + 0.0001 * x) ** (-0.5)
        sigma_z = 0.12 * x
    elif estabilidad == 'C':
        sigma_y = 0.11 * x * (1 + 0.0001 * x) ** (-0.5)
        sigma_z = 0.08 * x * (1 + 0.0002 * x) ** (-0.5)
    elif estabilidad == 'D':  # Neutra
        sigma_y = 0.08 * x * (1 + 0.0001 * x) ** (-0.5)
        sigma_z = 0.06 * x * (1 + 0.0015 * x) ** (-0.5)
    elif estabilidad == 'E':
        sigma_y = 0.06 * x * (1 + 0.0001 * x) ** (-0.5)
        sigma_z = 0.03 * x * (1 + 0.0003 * x) ** (-1)
    elif estabilidad == 'F':  # Muy estable
        sigma_y = 0.04 * x * (1 + 0.0001 * x) ** (-0.5)
        sigma_z = 0.016 * x * (1 + 0.0003 * x) ** (-1)
    else:
        raise ValueError("Clase de estabilidad no válida. Usa A, B, C, D, E o F.")
    
    return sigma_y, sigma_z
