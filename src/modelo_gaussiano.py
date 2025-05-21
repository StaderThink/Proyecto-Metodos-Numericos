import numpy as np

def calcular_sigma(x, estabilidad):
    estabilidad = estabilidad.upper()
    if estabilidad == 'A':
        sigma_y = 0.22 * x * (1 + 0.0001 * x) ** -0.5
        sigma_z = 0.20 * x
    elif estabilidad == 'B':
        sigma_y = 0.16 * x * (1 + 0.0001 * x) ** -0.5
        sigma_z = 0.12 * x
    elif estabilidad == 'C':
        sigma_y = 0.11 * x * (1 + 0.0001 * x) ** -0.5
        sigma_z = 0.08 * x * (1 + 0.0002 * x) ** -0.5
    elif estabilidad == 'D':
        sigma_y = 0.08 * x * (1 + 0.0001 * x) ** -0.5
        sigma_z = 0.06 * x * (1 + 0.0015 * x) ** -0.5
    elif estabilidad == 'E':
        sigma_y = 0.06 * x * (1 + 0.0001 * x) ** -0.5
        sigma_z = 0.03 * x * (1 + 0.0003 * x) ** -1
    elif estabilidad == 'F':
        sigma_y = 0.04 * x * (1 + 0.0001 * x) ** -0.5
        sigma_z = 0.016 * x * (1 + 0.0003 * x) ** -1
    else:
        raise ValueError("Estabilidad desconocida")
    return sigma_y, sigma_z

def calcular_concentracion(Q, u, H, sigma_y, sigma_z, x, y, z):
    if x == 0:
        return 0
    term1 = Q / (2 * np.pi * u * sigma_y * sigma_z)
    term2 = np.exp(-y**2 / (2 * sigma_y**2))
    term3 = np.exp(-(z - H)**2 / (2 * sigma_z**2)) + np.exp(-(z + H)**2 / (2 * sigma_z**2))
    return term1 * term2 * term3