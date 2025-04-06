import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def crear_grafico_2d(x, y, concentraciones, titulo="Concentración de Contaminantes (2D)"):
    """Crea un gráfico 2D de la concentración de contaminantes."""
    plt.figure(figsize=(10, 6))
    plt.contourf(x, y, concentraciones, levels=50, cmap='viridis')
    plt.colorbar(label="Concentración (g/m^3)")
    plt.xlabel("Distancia a lo largo del viento (m)")
    plt.ylabel("Distancia perpendicular al viento (m)")
    plt.title(titulo)
    plt.show()

def crear_grafico_3d(x, y, concentraciones, titulo="Concentración de Contaminantes (3D)"):
    """Crea un gráfico 3D de la concentración de contaminantes."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    x, y = np.meshgrid(x, y)
    ax.plot_surface(x, y, concentraciones, cmap='viridis')
    ax.set_xlabel("Distancia a lo largo del viento (m)")
    ax.set_ylabel("Distancia perpendicular al viento (m)")
    ax.set_zlabel("Concentración (g/m^3)")
    ax.set_title(titulo)
    plt.show()