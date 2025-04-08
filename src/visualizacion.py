import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd  # Asegúrate de importar pandas aquí

def crear_grafico_2d(x, y, concentraciones, titulo="Concentración de Contaminantes (2D)"):
    """
    Crea un gráfico 2D de la concentración de contaminantes.

    Args:
        x: Array con las distancias a lo largo del viento (m).
        y: Array con las distancias perpendiculares al viento (m).
        concentraciones: Matriz 2D con las concentraciones calculadas.
        titulo: Título del gráfico.
    """
    plt.figure(figsize=(10, 6))
    plt.contourf(x, y, concentraciones, levels=50, cmap='viridis')
    plt.colorbar(label="Concentración (g/m^3)")
    plt.xlabel("Distancia a lo largo del viento (m)")
    plt.ylabel("Distancia perpendicular al viento (m)")
    plt.title(titulo)
    plt.show()

def crear_grafico_3d(x, y, concentraciones, titulo="Concentración de Contaminantes (3D)"):
    """
    Crea un gráfico 3D de la concentración de contaminantes.

    Args:
        x: Array con las distancias a lo largo del viento (m).
        y: Array con las distancias perpendiculares al viento (m).
        concentraciones: Matriz 2D con las concentraciones calculadas.
        titulo: Título del gráfico.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    x, y = np.meshgrid(x, y)
    ax.plot_surface(x, y, concentraciones, cmap='viridis')
    ax.set_xlabel("Distancia a lo largo del viento (m)")
    ax.set_ylabel("Distancia perpendicular al viento (m)")
    ax.set_zlabel("Concentración (g/m^3)")
    ax.set_title(titulo)
    plt.show()

def crear_grafico_dispersion(df, concentraciones, titulo="Dispersión de Contaminantes"):
    """
    Crea un gráfico de dispersión de la concentración de contaminantes en función de la ubicación.

    Args:
        df: DataFrame de pandas con los datos (incluye latitud y longitud).
        concentraciones: Lista o array con las concentraciones calculadas para cada ubicación.
        titulo: Título del gráfico.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Longitud'], df['Latitud'], c=concentraciones, cmap='viridis', s=50)
    plt.colorbar(label="Concentración (g/m^3)")
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.title(titulo)
    plt.grid(True)
    plt.show()
def crear_mapa_calor(df, concentraciones, titulo="Mapa de Calor de Concentración de Contaminantes"):
    """
    Crea un mapa de calor de la concentración de contaminantes en función de la ubicación.

    Args:
        df: DataFrame de pandas con los datos (incluye latitud y longitud).
        concentraciones: Lista o array con las concentraciones calculadas para cada ubicación.
        titulo: Título del gráfico.
    """
    
    # Crear un DataFrame con las columnas necesarias
    data = {'Longitud': df['Longitud'], 'Latitud': df['Latitud'], 'Concentracion': concentraciones}
    df_mapa = pd.DataFrame(data)

    # Crear el mapa de calor usando Seaborn
    plt.figure(figsize=(10, 8))
    sns.kdeplot(x=df_mapa['Longitud'], y=df_mapa['Latitud'], weights=df_mapa['Concentracion'],
                cmap="viridis", fill=True, levels=100)

    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.title(titulo)
    plt.colorbar(label="Densidad de Concentración")
    plt.grid(True)
    plt.show()
