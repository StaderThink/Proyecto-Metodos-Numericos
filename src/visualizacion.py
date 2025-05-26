import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
import matplotlib as mpl

def crear_grafico_2d(x, y, concentraciones, titulo="Concentración de Contaminantes (2D)"):

    plt.figure(figsize=(10, 6))
    plt.contourf(x, y, concentraciones, levels=50, cmap='viridis')
    plt.colorbar(label="Concentración (g/m^3)")
    plt.xlabel("Distancia a lo largo del viento (m)")
    plt.ylabel("Distancia perpendicular al viento (m)")
    plt.title(titulo)
    plt.show()

def crear_grafico_3d(x, y, concentraciones, titulo="Concentración de Contaminantes (3D)"):

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

    plt.figure(figsize=(10, 6))
    plt.scatter(df['Longitud'], df['Latitud'], c=concentraciones, cmap='viridis', s=50)
    plt.colorbar(label="Concentración (g/m^3)")
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.title(titulo)
    plt.grid(True)
    plt.show()

def crear_mapa_calor(df, concentraciones, titulo="Mapa de Calor de Concentración de Contaminantes"):

    df_mapa = pd.DataFrame({
        'Longitud': df['Longitud'],
        'Latitud': df['Latitud'],
        'Concentracion': np.array(concentraciones).astype(float)  
    })
    
    df_mapa = df_mapa[np.isfinite(df_mapa['Concentracion'])]
    df_mapa = df_mapa[df_mapa['Concentracion'] >= 0]  
    
    if len(df_mapa) < 3:  
        raise ValueError("Insuficientes datos válidos para generar el mapa de calor")
    
    conc_min = df_mapa['Concentracion'].min()
    conc_max = df_mapa['Concentracion'].max()
    df_mapa['Concentracion_norm'] = (df_mapa['Concentracion'] - conc_min) / (conc_max - conc_min + 1e-10)
    
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    try:
        sns.kdeplot(
            x=df_mapa['Longitud'],
            y=df_mapa['Latitud'],
            weights=df_mapa['Concentracion_norm'],
            cmap='viridis',
            fill=True,
            levels=np.linspace(0, 1, 20), 
            thresh=0.02,
            ax=ax
        )
    except Exception:
        sns.kdeplot(
            x=df_mapa['Longitud'],
            y=df_mapa['Latitud'],
            cmap='viridis',
            fill=True,
            levels=20,
            thresh=0.02,
            ax=ax
        )
    
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_title(titulo)
    plt.grid(True)
    
    norm = mpl.colors.Normalize(vmin=conc_min, vmax=conc_max)
    sm = cm.ScalarMappable(norm=norm, cmap='viridis')
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Concentración (g/m³)")
    
    plt.tight_layout()
    plt.show()
