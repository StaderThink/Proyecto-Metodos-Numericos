import pandas as pd

def cargar_datos(ruta_archivo):
    """Carga datos desde un archivo CSV."""
    try:
        df = pd.read_csv(ruta_archivo)
        return df
    except FileNotFoundError:
        print(f"Error: El archivo '{ruta_archivo}' no se encuentra.")
        return None
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        return None

def limpiar_datos(df):
    """Limpia los datos (manejo de valores faltantes, etc.)."""
    df_limpio = df.dropna()
    return df_limpio

def preparar_datos_ml(df):
    """Prepara los datos para el aprendizaje automático (selección de características, etc.)."""
    # Selecciona las características que usarás para predecir la calidad del aire
    features = ['Emisiones_Vehiculares', 'Emisiones_Industriales', 'Velocidad_Viento', 'Direccion_Viento', 'Temperatura']
    X = df[features]
    # Selecciona la variable objetivo (ICA_PM10 o ICA_PM25)
    y = df['ICA_PM10']  # Puedes cambiar a 'ICA_PM25' si lo prefieres
    return X, y