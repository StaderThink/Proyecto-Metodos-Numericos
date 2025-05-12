import pandas as pd

def cargar_datos(ruta_archivo):
    try:
        return pd.read_csv(ruta_archivo)
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        return None

def preparar_datos_ml(df):
    X = df[['Emisiones_Vehiculares', 'Emisiones_Industriales', 'Velocidad_Viento', 'Direccion_Viento', 'Temperatura']]
    y = df[['ICA_PM10']]  # o 'ICA_PM25' si se prefiere
    return X, y
