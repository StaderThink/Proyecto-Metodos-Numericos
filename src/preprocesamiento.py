import pandas as pd

def cargar_datos(ruta):
    try:
        df = pd.read_csv(ruta)

        df.dropna(inplace=True)

        columnas_numericas = ['PM2.5', 'PM10', 'CO2', 'Velocidad_Viento', 'Direccion_Viento', 'Temperatura', 'Altura_Fuente']
        for col in columnas_numericas:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df[df['Estabilidad'].isin(['A', 'B', 'C', 'D', 'E', 'F'])]

        df.dropna(inplace=True)

        return df
    except Exception as e:
        print(f"Error cargando datos: {e}")
        return None

def preparar_datos_ml(df):
    X = df[['PM10', 'PM2.5', 'CO2', 'Velocidad_Viento', 'Direccion_Viento', 'Temperatura']]
    y = df['PM10']  
    return X, y
