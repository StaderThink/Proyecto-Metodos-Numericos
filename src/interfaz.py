import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import numpy as np
import pandas as pd
from modelo_gaussiano import calcular_concentracion, calcular_sigma
from visualizacion import crear_grafico_2d, crear_grafico_3d, crear_grafico_dispersion, crear_mapa_calor
from preprocesamiento import cargar_datos, limpiar_datos, preparar_datos_ml
from entrenamiento_ml import entrenar_modelo
from datetime import datetime, timedelta  # Importa las clases datetime y timedelta

class InterfazGrafica:
    def __init__(self, master):
        self.master = master
        master.title("Simulador de Dispersión de Contaminantes")

        # Etiquetas y campos de entrada para los parámetros
        ttk.Label(master, text="Ruta del Archivo CSV:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.ruta_archivo_entry = ttk.Entry(master)
        self.ruta_archivo_entry.grid(row=0, column=1, padx=5, pady=5)
        self.ruta_archivo_entry.insert(0, "data/datos_simulados.csv")

        ttk.Label(master, text="Altura de la Fuente (H):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.h_entry = ttk.Entry(master)
        self.h_entry.grid(row=1, column=1, padx=5, pady=5)
        self.h_entry.insert(0, "50")  # Altura de la fuente por defecto

        ttk.Label(master, text="Estabilidad Atmosférica:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.estabilidad_combo = ttk.Combobox(master, values=['A', 'B', 'C', 'D', 'E', 'F'])
        self.estabilidad_combo.grid(row=2, column=1, padx=5, pady=5)
        self.estabilidad_combo.set('D')  # Estabilidad neutra por defecto

        # Botón para cargar, procesar y visualizar
        self.cargar_button = ttk.Button(master, text="Cargar, Procesar y Visualizar", command=self.cargar_procesar_visualizar)
        self.cargar_button.grid(row=3, column=0, columnspan=2, pady=10)

    def cargar_procesar_visualizar(self):
        try:
            # 1. Cargar y limpiar datos
            ruta_archivo = self.ruta_archivo_entry.get()
            df = cargar_datos(ruta_archivo)
            if df is None:
                return  # Sale si no se pudieron cargar los datos
            df = limpiar_datos(df)

            # 2. Preparar datos para ML y entrenar el modelo
            X, y = preparar_datos_ml(df)
            self.modelo_ml = entrenar_modelo(X, y)  # Guarda el modelo entrenado

            # 3. Simulación de la dispersión gaussiana (para el gráfico 2D y 3D)
            H = float(self.h_entry.get())
            estabilidad = self.estabilidad_combo.get()

            # Define un rango espacial (para el gráfico 2D y 3D)
            x_rango = np.linspace(100, 1000, 50)  # Distancia a lo largo del viento
            y_rango = np.linspace(-200, 200, 50)  # Distancia perpendicular al viento
            z = 0  # Altura al nivel del suelo

            # Usa datos promedio para la simulación (esto es solo un ejemplo)
            Q = df['Emisiones_Vehiculares'].mean() + df['Emisiones_Industriales'].mean()  # Tasa de emisión (suma de fuentes)
            u = df['Velocidad_Viento'].mean()  # Velocidad del viento

            # Calcula sigma_y y sigma_z
            sigma_y, sigma_z = calcular_sigma(x_rango, estabilidad)

            # Calcula la concentración para cada punto (para el gráfico 2D y 3D)
            concentraciones_2d = np.zeros((len(y_rango), len(x_rango)))
            for i, yi in enumerate(y_rango):
                for j, xi in enumerate(x_rango):
                    concentraciones_2d[i, j] = calcular_concentracion(Q, u, H, sigma_y[j], sigma_z[j], xi, yi, z)
            
            # 4.Creacion de datos futuros
            fecha_inicio = datetime(2024, 1, 2)  # Fecha de inicio de la predicción (un dia despues de los datos)
            num_meses = 3 # Numero de meses a predecir
            fechas_futuras = [fecha_inicio + timedelta(days=i*30) for i in range(num_meses)] #Creacion de datos de fecha

            # Simula datos meteorologicos y de emisiones futuros
            datos_futuros = []
            for fecha in fechas_futuras:
                # Simula datos meteorologicos (esto es solo un ejemplo)
                velocidad_viento = np.random.uniform(1,6)
                direccion_viento = np.random.uniform(0,360)
                temperatura = np.random.uniform(5,25)

                #Simula datos de emisiones (esto tambien es solo un ejemplo)
                emisiones_vehiculares = np.random.uniform(30,100)
                emisiones_industriales = np.random.uniform(20,50)

                datos_futuros.append([emisiones_vehiculares, emisiones_industriales, velocidad_viento, direccion_viento, temperatura])
            
            df_futuro = pd.DataFrame(datos_futuros, columns=['Emisiones_Vehiculares', 'Emisiones_Industriales', 'Velocidad_Viento', 'Direccion_Viento', 'Temperatura'])


            #5. Predicción con GPR
            X_pred = df[['Emisiones_Vehiculares', 'Emisiones_Industriales', 'Velocidad_Viento', 'Direccion_Viento', 'Temperatura']]
            y_pred, sigma = self.modelo_ml.predict(X_pred, return_std=True)

            X_futuro = df_futuro[['Emisiones_Vehiculares', 'Emisiones_Industriales', 'Velocidad_Viento', 'Direccion_Viento', 'Temperatura']]
            y_pred_futuro, sigma_futuro = self.modelo_ml.predict(X_futuro, return_std=True)


            # 6. Visualizar los resultados
            crear_grafico_2d(x_rango, y_rango, concentraciones_2d, titulo="Concentración de Contaminantes (2D)")
            crear_grafico_3d(x_rango, y_rango, concentraciones_2d, titulo="Concentración de Contaminantes (3D)")
            crear_grafico_dispersion(df, y_pred, titulo="Predicciones GPR con Dispersion Historico")
            crear_mapa_calor(df, y_pred, titulo="Mapa de Calor de Predicciones GPR Historico")

            crear_grafico_dispersion(df_futuro, y_pred_futuro, titulo="Predicciones GPR con Dispersion Futuro")
            crear_mapa_calor(df_futuro, y_pred_futuro, titulo="Mapa de Calor de Predicciones GPR Futuro")



            messagebox.showinfo("Info", "Cálculo y visualización completados.")

        except ValueError:
            messagebox.showerror("Error", "Por favor, introduce valores numéricos válidos.")
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error: {e}")

def main():
    root = tk.Tk()
    interfaz = InterfazGrafica(root)
    root.mainloop()

if __name__ == "__main__":
    main()