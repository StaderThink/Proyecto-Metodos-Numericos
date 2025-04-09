import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import numpy as np
import pandas as pd
from modelo_gaussiano import calcular_concentracion, calcular_sigma
from visualizacion import crear_grafico_2d, crear_grafico_3d, crear_grafico_dispersion, crear_mapa_calor  # Importa la nueva función de visualización
from preprocesamiento import cargar_datos, limpiar_datos, preparar_datos_ml
from entrenamiento_ml import entrenar_modelo

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

            # 4. Simulación de la dispersión gaussiana (para el gráfico de dispersión y mapa de calor)
            concentraciones_dispersion = []
            for index, row in df.iterrows():
                # Obtiene los datos de la fila
                Q = row['Emisiones_Vehiculares'] + row['Emisiones_Industriales']  # Tasa de emisión (suma de fuentes)
                u = row['Velocidad_Viento']  # Velocidad del viento
                x = 100  # Distancia a lo largo del viento (puedes ajustarla)
                y_val = 0  # Distancia perpendicular al viento (puedes ajustarla)
                z = 0  # Altura al nivel del suelo

                # Calcula sigma_y y sigma_z
                sigma_y_disp, sigma_z_disp = calcular_sigma(x, estabilidad)

                # Calcula la concentración
                C = calcular_concentracion(Q, u, H, sigma_y_disp, sigma_z_disp, x, y_val, z)  # Usamos sigma_y[0] y sigma_z[0] porque x es un valor único
                concentraciones_dispersion.append(C)

            # 5. Visualizar los resultados
            crear_grafico_2d(x_rango, y_rango, concentraciones_2d, titulo="Concentración de Contaminantes (2D)")
            crear_grafico_3d(x_rango, y_rango, concentraciones_2d, titulo="Concentración de Contaminantes (3D)")
            crear_grafico_dispersion(df, concentraciones_dispersion, titulo="Dispersión de Contaminantes")
            crear_mapa_calor(df, concentraciones_dispersion, titulo="Mapa de Calor de Concentración de Contaminantes")


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
