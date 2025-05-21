import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
from modelo_gaussiano import calcular_concentracion, calcular_sigma
from visualizacion import crear_grafico_2d, crear_grafico_3d, crear_grafico_dispersion, crear_mapa_calor
from preprocesamiento import cargar_datos, preparar_datos_ml
from entrenamiento_ml import entrenar_modelo
from datetime import datetime, timedelta

class InterfazGrafica:
    def __init__(self, master):
        self.master = master
        master.title("Simulador de Dispersión de Contaminantes")
        
        # Configurar estilo para botones
        style = ttk.Style()
        style.configure('TButton', font=('Helvetica', 10), padding=5)
        style.configure('Success.TButton', foreground='green', font=('Helvetica', 10, 'bold'))
        
        # Frame para la carga de archivos
        file_frame = ttk.LabelFrame(master, text="Cargar Datos", padding=(10, 5))
        file_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        
        # Botón para seleccionar archivo
        self.select_button = ttk.Button(
            file_frame, 
            text="Seleccionar Archivo CSV", 
            command=self.seleccionar_archivo
        )
        self.select_button.grid(row=0, column=0, padx=5, pady=5)
        
        # Etiqueta para mostrar el nombre del archivo seleccionado
        self.file_label = ttk.Label(file_frame, text="Ningún archivo seleccionado")
        self.file_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Barra de progreso
        self.progress = ttk.Progressbar(
            file_frame, 
            orient="horizontal", 
            length=200, 
            mode="determinate"
        )
        self.progress.grid(row=1, column=0, columnspan=2, pady=5, sticky="ew")
        self.progress.grid_remove()  # Ocultar inicialmente
        
        # Botón para procesar y visualizar
        self.cargar_button = ttk.Button(
            master, 
            text="Procesar y Visualizar", 
            command=self.cargar_procesar_visualizar,
            state="disabled"  # Deshabilitado hasta que se seleccione un archivo
        )
        self.cargar_button.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        
        # Configurar peso de columnas para que se expandan
        master.columnconfigure(0, weight=1)
        file_frame.columnconfigure(1, weight=1)
        
        # Variable para almacenar la ruta del archivo
        self.ruta_archivo = ""
    
    def seleccionar_archivo(self):
        """Abre un diálogo para seleccionar archivo y actualiza la interfaz"""
        filetypes = [("CSV files", "*.csv"), ("All files", "*.*")]
        filename = filedialog.askopenfilename(
            title="Seleccionar archivo CSV", 
            filetypes=filetypes
        )
        
        if filename:
            self.ruta_archivo = filename
            self.file_label.config(text=filename.split("/")[-1])
            self.cargar_button.config(state="enabled")
            
            # Cambiar estilo del botón para indicar éxito
            self.select_button.config(style='Success.TButton', text="✓ Archivo Seleccionado")
    
    def mostrar_progreso(self, visible=True):
        """Muestra u oculta la barra de progreso"""
        if visible:
            self.progress.grid()
            self.progress["value"] = 0
            self.master.update_idletasks()
        else:
            self.progress.grid_remove()
    
    def actualizar_progreso(self, valor, mensaje=None):
        """Actualiza la barra de progreso y opcionalmente muestra un mensaje"""
        self.progress["value"] = valor
        if mensaje:
            self.file_label.config(text=mensaje)
        self.master.update_idletasks()
    
    def cargar_procesar_visualizar(self):
        try:
            # Mostrar barra de progreso
            self.mostrar_progreso(True)
            self.actualizar_progreso(10, "Cargando datos...")
            
            # 1. Cargar y limpiar datos
            df = cargar_datos(self.ruta_archivo)
            if df is None:
                self.mostrar_progreso(False)
                return
            
            self.actualizar_progreso(30, "Preparando datos para ML...")
            
            # 2. Preparar datos para ML y entrenar el modelo
            X, y = preparar_datos_ml(df)
            self.actualizar_progreso(50, "Entrenando modelo...")
            self.modelo_ml = entrenar_modelo(X, y)
            
            self.actualizar_progreso(60, "Realizando simulación gaussiana...")
            
            # 3. Simulación gaussiana usando datos promedio
            x_rango = np.linspace(100, 1000, 50)
            y_rango = np.linspace(-200, 200, 50)
            z = 0

            Q = df['PM2.5'].mean() + df['PM10'].mean()
            u = df['Velocidad_Viento'].mean()
            H = df['Altura_Fuente'].mean()
            estabilidad = df['Estabilidad'].mode()[0]

            sigma_y, sigma_z = calcular_sigma(x_rango, estabilidad)

            concentraciones_2d = np.zeros((len(y_rango), len(x_rango)))
            for i, yi in enumerate(y_rango):
                for j, xi in enumerate(x_rango):
                    concentraciones_2d[i, j] = calcular_concentracion(Q, u, H, sigma_y[j], sigma_z[j], xi, yi, z)
            
            self.actualizar_progreso(70, "Simulando datos futuros...")
            
            # 4. Simulación de datos futuros
            fecha_inicio = datetime(2024, 1, 2)
            num_meses = 3
            fechas_futuras = [fecha_inicio + timedelta(days=i*30) for i in range(num_meses)]

            datos_futuros = []
            for fecha in fechas_futuras:
                velocidad_viento = np.random.uniform(1, 6)
                direccion_viento = np.random.uniform(0, 360)
                temperatura = np.random.uniform(5, 25)
                pm25 = np.random.uniform(10, 80)
                pm10 = np.random.uniform(20, 120)
                co2 = np.random.uniform(350, 500)
                estabilidad = np.random.choice(['A', 'B', 'C', 'D', 'E', 'F'])
                altura_fuente = np.random.uniform(10, 80)
                latitud = 4.65 + np.random.uniform(-0.01, 0.01)
                longitud = -74.05 + np.random.uniform(-0.01, 0.01)

                datos_futuros.append([
                    fecha.strftime("%Y-%m-%d"),
                    fecha.strftime("%H:%M"),
                    latitud,
                    longitud,
                    pm25,
                    pm10,
                    co2,
                    velocidad_viento,
                    direccion_viento,
                    temperatura,
                    estabilidad,
                    altura_fuente
                ])

            df_futuro = pd.DataFrame(datos_futuros, columns=[
                'Fecha', 'Hora', 'Latitud', 'Longitud', 'PM2.5', 'PM10', 'CO2',
                'Velocidad_Viento', 'Direccion_Viento', 'Temperatura',
                'Estabilidad', 'Altura_Fuente'
            ])
            
            self.actualizar_progreso(80, "Realizando predicciones...")
            
            # 5. Predicciones utilizando las mismas columnas que en entrenamiento
            y_pred, sigma = self.modelo_ml.predict(X, return_std=True)
            X_futuro = df_futuro[X.columns]  # Asegurar mismo orden y nombres de columnas
            y_pred_futuro, sigma_futuro = self.modelo_ml.predict(X_futuro, return_std=True)
            
            self.actualizar_progreso(90, "Generando visualizaciones...")
            
            # 6. Visualización
            crear_grafico_2d(x_rango, y_rango, concentraciones_2d, titulo="Concentración de Contaminantes (2D)")
            crear_grafico_3d(x_rango, y_rango, concentraciones_2d, titulo="Concentración de Contaminantes (3D)")
            crear_grafico_dispersion(df, y_pred, titulo="Predicciones GPR con Dispersion Histórico")
            crear_mapa_calor(df, y_pred, titulo="Mapa de Calor de Predicciones GPR Histórico")
            crear_grafico_dispersion(df_futuro, y_pred_futuro, titulo="Predicciones GPR con Dispersion Futuro")
            crear_mapa_calor(df_futuro, y_pred_futuro, titulo="Mapa de Calor de Predicciones GPR Futuro")
            
            self.actualizar_progreso(100, "¡Proceso completado!")
            
            # Mostrar mensaje de éxito y ocultar progreso después de un breve retraso
            self.master.after(1500, lambda: [
                messagebox.showinfo("Éxito", "Procesamiento y visualización completados con éxito."),
                self.mostrar_progreso(False),
                self.file_label.config(text=f"Archivo procesado: {self.ruta_archivo.split('/')[-1]}")
            ])

        except Exception as e:
            self.mostrar_progreso(False)
            messagebox.showerror("Error", f"Ocurrió un error: {str(e)}")