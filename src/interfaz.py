import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
from modelo_gaussiano import calcular_concentracion, calcular_sigma
from visualizacion import crear_grafico_2d, crear_grafico_3d, crear_grafico_dispersion, crear_mapa_calor
from preprocesamiento import cargar_datos, preparar_datos_ml
from entrenamiento_ml import entrenar_modelo
from datetime import datetime, timedelta
from scipy.stats import multivariate_normal

class InterfazGrafica:
    def __init__(self, master):
        self.master = master
        master.title("Simulador de Dispersión de Contaminantes")
        
        style = ttk.Style()
        style.configure('TButton', font=('Helvetica', 10), padding=5)
        style.configure('Success.TButton', foreground='green', font=('Helvetica', 10, 'bold'))
        
        file_frame = ttk.LabelFrame(master, text="Cargar Datos", padding=(10, 5))
        file_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        
        self.select_button = ttk.Button(
            file_frame, 
            text="Seleccionar Archivo CSV", 
            command=self.seleccionar_archivo
        )
        self.select_button.grid(row=0, column=0, padx=5, pady=5)
        
        self.file_label = ttk.Label(file_frame, text="Ningún archivo seleccionado")
        self.file_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        self.progress = ttk.Progressbar(
            file_frame, 
            orient="horizontal", 
            length=200, 
            mode="determinate"
        )
        self.progress.grid(row=1, column=0, columnspan=2, pady=5, sticky="ew")
        self.progress.grid_remove()  
        
        self.cargar_button = ttk.Button(
            master, 
            text="Procesar y Visualizar", 
            command=self.cargar_procesar_visualizar,
            state="disabled"  
        )
        self.cargar_button.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        
        master.columnconfigure(0, weight=1)
        file_frame.columnconfigure(1, weight=1)
        
        self.ruta_archivo = ""
    
    def seleccionar_archivo(self):
        filetypes = [("CSV files", "*.csv"), ("All files", "*.*")]
        filename = filedialog.askopenfilename(
            title="Seleccionar archivo CSV", 
            filetypes=filetypes
        )
        
        if filename:
            self.ruta_archivo = filename
            self.file_label.config(text=filename.split("/")[-1])
            self.cargar_button.config(state="enabled")
            
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
        self.progress["value"] = valor
        if mensaje:
            self.file_label.config(text=mensaje)
        self.master.update_idletasks()
    
    def generar_datos_futuros(self, df, fechas_futuras):
        # Calcular estadísticas clave
        stats = {
            'PM2.5_mean': df['PM2.5'].mean(),
            'PM2.5_std': df['PM2.5'].std(),
            'PM10_mean': df['PM10'].mean(),
            'PM10_std': df['PM10'].std(),
            'CO2_mean': df['CO2'].mean(),
            'CO2_std': df['CO2'].std(),
            'lat_mean': df['Latitud'].mean(),
            'lon_mean': df['Longitud'].mean(),
            'geo_cov': np.cov(df['Latitud'], df['Longitud'])
        }
        
        geo_dist = multivariate_normal(
            mean=[stats['lat_mean'], stats['lon_mean']],
            cov=stats['geo_cov']
        )
        
        datos_futuros = []
        for i, fecha in enumerate(fechas_futuras):
            lat, lon = geo_dist.rvs()
            lat = np.clip(lat, stats['lat_mean']-0.05, stats['lat_mean']+0.05)
            lon = np.clip(lon, stats['lon_mean']-0.05, stats['lon_mean']+0.05)
            
            base = np.random.normal(0, 1)
            pm25 = stats['PM2.5_mean'] + base * stats['PM2.5_std']
            pm10 = stats['PM10_mean'] + (base * 0.8 + np.random.normal(0, 0.2)) * stats['PM10_std']
            co2 = stats['CO2_mean'] + (base * 0.5 + np.random.normal(0, 0.5)) * stats['CO2_std']
            
            hora = fecha.hour
            temperatura = 20 + 5 * np.sin(2*np.pi*hora/24) + np.random.normal(0, 2)
            velocidad_viento = np.random.weibull(2) * 3
            
            if velocidad_viento > 4:
                estabilidad = np.random.choice(['A', 'B', 'C'], p=[0.2, 0.5, 0.3])
            else:
                estabilidad = np.random.choice(['D', 'E', 'F'], p=[0.3, 0.5, 0.2])
            
            altura_fuente = np.random.normal(50, 15)
            
            datos_futuros.append([
                fecha.strftime("%Y-%m-%d"),
                fecha.strftime("%H:%M"),
                float(lat),
                float(lon),
                max(10, pm25),
                max(20, pm10),
                max(350, co2),
                float(velocidad_viento),
                float(np.random.uniform(0, 360)),
                float(temperatura),
                estabilidad,
                float(max(10, min(100, altura_fuente)))
            ])
        
        return datos_futuros
    
    def cargar_procesar_visualizar(self):
        try:
            self.mostrar_progreso(True)
            self.actualizar_progreso(10, "Cargando datos...")
            
            df = cargar_datos(self.ruta_archivo)
            if df is None:
                self.mostrar_progreso(False)
                messagebox.showerror("Error", "No se pudieron cargar los datos del archivo")
                return
            
            self.actualizar_progreso(30, "Preparando datos para ML...")
            
            X, y = preparar_datos_ml(df)
            self.actualizar_progreso(50, "Entrenando modelo...")
            self.modelo_ml = entrenar_modelo(X, y)
            
            self.actualizar_progreso(60, "Realizando simulación gaussiana...")
            
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
            
            fecha_inicio = datetime.now()  
            num_meses = 3
            fechas_futuras = [fecha_inicio + timedelta(days=i*30) for i in range(num_meses)]

            datos_futuros = self.generar_datos_futuros(df, fechas_futuras)

            df_futuro = pd.DataFrame(datos_futuros, columns=[
                'Fecha', 'Hora', 'Latitud', 'Longitud', 'PM2.5', 'PM10', 'CO2',
                'Velocidad_Viento', 'Direccion_Viento', 'Temperatura',
                'Estabilidad', 'Altura_Fuente'
            ])
            
            self.actualizar_progreso(80, "Realizando predicciones...")
            
            y_pred, sigma = self.modelo_ml.predict(X, return_std=True)
            
            X_futuro = df_futuro[X.columns]
            y_pred_futuro, sigma_futuro = self.modelo_ml.predict(X_futuro, return_std=True)
            
            self.actualizar_progreso(90, "Generando visualizaciones...")
            
            crear_grafico_2d(x_rango, y_rango, concentraciones_2d, titulo="Concentración de Contaminantes (2D)")
            crear_grafico_3d(x_rango, y_rango, concentraciones_2d, titulo="Concentración de Contaminantes (3D)")
            crear_grafico_dispersion(df, y_pred, titulo="Predicciones GPR con Dispersion Histórico")
            crear_mapa_calor(df, y_pred, titulo="Mapa de Calor de Predicciones GPR Histórico")
            crear_grafico_dispersion(df_futuro, y_pred_futuro, titulo="Predicciones GPR con Dispersion Futuro")
            crear_mapa_calor(df_futuro, y_pred_futuro, titulo="Mapa de Calor de Predicciones GPR Futuro")
            
            self.actualizar_progreso(100, "¡Proceso completado!")
            
            self.master.after(1500, lambda: [
                messagebox.showinfo("Éxito", "Procesamiento y visualización completados con éxito."),
                self.mostrar_progreso(False),
                self.file_label.config(text=f"Archivo procesado: {self.ruta_archivo.split('/')[-1]}")
            ])

        except Exception as e:
            self.mostrar_progreso(False)
            messagebox.showerror("Error", f"Ocurrió un error: {str(e)}")
            import traceback
            traceback.print_exc() 