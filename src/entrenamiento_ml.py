from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def entrenar_modelo(X, y):
    """Entrena un modelo de aprendizaje automático (Random Forest en este caso)."""
    # Divide los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crea un modelo Random Forest
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)

    # Entrena el modelo
    modelo.fit(X_train, y_train)

    # Evalúa el modelo
    y_pred = modelo.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"RMSE: {rmse}")
    print(f"R^2: {r2}")

    return modelo