from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def entrenar_modelo(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-2, 1.0))

    modelo = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42) 

    modelo.fit(X_train, y_train)

    y_pred, sigma = modelo.predict(X_test, return_std=True)  

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"RMSE: {rmse}")
    print(f"R^2: {r2}")

    return modelo