import csv
import numpy as np
import matplotlib.pyplot as plt

# Cargar datos desde el archivo CSV
data = []
with open('polynomial-regression.csv', 'r') as file:
    csv_reader = csv.reader(file)
    headers = next(csv_reader)  # Ignorar la primera fila (encabezados)
    for row in csv_reader:
        data.append([float(row[0]), float(row[1])])

data = np.array(data)
  
# Separar las características (X) y la variable objetivo (y)
X = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)

# Grado del polinomio
degree = 2

# Inicializar variables para almacenar los errores
errors = []

# Implementación de leave-one-out cross-validation
for i in range(len(X)):
    # Separar un punto para validación
    X_val = X[i:i+1]
    y_val = y[i:i+1]

    # Resto de los puntos para entrenamiento
    X_train = np.concatenate([X[:i], X[i+1:]])
    y_train = np.concatenate([y[:i], y[i+1:]])

    # Crear matriz de características polinomiales
    X_poly = np.c_[np.ones((len(X_train), 1)), X_train, X_train**degree]

    # Calcular coeficientes utilizando la fórmula cerrada de la regresión lineal
    theta = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y_train

    # Predecir el punto de validación
    X_val_poly = np.c_[1, X_val, X_val**degree]
    y_pred = X_val_poly @ theta

    # Calcular el error absoluto del modelo
    error = np.abs(y_val - y_pred)
    errors.append(error)

# Calcular el error absoluto medio (MAE)
mae = np.mean(errors)

# Visualización del modelo
X_plot = np.linspace(min(X), max(X), 100).reshape(-1, 1)
X_plot_poly = np.c_[np.ones((100, 1)), X_plot, X_plot**degree]
y_plot = X_plot_poly @ theta

plt.scatter(X, y, label='Datos')
plt.plot(X_plot, y_plot, 'r-', label='Modelo de regresión polinomial')
plt.xlabel('X araba_fiyat')
plt.ylabel('y araba_fiyat')
plt.title('Regresión Polinomial Simple\nError cuadratico medio: ' + str(mae))
plt.legend()
plt.show()