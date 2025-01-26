import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

from preprocessing import procesar_logs

# Obtener datos procesados
data, X_train = procesar_logs()

if X_train is not None:
    # Crear el modelo autoencoder
    autoencoder = Sequential([
        Dense(32, activation='relu', input_shape=(3,)),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(16, activation='relu'),
        Dense(32, activation='relu'),
        Dense(3, activation='sigmoid')
    ])

    # Compilar con 'mean_squared_error'
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    # Entrenar el modelo
    history = autoencoder.fit(
        X_train, X_train, 
        epochs=50, 
        batch_size=32, 
        validation_split=0.2, 
        verbose=1
    )

    # Guardar el modelo
    autoencoder.save("anomaly_detector_model.h5")

    # Graficar la pérdida
    plt.plot(history.history['loss'], label='Entrenamiento')
    plt.plot(history.history['val_loss'], label='Validación')
    plt.title("Pérdida del Modelo")
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida")
    plt.legend()
    plt.show()

    print("Modelo entrenado y guardado como 'anomaly_detector_model.h5'.")
else:
    print("No se pudo entrenar el modelo porque no hay datos en X_train.")
