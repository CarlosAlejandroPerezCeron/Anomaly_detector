import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from preprocessing import procesar_logs
import numpy as np

# Obtener datos procesados
data, X_train = procesar_logs()

if X_train is not None:
    # Crear el modelo Autoencoder
    autoencoder = Sequential([
        Dense(32, activation='relu', input_shape=(3,)),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(16, activation='relu'),
        Dense(32, activation='relu'),
        Dense(3, activation='sigmoid')
    ])

    autoencoder.compile(optimizer='adam', loss='mse')

    # Entrenar el modelo
    autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

    # Guardar el modelo entrenado
    autoencoder.save("anomaly_detector_model.h5")
    print("Modelo guardado como 'anomaly_detector_model.h5'.")
