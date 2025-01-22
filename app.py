import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import time

st.title("Dashboard de Detección de Anomalías en Tiempo Real")

# Cargar el modelo entrenado
model = load_model("anomaly_detector_model.h5")

# Contenedor para los datos en tiempo real
placeholder = st.empty()

while True:
    try:
        data = pd.read_csv("real_time_logs.csv")

        if len(data) > 0:
            # Normalizar los datos
            scaler = MinMaxScaler()
            features = ["cpu_usage", "memory_usage", "disk_io"]
            data_normalized = scaler.fit_transform(data[features])

            # Predicciones del modelo
            reconstructions = model.predict(data_normalized)
            reconstruction_errors = np.mean(np.square(data_normalized - reconstructions), axis=1)

            # Definir umbral de anomalías
            threshold = reconstruction_errors.mean() + 3 * reconstruction_errors.std()
            data["anomaly"] = reconstruction_errors > threshold

            with placeholder.container():
                st.dataframe(data.tail(10))

                # Graficar anomalías
                fig, ax = plt.subplots()
                ax.plot(data["cpu_usage"], label="Uso de CPU")
                ax.scatter(data.index[data["anomaly"]], data["cpu_usage"][data["anomaly"]], color="red", label="Anomalías")
                ax.legend()
                st.pyplot(fig)

            time.sleep(2)
    except Exception as e:
        st.warning(f"Error: {e}")
