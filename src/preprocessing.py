import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def procesar_logs():
    try:
        # Leer el archivo de logs en tiempo real
        data = pd.read_csv("real_time_logs.csv")

        # Seleccionar características
        features = ["cpu_usage", "memory_usage", "disk_io"]
        scaler = MinMaxScaler()
        data_normalized = scaler.fit_transform(data[features])

        return data, data_normalized
    except Exception as e:
        print("Error procesando los logs:", e)
        return None, None
