import time
import pandas as pd
import numpy as np

def generar_logs_tiempo_real():
    while True:
        log = {
            "timestamp": pd.Timestamp.now(),
            "cpu_usage": np.random.normal(50, 10),
            "memory_usage": np.random.normal(30, 5),
            "disk_io": np.random.normal(70, 15)
        }

        # Guardar en CSV (modo append)
        with open("real_time_logs.csv", "a") as f:
            f.write(f'{log["timestamp"]},{log["cpu_usage"]},{log["memory_usage"]},{log["disk_io"]}\n')

        print("Log generado:", log)
        time.sleep(1)  # Generar un log cada segundo

if __name__ == "__main__":
    with open("real_time_logs.csv", "w") as f:
        f.write("timestamp,cpu_usage,memory_usage,disk_io\n")
    generar_logs_tiempo_real()
