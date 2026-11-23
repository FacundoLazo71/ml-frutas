from ultralytics import YOLO
import time
import statistics
import sys
import os
import psutil

ruta_modelo = sys.argv[1]
imagen = "datos/images/cf376922-manzana1.png"

proceso = psutil.Process()
memoria_inicial = proceso.memory_info().rss / 1048576

modelo = YOLO(ruta_modelo, task="obb")

def predecir():
    return modelo.predict(
        imagen,
        imgsz=640,
        conf=0.25,
        device="cpu",
        verbose=False
    )[0]

# Calentamiento
for _ in range(5):
    predecir()

tiempos_inferencia = []
tiempos_totales = []

for _ in range(30):
    inicio = time.perf_counter()
    resultado = predecir()
    duracion = (time.perf_counter() - inicio) * 1000

    tiempos_totales.append(duracion)
    tiempos_inferencia.append(resultado.speed["inference"])

memoria_final = proceso.memory_info().rss / 1048576

print("\nRESULTADOS")
print("Modelo:", ruta_modelo)
print("Tamano del archivo (MB):",
      round(os.path.getsize(ruta_modelo) / 1048576, 2))
print("Detecciones:", len(resultado.obb))
print("Inferencia media (ms):",
      round(statistics.mean(tiempos_inferencia), 2))
print("Tiempo total medio (ms):",
      round(statistics.mean(tiempos_totales), 2))
print("Mediana total (ms):",
      round(statistics.median(tiempos_totales), 2))
print("RAM del proceso al finalizar (MB):",
      round(memoria_final, 2))
print("Variacion RAM desde el inicio (MB):",
      round(memoria_final - memoria_inicial, 2))
