import json
import math
import statistics
import time
from pathlib import Path
from urllib.request import Request, urlopen

URL = "http://127.0.0.1:8000/predict"
IMAGEN = Path(__file__).parent / "test_image.png"
LIMITE_P95_MS = 3000

contenido = IMAGEN.read_bytes()
boundary = "ML_EVA4_TEST"

cuerpo = (
    f"--{boundary}\r\n"
    'Content-Disposition: form-data; name="file"; filename="test_image.png"\r\n'
    "Content-Type: image/png\r\n\r\n"
).encode() + contenido + f"\r\n--{boundary}--\r\n".encode()

def predecir():
    solicitud = Request(
        URL,
        data=cuerpo,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}"
        },
        method="POST"
    )

    inicio = time.perf_counter()

    with urlopen(solicitud, timeout=30) as respuesta:
        datos = json.load(respuesta)

    total = (time.perf_counter() - inicio) * 1000

    assert datos["total_detecciones"] == len(datos["detecciones"])
    assert datos["tiempo_inferencia_ms"] >= 0

    return total, datos["tiempo_inferencia_ms"]

# Calentamiento
for _ in range(3):
    predecir()

tiempos_totales = []
tiempos_inferencia = []

for _ in range(10):
    total, inferencia = predecir()
    tiempos_totales.append(total)
    tiempos_inferencia.append(inferencia)

ordenados = sorted(tiempos_totales)
p95 = ordenados[math.ceil(0.95 * len(ordenados)) - 1]

reporte = {
    "modelo": "YOLOv8n-OBB ONNX",
    "ejecuciones": len(tiempos_totales),
    "inferencia_media_ms": round(
        statistics.mean(tiempos_inferencia), 2
    ),
    "respuesta_media_ms": round(
        statistics.mean(tiempos_totales), 2
    ),
    "respuesta_p95_ms": round(p95, 2),
    "limite_p95_ms": LIMITE_P95_MS
}

print(json.dumps(reporte, indent=2))

Path("latencia_ci.json").write_text(
    json.dumps(reporte, indent=2),
    encoding="utf-8"
)

assert p95 < LIMITE_P95_MS, (
    f"Latencia excesiva: p95={p95:.2f} ms"
)

print("Prueba de latencia aprobada")
