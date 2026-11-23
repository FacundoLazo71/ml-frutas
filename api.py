from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from pathlib import Path
from threading import Lock

app = FastAPI(
    title="API de Deteccion de Frutas",
    description="Modelo YOLOv8n-OBB optimizado con ONNX",
    version="1.0.0"
)

RUTA_MODELO = Path(__file__).parent / "modelo" / "best.onnx"
modelo = YOLO(str(RUTA_MODELO), task="obb")
modelo_lock = Lock()

@app.get("/health")
def health():
    return {
        "estado": "activo",
        "modelo": "YOLOv8n-OBB ONNX"
    }

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    if file.content_type not in [
        "image/jpeg", "image/png", "image/webp"
    ]:
        raise HTTPException(
            status_code=415,
            detail="Formato no compatible. Usa JPG, PNG o WEBP."
        )

    contenido = file.file.read(10 * 1024 * 1024 + 1)

    if len(contenido) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="La imagen supera el limite de 10 MB."
        )

    try:
        with Image.open(BytesIO(contenido)) as img:
            imagen = img.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="El archivo no contiene una imagen valida."
        )

    with modelo_lock:
        resultado = modelo.predict(
            imagen,
            imgsz=640,
            conf=0.25,
            device="cpu",
            verbose=False
        )[0]

    detecciones = []

    if resultado.obb is not None:
        clases = resultado.obb.cls.tolist()
        confianzas = resultado.obb.conf.tolist()
        coordenadas = resultado.obb.xyxyxyxy.tolist()

        for clase, confianza, puntos in zip(
            clases, confianzas, coordenadas
        ):
            detecciones.append({
                "fruta": resultado.names[int(clase)],
                "confianza": round(float(confianza), 4),
                "coordenadas": puntos
            })

    return {
        "archivo": file.filename,
        "total_detecciones": len(detecciones),
        "tiempo_inferencia_ms": round(
            resultado.speed["inference"], 2
        ),
        "detecciones": detecciones
    }
