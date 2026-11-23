from pathlib import Path
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["estado"] == "activo"

def test_predict():
    imagen = Path("tests/test_image.png").read_bytes()

    response = client.post(
        "/predict",
        files={
            "file": ("prueba.png", imagen, "image/png")
        }
    )

    assert response.status_code == 200
    datos = response.json()
    assert "detecciones" in datos
    assert "tiempo_inferencia_ms" in datos
    assert datos["total_detecciones"] == len(datos["detecciones"])

def test_formato_invalido():
    response = client.post(
        "/predict",
        files={
            "file": ("archivo.txt", b"texto", "text/plain")
        }
    )
    assert response.status_code == 415

def test_imagen_corrupta():
    response = client.post(
        "/predict",
        files={
            "file": ("archivo.png", b"no es una imagen", "image/png")
        }
    )
    assert response.status_code == 400
