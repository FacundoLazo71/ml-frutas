from pathlib import Path
import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static
)

ORIGINAL = "modelo/best.onnx"
CUANTIZADO = "modelo/best_int8.onnx"

sesion = ort.InferenceSession(
    ORIGINAL,
    providers=["CPUExecutionProvider"]
)

entrada = sesion.get_inputs()[0]
nombre_entrada = entrada.name
forma = entrada.shape

alto = forma[2] if isinstance(forma[2], int) else 640
ancho = forma[3] if isinstance(forma[3], int) else 640

def preparar_imagen(ruta):
    imagen = cv2.imread(str(ruta))

    if imagen is None:
        raise ValueError(f"No se puede leer: {ruta}")

    h, w = imagen.shape[:2]
    escala = min(alto / h, ancho / w)

    nuevo_w = round(w * escala)
    nuevo_h = round(h * escala)

    imagen = cv2.resize(
        imagen, (nuevo_w, nuevo_h),
        interpolation=cv2.INTER_LINEAR
    )

    dw = ancho - nuevo_w
    dh = alto - nuevo_h

    izquierda = round(dw / 2 - 0.1)
    derecha = round(dw / 2 + 0.1)
    arriba = round(dh / 2 - 0.1)
    abajo = round(dh / 2 + 0.1)

    imagen = cv2.copyMakeBorder(
        imagen, arriba, abajo,
        izquierda, derecha,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114)
    )

    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    imagen = imagen.transpose(2, 0, 1)
    imagen = np.ascontiguousarray(imagen)

    return imagen[None].astype(np.float32) / 255.0


class LectorCalibracion(CalibrationDataReader):
    def __init__(self, archivos):
        self.archivos = archivos
        self.indice = 0

    def get_next(self):
        if self.indice >= len(self.archivos):
            return None

        ruta = self.archivos[self.indice]
        self.indice += 1

        return {
            nombre_entrada: preparar_imagen(ruta)
        }

    def rewind(self):
        self.indice = 0


carpeta = Path("datos/images")
extensiones = {".jpg", ".jpeg", ".png", ".webp"}

archivos = sorted(
    ruta for ruta in carpeta.rglob("*")
    if ruta.suffix.lower() in extensiones
)[:15]

if not archivos:
    raise RuntimeError("No se encontraron imagenes de calibracion")

print("Modelo original:", ORIGINAL)
print("Entrada:", nombre_entrada, forma)
print("Imagenes de calibracion:", len(archivos))
print("Iniciando cuantizacion...")

lector = LectorCalibracion(archivos)

quantize_static(
    model_input=ORIGINAL,
    model_output=CUANTIZADO,
    calibration_data_reader=lector,
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8,
    op_types_to_quantize=["Conv"],
    per_channel=False
)

# Conservar los metadatos del modelo original
original = onnx.load(ORIGINAL)
cuantizado = onnx.load(CUANTIZADO)

claves = {p.key for p in cuantizado.metadata_props}

for propiedad in original.metadata_props:
    if propiedad.key not in claves:
        nueva = cuantizado.metadata_props.add()
        nueva.key = propiedad.key
        nueva.value = propiedad.value

onnx.save(cuantizado, CUANTIZADO)
onnx.checker.check_model(CUANTIZADO)

print("Modelo cuantizado generado:", CUANTIZADO)
print("Modelo ONNX valido: True")

# Verificar que ONNX Runtime puede ejecutarlo
prueba = preparar_imagen(archivos[0])

sesion_int8 = ort.InferenceSession(
    CUANTIZADO,
    providers=["CPUExecutionProvider"]
)

salidas = sesion_int8.run(
    None, {nombre_entrada: prueba}
)

print("Inferencia INT8 ejecutada correctamente")
print("Forma de salida:", [s.shape for s in salidas])
