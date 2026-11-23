# Resultados de optimizacion del modelo

## Modelo
YOLOv8n-OBB para deteccion de manzanas, naranjas y platanos.

## Metodologia
- Entorno: Windows, CPU.
- Modelo original: PyTorch.
- Primera optimizacion: exportacion a ONNX.
- Segunda optimizacion: cuantizacion estatica QDQ de operaciones Conv.
- Calibracion INT8: 15 imagenes.
- Benchmark: 5 ejecuciones de calentamiento y 30 mediciones.
- Imagen evaluada: cf376922-manzana1.png.

## Resultados

| Metrica | PyTorch | ONNX | ONNX INT8 |
|---|---:|---:|---:|
| Tamano MB | 6.11 | 11.92 | 3.26 |
| Inferencia media ms | 31.48 | 27.68 | 27.23 |
| Tiempo total medio ms | 38.56 | 34.99 | 35.24 |
| RAM final del proceso MB | 376.71 | 400.53 | 402.79 |
| Detecciones | 1 | 1 | 1 |

## Interpretacion

ONNX redujo el tiempo de inferencia aproximadamente 12.1%
respecto de PyTorch.

La cuantizacion INT8 redujo el archivo ONNX aproximadamente
72.7%, con una mejora marginal en inferencia, pero sin
reduccion del tiempo total medio ni de la RAM final observada.

Las mediciones corresponden a ejecuciones de referencia.
La RAM representa el proceso completo y no exclusivamente
el modelo. La calidad de deteccion solo se verifico
puntualmente; no se demuestra equivalencia de precision.

## Decision

Mantener best.onnx como modelo principal del servicio.
Conservar best_int8.onnx como alternativa experimental.
