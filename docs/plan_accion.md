# Plan de implementacion del modelo Deep Learning

## 1. Objetivo

Implementar el modelo YOLOv8n-OBB de deteccion de frutas
en un entorno local, optimizando su rendimiento y
habilitando su uso mediante una API REST.

La solucion contempla exportacion, optimizacion,
empaquetado, pruebas automatizadas, versionado
y despliegue local.

## 2. Fases de implementacion

### Fase 1. Preparacion del modelo

Actividades:
- Recuperar el modelo entrenado best.pt.
- Verificar su funcionamiento fuera de Google Colab.
- Organizar el dataset y las dependencias de Python.

Entregable:
Modelo original funcional y entorno local preparado.

Validacion:
Prediccion ejecutada correctamente sobre una imagen.

### Fase 2. Exportacion a ONNX

Actividades:
- Exportar YOLOv8n-OBB desde PyTorch a ONNX.
- Verificar la integridad del archivo exportado.
- Comprobar la ejecucion con ONNX Runtime.

Entregable:
Archivo best.onnx funcional.

Validacion:
Inferencia realizada correctamente con ONNX.

### Fase 3. Optimizacion de recursos

Actividades:
- Medir el rendimiento del modelo PyTorch.
- Comparar la version original con ONNX.
- Aplicar cuantizacion estatica INT8.
- Evaluar latencia, memoria y tamano de los archivos.
- Seleccionar el modelo para el servicio.

Entregable:
Comparacion tecnica de las tres versiones.

Validacion:
30 mediciones por modelo, despues de 5 ejecuciones
de calentamiento.

Decision:
Mantener best.onnx en el servicio principal.
Conservar INT8 como alternativa experimental.

### Fase 4. Implementacion de API

Actividades:
- Crear el servicio REST mediante FastAPI.
- Implementar el endpoint GET /health.
- Implementar el endpoint POST /predict.
- Validar imagenes y controlar errores.
- Devolver predicciones mediante JSON.

Entregable:
API funcional conectada al modelo ONNX.

Validacion:
Prediccion exitosa desde Swagger y pruebas pytest.

### Fase 5. Empaquetado Docker

Actividades:
- Crear Dockerfile y requirements.txt.
- Configurar las dependencias del servicio.
- Incorporar el modelo ONNX.
- Construir y ejecutar el contenedor.

Entregable:
Imagen Docker frutas-eva4.

Validacion:
API accesible desde localhost:8000/docs
con el contenedor en ejecucion.

### Fase 6. CI/CD y MLOps

Actividades:
- Almacenar el codigo mediante Git.
- Versionar dataset y pesos originales con DVC.
- Configurar pruebas automaticas en GitHub Actions.
- Integrar verificaciones de inferencia y latencia.
- Publicar la imagen en GitHub Container Registry.

Entregable:
Pipeline automatizado de integracion y entrega,
con versionado de datos y modelos.

Validacion:
Ejecuciones exitosas en GitHub Actions
e imagen publicada en GHCR.

Nota:
La publicacion en GHCR constituye entrega continua.
No implica despliegue automatico en un servidor.

### Fase 7. Despliegue y validacion final

Actividades:
- Ejecutar la API dentro de Docker localmente.
- Verificar disponibilidad y endpoints.
- Probar solicitudes validas e invalidas.
- Registrar resultados y evidencias.

Entregable:
Servicio de inferencia local funcional.

Validacion:
Contenedor operativo, respuestas HTTP correctas
y pruebas automaticas satisfactorias.

## 3. Riesgos y medidas

Riesgo: diferencias entre motores de inferencia.
Medida: comparar detecciones y tiempos de respuesta.

Riesgo: incompatibilidades de dependencias.
Medida: definir versiones y utilizar Docker.

Riesgo: errores durante actualizaciones.
Medida: pruebas automaticas con GitHub Actions.

Riesgo: baja generalizacion del modelo.
Medida: conservar los resultados de evaluacion
y documentar las limitaciones del dataset.

Riesgo: falta de acceso compartido a datos DVC.
Medida: conservar el respaldo local y configurar
almacenamiento compartido cuando sea necesario.

## 4. Resultado esperado

Disponer de un servicio local de Deep Learning
que reciba imagenes, ejecute inferencias mediante
YOLOv8n-OBB optimizado y entregue respuestas JSON,
con empaquetado Docker, versionado y pruebas
automatizadas para facilitar su mantenimiento.
