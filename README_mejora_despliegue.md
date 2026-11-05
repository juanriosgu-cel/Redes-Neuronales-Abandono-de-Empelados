# Mejora del despliegue: clasificación y redes neuronales

## Archivos
- `aplicacion_redes_neuronales.py`: nueva app Streamlit, conserva el EDA y organiza clasificación, redes neuronales, comparación global, desempeño y predicción.
- `requirements_redes_neuronales.txt`: dependencias para copiar como `requirements.txt` en GitHub.
- `ACA2_Redes_Neuronales_Comparacion_v6.ipynb`: notebook que entrena MLP, Wide & Deep y Residual MLP y exporta los artefactos.

## Flujo de trabajo
1. Ejecutar el notebook completo en Colab con el CSV IBM HR Analytics.
2. En la última celda se genera `artifacts_neural.zip`. Descargarlo y extraer la carpeta `artifacts_neural`.
3. En el repositorio de GitHub de Streamlit, subir `aplicacion_redes_neuronales.py` renombrado como `app.py`, el CSV, `requirements_redes_neuronales.txt` renombrado como `requirements.txt` y la carpeta `artifacts_neural/` completa.
4. Mantener Python 3.11 en Streamlit Community Cloud.
5. Reiniciar la app.

## Artefactos exportados
`artifacts_neural/` contiene los tres modelos `.keras`, el preprocesador, metadatos, métricas reales, curvas y tabla comparativa global. La aplicación no entrena modelos durante el arranque.

## Nota sobre modelos clásicos
Los cuatro modelos clásicos se muestran con las métricas de referencia previamente reportadas. Si no se dispone de sus artefactos y no fueron evaluados sobre la misma partición, se presentan como comparación descriptiva y no como evaluación estrictamente equivalente.
