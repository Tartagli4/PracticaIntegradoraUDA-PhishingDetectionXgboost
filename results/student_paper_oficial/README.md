# Resultados oficiales

Esta carpeta es la fuente única de resultados del Student Paper. Todos los
archivos corresponden al split aleatorio estratificado 70/15/15 con semilla
42 y al mismo test de 8.597 casos.

- `resumen_corpus.json`: composición del corpus y tamaños del split.
- `tabla_ablacion.csv`: métricas de los cuatro escenarios.
- `matrices_ablacion.json`: matrices enteras y métricas recalculadas.
- `matriz_confusion.png`: matriz del sistema híbrido.
- `metricas_hibrido.json`: resultado detallado del sistema híbrido.
- `config_modelo.json`: protocolo, hiperparámetros y versiones.
- `predicciones_test.csv`: predicciones por muestra de los cuatro escenarios.
- `MANIFIESTO_SHA256.json`: hashes de integridad.

Todos los archivos de esta carpeta son la salida directa de
`correr_pipeline_completo()` (`src/fase5_6_entrenamiento.py`), no derivaciones
posteriores. `predicciones_test.csv` conserva la predicción de cada uno de los
8.597 casos de test, de modo que las cuatro matrices de confusión pueden
reconstruirse caso por caso.

Verificación desde la raíz del repositorio:

```bash
python scripts/verificar_resultados.py
```
