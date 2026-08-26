# Detección de phishing con XGBoost — Student Paper CACIC 2026

Implementación de un clasificador de páginas de phishing que combina nueve
características léxicas de URL, TF-IDF de caracteres sobre el texto extraído
del HTML y tres indicadores estructurales de hipervínculos.

## Experimento oficial

El Student Paper reporta una única ejecución: partición aleatoria
estratificada 70/15/15 con `random_state=42`.

| Partición | Casos |
|---|---:|
| Corpus final | 57.309 |
| Phishing | 39.684 |
| Benignas | 17.625 |
| Train | 40.116 |
| Validación | 8.596 |
| Test | 8.597 |

El test contiene 2.644 páginas benignas y 5.953 de phishing. La matriz del
sistema híbrido es:

```text
                Predicho
              Benigno  Phishing
Real Benigno     2631        13
Real Phishing      66      5887
```

Las métricas se recalcularon desde matrices enteras. Las cuatro filas de
ablación corresponden al mismo test de 8.597 casos.

| Escenario | Exactitud | Precisión | Exhaustividad | F1 | TFP |
|---|---:|---:|---:|---:|---:|
| Solo URL | 98,31 % | 99,37 % | 98,19 % | 0,9877 | 1,40 % |
| Solo HTML (TF-IDF) | 93,21 % | 94,76 % | 95,46 % | 0,9511 | 11,88 % |
| Solo Hipervínculos | 83,01 % | 82,67 % | 95,46 % | 0,8861 | 45,05 % |
| **Sistema híbrido** | **99,08 %** | **99,78 %** | **98,89 %** | **0,9933** | **0,49 %** |

Solo HTML y Solo Hipervínculos coinciden en exhaustividad (95,46 %): ambos
dejan 270 falsos negativos sobre las 5.953 páginas de phishing del test. No
se verificó si además fallan sobre el mismo subconjunto de páginas, ya que
los artefactos publicados conservan las matrices de confusión y no las
predicciones por muestra.

## Protocolo reproducible

El código principal es `src/fase5_6_entrenamiento.py`. El corpus se divide en
dos pasos estratificados (70/30 y luego 50/50 del remanente), ambos con
semilla 42. Cada escenario usa una grilla unitaria con validación cruzada
estratificada de tres pliegues sobre train:

```text
max_depth=6
n_estimators=300
learning_rate=0.1
objective=binary:logistic
eval_metric=logloss
random_state=42
```

Se ejecutó una sola semilla. La partición de validación queda reservada y no
interviene en las métricas reportadas. Los artefactos conservados registran
XGBoost 3.3.0 y scikit-learn 1.9.0; esas versiones están fijadas en
`requirements.txt`.

Para ejecutar el entrenamiento completo desde la raíz del repositorio:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
cd src
python fase5_6_entrenamiento.py
```

El entrenamiento requiere `data/corpus_normalizado.parquet`, que no se
distribuye en el repositorio. Para comprobar los resultados publicados sin
instalar las dependencias de aprendizaje automático:

```bash
python scripts/verificar_resultados.py
```

El verificador usa sólo la biblioteca estándar, recalcula todas las métricas
y falla si el corpus, la ablación, las matrices, la configuración o los hashes
son inconsistentes.

## Estructura relevante

```text
phishing-xgboost/
├── notebooks/pipeline_completo.ipynb
├── scripts/verificar_resultados.py
├── src/
│   ├── fase1_recoleccion.py
│   ├── fase2_3_normalizacion.py
│   ├── fase4_features.py
│   └── fase5_6_entrenamiento.py
├── data/                              # datos locales, no versionados
├── results/student_paper_oficial/     # artefactos livianos verificados
└── requirements.txt
```

Los datos provienen de PhishTank y OpenPhish para phishing y principalmente
de Tranco para páginas benignas.

## Alcance y limitaciones

El split aleatorio no agrupa por dominio o campaña ni deduplica por similitud
semántica o de HTML. La clase benigna, basada principalmente en Tranco, puede
ser menos desafiante que portales de autenticación, pasarelas de pago o
infraestructura compartida. Se informa una sola partición y una sola semilla.
Por estas razones, los resultados no demuestran generalización cronológica,
desempeño ante campañas posteriores ni estabilidad frente a cambios de
distribución.

## Autores

Víctor Córdoba · Lionel Gutiérrez · Nahuel Leyes · Giuliana Pessina · Juan Ignacio Tartaglia · Jorge Tohme

Laboratorio de Investigación en Ciencia y Tecnología — Universidad del
Aconcagua, Mendoza, Argentina.

## Nota sobre asistencia generativa

Se utilizaron herramientas de IA generativa como asistencia para la generación
y revisión de código y para la revisión editorial. La ejecución del pipeline,
la validación de los resultados y la verificación experimental fueron
realizadas por los autores.
