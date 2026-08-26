"""
Fase 5 - Escalado
Fase 6 - Particion + Entrenamiento XGBoost + Estudio de ablacion
===================================================================
Este script:
1. Particiona el corpus 70/15/15 (train/val/test) estratificado.
2. Ajusta StandardScaler y TfidfVectorizer SOLO sobre train.
3. Reproduce la configuracion oficial del Student Paper mediante una grilla
   unitaria y CV estratificada de 3 pliegues sobre train.
4. Evalua en test: exactitud, precision, exhaustividad, F1, TFP y matriz de confusion.
5. Corre el estudio de ablacion: Solo-URL / Solo-HTML / Solo-Hipervinculos / Hibrido.
6. Guarda los artefactos del experimento oficial en ../results/student_paper_oficial/
"""
import csv
import hashlib
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from fase4_features import construir_features_lexicas_y_links, construir_tfidf

RESULTS_DIR = "../results/student_paper_oficial"
RANDOM_STATE = 42
OFFICIAL_CV_FOLDS = 3
OFFICIAL_MODEL_GRID = {
    "max_depth": [6],
    "n_estimators": [300],
    "learning_rate": [0.1],
}
MANIFEST_FILES = (
    "resumen_corpus.json", "tabla_ablacion.csv", "matriz_confusion.png",
    "metricas_hibrido.json", "config_modelo.json", "matrices_ablacion.json",
    "predicciones_test.csv",
)
COLUMNAS_PREDICCION = {
    "Solo URL": "solo_url",
    "Solo HTML (TF-IDF)": "solo_html",
    "Solo Hipervinculos": "solo_hipervinculos",
    "Sistema Hibrido": "sistema_hibrido",
}

COLS_LEXICAS = [
    "url_longitud", "url_num_subdominios", "url_tiene_ip", "url_tiene_arroba",
    "url_doble_barra_path", "url_guiones_dominio", "url_https_en_dominio",
    "url_profundidad_ruta", "url_tld_en_ruta",
]
COLS_HIPERVINCULOS = [
    "frac_enlaces_externos", "frac_anclas_nulas", "frac_forms_accion_externa"
]


def particionar(df):
    """70/15/15 estratificado con semilla 42."""
    train, temp = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=RANDOM_STATE
    )
    val, test = train_test_split(
        temp, test_size=0.50, stratify=temp["label"], random_state=RANDOM_STATE
    )
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def calcular_tfp(y_true, y_pred):
    """Tasa de falsos positivos = FP / (FP + TN)."""
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def entrenar_xgboost(X_train, y_train, cv_folds=OFFICIAL_CV_FOLDS):
    """Reproduce la grilla unitaria y CV de la ejecucion oficial."""
    base = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    folds = StratifiedKFold(
        n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE
    )
    grid = GridSearchCV(
        base, OFFICIAL_MODEL_GRID, cv=folds,
        scoring="accuracy", n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)
    print(f"Mejores hiperparametros: {grid.best_params_}")
    return grid.best_estimator_


def evaluar(modelo, X_test, y_test, nombre_escenario):
    """Devuelve metricas, matriz y las predicciones por muestra del escenario."""
    y_pred = modelo.predict(X_test)
    metrics = {
        "escenario": nombre_escenario,
        "exactitud": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "exhaustividad": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "tfp": calcular_tfp(y_test, y_pred),
    }
    return metrics, confusion_matrix(y_test, y_pred), y_pred


def guardar_predicciones(test, predicciones):
    """Escribe las predicciones por muestra, en orden del corpus.

    Permite que scripts/verificar_resultados.py reconstruya cada matriz de
    confusion caso por caso, sin depender de numpy.
    """
    indices = test["indice_corpus"].to_list()
    orden = sorted(range(len(indices)), key=lambda k: indices[k])
    ruta = os.path.join(RESULTS_DIR, "predicciones_test.csv")
    with open(ruta, "w", encoding="utf-8", newline="") as archivo:
        escritor = csv.writer(archivo, lineterminator="\n")
        escritor.writerow(["indice_corpus", "y_real"] + list(COLUMNAS_PREDICCION.values()))
        etiquetas = test["label"].to_list()
        for k in orden:
            escritor.writerow(
                [int(indices[k]), int(etiquetas[k])]
                + [int(predicciones[nombre][k]) for nombre in COLUMNAS_PREDICCION]
            )


def graficar_matriz_confusion(cm, nombre_archivo):
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Benigno", "Phishing"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Benigno", "Phishing"])
    ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusión - Sistema Híbrido")
    fig.colorbar(image)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, nombre_archivo), dpi=150)
    plt.close(fig)


def graficar_feature_importance(modelo, nombres_features, nombre_archivo, top_n=20):
    importancias = modelo.feature_importances_
    idx_top = np.argsort(importancias)[-top_n:]
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(range(len(idx_top)), importancias[idx_top])
    ax.set_yticks(range(len(idx_top)))
    ax.set_yticklabels([nombres_features[i] for i in idx_top])
    ax.set_xlabel("Importancia (gain)")
    ax.set_title(f"Top {top_n} Features - XGBoost")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, nombre_archivo), dpi=150)
    plt.close(fig)


def guardar_json(nombre_archivo, contenido):
    with open(os.path.join(RESULTS_DIR, nombre_archivo), "w", encoding="utf-8") as archivo:
        json.dump(contenido, archivo, ensure_ascii=False, indent=2)
        archivo.write("\n")


def sha256_archivo(ruta):
    digest = hashlib.sha256()
    with open(ruta, "rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def correr_pipeline_completo():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.read_parquet("../data/corpus_normalizado.parquet")
    df["indice_corpus"] = range(len(df))
    train, val, test = particionar(df)

    train_scalar = construir_features_lexicas_y_links(train)
    val_scalar = construir_features_lexicas_y_links(val)
    test_scalar = construir_features_lexicas_y_links(test)

    scaler = StandardScaler().fit(train_scalar)
    train_scalar_s = scaler.transform(train_scalar)
    val_scalar_s = scaler.transform(val_scalar)
    test_scalar_s = scaler.transform(test_scalar)
    joblib.dump(scaler, os.path.join(RESULTS_DIR, "scaler.joblib"))

    vectorizer, X_tfidf_train, X_tfidf_val = construir_tfidf(
        train["texto_plano"], val["texto_plano"], max_features=5000
    )
    X_tfidf_test = vectorizer.transform(test["texto_plano"])
    joblib.dump(vectorizer, os.path.join(RESULTS_DIR, "tfidf_vectorizer.joblib"))

    nombres_features = COLS_LEXICAS + COLS_HIPERVINCULOS + [
        f"tfidf_{token}" for token in vectorizer.get_feature_names_out()
    ]
    n_lex = len(COLS_LEXICAS)
    n_link = len(COLS_HIPERVINCULOS)
    escenarios = {
        "Solo URL": (
            train_scalar_s[:, :n_lex], test_scalar_s[:, :n_lex]
        ),
        "Solo HTML (TF-IDF)": (
            X_tfidf_train, X_tfidf_test
        ),
        "Solo Hipervinculos": (
            train_scalar_s[:, n_lex:n_lex + n_link],
            test_scalar_s[:, n_lex:n_lex + n_link]
        ),
        "Sistema Hibrido": (
            hstack([csr_matrix(train_scalar_s), X_tfidf_train]).tocsr(),
            hstack([csr_matrix(test_scalar_s), X_tfidf_test]).tocsr(),
        ),
    }

    resultados_ablacion = []
    matrices = []
    predicciones = {}
    modelo_hibrido = None
    metricas_hibrido = None
    matriz_hibrido = None

    for nombre, (X_train, X_test) in escenarios.items():
        print(f"\n=== Entrenando escenario: {nombre} ===")
        modelo = entrenar_xgboost(X_train, train["label"])
        metrics, cm, y_pred = evaluar(modelo, X_test, test["label"], nombre)
        predicciones[nombre] = y_pred
        resultados_ablacion.append(metrics)
        tn, fp, fn, tp = (int(value) for value in cm.ravel())
        matrices.append({
            "escenario": nombre,
            "matriz_confusion": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
            "metricas": {
                key: float(metrics[key])
                for key in ("exactitud", "precision", "exhaustividad", "f1", "tfp")
            },
        })
        print(metrics)
        if nombre == "Sistema Hibrido":
            modelo_hibrido = modelo
            metricas_hibrido = metrics
            matriz_hibrido = cm
            graficar_matriz_confusion(cm, "matriz_confusion.png")
            joblib.dump(
                modelo, os.path.join(RESULTS_DIR, "modelo_xgboost_hibrido.joblib")
            )

    guardar_predicciones(test, predicciones)

    df_ablacion = pd.DataFrame(resultados_ablacion)
    df_ablacion.to_csv(os.path.join(RESULTS_DIR, "tabla_ablacion.csv"), index=False)
    print("\n=== TABLA DE ABLACION ===")
    print(df_ablacion.to_string(index=False))

    if modelo_hibrido is not None:
        graficar_feature_importance(
            modelo_hibrido, nombres_features, "feature_importance.png"
        )

    resumen = {
        "total_paginas": len(df),
        "benignas": int((df["label"] == 0).sum()),
        "phishing": int((df["label"] == 1).sum()),
        "train": len(train), "val": len(val), "test": len(test),
    }
    guardar_json("resumen_corpus.json", resumen)

    test_benignas = int((test["label"] == 0).sum())
    test_phishing = int((test["label"] == 1).sum())
    guardar_json("matrices_ablacion.json", {
        "experimento": "random_stratified_70_15_15",
        "test_total": len(test),
        "test_benignas": test_benignas,
        "test_phishing": test_phishing,
        "escenarios": matrices,
    })

    if modelo_hibrido is None or metricas_hibrido is None or matriz_hibrido is None:
        raise RuntimeError("No se genero el escenario hibrido")

    tn, fp, fn, tp = (int(value) for value in matriz_hibrido.ravel())
    guardar_json("metricas_hibrido.json", {
        "experimento": "random_stratified_70_15_15",
        "clase_positiva": "phishing",
        "test": {
            "total": len(test),
            "benignas": test_benignas,
            "phishing": test_phishing,
        },
        "matriz_confusion": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
        "metricas": {
            "accuracy": float(metricas_hibrido["exactitud"]),
            "precision": float(metricas_hibrido["precision"]),
            "recall": float(metricas_hibrido["exhaustividad"]),
            "f1": float(metricas_hibrido["f1"]),
            "false_positive_rate": float(metricas_hibrido["tfp"]),
        },
    })

    params = modelo_hibrido.get_params()
    guardar_json("config_modelo.json", {
        "experimento": "random_stratified_70_15_15",
        "implementacion_oficial": "src/fase5_6_entrenamiento.py",
        "dataset": "data/corpus_normalizado.parquet",
        "split": {
            "procedimiento": "dos train_test_split estratificados: 70/30 y 50/50 del remanente",
            "random_state": RANDOM_STATE,
            "train": len(train), "validation": len(val), "test": len(test),
            "validation_usage": "reservado; no se usa para calcular las metricas reportadas",
        },
        "seleccion_modelo": {
            "metodo": "GridSearchCV con grilla unitaria",
            "cv": "StratifiedKFold(n_splits=3, shuffle=True, random_state=42)",
            "scoring": "accuracy",
            "semillas_ejecutadas": [RANDOM_STATE],
        },
        "modelo_hibrido": {
            "n_estimators": int(params["n_estimators"]),
            "objective": params["objective"],
            "max_depth": int(params["max_depth"]),
            "learning_rate": float(params["learning_rate"]),
            "random_state": int(params["random_state"]),
            "n_jobs": int(params["n_jobs"]),
            "eval_metric": params["eval_metric"],
        },
        "caracteristicas": {
            "url_lexicas": len(COLS_LEXICAS),
            "hipervinculos": len(COLS_HIPERVINCULOS),
            "tfidf_max_features": 5000,
            "tfidf_analyzer": "char",
            "tfidf_ngram_range": [1, 3],
            "total_modelo": len(nombres_features),
        },
        "versiones": {
            "xgboost": xgb.__version__,
            "scikit_learn": sklearn.__version__,
        },
    })

    manifiesto = {
        nombre: sha256_archivo(os.path.join(RESULTS_DIR, nombre))
        for nombre in MANIFEST_FILES
    }
    guardar_json("MANIFIESTO_SHA256.json", manifiesto)

    print(f"\nResultados guardados en {RESULTS_DIR}/")
    return df_ablacion


if __name__ == "__main__":
    correr_pipeline_completo()
