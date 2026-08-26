#!/usr/bin/env python3
"""Verifica los artefactos oficiales del Student Paper sin dependencias externas."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "student_paper_oficial"
TRAINING_SCRIPT = ROOT / "src" / "fase5_6_entrenamiento.py"
NOTEBOOK = ROOT / "notebooks" / "pipeline_completo.ipynb"

EXPECTED_COUNTS = {
    "total_paginas": 57309,
    "benignas": 17625,
    "phishing": 39684,
    "train": 40116,
    "val": 8596,
    "test": 8597,
}
EXPECTED_TEST = {"total": 8597, "benignas": 2644, "phishing": 5953}
EXPECTED_HYBRID_MATRIX = {"TN": 2631, "FP": 13, "FN": 66, "TP": 5887}
EXPECTED_MATRIX_PNG_SHA256 = (
    "5c8967e866f073ef81c8744ee32a3e0037c05ea1b80be4dd991918686a46819c"
)
SCENARIOS = (
    "Solo URL",
    "Solo HTML (TF-IDF)",
    "Solo Hipervinculos",
    "Sistema Hibrido",
)
METRIC_KEYS = ("exactitud", "precision", "exhaustividad", "f1", "tfp")
MANIFEST_FILES = {
    "resumen_corpus.json",
    "tabla_ablacion.csv",
    "matriz_confusion.png",
    "metricas_hibrido.json",
    "config_modelo.json",
    "matrices_ablacion.json",
}
ABS_TOL = 1e-15


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def load_json(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics_from_matrix(matrix: dict[str, int]) -> dict[str, float]:
    tn, fp, fn, tp = (matrix[key] for key in ("TN", "FP", "FN", "TP"))
    total = tn + fp + fn + tp
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return {
        "exactitud": (tn + tp) / total,
        "precision": precision,
        "exhaustividad": recall,
        "f1": 2 * precision * recall / (precision + recall),
        "tfp": fp / (fp + tn),
    }


def require_close(observed: float, expected: float, context: str) -> None:
    require(
        math.isclose(observed, expected, rel_tol=0.0, abs_tol=ABS_TOL),
        f"{context}: {observed} != {expected}",
    )


def verify_counts() -> dict:
    summary = load_json("resumen_corpus.json")
    require(summary == EXPECTED_COUNTS, f"Conteos incompatibles: {summary}")
    require(
        summary["train"] + summary["val"] + summary["test"] == summary["total_paginas"],
        "Train + validacion + test no suma el corpus",
    )
    require(
        summary["benignas"] + summary["phishing"] == summary["total_paginas"],
        "Las clases no suman el corpus",
    )
    return summary


def verify_ablation() -> tuple[dict[str, dict[str, int]], dict[str, float]]:
    with (RESULTS / "tabla_ablacion.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    require(
        tuple(row["escenario"] for row in rows) == SCENARIOS,
        "La tabla no contiene los cuatro escenarios oficiales en orden",
    )

    document = load_json("matrices_ablacion.json")
    require(
        {
            "total": document["test_total"],
            "benignas": document["test_benignas"],
            "phishing": document["test_phishing"],
        }
        == EXPECTED_TEST,
        "Distribucion de test incompatible",
    )
    entries = document["escenarios"]
    require(
        tuple(entry["escenario"] for entry in entries) == SCENARIOS,
        "Las matrices no contienen los cuatro escenarios oficiales en orden",
    )

    matrices = {}
    for row, entry in zip(rows, entries, strict=True):
        matrix = entry["matriz_confusion"]
        matrices[row["escenario"]] = matrix
        require(
            sum(matrix.values()) == EXPECTED_TEST["total"],
            f"{row['escenario']}: la matriz no suma el test",
        )
        require(
            matrix["TN"] + matrix["FP"] == EXPECTED_TEST["benignas"],
            f"{row['escenario']}: cantidad benigna incompatible",
        )
        require(
            matrix["FN"] + matrix["TP"] == EXPECTED_TEST["phishing"],
            f"{row['escenario']}: cantidad phishing incompatible",
        )

        calculated = metrics_from_matrix(matrix)
        for key in METRIC_KEYS:
            require_close(float(row[key]), calculated[key], f"{row['escenario']} CSV {key}")
            require_close(
                float(entry["metricas"][key]),
                calculated[key],
                f"{row['escenario']} JSON {key}",
            )

    require(
        matrices["Sistema Hibrido"] == EXPECTED_HYBRID_MATRIX,
        "Matriz hibrida incompatible",
    )
    return matrices, metrics_from_matrix(matrices["Sistema Hibrido"])


def verify_hybrid(hybrid_metrics: dict[str, float]) -> None:
    document = load_json("metricas_hibrido.json")
    require(
        document["experimento"] == "random_stratified_70_15_15",
        "Identificador de experimento incompatible",
    )
    require(document["test"] == EXPECTED_TEST, "Test hibrido incompatible")
    require(
        document["matriz_confusion"] == EXPECTED_HYBRID_MATRIX,
        "Matriz hibrida detallada incompatible",
    )
    mapping = {
        "accuracy": "exactitud",
        "precision": "precision",
        "recall": "exhaustividad",
        "f1": "f1",
        "false_positive_rate": "tfp",
    }
    for external, local in mapping.items():
        require_close(
            float(document["metricas"][external]),
            hybrid_metrics[local],
            f"Metrica hibrida {external}",
        )


def verify_config() -> None:
    config = load_json("config_modelo.json")
    require(
        config["experimento"] == "random_stratified_70_15_15",
        "Configuracion de experimento incompatible",
    )
    require(
        config["split"]["random_state"] == 42
        and config["split"]["train"] == 40116
        and config["split"]["validation"] == 8596
        and config["split"]["test"] == 8597,
        "Configuracion del split incompatible",
    )
    require(
        config["seleccion_modelo"]["semillas_ejecutadas"] == [42],
        "La configuracion declara semillas adicionales",
    )
    require(
        config["seleccion_modelo"]["cv"]
        == "StratifiedKFold(n_splits=3, shuffle=True, random_state=42)",
        "Configuracion de validacion cruzada incompatible",
    )
    expected_model = {
        "n_estimators": 300,
        "objective": "binary:logistic",
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "logloss",
    }
    require(config["modelo_hibrido"] == expected_model, "Hiperparametros incompatibles")

    source = TRAINING_SCRIPT.read_text(encoding="utf-8")
    for fragment in (
        "RANDOM_STATE = 42",
        "OFFICIAL_CV_FOLDS = 3",
        '"max_depth": [6]',
        '"n_estimators": [300]',
        '"learning_rate": [0.1]',
        "test_size=0.30",
        "test_size=0.50",
        'stratify=df["label"]',
    ):
        require(fragment in source, f"El script no contiene el protocolo oficial: {fragment}")

    notebook = NOTEBOOK.read_text(encoding="utf-8").lower()
    require("random_state=42" in notebook, "El notebook no declara la semilla oficial")
    require("## 8." not in notebook, "El notebook contiene una seccion experimental adicional")


def verify_manifest() -> None:
    manifest = load_json("MANIFIESTO_SHA256.json")
    require(set(manifest) == MANIFEST_FILES, "Contenido inesperado en el manifiesto")
    for name, expected_hash in manifest.items():
        require(
            sha256(RESULTS / name) == expected_hash,
            f"Hash incompatible para {name}",
        )
    require(
        sha256(RESULTS / "matriz_confusion.png") == EXPECTED_MATRIX_PNG_SHA256,
        "La figura no es la matriz oficial validada",
    )


def main() -> int:
    try:
        required = MANIFEST_FILES | {"MANIFIESTO_SHA256.json"}
        for name in required:
            require((RESULTS / name).is_file(), f"Falta {name}")
        verify_counts()
        _, hybrid_metrics = verify_ablation()
        verify_hybrid(hybrid_metrics)
        verify_config()
        verify_manifest()
    except (VerificationError, KeyError, ValueError, OSError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print("OK: corpus 57309 = train 40116 + validacion 8596 + test 8597")
    print("OK: cuatro escenarios sobre 8597 casos (2644 benignos, 5953 phishing)")
    print("OK: matriz hibrida TN=2631 FP=13 FN=66 TP=5887")
    print("OK: accuracy=0.9908107479353263 precision=0.9977966101694915")
    print("OK: recall=0.9889131530320847 f1=0.9933350206698726")
    print("OK: TFP=0.004916792738275341 y hashes verificados")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
