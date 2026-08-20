"""
Fase 4 - Construccion de caracteristicas
==========================================
Construye los tres bloques de features descritos en el paper:
  (a) 9 caracteristicas lexicas escalares de la URL
  (b) Vectores TF-IDF (n-gramas de caracteres 1-3) sobre el texto plano
      extraido del HTML
  (c) 3 razones estructurales de hipervinculos
"""
import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import joblib


# ---------- (a) Caracteristicas lexicas de URL (9 features) ----------

def features_lexicas_url(url: str) -> dict:
    parsed = urlparse(url if "://" in url else "http://" + url)
    host = parsed.netloc or ""
    path = parsed.path or ""

    es_ip = bool(re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", host.split(":")[0]))
    num_subdominios = max(host.count(".") - 1, 0)
    tiene_arroba = "@" in url
    doble_barra = "//" in path
    guiones_dominio = "-" in host
    https_en_dominio = "https" in host.replace("www.", "")
    profundidad_ruta = len([p for p in path.split("/") if p])
    tlds_sospechosos = ["com", "net", "org", "info", "biz"]
    tld_incrustado_en_ruta = any(f".{t}" in path.lower() for t in tlds_sospechosos)

    return {
        "url_longitud": len(url),
        "url_num_subdominios": num_subdominios,
        "url_tiene_ip": int(es_ip),
        "url_tiene_arroba": int(tiene_arroba),
        "url_doble_barra_path": int(doble_barra),
        "url_guiones_dominio": int(guiones_dominio),
        "url_https_en_dominio": int(https_en_dominio),
        "url_profundidad_ruta": profundidad_ruta,
        "url_tld_en_ruta": int(tld_incrustado_en_ruta),
    }


# ---------- (c) Indicadores estructurales de hipervinculos (3 razones) ----------

def features_hipervinculos(html: str, url_pagina: str) -> dict:
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return {"frac_enlaces_externos": 0.0, "frac_anclas_nulas": 0.0, "frac_forms_accion_externa": 0.0}

    dominio_pagina = urlparse(url_pagina if "://" in url_pagina else "http://" + url_pagina).netloc

    enlaces = soup.find_all("a", href=True)
    total_enlaces = len(enlaces)
    externos = 0
    nulos = 0
    for a in enlaces:
        href = a["href"].strip()
        if href in ("#", "", "javascript:void(0)", "javascript:;"):
            nulos += 1
            continue
        dominio_href = urlparse(href).netloc
        if dominio_href and dominio_href != dominio_pagina:
            externos += 1
    frac_externos = externos / total_enlaces if total_enlaces else 0.0
    frac_nulas = nulos / total_enlaces if total_enlaces else 0.0

    forms = soup.find_all("form")
    total_forms = len(forms)
    forms_externos = 0
    for f in forms:
        action = f.get("action", "").strip()
        if action:
            dominio_action = urlparse(action).netloc
            if dominio_action and dominio_action != dominio_pagina:
                forms_externos += 1
    frac_forms_ext = forms_externos / total_forms if total_forms else 0.0

    return {
        "frac_enlaces_externos": frac_externos,
        "frac_anclas_nulas": frac_nulas,
        "frac_forms_accion_externa": frac_forms_ext,
    }


def construir_features_lexicas_y_links(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica (a) y (c) fila por fila. Devuelve dataframe con las 12 columnas escalares."""
    lex = df["url"].apply(features_lexicas_url).apply(pd.Series)
    links = df.apply(lambda r: features_hipervinculos(r["html_limpio"], r["url"]), axis=1).apply(pd.Series)
    return pd.concat([lex, links], axis=1)


# ---------- (b) TF-IDF caracter n-gramas sobre texto extraido del HTML ----------

def construir_tfidf(textos_train, textos_resto=None, max_features=5000):
    """
    Ajusta el vectorizador TF-IDF SOLO sobre el set de entrenamiento (Fase 5
    dice que el escalado/ajuste se hace exclusivamente sobre train).
    n-gramas de caracteres 1,2,3 como especifica el paper.
    """
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 3),
        max_features=max_features,
        norm="l2",
    )
    X_train = vectorizer.fit_transform(textos_train)
    if textos_resto is not None:
        X_resto = vectorizer.transform(textos_resto)
        return vectorizer, X_train, X_resto
    return vectorizer, X_train


if __name__ == "__main__":
    df = pd.read_parquet("../data/corpus_normalizado.parquet")

    print("Construyendo features lexicas de URL + hipervinculos...")
    df_scalar = construir_features_lexicas_y_links(df)
    df_scalar["label"] = df["label"].values
    df_scalar["url"] = df["url"].values
    df_scalar.to_parquet("../data/features_escalares.parquet", index=False)
    print(df_scalar.describe())

    print("\nFeatures escalares listas. El TF-IDF se ajusta en fase5_6_entrenamiento.py")
    print("(debe ajustarse solo sobre train, por eso se separa del resto del pipeline).")
