"""
M8 - Métricas de validación del MODELO REAL de Astronomus (Clase #8).

El modelo de producción NO es un clasificador supervisado, sino un pipeline de
detección de anomalías + ranking (Autoencoder ponderado + Isolation Forest + índice
de Heller → IHP), implementado en backend/src/models/train.py.

Por eso las métricas correctas no son las de un clasificador plano, sino métricas de
RANKING/RECUPERACIÓN: ¿el modelo rankea arriba a los Griales (Tierra 2.0) conocidos,
que quedaron FUERA del entrenamiento (validación held-out)?

Requiere haber corrido antes:
    py backend/main.py --all
    py backend/src/models/train.py        # genera ranking_anomalias.csv

Ejecutar desde la RAÍZ del repo:
    py backend/tests/Simple/M8_metricas_anomalias.py

No modifica nada del proyecto: solo lee ranking_anomalias.csv y escribe reportes/figuras.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ----------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[3]
BACKEND = RAIZ / "backend"
sys.path.append(str(BACKEND))
from src.visualization.config_plots import configurar_estilo  # noqa: E402

RANKING = BACKEND / "artifacts" / "models" / "ranking_anomalias.csv"
REPORTES = BACKEND / "reports"
FIGURAS = REPORTES / "figures"
FIGURAS.mkdir(parents=True, exist_ok=True)

# Las 3 señales que produce el modelo
SENALES = ["score_ia", "score_heller", "ihp"]
NOMBRE_SENAL = {
    "score_ia": "Rareza IA (AE + IsoForest)",
    "score_heller": "Física (Heller ≈ ESI)",
    "ihp": "IHP (combinado)",
}
KS = [10, 20, 50, 100]


def log_factory():
    lineas = []
    def log(txt=""):
        print(txt)
        lineas.append(str(txt))
    return log, lineas


# ======================================================================
# CARGA
# ======================================================================
def cargar():
    if not RANKING.exists():
        raise FileNotFoundError(
            f"No se encontró {RANKING}.\n"
            "Corré primero:  py backend/src/models/train.py"
        )
    df = pd.read_csv(RANKING, low_memory=False)
    # Universo etiquetado: clases 0/1/2 (los -1/3 son desconocidos/pseudo-labels)
    df_lab = df[df["target_class"].isin([0, 1, 2])].copy().reset_index(drop=True)
    return df, df_lab


# ======================================================================
# MÉTRICAS DE RANKING
# ======================================================================
def evaluar():
    configurar_estilo()
    log, lineas = log_factory()

    df, df_lab = cargar()
    is_grial = (df_lab["target_class"] == 2).astype(int).values
    is_hab = df_lab["target_class"].isin([1, 2]).astype(int).values

    log("=" * 70)
    log(" MÉTRICAS DEL MODELO REAL — Detección de anomalías + ranking (IHP)")
    log("=" * 70)
    log(f" Universo etiquetado: {len(df_lab)} planetas "
        f"(clase 0={int((df_lab.target_class==0).sum())}, "
        f"1={int((df_lab.target_class==1).sum())}, "
        f"2/Grial={int((df_lab.target_class==2).sum())})")
    log(f" Catálogo total rankeado: {len(df)} planetas")
    log("")

    # ---------- 1. AUC y PR-AUC por señal ----------
    log("-" * 70)
    log(" 1. PODER DE RANKING (ROC-AUC y PR-AUC / Average Precision)")
    log("    Pregunta: ¿la señal rankea arriba a los positivos reales?")
    log("-" * 70)
    filas_auc = []
    for objetivo_nombre, objetivo in [("GRIAL (clase 2)", is_grial),
                                      ("HABITABLE (clase 1∪2)", is_hab)]:
        log(f"\n  Objetivo: {objetivo_nombre}  (positivos={int(objetivo.sum())})")
        log(f"    {'Señal':<28}{'ROC-AUC':>10}{'PR-AUC':>10}")
        for s in SENALES:
            try:
                a = roc_auc_score(objetivo, df_lab[s].values)
                p = average_precision_score(objetivo, df_lab[s].values)
            except Exception as e:
                a, p = float("nan"), float("nan")
                log(f"    [aviso] {s}: {e}")
            log(f"    {NOMBRE_SENAL[s]:<28}{a:>10.4f}{p:>10.4f}")
            filas_auc.append((objetivo_nombre, s, round(a, 4), round(p, 4)))

    # ---------- 2. Precision@k / Recall@k (IHP, recuperando Griales) ----------
    log("\n" + "-" * 70)
    log(" 2. PRECISION@k / RECALL@k — recuperando los Griales por IHP")
    log("-" * 70)
    n_grial = int(is_grial.sum())
    orden_ihp = df_lab.sort_values("ihp", ascending=False).reset_index(drop=True)
    pos_orden = (orden_ihp["target_class"] == 2).values
    log(f"    {'k':>5}{'#Griales en top-k':>20}{'Precision@k':>14}{'Recall@k':>12}")
    filas_pk = []
    for k in KS:
        hits = int(pos_orden[:k].sum())
        prec = hits / k
        rec = hits / n_grial if n_grial else 0.0
        log(f"    {k:>5}{hits:>20}{prec:>14.4f}{rec:>12.4f}")
        filas_pk.append((k, hits, round(prec, 4), round(rec, 4)))

    # ---------- 3. Ranking de cada Grial conocido ----------
    log("\n" + "-" * 70)
    log(" 3. POSICIÓN DE LOS GRIALES CONOCIDOS (ranking por IHP, universo etiquetado)")
    log("-" * 70)
    total = len(orden_ihp)
    griales = orden_ihp[orden_ihp["target_class"] == 2]
    log(f"    {'pl_name':<16}{'IHP':>8}{'rank':>8}{'percentil':>12}")
    for _, row in griales.iterrows():
        rank = int(orden_ihp.index[orden_ihp["pl_name"] == row["pl_name"]][0]) + 1
        pct = 100 * (1 - rank / total)
        log(f"    {str(row['pl_name']):<16}{row['ihp']:>8.2f}{rank:>8}{pct:>11.1f}%")

    # ---------- 4. Vista estilo Clase #8: matriz de confusión umbralizada ----------
    log("\n" + "-" * 70)
    log(" 4. VISTA CLASIFICACIÓN (Clase #8): umbral del modelo (cuantil 0.96 de IHP>0)")
    log("    Convierte el ranking en decisión binaria: ¿candidato excepcional?")
    log("-" * 70)
    umbral = df.loc[df["ihp"] > 0, "ihp"].quantile(0.96)
    y_pred = (df_lab["ihp"].values >= umbral).astype(int)
    cm = confusion_matrix(is_grial, y_pred, labels=[0, 1])
    log(f"    Umbral IHP = {umbral:.2f}")
    df_cm = pd.DataFrame(
        cm,
        index=["real: NO Grial", "real: Grial"],
        columns=["pred: no-cand", "pred: candidato"],
    )
    log(df_cm.to_string())
    df_cm.to_csv(REPORTES / "anomalias_matriz_confusion.csv", encoding="utf-8")
    acc_g = accuracy_score(is_grial, y_pred)
    prec_g = precision_score(is_grial, y_pred, zero_division=0)
    rec_g = recall_score(is_grial, y_pred, zero_division=0)
    f1_g = f1_score(is_grial, y_pred, zero_division=0)
    tp = int(cm[1, 1]); fp = int(cm[0, 1])
    log(f"\n    Accuracy  (aciertos totales / total):                        {acc_g:.4f}"
        f"   <- engañoso: el 99.8% son no-Grial")
    log(f"    Precision (de los marcados candidatos, ¿cuántos son Grial): {prec_g:.4f}  ({tp}/{tp+fp})")
    log(f"    Recall    (de los Griales, ¿cuántos marcó):                  {rec_g:.4f}")
    log(f"    F1                                                           {f1_g:.4f}")

    # ---------- 5. Insight honesto: IA vs Física ----------
    log("\n" + "-" * 70)
    log(" 5. ¿QUIÉN HACE EL TRABAJO? — IA no supervisada vs física (Heller)")
    log("-" * 70)
    corr_heller_esi = df_lab["score_heller"].corr(df_lab["phl_esi"]) if "phl_esi" in df_lab else float("nan")
    corr_ia_esi = df_lab["score_ia"].corr(df_lab["phl_esi"]) if "phl_esi" in df_lab else float("nan")
    log(f"    Correlación score_heller ↔ phl_esi : {corr_heller_esi:+.3f}")
    log(f"    Correlación score_ia     ↔ phl_esi : {corr_ia_esi:+.3f}")
    log("    Nota: el índice de Heller se construye con la misma física que el ESI, y")
    log("    el Grial se DEFINE por ESI>=0.80. Si su AUC es alto pero el de score_ia no,")
    log("    significa que quien 'encuentra' los Griales es la FÍSICA, no la IA.")

    # ---------- 6. R² (vista regresión: predecir el ESI continuo desde el score) ----------
    log("\n" + "-" * 70)
    log(" 6. R² SCORE (vista regresión) — varianza del ESI explicada por cada señal")
    log("    El modelo no es un regresor, pero su score es continuo: medimos cuánta")
    log("    varianza del ESI (target continuo) explica un ajuste lineal de cada señal.")
    log("-" * 70)
    filas_r2 = []
    esi = df_lab["phl_esi"].values
    mask_esi = np.isfinite(esi)
    log(f"    {'Señal':<28}{'R²':>10}")
    for s in SENALES:
        x = df_lab[s].values[mask_esi]
        y = esi[mask_esi]
        coef = np.polyfit(x, y, 1)          # ajuste lineal simple (OLS 1D)
        y_hat = np.polyval(coef, x)
        r2 = r2_score(y, y_hat)
        log(f"    {NOMBRE_SENAL[s]:<28}{r2:>10.4f}")
        filas_r2.append((s, round(r2, 4)))
    log("    (R² alto en 'Física' confirma la circularidad Heller≈ESI; bajo en 'Rareza IA'.)")
    log("=" * 70)

    # ---------- Persistencia ----------
    (REPORTES / "anomalias_metricas.txt").write_text("\n".join(lineas), encoding="utf-8")
    pd.DataFrame(filas_auc, columns=["objetivo", "senal", "roc_auc", "pr_auc"]).to_csv(
        REPORTES / "anomalias_auc.csv", index=False, encoding="utf-8")
    pd.DataFrame(filas_pk, columns=["k", "griales_en_topk", "precision_at_k", "recall_at_k"]).to_csv(
        REPORTES / "anomalias_precision_at_k.csv", index=False, encoding="utf-8")
    pd.DataFrame(
        [("Accuracy", round(acc_g, 4)), ("Precision", round(prec_g, 4)),
         ("Recall", round(rec_g, 4)), ("F1", round(f1_g, 4)), ("Umbral_IHP", round(umbral, 2))],
        columns=["metrica", "valor"],
    ).to_csv(REPORTES / "anomalias_clasificacion.csv", index=False, encoding="utf-8")
    pd.DataFrame(filas_r2, columns=["senal", "r2_vs_esi"]).to_csv(
        REPORTES / "anomalias_r2.csv", index=False, encoding="utf-8")
    print(f"\n ✓ Reporte: {REPORTES / 'anomalias_metricas.txt'}")
    print(f" ✓ CSVs: anomalias_auc.csv, anomalias_precision_at_k.csv, "
          f"anomalias_clasificacion.csv, anomalias_r2.csv")

    # ---------- Figuras ----------
    graficar_matriz_confusion(cm, umbral)
    graficar_roc_pr(df_lab, is_grial)
    graficar_dispersion(df_lab)
    return df_lab


def graficar_matriz_confusion(cm, umbral):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", cbar=True,
                xticklabels=["no-candidato", "candidato"],
                yticklabels=["NO Grial", "Grial"], ax=ax)
    ax.set_xlabel("Predicho (IHP ≥ umbral)")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de Confusión — modelo IHP (umbral={umbral:.1f})")
    fig.savefig(FIGURAS / "anomalias_matriz_confusion.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f" ✓ Figura: {FIGURAS / 'anomalias_matriz_confusion.png'}")


# ======================================================================
# FIGURAS
# ======================================================================
def graficar_roc_pr(df_lab, is_grial):
    # ROC
    fig, ax = plt.subplots(figsize=(7, 6))
    for s in SENALES:
        fpr, tpr, _ = roc_curve(is_grial, df_lab[s].values)
        ax.plot(fpr, tpr, lw=2, label=f"{NOMBRE_SENAL[s]} (AUC={roc_auc_score(is_grial, df_lab[s].values):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Aleatorio")
    ax.set_xlabel("FPR = FP / (FP + TN)")
    ax.set_ylabel("TPR = Recall = TP / (TP + FN)")
    ax.set_title("ROC — recuperación de Griales por señal")
    ax.legend(loc="lower right")
    fig.savefig(FIGURAS / "anomalias_roc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f" ✓ Figura: {FIGURAS / 'anomalias_roc.png'}")

    # PR
    fig, ax = plt.subplots(figsize=(7, 6))
    for s in SENALES:
        prec, rec, _ = precision_recall_curve(is_grial, df_lab[s].values)
        ax.plot(rec, prec, lw=2, label=f"{NOMBRE_SENAL[s]} (AP={average_precision_score(is_grial, df_lab[s].values):.3f})")
    base = is_grial.mean()
    ax.axhline(base, ls="--", color="k", lw=1, label=f"Base (prevalencia={base:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall — recuperación de Griales")
    ax.legend(loc="upper right")
    fig.savefig(FIGURAS / "anomalias_pr.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f" ✓ Figura: {FIGURAS / 'anomalias_pr.png'}")


def graficar_dispersion(df_lab):
    fig, ax = plt.subplots(figsize=(8, 6))
    colores = {0: "#888888", 1: "#2a9d8f", 2: "#f1c40f"}
    nombres = {0: "Inhóspito", 1: "Exótico", 2: "Tierra 2.0 (Grial)"}
    for c in [0, 1, 2]:
        sub = df_lab[df_lab["target_class"] == c]
        ax.scatter(sub["score_ia"], sub["score_heller"],
                   s=80 if c == 2 else 12, alpha=0.7 if c == 2 else 0.35,
                   c=colores[c], label=f"{nombres[c]} (n={len(sub)})",
                   edgecolors="k" if c == 2 else "none", zorder=3 if c == 2 else 1)
    ax.set_xlabel("score_ia (rareza IA)")
    ax.set_ylabel("score_heller (física)")
    ax.set_title("Señales del modelo por clase — ¿los Griales se separan?")
    ax.legend()
    fig.savefig(FIGURAS / "anomalias_dispersion.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f" ✓ Figura: {FIGURAS / 'anomalias_dispersion.png'}")


if __name__ == "__main__":
    evaluar()
    print("\n✓ Listo. Métricas del modelo real (anomalías/IHP) generadas.")
