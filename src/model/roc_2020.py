"""
roc_2020.py
ROC curve y analisis de similitud entre la prediccion del modelo y los datos
reales de 2020.

Flujo:
  1. Carga urban_features
  2. Entrena el mismo modelo que prediccion_2020.py (logistic + poly2, 1975-2010)
  3. Predice sobre datos de 2015 (target = expansion en 2020)
  4. Compara con la columna 'expanded' de year=2015, que contiene la realidad 2020
  5. Genera figura 2x2:
       Panel 1 — Curva ROC con AUC y punto optimo de Youden J
       Panel 2 — Curva Precision-Recall con Average Precision
       Panel 3 — Calibracion del modelo (probabilidad predicha vs. tasa real)
       Panel 4 — Distribucion de scores: como separa el modelo las dos clases

Output: outputs/figures/roc_comparacion_2020.png
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import cross_val_score
from sqlalchemy import create_engine, text

warnings.filterwarnings('ignore')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT     = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data" / "processed" / "urban_features.csv"
OUT_FIGS = ROOT / "outputs" / "figures"
OUT_FIGS.mkdir(parents=True, exist_ok=True)

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/urbancast"
engine       = create_engine(DATABASE_URL)

FEATURES_POLY = ['bs_t', 'lat', 'lon', 'dist_centro_km']
ANOS_TRAIN    = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010]


# ─────────────────────────────────────────────────────────────────
# CARGA, SPLIT, MODELO
# ─────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text("SELECT * FROM urban_features"), conn)
        print(f"Datos desde PostGIS: {len(df):,} filas")
    except Exception as e:
        print(f"PostGIS no disponible ({e}) — usando CSV")
        df = pd.read_csv(CSV_PATH)
    return df


def entrenar_y_predecir(df: pd.DataFrame) -> tuple:
    """
    Replica exacta del pipeline de prediccion_2020.py:
      - Train: años 1975-2010
      - Prediccion: año 2015 (realidad = expansion en 2020)
      - PolynomialFeatures grado 2 + StandardScaler + LogisticRegression
      - Busca el mejor C por cross-validation (AUC-ROC) sobre submuestra
      - Umbral optimo de Youden J

    Retorna (y_true, y_proba, y_pred, umbral):
      y_true  → 0/1 real de 2020 (columna 'expanded' del periodo 2015->2020)
      y_proba → probabilidad predicha P(expanded=1) para cada pixel de 2015
      y_pred  → clasificacion binaria con umbral optimo
      umbral  → umbral de Youden J
    """
    cols_req = FEATURES_POLY + ['expanded']

    df_train = df[df['year'].isin(ANOS_TRAIN)].dropna(subset=cols_req).copy()
    df_pred  = df[df['year'] == 2015].dropna(subset=cols_req).copy()

    X_train = df_train[FEATURES_POLY].values
    y_train = df_train['expanded'].values
    X_pred  = df_pred[FEATURES_POLY].values
    y_true  = df_pred['expanded'].values

    print(f"  Train:  {len(X_train):,} filas (1975-2010)")
    print(f"  Test:   {len(X_pred):,} filas  (2015, target = 2020)")

    # Buscar mejor C sobre submuestra para no esperar mucho
    rng   = np.random.default_rng(42)
    n_cv  = min(25_000, len(X_train))
    idx   = rng.choice(len(X_train), n_cv, replace=False)
    X_cv, y_cv = X_train[idx], y_train[idx]

    candidatos  = [0.01, 0.05, 0.1, 0.5, 1.0]
    aucs_cv     = {}
    print("  CV (3-fold, AUC-ROC):")
    for C in candidatos:
        pipe = _pipeline(C)
        auc  = cross_val_score(pipe, X_cv, y_cv, cv=3,
                               scoring='roc_auc', n_jobs=1).mean()
        aucs_cv[C] = auc
        print(f"    C={C:.2f}  AUC={auc:.4f}")

    mejor_C = max(aucs_cv, key=aucs_cv.get)
    print(f"  -> Mejor C: {mejor_C}")

    # Entrenar sobre TODOS los datos de train con el mejor C
    pipe_final = _pipeline(mejor_C)
    pipe_final.fit(X_train, y_train)

    # Probabilidades predichas para los pixeles de 2015
    y_proba = pipe_final.predict_proba(X_pred)[:, 1]

    # Umbral optimo de Youden J: maximiza (TPR - FPR)
    # Es mejor que 0.5 cuando las clases estan desbalanceadas
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    j      = tpr - fpr
    umbral = float(thresholds[np.argmax(j)])
    y_pred = (y_proba >= umbral).astype(int)

    return y_true, y_proba, y_pred, umbral


def _pipeline(C: float) -> Pipeline:
    return Pipeline([
        ('poly',   PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler()),
        ('lr',     LogisticRegression(class_weight='balanced', C=C,
                                      max_iter=1000, solver='lbfgs',
                                      random_state=42))
    ])


# ─────────────────────────────────────────────────────────────────
# FIGURA PRINCIPAL
# ─────────────────────────────────────────────────────────────────

def graficar_roc_completo(y_true: np.ndarray, y_proba: np.ndarray,
                           y_pred: np.ndarray, umbral: float) -> None:
    """
    Figura 2x2 con cuatro perspectivas distintas de "que tan bien predice
    el modelo la realidad de 2020":

    Panel 1 — Curva ROC:
      Traza TPR (pixeles que expandieron y el modelo los detecto) vs.
      FPR (pixeles que NO expandieron pero el modelo dijo que si) para
      todos los umbrales posibles.
      AUC = Area Bajo la Curva:
        1.0 = prediccion perfecta
        0.5 = equivale a lanzar una moneda (aleatorio)
        El punto dorado marca el umbral optimo de Youden J.

    Panel 2 — Curva Precision-Recall:
      Mas relevante que ROC cuando hay desbalance de clases (pocas expansiones
      vs. muchos pixeles sin cambio). Muestra como varia la precision del
      modelo segun cuantos pixeles decide "marcar como expansion".
      AP = Average Precision (area bajo esta curva). Linea punteada = baseline.

    Panel 3 — Curva de Calibracion:
      Responde: "cuando el modelo dice 70% de probabilidad de expansion,
      realmente lo hace el 70%?"
      Si la curva sigue la diagonal perfecta → modelo perfectamente calibrado.
      Si la curva esta por encima de la diagonal → el modelo es conservador
      (subestima la probabilidad).
      Si esta por debajo → el modelo es demasiado confiado.

    Panel 4 — Distribucion de Scores:
      Histograma de las probabilidades predichas separadas por clase real.
      Clase 0 (no expandio) vs. Clase 1 (si expandio).
      Si las dos distribuciones se separan bien → el modelo discrimina bien.
      Mucho solapamiento → el modelo confunde las dos clases.
    """
    auc_val = roc_auc_score(y_true, y_proba)
    ap_val  = average_precision_score(y_true, y_proba)

    fpr_c, tpr_c, thresholds_c = roc_curve(y_true, y_proba)
    prec_c, rec_c, _            = precision_recall_curve(y_true, y_proba)

    # Punto de Youden J en la curva ROC
    j_idx      = np.argmax(tpr_c - fpr_c)
    youden_fpr = fpr_c[j_idx]
    youden_tpr = tpr_c[j_idx]

    # Curva de calibracion: probabilidad media predicha vs. fraccion de positivos reales
    # n_bins=10 → dividimos el espacio de probabilidades en 10 intervalos
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)

    # ── Figura ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    fig.suptitle(
        "Similitud entre Prediccion del Modelo y Realidad 2020 — UrbanCast\n"
        "Logistic Regression + Polinomios Grado 2  |  Entrenado 1975-2010  |  Evaluado en 2015->2020",
        fontsize=13, fontweight='bold', y=0.99
    )

    # ── Panel 1: Curva ROC ──────────────────────────────────────────────────────
    ax = axes[0, 0]

    # Rellenar el area bajo la curva
    ax.fill_between(fpr_c, tpr_c, alpha=0.15, color='#1565c0')
    ax.plot(fpr_c, tpr_c, color='#1565c0', linewidth=2.5,
            label=f'Modelo (AUC = {auc_val:.4f})')

    # Linea de referencia: clasificador aleatorio
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.6,
            label='Aleatorio (AUC = 0.50)')

    # Punto optimo de Youden J
    ax.scatter([youden_fpr], [youden_tpr], s=120, color='#f9a825',
               zorder=5, edgecolors='black', linewidths=1.5,
               label=f'Umbral optimo (J={youden_tpr - youden_fpr:.3f})')

    # Lineas punteadas hacia los ejes desde el punto optimo
    ax.axhline(youden_tpr, color='#f9a825', linestyle=':', linewidth=1, alpha=0.6)
    ax.axvline(youden_fpr, color='#f9a825', linestyle=':', linewidth=1, alpha=0.6)

    # Anotacion de AUC en el area
    ax.text(0.55, 0.25, f'AUC = {auc_val:.4f}',
            fontsize=14, fontweight='bold', color='#1565c0',
            transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#1565c0', alpha=0.85))

    ax.set_xlabel('Tasa de Falsos Positivos  (FPR = FP / (FP+TN))', fontsize=11)
    ax.set_ylabel('Tasa de Verdaderos Positivos  (TPR = TP / (TP+FN))', fontsize=11)
    ax.set_title('Curva ROC\nPrediccion 2020 vs. Realidad 2020', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.25)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])

    # ── Panel 2: Curva Precision-Recall ────────────────────────────────────────
    ax = axes[0, 1]

    ax.fill_between(rec_c, prec_c, alpha=0.15, color='#2e7d32')
    ax.plot(rec_c, prec_c, color='#2e7d32', linewidth=2.5,
            label=f'Modelo (AP = {ap_val:.4f})')

    # Baseline: si siempre predijera positivo
    baseline = float(y_true.mean())
    ax.axhline(baseline, color='k', linestyle='--', linewidth=1.5, alpha=0.6,
               label=f'Baseline (AP = {baseline:.4f})')

    # Marcar el punto en la curva PR correspondiente al umbral de Youden
    prec_opt = float(y_pred.sum() and (y_true[y_pred == 1].sum() / y_pred.sum()))
    rec_opt  = float(y_true[y_pred == 1].sum() / y_true.sum()) if y_true.sum() > 0 else 0
    ax.scatter([rec_opt], [prec_opt], s=120, color='#f9a825',
               zorder=5, edgecolors='black', linewidths=1.5,
               label=f'Umbral optimo\n(P={prec_opt:.3f}, R={rec_opt:.3f})')

    # Anotacion de AP
    ax.text(0.55, 0.85, f'AP = {ap_val:.4f}',
            fontsize=14, fontweight='bold', color='#2e7d32',
            transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#2e7d32', alpha=0.85))

    ax.set_xlabel('Recall  (fraccion de expansiones reales detectadas)', fontsize=11)
    ax.set_ylabel('Precision  (fraccion de predicciones correctas)', fontsize=11)
    ax.set_title('Curva Precision-Recall\nMas area = mejor (especialmente con clases desbalanceadas)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.25)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])

    # ── Panel 3: Calibracion ───────────────────────────────────────────────────
    ax = axes[1, 0]

    # Diagonal perfecta (calibracion ideal)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.6,
            label='Calibracion perfecta')

    # Curva de calibracion real del modelo
    ax.plot(prob_pred, prob_true, 'o-', color='#6a1b9a', linewidth=2.5,
            markersize=8, markerfacecolor='white', markeredgewidth=2,
            label='Modelo calibrado')
    ax.fill_between(prob_pred, prob_true, prob_pred,
                    alpha=0.12, color='#6a1b9a',
                    label='Brecha de calibracion')

    # Histograma de distribuciones de scores (eje secundario)
    ax2 = ax.twinx()
    ax2.hist(y_proba[y_true == 0], bins=30, alpha=0.18,
             color='#1565c0', density=True, label='_')
    ax2.hist(y_proba[y_true == 1], bins=30, alpha=0.18,
             color='#c62828', density=True, label='_')
    ax2.set_ylabel('Densidad (distribuciones)', fontsize=9, color='gray')
    ax2.tick_params(axis='y', colors='gray')
    ax2.set_ylim(bottom=0)

    ax.set_xlabel('Probabilidad predicha  P(expanded=1)', fontsize=11)
    ax.set_ylabel('Fraccion real de pixeles que expandieron', fontsize=11)
    ax.set_title('Curva de Calibracion\n"Cuando el modelo dice X%, cuantos realmente expanden?"',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.25)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    # ── Panel 4: Distribucion de scores por clase ──────────────────────────────
    ax = axes[1, 1]

    scores_neg = y_proba[y_true == 0]   # pixeles que NO expandieron en 2020
    scores_pos = y_proba[y_true == 1]   # pixeles que SI expandieron en 2020

    bins = np.linspace(0, 1, 51)

    ax.hist(scores_neg, bins=bins, density=True, alpha=0.65,
            color='#1565c0', edgecolor='white', linewidth=0.4,
            label=f'No expandio (n={len(scores_neg):,})')
    ax.hist(scores_pos, bins=bins, density=True, alpha=0.65,
            color='#c62828', edgecolor='white', linewidth=0.4,
            label=f'Si expandio  (n={len(scores_pos):,})')

    # Linea del umbral optimo
    ax.axvline(umbral, color='#f9a825', linewidth=2.5, linestyle='--',
               label=f'Umbral optimo = {umbral:.3f}')

    # Anotaciones de solapamiento
    ax.text(0.02, 0.95,
            'Azul = clase 0 (no expande)\nRojo = clase 1 (si expande)\n'
            'Mayor separacion = mejor discriminacion',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))

    ax.set_xlabel('Probabilidad predicha  P(expanded=1)', fontsize=11)
    ax.set_ylabel('Densidad', fontsize=11)
    ax.set_title('Distribucion de Scores por Clase Real\n'
                 'Separacion entre clases = poder discriminativo del modelo',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)

    # ── Guardar ────────────────────────────────────────────────────────────────
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    ruta = str(OUT_FIGS / "roc_comparacion_2020.png")
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  -> Figura guardada: {ruta}")


# ─────────────────────────────────────────────────────────────────
# RESUMEN NUMERICO
# ─────────────────────────────────────────────────────────────────

def imprimir_resumen(y_true: np.ndarray, y_proba: np.ndarray,
                     y_pred: np.ndarray, umbral: float) -> None:
    """
    Imprime los numeros clave de similitud entre prediccion y realidad 2020.
    """
    from sklearn.metrics import (confusion_matrix, accuracy_score,
                                  precision_score, recall_score, f1_score)

    auc_val = roc_auc_score(y_true, y_proba)
    ap_val  = average_precision_score(y_true, y_proba)
    cm      = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    n = len(y_true)

    print("\n" + "=" * 52)
    print("  SIMILITUD PREDICCION 2020 vs. REALIDAD 2020")
    print("=" * 52)
    print(f"  Pixeles evaluados          : {n:,}")
    print(f"  Expandieron en 2020 (real) : {int(y_true.sum()):,}  ({y_true.mean()*100:.1f}%)")
    print(f"  Predichos como expansion   : {int(y_pred.sum()):,}  ({y_pred.mean()*100:.1f}%)")
    print()
    print(f"  AUC-ROC                    : {auc_val:.4f}")
    print(f"  Average Precision (AP)     : {ap_val:.4f}")
    print(f"  Umbral optimo (Youden J)   : {umbral:.4f}")
    print()
    print(f"  Verdaderos Positivos (TP)  : {tp:,}  (expansion detectada correctamente)")
    print(f"  Verdaderos Negativos (TN)  : {tn:,}  (no expansion, correcto)")
    print(f"  Falsos Positivos    (FP)   : {fp:,}  (predijo expansion, no ocurrio)")
    print(f"  Falsos Negativos    (FN)   : {fn:,}  (no predijo, pero si ocurrio)")
    print()
    print(f"  Accuracy                   : {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision                  : {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  Recall                     : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  F1-Score                   : {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print()

    # Interpretacion del AUC
    if auc_val >= 0.90:
        nivel = "Excelente (AUC >= 0.90)"
    elif auc_val >= 0.80:
        nivel = "Bueno (AUC >= 0.80)"
    elif auc_val >= 0.70:
        nivel = "Aceptable (AUC >= 0.70)"
    else:
        nivel = "Debil (AUC < 0.70) — considerar mas features o modelo mas complejo"
    print(f"  Interpretacion AUC: {nivel}")
    print("=" * 52)


# ─────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 52)
    print("  UrbanCast — ROC: Prediccion vs. Realidad 2020")
    print("=" * 52)

    df = load_data()

    print("\n-- Entrenando modelo y generando predicciones --")
    y_true, y_proba, y_pred, umbral = entrenar_y_predecir(df)

    imprimir_resumen(y_true, y_proba, y_pred, umbral)

    print("-- Generando figura ROC (4 paneles) --")
    graficar_roc_completo(y_true, y_proba, y_pred, umbral)
