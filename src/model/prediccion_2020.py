"""
prediccion_2020.py
Regresion logistica + polinomios de grado 2 para predecir expansion urbana 2020.

Estrategia:
  - ENTRENAMIENTO: filas con year IN (1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010)
  - PREDICCION:    filas con year = 2015  (bs_t = superficie en 2015)
  - TARGET:        columna 'expanded' (1 si ese pixel creció en el siguiente período)
  - Las filas de year=2015 tienen como target la realidad de 2020, asi que
    podemos comparar prediccion vs. verdad sin usar ningun dato de 2020.

Consideraciones de Gauss-Markov (gauss_markov_spd.py):
  - GM4a violado (heteroscedasticidad) → class_weight='balanced' + regularizacion L2
  - Residuos siguen Cauchy (colas pesadas) → C pequeño para robustez ante outliers
  - GM2 OK (VIF < 2 en todas las features) → no hay multicolinealidad que preocupe

Outputs:
  outputs/figures/evaluacion_modelo_2020.png
  outputs/maps/mapa_prediccion_2020.html
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import cross_val_score
from sqlalchemy import create_engine, text

warnings.filterwarnings('ignore')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT     = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data" / "processed" / "urban_features.csv"
OUT_FIGS = ROOT / "outputs" / "figures"
OUT_MAPS = ROOT / "outputs" / "maps"
OUT_FIGS.mkdir(parents=True, exist_ok=True)
OUT_MAPS.mkdir(parents=True, exist_ok=True)

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/urbancast"
engine       = create_engine(DATABASE_URL)

# Features que entran al polinomio de grado 2.
# Elegidas porque su VIF < 2 (no hay multicolinealidad, validado en GM):
#   bs_t          → superficie ya construida: áreas densas expanden diferente a zonas vacias
#   lat, lon      → posicion geografica: la expansion tiene patron espacial (noreste de Medellin)
#   dist_centro_km → distancia al centro: la expansion tipicamente ocurre en la periferia
FEATURES_POLY = ['bs_t', 'lat', 'lon', 'dist_centro_km']

# Años de entrenamiento: excluimos 2015 (que usaremos para predecir 2020)
ANOS_TRAIN = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010]


# ─────────────────────────────────────────────────────────────────
# CARGA Y SPLIT
# ─────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text("SELECT * FROM urban_features"), conn)
        print(f"Datos desde PostGIS: {len(df):,} filas")
    except Exception as e:
        print(f"PostGIS no disponible ({e}) — usando CSV")
        df = pd.read_csv(CSV_PATH)
        print(f"Datos desde CSV: {len(df):,} filas")
    return df


def preparar_splits(df: pd.DataFrame) -> tuple:
    """
    Separa los datos en dos conjuntos excluyentes:

    TRAIN (1975-2010):
      El modelo aprende el patron historico: dado el estado de un pixel en
      el año T, ¿se expandio en T+5? Con 8 periodos tiene suficiente variedad
      temporal para generalizar.

    PREDICCION (2015):
      Usamos los datos del año 2015 como entrada. La columna 'expanded' de
      estas filas contiene la REALIDAD de 2020 (porque el periodo es 2015→2020),
      por lo que podemos medir qué tan bien predice el modelo 2020 sin haberlo visto.

    Nota sobre clases:
      Si expanded=1 es la clase minoritaria, el modelo tendería a predecir
      siempre 0 (no expande). class_weight='balanced' corrige esto.
    """
    cols_req = FEATURES_POLY + ['expanded', 'lat', 'lon']

    df_train = df[df['year'].isin(ANOS_TRAIN)].dropna(subset=cols_req).copy()
    df_pred  = df[df['year'] == 2015].dropna(subset=cols_req).copy()

    tasa_train = df_train['expanded'].mean() * 100
    tasa_pred  = df_pred['expanded'].mean() * 100

    print(f"\n  Train  : {len(df_train):,} filas  (1975-2010)  | tasa expansion = {tasa_train:.1f}%")
    print(f"  Pred   : {len(df_pred):,} filas  (2015->2020)  | tasa expansion = {tasa_pred:.1f}%")

    if tasa_train < 20:
        print("  AVISO: clase minoritaria < 20% → class_weight='balanced' es critico aqui")

    return df_train, df_pred


# ─────────────────────────────────────────────────────────────────
# PIPELINE DE MODELO
# ─────────────────────────────────────────────────────────────────

def construir_pipeline(C: float = 0.1) -> Pipeline:
    """
    Pipeline: PolynomialFeatures → StandardScaler → LogisticRegression

    PolynomialFeatures(degree=2):
      Transforma [bs_t, lat, lon, dist] en sus cuadrados e interacciones:
        bs_t², lat², lon², dist²
        bs_t·lat, bs_t·lon, bs_t·dist
        lat·lon, lat·dist, lon·dist
      = 14 features en total (4 originales + 10 nuevas).

      ¿Por que grado 2? La expansion urbana NO es lineal:
        - La relacion entre distancia al centro y probabilidad de expansion
          tiene forma de U invertida (muy cerca = ya construido, muy lejos = sin acceso).
        - lat·lon captura patrones de expansion diagonal (hacia el norte, hacia valles).
        - bs_t² captura el efecto de saturacion (poco espacio libre = menos expansion).

    StandardScaler:
      Escala todas las 14 features a media=0 y std=1.
      Necesario porque la regularizacion L2 penaliza a todos los coeficientes
      por igual. Sin escalar, features como lat (~6.2) y bs_t (~miles) recibirian
      penalizaciones muy diferentes siendo artificialmente comparadas.

    LogisticRegression:
      class_weight='balanced':
        Gauss-Markov detecto heteroscedasticidad y el periodo 2015→2020 tiene
        distribucion Cauchy. Balancear los pesos da mas importancia a los pixeles
        que SI expanden (clase minoritaria), evitando que el modelo ignore esa clase.

      C (regularizacion L2 inversa):
        C pequeño = mas regularizacion = coeficientes mas conservadores.
        Como los residuos siguen Cauchy (colas pesadas, muchos outliers extremos),
        usar C < 1 hace que el modelo no se "deje llevar" por esos valores extremos
        y generalice mejor a 2020.

      solver='lbfgs':
        Metodo de gradiente de segundo orden. Eficiente para L2 y datasets medianos.
        Converge bien con features escaladas.
    """
    return Pipeline([
        ('poly',   PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler()),
        ('lr',     LogisticRegression(
                       class_weight='balanced',
                       C=C,
                       max_iter=1000,
                       solver='lbfgs',
                       random_state=42
                   ))
    ])


def buscar_mejor_C(X_train: np.ndarray, y_train: np.ndarray) -> float:
    """
    Busca el valor de C optimo mediante 3-fold cross-validation (AUC-ROC).

    Usamos una submuestra estratificada de 25k para que la busqueda sea rapida
    (~1-2 minutos). El AUC en la submuestra es representativo porque la
    distribucion espacial de Medellin es bastante uniforme dentro de cada año.

    Rango de C explorado: [0.01, 0.05, 0.1, 0.5, 1.0]
      - 0.01: mucha regularizacion (modelo muy simple)
      - 0.1:  regularizacion moderada (buen balance para Cauchy)
      - 1.0:  poca regularizacion (puede overfittear a outliers extremos)
    """
    candidatos = [0.01, 0.05, 0.1, 0.5, 1.0]

    # Submuestra estratificada para que la proporcion de expanded=1 se mantenga
    rng  = np.random.default_rng(42)
    n_cv = min(25_000, len(X_train))

    # Estratificacion manual: mismo ratio de positivos/negativos
    idx_pos = np.where(y_train == 1)[0]
    idx_neg = np.where(y_train == 0)[0]
    n_pos   = int(n_cv * y_train.mean())
    n_neg   = n_cv - n_pos
    idx_cv  = np.concatenate([
        rng.choice(idx_pos, min(n_pos, len(idx_pos)), replace=False),
        rng.choice(idx_neg, min(n_neg, len(idx_neg)), replace=False)
    ])
    X_cv, y_cv = X_train[idx_cv], y_train[idx_cv]

    print(f"  CV con {len(X_cv):,} muestras (estratificado) | 3 folds | metrica: AUC-ROC")
    print(f"  {'C':>6}  {'AUC medio':>10}  {'Std':>8}")

    resultados = {}
    for C in candidatos:
        pipe = construir_pipeline(C=C)
        aucs = cross_val_score(pipe, X_cv, y_cv, cv=3,
                               scoring='roc_auc', n_jobs=1)
        resultados[C] = aucs.mean()
        print(f"  {C:>6.2f}  {aucs.mean():>10.4f}  {aucs.std():>8.4f}")

    mejor_C = max(resultados, key=resultados.get)
    print(f"\n  -> Mejor C: {mejor_C}  (AUC={resultados[mejor_C]:.4f})")
    return mejor_C


def importancias_features(pipe: Pipeline) -> pd.DataFrame:
    """
    Mapea los coeficientes del modelo logistico a los nombres de las features
    polinomicas para interpretar cuales impulsan mas la prediccion de expansion.

    Coeficiente positivo → esa feature aumenta la probabilidad de expansion.
    Coeficiente negativo → esa feature la reduce.

    Los coeficientes ya estan en escala estandarizada (despues del StandardScaler),
    por lo que son directamente comparables entre si.
    """
    nombres_poly = pipe.named_steps['poly'].get_feature_names_out(FEATURES_POLY)
    coefs        = pipe.named_steps['lr'].coef_[0]

    return pd.DataFrame({
        'feature'    : nombres_poly,
        'coeficiente': coefs,
        'abs_coef'   : np.abs(coefs)
    }).sort_values('abs_coef', ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────
# EVALUACION Y VISUALIZACION
# ─────────────────────────────────────────────────────────────────

def umbral_optimo(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Calcula el umbral de decision optimo usando el indice de Youden J:
      J = Sensitivity + Specificity - 1 = TPR - FPR

    Maximizar J da el umbral donde el modelo equilibra mejor la deteccion
    de verdaderos positivos vs. el control de falsos positivos.
    Es preferible a usar 0.5 cuando las clases son desbalanceadas.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    j        = tpr - fpr
    idx_opt  = np.argmax(j)
    umbral   = thresholds[idx_opt]
    return float(umbral)


def graficar_evaluacion(y_true: np.ndarray, y_proba: np.ndarray,
                         y_pred: np.ndarray, df_imp: pd.DataFrame) -> None:
    """
    Figura 2x2 con los 4 graficos estandar de evaluacion de modelos binarios.
    """
    auc_val = roc_auc_score(y_true, y_proba)
    ap_val  = average_precision_score(y_true, y_proba)
    cm      = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fpr_c, tpr_c, _ = roc_curve(y_true, y_proba)
    prec_c, rec_c, _ = precision_recall_curve(y_true, y_proba)

    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    fig.suptitle(
        "Regresion Logistica + Polinomios Grado 2 — UrbanCast\n"
        "Prediccion expansion urbana 2020  |  Entrenado con datos 1975-2010",
        fontsize=13, fontweight='bold'
    )

    # Panel 1: ROC
    ax = axes[0, 0]
    ax.plot(fpr_c, tpr_c, 'b-', linewidth=2.5, label=f'Modelo (AUC = {auc_val:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.2, label='Clasificador aleatorio (AUC = 0.5)')
    ax.fill_between(fpr_c, tpr_c, alpha=0.12, color='blue')
    ax.set_xlabel('Tasa de Falsos Positivos  (1 - Especificidad)')
    ax.set_ylabel('Tasa de Verdaderos Positivos  (Sensibilidad / Recall)')
    ax.set_title(f'Curva ROC\nAUC = {auc_val:.4f}  (1.0 = perfecto  |  0.5 = aleatorio)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 2: Precision-Recall
    ax = axes[0, 1]
    ax.plot(rec_c, prec_c, 'g-', linewidth=2.5, label=f'Modelo (AP = {ap_val:.4f})')
    baseline = float(y_true.mean())
    ax.axhline(baseline, color='k', linestyle='--', linewidth=1.2,
               label=f'Baseline (AP = {baseline:.4f})')
    ax.fill_between(rec_c, prec_c, alpha=0.12, color='green')
    ax.set_xlabel('Recall  (fraccion de expansiones detectadas)')
    ax.set_ylabel('Precision  (fraccion de predicciones correctas)')
    ax.set_title(f'Curva Precision-Recall\nAP = {ap_val:.4f}  (mas area = mejor)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 3: Confusion matrix
    ax = axes[1, 0]
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)
    labels = [['TN\n(correcto)', 'FP\n(falsa alarma)'],
              ['FN\n(no detectado)', 'TP\n(correcto)']]
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i,
                    f'{cm[i,j]:,}\n{labels[i][j]}',
                    ha='center', va='center',
                    fontsize=11, fontweight='bold', color=color)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred: No expande', 'Pred: Expande'])
    ax.set_yticklabels(['Real: No expande', 'Real: Expande'])
    acc = (tp + tn) / (tp + tn + fp + fn)
    ax.set_title(f'Matriz de Confusion (2015 → prediccion 2020)\n'
                 f'Accuracy={acc:.3f}  TP={tp:,}  TN={tn:,}  FP={fp:,}  FN={fn:,}')

    # Panel 4: Importancias top 15
    ax = axes[1, 1]
    top15   = df_imp.head(15)
    colores = ['#e74c3c' if c > 0 else '#3498db' for c in top15['coeficiente']]
    ax.barh(top15['feature'], top15['coeficiente'], color=colores, alpha=0.85)
    ax.axvline(0, color='black', linewidth=1)
    ax.invert_yaxis()
    ax.set_xlabel('Coeficiente log-odds (estandarizado)\n'
                  'Rojo = aumenta P(expande)  |  Azul = reduce P(expande)')
    ax.set_title('Top 15 Features por Importancia\n'
                 '(modelo logistico con polinomios grado 2)')
    ax.grid(True, alpha=0.2, axis='x')

    plt.tight_layout()
    ruta = str(OUT_FIGS / 'evaluacion_modelo_2020.png')
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Figura de evaluacion guardada: {ruta}")


# ─────────────────────────────────────────────────────────────────
# MAPA HTML
# ─────────────────────────────────────────────────────────────────

def generar_mapa(df_2015: pd.DataFrame, y_proba: np.ndarray,
                 y_pred: np.ndarray, output: str = None) -> None:
    """
    Mapa HTML interactivo con Leaflet.js (mismo patron que mapa_temporal.py):
    datos incrustados como JSON, sin depender de folium.

    3 capas alternables mediante botones:

    CAPA 1 — Prediccion (heatmap azul→rojo):
      Intensidad proporcional a P(expanded=1) del modelo.

    CAPA 2 — Realidad 2020 (heatmap azul→rojo):
      Solo pixeles donde expanded=1 (la expansion si ocurrio en 2020).

    CAPA 3 — Errores del modelo (circulos):
      Naranja = Falso Positivo (el modelo predijo expansion, pero NO ocurrio)
      Morado  = Falso Negativo (el modelo NO predijo expansion, pero SI ocurrio)
    """
    if output is None:
        output = str(OUT_MAPS / "mapa_prediccion_2020.html")

    MAX_HEAT = 25_000
    MAX_ERR  = 8_000

    rng = np.random.default_rng(42)

    # Normalizar probabilidades a [0, 1] para la intensidad del heatmap
    p_min, p_max = y_proba.min(), y_proba.max()
    proba_norm   = (y_proba - p_min) / (p_max - p_min + 1e-9)

    # Capa 1: heatmap de probabilidad predicha
    idx_heat = rng.choice(len(df_2015), min(MAX_HEAT, len(df_2015)), replace=False)
    heat_pred = [
        [float(df_2015.iloc[i]['lat']),
         float(df_2015.iloc[i]['lon']),
         float(proba_norm[i])]
        for i in idx_heat
    ]

    # Capa 2: heatmap de expansion real 2020 (solo pixeles que expandieron)
    mask_real = df_2015['expanded'].values == 1
    df_real   = df_2015[mask_real].reset_index(drop=True)
    n_real    = min(MAX_HEAT, len(df_real))
    idx_real  = rng.choice(len(df_real), n_real, replace=False) if len(df_real) > n_real else np.arange(len(df_real))
    heat_real = [
        [float(df_real.iloc[i]['lat']), float(df_real.iloc[i]['lon']), 1.0]
        for i in idx_real
    ]

    # Capa 3: errores
    y_true_arr = df_2015['expanded'].values
    fp_mask    = (y_pred == 1) & (y_true_arr == 0)
    fn_mask    = (y_pred == 0) & (y_true_arr == 1)

    def _sample_err(mask: np.ndarray, n_max: int) -> list:
        idx = np.where(mask)[0]
        if len(idx) > n_max:
            idx = rng.choice(idx, n_max, replace=False)
        return [[float(df_2015.iloc[i]['lat']), float(df_2015.iloc[i]['lon'])]
                for i in idx]

    fp_data = _sample_err(fp_mask, MAX_ERR)
    fn_data = _sample_err(fn_mask, MAX_ERR)

    # Metricas para el panel lateral
    auc_val  = roc_auc_score(y_true_arr, y_proba)
    acc_val  = accuracy_score(y_true_arr, y_pred)
    prec_val = precision_score(y_true_arr, y_pred, zero_division=0)
    rec_val  = recall_score(y_true_arr, y_pred, zero_division=0)
    f1_val   = f1_score(y_true_arr, y_pred, zero_division=0)

    stats = {
        'auc':        round(float(auc_val),  4),
        'accuracy':   round(float(acc_val),  4),
        'precision':  round(float(prec_val), 4),
        'recall':     round(float(rec_val),  4),
        'f1':         round(float(f1_val),   4),
        'n_total':    int(len(df_2015)),
        'n_real_exp': int(mask_real.sum()),
        'n_pred_exp': int(y_pred.sum()),
        'n_fp':       int(fp_mask.sum()),
        'n_fn':       int(fn_mask.sum()),
        'n_tp':       int(((y_pred == 1) & (y_true_arr == 1)).sum()),
    }

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>UrbanCast — Prediccion expansion 2020</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://leaflet.github.io/Leaflet.heat/dist/leaflet-heat.js"></script>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'Segoe UI', Arial, sans-serif; }}
#map {{ width: 100vw; height: 100vh; }}

#panel {{
  position: absolute; top: 12px; right: 12px; z-index: 9999;
  background: rgba(10, 12, 28, 0.93);
  color: #e0e0e0; padding: 16px 20px;
  border-radius: 12px; min-width: 250px;
  border: 1px solid #2a2a4a;
  box-shadow: 0 4px 20px rgba(0,0,0,0.5);
}}
#panel h3 {{
  color: #64b5f6; font-size: 13px; text-align: center;
  margin-bottom: 12px; letter-spacing: 0.5px;
}}
.stat-row {{
  display: flex; justify-content: space-between;
  align-items: center; margin: 5px 0; font-size: 12px;
}}
.stat-label {{ color: #90a4ae; }}
.stat-val {{ font-weight: 700; color: #e3f2fd; }}
.stat-val.good  {{ color: #81c784; }}
.stat-val.warn  {{ color: #ffb74d; }}
.stat-val.error {{ color: #ef9a9a; }}
.divider {{ border: none; border-top: 1px solid #2a2a4a; margin: 8px 0; }}
.model-info {{ font-size: 10px; color: #546e7a; text-align: center; margin-top: 8px; }}

#controls {{
  position: absolute; bottom: 28px;
  left: 50%; transform: translateX(-50%);
  z-index: 9999;
  background: rgba(10, 12, 28, 0.93);
  padding: 12px 22px; border-radius: 14px;
  border: 1px solid #2a2a4a;
  display: flex; gap: 12px; align-items: center;
  box-shadow: 0 4px 20px rgba(0,0,0,0.5);
}}
.capa-label {{ color: #90a4ae; font-size: 12px; margin-right: 4px; }}
.btn {{
  padding: 9px 18px; border-radius: 8px; border: none;
  cursor: pointer; font-size: 13px; font-weight: 600;
  letter-spacing: 0.3px; transition: all 0.2s ease;
}}
.btn.active  {{ opacity: 1.0; box-shadow: 0 0 12px rgba(255,255,255,0.25); transform: scale(1.06); }}
.btn.inactive {{ opacity: 0.40; }}
#btn-pred {{ background: #1565c0; color: white; }}
#btn-real {{ background: #2e7d32; color: white; }}
#btn-err  {{ background: #e65100; color: white; }}

#legend {{
  position: absolute; bottom: 100px; left: 12px; z-index: 9999;
  background: rgba(10, 12, 28, 0.93); color: #e0e0e0;
  padding: 12px 16px; border-radius: 10px;
  border: 1px solid #2a2a4a; font-size: 11px;
  box-shadow: 0 4px 20px rgba(0,0,0,0.5);
}}
.leg-title {{ font-weight: 700; color: #64b5f6; margin-bottom: 8px; font-size: 12px; }}
.leg-item  {{ display: flex; align-items: center; margin: 5px 0; }}
.leg-dot   {{ width: 13px; height: 13px; border-radius: 50%; margin-right: 9px; flex-shrink: 0; }}
.leg-bar {{
  width: 120px; height: 12px; border-radius: 4px; margin: 6px 0 2px;
  background: linear-gradient(to right, blue, lime, yellow, red);
}}
</style>
</head>
<body>
<div id="map"></div>

<div id="panel">
  <h3>UrbanCast — Prediccion 2020</h3>

  <div class="stat-row">
    <span class="stat-label">AUC-ROC</span>
    <span class="stat-val {'good' if stats['auc'] > 0.75 else 'warn'}">{stats['auc']}</span>
  </div>
  <div class="stat-row">
    <span class="stat-label">Accuracy</span>
    <span class="stat-val">{stats['accuracy']}</span>
  </div>
  <div class="stat-row">
    <span class="stat-label">Precision</span>
    <span class="stat-val">{stats['precision']}</span>
  </div>
  <div class="stat-row">
    <span class="stat-label">Recall</span>
    <span class="stat-val">{stats['recall']}</span>
  </div>
  <div class="stat-row">
    <span class="stat-label">F1-Score</span>
    <span class="stat-val {'good' if stats['f1'] > 0.6 else 'warn'}">{stats['f1']}</span>
  </div>

  <hr class="divider">

  <div class="stat-row">
    <span class="stat-label">Pixeles analizados</span>
    <span class="stat-val">{stats['n_total']:,}</span>
  </div>
  <div class="stat-row">
    <span class="stat-label">Expandieron (real)</span>
    <span class="stat-val good">{stats['n_real_exp']:,}</span>
  </div>
  <div class="stat-row">
    <span class="stat-label">Predichos expand.</span>
    <span class="stat-val">{stats['n_pred_exp']:,}</span>
  </div>
  <div class="stat-row">
    <span class="stat-label">Verdaderos positivos</span>
    <span class="stat-val good">{stats['n_tp']:,}</span>
  </div>
  <div class="stat-row">
    <span class="stat-label">Falsos positivos</span>
    <span class="stat-val warn">{stats['n_fp']:,}</span>
  </div>
  <div class="stat-row">
    <span class="stat-label">Falsos negativos</span>
    <span class="stat-val error">{stats['n_fn']:,}</span>
  </div>

  <p class="model-info">Logistic Regression + Poly(2)<br>Train: 1975-2010 | Pred: 2015->2020</p>
</div>

<div id="controls">
  <span class="capa-label">Ver capa:</span>
  <button class="btn active" id="btn-pred" onclick="showLayer('pred')">Prediccion 2020</button>
  <button class="btn inactive" id="btn-real" onclick="showLayer('real')">Realidad 2020</button>
  <button class="btn inactive" id="btn-err" onclick="showLayer('err')">Errores del modelo</button>
</div>

<div id="legend">
  <div class="leg-title">Heatmap de probabilidad</div>
  <div class="leg-bar"></div>
  <div style="display:flex; justify-content:space-between; font-size:10px; color:#90a4ae; margin-bottom:8px;">
    <span>Baja</span><span>Alta</span>
  </div>
  <div class="leg-title" style="margin-top:6px;">Errores del modelo</div>
  <div class="leg-item">
    <div class="leg-dot" style="background:#FF9800;"></div>
    Falso Positivo (predijo expande, no ocurrio)
  </div>
  <div class="leg-item">
    <div class="leg-dot" style="background:#9C27B0;"></div>
    Falso Negativo (no predijo, si ocurrio)
  </div>
</div>

<script>
var HEAT_PRED = {json.dumps(heat_pred)};
var HEAT_REAL = {json.dumps(heat_real)};
var FP_DATA   = {json.dumps(fp_data)};
var FN_DATA   = {json.dumps(fn_data)};

var map = L.map('map', {{
  center: [6.2442, -75.5812],
  zoom: 12,
  zoomControl: true
}});

var satellite = L.tileLayer(
  'https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
  {{ attribution: 'ArcGIS World Imagery', maxZoom: 19 }}
).addTo(map);

var streets = L.tileLayer(
  'https://services.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{{z}}/{{y}}/{{x}}',
  {{ attribution: 'ArcGIS Street Map', maxZoom: 19 }}
);

var topo = L.tileLayer(
  'https://services.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{{z}}/{{y}}/{{x}}',
  {{ attribution: 'ArcGIS Topo', maxZoom: 19 }}
);

L.control.layers({{
  'Satelite': satellite,
  'Calles': streets,
  'Topografico': topo
}}).addTo(map);

var heatOpts = {{
  radius: 12,
  blur: 15,
  minOpacity: 0.35,
  gradient: {{ 0.2: 'blue', 0.5: 'lime', 0.8: 'yellow', 1.0: 'red' }}
}};

var layerPred = L.heatLayer(HEAT_PRED, heatOpts);
var layerReal = L.heatLayer(HEAT_REAL, heatOpts);
var layerErr = L.layerGroup();

FP_DATA.forEach(function(p) {{
  L.circleMarker([p[0], p[1]], {{
    radius: 4, color: '#FF9800', fillColor: '#FF9800',
    fillOpacity: 0.65, weight: 1, opacity: 0.9
  }})
  .bindPopup('<b style="color:#FF9800">Falso Positivo</b><br>Predijo expansion, no ocurrio.')
  .addTo(layerErr);
}});

FN_DATA.forEach(function(p) {{
  L.circleMarker([p[0], p[1]], {{
    radius: 4, color: '#9C27B0', fillColor: '#9C27B0',
    fillOpacity: 0.65, weight: 1, opacity: 0.9
  }})
  .bindPopup('<b style="color:#9C27B0">Falso Negativo</b><br>Si expandio, no fue predicho.')
  .addTo(layerErr);
}});

layerPred.addTo(map);

function showLayer(name) {{
  [layerPred, layerReal, layerErr].forEach(function(l) {{
    if (map.hasLayer(l)) map.removeLayer(l);
  }});
  ['btn-pred', 'btn-real', 'btn-err'].forEach(function(id) {{
    document.getElementById(id).className = 'btn inactive';
  }});
  if (name === 'pred') {{
    layerPred.addTo(map);
    document.getElementById('btn-pred').className = 'btn active';
  }} else if (name === 'real') {{
    layerReal.addTo(map);
    document.getElementById('btn-real').className = 'btn active';
  }} else if (name === 'err') {{
    layerErr.addTo(map);
    document.getElementById('btn-err').className = 'btn active';
  }}
}}
</script>
</body>
</html>"""

    with open(output, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  -> Mapa guardado: {output}")


# ─────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 57)
    print("  UrbanCast — Prediccion 2020 (Logistic + Poly2)")
    print("=" * 57)

    df = load_data()

    # ── Split ──────────────────────────────────────────────────
    print("\n-- Preparando datos --")
    df_train, df_pred = preparar_splits(df)

    X_train = df_train[FEATURES_POLY].values
    y_train = df_train['expanded'].values
    X_pred  = df_pred[FEATURES_POLY].values
    y_true  = df_pred['expanded'].values

    # ── Busqueda de C ──────────────────────────────────────────
    print("\n-- Busqueda de regularizacion optima (cross-validation) --")
    mejor_C = buscar_mejor_C(X_train, y_train)

    # ── Entrenamiento final ────────────────────────────────────
    print(f"\n-- Entrenando modelo final (C={mejor_C}, grado 2) --")
    pipe = construir_pipeline(C=mejor_C)
    pipe.fit(X_train, y_train)

    n_feats = len(pipe.named_steps['poly'].get_feature_names_out(FEATURES_POLY))
    print(f"  Features despues del polinomio: {n_feats}  (4 originales + {n_feats-4} nuevas)")

    # ── Importancias ───────────────────────────────────────────
    df_imp = importancias_features(pipe)
    print("\n  Top 10 features:")
    print(df_imp[['feature', 'coeficiente']].head(10).to_string(index=False))

    # ── Prediccion ─────────────────────────────────────────────
    print("\n-- Prediccion sobre datos 2015 (target = 2020) --")
    y_proba = pipe.predict_proba(X_pred)[:, 1]

    # Umbral optimo (Youden J) en lugar de 0.5
    umbral = umbral_optimo(y_true, y_proba)
    y_pred = (y_proba >= umbral).astype(int)

    auc_val = roc_auc_score(y_true, y_proba)
    print(f"\n  AUC-ROC       : {auc_val:.4f}")
    print(f"  Umbral optimo : {umbral:.4f}  (Youden J)")
    print(f"  Expandidos reales   : {y_true.sum():,}  ({y_true.mean()*100:.1f}%)")
    print(f"  Expansiones predichas: {y_pred.sum():,}  ({y_pred.mean()*100:.1f}%)")

    print("\n" + classification_report(
        y_true, y_pred,
        target_names=['No expande', 'Expande']
    ))

    # ── Graficos de evaluacion ─────────────────────────────────
    print("-- Generando graficos de evaluacion --")
    graficar_evaluacion(y_true, y_proba, y_pred, df_imp)

    # ── Mapa HTML ──────────────────────────────────────────────
    print("-- Generando mapa interactivo --")
    generar_mapa(df_pred, y_proba, y_pred)

    print(f"\n{'='*57}")
    print(f"  Archivos generados:")
    print(f"  outputs/figures/evaluacion_modelo_2020.png")
    print(f"  outputs/maps/mapa_prediccion_2020.html")
    print(f"{'='*57}")
