"""
riemann_2020.py
Similitud espacial entre expansion PREDICHA 2020 y expansion REAL 2020
mediante integracion de Riemann (mismo enfoque que spd.py).

Pregunta central:
  "El modelo coloca la probabilidad de expansion en las mismas zonas
   geograficas donde la expansion realmente ocurrio en 2020?"

Enfoque correcto para la comparacion:
  p(x) = histograma de la variable X ponderado por y_proba
          → distribucion espacial PREDICHA: donde el modelo cree que expande
  q(x) = histograma de la variable X ponderado por y_true (0/1)
          → distribucion espacial REAL: donde la expansion ocurrio

  Ambas se normalizan con Simpson a integrales = 1 y se comparan con
  las 6 metricas de Riemann de spd.py.

  Fisher-Rao = 0    → distribuciones identicas, prediccion perfecta
  Fisher-Rao = pi/2 → sin solapamiento, prediccion completamente errada

Ademas se identifican los puntos palanca (leverage) del modelo logistico
para ver cuanto distorsionan la comparacion cuando se incluyen.

Outputs:
  outputs/figures/riemann_similitud_2020.png
  outputs/results/riemann_resultados_2020.csv
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import simpson
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score
from sqlalchemy import create_engine, text

warnings.filterwarnings('ignore')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT     = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data" / "processed" / "urban_features.csv"
OUT_FIGS = ROOT / "outputs" / "figures"
OUT_RES  = ROOT / "outputs" / "results"
OUT_FIGS.mkdir(parents=True, exist_ok=True)
OUT_RES.mkdir(parents=True, exist_ok=True)

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/urbancast"
engine       = create_engine(DATABASE_URL)

FEATURES_POLY = ['bs_t', 'lat', 'lon', 'dist_centro_km']
ANOS_TRAIN    = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010]

VARIABLES = ['lat', 'lon', 'dist_centro_km', 'bs_t']
LABELS = {
    'lat'           : 'Latitud  (patron Norte-Sur)',
    'lon'           : 'Longitud  (patron Este-Oeste)',
    'dist_centro_km': 'Distancia al centro (km)',
    'bs_t'          : 'Sup. construida en 2015 (m2)',
}


# ─────────────────────────────────────────────────────────────────
# MODELO
# ─────────────────────────────────────────────────────────────────

def _pipeline(c_reg: float) -> Pipeline:
    return Pipeline([
        ('poly',   PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler()),
        ('lr',     LogisticRegression(class_weight='balanced', C=c_reg,
                                      max_iter=1000, solver='lbfgs',
                                      random_state=42))
    ])


def load_y_predecir() -> tuple:
    """
    Entrena el modelo (años 1975-2010) y genera predicciones para 2015.
    Retorna (df, X_poly_scaled) donde df contiene y_true, y_proba, y_pred
    y todas las features espaciales para el set de año=2015.
    """
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text("SELECT * FROM urban_features"), conn)
        print(f"Datos desde PostGIS: {len(df):,} filas")
    except Exception as exc:
        print(f"PostGIS no disponible ({exc}) — usando CSV")
        df = pd.read_csv(CSV_PATH)

    cols = FEATURES_POLY + ['expanded']
    df_train = df[df['year'].isin(ANOS_TRAIN)].dropna(subset=cols).copy()
    df_test  = df[df['year'] == 2015].dropna(subset=cols).copy()

    X_tr = df_train[FEATURES_POLY].values
    y_tr = df_train['expanded'].values
    X_te = df_test[FEATURES_POLY].values

    # Seleccion de C por CV en submuestra estratificada
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_tr), min(25_000, len(X_tr)), replace=False)
    mejor_c = max(
        [0.01, 0.05, 0.1, 0.5, 1.0],
        key=lambda c: cross_val_score(
            _pipeline(c), X_tr[idx], y_tr[idx],
            cv=3, scoring='roc_auc', n_jobs=1).mean()
    )
    print(f"  Mejor C: {mejor_c}")

    pipe = _pipeline(mejor_c)
    pipe.fit(X_tr, y_tr)

    # X transformado (poly+scaled) — necesario para calcular palancas
    X_poly_scaled = pipe.named_steps['scaler'].transform(
        pipe.named_steps['poly'].transform(X_te)
    )

    y_proba = pipe.predict_proba(X_te)[:, 1]
    fpr, tpr, thresholds = roc_curve(df_test['expanded'].values, y_proba)
    umbral  = float(thresholds[np.argmax(tpr - fpr)])
    y_pred  = (y_proba >= umbral).astype(int)

    resultado = df_test[['lat', 'lon', 'dist_centro_km', 'bs_t', 'expanded']].copy()
    resultado = resultado.rename(columns={'expanded': 'y_true'})
    resultado['y_proba'] = y_proba
    resultado['y_pred']  = y_pred
    resultado = resultado.reset_index(drop=True)

    print(f"  Pixeles en 2015        : {len(resultado):,}")
    print(f"  Expansion real 2020    : {resultado['y_true'].sum():,}  ({resultado['y_true'].mean()*100:.1f}%)")
    print(f"  Umbral Youden          : {umbral:.4f}")
    return resultado, X_poly_scaled


# ─────────────────────────────────────────────────────────────────
# PUNTOS PALANCA
# ─────────────────────────────────────────────────────────────────

def calcular_palancas(x_scaled: np.ndarray, y_proba: np.ndarray) -> tuple:
    """
    Calcula leverage h_ii y distancia de Cook para cada pixel.

    En regresion logistica la matriz hat es:
      H = M (M^T M)^{-1} M^T
      M = X_scaled * sqrt(W),  W_i = p_i*(1-p_i)  (pesos de Fisher)

    h_ii = M_i^T (M^T M)^{-1} M_i  (vectorizado sin formar H completa)

    Cook's D_i ≈ r_i^2 * h_ii / (p * (1 - h_ii)^2)
      r_i = (y_proba_i - 0.5) / sqrt(W_i)  (residuo centrado en 0.5)

    Umbrales:
      Palanca    : h_ii > 2p/n
      Influyente : Cook's D > 4/n
    """
    n_obs, n_feat = x_scaled.shape
    weights = np.clip(y_proba * (1.0 - y_proba), 1e-6, 1.0 - 1e-6)
    m_mat   = x_scaled * np.sqrt(weights)[:, None]

    mtm_inv = np.linalg.pinv(m_mat.T @ m_mat)
    h_diag  = np.sum((m_mat @ mtm_inv) * m_mat, axis=1)

    residuo = (y_proba - 0.5) / np.sqrt(weights)
    cook_d  = (residuo**2 * h_diag) / (n_feat * (1.0 - h_diag + 1e-8)**2)

    mask_pal = h_diag > 2.0 * n_feat / n_obs
    mask_inf = cook_d  > 4.0 / n_obs

    return h_diag, cook_d, mask_pal, mask_inf


# ─────────────────────────────────────────────────────────────────
# RIEMANN: COMPARACION PREDICHO vs REAL
# ─────────────────────────────────────────────────────────────────

def _distancia_spd(var1: float, var2: float) -> float:
    """Distancia Riemanniana en SPD(2) diagonal: |log(var2/var1)|"""
    if var1 <= 0 or var2 <= 0:
        return float('nan')
    return abs(np.log(var2 / var1))


def riemann_predicho_vs_real(df: pd.DataFrame,
                              variable: str,
                              mask_excluir: np.ndarray = None) -> dict:
    """
    Compara la distribucion espacial PREDICHA con la REAL usando Riemann.

    p(x) = histograma de 'variable' ponderado por y_proba (normalizado)
            → "segun el modelo, la expansion tiene esta distribucion espacial"

    q(x) = histograma de 'variable' ponderado por y_true (normalizado)
            → "segun los datos, la expansion tuvo esta distribucion espacial"

    Si el modelo es bueno, p(x) ≈ q(x) → distancias cercanas a 0.

    mask_excluir : boolean array, True = excluir ese pixel del computo
                   (usado para filtrar puntos palanca)
    """
    if mask_excluir is not None:
        datos = df[~mask_excluir].copy()
    else:
        datos = df.copy()

    if len(datos) == 0:
        nan_dict = {k: float('nan') for k in
                    ['fisher_rao', 'hellinger', 'ks', 'l1', 'bhatta', 'spd_riemann']}
        nan_dict['variable'] = variable
        return nan_dict

    vals   = datos[variable].values
    pesos_p = datos['y_proba'].values           # pesos del modelo
    pesos_q = datos['y_true'].values.astype(float)  # pesos reales (0/1)

    # Si no hay ninguna expansion real, comparacion no tiene sentido
    if pesos_q.sum() == 0:
        nan_dict = {k: float('nan') for k in
                    ['fisher_rao', 'hellinger', 'ks', 'l1', 'bhatta', 'spd_riemann']}
        nan_dict['variable'] = variable
        return nan_dict

    x_min, x_max = vals.min(), vals.max()
    if x_min == x_max:
        return {'variable': variable, 'fisher_rao': 0.0, 'hellinger': 0.0,
                'ks': 0.0, 'l1': 0.0, 'bhatta': 0.0, 'spd_riemann': 0.0}

    bins        = np.linspace(x_min, x_max, 101)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    dx          = bin_centers[1] - bin_centers[0]

    # Histogramas ponderados: cada bin acumula la suma de pesos de los
    # pixeles que caen en ese bin. Usar pesos en lugar de conteos captura
    # "cuanta expansion (real o predicha) hay en cada zona geografica".
    h_pred, _ = np.histogram(vals, bins=bins, weights=pesos_p, density=False)
    h_real, _ = np.histogram(vals, bins=bins, weights=pesos_q, density=False)

    # Normalizar con Simpson para que ambas integren a 1 (densidades)
    area_p = simpson(h_pred.astype(float), x=bin_centers)
    area_q = simpson(h_real.astype(float), x=bin_centers)

    if area_p <= 0 or area_q <= 0:
        nan_dict = {k: float('nan') for k in
                    ['fisher_rao', 'hellinger', 'ks', 'l1', 'bhatta', 'spd_riemann']}
        nan_dict['variable'] = variable
        return nan_dict

    p = h_pred / area_p
    q = h_real / area_q

    # ── Metricas de Riemann (identicas a spd.py) ─────────────────────────────
    l1         = float(simpson(np.abs(p - q), x=bin_centers))
    hellinger  = float(np.sqrt(0.5 * simpson(
        (np.sqrt(np.maximum(p, 0)) - np.sqrt(np.maximum(q, 0)))**2,
        x=bin_centers)))
    rho_b      = float(np.clip(
        simpson(np.sqrt(np.maximum(p * q, 0)), x=bin_centers), 1e-8, 1.0))
    bhatta     = float(-np.log(rho_b))
    fisher_rao = float(np.arccos(rho_b))

    cdf_p = np.clip(np.cumsum(p * dx), 0.0, 1.0)
    cdf_q = np.clip(np.cumsum(q * dx), 0.0, 1.0)
    ks    = float(np.max(np.abs(cdf_p - cdf_q)))

    mu_p   = float(np.sum(p * bin_centers) * dx)
    mu_q   = float(np.sum(q * bin_centers) * dx)
    var_p  = max(float(np.sum(p * (bin_centers - mu_p)**2) * dx), 1e-12)
    var_q  = max(float(np.sum(q * (bin_centers - mu_q)**2) * dx), 1e-12)
    spd    = _distancia_spd(var_p, var_q)

    return {
        'variable'   : variable,
        'fisher_rao' : round(fisher_rao, 6),
        'hellinger'  : round(hellinger,  6),
        'ks'         : round(ks,         6),
        'l1'         : round(l1,         6),
        'bhatta'     : round(bhatta,     6),
        'spd_riemann': round(spd,        6),
    }


def correr_comparaciones(df: pd.DataFrame,
                          mask_pal: np.ndarray,
                          mask_inf: np.ndarray) -> pd.DataFrame:
    """
    Corre la comparacion Riemann para 3 escenarios por variable:
      todos          : todos los pixeles
      sin_palancas   : excluye h_ii > 2p/n
      sin_influyentes: excluye Cook's D > 4/n  (el mas conservador)
    """
    filas = []
    print(f"\n  {'Variable':<22} {'Escenario':<18} "
          f"{'Fisher-Rao':>11} {'Hellinger':>10} {'KS':>8} {'SPD':>9}")
    print("  " + "-" * 85)

    for var in VARIABLES:
        for escenario, mask_ex in [
            ('todos',            None),
            ('sin_palancas',     mask_pal),
            ('sin_influyentes',  mask_inf),
        ]:
            m = riemann_predicho_vs_real(df, var, mask_excluir=mask_ex)
            m['escenario'] = escenario
            filas.append(m)
            print(f"  {var:<22} {escenario:<18} "
                  f"{m['fisher_rao']:>11.4f} {m['hellinger']:>10.4f} "
                  f"{m['ks']:>8.4f} {m['spd_riemann']:>9.4f}")

        # Delta: cuanto mejora al quitar influyentes
        fr_todos = filas[-3]['fisher_rao']
        fr_sin   = filas[-1]['fisher_rao']
        if not (np.isnan(fr_todos) or np.isnan(fr_sin)):
            delta = fr_todos - fr_sin
            if abs(delta) > 0.01:
                print(f"  {'':22} {'-> delta':18} {delta:>+11.4f}"
                      f"  {'palancas inflan la dist.' if delta > 0 else 'palancas la reducen'}")
        print()

    return pd.DataFrame(filas)


# ─────────────────────────────────────────────────────────────────
# VISUALIZACION
# ─────────────────────────────────────────────────────────────────

def _interpretacion_fr(fr: float) -> str:
    if fr < 0.05:  return "Identicas"
    if fr < 0.15:  return "Muy similar"
    if fr < 0.30:  return "Similar"
    if fr < 0.50:  return "Diferente"
    return "Muy diferente"


def graficar(df: pd.DataFrame,
             h_diag: np.ndarray,
             mask_pal: np.ndarray,
             mask_inf: np.ndarray,
             df_res: pd.DataFrame) -> None:
    """
    Figura 3x2 con 6 paneles:

    Panel 1 — Distribucion de leverage h_ii:
      Muestra cuantos puntos son palanca (h_ii > 2p/n).
      Estos puntos tienen ubicaciones atipicas en el espacio de features
      y pueden sesgar la comparacion Riemann si se incluyen.

    Panel 2 — Leverage vs probabilidad predicha:
      Scatter que muestra cuales puntos son influyentes (alta palanca + residuo).
      Puntos rojos = los que mas distorsionan el modelo.

    Paneles 3-6 — Una por variable espacial:
      Azul  = p(x): distribucion ponderada por y_proba (donde el modelo predice)
      Rojo  = q(x): distribucion ponderada por y_true  (donde realmente expandio)
      El solapamiento visual entre azul y rojo = similitud entre prediccion y realidad.
      Las lineas punteadas muestran la misma comparacion sin puntos influyentes.
      La anotacion muestra Fisher-Rao con interpretacion verbal de similitud.
    """
    mask_limpio = ~mask_inf
    n_obs  = len(df)
    n_feat = 14

    fig = plt.figure(figsize=(18, 20))
    fig.suptitle(
        "Similitud Espacial: Prediccion 2020 vs Realidad 2020\n"
        "Riemann (Simpson): p(x)=modelo ponderado por y_proba  vs  q(x)=real ponderado por y_true\n"
        "Solapamiento azul-rojo = similitud  |  Fisher-Rao = 0 (perfecto) hasta pi/2 (sin similitud)",
        fontsize=12, fontweight='bold', y=0.995
    )
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.32)

    # ── Panel 1: distribucion de leverage ────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    umbral_h = 2.0 * n_feat / n_obs
    ax.hist(h_diag, bins=80, color='#1565c0', alpha=0.7,
            edgecolor='white', linewidth=0.3)
    ax.axvline(umbral_h, color='#e53935', linewidth=2.5, linestyle='--',
               label=f'Umbral 2p/n = {umbral_h:.4f}')
    n_pal = mask_pal.sum()
    ax.set_xlabel('Leverage h_ii  (diagonal de la matriz hat H)')
    ax.set_ylabel('Frecuencia')
    ax.set_title(f'Distribucion de Leverage (Puntos Palanca)\n'
                 f'{n_pal:,} palancas = {n_pal/n_obs*100:.1f}% del total  '
                 f'| {mask_inf.sum():,} influyentes (Cook>4/n)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    # ── Panel 2: leverage vs y_proba ─────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    y_proba = df['y_proba'].values
    normales = ~mask_inf
    solo_pal = mask_pal & ~mask_inf

    ax.scatter(h_diag[normales], y_proba[normales],
               alpha=0.07, s=2, color='#90a4ae', label='Normal')
    ax.scatter(h_diag[solo_pal], y_proba[solo_pal],
               alpha=0.5, s=8, color='#ff9800',
               label=f'Solo palanca ({solo_pal.sum():,})')
    ax.scatter(h_diag[mask_inf], y_proba[mask_inf],
               alpha=0.8, s=12, color='#e53935',
               label=f'Influyente ({mask_inf.sum():,})')

    ax.axvline(umbral_h, color='#e53935', linewidth=1.5, linestyle='--', alpha=0.5)
    ax.axhline(0.5, color='gray', linewidth=1, linestyle=':', alpha=0.4)
    ax.set_xlabel('Leverage h_ii')
    ax.set_ylabel('Probabilidad predicha P(expande)')
    ax.set_title('Leverage vs Probabilidad Predicha\n'
                 'Rojo = influyentes (distorsionan la comparacion Riemann)')
    ax.legend(fontsize=9, markerscale=3)
    ax.grid(True, alpha=0.2)

    # ── Paneles 3-6: comparacion p(x) vs q(x) por variable ───────────────────
    axes_grid = [
        fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1]),
    ]

    for ax_var, var in zip(axes_grid, VARIABLES):
        label = LABELS[var]
        vals  = df[var].values

        x_min, x_max = vals.min(), vals.max()
        bins        = np.linspace(x_min, x_max, 70)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        # ── CON todos los puntos (linea solida, relleno) ──────────────────────
        h_pred_t, _ = np.histogram(vals, bins=bins,
                                    weights=df['y_proba'].values, density=False)
        h_real_t, _ = np.histogram(vals, bins=bins,
                                    weights=df['y_true'].values, density=False)
        if h_pred_t.sum() > 0:
            h_pred_t = h_pred_t / (h_pred_t.sum() * (bins[1]-bins[0]))
        if h_real_t.sum() > 0:
            h_real_t = h_real_t / (h_real_t.sum() * (bins[1]-bins[0]))

        ax_var.fill_between(bin_centers, h_pred_t, alpha=0.4,
                            color='#1565c0', label='Predicho — y_proba')
        ax_var.fill_between(bin_centers, h_real_t, alpha=0.4,
                            color='#c62828', label='Real 2020 — y_true')
        ax_var.plot(bin_centers, h_pred_t, color='#1565c0',
                    linewidth=1.8, alpha=0.9)
        ax_var.plot(bin_centers, h_real_t, color='#c62828',
                    linewidth=1.8, alpha=0.9)

        # ── SIN influyentes (linea punteada, sin relleno) ─────────────────────
        vals_l  = df.loc[mask_limpio, var].values
        wp_l    = df.loc[mask_limpio, 'y_proba'].values
        wq_l    = df.loc[mask_limpio, 'y_true'].values.astype(float)

        h_pred_l, _ = np.histogram(vals_l, bins=bins, weights=wp_l, density=False)
        h_real_l, _ = np.histogram(vals_l, bins=bins, weights=wq_l, density=False)
        if h_pred_l.sum() > 0:
            h_pred_l = h_pred_l / (h_pred_l.sum() * (bins[1]-bins[0]))
        if h_real_l.sum() > 0:
            h_real_l = h_real_l / (h_real_l.sum() * (bins[1]-bins[0]))

        ax_var.plot(bin_centers, h_pred_l, color='#1565c0',
                    linewidth=1.2, linestyle='--', alpha=0.7,
                    label='Predicho (sin influyentes)')
        ax_var.plot(bin_centers, h_real_l, color='#c62828',
                    linewidth=1.2, linestyle='--', alpha=0.7,
                    label='Real (sin influyentes)')

        # ── Anotacion Fisher-Rao con interpretacion ───────────────────────────
        fr_t = df_res.loc[(df_res['variable'] == var) &
                          (df_res['escenario'] == 'todos'), 'fisher_rao']
        fr_s = df_res.loc[(df_res['variable'] == var) &
                          (df_res['escenario'] == 'sin_influyentes'), 'fisher_rao']

        if len(fr_t) and len(fr_s):
            fr_tv = fr_t.values[0]
            fr_sv = fr_s.values[0]
            delta  = fr_tv - fr_sv
            interp = _interpretacion_fr(fr_sv)
            color_bx = ('#e8f5e9' if fr_sv < 0.15
                        else '#fff8e1' if fr_sv < 0.30
                        else '#ffebee')
            txt = (f"Fisher-Rao (todos)     = {fr_tv:.4f}\n"
                   f"Fisher-Rao (sin pal.)  = {fr_sv:.4f}\n"
                   f"Delta (impacto palanca)= {delta:+.4f}\n"
                   f"Similitud              : {interp}")
            ax_var.text(0.98, 0.97, txt,
                        transform=ax_var.transAxes,
                        fontsize=8.5, va='top', ha='right',
                        family='monospace',
                        bbox=dict(boxstyle='round,pad=0.45',
                                  facecolor=color_bx, edgecolor='#aaa',
                                  alpha=0.93))

        ax_var.set_xlabel(label, fontsize=10)
        ax_var.set_ylabel('Densidad ponderada', fontsize=9)
        ax_var.set_title(f'{label}\nSolido+relleno = todos  |  Punteado = sin influyentes',
                         fontsize=10)
        ax_var.legend(fontsize=8, ncol=2)
        ax_var.grid(True, alpha=0.2)

    ruta = str(OUT_FIGS / "riemann_similitud_2020.png")
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Figura guardada: {ruta}")


# ─────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  UrbanCast — Riemann: Prediccion vs Realidad 2020")
    print("=" * 60)

    print("\n-- Entrenando modelo --")
    df_datos, x_scaled = load_y_predecir()

    print("\n-- Calculando puntos palanca --")
    h_diag, cook_d, mask_pal, mask_inf = calcular_palancas(
        x_scaled, df_datos['y_proba'].values
    )
    df_datos['h_ii']   = h_diag
    df_datos['cook_d'] = cook_d
    n_obs, n_feat = len(df_datos), x_scaled.shape[1]
    print(f"  Puntos palanca  (h>2p/n): {mask_pal.sum():,}  ({mask_pal.mean()*100:.1f}%)")
    print(f"  Puntos influyentes (Cook): {mask_inf.sum():,}  ({mask_inf.mean()*100:.1f}%)")

    print("\n-- Similitud Riemann: p(x)=modelo  vs  q(x)=real --")
    df_res = correr_comparaciones(df_datos, mask_pal, mask_inf)

    # ── Resumen interpretado ──────────────────────────────────────────────────
    print("\n-- Resumen de similitud (sin puntos influyentes) --")
    sin_inf = df_res[df_res['escenario'] == 'sin_influyentes'].copy()
    for _, row in sin_inf.iterrows():
        interp = _interpretacion_fr(row['fisher_rao'])
        barra  = '|' * int((1 - row['fisher_rao'] / 1.5708) * 30)
        print(f"  {row['variable']:<22}  FR={row['fisher_rao']:.4f}  "
              f"[{barra:<30}]  {interp}")

    fr_global = sin_inf['fisher_rao'].mean()
    print(f"\n  Fisher-Rao promedio global: {fr_global:.4f}  "
          f"-> {_interpretacion_fr(fr_global)}")

    csv_out = str(OUT_RES / "riemann_resultados_2020.csv")
    df_res.to_csv(csv_out, index=False)
    print(f"  -> CSV guardado: {csv_out}")

    print("\n-- Generando figura --")
    graficar(df_datos, h_diag, mask_pal, mask_inf, df_res)

    print(f"\n{'='*60}")
    print(f"  outputs/figures/riemann_similitud_2020.png")
    print(f"  outputs/results/riemann_resultados_2020.csv")
    print(f"{'='*60}")
