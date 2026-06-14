"""
gauss_markov_spd.py
Verifica los supuestos del Teorema de Gauss-Markov sobre el modelo OLS de
urban_features y detecta la distribución de probabilidad de delta_built y
los residuos mediante integración de Riemann (adaptado de spd.py).

Dependencias: pip install scipy statsmodels scikit-learn

Outputs (outputs/figures/ y outputs/results/):
  gauss_markov_supuestos.png        → 6 gráficos diagnósticos GM
  dist_delta_built_global.png       → PDF + CDF ajustada, delta_built todos los años
  dist_delta_built_2015_2020.png    → ídem, solo período más atípico (15.4% outliers)
  dist_delta_built_sin_outliers.png → ídem, delta_built depurado de outliers IQR
  dist_residuos_ols.png             → ídem, residuos del modelo lineal
  resultados_distribuciones.csv     → tabla de distancias y mejor ajuste por campo
"""

import os
import sys
import warnings
import numpy as np

# Forzar UTF-8 en la salida de consola (necesario en Windows con cp1252)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.integrate import simpson
from sklearn.linear_model import LinearRegression
from sqlalchemy import create_engine, text

warnings.filterwarnings('ignore')

ROOT     = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data" / "processed" / "urban_features.csv"
OUT_FIGS = ROOT / "outputs" / "figures"
OUT_RES  = ROOT / "outputs" / "results"
OUT_FIGS.mkdir(parents=True, exist_ok=True)
OUT_RES.mkdir(parents=True, exist_ok=True)

# -- Conexión a PostGIS ---------------------------------------------
DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/urbancast"
engine       = create_engine(DATABASE_URL)

# -- Distribuciones candidatas --------------------------------------
# Todas admiten valores negativos via el parámetro loc.
# Ordenadas de menor a mayor complejidad de cola.
DISTRIBUCIONES = [
    stats.norm,      # Normal/Gaussiana — hipótesis base de Gauss-Markov
    stats.laplace,   # Doble exponencial — colas más pesadas, simétrica
    stats.logistic,  # Logística — similar a normal, colas un poco más pesadas
    stats.skewnorm,  # Normal asimétrica — captura sesgo
    stats.t,         # t de Student — colas pesadas, robusta a outliers
    stats.gennorm,   # Normal generalizada — β=1→Laplace, β=2→Normal
    stats.cauchy,    # Cauchy — colas muy pesadas, sin media ni varianza definida
]

# -- Features del modelo OLS ----------------------------------------
# Variable dependiente: delta_built (cambio de superficie construida)
# Regressores: superficie en T, posición, distancia al centro, año
FEATURES = ['bs_t', 'lat', 'lon', 'dist_centro_km', 'year']


# -----------------------------------------------------------------
# CARGA DE DATOS
# -----------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """Carga urban_features desde PostGIS; cae a CSV si no hay conexión."""
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text("SELECT * FROM urban_features"), conn)
        print(f"Datos desde PostGIS: {len(df):,} filas")
    except Exception as e:
        print(f"PostGIS no disponible ({e}) — usando CSV")
        df = pd.read_csv(CSV_PATH)
        print(f"Datos desde CSV: {len(df):,} filas")
    return df


# -----------------------------------------------------------------
# SECCIÓN 1: TEOREMA DE GAUSS-MARKOV
# -----------------------------------------------------------------

def _vif(X: np.ndarray, j: int) -> float:
    """
    Variance Inflation Factor (VIF) para el regresor j.
    VIF(j) = 1 / (1 - R²_j)
    donde R²_j es el R² de regresar la columna j sobre las demás.

    Interpretación:
      VIF < 5  → sin multicolinealidad
      VIF 5-10 → moderada (advertencia)
      VIF > 10 → severa (viola GM2)
    """
    y       = X[:, j]
    X_otros = np.delete(X, j, axis=1)
    r2      = LinearRegression().fit(X_otros, y).score(X_otros, y)
    return 1.0 / (1.0 - r2) if r2 < 0.9999 else float('inf')


def _durbin_watson(residuos: np.ndarray) -> float:
    """
    Estadístico de Durbin-Watson para autocorrelación de primer orden:
      DW = Σ(εᵢ - εᵢ₋₁)² / Σεᵢ²

    Interpretación:
      DW ≈ 2.0          → sin autocorrelación    (GM4b OK)
      DW < 1.5           → autocorrelación positiva
      DW > 2.5           → autocorrelación negativa
    """
    diff = np.diff(residuos)
    return np.sum(diff**2) / np.sum(residuos**2)


def _breusch_pagan(residuos: np.ndarray, X: np.ndarray) -> tuple:
    """
    Test de Breusch-Pagan para homoscedasticidad (GM4a).
    H0: Var(ε|X) = σ² constante (homoscedástico) ← queremos NO rechazar esto
    H1: Var(ε|X) = f(X) (heteroscedástico)

    Método:
      1. Regresar ε² sobre X
      2. LM = n · R²   ~  χ²(k) bajo H0
    Si p < 0.05 → rechazamos H0 → hay heteroscedasticidad → viola GM4a.
    """
    n   = len(residuos)
    e2  = residuos ** 2
    r2  = LinearRegression().fit(X, e2).score(X, e2)
    lm  = n * r2
    p   = 1.0 - stats.chi2.cdf(lm, df=X.shape[1])
    return lm, p


def ajustar_ols(df: pd.DataFrame) -> tuple:
    """
    Ajusta el modelo OLS:
      delta_built ~ bs_t + lat + lon + dist_centro_km + year

    Filtramos delta_built == 0 porque no son cambios reales:
    son pixeles que no variaron entre períodos y contaminarían los residuos.

    Submuestra a 100k filas si el dataset es mayor (performance sin
    afectar la estimación de coeficientes en un dataset tan grande).

    Retorna: (X, y, coeficientes, intercepto, ŷ, residuos)
    """
    subset = df[df['delta_built'] != 0].copy()
    if len(subset) > 100_000:
        subset = subset.sample(100_000, random_state=42)

    X     = subset[FEATURES].values
    y     = subset['delta_built'].values
    model = LinearRegression().fit(X, y)
    y_hat = model.predict(X)
    res   = y - y_hat

    r2 = model.score(X, y)
    print(f"\n  Modelo OLS ajustado")
    print(f"  n = {len(y):,}  |  R² = {r2:.4f}")
    print(f"  Intercepto: {model.intercept_:.4f}")
    for nombre, coef in zip(FEATURES, model.coef_):
        print(f"  {nombre:20s}: {coef:.6f}")

    return X, y, model.coef_, model.intercept_, y_hat, res


def verificar_gm(X: np.ndarray, y_hat: np.ndarray,
                 res: np.ndarray) -> None:
    """
    Verifica los 5 supuestos del Teorema de Gauss-Markov e imprime
    los resultados numéricos. Genera una figura de 2×3 subgráficos.

    GM1 — Linealidad del modelo
        Diagnóstico: gráfico residuos vs ŷ (debe ser nube sin patrón).
        Si hay curvatura → el modelo es no lineal en los parámetros.

    GM2 — No multicolinealidad perfecta
        Diagnóstico: VIF de cada regresor.
        VIF > 10 → el regresor es casi combinación lineal de los demás.

    GM3 — Media condicional cero: E[ε|X] = 0
        Diagnóstico: t-test de que la media de los residuos = 0.
        También visible en el gráfico residuos vs ŷ (línea roja en 0).

    GM4a — Homoscedasticidad: Var(ε|X) = σ² constante
        Diagnóstico: test de Breusch-Pagan + gráfico √|ε| vs ŷ.
        Si la dispersión crece con ŷ → heteroscedasticidad.

    GM4b — Sin autocorrelación: Cov(εᵢ, εⱼ|X) = 0
        Diagnóstico: estadístico de Durbin-Watson.
        DW ≈ 2 → errores no correlacionados en el tiempo.

    Normalidad (bonus, no es supuesto GM pero necesaria para t/F tests):
        Diagnóstico: QQ plot + Shapiro-Wilk.
    """
    print("\n====== Verificación Teorema de Gauss-Markov ======")

    # GM3: E[ε] = 0
    t_stat, p_media = stats.ttest_1samp(res, 0)
    ok_gm3 = p_media > 0.05
    print(f"\n  GM3  Media de residuos   : {res.mean():.6f}"
          f"  t={t_stat:.4f}  p={p_media:.4f}"
          f"  {'OK' if ok_gm3 else 'VIOLA (E[e] != 0)'}")

    # GM4b: Durbin-Watson
    dw = _durbin_watson(res)
    ok_dw = 1.5 < dw < 2.5
    print(f"  GM4b Durbin-Watson       : {dw:.4f}"
          f"  {'Sin autocorrelacion' if ok_dw else 'Posible autocorrelacion'}")

    # GM4a: Breusch-Pagan
    lm_bp, p_bp = _breusch_pagan(res, X)
    ok_bp = p_bp > 0.05
    print(f"  GM4a Breusch-Pagan       : LM={lm_bp:.2f}  p={p_bp:.6f}"
          f"  {'Homoscedastico' if ok_bp else 'Heteroscedasticidad detectada'}")

    # GM2: VIF
    vifs = [_vif(X, j) for j in range(X.shape[1])]
    print("  GM2  VIF por regresor:")
    for nombre, v in zip(FEATURES, vifs):
        ok_v = v < 10
        print(f"       {nombre:20s}: {v:7.2f}"
              f"  {'OK' if ok_v else 'Multicolinealidad'}")

    # Normalidad (bonus)
    muestra_sw = res[np.random.choice(len(res), min(5000, len(res)), replace=False)]
    sw_stat, p_sw = stats.shapiro(muestra_sw)
    print(f"  Normalidad (Shapiro-Wilk): W={sw_stat:.4f}  p={p_sw:.6f}"
          f"  {'~ Normal' if p_sw > 0.05 else 'No normal (comun en datasets grandes)'}")

    print("\n  RESUMEN:")
    print(f"  {'Supuesto':<35} {'Estado'}")
    print(f"  {'-'*50}")
    print(f"  {'GM1 Linealidad':<35} {'Ver grafico residuos vs y_hat'}")
    print(f"  {'GM2 No multicolinealidad (VIF<10)':<35} {'OK' if all(v < 10 for v in vifs) else 'VIOLA'}")
    print(f"  {'GM3 E[e|X] = 0':<35} {'OK' if ok_gm3 else 'VIOLA'}")
    print(f"  {'GM4a Homoscedasticidad (BP)':<35} {'OK' if ok_bp else 'VIOLA'}")
    print(f"  {'GM4b Sin autocorrelacion (DW)':<35} {'OK' if ok_dw else 'VIOLA'}")

    # -- Figura diagnóstica -----------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(19, 12))
    fig.suptitle(
        "Diagnostico Gauss-Markov — UrbanCast\n"
        "Modelo: delta_built ~ bs_t + lat + lon + dist_centro_km + year",
        fontsize=14, fontweight='bold'
    )

    # Panel 1 — GM1 + GM3: Residuos vs Ajustados
    ax = axes[0, 0]
    idx = np.random.choice(len(res), min(10_000, len(res)), replace=False)
    ax.scatter(y_hat[idx], res[idx], alpha=0.08, s=2, color='steelblue')
    ax.axhline(0, color='red', linewidth=2, linestyle='--', label='e = 0')
    ax.set_xlabel('Valores ajustados (y_hat)')
    ax.set_ylabel('Residuos (e)')
    ax.set_title(f'GM1 + GM3: Residuos vs Ajustados\nMedia e = {res.mean():.4f}  p={p_media:.4f}')
    ax.legend()

    # Panel 2 — GM4a: Scale-Location (homoscedasticidad)
    ax = axes[0, 1]
    ax.scatter(y_hat[idx], np.sqrt(np.abs(res[idx])),
               alpha=0.08, s=2, color='darkorange')
    ax.set_xlabel('Valores ajustados (y_hat)')
    ax.set_ylabel('sqrt(|Residuos|)')
    ax.set_title(f'GM4a: Scale-Location (Homoscedasticidad)\nBreusch-Pagan p = {p_bp:.4f}'
                 f'  {"OK" if ok_bp else "VIOLA"}')

    # Panel 3 — Normalidad: QQ plot de residuos
    ax = axes[0, 2]
    stats.probplot(res[idx], dist='norm', plot=ax)
    ax.get_lines()[0].set(markersize=1, alpha=0.4)
    ax.get_lines()[1].set_color('red')
    ax.set_title(f'Normalidad de Residuos (Q-Q)\nShapiro-Wilk p = {p_sw:.4f}')

    # Panel 4 — GM2: VIF bar chart
    ax = axes[1, 0]
    colores = ['#27ae60' if v < 5 else '#f39c12' if v < 10 else '#e74c3c'
               for v in vifs]
    bars = ax.barh(FEATURES, vifs, color=colores, alpha=0.85)
    ax.axvline(5,  color='#f39c12', linestyle='--', linewidth=1.5, label='VIF = 5')
    ax.axvline(10, color='#e74c3c', linestyle='--', linewidth=1.5, label='VIF = 10 (umbral)')
    for bar, v in zip(bars, vifs):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f'{v:.1f}', va='center', fontsize=10)
    ax.set_xlabel('VIF')
    ax.set_title('GM2: Multicolinealidad (VIF)\nverde<5 OK | naranja<10 alerta | rojo>10 viola')
    ax.legend()

    # Panel 5 — Normalidad: Histograma de residuos vs curva normal teórica
    ax = axes[1, 1]
    ax.hist(res, bins=120, density=True, color='steelblue',
            alpha=0.65, edgecolor='white', linewidth=0.3,
            label='Residuos')
    x_rng = np.linspace(res.min(), res.max(), 400)
    ax.plot(x_rng, stats.norm.pdf(x_rng, res.mean(), res.std()),
            'r-', linewidth=2.5, label='Normal teorica')
    ax.set_xlabel('Residuos (e)')
    ax.set_ylabel('Densidad')
    ax.set_title('Distribucion de Residuos\nvs Normal Teorica (Gauss-Markov)')
    ax.legend()

    # Panel 6 — GM4b: Residuos al cuadrado (autocorrelación)
    ax = axes[1, 2]
    ax.scatter(np.arange(len(idx)), res[idx]**2,
               alpha=0.15, s=2, color='purple')
    ax.set_xlabel('Indice de observacion (muestra aleatoria)')
    ax.set_ylabel('Residuos^2')
    ax.set_title(f'GM4b: Autocorrelacion\nDurbin-Watson = {dw:.4f}'
                 f'  {"OK" if ok_dw else "VIOLA"}'
                 f'  (2.0 = ideal)')

    plt.tight_layout()
    ruta = str(OUT_FIGS / 'gauss_markov_supuestos.png')
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  -> Figura GM guardada: {ruta}")


# -----------------------------------------------------------------
# SECCIÓN 2: RIEMANN / DETECCIÓN DE DISTRIBUCIÓN
# -----------------------------------------------------------------

def _distancia_spd(var_emp: float, var_fit: float) -> float:
    """
    Distancia Riemanniana entre dos matrices SPD(2) diagonales.
    Equivalente a spd.metric.dist() de geomstats para esta estructura:

      A = [[var_emp, 0], [0, 1]]
      B = [[var_fit, 0], [0, 1]]

    Para matrices diagonales, la distancia geodésica en SPD(n) se reduce a:
      d(A, B) = ||log(A^{-1/2} B A^{-1/2})||_F
              = sqrt( log(var_fit/var_emp)^2 + log(1)^2 )
              = |log(var_fit / var_emp)|

    Resultado: 0 si las varianzas son iguales; crece logarítmicamente con
    la diferencia relativa de varianzas.
    """
    if var_emp <= 0 or var_fit <= 0:
        return float('nan')
    return abs(np.log(var_fit / var_emp))


def ajustar_distribuciones(valores: np.ndarray, nombre: str) -> pd.DataFrame:
    """
    Ajusta todas las distribuciones candidatas a 'valores' y calcula
    las métricas de distancia de spd.py mediante integración de Riemann
    (regla de Simpson compuesta vía scipy.integrate.simpson).

    Métricas calculadas (de menor a mayor tolerancia al error):
      Fisher-Rao   : arccos(ρ_B) en [0, π/2] — métrica geodésica en la variedad
                     estadística (M, g_Fisher). Es la métrica principal de selección.
      Kolmogorov-Smirnov : max|F_emp - F_fit| — diferencia máxima de CDFs acumuladas.
      Hellinger    : sqrt(0.5 · ∫(√p - √q)² dx) — raíz de semidivergencia de Hellinger.
      L1           : ∫|p - q| dx — distancia L1 entre densidades normalizadas.
      Bhattacharyya: -log(∫√(pq) dx) — relacionada con overlap de distribuciones.
      SPD Riemann  : distancia Riemanniana entre matrices de covarianza en SPD(2).

    Selección del mejor ajuste (igual que spd.py):
      Primero filtra FR < π/12 (~0.26 rad) Y KS < 0.1 Y Hellinger < 0.1 Y L1 < 0.1.
      Si nadie pasa, ordena todos por Fisher-Rao.
    """
    print(f"\n  -- [{nombre}]  n={len(valores):,} --")

    # Histograma de densidad (100 bins, misma base que spd.py)
    hist_vals, bin_edges = np.histogram(valores, bins=100, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    dx          = bin_centers[1] - bin_centers[0]   # ancho uniforme de bin

    # Varianza empírica mediante integración Riemann (no np.var)
    # Var(X) = ∫ p(x)·(x − μ)² dx ≈ Σ p(xᵢ)·(xᵢ − μ)²·Δx
    mu_emp  = np.sum(hist_vals * bin_centers) * dx
    var_emp = max(np.sum(hist_vals * (bin_centers - mu_emp)**2) * dx, 1e-12)

    # Para el fit usamos una muestra si hay muchos datos (el histograma
    # ya captura la forma; el ajuste no necesita todos los puntos)
    vals_fit = (np.random.default_rng(42).choice(valores, 50_000, replace=False)
                if len(valores) > 50_000 else valores)

    resultados = []

    for dist in DISTRIBUCIONES:
        try:
            params   = dist.fit(vals_fit)
            pdf_vals = dist.pdf(bin_centers, *params)

            # Normalizamos histograma y PDF a densidades que integran a 1
            # usando Simpson (Riemann compuesto de orden 4)
            area_hist = simpson(hist_vals, x=bin_centers)
            area_pdf  = simpson(pdf_vals,  x=bin_centers)
            if area_hist <= 0 or area_pdf <= 0:
                continue

            h = hist_vals / area_hist   # densidad empírica normalizada
            p = pdf_vals  / area_pdf    # densidad teórica normalizada

            # -- L1 ----------------------------------------------
            l1 = simpson(np.abs(h - p), x=bin_centers)

            # -- Hellinger ----------------------------------------
            # d_H = sqrt(0.5 · ∫(√h - √p)²)
            hellinger = np.sqrt(0.5 * simpson(
                (np.sqrt(np.maximum(h, 0)) - np.sqrt(np.maximum(p, 0)))**2,
                x=bin_centers
            ))

            # -- Bhattacharyya -------------------------------------
            # ρ_B = ∫ √(h·p) dx  (coeficiente de Bhattacharyya)
            # d_B = -log(ρ_B)   (distancia de Bhattacharyya)
            bc = np.clip(simpson(np.sqrt(np.maximum(h * p, 0)), x=bin_centers),
                         1e-8, 1.0)
            bhatta = -np.log(bc)

            # -- Fisher-Rao ---------------------------------------
            # Distancia geodésica en la variedad estadística (M, g_Fisher).
            # d_FR = arccos(ρ_B) en [0, π/2]
            fisher_rao = np.arccos(bc)

            # -- Kolmogorov-Smirnov -------------------------------
            cdf_emp = np.clip(np.cumsum(h * dx), 0.0, 1.0)
            cdf_fit = np.clip(np.cumsum(p * dx), 0.0, 1.0)
            ks_D    = np.max(np.abs(cdf_emp - cdf_fit))

            # -- SPD Riemanniana ------------------------------------
            mu_fit  = np.sum(p * bin_centers) * dx
            var_fit = max(np.sum(p * (bin_centers - mu_fit)**2) * dx, 1e-12)
            spd_d   = _distancia_spd(var_emp, var_fit)

            resultados.append({
                'campo'       : nombre,
                'distribucion': dist.name,
                'params'      : str(tuple(round(float(q), 5) for q in params)),
                'fisher_rao'  : round(fisher_rao, 6),
                'ks'          : round(ks_D,        6),
                'hellinger'   : round(hellinger,   6),
                'l1'          : round(l1,           6),
                'bhatta'      : round(bhatta,       6),
                'spd_riemann' : round(spd_d,        6),
                '_params_raw' : params,   # sin redondear, para graficar
            })

            print(f"    {dist.name:12s}  FR={fisher_rao:.4f}  "
                  f"KS={ks_D:.4f}  H={hellinger:.4f}  "
                  f"L1={l1:.4f}  SPD={spd_d:.4f}")

        except Exception as e:
            print(f"    {dist.name:12s}  ! {e}")

    if not resultados:
        print("    Sin resultados.")
        return pd.DataFrame()

    df_res = pd.DataFrame(resultados)

    # Selección del mejor ajuste (criterio de spd.py)
    filtrado = df_res[
        (df_res['fisher_rao'] < 0.26) &
        (df_res['ks']         < 0.10) &
        (df_res['hellinger']  < 0.10) &
        (df_res['l1']         < 0.10)
    ]
    df_ord = (filtrado if not filtrado.empty else df_res).sort_values('fisher_rao')
    mejor  = df_ord.iloc[0]

    print(f"\n  Mejor ajuste [{nombre}]: {mejor['distribucion'].upper()}"
          f"  FR={mejor['fisher_rao']}  KS={mejor['ks']}"
          f"  SPD={mejor['spd_riemann']}")

    # Generar gráficos del mejor ajuste
    dist_obj = next(d for d in DISTRIBUCIONES if d.name == mejor['distribucion'])
    _graficar_ajuste(valores, dist_obj, mejor['_params_raw'],
                     hist_vals, bin_centers, bin_edges, mejor, nombre)

    # Eliminar columna auxiliar antes de retornar
    return df_res.drop(columns=['_params_raw'])


def _graficar_ajuste(valores: np.ndarray, dist_obj, params: tuple,
                     hist_vals: np.ndarray, bin_centers: np.ndarray,
                     bin_edges: np.ndarray, mejor: pd.Series,
                     nombre: str) -> None:
    """
    Genera dos paneles para el mejor ajuste de distribución:

    Panel izquierdo — PDF ajustada + histograma + zonas sombreadas de outliers IQR:
      La zona naranja (derecha) = pixeles con delta_built > Q3+1.5·IQR
      La zona morada (izquierda) = pixeles con delta_built < Q1-1.5·IQR

    Panel derecho — CDF empírica vs CDF teórica:
      Si las dos curvas se superponen → buen ajuste.
      La brecha máxima es el estadístico KS.
    """
    # Límites IQR para zonas de outlier
    Q1, Q3 = np.percentile(valores, 25), np.percentile(valores, 75)
    iqr     = Q3 - Q1
    lim_sup = Q3 + 1.5 * iqr
    lim_inf = Q1 - 1.5 * iqr

    # Probabilidad teórica de outlier (área bajo la PDF fuera de los límites)
    try:
        p_sup = 1.0 - dist_obj.cdf(lim_sup, *params)
        p_inf = dist_obj.cdf(lim_inf, *params)
        p_out = round((p_sup + p_inf) * 100, 2)
    except Exception:
        p_out = float('nan')

    x_plot = np.linspace(valores.min(), valores.max(), 1000)
    try:
        pdf_plot = dist_obj.pdf(x_plot, *params)
    except Exception:
        pdf_plot = np.zeros_like(x_plot)

    fig, axes = plt.subplots(1, 2, figsize=(17, 6))
    fig.suptitle(
        f"Campo: {nombre}  →  Mejor distribucion: {mejor['distribucion'].upper()}\n"
        f"Fisher-Rao={mejor['fisher_rao']}  KS={mejor['ks']}  "
        f"Hellinger={mejor['hellinger']}  SPD Riemann={mejor['spd_riemann']}",
        fontsize=12, fontweight='bold'
    )

    # -- Panel 1: Histograma + PDF + zonas de outlier ---------------
    ax1 = axes[0]
    ax1.bar(bin_centers, hist_vals, width=np.diff(bin_edges),
            alpha=0.5, color='steelblue', align='center',
            label='Histograma (densidad)')
    ax1.plot(x_plot, pdf_plot, 'k-', linewidth=2.5,
             label=f'PDF {mejor["distribucion"]}')

    # Zona de outliers superiores (expansión extrema)
    m_sup = x_plot >= lim_sup
    if m_sup.any():
        ax1.fill_between(x_plot[m_sup], pdf_plot[m_sup],
                         alpha=0.45, color='#ff6b00',
                         label=f'Outlier sup > {lim_sup:,.0f} m2')

    # Zona de outliers inferiores (declive extremo)
    m_inf = x_plot <= lim_inf
    if m_inf.any():
        ax1.fill_between(x_plot[m_inf], pdf_plot[m_inf],
                         alpha=0.45, color='#9b59b6',
                         label=f'Outlier inf < {lim_inf:,.0f} m2')

    ax1.axvline(lim_sup, color='#ff6b00', linestyle='--', linewidth=1.5)
    ax1.axvline(lim_inf, color='#9b59b6', linestyle='--', linewidth=1.5)
    ax1.set_xlabel('Delta Superficie construida (m2)')
    ax1.set_ylabel('Densidad de probabilidad')
    ax1.set_title(f'PDF + zonas de atipicos IQR\n'
                  f'P(outlier) teorica = {p_out}%')
    ax1.legend(fontsize=9)

    # -- Panel 2: CDF empírica vs CDF teórica ----------------------
    ax2 = axes[1]
    sorted_vals = np.sort(valores)
    ecdf_y      = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    try:
        cdf_teo = dist_obj.cdf(sorted_vals, *params)
    except Exception:
        cdf_teo = np.zeros_like(sorted_vals)

    ax2.step(sorted_vals, ecdf_y,  where='post',
             color='steelblue', linewidth=2, label='CDF empirica')
    ax2.plot(sorted_vals, cdf_teo,
             'k-', linewidth=2.5, label=f'CDF {mejor["distribucion"]}')
    ax2.axvline(lim_sup, color='#ff6b00', linestyle='--', linewidth=1.5,
                label=f'LS IQR ({lim_sup:,.0f})')
    ax2.axvline(lim_inf, color='#9b59b6', linestyle='--', linewidth=1.5,
                label=f'LI IQR ({lim_inf:,.0f})')
    ax2.set_xlabel('Delta Superficie construida (m2)')
    ax2.set_ylabel('Probabilidad acumulada F(x)')
    ax2.set_title('CDF empirica vs CDF teorica\n'
                  '(brecha maxima = KS · cuanto mas se superponen = mejor)')
    ax2.legend(fontsize=9)

    plt.tight_layout()
    slug = nombre.replace(' ', '_').replace('->', '_').replace('/', '_')
    ruta = str(OUT_FIGS / f'dist_{slug}.png')
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Grafico guardado: {ruta}")


def analizar_distribuciones(df: pd.DataFrame,
                             residuos: np.ndarray) -> pd.DataFrame:
    """
    Corre la detección de distribución (Riemann/SPD) para 4 series:

      1. delta_built global (todos los períodos, sin ceros)
      2. delta_built período 2015→2020 (el más atípico: 15.4% outliers)
      3. delta_built sin outliers IQR
      4. Residuos del modelo OLS
    """
    todos = []

    # 1. Global
    vals_global = df[df['delta_built'] != 0]['delta_built'].values
    todos.append(ajustar_distribuciones(vals_global, 'delta_built_global'))

    # 2. Período 2015→2020 (más atípico)
    vals_2015 = df[(df['year'] == 2015) & (df['delta_built'] != 0)]['delta_built'].values
    if len(vals_2015) > 0:
        todos.append(ajustar_distribuciones(vals_2015, 'delta_built_2015_2020'))

    # 3. Sin outliers IQR (la distribución "central")
    Q1  = np.percentile(vals_global, 25)
    Q3  = np.percentile(vals_global, 75)
    IQR = Q3 - Q1
    sin_out = vals_global[
        (vals_global >= Q1 - 1.5*IQR) & (vals_global <= Q3 + 1.5*IQR)
    ]
    todos.append(ajustar_distribuciones(sin_out, 'delta_built_sin_outliers'))

    # 4. Residuos OLS
    todos.append(ajustar_distribuciones(residuos, 'residuos_ols'))

    return pd.concat([t for t in todos if not t.empty], ignore_index=True)


# -----------------------------------------------------------------
# PUNTO DE ENTRADA
# -----------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("  UrbanCast — Gauss-Markov + Riemann SPD")
    print("=" * 55)

    df = load_data()

    # -- Sección 1: Gauss-Markov ----------------------------------
    print("\n-- Sección 1: OLS + Verificación Gauss-Markov --")
    X, y, coef, intercept, y_hat, residuos = ajustar_ols(df)
    verificar_gm(X, y_hat, residuos)

    # -- Sección 2: Riemann / Distribuciones ----------------------
    print("\n-- Sección 2: Detección de Distribución (Riemann/SPD) --")
    df_resultados = analizar_distribuciones(df, residuos)

    # Guardar CSV con todas las métricas de distancia
    csv_path = str(OUT_RES / 'resultados_distribuciones.csv')
    df_resultados.to_csv(csv_path, index=False)

    print(f"\n{'='*55}")
    print(f"  Archivos generados en:")
    print(f"  outputs/figures/gauss_markov_supuestos.png")
    print(f"  outputs/figures/dist_delta_built_global.png")
    print(f"  outputs/figures/dist_delta_built_2015_2020.png")
    print(f"  outputs/figures/dist_delta_built_sin_outliers.png")
    print(f"  outputs/figures/dist_residuos_ols.png")
    print(f"  outputs/results/resultados_distribuciones.csv")
    print(f"{'='*55}")
