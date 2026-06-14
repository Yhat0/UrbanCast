"""
explore_visual.py
Visualización exploratoria de la expansión urbana de Medellín.

Genera:
  1. outputs/maps/mapa_expansion_medellin.html  → mapa interactivo con tiles ArcGIS/Esri
  2. outputs/figures/histogramas_por_año.png    → 9 histogramas (uno por par de años) con outliers marcados
  3. outputs/figures/histograma_global.png      → histograma acumulado de todos los períodos + KDE

Los histogramas preparan el análisis para:
  - Teorema de Gauss-Markov: verificar que los errores sean ~normales y homoscedásticos
  - Integración de Riemann: el área bajo la curva KDE aproxima la probabilidad acumulada
"""

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import folium
from folium.plugins import HeatMap
from pathlib import Path
from sqlalchemy import create_engine, text

warnings.filterwarnings('ignore')

ROOT     = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data" / "processed" / "urban_features.csv"
OUT_MAPS = ROOT / "outputs" / "maps"
OUT_FIGS = ROOT / "outputs" / "figures"
OUT_MAPS.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/urbancast"
engine = create_engine(DATABASE_URL)

LAT_CENTRO = 6.2442
LON_CENTRO = -75.5812

# Los 9 pares de años consecutivos que produjo feature_engineering.py
PARES_AÑOS = [
    (1975, 1980), (1980, 1985), (1985, 1990),
    (1990, 1995), (1995, 2000), (2000, 2005),
    (2005, 2010), (2010, 2015), (2015, 2020),
]

# Los 10 años individuales — se usan para el slider temporal del mapa
AÑOS_TODOS = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020]

# URLs de tiles públicos de ArcGIS/Esri — no requieren cuenta ni API key.
ARCGIS_SATELITE    = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
ARCGIS_CALLES      = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}'
ARCGIS_TOPOGRAFICO = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}'

sns.set_theme(style='darkgrid', palette='muted')


# ─────────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """
    Intenta cargar urban_features desde PostGIS.
    Si la base no está disponible, cae a urban_features.csv como respaldo.
    """
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text("SELECT * FROM urban_features"), conn)
        print(f"Datos cargados desde PostGIS: {len(df):,} filas")
    except Exception as e:
        print(f"PostGIS no disponible ({e})\nCargando desde CSV...")
        df = pd.read_csv(CSV_PATH)
        print(f"Datos cargados desde CSV: {len(df):,} filas")
    return df


# ─────────────────────────────────────────────────────────────────
# DETECCIÓN DE ATÍPICOS (IQR)
# ─────────────────────────────────────────────────────────────────

def detectar_atipicos_iqr(serie: pd.Series):
    """
    Detecta valores atípicos usando el Rango Intercuartílico (IQR).

    Lógica:
      - Q1 = percentil 25, Q3 = percentil 75
      - IQR = Q3 - Q1  (rango donde vive el 50% central de los datos)
      - Límite inferior = Q1 - 1.5 * IQR
      - Límite superior = Q3 + 1.5 * IQR
      - Todo lo que quede fuera de esos límites es atípico

    Este es el mismo criterio que usa un boxplot estándar.
    Usamos 1.5×IQR y no z-score porque IQR no asume distribución normal,
    lo cual es importante cuando todavía no sabemos si los datos son normales.

    Retorna:
      mascara       → Serie booleana, True donde el dato es atípico
      limite_inf    → valor del límite inferior
      limite_sup    → valor del límite superior
    """
    Q1 = serie.quantile(0.25)
    Q3 = serie.quantile(0.75)
    IQR = Q3 - Q1
    limite_inf = Q1 - 1.5 * IQR
    limite_sup = Q3 + 1.5 * IQR
    mascara = (serie < limite_inf) | (serie > limite_sup)
    return mascara, limite_inf, limite_sup


# ─────────────────────────────────────────────────────────────────
# MAPA INTERACTIVO (FOLIUM + TILES ARCGIS/ESRI)
# ─────────────────────────────────────────────────────────────────

def _preparar_heat_por_año(df: pd.DataFrame):
    """
    Para cada uno de los 10 años, extrae los pixeles con superficie construida
    y los normaliza para que el peso esté en [0, 1] con escala global consistente.

    Estrategia de extracción:
      - Años 1975–2015: columna bs_t del par (año, año+5).
        Cada año aparece como T en exactamente un par, sin duplicados.
      - Año 2020: solo aparece como year_next en el par (2015→2020),
        se toma de bs_t1.

    Retorna:
      heat_data → lista de 10 listas [[lat, lon, peso], ...], una por año
    """
    # Máximo global: garantiza que la escala de color sea consistente entre años
    max_bs = max(df['bs_t'].max(), df['bs_t1'].max())

    heat_data = []
    for año in AÑOS_TODOS:
        if año < 2020:
            subset = (df[df['year'] == año][['lat', 'lon', 'bs_t']]
                        .rename(columns={'bs_t': 'bs'}))
        else:
            # 2020 solo existe como year_next en el último par
            subset = (df[df['year_next'] == 2020][['lat', 'lon', 'bs_t1']]
                        .rename(columns={'bs_t1': 'bs'}))

        subset = (subset[subset['bs'] > 0]
                  .drop_duplicates(subset=['lat', 'lon'])
                  .copy())

        subset['peso'] = subset['bs'] / max_bs
        heat_data.append(subset[['lat', 'lon', 'peso']].values.tolist())
        print(f"    {año}: {len(subset):,} pixeles")

    return heat_data


def crear_mapa(df: pd.DataFrame) -> None:
    """
    Crea un mapa interactivo de Medellín con slider temporal año por año.

    Estrategia: un FeatureGroup (capa) por año con su propio HeatMap.
    Un slider HTML/JS custom controla qué capa es visible.

    Controles del mapa:
      - Slider inferior: arrastrá para cambiar de año (1975 → 2020)
      - Botón Play/Pausa: animación automática
      - Selector de velocidad: Lento / Normal / Rápido
      - Panel superior derecho: cambiá entre los tiles base de ArcGIS
    """
    m = folium.Map(
        location=[LAT_CENTRO, LON_CENTRO],
        zoom_start=12,
        tiles=None
    )

    # ── Capas base ArcGIS (van al LayerControl) ────────────────────
    for nombre, url, attr in [
        ('Satelite (ArcGIS)',    ARCGIS_SATELITE,    'Esri World Imagery'),
        ('Calles (ArcGIS)',      ARCGIS_CALLES,      'Esri World Street Map'),
        ('Topografico (ArcGIS)', ARCGIS_TOPOGRAFICO, 'Esri World Topo Map'),
    ]:
        folium.TileLayer(tiles=url, attr=attr, name=nombre,
                         overlay=False, control=True).add_to(m)

    # ── Un FeatureGroup por año ────────────────────────────────────
    heat_data = _preparar_heat_por_año(df)

    fg_vars = {}   # { año: 'feature_group_abc123' }

    for año, datos in zip(AÑOS_TODOS, heat_data):
        fg = folium.FeatureGroup(
            name=str(año),
            show=True,
            control=False
        )
        HeatMap(
            datos,
            radius=8,
            blur=10,
            min_opacity=0.4,
            gradient={0.2: 'blue', 0.5: 'lime', 0.8: 'yellow', 1.0: 'red'}
        ).add_to(fg)
        fg.add_to(m)
        fg_vars[año] = fg.get_name()

    # Marcador del centro como referencia visual
    folium.Marker(
        location=[LAT_CENTRO, LON_CENTRO],
        popup='Centro de Medellin',
        icon=folium.Icon(color='red', icon='home')
    ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # ── Slider temporal (HTML + JS inyectado en el HTML del mapa) ──
    map_var  = f"map_{m._id}"
    years_js = str(AÑOS_TODOS)

    fgs_js = '{' + ', '.join(f'"{a}": {v}' for a, v in fg_vars.items()) + '}'

    slider_html = f"""
    <div id="yr-ctrl" style="
        position:fixed; bottom:28px; left:50%; transform:translateX(-50%);
        z-index:9999; background:rgba(255,255,255,0.96);
        padding:16px 30px; border-radius:14px;
        box-shadow:0 4px 18px rgba(0,0,0,0.28);
        font-family:Arial,sans-serif; min-width:480px; text-align:center;">
      <div style="font-size:18px; font-weight:bold; color:#2c3e50; margin-bottom:10px;">
        Medell&iacute;n &mdash; Superficie Construida &bull;
        <span id="yr-label" style="color:#e74c3c; font-size:22px;">1975</span>
      </div>
      <input type="range" id="yr-range" min="0" max="9" value="0" step="1"
             style="width:100%; accent-color:#e74c3c; cursor:pointer; height:6px;">
      <div style="display:flex; justify-content:space-between;
                  font-size:10px; color:#95a5a6; margin-top:5px;">
        <span>1975</span><span>1980</span><span>1985</span><span>1990</span><span>1995</span>
        <span>2000</span><span>2005</span><span>2010</span><span>2015</span><span>2020</span>
      </div>
      <div style="margin-top:12px; display:flex; align-items:center; justify-content:center; gap:12px;">
        <button id="yr-play" style="
            padding:7px 24px; background:#e74c3c; color:#fff;
            border:none; border-radius:7px; cursor:pointer;
            font-size:14px; font-weight:bold; letter-spacing:.5px;">
          &#9654; Play
        </button>
        <label style="font-size:12px; color:#555;">Velocidad:</label>
        <select id="yr-speed" style="padding:5px 10px; border-radius:6px; border:1px solid #ccc; font-size:13px;">
          <option value="1500">Lenta</option>
          <option value="900" selected>Normal</option>
          <option value="400">R&aacute;pida</option>
        </select>
      </div>
    </div>

    <script>
    (function waitForMap() {{
      if (typeof {map_var} === 'undefined') {{
        setTimeout(waitForMap, 80);
        return;
      }}

      var MAP   = {map_var};
      var YEARS = {years_js};
      var fgs   = {fgs_js};

      var curr   = 0;
      var active = false;
      var timer  = null;

      function showYear(idx) {{
        curr = idx;
        var yr = String(YEARS[idx]);
        document.getElementById('yr-label').textContent = yr;
        document.getElementById('yr-range').value = idx;

        Object.keys(fgs).forEach(function(y) {{
          if (y === yr) {{
            if (!MAP.hasLayer(fgs[y])) MAP.addLayer(fgs[y]);
          }} else {{
            if (MAP.hasLayer(fgs[y]))  MAP.removeLayer(fgs[y]);
          }}
        }});
      }}

      document.getElementById('yr-range').addEventListener('input', function() {{
        showYear(parseInt(this.value));
      }});

      document.getElementById('yr-play').addEventListener('click', function() {{
        if (active) {{
          clearInterval(timer);
          active = false;
          this.innerHTML = '&#9654; Play';
        }} else {{
          active = true;
          this.innerHTML = '&#9646;&#9646; Pausa';
          var spd = parseInt(document.getElementById('yr-speed').value);
          timer = setInterval(function() {{
            curr = (curr + 1) % YEARS.length;
            showYear(curr);
          }}, spd);
        }}
      }});

      showYear(0);
    }})();
    </script>
    """

    m.get_root().html.add_child(folium.Element(slider_html))

    ruta = str(OUT_MAPS / "mapa_expansion_medellin.html")
    m.save(ruta)
    print(f"  -> Mapa guardado: {ruta}  (abrilo en el navegador)")


# ─────────────────────────────────────────────────────────────────
# HISTOGRAMAS POR AÑO
# ─────────────────────────────────────────────────────────────────

def histogramas_por_año(df: pd.DataFrame) -> None:
    """
    Genera una grilla 3×3 con un histograma por cada par de años.

    Cada histograma muestra la distribución de delta_built (cambio de
    superficie construida) con:
      - Datos normales en azul
      - Outliers IQR en rojo
      - Líneas verticales para los límites IQR y la mediana
    """
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle(
        'Distribución del cambio en superficie construida por período\n'
        'Medellín 1975–2020  |  delta_built = bs_t1 − bs_t  |  Outliers por IQR',
        fontsize=15, fontweight='bold', y=1.01
    )

    axes_flat = axes.flatten()
    resumen   = []

    for i, (año_t, año_t1) in enumerate(PARES_AÑOS):
        ax = axes_flat[i]

        subset = df[(df['year'] == año_t) & (df['delta_built'] != 0)]['delta_built']

        if subset.empty:
            ax.set_title(f'{año_t}→{año_t1}\n(sin datos)', fontsize=11)
            continue

        mascara, lim_inf, lim_sup = detectar_atipicos_iqr(subset)
        n_outliers  = mascara.sum()
        pct_outliers = n_outliers / len(subset) * 100

        resumen.append({
            'período'     : f'{año_t}→{año_t1}',
            'n_pixeles'   : len(subset),
            'n_outliers'  : n_outliers,
            'pct_outliers': round(pct_outliers, 2),
            'media'       : round(subset.mean(), 1),
            'mediana'     : round(subset.median(), 1),
            'desvio'      : round(subset.std(), 1),
        })

        normales  = subset[~mascara]
        atipicos  = subset[mascara]

        ax.hist(normales, bins=60, color='steelblue', alpha=0.75,
                edgecolor='white', linewidth=0.4, label='Normal')

        if not atipicos.empty:
            ax.hist(atipicos, bins=30, color='red', alpha=0.65,
                    edgecolor='white', linewidth=0.4,
                    label=f'Atípicos: {n_outliers:,}')

        ax.axvline(lim_inf, color='darkorange', linestyle='--', linewidth=1.5,
                   label=f'LI: {lim_inf:,.0f}')
        ax.axvline(lim_sup, color='darkorange', linestyle='--', linewidth=1.5,
                   label=f'LS: {lim_sup:,.0f}')
        ax.axvline(subset.median(), color='limegreen', linestyle='-', linewidth=2,
                   label=f'Mediana: {subset.median():,.0f}')

        ax.set_title(f'{año_t} → {año_t1}\n{pct_outliers:.1f}% atípicos | n={len(subset):,}',
                     fontsize=11)
        ax.set_xlabel('Δ Superficie construida (m²)', fontsize=9)
        ax.set_ylabel('Frecuencia', fontsize=9)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
        ax.legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    ruta = str(OUT_FIGS / "histogramas_por_año.png")
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Histogramas por año guardados: {ruta}")

    df_res = pd.DataFrame(resumen)
    print("\n  === Atípicos por período (IQR) ===")
    print(df_res.to_string(index=False))

    peor = df_res.loc[df_res['pct_outliers'].idxmax()]
    print(f"\n  Período con más atípicos: {peor['período']}  "
          f"({peor['pct_outliers']}% — {peor['n_outliers']:,} pixeles)")


# ─────────────────────────────────────────────────────────────────
# HISTOGRAMA GLOBAL (todos los años juntos)
# ─────────────────────────────────────────────────────────────────

def histograma_global(df: pd.DataFrame) -> None:
    """
    Genera un histograma único con delta_built de todos los períodos.

    Panel izquierdo  → distribución completa (con outliers): muestra el rango total
    Panel derecho    → distribución sin outliers + curva KDE: muestra la forma real

    La curva KDE (Kernel Density Estimation) es una estimación suavizada de la
    distribución de probabilidad. Si se parece a una campana simétrica (normal),
    los supuestos de Gauss-Markov tienen más chances de cumplirse.
    """
    datos = df[df['delta_built'] != 0]['delta_built']
    mascara, lim_inf, lim_sup = detectar_atipicos_iqr(datos)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        'Distribución global de Δ superficie construida — todos los períodos 1975–2020\n'
        f'N total: {len(datos):,}  |  N outliers IQR: {mascara.sum():,} ({mascara.mean()*100:.2f}%)',
        fontsize=14, fontweight='bold'
    )

    # ── Panel izquierdo: distribución COMPLETA ─────────────────────
    ax1 = axes[0]
    ax1.hist(datos, bins=150, color='royalblue', alpha=0.7,
             edgecolor='white', linewidth=0.2)
    ax1.axvline(lim_inf, color='red', linestyle='--', linewidth=2,
                label=f'LI IQR: {lim_inf:,.0f}')
    ax1.axvline(lim_sup, color='red', linestyle='--', linewidth=2,
                label=f'LS IQR: {lim_sup:,.0f}')
    ax1.axvline(datos.mean(), color='limegreen', linestyle='-', linewidth=2,
                label=f'Media: {datos.mean():,.0f}')
    ax1.set_title('Distribución completa (con outliers)')
    ax1.set_xlabel('Δ Superficie construida (m²)')
    ax1.set_ylabel('Frecuencia')
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax1.legend()

    # ── Panel derecho: SIN outliers + KDE ─────────────────────────
    ax2 = axes[1]
    normales = datos[~mascara]

    ax2.hist(normales, bins=100, color='royalblue', alpha=0.7,
             edgecolor='white', linewidth=0.2, density=True,
             label='Histograma (densidad)')

    # KDE manual con numpy puro (evita la dependencia de scipy).
    # Regla de Silverman: bw = 1.06 * σ * n^(-1/5)
    muestra = normales.sample(min(15_000, len(normales)), random_state=42).values
    bw = 1.06 * muestra.std() * len(muestra) ** (-0.2)

    x_kde = np.linspace(normales.min(), normales.max(), 300)

    # diff[i, j] = distancia entre el i-ésimo punto del eje X y el j-ésimo dato
    diff = x_kde[:, None] - muestra[None, :]
    kde_vals = np.mean(np.exp(-0.5 * (diff / bw) ** 2), axis=1) / (bw * np.sqrt(2 * np.pi))

    ax2.plot(x_kde, kde_vals, color='darkorange', linewidth=2.5,
             label='KDE (densidad estimada)')

    ax2.axvline(normales.mean(), color='limegreen', linestyle='-', linewidth=2,
                label=f'Media: {normales.mean():,.0f}')
    ax2.axvline(normales.median(), color='violet', linestyle='--', linewidth=2,
                label=f'Mediana: {normales.median():,.0f}')

    stats_txt = (
        f'Skewness : {normales.skew():.3f}\n'
        f'Curtosis  : {normales.kurtosis():.3f}\n'
        f'Std Dev   : {normales.std():,.1f}'
    )
    ax2.text(0.02, 0.95, stats_txt, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax2.set_title('Sin outliers + KDE  (relevante para Gauss-Markov)')
    ax2.set_xlabel('Δ Superficie construida (m²)')
    ax2.set_ylabel('Densidad')
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax2.legend()

    plt.tight_layout()
    ruta = str(OUT_FIGS / "histograma_global.png")
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Histograma global guardado: {ruta}")

    print("\n  === Estadísticos globales de delta_built ===")
    print(f"  N total          : {len(datos):,}")
    print(f"  N outliers IQR   : {mascara.sum():,}  ({mascara.mean()*100:.2f}%)")
    print(f"  Media            : {datos.mean():.2f}")
    print(f"  Mediana          : {datos.median():.2f}")
    print(f"  Desv. estándar   : {datos.std():.2f}")
    print(f"  Asimetría (skew) : {datos.skew():.4f}  (0 = normal simétrica)")
    print(f"  Curtosis         : {datos.kurtosis():.4f}  (0 = normal; >0 = colas pesadas)")


# ─────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  UrbanCast — Exploración Visual")
    print("=" * 55)

    df = load_data()

    print("\n[1/3] Generando mapa interactivo con tiles ArcGIS...")
    crear_mapa(df)

    print("\n[2/3] Generando histogramas por par de años...")
    histogramas_por_año(df)

    print("\n[3/3] Generando histograma global con KDE...")
    histograma_global(df)

    print("\n" + "=" * 55)
    print("Archivos generados:")
    print("  outputs/maps/mapa_expansion_medellin.html  ← abrir en navegador")
    print("  outputs/figures/histogramas_por_año.png")
    print("  outputs/figures/histograma_global.png")
    print("=" * 55)
