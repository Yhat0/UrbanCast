"""
mapa_temporal.py
Genera outputs/maps/mapa_expansion_medellin.html con:
  - Heatmap de superficie construida por año (slider 1975-2020)
  - Capa de outliers IQR sobre delta_built:
      · Naranja = expansión extrema (delta > Q3 + 1.5·IQR)
      · Morado  = declive extremo   (delta < Q1 - 1.5·IQR)
  - Toggle para activar/desactivar outliers
  - Panel de estadísticos por año (n outliers, %, límites IQR)
  - Tiles base intercambiables de ArcGIS (satélite / calles / topográfico)
"""

import json
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text

ROOT     = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data" / "processed" / "urban_features.csv"
OUT_MAPS = ROOT / "outputs" / "maps"
OUT_MAPS.mkdir(parents=True, exist_ok=True)

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/urbancast"
engine       = create_engine(DATABASE_URL)

AÑOS       = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020]
LAT_CENTRO = 6.2442
LON_CENTRO = -75.5812

# Máximo de puntos por año para el heatmap principal.
# ~35k puntos/año; 20k mantiene buen detalle sin superar 6MB de HTML.
MAX_PUNTOS_HEAT = 20_000


def load_data() -> pd.DataFrame:
    """Carga urban_features desde PostGIS; cae a CSV si la DB no está disponible."""
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text("SELECT * FROM urban_features"), conn)
        print(f"Datos desde PostGIS: {len(df):,} filas")
    except Exception as e:
        print(f"PostGIS no disponible ({e}) — usando CSV")
        df = pd.read_csv(CSV_PATH)
        print(f"Datos desde CSV: {len(df):,} filas")
    return df


def iqr_outliers(serie: pd.Series):
    """
    Detecta outliers usando el Rango Intercuartílico (IQR).
    Retorna (mascara_bool, limite_inferior, limite_superior).
    """
    Q1  = serie.quantile(0.25)
    Q3  = serie.quantile(0.75)
    IQR = Q3 - Q1
    li  = Q1 - 1.5 * IQR
    ls  = Q3 + 1.5 * IQR
    return (serie < li) | (serie > ls), li, ls


def preparar_datos(df: pd.DataFrame) -> tuple[dict, dict]:
    """
    Prepara dos diccionarios indexados por año (string):

    heat_data[año]  →  [[lat, lon, intensidad], ...]
        Datos para L.heatLayer. Intensidad normalizada a [0,1] con escala global.

    outliers[año]   →  {
        'pos':     [[lat, lon], ...]   # delta > límite superior IQR (expansión extrema)
        'neg':     [[lat, lon], ...]   # delta < límite inferior IQR (declive extremo)
        'n_pos':   int,
        'n_neg':   int,
        'n_total': int,                # total pixeles del período (excluyendo delta=0)
        'pct':     float,              # % de outliers sobre el total
        'li':      float,              # límite inferior IQR de delta_built
        'ls':      float,              # límite superior IQR de delta_built
    }

    Para outliers, cada año T usa el período (T → T+5).
    El año 2020 (sin período siguiente) reutiliza el período 2015→2020.
    """
    # Máximo global de superficie construida → escala de color consistente entre años
    max_bs = max(df['bs_t'].max(), df['bs_t1'].max())

    heat_data = {}
    outliers  = {}

    for año in AÑOS:
        # ── Heatmap: superficie construida en el año T ─────────────
        if año < 2020:
            sub_heat = (df[df['year'] == año][['lat', 'lon', 'bs_t']]
                        .rename(columns={'bs_t': 'bs'}))
        else:
            sub_heat = (df[df['year_next'] == 2020][['lat', 'lon', 'bs_t1']]
                        .rename(columns={'bs_t1': 'bs'}))

        sub_heat = (sub_heat[sub_heat['bs'] > 0]
                    .drop_duplicates(subset=['lat', 'lon'])
                    .copy())

        if len(sub_heat) > MAX_PUNTOS_HEAT:
            sub_heat = sub_heat.sample(MAX_PUNTOS_HEAT, random_state=42)

        sub_heat['peso'] = (sub_heat['bs'] / max_bs).round(4)
        sub_heat['lat']  = sub_heat['lat'].round(4)
        sub_heat['lon']  = sub_heat['lon'].round(4)
        heat_data[str(año)] = sub_heat[['lat', 'lon', 'peso']].values.tolist()

        # ── Outliers: delta_built del período que inicia en año T ──
        # Para 2020 no hay período siguiente; usamos el período 2015→2020
        # porque es el que explica cómo llegó el territorio al estado de 2020.
        if año < 2020:
            periodo = df[df['year'] == año].copy()
        else:
            periodo = df[df['year_next'] == 2020].copy()

        # Excluimos delta=0 (sin cambio) para no contaminar la distribución IQR
        delta_nz = periodo[periodo['delta_built'] != 0]['delta_built']

        if delta_nz.empty:
            outliers[str(año)] = {
                'pos': [], 'neg': [], 'n_pos': 0, 'n_neg': 0,
                'n_total': 0, 'pct': 0.0, 'li': 0.0, 'ls': 0.0
            }
            continue

        mascara, li, ls = iqr_outliers(delta_nz)

        # Filtramos los outliers en el DataFrame original del período
        # (que sí tiene lat/lon) usando el índice compartido
        idx_outlier    = delta_nz[mascara].index
        outlier_rows   = periodo.loc[periodo.index.intersection(idx_outlier)]

        pos_rows = outlier_rows[outlier_rows['delta_built'] > ls][['lat', 'lon']].round(4)
        neg_rows = outlier_rows[outlier_rows['delta_built'] < li][['lat', 'lon']].round(4)

        n_total = len(delta_nz)
        n_out   = len(outlier_rows)

        outliers[str(año)] = {
            'pos':     pos_rows.values.tolist(),
            'neg':     neg_rows.values.tolist(),
            'n_pos':   len(pos_rows),
            'n_neg':   len(neg_rows),
            'n_total': n_total,
            'pct':     round(n_out / max(n_total, 1) * 100, 1),
            'li':      round(float(li), 1),
            'ls':      round(float(ls), 1),
        }

        print(f"  {año}: {len(sub_heat):,} pts heatmap | "
              f"outliers: {len(pos_rows):,} pos, {len(neg_rows):,} neg "
              f"({outliers[str(año)]['pct']}%)")

    return heat_data, outliers


def generar_html(heat_data: dict, outliers: dict,
                 output: str = None) -> None:
    """Escribe el HTML completo con Leaflet + leaflet-heat, datos embebidos como JSON."""
    if output is None:
        output = str(OUT_MAPS / "mapa_expansion_medellin.html")

    heat_json     = json.dumps(heat_data, separators=(',', ':'))
    outliers_json = json.dumps(outliers,  separators=(',', ':'))
    años_json     = json.dumps([str(a) for a in AÑOS])

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>UrbanCast — Expansión Urbana Medellín</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    body {{ background:#1a1a2e; font-family:Arial,sans-serif; }}
    #map {{ width:100vw; height:100vh; }}

    /* ── Slider temporal ── */
    #ctrl {{
      position:fixed; bottom:24px; left:50%; transform:translateX(-50%);
      z-index:9999; background:rgba(255,255,255,0.96);
      padding:16px 30px; border-radius:14px;
      box-shadow:0 4px 18px rgba(0,0,0,0.35);
      min-width:520px; text-align:center;
    }}
    #ctrl h2 {{ font-size:17px; color:#2c3e50; margin-bottom:10px; font-weight:bold; }}
    #yr-label {{ color:#e74c3c; font-size:24px; font-weight:bold; }}
    #yr-range {{
      width:100%; accent-color:#e74c3c; cursor:pointer;
      height:6px; margin:8px 0 4px;
    }}
    .ticks {{
      display:flex; justify-content:space-between;
      font-size:10px; color:#95a5a6; margin-bottom:10px;
    }}
    #btns {{
      display:flex; align-items:center; justify-content:center;
      gap:10px; flex-wrap:wrap; margin-bottom:10px;
    }}
    .btn {{
      padding:7px 18px; border:none; border-radius:7px;
      cursor:pointer; font-size:13px; font-weight:bold;
    }}
    #yr-play   {{ background:#e74c3c; color:#fff; }}
    #yr-play:hover {{ background:#c0392b; }}
    #btn-outliers {{ background:#f39c12; color:#fff; }}
    #btn-outliers.activo {{ background:#27ae60; }}
    #yr-speed {{
      padding:5px 10px; border-radius:6px;
      border:1px solid #ccc; font-size:13px;
    }}

    /* ── Panel de estadísticos de outliers ── */
    #stats-panel {{
      background:#f8f9fa; border-radius:8px; padding:8px 14px;
      font-size:12px; color:#2c3e50; text-align:left;
      border-left:4px solid #f39c12; display:none;
    }}
    #stats-panel .row {{
      display:flex; justify-content:space-between; gap:16px;
      flex-wrap:wrap;
    }}
    #stats-panel span {{ white-space:nowrap; }}
    .dot-pos {{ color:#ff6b00; font-weight:bold; }}
    .dot-neg {{ color:#9b59b6; font-weight:bold; }}

    /* ── Leyenda de outliers ── */
    #leyenda {{
      position:fixed; bottom:24px; right:14px; z-index:9999;
      background:rgba(255,255,255,0.95); padding:10px 14px;
      border-radius:10px; box-shadow:0 2px 10px rgba(0,0,0,0.25);
      font-size:12px; color:#2c3e50; display:none;
    }}
    #leyenda b {{ display:block; margin-bottom:6px; }}
    .leg-item {{ display:flex; align-items:center; gap:8px; margin-bottom:4px; }}
    .dot {{ width:10px; height:10px; border-radius:50%; display:inline-block; }}

    /* ── Selector de mapa base ── */
    #basemap {{
      position:fixed; top:12px; right:12px; z-index:9999;
      background:rgba(255,255,255,0.95); padding:10px 14px;
      border-radius:10px; box-shadow:0 2px 10px rgba(0,0,0,0.25); font-size:13px;
    }}
    #basemap b {{ display:block; margin-bottom:6px; color:#2c3e50; }}
    #tile-sel {{
      width:100%; padding:5px 8px; border-radius:5px;
      border:1px solid #ccc; font-size:13px;
    }}
  </style>
</head>
<body>

<div id="map"></div>

<!-- Selector de mapa base -->
<div id="basemap">
  <b>Mapa base (ArcGIS)</b>
  <select id="tile-sel" onchange="cambiarBase(this.value)">
    <option value="sat">Satelite</option>
    <option value="str">Calles</option>
    <option value="top">Topografico</option>
  </select>
</div>

<!-- Leyenda de outliers -->
<div id="leyenda">
  <b>Atípicos (IQR)</b>
  <div class="leg-item"><span class="dot" style="background:#ff6b00"></span> Expansión extrema</div>
  <div class="leg-item"><span class="dot" style="background:#9b59b6"></span> Declive extremo</div>
</div>

<!-- Slider y controles -->
<div id="ctrl">
  <h2>Medell&iacute;n &mdash; Superficie Construida &bull; <span id="yr-label">1975</span></h2>
  <input type="range" id="yr-range" min="0" max="9" value="0" step="1">
  <div class="ticks">
    <span>1975</span><span>1980</span><span>1985</span><span>1990</span><span>1995</span>
    <span>2000</span><span>2005</span><span>2010</span><span>2015</span><span>2020</span>
  </div>
  <div id="btns">
    <button class="btn" id="yr-play">&#9654; Play</button>
    <label style="font-size:12px;color:#555;">Velocidad:</label>
    <select id="yr-speed">
      <option value="1500">Lenta</option>
      <option value="900" selected>Normal</option>
      <option value="400">R&aacute;pida</option>
    </select>
    <button class="btn" id="btn-outliers" onclick="toggleOutliers()">
      &#9679; Mostrar At&iacute;picos
    </button>
  </div>
  <!-- Panel de estadísticos (visible solo cuando outliers están activos) -->
  <div id="stats-panel">
    <div class="row">
      <span><b>At&iacute;picos:</b> <span id="s-total">—</span> (<span id="s-pct">—</span>%)</span>
      <span class="dot-pos">&#9679; Expansi&oacute;n extrema: <span id="s-pos">—</span></span>
      <span class="dot-neg">&#9679; Declive extremo: <span id="s-neg">—</span></span>
      <span><b>L&iacute;mites IQR:</b> [<span id="s-li">—</span>, <span id="s-ls">—</span>] m&sup2;</span>
    </div>
  </div>
</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>

<script>
// ── Tiles ArcGIS ───────────────────────────────────────────────────
var TILES = {{
  sat: L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
    {{ attribution: 'Esri World Imagery', maxZoom: 19 }}),
  str: L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{{z}}/{{y}}/{{x}}',
    {{ attribution: 'Esri World Street Map', maxZoom: 19 }}),
  top: L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{{z}}/{{y}}/{{x}}',
    {{ attribution: 'Esri World Topo Map', maxZoom: 19 }})
}};

var map = L.map('map', {{ center: [{LAT_CENTRO}, {LON_CENTRO}], zoom: 12 }});
var baseActiva = 'sat';
TILES.sat.addTo(map);

L.marker([{LAT_CENTRO}, {LON_CENTRO}])
  .bindPopup('<b>Centro de Medell&iacute;n</b>')
  .addTo(map);

function cambiarBase(tipo) {{
  map.removeLayer(TILES[baseActiva]);
  TILES[tipo].addTo(map);
  baseActiva = tipo;
}}

// ── Datos embebidos ────────────────────────────────────────────────
var DATOS = {heat_json};
var OUTL  = {outliers_json};
var YEARS = {años_json};

// ── Estado de la aplicación ────────────────────────────────────────
var heatLayer    = null;
var outlierLayer = null;
var verOutliers  = false;
var curr         = 0;
var active       = false;
var timer        = null;

// ── Función principal: mostrar un año ─────────────────────────────
function mostrarAño(idx) {{
  curr = idx;
  var yr = YEARS[idx];

  document.getElementById('yr-label').textContent = yr;
  document.getElementById('yr-range').value = idx;

  if (heatLayer) {{ map.removeLayer(heatLayer); heatLayer = null; }}
  heatLayer = L.heatLayer(DATOS[yr], {{
    radius:     10,
    blur:       12,
    minOpacity: 0.4,
    gradient: {{ 0.2: 'blue', 0.5: 'lime', 0.8: 'yellow', 1.0: 'red' }}
  }});
  heatLayer.addTo(map);

  if (verOutliers) actualizarOutliers(yr);
  actualizarStats(yr);
}}

// ── Capa de outliers ───────────────────────────────────────────────
function actualizarOutliers(yr) {{
  if (outlierLayer) {{ map.removeLayer(outlierLayer); outlierLayer = null; }}

  var data = OUTL[yr];
  if (!data) return;

  var grupo = L.layerGroup();

  data.pos.forEach(function(p) {{
    L.circleMarker([p[0], p[1]], {{
      radius:      4,
      fillColor:   '#ff6b00',
      color:       '#cc5500',
      weight:      1,
      fillOpacity: 0.85
    }}).bindTooltip('Expansi&oacute;n extrema').addTo(grupo);
  }});

  data.neg.forEach(function(p) {{
    L.circleMarker([p[0], p[1]], {{
      radius:      4,
      fillColor:   '#9b59b6',
      color:       '#6c3483',
      weight:      1,
      fillOpacity: 0.85
    }}).bindTooltip('Declive extremo').addTo(grupo);
  }});

  outlierLayer = grupo;
  map.addLayer(outlierLayer);
}}

// ── Panel de estadísticos ──────────────────────────────────────────
function actualizarStats(yr) {{
  var data = OUTL[yr];
  if (!data || !verOutliers) return;

  document.getElementById('s-total').textContent =
    (data.n_pos + data.n_neg).toLocaleString();
  document.getElementById('s-pct').textContent   = data.pct;
  document.getElementById('s-pos').textContent   = data.n_pos.toLocaleString();
  document.getElementById('s-neg').textContent   = data.n_neg.toLocaleString();
  document.getElementById('s-li').textContent    = data.li.toLocaleString();
  document.getElementById('s-ls').textContent    = data.ls.toLocaleString();
}}

// ── Toggle outliers ────────────────────────────────────────────────
function toggleOutliers() {{
  verOutliers = !verOutliers;
  var btn = document.getElementById('btn-outliers');
  var panel = document.getElementById('stats-panel');
  var leyenda = document.getElementById('leyenda');

  if (verOutliers) {{
    btn.classList.add('activo');
    btn.innerHTML = '&#10003; Ocultar At&iacute;picos';
    panel.style.display   = 'block';
    leyenda.style.display = 'block';
    actualizarOutliers(YEARS[curr]);
    actualizarStats(YEARS[curr]);
  }} else {{
    btn.classList.remove('activo');
    btn.innerHTML = '&#9679; Mostrar At&iacute;picos';
    panel.style.display   = 'none';
    leyenda.style.display = 'none';
    if (outlierLayer) {{ map.removeLayer(outlierLayer); outlierLayer = null; }}
  }}
}}

// ── Slider y Play ─────────────────────────────────────────────────
document.getElementById('yr-range').addEventListener('input', function () {{
  mostrarAño(parseInt(this.value));
}});

document.getElementById('yr-play').addEventListener('click', function () {{
  if (active) {{
    clearInterval(timer); active = false;
    this.innerHTML = '&#9654; Play';
  }} else {{
    active = true;
    this.innerHTML = '&#9646;&#9646; Pausa';
    var spd = parseInt(document.getElementById('yr-speed').value);
    timer = setInterval(function () {{
      curr = (curr + 1) % YEARS.length;
      mostrarAño(curr);
    }}, spd);
  }}
}});

mostrarAño(0);
</script>
</body>
</html>"""

    with open(output, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\nMapa guardado: {output}  — abrilo en el navegador")


if __name__ == "__main__":
    print("=== UrbanCast — Mapa Temporal con Outliers ===\n")
    df = load_data()
    print("\nPreparando heatmap y outliers por año...")
    heat_data, outliers = preparar_datos(df)
    print("\nGenerando HTML...")
    generar_html(heat_data, outliers)
