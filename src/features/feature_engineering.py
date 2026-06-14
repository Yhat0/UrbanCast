import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from pathlib import Path

ROOT     = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data" / "processed" / "urban_features.csv"

# ─────────────────────────────────────────────────────────────────
# CONEXIÓN A LA BASE DE DATOS
# create_engine crea un "pool" de conexiones reutilizables a PostgreSQL.
# No abre la conexión aquí; solo la configura para usarla después.
# ─────────────────────────────────────────────────────────────────
DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/urbancast"
engine = create_engine(DATABASE_URL)

# Lista ordenada de los años que cargamos en la Fase 1.
# La usamos para construir los pares consecutivos (1975→1980, 1980→1985, ...).
AÑOS = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020]

# Coordenadas del centro de Medellín (Plaza Botero).
# Las usamos para calcular qué tan lejos está cada pixel del centro urbano,
# un feature importante: zonas periféricas tienden a expandirse más rápido.
LAT_CENTRO = 6.2442
LON_CENTRO = -75.5812


def load_data() -> pd.DataFrame:
    """
    Carga todos los puntos de la tabla urban_expansion desde PostGIS
    y extrae las coordenadas lat/lon de la columna de geometría.
    """
    # ST_X y ST_Y son funciones de PostGIS que extraen longitud y latitud
    # de una columna de tipo geometry. Sin ellas, obtendrías un objeto
    # binario (WKB) que pandas no puede leer directamente.
    query = text("""
        SELECT
            year,
            built_surface,
            ST_X(geometry) AS lon,
            ST_Y(geometry) AS lat
        FROM urban_expansion
        ORDER BY year, lat, lon
    """)

    # Abrimos la conexión solo dentro del bloque 'with' para que se cierre
    # automáticamente al terminar, evitando conexiones huérfanas.
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    print(f"Datos cargados: {len(df):,} filas | años: {sorted(df['year'].unique())}")
    return df


def distancia_km(lat, lon, lat_ref=LAT_CENTRO, lon_ref=LON_CENTRO) -> pd.Series:
    """
    Calcula la distancia aproximada en km entre cada punto y el centro de Medellín.

    Usamos la aproximación euclidiana en lugar de la fórmula de Haversine (esférica)
    porque el bounding box es pequeño (~33x33 km), donde el error de Haversine
    sería menor al 0.1%. Es más rápido y suficientemente preciso.

    1 grado de latitud ≈ 111 km siempre.
    1 grado de longitud ≈ 111 km * cos(latitud) — se achica hacia los polos.
    """
    lat_km = (lat - lat_ref) * 111.0
    lon_km = (lon - lon_ref) * 111.0 * np.cos(np.radians(lat_ref))

    # Teorema de Pitágoras: distancia = √(Δlat² + Δlon²)
    return np.sqrt(lat_km**2 + lon_km**2)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye el dataset de features para el modelo.

    Para cada par de años consecutivos (T → T+5), hace un outer join
    de los puntos de ambos años usando lat/lon como clave, y calcula:
    - cuánto cambió la superficie construida
    - si el pixel se expandió o no (variable objetivo binaria)
    - distancia al centro de Medellín
    """
    # zip(AÑOS[:-1], AÑOS[1:]) empareja cada año con el siguiente:
    # [(1975,1980), (1980,1985), ..., (2015,2020)] → 9 pares en total
    pares = list(zip(AÑOS[:-1], AÑOS[1:]))
    registros = []

    for año_t, año_t1 in pares:
        print(f"  Procesando {año_t} → {año_t1}...")

        # Filtramos los puntos del año T y los del año T+5 por separado,
        # renombrando built_surface para distinguirlos después del merge.
        df_t  = (df[df['year'] == año_t ]
                   [['lat', 'lon', 'built_surface']]
                   .rename(columns={'built_surface': 'bs_t'}))

        df_t1 = (df[df['year'] == año_t1]
                   [['lat', 'lon', 'built_surface']]
                   .rename(columns={'built_surface': 'bs_t1'}))

        # OUTER JOIN: une los dos DataFrames por (lat, lon).
        # 'outer' incluye TODOS los puntos de ambos años, incluso si no
        # aparecen en el otro. Esto captura tres casos:
        #   - Pixel solo en T:     bs_t1 → NaN (ya no existe o no tiene datos)
        #   - Pixel solo en T+5:   bs_t  → NaN (urbanización nueva desde cero)
        #   - Pixel en ambos años: comparación directa
        merged = df_t.merge(df_t1, on=['lat', 'lon'], how='outer')

        # Reemplazamos NaN con 0: si un pixel no aparece en un año,
        # asumimos que su superficie construida era 0 (vacío / sin datos).
        merged['bs_t']  = merged['bs_t'].fillna(0.0)
        merged['bs_t1'] = merged['bs_t1'].fillna(0.0)

        # Metadatos del período
        merged['year']      = año_t
        merged['year_next'] = año_t1

        # Delta: cuántos m² de superficie construida ganó o perdió el pixel.
        # Valor positivo = urbanización; negativo = demolición o error de datos.
        merged['delta_built'] = merged['bs_t1'] - merged['bs_t']

        # Variable objetivo binaria para el modelo:
        # 1 = el pixel ganó superficie construida entre T y T+5
        # 0 = no cambió o disminuyó
        merged['expanded'] = (merged['delta_built'] > 0).astype(int)

        # Feature espacial: distancia al centro de Medellín.
        # La intuición es que la expansión urbana sigue gradientes de distancia.
        merged['dist_centro_km'] = distancia_km(merged['lat'], merged['lon'])

        # Seleccionamos solo las columnas que necesita el modelo, en orden.
        registros.append(merged[['lat', 'lon', 'year', 'year_next',
                                  'bs_t', 'bs_t1', 'delta_built',
                                  'expanded', 'dist_centro_km']])

    # Apilamos los 9 DataFrames en uno solo.
    # reset_index(drop=True) recalcula los índices del 0 al N para evitar duplicados.
    features = pd.concat(registros, ignore_index=True)
    return features


def save_features(df: pd.DataFrame) -> None:
    """
    Guarda el dataset de features en una tabla nueva llamada 'urban_features'.
    if_exists='replace' borra y recrea la tabla si ya existe,
    útil para re-correr el script sin errores de duplicados.
    """
    df.to_sql('urban_features', engine, if_exists='replace', index=False)
    print(f"Tabla 'urban_features' guardada: {len(df):,} filas")


def print_stats(df: pd.DataFrame) -> None:
    """Imprime un resumen del dataset para validar que los datos tienen sentido."""
    print("\n=== Estadísticas del dataset ===")
    print(f"Total observaciones : {len(df):,}")
    print(f"Expansiones (y=1)   : {df['expanded'].sum():,}  ({df['expanded'].mean()*100:.1f}%)")
    print(f"Sin cambio  (y=0)   : {(df['expanded']==0).sum():,}  ({(df['expanded']==0).mean()*100:.1f}%)")

    print("\nDistribución por par de años:")
    resumen = (df.groupby(['year', 'year_next'])
                 .agg(
                     n_obs=('expanded', 'count'),
                     n_expand=('expanded', 'sum'),
                     pct_expand=('expanded', 'mean')
                 )
                 .reset_index())
    resumen['pct_expand'] = (resumen['pct_expand'] * 100).round(1)
    print(resumen.to_string(index=False))

    print(f"\nFeatures disponibles: {list(df.columns)}")


# ─────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# Este bloque solo se ejecuta cuando corres el script directamente
# (python feature_engineering.py), no cuando lo importas desde otro módulo.
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("--- Fase 2: Ingeniería de Features ---\n")

    df_raw = load_data()
    df_features = build_features(df_raw)
    print_stats(df_features)

    print("\nGuardando en PostGIS...")
    save_features(df_features)

    # Exportamos también a CSV como respaldo local.
    # data/processed/ versiona el CSV limpio; data/raw/ queda fuera del repo.
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(CSV_PATH, index=False)
    print(f"CSV exportado: {CSV_PATH}")

    print("\nListo. Próximo paso: entrenar modelo de regresión logística.")
