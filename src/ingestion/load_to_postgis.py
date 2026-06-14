import rasterio
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import geopandas as gpd
from shapely.geometry import Point
from geoalchemy2 import Geometry
from pathlib import Path
import os

ROOT     = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"

# Conexion a PostGIS
DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/urbancast"
engine = create_engine(DATABASE_URL)

# Leer GeoTIFF y cargar a PostGIS
def tiff_a_postgis(ruta_tiff, year, if_exists='append'):
    print(f"Procesando {year}...")

    with rasterio.open(ruta_tiff) as src:
        data = src.read(1).astype(float)
        transform = src.transform
        crs = src.crs

        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan

    rows, cols = np.where(~np.isnan(data) & (data > 0))
    values = data[rows, cols]

    xs, ys = rasterio.transform.xy(transform, rows, cols)

    gdf = gpd.GeoDataFrame({
        'year': year,
        'built_surface': values,
        'geometry': [Point(x, y) for x, y in zip(xs, ys)]
    }, crs=crs)

    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    gdf.to_postgis(
        name='urban_expansion',
        con=engine,
        if_exists=if_exists,
        index=False,
        dtype={'geometry': Geometry('POINT', srid=4326)}
    )

    print(f"  {len(gdf)} puntos cargados para {year}")

# Correr para todos los años
def cargar_todos(carpeta):
    años = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020]
    primer_año = True

    for year in años:
        ruta = os.path.join(carpeta, f'built_{year}_medellin.tif')
        if os.path.exists(ruta):
            if_exists = 'replace' if primer_año else 'append'
            tiff_a_postgis(ruta, year, if_exists)
            primer_año = False
        else:
            print(f"No encontrado: {ruta}")

if __name__ == "__main__":
    # Los GeoTIFFs deben estar en data/raw/
    cargar_todos(str(DATA_RAW))
