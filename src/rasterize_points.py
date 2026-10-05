"""Turn field-survey points (CSV: lon,lat,label) into a sparse label raster (255 = unlabeled).

Labels: 0 = healthy, 1 = water stress, 2 = crop anomaly.
Tip: create the points in QGIS (Layer > Add Delimited Text) and export as CSV.
Then:  python -m src.train --raster data/raw/scene_after.tif --labels data/raw/labels.tif
"""
import argparse
import numpy as np
import pandas as pd
import rasterio
from config import DATA_DIR


def main(raster, csv, out, radius=1):
    df = pd.read_csv(csv)
    with rasterio.open(raster) as src:
        lab = np.full((src.height, src.width), 255, dtype="uint8")
        for lon, lat, label in df[["lon", "lat", "label"]].itertuples(index=False):
            r, c = src.index(lon, lat)
            if 0 <= r < src.height and 0 <= c < src.width:
                lab[max(r - radius, 0):r + radius + 1, max(c - radius, 0):c + radius + 1] = int(label)
        profile = src.profile
    profile.update(count=1, dtype="uint8", nodata=255)
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(lab, 1)
    print(f"Labelled pixels: {(lab != 255).sum()} -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raster", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default=str(DATA_DIR / "labels.tif"))
    ap.add_argument("--radius", type=int, default=1, help="pixels around each point to label")
    a = ap.parse_args()
    main(a.raster, a.csv, a.out, a.radius)
