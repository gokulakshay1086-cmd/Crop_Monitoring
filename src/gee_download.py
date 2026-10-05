"""Download a cloud-masked Sentinel-2 L2A median composite from Google Earth Engine.

Usage:
    earthengine authenticate            # once
    python -m src.gee_download --project YOUR_GCP_PROJECT \
        --bbox 79.85 13.10 79.95 13.20 --start 2025-08-01 --end 2025-09-15 \
        --out data/raw/scene_after.tif

Keep the AOI small (~<50 MB download limit, e.g. a ~10 x 10 km block at 10 m).
For bigger areas use ee.batch.Export.image.toDrive instead.
"""
import argparse
import io
import zipfile
import urllib.request
import numpy as np
from config import BANDS, DATA_DIR

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"


def mask_clouds(img):
    """Keep vegetation(4), bare soil(5), water(6), unclassified(7) using the SCL band."""
    scl = img.select("SCL")
    good = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
    return img.updateMask(good)


def build_composite(ee, bbox, start, end, max_cloud=30):
    region = ee.Geometry.Rectangle(bbox)
    col = (ee.ImageCollection(S2_COLLECTION)
           .filterBounds(region).filterDate(start, end)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
           .map(mask_clouds))
    n = col.size().getInfo()
    if n == 0:
        raise RuntimeError("No Sentinel-2 scenes found - widen dates or cloud threshold.")
    print(f"Using {n} Sentinel-2 scenes")
    img = col.select(BANDS).median().divide(10000).clip(region).toFloat()
    return img, region


def download(img, region, out_path, scale=10):
    url = img.getDownloadURL({"region": region, "scale": scale, "format": "GEO_TIFF",
                              "crs": "EPSG:4326"})
    print("Downloading ...")
    data = urllib.request.urlopen(url).read()
    if zipfile.is_zipfile(io.BytesIO(data)):          # GEE returns a zip of per-band tifs
        import rasterio
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            tifs = {n.split(".")[-2]: n for n in z.namelist() if n.endswith(".tif")}
            arrays, profile = [], None
            for b in BANDS:
                with rasterio.MemoryFile(z.read(tifs[b])) as mf, mf.open() as src:
                    arrays.append(src.read(1)); profile = src.profile
        profile.update(count=len(BANDS), dtype="float32", compress="deflate")
        with rasterio.open(out_path, "w", **profile) as dst:
            for i, a in enumerate(arrays, 1):
                dst.write(a.astype("float32"), i)
    else:
        with open(out_path, "wb") as f:
            f.write(data)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="Google Cloud project registered for Earth Engine")
    ap.add_argument("--bbox", type=float, nargs=4, required=True,
                    metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"))
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", default=str(DATA_DIR / "scene_after.tif"))
    ap.add_argument("--max-cloud", type=float, default=30)
    a = ap.parse_args()

    import ee
    try:
        ee.Initialize(project=a.project)
    except Exception:
        ee.Authenticate(); ee.Initialize(project=a.project)
    image, region = build_composite(ee, a.bbox, a.start, a.end, a.max_cloud)
    download(image, region, a.out)
