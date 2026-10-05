"""Raster I/O. Uses rasterio (GeoTIFF, QGIS-ready); falls back to .npy if unavailable."""
from pathlib import Path
import json
import numpy as np

try:
    import rasterio
    from rasterio.transform import from_origin
    HAS_RASTERIO = True
except ImportError:  # pragma: no cover
    HAS_RASTERIO = False

# Default georeference (Tamil Nadu, India) used for synthetic demo data
DEFAULT_ORIGIN = (79.90, 13.20)   # lon, lat of upper-left corner
DEFAULT_RES_DEG = 0.00009         # ~10 m


def _npy_path(path):
    return Path(path).with_suffix(".npy")


def save_raster(array, path, dtype=None, meta=None):
    """Save (bands, H, W) or (H, W) array. GeoTIFF if rasterio present, else .npy."""
    array = np.asarray(array)
    if array.ndim == 2:
        array = array[None]
    path = Path(path)
    if dtype:
        array = array.astype(dtype)
    if HAS_RASTERIO:
        meta = meta or {}
        transform = meta.get("transform") or from_origin(*DEFAULT_ORIGIN, DEFAULT_RES_DEG, DEFAULT_RES_DEG)
        with rasterio.open(
            path, "w", driver="GTiff", height=array.shape[1], width=array.shape[2],
            count=array.shape[0], dtype=array.dtype, crs=meta.get("crs", "EPSG:4326"),
            transform=transform, compress="deflate",
        ) as dst:
            dst.write(array)
        return path
    np.save(_npy_path(path), array)
    return _npy_path(path)


def load_raster(path):
    """Return (array[bands,H,W], meta dict). Accepts .tif or .npy."""
    path = Path(path)
    if path.suffix == ".tif" and HAS_RASTERIO and path.exists():
        with rasterio.open(path) as src:
            return src.read().astype("float32"), {"transform": src.transform, "crs": src.crs}
    npy = _npy_path(path)
    if npy.exists():
        return np.load(npy).astype("float32"), {}
    raise FileNotFoundError(f"Could not find raster {path} (or {npy})")


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=float))
