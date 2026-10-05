"""Spectral index computation and feature engineering."""
import numpy as np
from config import BANDS

EPS = 1e-6


def _split(arr):
    """arr: (6, H, W) in BANDS order -> dict of 2D arrays."""
    if arr.shape[0] != len(BANDS):
        raise ValueError(f"Expected {len(BANDS)} bands {BANDS}, got {arr.shape[0]}")
    return {b: arr[i].astype("float32") for i, b in enumerate(BANDS)}


def compute_indices(arr):
    """Compute vegetation / water / moisture indices from a reflectance stack."""
    b = _split(arr)
    blue, green, red, re, nir, swir = (b[k] for k in ["B2", "B3", "B4", "B5", "B8", "B11"])
    return {
        "NDVI": (nir - red) / (nir + red + EPS),                                 # greenness
        "EVI": 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + EPS),       # enhanced veg.
        "SAVI": 1.5 * (nir - red) / (nir + red + 0.5),                           # soil-adjusted
        "GNDVI": (nir - green) / (nir + green + EPS),                            # chlorophyll
        "NDRE": (nir - re) / (nir + re + EPS),                                   # N / chlorophyll
        "NDWI": (green - nir) / (green + nir + EPS),                             # leaf/open water
        "NDMI": (nir - swir) / (nir + swir + EPS),                               # canopy moisture
    }


def build_feature_stack(arr):
    """Return (H, W, F) feature cube and list of feature names (bands + indices)."""
    b = _split(arr)
    idx = compute_indices(arr)
    names = list(b.keys()) + list(idx.keys())
    cube = np.stack(list(b.values()) + list(idx.values()), axis=-1)
    return np.nan_to_num(cube, nan=0.0, posinf=0.0, neginf=0.0), names
