"""Generate a *synthetic* Sentinel-2-like scene so the project runs with no downloads.

Two dates are produced:
  * before : healthy peak-season canopy
  * after  : same field with water-stress and anomaly (pest/disease/nutrient) patches
Ground-truth labels are returned for supervised training.
Replace with real data via src/gee_download.py (see README).
"""
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter
from config import BANDS, DATA_DIR, RANDOM_STATE
from src.io_utils import save_raster

# mean reflectance per class: B2, B3, B4, B5, B8, B11
SIGNATURES = {
    0: [0.040, 0.085, 0.040, 0.150, 0.450, 0.200],   # healthy
    1: [0.050, 0.090, 0.075, 0.180, 0.320, 0.310],   # water stress (lower NIR, high SWIR)
    2: [0.060, 0.105, 0.115, 0.200, 0.240, 0.260],   # anomaly (disease/pest/nutrient)
}


def _smooth_field(size, sigma, rng):
    f = gaussian_filter(rng.standard_normal((size, size)), sigma)
    return (f - f.mean()) / f.std()


def make_labels(size, rng):
    labels = np.zeros((size, size), dtype="uint8")
    labels[_smooth_field(size, 18, rng) > 0.95] = 1
    labels[_smooth_field(size, 12, rng) > 1.15] = 2
    return labels


def render_scene(labels, rng, noise=0.03):
    """Render reflectance. Stressed pixels have variable severity (blend with healthy
    signature), so classes overlap like real imagery -> realistic, not perfect, accuracy."""
    size = labels.shape[0]
    scene = np.zeros((len(BANDS), size, size), dtype="float32")
    texture = 1 + 0.10 * _smooth_field(size, 3, rng)          # within-field variability
    severity = np.clip(0.25 + 0.75 * rng.random((size, size)) + 0.2 * _smooth_field(size, 6, rng), 0.05, 1.0)
    healthy = np.array(SIGNATURES[0])
    for cls, sig in SIGNATURES.items():
        m = labels == cls
        sig = np.array(sig)
        for i in range(len(BANDS)):
            w = severity if cls != 0 else 1.0
            mean = healthy[i] + w * (sig[i] - healthy[i]) if cls != 0 else sig[i]
            band = mean * texture + rng.normal(0, noise, (size, size)) * (0.6 + sig[i])
            scene[i][m] = band[m]
    return np.clip(scene, 0.001, 1.0)


def generate(size=256, seed=RANDOM_STATE, out_dir=DATA_DIR):
    rng = np.random.default_rng(seed)
    labels = make_labels(size, rng)
    before = render_scene(np.zeros_like(labels), rng)   # everything healthy earlier in season
    after = render_scene(labels, rng)
    return {
        "before": save_raster(before, out_dir / "scene_before.tif", dtype="float32"),
        "after": save_raster(after, out_dir / "scene_after.tif", dtype="float32"),
        "labels": save_raster(labels, out_dir / "labels.tif", dtype="uint8"),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=RANDOM_STATE)
    a = ap.parse_args()
    for k, v in generate(a.size, a.seed).items():
        print(f"{k:7s} -> {v}")
