"""Vegetation change & anomaly detection from two dates (NDVI differencing + z-score)."""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from config import (DATA_DIR, OUT_DIR, PIXEL_AREA_HA, NDVI_LOSS_THRESHOLD,
                    NDVI_DECLINE_THRESHOLD, NDVI_GAIN_THRESHOLD)
from src.features import compute_indices
from src.io_utils import load_raster, save_raster, save_json

# 0 = stable, 1 = gain, 2 = decline, 3 = severe loss
CHANGE_LABELS = {0: "Stable", 1: "Gain", 2: "Decline", 3: "Severe loss"}
CHANGE_COLORS = ["#bdbdbd", "#43a047", "#fb8c00", "#b71c1c"]


def classify_change(delta):
    out = np.zeros(delta.shape, dtype="uint8")
    out[delta >= NDVI_GAIN_THRESHOLD] = 1
    out[delta <= NDVI_DECLINE_THRESHOLD] = 2
    out[delta <= NDVI_LOSS_THRESHOLD] = 3
    return out


def zscore_anomaly(ndvi, z_thresh=-2.0):
    """Pixels whose NDVI is >2 std-dev below the scene mean (spatial anomalies)."""
    z = (ndvi - ndvi.mean()) / (ndvi.std() + 1e-9)
    return z, z <= z_thresh


def main(before=None, after=None):
    before = before or DATA_DIR / "scene_before.tif"
    after = after or DATA_DIR / "scene_after.tif"
    b, meta = load_raster(before)
    a, _ = load_raster(after)
    ndvi_b, ndvi_a = compute_indices(b)["NDVI"], compute_indices(a)["NDVI"]
    delta = ndvi_a - ndvi_b
    change = classify_change(delta)
    z, anomaly = zscore_anomaly(ndvi_a)

    save_raster(delta.astype("float32"), OUT_DIR / "ndvi_change.tif", meta=meta)
    save_raster(change, OUT_DIR / "change_class.tif", dtype="uint8", meta=meta)
    save_raster(anomaly.astype("uint8"), OUT_DIR / "ndvi_anomaly.tif", dtype="uint8", meta=meta)

    stats = {CHANGE_LABELS[c]: {"hectares": round(float((change == c).sum() * PIXEL_AREA_HA), 2),
                                "percent": round(100 * float((change == c).mean()), 2)}
             for c in CHANGE_LABELS}
    stats["NDVI anomaly (z<=-2)"] = {"hectares": round(float(anomaly.sum() * PIXEL_AREA_HA), 2),
                                     "percent": round(100 * float(anomaly.mean()), 2)}
    save_json(stats, OUT_DIR / "change_stats.json")

    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    im = ax[0].imshow(delta, cmap="RdYlGn", vmin=-0.6, vmax=0.6)
    ax[0].set_title("ΔNDVI (after − before)"); ax[0].axis("off"); fig.colorbar(im, ax=ax[0], fraction=0.046)
    ax[1].imshow(change, cmap=ListedColormap(CHANGE_COLORS), vmin=0, vmax=3, interpolation="nearest")
    ax[1].set_title("Vegetation change class"); ax[1].axis("off")
    ax[1].legend(handles=[Patch(color=CHANGE_COLORS[i], label=CHANGE_LABELS[i]) for i in CHANGE_LABELS],
                 loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False)
    fig.tight_layout(); fig.savefig(OUT_DIR / "change_map.png", dpi=150); plt.close(fig)

    for k, v in stats.items():
        print(f"{k:24s} {v['hectares']:>8.2f} ha ({v['percent']:.1f}%)")
    return delta, change, stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default=None)
    ap.add_argument("--after", default=None)
    a = ap.parse_args()
    main(a.before, a.after)
