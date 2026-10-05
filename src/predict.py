"""Run the trained model over a scene -> crop-stress map (GeoTIFF) + PNG + area stats."""
import argparse
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from config import DATA_DIR, OUT_DIR, MODEL_DIR, CLASS_NAMES, CLASS_COLORS, PIXEL_AREA_HA
from src.features import build_feature_stack, compute_indices
from src.io_utils import load_raster, save_raster, save_json


def predict_scene(raster_path, model_path=None):
    bundle = joblib.load(model_path or MODEL_DIR / "best_model.joblib")
    arr, meta = load_raster(raster_path)
    cube, names = build_feature_stack(arr)
    assert names == bundle["features"], "Feature mismatch between model and raster"
    h, w, f = cube.shape
    pred = bundle["model"].predict(cube.reshape(-1, f)).reshape(h, w).astype("uint8")
    return pred, arr, meta


def area_stats(pred):
    total = pred.size
    return {
        CLASS_NAMES[c]: {
            "pixels": int((pred == c).sum()),
            "hectares": round(float((pred == c).sum() * PIXEL_AREA_HA), 2),
            "percent": round(100 * float((pred == c).sum()) / total, 2),
        } for c in CLASS_NAMES
    }


def plot_stress_map(pred, ndvi, path):
    cmap = ListedColormap([CLASS_COLORS[i] for i in sorted(CLASS_COLORS)])
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    im = axes[0].imshow(ndvi, cmap="RdYlGn", vmin=-0.1, vmax=0.9)
    axes[0].set_title("NDVI"); axes[0].axis("off")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    axes[1].imshow(pred, cmap=cmap, vmin=0, vmax=len(CLASS_COLORS) - 1, interpolation="nearest")
    axes[1].set_title("Predicted crop condition"); axes[1].axis("off")
    axes[1].legend(handles=[Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i]) for i in CLASS_NAMES],
                   loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main(raster=None, model=None):
    raster = raster or DATA_DIR / "scene_after.tif"
    pred, arr, meta = predict_scene(raster, model)
    save_raster(pred, OUT_DIR / "stress_map.tif", dtype="uint8", meta=meta)
    plot_stress_map(pred, compute_indices(arr)["NDVI"], OUT_DIR / "stress_map.png")
    stats = area_stats(pred)
    save_json(stats, OUT_DIR / "area_stats.json")
    for k, v in stats.items():
        print(f"{k:14s} {v['hectares']:>8.2f} ha  ({v['percent']:.1f}%)")
    return pred, stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raster", default=None)
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    main(a.raster, a.model)
