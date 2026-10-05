"""Field-level yield estimation from aggregated spectral features.

IMPORTANT: with no ground-truth yield data, the demo target is *synthetic*
(a noisy function of NDVI/NDMI). To use real data, pass --csv with a column
`yield_t_ha` and the same feature columns (see README).
"""
import argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from config import DATA_DIR, OUT_DIR, MODEL_DIR, RANDOM_STATE
from src.features import build_feature_stack
from src.io_utils import load_raster, save_json

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:  # pragma: no cover
    HAS_XGB = False

FEATURES = ["NDVI", "EVI", "NDRE", "NDWI", "NDMI", "GNDVI"]


def aggregate_to_fields(raster_path, block=16):
    """Split the scene into block x block 'fields'; return per-field mean features."""
    arr, _ = load_raster(raster_path)
    cube, names = build_feature_stack(arr)
    h, w, _ = cube.shape
    rows = []
    for i in range(0, h - block + 1, block):
        for j in range(0, w - block + 1, block):
            m = cube[i:i + block, j:j + block].reshape(-1, cube.shape[-1]).mean(axis=0)
            rows.append(dict(zip(names, m), field_id=f"F{i // block:02d}_{j // block:02d}"))
    return pd.DataFrame(rows)


def synthetic_yield(df, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    y = 2.0 + 6.5 * df["NDVI"] + 2.5 * df["NDMI"] + rng.normal(0, 0.35, len(df))
    return y.clip(0.5, None)


def main(csv=None):
    if csv:
        df = pd.read_csv(csv)
    else:
        df = aggregate_to_fields(DATA_DIR / "scene_after.tif")
        df["yield_t_ha"] = synthetic_yield(df)
    X, y = df[FEATURES], df["yield_t_ha"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)

    models = {"RandomForest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)}
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                         subsample=0.9, random_state=RANDOM_STATE)
    res, best, best_r2 = {}, None, -np.inf
    for n, m in models.items():
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        r2, rmse = r2_score(yte, p), float(np.sqrt(mean_squared_error(yte, p)))
        res[n] = {"r2": r2, "rmse_t_ha": rmse}
        print(f"{n:13s} R2={r2:.3f}  RMSE={rmse:.3f} t/ha")
        if r2 > best_r2:
            best, best_r2, best_pred = n, r2, p
            joblib.dump({"model": m, "features": FEATURES}, MODEL_DIR / "yield_model.joblib")
    res["best_model"] = best
    save_json(res, OUT_DIR / "yield_metrics.json")

    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    ax.scatter(yte, best_pred, s=18, c="#2e7d32", alpha=0.7)
    lim = [min(yte.min(), best_pred.min()), max(yte.max(), best_pred.max())]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("Actual yield (t/ha)"); ax.set_ylabel("Predicted yield (t/ha)")
    ax.set_title(f"Yield estimation ({best})"); fig.tight_layout()
    fig.savefig(OUT_DIR / "yield_scatter.png", dpi=150); plt.close(fig)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="CSV with FEATURES + yield_t_ha columns")
    main(ap.parse_args().csv)
