"""Train Random Forest and XGBoost crop-stress classifiers with a spatial block split.

A *spatial* hold-out (whole 32x32 blocks go to train OR test) avoids the optimistic
bias you get from randomly splitting neighbouring, highly-correlated pixels.
"""
import argparse
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from config import DATA_DIR, OUT_DIR, MODEL_DIR, CLASS_NAMES, RANDOM_STATE
from src.features import build_feature_stack
from src.io_utils import load_raster, save_json

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:  # pragma: no cover
    HAS_XGB = False


def spatial_block_split(h, w, block=32, test_frac=0.3, seed=RANDOM_STATE):
    """Boolean (h, w) mask: True = test pixel."""
    rng = np.random.default_rng(seed)
    nby, nbx = -(-h // block), -(-w // block)
    is_test_block = rng.random((nby, nbx)) < test_frac
    return np.kron(is_test_block, np.ones((block, block), bool))[:h, :w]


def prepare_data(raster_path, labels_path, max_train=60000):
    arr, _ = load_raster(raster_path)
    labels, _ = load_raster(labels_path)
    labels = labels[0].astype(int)
    cube, names = build_feature_stack(arr)
    h, w, f = cube.shape
    test_mask = spatial_block_split(h, w)
    X, y = cube.reshape(-1, f), labels.ravel()
    tm = test_mask.ravel()
    valid = y != 255                      # 255 = unlabeled (real-world sparse labels)
    Xtr, ytr = X[~tm & valid], y[~tm & valid]
    Xte, yte = X[tm & valid], y[tm & valid]
    if len(ytr) > max_train:  # subsample for speed
        idx = np.random.default_rng(RANDOM_STATE).choice(len(ytr), max_train, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]
    return Xtr, ytr, Xte, yte, names


def get_models():
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=14, n_jobs=-1, class_weight="balanced",
            random_state=RANDOM_STATE),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.08, subsample=0.85,
            colsample_bytree=0.85, objective="multi:softprob", eval_metric="mlogloss",
            n_jobs=-1, random_state=RANDOM_STATE)
    else:
        print("[warn] xgboost not installed -> training Random Forest only "
              "(pip install xgboost)")
    return models


def plot_confusion(cm, title, path):
    labels = [CLASS_NAMES[i] for i in range(len(CLASS_NAMES))]
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_importance(names, imp, title, path):
    order = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.barh(np.array(names)[order], imp[order], color="#2e7d32")
    ax.set_title(title); ax.set_xlabel("Importance")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main(raster=None, labels=None):
    raster = raster or DATA_DIR / "scene_after.tif"
    labels = labels or DATA_DIR / "labels.tif"
    Xtr, ytr, Xte, yte, names = prepare_data(raster, labels)
    print(f"train pixels: {len(ytr):,} | test pixels (spatial hold-out): {len(yte):,}")

    results, best_name, best_f1, best_model = {}, None, -1, None
    for name, model in get_models().items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        acc = accuracy_score(yte, pred)
        f1 = f1_score(yte, pred, average="macro")
        cm = confusion_matrix(yte, pred, labels=list(CLASS_NAMES))
        print(f"\n=== {name} ===  accuracy={acc:.4f}  macro-F1={f1:.4f}")
        print(classification_report(yte, pred, target_names=list(CLASS_NAMES.values()), digits=3))
        results[name] = {"accuracy": acc, "macro_f1": f1, "confusion_matrix": cm.tolist()}
        plot_confusion(cm, f"{name} confusion matrix", OUT_DIR / f"confusion_{name}.png")
        plot_importance(names, model.feature_importances_, f"{name} feature importance",
                        OUT_DIR / f"importance_{name}.png")
        results[name]["feature_importance"] = dict(zip(names, map(float, model.feature_importances_)))
        joblib.dump({"model": model, "features": names}, MODEL_DIR / f"{name}.joblib")
        if f1 > best_f1:
            best_name, best_f1, best_model = name, f1, model

    joblib.dump({"model": best_model, "features": names, "name": best_name},
                MODEL_DIR / "best_model.joblib")
    results["best_model"] = best_name
    save_json(results, OUT_DIR / "metrics.json")
    print(f"\nBest model: {best_name} (macro-F1={best_f1:.4f}) -> models/best_model.joblib")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raster", default=None, help="multi-band reflectance raster (6 bands)")
    ap.add_argument("--labels", default=None, help="single-band label raster (0/1/2)")
    a = ap.parse_args()
    main(a.raster, a.labels)
