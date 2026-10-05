"""Interactive crop-monitoring dashboard.  Run:  streamlit run app/dashboard.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from config import DATA_DIR, OUT_DIR, CLASS_NAMES, CLASS_COLORS, PIXEL_AREA_HA
from src.features import compute_indices
from src.io_utils import load_raster

st.set_page_config(page_title="Satellite Crop Monitoring", page_icon="🌾", layout="wide")
st.title("🌾 AI-Powered Satellite Crop Monitoring")
st.caption("Sentinel-2 spectral indices · Random Forest / XGBoost · stress, anomaly & yield analytics")


@st.cache_data
def load(name, folder):
    try:
        arr, _ = load_raster(Path(folder) / name)
        return arr
    except FileNotFoundError:
        return None


def read_json(name):
    p = OUT_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


stress = load("stress_map.tif", OUT_DIR)
after = load("scene_after.tif", DATA_DIR)
before = load("scene_before.tif", DATA_DIR)
if stress is None or after is None:
    st.error("No outputs found. Run `python run_pipeline.py` first, then refresh.")
    st.stop()
stress = stress[0].astype(int)
idx_after = compute_indices(after)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Crop condition", "Index explorer", "Change detection", "Model performance", "Yield"])

# ---------------------------------------------------------------- tab 1
with tab1:
    stats = read_json("area_stats.json")
    cols = st.columns(len(CLASS_NAMES))
    for c, (k, name) in zip(cols, CLASS_NAMES.items()):
        c.metric(name, f"{stats[name]['hectares']:.1f} ha", f"{stats[name]['percent']:.1f}% of field")
    cmap = ListedColormap([CLASS_COLORS[i] for i in sorted(CLASS_COLORS)])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(stress, cmap=cmap, vmin=0, vmax=2, interpolation="nearest"); ax.axis("off")
    ax.legend(handles=[Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i]) for i in CLASS_NAMES],
              loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
    st.pyplot(fig)
    st.download_button("⬇ Download stress map (GeoTIFF for QGIS)",
                       data=open(OUT_DIR / "stress_map.tif", "rb").read() if (OUT_DIR / "stress_map.tif").exists() else b"",
                       file_name="stress_map.tif", disabled=not (OUT_DIR / "stress_map.tif").exists())

# ---------------------------------------------------------------- tab 2
with tab2:
    c1, c2 = st.columns([1, 2])
    with c1:
        index = st.selectbox("Spectral index", list(idx_after))
        arr = idx_after[index]
        thr = st.slider("Flag pixels below", float(np.percentile(arr, 1)), float(np.percentile(arr, 99)),
                        float(np.percentile(arr, 15)))
        st.metric("Area below threshold", f"{(arr < thr).sum() * PIXEL_AREA_HA:.1f} ha",
                  f"{100 * (arr < thr).mean():.1f}%")
        st.write(pd.Series(arr.ravel()).describe().round(3).to_frame(index))
    with c2:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
        im = ax[0].imshow(arr, cmap="RdYlGn"); ax[0].set_title(index); ax[0].axis("off")
        fig.colorbar(im, ax=ax[0], fraction=0.046)
        ax[1].imshow(arr < thr, cmap="Reds", interpolation="nearest")
        ax[1].set_title(f"{index} < {thr:.2f}"); ax[1].axis("off")
        st.pyplot(fig)

# ---------------------------------------------------------------- tab 3
with tab3:
    if before is None:
        st.info("Add scene_before.tif to enable change detection.")
    else:
        st.image(str(OUT_DIR / "change_map.png"), caption="NDVI differencing between dates")
        cs = read_json("change_stats.json")
        st.dataframe(pd.DataFrame(cs).T, use_container_width=True)

# ---------------------------------------------------------------- tab 4
with tab4:
    m = read_json("metrics.json")
    if m:
        rows = {k: {"Accuracy": v["accuracy"], "Macro F1": v["macro_f1"]}
                for k, v in m.items() if isinstance(v, dict)}
        st.dataframe(pd.DataFrame(rows).T.round(4), use_container_width=True)
        st.caption(f"Best model: **{m.get('best_model')}** (spatial block hold-out evaluation)")
        for name in [k for k in m if isinstance(m[k], dict)]:
            a, b = st.columns(2)
            a.image(str(OUT_DIR / f"confusion_{name}.png"))
            b.image(str(OUT_DIR / f"importance_{name}.png"))

# ---------------------------------------------------------------- tab 5
with tab5:
    ym = read_json("yield_metrics.json")
    if ym:
        st.dataframe(pd.DataFrame({k: v for k, v in ym.items() if isinstance(v, dict)}).T.round(3),
                     use_container_width=True)
        st.image(str(OUT_DIR / "yield_scatter.png"))
        st.warning("Demo yield target is synthetic. Supply real field yields via "
                   "`python -m src.yield_model --csv your_data.csv` for meaningful estimates.")
