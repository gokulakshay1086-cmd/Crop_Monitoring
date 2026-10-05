"""Central configuration for the crop monitoring pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
for _d in (DATA_DIR, OUT_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Sentinel-2 L2A bands used (order matters everywhere in the project)
BANDS = ["B2", "B3", "B4", "B5", "B8", "B11"]   # Blue, Green, Red, RedEdge1, NIR, SWIR1

CLASS_NAMES = {0: "Healthy", 1: "Water stress", 2: "Crop anomaly"}
CLASS_COLORS = {0: "#2e7d32", 1: "#f9a825", 2: "#c62828"}

PIXEL_SIZE_M = 10
PIXEL_AREA_HA = (PIXEL_SIZE_M ** 2) / 10_000   # 0.01 ha for Sentinel-2 10 m

RANDOM_STATE = 42

# NDVI change thresholds (after - before)
NDVI_LOSS_THRESHOLD = -0.20
NDVI_DECLINE_THRESHOLD = -0.10
NDVI_GAIN_THRESHOLD = 0.10
