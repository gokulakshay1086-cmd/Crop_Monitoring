"""One-command end-to-end run:  python run_pipeline.py

1. generate synthetic Sentinel-2-like data (skip with --use-existing if you have real rasters)
2. train Random Forest / XGBoost
3. predict crop-stress map
4. detect vegetation change / anomalies
5. train yield model
"""
import argparse
import time
from src import synthetic_data, train, predict, change_detection, yield_model


def step(title):
    print(f"\n{'=' * 8} {title} {'=' * 8}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-existing", action="store_true",
                    help="use rasters already in data/raw (scene_before/after.tif + labels.tif)")
    ap.add_argument("--size", type=int, default=256, help="synthetic scene size in pixels")
    a = ap.parse_args()
    t0 = time.time()

    if not a.use_existing:
        step("1/5 Generating synthetic scene"); synthetic_data.generate(a.size)
    step("2/5 Training classifiers");           train.main()
    step("3/5 Predicting crop stress");         predict.main()
    step("4/5 Change & anomaly detection");     change_detection.main()
    step("5/5 Yield estimation");               yield_model.main()

    print(f"\nDone in {time.time() - t0:.1f}s. Outputs in ./outputs  |  Dashboard: streamlit run app/dashboard.py")
