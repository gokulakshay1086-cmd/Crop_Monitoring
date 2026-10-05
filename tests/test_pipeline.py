import numpy as np
from src.features import compute_indices, build_feature_stack
from src.change_detection import classify_change
from src.train import spatial_block_split
from src import synthetic_data


def _stack(nir, red):
    a = np.full((6, 4, 4), 0.1, dtype="float32")
    a[4], a[2] = nir, red
    return a


def test_ndvi_values():
    idx = compute_indices(_stack(0.5, 0.1))
    assert np.allclose(idx["NDVI"], (0.5 - 0.1) / (0.6), atol=1e-4)
    assert (compute_indices(_stack(0.1, 0.5))["NDVI"] < 0).all()


def test_feature_stack_shape():
    cube, names = build_feature_stack(_stack(0.4, 0.05))
    assert cube.shape == (4, 4, len(names)) and len(names) == 13
    assert np.isfinite(cube).all()


def test_change_classes():
    d = np.array([[-0.5, -0.15, 0.0, 0.3]])
    assert classify_change(d).tolist() == [[3, 2, 0, 1]]


def test_spatial_split_no_leak_shape():
    m = spatial_block_split(100, 90)
    assert m.shape == (100, 90) and 0 < m.mean() < 1


def test_synthetic_generation(tmp_path):
    paths = synthetic_data.generate(size=64, out_dir=tmp_path)
    assert all(p.exists() for p in paths.values())
