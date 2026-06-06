from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.isp.simple_isp import apply_rgb_degradation


def test_apply_rgb_degradation_is_noop_without_config():
    image = np.linspace(0.0, 1.0, 3 * 16 * 16, dtype=np.float32).reshape(16, 16, 3)
    degraded = apply_rgb_degradation(image, None)

    assert degraded.shape == image.shape
    assert degraded.dtype == np.float32
    assert np.allclose(degraded, image)
    assert degraded is not image


def test_apply_rgb_degradation_jpeg_preserves_shape_and_changes_pixels():
    yy, xx = np.indices((64, 64), dtype=np.float32)
    image = np.stack(
        [
            yy / 63.0,
            xx / 63.0,
            0.5 * (np.sin(xx / 5.0) + 1.0),
        ],
        axis=-1,
    ).astype(np.float32)

    degraded = apply_rgb_degradation(image, {"type": "jpeg_compression", "quality": 35})

    assert degraded.shape == image.shape
    assert degraded.dtype == np.float32
    assert np.isfinite(degraded).all()
    assert degraded.min() >= 0.0
    assert degraded.max() <= 1.0
    assert not np.allclose(degraded, image)
