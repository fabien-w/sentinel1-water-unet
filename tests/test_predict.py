import numpy as np
import pytest
import torch

from s1water.predict import STEP, TILE, predict_image, tile_origins


class MeanModel(torch.nn.Module):
    """Fake model: probability = mean of the 4 input bands, cropped like the U-Net."""

    def forward(self, x, crop=0):
        p = x.mean(1, keepdim=True)
        return p[..., crop:-crop, crop:-crop] if crop else p


def test_original_tile_grid():
    # a 10000 x 9000 image is padded to 12288 + 128 and cut every 4096 px
    o = tile_origins(10000, 9000)
    assert o[0] == (0, 0)
    assert sorted({r for r, _ in o}) == [0, 4096, 8192]
    assert sorted({c for _, c in o}) == [0, 4096, 8192]
    assert TILE == STEP + 128


def test_stitching_covers_image_on_same_grid():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(2, 150, 130), dtype=np.uint8)
    prob = predict_image(MeanModel(), img, "cpu", return_probability=True,
                         progress=False, tile=48, step=32, crop=8)
    assert prob.shape == (150, 130)
    # first `crop` rows/cols are never predicted (as in the original workflow)
    assert np.isnan(prob[:8]).all() and np.isnan(prob[:, :8]).all()
    expected = img.astype(np.float32).mean(0) / 255
    np.testing.assert_allclose(prob[8:, 8:], expected[8:, 8:], rtol=1e-6)


def test_mask_coding():
    img = np.full((2, 100, 100), 255, dtype=np.uint8)
    img[:, 50:, :] = 0
    m = predict_image(MeanModel(), img, "cpu", progress=False, tile=48, step=32, crop=8)
    assert m.dtype == np.uint8
    assert set(np.unique(m)) <= {0, 1, 255}
    assert (m[:8] == 255).all()
    assert (m[20:40, 20:40] == 1).all() and (m[60:90, 20:40] == 0).all()


def test_input_checks():
    with pytest.raises(ValueError):
        predict_image(MeanModel(), np.zeros((4, 10, 10), np.uint8), "cpu")
    with pytest.raises(ValueError):
        predict_image(MeanModel(), np.zeros((2, 10, 10), np.uint8), "cpu", tile=50, step=32, crop=8)


def test_input_255_is_valid_data_not_no_data():
    # a valid input pixel at 255 (bright target, >= +1 dB) must give 0/1, never 255, in the output
    img = np.full((2, 120, 120), 120, dtype=np.uint8)
    img[:, 40:60, 40:60] = 255
    m = predict_image(MeanModel(), img, "cpu", progress=False, tile=48, step=32, crop=8)
    assert (m[8:, 8:] != 255).all()
    assert (m[40:60, 40:60] == 1).all()
