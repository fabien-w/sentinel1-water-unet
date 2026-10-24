import numpy as np
from rasterio.crs import CRS
from rasterio.transform import from_origin

from s1water.postprocess import apply_masks, slope_degrees, valid_data_mask

RES = 8.983152841195215e-05  # 10 m in degrees, SNAP default pixel spacing
TR = from_origin(-61.0, -2.5, RES, RES)
WGS84 = CRS.from_epsg(4326)


def test_valid_mask_erodes_scene_borders():
    img = np.full((2, 1600, 1600), 120, dtype=np.uint8)
    img[:, :, :400] = 250  # no-data on the west (range edge): erode ~1000 m
    img[:, :300, :] = 250  # no-data on the north (azimuth edge): erode ~500 m
    v = valid_data_mask(img, TR, WGS84)
    cols = np.nonzero(v[1000])[0]
    rows = np.nonzero(v[:, 1000])[0]
    assert abs((cols.min() - 400) * 10 - 1000) <= 2 * 160  # within one coarse cell (16 px = 160 m)
    assert abs((rows.min() - 300) * 10 - 500) <= 2 * 160
    # image edges that are not scene borders (east, south) are not eroded
    assert v[1599, 1599] and v[1000, 1599]


def test_valid_mask_fills_holes_but_keeps_nodata_pixels():
    img = np.full((2, 640, 640), 120, dtype=np.uint8)
    img[:, 300:302, 300:302] = 250
    v = valid_data_mask(img, TR, WGS84)
    assert v[100, 100] and v[310, 310]
    assert not v[300, 300]


def test_slope_of_a_plane():
    x = np.arange(50, dtype=np.float32) * 30 * np.tan(np.radians(25))
    dem = np.tile(x, (50, 1))
    s = slope_degrees(dem, 30.0, 30.0)
    np.testing.assert_allclose(s[5:-5, 5:-5], 25.0, atol=1e-3)


def test_apply_masks():
    pred = np.array([[0, 1], [1, 255]], dtype=np.uint8)
    valid = np.array([[True, False], [True, True]])
    shade = np.array([[False, False], [True, False]])
    np.testing.assert_array_equal(apply_masks(pred, valid, shade), [[0, 255], [255, 255]])
