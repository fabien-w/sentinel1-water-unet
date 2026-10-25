import numpy as np
import pytest

from s1water.data import s3_prefix
from s1water.preprocess import NODATA_8BIT, db_to_uint8, footprint_wkt, polarisations_from_name

NAME = "S1D_IW_GRDH_1SDV_20260828T094657_20260828T094722_004324_007F97_2C40"


def test_8bit_formula():
    db = np.array([-60.0, -49.0, -25.0, -10.0, 0.0, 1.0, 5.0, np.nan], dtype=np.float32)
    # (maximum(-49, minimum(1, x)) + 50) * 5
    np.testing.assert_array_equal(db_to_uint8(db), [5, 5, 125, 200, 250, 255, 255, 250])
    assert NODATA_8BIT == 250


def test_8bit_rounding_to_nearest():
    assert db_to_uint8(np.float32([-15.34]))[0] == 173  # 173.3 -> 173
    assert db_to_uint8(np.float32([-15.26]))[0] == 174  # 173.7 -> 174


def test_polarisations():
    assert polarisations_from_name(NAME) == "VH,VV"
    assert polarisations_from_name(NAME.replace("1SDV", "1SDH")) == "HH,HV"
    with pytest.raises(ValueError):
        polarisations_from_name(NAME.replace("1SDV", "1SXX"))


def test_footprint_from_kml(tmp_path):
    safe = tmp_path / NAME
    (safe / "preview").mkdir(parents=True)
    (safe / "preview" / "map-overlay.kml").write_text(
        "<kml><Document><Folder><GroundOverlay><gx:LatLonQuad><coordinates>"
        "-60.1,-2.1 -62.3,-1.7 -62.0,-3.2 -59.8,-3.6"
        "</coordinates></gx:LatLonQuad></GroundOverlay></Folder></Document></kml>"
    )
    wkt = footprint_wkt(safe)
    assert wkt.startswith("POLYGON((-60.1 -2.1")
    assert wkt.endswith("-60.1 -2.1))")


def test_bucket_prefix():
    # month and day without leading zeros, as in the original R download script
    assert s3_prefix(NAME) == f"GRD/2026/8/28/IW/DV/{NAME}/"
    assert s3_prefix(NAME.replace("20260828T094657", "20260105T094657")).startswith("GRD/2026/1/5/IW/DV/")


def test_dates_typed_by_users():
    import pytest

    from s1water.data import normalize_date

    # June and September have 30 days: the 31st is moved to the 30th instead of failing
    assert normalize_date("2026-06-31", end=True) == "2026-06-30"
    assert normalize_date("2026-09-31", end=True) == "2026-09-30"
    assert normalize_date("2026-02-30", end=True) == "2026-02-28"
    assert normalize_date("2026-08-31", end=True) == "2026-08-31"
    # month only: first or last day
    assert normalize_date("2026-06") == "2026-06-01"
    assert normalize_date("2026-06", end=True) == "2026-06-30"
    assert normalize_date(" 2026-6-5 ") == "2026-06-05"
    for bad in ("2026-13-01", "June 2026", "2026-06-00", ""):
        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            normalize_date(bad)


def test_start_after_end_is_rejected():
    import pytest

    from s1water.data import search_scenes

    with pytest.raises(ValueError, match="after the end date"):
        search_scenes(-60.8, -2.7, "2026-09-01", "2026-06-30")
