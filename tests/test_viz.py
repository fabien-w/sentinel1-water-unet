import pytest

pytest.importorskip("ipyleaflet")

from s1water.viz import ESRI_IMAGERY, PointPicker  # noqa: E402


def test_click_on_map_fills_the_boxes():
    p = PointPicker(-2.70, -60.80, half_size=0.25)
    assert (p.lat, p.lon) == (-2.70, -60.80)
    # simulated click on the map (ipyleaflet sends type="click" and coordinates=[lat, lon])
    p._on_map(type="click", coordinates=[-2.4512345, -61.1234567])
    assert (p.lat, p.lon) == (-2.45123, -61.12346)
    assert tuple(p.marker.location) == (-2.45123, -61.12346)
    assert p.square.bounds[0][0] == pytest.approx(-2.70123, abs=1e-5)
    # other map events (mouse moves) change nothing
    p._on_map(type="mousemove", coordinates=[0.0, 0.0])
    assert p.lat == -2.45123


def test_typing_in_the_boxes_moves_the_marker():
    p = PointPicker(-2.70, -60.80)
    p.lat_box.value, p.lon_box.value = -3.1, -60.0
    assert (p.lat, p.lon) == (-3.1, -60.0)
    assert tuple(p.marker.location) == (-3.1, -60.0)


def test_longitude_wrapped_and_no_openstreetmap():
    p = PointPicker(0, 0)
    p._on_map(type="click", coordinates=[1.0, 299.2])  # map panned once around the globe
    assert p.lon == pytest.approx(-60.8)
    urls = [layer.url for layer in p.map.layers if hasattr(layer, "url")]
    assert urls[0] == ESRI_IMAGERY and not any("openstreetmap" in u for u in urls)
