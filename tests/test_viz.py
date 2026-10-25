import pytest

pytest.importorskip("ipyleaflet")

from s1water.viz import ESRI_IMAGERY, PointPicker  # noqa: E402

DEFAULT = (-2.70, -60.80)  # Anavilhanas


def click(p, lat, lon):
    # what ipyleaflet sends when the user clicks on the map
    p._on_map(type="click", coordinates=[lat, lon])


def test_default_state_toggle_off_and_clicks_ignored():
    p = PointPicker(*DEFAULT, half_size=0.25)
    assert p.click_toggle.value is False
    assert (p.lat, p.lon) == DEFAULT
    click(p, -2.45, -61.12)
    assert (p.lat, p.lon) == DEFAULT


def test_toggle_on_each_click_replaces_the_coordinates():
    p = PointPicker(*DEFAULT, half_size=0.25)
    p.click_toggle.value = True
    assert (p.lat, p.lon) == DEFAULT  # switching on does not move the point
    click(p, -2.4512345, -61.1234567)
    assert (p.lat, p.lon) == (-2.45123, -61.12346)
    assert tuple(p.marker.location) == (-2.45123, -61.12346)
    assert p.square.bounds[0][0] == pytest.approx(-2.70123, abs=1e-5)
    click(p, -3.0, -60.0)  # a new click replaces the previous one
    assert (p.lat, p.lon) == (-3.0, -60.0)
    p._on_map(type="mousemove", coordinates=[0.0, 0.0])  # other events change nothing
    assert (p.lat, p.lon) == (-3.0, -60.0)


def test_toggle_off_resets_to_default_and_ignores_clicks():
    p = PointPicker(*DEFAULT)
    p.click_toggle.value = True
    click(p, -3.0, -60.0)
    p.click_toggle.value = False
    assert (p.lat, p.lon) == DEFAULT
    assert tuple(p.marker.location) == DEFAULT
    click(p, -1.0, -62.0)
    assert (p.lat, p.lon) == DEFAULT


def test_typing_in_the_boxes_moves_the_marker():
    p = PointPicker(*DEFAULT)
    p.lat_box.value, p.lon_box.value = -3.1, -60.0
    assert (p.lat, p.lon) == (-3.1, -60.0)
    assert tuple(p.marker.location) == (-3.1, -60.0)


def test_longitude_wrapped_and_no_openstreetmap():
    p = PointPicker(0, 0, click_enabled=True)
    click(p, 1.0, 299.2)  # map panned once around the globe
    assert p.lon == pytest.approx(-60.8)
    urls = [layer.url for layer in p.map.layers if hasattr(layer, "url")]
    assert urls[0] == ESRI_IMAGERY and not any("openstreetmap" in u for u in urls)
