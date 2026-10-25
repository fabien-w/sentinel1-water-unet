"""Maps and figures for the notebooks (folium for interactive maps, matplotlib for static figures)."""

from __future__ import annotations

import base64
import io

import numpy as np

ESRI_IMAGERY = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
ESRI_ATTR = "Esri, Maxar, Earthstar Geographics"
# Place names and borders drawn over the imagery. OpenStreetMap tiles are not used: the OSM volunteer
# servers block maps embedded in notebooks such as Colab (HTTP 403, tile usage policy).
ESRI_LABELS = "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"


def _basemap(m):
    """Esri satellite imagery (default) with place-name labels."""
    import folium

    folium.TileLayer(ESRI_IMAGERY, attr=ESRI_ATTR, name="Satellite (Esri)", overlay=False, control=False).add_to(m)
    folium.TileLayer(ESRI_LABELS, attr="Esri", name="Place names", overlay=True).add_to(m)


def point_map(lat: float, lon: float, zoom: int = 10, footprints: dict | None = None, aoi_wkt: str | None = None):
    """Folium map centred on the point. Click anywhere to read the coordinates.

    ``footprints``: optional {label: WKT polygon} of candidate scenes.
    """
    import folium
    from shapely import wkt as swkt

    m = folium.Map(location=[lat, lon], zoom_start=zoom, tiles=None)
    _basemap(m)
    folium.Marker([lat, lon], tooltip=f"{lat:.4f}, {lon:.4f}").add_to(m)
    if aoi_wkt:
        folium.GeoJson(swkt.loads(aoi_wkt).__geo_interface__, name="area processed",
                       style_function=lambda _: {"color": "yellow", "fill": False, "weight": 2}).add_to(m)
    for label, w in (footprints or {}).items():
        folium.GeoJson(swkt.loads(w).__geo_interface__, name=label, tooltip=label,
                       style_function=lambda _: {"color": "#ff7f0e", "fillOpacity": 0.05, "weight": 1}).add_to(m)
    folium.LatLngPopup().add_to(m)
    folium.LayerControl().add_to(m)
    return m


class PointPicker:
    """Interactive map (ipyleaflet) to choose a point by clicking, with Latitude / Longitude boxes.

    A click moves the marker and fills the boxes; typing in the boxes moves the marker. The chosen
    point is read with ``.lat`` and ``.lon``. In Colab, call
    ``google.colab.output.enable_custom_widget_manager()`` before displaying ``.widget``.
    """

    def __init__(self, lat: float, lon: float, zoom: int = 9, half_size: float | None = None):
        import ipyleaflet as L
        import ipywidgets as W

        self._half = half_size
        # Esri imagery as basemap: the default OpenStreetMap tiles are blocked in Colab (HTTP 403)
        base = L.TileLayer(url=ESRI_IMAGERY, attribution=ESRI_ATTR, name="Satellite (Esri)", base=True)
        self.map = L.Map(center=(lat, lon), zoom=zoom, basemap=base, scroll_wheel_zoom=True,
                         layout=W.Layout(height="450px", width="100%"))
        self.map.add(L.TileLayer(url=ESRI_LABELS, attribution="Esri", name="Place names"))
        self.marker = L.Marker(location=(lat, lon), draggable=False, title="chosen point")
        self.map.add(self.marker)
        self.square = None
        if half_size:
            self.square = L.Rectangle(bounds=self._bounds(lat, lon), color="yellow", fill=False, weight=2)
            self.map.add(self.square)
        style = {"description_width": "70px"}
        self.lat_box = W.BoundedFloatText(value=round(lat, 5), min=-90, max=90, step=0.01,
                                          description="Latitude", style=style, layout=W.Layout(width="200px"))
        self.lon_box = W.BoundedFloatText(value=round(lon, 5), min=-180, max=180, step=0.01,
                                          description="Longitude", style=style, layout=W.Layout(width="200px"))
        help_text = W.HTML("<b>Click on the map</b> to choose the point,<br>or type the coordinates.")
        self.widget = W.HBox([self.map, W.VBox([help_text, self.lat_box, self.lon_box])])
        self.map.on_interaction(self._on_map)
        self.lat_box.observe(self._on_box, names="value")
        self.lon_box.observe(self._on_box, names="value")

    def _bounds(self, lat, lon):
        h = self._half
        return ((lat - h, lon - h), (lat + h, lon + h))

    def _move(self, lat, lon):
        self.marker.location = (lat, lon)
        if self.square is not None:
            self.square.bounds = self._bounds(lat, lon)

    def _on_map(self, **event):
        if event.get("type") == "click":
            lat, lon = event["coordinates"]
            lon = (lon + 180) % 360 - 180  # the map can be panned around the globe
            self.lat_box.value, self.lon_box.value = round(lat, 5), round(lon, 5)  # fills the boxes

    def _on_box(self, _change):
        self._move(self.lat_box.value, self.lon_box.value)

    @property
    def lat(self) -> float:
        return float(self.lat_box.value)

    @property
    def lon(self) -> float:
        return float(self.lon_box.value)


def _png_data_url(rgba: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _downsample(a: np.ndarray, max_size: int) -> np.ndarray:
    step = max(1, int(np.ceil(max(a.shape[-2:]) / max_size)))
    return a[..., ::step, ::step]


def sar_rgba(img8: np.ndarray, max_size: int = 2048) -> np.ndarray:
    """Grey VV image (8-bit) as RGBA, no-data (250) transparent."""
    vv = _downsample(img8[0], max_size)
    rgba = np.dstack([vv, vv, vv, np.where(vv == 250, 0, 255).astype(np.uint8)])
    return rgba


def mask_rgba(mask: np.ndarray, max_size: int = 2048, color=(30, 110, 255)) -> np.ndarray:
    m = _downsample(mask, max_size)
    rgba = np.zeros(m.shape + (4,), dtype=np.uint8)
    rgba[m == 1] = (*color, 200)
    return rgba


def result_map(img8: np.ndarray, mask: np.ndarray, transform, max_size: int = 2048):
    """Folium map with the satellite basemap, the SAR image and the water mask (lat/lon grids)."""
    import folium

    h, w = mask.shape
    west, north = transform.c, transform.f
    east, south = west + transform.a * w, north + transform.e * h
    bounds = [[south, west], [north, east]]
    m = folium.Map(location=[(south + north) / 2, (west + east) / 2], zoom_start=10, tiles=None)
    _basemap(m)
    folium.raster_layers.ImageOverlay(_png_data_url(sar_rgba(img8, max_size)), bounds=bounds,
                                      name="Sentinel-1 VV", opacity=0.9).add_to(m)
    folium.raster_layers.ImageOverlay(_png_data_url(mask_rgba(mask, max_size)), bounds=bounds,
                                      name="Water (U-Net)", opacity=0.8).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def figure(img8: np.ndarray, mask: np.ndarray, title: str = "", max_size: int = 1500):
    """Static 3-panel figure: VV, VH, water mask over VV."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    vv, vh = _downsample(img8[0], max_size), _downsample(img8[1], max_size)
    m = _downsample(mask, max_size)
    fig, ax = plt.subplots(1, 3, figsize=(18, 6.5))
    for a, im, t in ((ax[0], vv, "VV (8-bit)"), (ax[1], vh, "VH (8-bit)")):
        a.imshow(np.ma.masked_equal(im, 250), cmap="gray", vmin=100, vmax=230)
        a.set_title(t)
    ax[2].imshow(np.ma.masked_equal(vv, 250), cmap="gray", vmin=100, vmax=230)
    ax[2].imshow(np.ma.masked_where(m != 1, m), cmap=ListedColormap(["#1e6eff"]), alpha=0.75)
    ax[2].set_title("Water (blue) over VV")
    for a in ax:
        a.set_axis_off()
    fig.suptitle(title)
    fig.tight_layout()
    return fig
