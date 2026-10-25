"""Command line interface: ``s1water search | download | preprocess | predict | run``."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from . import config


def main(argv=None):
    p = argparse.ArgumentParser(prog="s1water", description="Sentinel-1 surface water segmentation (U-Net)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="list Sentinel-1 IW GRD VV+VH scenes over a point")
    s.add_argument("--lat", type=float, default=config.DEFAULT_LAT)
    s.add_argument("--lon", type=float, default=config.DEFAULT_LON)
    s.add_argument("--start", default=config.DEFAULT_START)
    s.add_argument("--end", default=config.DEFAULT_END)

    d = sub.add_parser("download", help="download a scene from the requester-pays bucket")
    d.add_argument("name")
    d.add_argument("--out", default="data")

    pp = sub.add_parser("preprocess", help="SNAP preprocessing to the 8-bit VV/VH model input")
    pp.add_argument("safe_dir")
    pp.add_argument("--out", default="data")
    pp.add_argument("--aoi", help="WKT polygon (lon/lat); default: whole scene")

    pr = sub.add_parser("predict", help="predict water on a preprocessed 8-bit scene")
    pr.add_argument("img8")
    pr.add_argument("out")
    pr.add_argument("--weights", help="local .safetensors/.pt; default: download from Hugging Face")
    pr.add_argument("--device")
    pr.add_argument("--shade-mask", action="store_true", help="also mask SRTM mountain shade (off by default)")
    pr.add_argument("--probability", action="store_true", help="also save the probability raster")

    r = sub.add_parser("run", help="search + download + preprocess + predict (latest scene)")
    r.add_argument("--lat", type=float, default=config.DEFAULT_LAT)
    r.add_argument("--lon", type=float, default=config.DEFAULT_LON)
    r.add_argument("--start", default=config.DEFAULT_START)
    r.add_argument("--end", default=config.DEFAULT_END)
    r.add_argument("--half-size", type=float, default=config.DEFAULT_AOI_HALF_SIZE)
    r.add_argument("--out", default="data")

    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if a.cmd == "search":
        from .data import search_scenes

        for sc in search_scenes(a.lon, a.lat, a.start, a.end):
            print(sc.name, sc.start.isoformat(), sc.orbit_direction, sc.relative_orbit)
    elif a.cmd == "download":
        from .data import download_scene

        print(download_scene(a.name, a.out))
    elif a.cmd == "preprocess":
        from .preprocess import preprocess_scene

        print(preprocess_scene(a.safe_dir, a.out, aoi_wkt=a.aoi))
    elif a.cmd == "predict":
        from .pipeline import predict_scene

        print(json.dumps(predict_scene(a.img8, a.out, a.weights, a.device, a.shade_mask,
                                       save_probability=a.probability), indent=2))
    elif a.cmd == "run":
        from .data import download_scene, search_scenes
        from .pipeline import aoi_wkt, predict_scene
        from .preprocess import preprocess_scene

        scenes = search_scenes(a.lon, a.lat, a.start, a.end)
        if not scenes:
            raise SystemExit("no scene found")
        sc = scenes[-1]
        safe = download_scene(sc.name, Path(a.out) / "safe")
        img8 = preprocess_scene(safe, a.out, aoi_wkt=aoi_wkt(a.lon, a.lat, a.half_size))
        print(json.dumps(predict_scene(img8, Path(a.out) / f"{sc.name}_water.tif"), indent=2))


if __name__ == "__main__":
    main()
