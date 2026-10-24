"""Search and download Sentinel-1 IW GRD scenes, as in the original workflow.

Search: Alaska Satellite Facility catalogue (``asf_search``), no account needed.
Download: the ``sentinel-s1-l1c`` S3 bucket (Sentinel-1 on the AWS Registry of Open Data,
https://registry.opendata.aws/sentinel-1/), region eu-central-1. The bucket is
**requester pays**: you need an AWS account, and your account pays the data transfer
(about US$0.09 per GB to the internet, about 1.3 GB per scene in 2026). Inside AWS
eu-central-1 the transfer is free.

Credentials are read by boto3 in the usual way: environment variables
(AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY), AWS_PROFILE, or ~/.aws/credentials.
They are never stored by this package.

Bucket layout (from the original R download script):
    s3://sentinel-s1-l1c/GRD/{YYYY}/{M}/{D}/IW/DV/{product_name}/
    with month and day without leading zeros, and unzipped SAFE content inside.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

BUCKET = "sentinel-s1-l1c"
BUCKET_REGION = "eu-central-1"


@dataclass
class Scene:
    name: str
    start: datetime
    orbit_direction: str
    relative_orbit: int
    footprint_wkt: str
    size_bytes: int | None = None

    @property
    def s3_prefix(self) -> str:
        return s3_prefix(self.name)


def s3_prefix(name: str) -> str:
    """Bucket prefix of a product, e.g. GRD/2026/8/28/IW/DV/S1D_IW_GRDH_1SDV_.../"""
    t = datetime.strptime(name.split("_")[4], "%Y%m%dT%H%M%S")
    mode, pol = name.split("_")[1], name.split("_")[3][2:4]
    return f"GRD/{t.year}/{t.month}/{t.day}/{mode}/{pol}/{name}/"


def search_scenes(lon: float, lat: float, start: str, end: str) -> list[Scene]:
    """Sentinel-1 IW GRDH dual-pol (VV+VH) scenes intersecting a point, sorted by date."""
    import asf_search as asf

    results = asf.geo_search(
        platform=[asf.PLATFORM.SENTINEL1],
        processingLevel=asf.PRODUCT_TYPE.GRD_HD,
        beamMode=asf.BEAMMODE.IW,
        polarization=asf.POLARIZATION.VV_VH,
        intersectsWith=f"POINT({lon} {lat})",
        start=start,
        end=end,
    )
    from shapely.geometry import shape

    scenes = []
    for r in results:
        p = r.properties
        scenes.append(
            Scene(
                name=p["sceneName"],
                start=datetime.fromisoformat(p["startTime"].replace("Z", "+00:00")).replace(tzinfo=None),
                orbit_direction=p["flightDirection"],
                relative_orbit=int(p["pathNumber"]),
                footprint_wkt=shape(r.geometry).wkt,
                size_bytes=p.get("bytes"),
            )
        )
    return sorted(scenes, key=lambda s: s.start)


def _s3_client(pool: int = 32):
    import boto3
    from botocore.config import Config

    cfg = Config(max_pool_connections=pool, retries={"max_attempts": 10, "mode": "adaptive"})
    return boto3.session.Session().client("s3", region_name=BUCKET_REGION, config=cfg)


def scene_in_bucket(name: str) -> bool:
    """True if the product exists in the bucket (one small requester-pays LIST request)."""
    s3 = _s3_client()
    r = s3.list_objects_v2(Bucket=BUCKET, Prefix=s3_prefix(name) + "manifest.safe", RequestPayer="requester")
    return r.get("KeyCount", 0) > 0


def download_scene(name: str, out_dir: str | Path, workers: int = 8, skip_existing: bool = True) -> Path:
    """Download the SAFE content of a scene from the requester-pays bucket.

    Only the files needed by SNAP are downloaded (manifest, annotation, measurement,
    preview, support). Returns the local product folder ``out_dir/<name>``.
    """
    s3 = _s3_client()
    prefix = s3_prefix(name)
    dest = Path(out_dir) / name
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix, RequestPayer="requester"):
        keys += [(o["Key"], o["Size"]) for o in page.get("Contents", [])]
    if not keys:
        raise FileNotFoundError(f"{name} not found in s3://{BUCKET}/{prefix}")
    keys = [(k, s) for k, s in keys if not k.endswith(".pdf")]  # quality report not needed
    total = sum(s for _, s in keys)
    log.info("Downloading %s (%.2f GB, requester pays)", name, total / 1e9)

    def get(item):
        key, size = item
        local = dest / key[len(prefix):]
        if skip_existing and local.exists() and local.stat().st_size == size:
            return
        local.parent.mkdir(parents=True, exist_ok=True)
        tmp = local.with_suffix(local.suffix + ".part")
        s3.download_file(BUCKET, key, str(tmp), ExtraArgs={"RequestPayer": "requester"})
        tmp.rename(local)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(get, keys))
    return dest
