"""Compare a scene preprocessed with this package to tiles preprocessed by the original workflow.

The original training tiles are 256 x 256 GeoTIFFs (4 bands VV, VH, VV, VH, uint8) cut from
the original SNAP 9 output. We read the same window in the new dB output (SNAP 14), convert it
to 8 bits with two rounding rules (round to nearest, truncate), and report agreement.

Usage:
    python scripts/validate_preprocessing.py NEW_DB_TIF REF_TILES_DIR OUT_JSON
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds

from s1water.preprocess import DB_MAX, DB_MIN, NODATA_8BIT, band_index, db_to_uint8


def db_to_uint8_trunc(db):
    x = np.where(np.isnan(db), 0, db).astype(np.float32)
    return ((np.maximum(np.float32(DB_MIN), np.minimum(np.float32(DB_MAX), x)) + 50) * 5).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("new_db_tif", type=Path)
    ap.add_argument("ref_dir", type=Path)
    ap.add_argument("out_json", type=Path)
    ap.add_argument("--weights", help="PyTorch weights: also compare the water masks predicted on both inputs")
    args = ap.parse_args()
    model = None
    if args.weights:
        import torch

        from s1water.model import load_model

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_model(args.weights, dev)
        mstats = {"agree": 0, "n": 0, "inter": 0, "union": 0, "water_ref": 0, "water_new": 0}

    def predict(img2):
        x = torch.from_numpy(np.concatenate([img2, img2]).astype(np.float32) / 255)[None].to(dev)
        with torch.inference_mode():
            return model(x.to(next(model.parameters()).dtype))[0, 0].float().cpu().numpy() >= 0.5

    refs = sorted(args.ref_dir.glob("*.tif"))
    stats = {"round": [], "trunc": []}
    offsets = []
    db_diffs = []
    with rasterio.open(args.new_db_tif) as src:
        res = {"new_crs": str(src.crs), "new_res": src.res, "n_ref_tiles": len(refs)}
        bands = [band_index(src, "VV"), band_index(src, "VH")]
        for rp in refs:
            with rasterio.open(rp) as ref:
                a = ref.read()
                assert a.shape[0] == 4 and np.array_equal(a[0], a[2]) and np.array_equal(a[1], a[3])
                win = from_bounds(*ref.bounds, transform=src.transform)
                offsets.append((win.col_off - round(win.col_off), win.row_off - round(win.row_off)))
                win = win.round_offsets().round_lengths()
                db = src.read(bands, window=win, boundless=True, fill_value=np.nan)
            ref_vvvh = a[:2]
            valid = (ref_vvvh != NODATA_8BIT).all(0) & (ref_vvvh != 255).all(0) & ~np.isnan(db).any(0)
            if valid.sum() == 0:
                continue
            for k, fn in (("round", db_to_uint8), ("trunc", db_to_uint8_trunc)):
                new = fn(db).astype(np.int16)
                d = np.abs(new - ref_vvvh.astype(np.int16))[:, valid]
                stats[k].append((d.size, int((d == 0).sum()), int((d <= 1).sum()), float(d.mean()), int(d.max())))
            if model is not None:
                m_ref = predict(ref_vvvh)
                m_new = predict(db_to_uint8(db))
                mstats["agree"] += int((m_ref == m_new)[valid].sum())
                mstats["n"] += int(valid.sum())
                mstats["inter"] += int((m_ref & m_new)[valid].sum())
                mstats["union"] += int((m_ref | m_new)[valid].sum())
                mstats["water_ref"] += int(m_ref[valid].sum())
                mstats["water_new"] += int(m_new[valid].sum())
            # difference expressed in dB (reference back-transformed with its 0.2 dB quantisation)
            ref_db = ref_vvvh.astype(np.float32) / 5 - 50
            db_diffs.append(np.abs(np.clip(db, DB_MIN, DB_MAX) - ref_db)[:, valid].ravel())

    for k, v in stats.items():
        v = np.array(v, dtype=np.float64)
        n = v[:, 0].sum()
        res[f"8bit_{k}"] = {
            "tiles_compared": len(v),
            "pixels_compared": int(n),
            "exact_match": v[:, 1].sum() / n,
            "within_1_dn": v[:, 2].sum() / n,
            "mean_abs_diff_dn": float((v[:, 3] * v[:, 0]).sum() / n),
            "max_abs_diff_dn": int(v[:, 4].max()),
        }
    dd = np.concatenate(db_diffs)
    res["db_abs_diff"] = {"median": float(np.median(dd)), "p95": float(np.percentile(dd, 95)), "p99": float(np.percentile(dd, 99))}
    off = np.array(offsets)
    res["subpixel_offset_cols_rows"] = {"mean": off.mean(0).tolist(), "max_abs": np.abs(off).max(0).tolist()}
    if model is not None:
        res["water_mask_original_vs_new_input"] = {
            "pixel_agreement": mstats["agree"] / mstats["n"],
            "iou_water": mstats["inter"] / max(mstats["union"], 1),
            "water_fraction_original_input": mstats["water_ref"] / mstats["n"],
            "water_fraction_new_input": mstats["water_new"] / mstats["n"],
        }
    args.out_json.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
