"""Validate the PyTorch port against the original Keras model and the original R predictions.

Uses the validation tiles of the paper (``data_validation_for_f1_paper``), written by
``35_computation_F1_score.r``:
  img/<name>.png                        4-band 8-bit tile (VV, VH, VV, VH), 256 x 256
  mask_obs_n_preds/<name>.png           reference mask + 1 (1 = no water, 255 = water)
  mask_obs_n_preds/<name>_pred.png      original R/Keras prediction (0 or 255), mixed_float16

Checks:
  1. Keras float32 vs PyTorch float32 probabilities (max abs diff, mask agreement)
  2. PyTorch (float32 and float16 model) masks vs original R masks (agreement, IoU)
  3. F1 / precision / recall against the reference masks (paper Table 2: 0.934 / 0.926 / 0.930)

Usage:
    python scripts/validate_port.py VAL_DIR KERAS_H5 TORCH_PT OUT_JSON [--n-keras 300]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from s1water.model import load_model


def read_png(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path))


def scores(obs: np.ndarray, pred: np.ndarray) -> dict:
    tp = np.sum(obs & pred)
    fp = np.sum(~obs & pred)
    fn = np.sum(obs & ~pred)
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn)}


def prf(tp, fp, fn) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("val_dir", type=Path)
    ap.add_argument("keras_h5", type=Path)
    ap.add_argument("torch_pt", type=Path)
    ap.add_argument("out_json", type=Path)
    ap.add_argument("--n-keras", type=int, default=300, help="tiles used for the Keras vs PyTorch comparison")
    args = ap.parse_args()

    imgs = sorted((args.val_dir / "img").glob("*.png"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # strict float32 on GPU (TF32 is on by default for cuDNN convolutions)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = load_model(args.torch_pt, device, half=False)
    model16 = load_model(args.torch_pt, device, half=True) if device == "cuda" else model

    kmodel = None
    if args.n_keras > 0:
        import sys

        sys.path.insert(0, str(Path(__file__).parent))
        from convert_keras_to_pytorch import build_keras_unet

        kmodel, _ = build_keras_unet(crop=0)
        kmodel.load_weights(str(args.keras_h5))

    res = {"n_tiles": len(imgs), "device": device}
    max_abs = 0.0
    mean_abs = []
    k_agree = []
    k_vs_orig = []
    agg = {k: {"tp": 0, "fp": 0, "fn": 0} for k in ("orig", "torch32", "torch16")}
    vs_orig = {"torch32": {"agree": 0, "inter": 0, "union": 0}, "torch16": {"agree": 0, "inter": 0, "union": 0}}
    n_pix = 0
    tiles_identical = {"torch32": 0, "torch16": 0}

    for i, ip in enumerate(imgs):
        x8 = read_png(ip)  # (256, 256, 4) uint8
        x = x8.astype(np.float32) / 255.0  # tf.image.convert_image_dtype(uint8 -> float32)
        obs_png = args.val_dir / "mask_obs_n_preds" / ip.name
        pred_png = args.val_dir / "mask_obs_n_preds" / (ip.stem + "_pred.png")
        obs = ((read_png(obs_png).astype(np.int32) - 1) / 254.0) >= 0.5
        orig = read_png(pred_png) > 127
        if obs.ndim == 3:
            obs = obs[..., 0]
        if orig.ndim == 3:
            orig = orig[..., 0]

        xt = torch.from_numpy(x).permute(2, 0, 1)[None].to(device)
        with torch.no_grad():
            p32 = model(xt)[0, 0].float().cpu().numpy()
            p16 = model16(xt.to(next(model16.parameters()).dtype))[0, 0].float().cpu().numpy()
        m32, m16 = p32 > 0.5, p16 > 0.5  # 35_computation_F1_score.r uses tf$where(mask_pred > 0.5, 1, 0)

        if kmodel is not None and i < args.n_keras:
            pk = kmodel(x[None], training=False).numpy()[0, ..., 0]
            d = np.abs(pk - p32)
            max_abs = max(max_abs, float(d.max()))
            mean_abs.append(float(d.mean()))
            k_agree.append(float(np.mean((pk > 0.5) == m32)))
            k_vs_orig.append(float(np.mean((pk > 0.5) == orig)))

        for k, m in (("orig", orig), ("torch32", m32), ("torch16", m16)):
            s = scores(obs, m)
            for kk in s:
                agg[k][kk] += s[kk]
        for k, m in (("torch32", m32), ("torch16", m16)):
            vs_orig[k]["agree"] += int(np.sum(m == orig))
            vs_orig[k]["inter"] += int(np.sum(m & orig))
            vs_orig[k]["union"] += int(np.sum(m | orig))
            tiles_identical[k] += int(np.array_equal(m, orig))
        n_pix += orig.size
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(imgs)}", flush=True)

    if kmodel is not None:
        res["keras32_vs_torch32"] = {
            "n_tiles": min(args.n_keras, len(imgs)),
            "max_abs_diff_probability": max_abs,
            "mean_abs_diff_probability": float(np.mean(mean_abs)),
            "mask_agreement": float(np.mean(k_agree)),
            "keras32_vs_original_R_mask_agreement": float(np.mean(k_vs_orig)),
        }
    for k in vs_orig:
        res[f"{k}_vs_original_R_masks"] = {
            "pixel_agreement": vs_orig[k]["agree"] / n_pix,
            "iou_water": vs_orig[k]["inter"] / vs_orig[k]["union"],
            "tiles_identical": tiles_identical[k],
        }
    res["scores_vs_reference"] = {k: prf(**v) for k, v in agg.items()}
    res["paper_table2_validation"] = {"precision": 0.934, "recall": 0.926, "f1": 0.930, "n_tiles": 6057}
    args.out_json.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
