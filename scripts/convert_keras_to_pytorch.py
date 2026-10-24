"""Convert the original Keras (R keras / TF 2.10) weights to PyTorch.

The original model was trained in R with the keras package and saved with
``save_model_weights_hdf5()``: the .h5 file holds only weights, no graph.
We rebuild the exact same graph in Python Keras (same layer creation order as the
R ``get_unet_128()`` function), load the .h5, then copy every layer into the
PyTorch model by its role (encoder level, decoder level, conv index), not by
position, so a wrong mapping cannot happen silently.

Requires TensorFlow 2.x with Keras 2 (tested with tensorflow-cpu 2.15.1).
This script is only needed once, to produce the published PyTorch weights.

Usage:
    python scripts/convert_keras_to_pytorch.py KERAS_WEIGHTS.h5 OUT_DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from s1water.model import CENTER_FILTERS, ENCODER_FILTERS, UNetS1Water


def build_keras_unet(crop: int = 0):
    """Python copy of the R keras get_unet_128() (input c(NA, NA, 4), 1 class).

    Returns the keras model and a dict {role: keras layer} for weight mapping.
    """
    from tensorflow import keras
    from tensorflow.keras import layers as L

    roles = {}

    def conv_bn_relu(x, filters, role):
        conv = L.Conv2D(filters, (3, 3), padding="same")
        bn = L.BatchNormalization()
        roles[role] = (conv, bn)
        return L.Activation("relu")(bn(conv(x)))

    inputs = keras.Input(shape=(None, None, 4))
    x = inputs
    skips = []
    for i, f in enumerate(ENCODER_FILTERS):
        x = conv_bn_relu(x, f, f"down{i}.0")
        x = conv_bn_relu(x, f, f"down{i}.1")
        skips.append(x)
        x = L.MaxPooling2D((2, 2), strides=(2, 2))(x)
        x = L.Dropout(0.5)(x)
    x = conv_bn_relu(x, CENTER_FILTERS, "center.0")
    x = conv_bn_relu(x, CENTER_FILTERS, "center.1")
    for j, f in enumerate(reversed(ENCODER_FILTERS)):
        x = L.UpSampling2D((2, 2))(x)
        x = L.Concatenate(axis=3)([skips[-1 - j], x])
        for k in range(3):
            x = conv_bn_relu(x, f, f"up{j}.{k}")
    x = L.Cropping2D(cropping=crop)(x)
    final = L.Conv2D(1, (1, 1), activation="sigmoid", dtype="float32")
    roles["classify"] = (final, None)
    outputs = final(x)
    return keras.Model(inputs, outputs), roles


def torch_target(model: UNetS1Water, role: str):
    """Return (conv, bn) PyTorch modules for a role name used in build_keras_unet."""
    if role == "classify":
        return model.classify, None
    name, idx = role.split(".")
    idx = int(idx)
    if name == "center":
        block = model.center[idx]
    elif name.startswith("down"):
        block = model.down[int(name[4:])][idx]
    else:
        block = model.up[int(name[2:])][idx]
    return block[0], block[1]


def copy_weights(keras_roles: dict, model: UNetS1Water) -> int:
    """Copy all Keras weights into the PyTorch model, return number of tensors copied."""
    n = 0
    for role, (kconv, kbn) in keras_roles.items():
        tconv, tbn = torch_target(model, role)
        kernel, bias = kconv.get_weights()
        # Keras kernel (kh, kw, in, out) -> PyTorch (out, in, kh, kw)
        assert tuple(tconv.weight.shape) == (kernel.shape[3], kernel.shape[2], kernel.shape[0], kernel.shape[1]), role
        tconv.weight.data = torch.from_numpy(np.ascontiguousarray(kernel.transpose(3, 2, 0, 1)))
        tconv.bias.data = torch.from_numpy(bias)
        n += 2
        if kbn is not None:
            gamma, beta, mean, var = kbn.get_weights()
            assert abs(kbn.epsilon - tbn.eps) < 1e-12, role
            tbn.weight.data = torch.from_numpy(gamma)
            tbn.bias.data = torch.from_numpy(beta)
            tbn.running_mean.data = torch.from_numpy(mean)
            tbn.running_var.data = torch.from_numpy(var)
            n += 4
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("keras_weights", type=Path)
    p.add_argument("out_dir", type=Path)
    args = p.parse_args()

    kmodel, roles = build_keras_unet(crop=0)
    # the .h5 is a weights-only file saved by R keras: load by topological order,
    # which is identical because the graph is built in the same order
    kmodel.load_weights(str(args.keras_weights))

    model = UNetS1Water()
    n = copy_weights(roles, model)
    n_torch = len(model.state_dict()) - sum(1 for k in model.state_dict() if k.endswith("num_batches_tracked"))
    assert n == n_torch, (n, n_torch)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    torch.save(state, args.out_dir / "s1_water_unet.pt")
    from safetensors.torch import save_file

    save_file(state, str(args.out_dir / "s1_water_unet.safetensors"))
    info = {"source": args.keras_weights.name, "tensors": n, "parameters": int(sum(p.numel() for p in model.parameters()))}
    (args.out_dir / "conversion_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
