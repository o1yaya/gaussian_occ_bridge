from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an ILGS classifier.pth to the NumPy format used by the PLY adapter."
    )
    parser.add_argument("--input", required=True, help="Path to classifier.pth")
    parser.add_argument("--output", required=True, help="Output classifier.npz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required only for this one-time checkpoint conversion. "
            "Run the script in the ILGS training environment."
        ) from error

    state = torch.load(args.input, map_location="cpu")
    if "weight" not in state:
        raise KeyError("classifier checkpoint does not contain a weight tensor")
    weight = state["weight"].detach().cpu().numpy()
    if weight.ndim == 4 and weight.shape[2:] == (1, 1):
        weight = weight[:, :, 0, 0]
    if weight.ndim != 2:
        raise ValueError(f"expected weight [C, D] or [C, D, 1, 1], got {weight.shape}")
    if state.get("bias") is None:
        bias = np.zeros(weight.shape[0], dtype=np.float32)
    else:
        bias = state["bias"].detach().cpu().numpy()
    if bias.shape != (weight.shape[0],):
        raise ValueError(f"expected bias shape {(weight.shape[0],)}, got {bias.shape}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, weight=weight.astype(np.float32), bias=bias.astype(np.float32))
    print(
        json.dumps(
            {
                "output": str(output_path),
                "class_count": int(weight.shape[0]),
                "object_feature_channels": int(weight.shape[1]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

