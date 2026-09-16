from __future__ import annotations

import argparse
import collections
import io
import json
import pickle
import zipfile
from pathlib import Path

import numpy as np


class _FloatStorage:
    pass


class _Storage:
    def __init__(self, values: np.ndarray):
        self.values = values


def _rebuild_tensor_v2(
    storage: _Storage,
    storage_offset: int,
    size: tuple[int, ...],
    stride: tuple[int, ...],
    requires_grad: bool,
    backward_hooks: object,
) -> np.ndarray:
    del requires_grad, backward_hooks
    itemsize = storage.values.dtype.itemsize
    base = storage.values[int(storage_offset) :]
    return np.lib.stride_tricks.as_strided(
        base,
        shape=tuple(int(value) for value in size),
        strides=tuple(int(value) * itemsize for value in stride),
    ).copy()


class _RestrictedTorchUnpickler(pickle.Unpickler):
    def __init__(self, stream: io.BytesIO, archive: zipfile.ZipFile, prefix: str):
        super().__init__(stream)
        self.archive = archive
        self.prefix = prefix

    def find_class(self, module: str, name: str) -> object:
        allowed = {
            ("collections", "OrderedDict"): collections.OrderedDict,
            ("torch", "FloatStorage"): _FloatStorage,
            ("torch._utils", "_rebuild_tensor_v2"): _rebuild_tensor_v2,
        }
        try:
            return allowed[(module, name)]
        except KeyError as error:
            raise pickle.UnpicklingError(f"unsupported checkpoint global: {module}.{name}") from error

    def persistent_load(self, identifier: tuple[object, ...]) -> _Storage:
        if len(identifier) != 5 or identifier[0] != "storage":
            raise pickle.UnpicklingError(f"unsupported persistent id: {identifier}")
        _, storage_type, key, _location, size = identifier
        if storage_type is not _FloatStorage:
            raise pickle.UnpicklingError("only FloatStorage checkpoints are supported")
        raw = self.archive.read(f"{self.prefix}/data/{key}")
        values = np.frombuffer(raw, dtype="<f4", count=int(size)).copy()
        return _Storage(values)


def _load_restricted_state_dict(path: Path) -> collections.OrderedDict:
    if not zipfile.is_zipfile(path):
        raise ValueError("checkpoint is not a PyTorch zip archive; convert it with PyTorch")
    with zipfile.ZipFile(path) as archive:
        data_pickle = next(
            (name for name in archive.namelist() if name.endswith("/data.pkl")), None
        )
        if data_pickle is None:
            raise ValueError("checkpoint archive does not contain data.pkl")
        prefix = data_pickle.rsplit("/", 1)[0]
        state = _RestrictedTorchUnpickler(
            io.BytesIO(archive.read(data_pickle)), archive, prefix
        ).load()
    if not isinstance(state, collections.OrderedDict):
        raise ValueError("checkpoint root is not an OrderedDict state_dict")
    return state


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
    except ImportError:
        torch = None

    input_path = Path(args.input)
    if torch is None:
        state = _load_restricted_state_dict(input_path)
        loader = "restricted-numpy"
    else:
        state = torch.load(input_path, map_location="cpu", weights_only=True)
        loader = "torch"
    if "weight" not in state:
        raise KeyError("classifier checkpoint does not contain a weight tensor")
    weight_value = state["weight"]
    weight = (
        weight_value.detach().cpu().numpy()
        if hasattr(weight_value, "detach")
        else np.asarray(weight_value)
    )
    if weight.ndim == 4 and weight.shape[2:] == (1, 1):
        weight = weight[:, :, 0, 0]
    if weight.ndim != 2:
        raise ValueError(f"expected weight [C, D] or [C, D, 1, 1], got {weight.shape}")
    if state.get("bias") is None:
        bias = np.zeros(weight.shape[0], dtype=np.float32)
    else:
        bias_value = state["bias"]
        bias = (
            bias_value.detach().cpu().numpy()
            if hasattr(bias_value, "detach")
            else np.asarray(bias_value)
        )
    if bias.shape != (weight.shape[0],):
        raise ValueError(f"expected bias shape {(weight.shape[0],)}, got {bias.shape}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, weight=weight.astype(np.float32), bias=bias.astype(np.float32))
    print(
        json.dumps(
            {
                "output": str(output_path),
                "loader": loader,
                "class_count": int(weight.shape[0]),
                "object_feature_channels": int(weight.shape[1]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
