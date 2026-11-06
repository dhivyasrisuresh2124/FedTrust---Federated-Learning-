"""Compact wire format for numpy arrays exchanged over HTTP/JSON."""
from __future__ import annotations

import base64
from typing import Any

import numpy as np

_ALLOWED = {"float16", "float32", "float64", "int32", "int64"}


def encode_array(a: np.ndarray, dtype: str = "float16") -> dict[str, Any]:
    if dtype not in _ALLOWED:
        raise ValueError(f"unsupported wire dtype {dtype!r}")
    arr = np.ascontiguousarray(np.asarray(a, dtype=dtype))
    return {
        "dtype": dtype,
        "shape": list(arr.shape),
        "b64": base64.b64encode(arr.tobytes()).decode("ascii"),
    }


def decode_array(d: dict[str, Any]) -> np.ndarray:
    dtype = d["dtype"]
    if dtype not in _ALLOWED:
        raise ValueError(f"unsupported wire dtype {dtype!r}")
    raw = base64.b64decode(d["b64"])
    arr = np.frombuffer(raw, dtype=dtype).reshape(d["shape"])
    return arr.astype(np.float32) if arr.dtype.kind == "f" else arr.copy()
