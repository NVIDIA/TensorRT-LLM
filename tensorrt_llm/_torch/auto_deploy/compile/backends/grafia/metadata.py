# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FX metadata and dtype helpers for Grafia lowering."""

from __future__ import annotations

from typing import Any

import torch
from torch.fx import GraphModule, Node

from .errors import GrafiaUnsupportedError


def _torch_dtype_to_ctm(dtype: torch.dtype, types_mod):
    mapping = {
        torch.float32: types_mod.DType.FP32,
        torch.float16: types_mod.DType.FP16,
        torch.bfloat16: types_mod.DType.BF16,
        torch.int64: types_mod.DType.INT64,
        torch.int32: types_mod.DType.INT32,
        torch.int8: types_mod.DType.INT8,
        torch.uint8: types_mod.DType.UINT8,
        torch.bool: types_mod.DType.BOOL,
    }
    try:
        return mapping[dtype]
    except KeyError:
        raise GrafiaUnsupportedError(
            f"compile_backend='grafia' does not support dtype {dtype}"
        ) from None


def _ctm_dtype_to_torch(dtype) -> torch.dtype:
    name = getattr(dtype, "name", None)
    mapping = {
        "FP32": torch.float32,
        "FP16": torch.float16,
        "BF16": torch.bfloat16,
        "INT64": torch.int64,
        "INT32": torch.int32,
        "INT8": torch.int8,
        "UINT8": torch.uint8,
        "BOOL": torch.bool,
    }
    try:
        return mapping[name]
    except KeyError:
        raise GrafiaUnsupportedError(
            f"compile_backend='grafia' cannot map CTM dtype {dtype} to torch"
        ) from None


def _node_meta_val(node: Node) -> Any:
    val = node.meta.get("val")
    if val is None:
        raise GrafiaUnsupportedError(
            f"FX node {node.name!r} is missing meta['val']; "
            "compile_backend='grafia' requires canonical AutoDeploy FX with "
            "static tensor metadata."
        )
    return val


def _shape_from_meta(node: Node) -> tuple[int, ...]:
    val = _node_meta_val(node)
    try:
        return tuple(int(d) for d in val.shape)
    except Exception as exc:
        raise GrafiaUnsupportedError(
            f"FX node {node.name!r} has unsupported or symbolic shape metadata: "
            f"{getattr(val, 'shape', None)!r}"
        ) from exc


def _dtype_from_meta(node: Node) -> torch.dtype:
    val = _node_meta_val(node)
    dtype = getattr(val, "dtype", None)
    if not isinstance(dtype, torch.dtype):
        raise GrafiaUnsupportedError(f"FX node {node.name!r} has invalid dtype metadata: {dtype!r}")
    return dtype


def _is_contiguous_meta(node: Node) -> bool:
    val = _node_meta_val(node)
    is_contiguous = getattr(val, "is_contiguous", None)
    if callable(is_contiguous):
        return bool(is_contiguous())
    shape = _shape_from_meta(node)
    stride_fn = getattr(val, "stride", None)
    if not callable(stride_fn):
        return False
    expected = 1
    for dim in range(len(shape) - 1, -1, -1):
        if int(stride_fn(dim)) != expected:
            return False
        expected *= shape[dim]
    return True


def _get_attr(gm: GraphModule, target: str) -> Any:
    obj: Any = gm
    for atom in target.split("."):
        obj = getattr(obj, atom)
    return obj
