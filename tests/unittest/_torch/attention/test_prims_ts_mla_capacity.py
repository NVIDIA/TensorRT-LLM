# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only capacity contracts, runnable without the TensorRT-LLM extension."""

import ast
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.cpu_only

_ROOT = Path(__file__).resolve().parents[4]
_ADAPTER = _ROOT / "tensorrt_llm/_torch/attention/backends/fmha/prims_ts.py"


def _load_capacities() -> Callable[..., tuple[int, int]]:
    tree = ast.parse(_ADAPTER.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_mla_context_capacities"
    )
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(_ADAPTER), "exec"), namespace)
    return namespace[function.name]


_capacities = _load_capacities()
_GEOMETRY = dict(
    head_dim=192,
    head_dim_vo=128,
    qk_dtype=torch.float8_e4m3fn,
    pv_dtype=torch.float8_e4m3fn,
    output_dtype=torch.bfloat16,
    packed=True,
    uniform_packed_lengths=False,
    mask_type="causal",
    window_left=-1,
)


@pytest.mark.parametrize(
    "query_length,kv_length",
    [
        (1, 0),
        (127, 1),
        (128, 127),
        (129, 128),
        (255, 129),
        (256, 255),
        (257, 256),
        (2048, 257),
        (1, 4607),
    ],
)
@pytest.mark.parametrize("mask_type", ["causal", "dense"])
def test_capacity_floor_preserves_tile_counts(
    query_length: int, kv_length: int, mask_type: str
) -> None:
    actual = _capacities(query_length, kv_length, **(_GEOMETRY | {"mask_type": mask_type}))
    original = tuple(1 << (max(length, 1) - 1).bit_length() for length in (query_length, kv_length))
    assert actual == (max(original[0], 256), max(original[1], 128))
    assert (actual[0] + 255) // 256 == (original[0] + 255) // 256
    assert (actual[1] + 127) // 128 == (original[1] + 127) // 128
    assert actual[0] >= query_length and actual[1] >= kv_length


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("output_dtype", [torch.float16, torch.bfloat16])
def test_capacity_floor_supported_precisions(dtype: torch.dtype, output_dtype: torch.dtype) -> None:
    assert _capacities(
        1,
        1,
        **(_GEOMETRY | {"qk_dtype": dtype, "pv_dtype": dtype, "output_dtype": output_dtype}),
    ) == (256, 128)


@pytest.mark.parametrize(
    "override",
    [
        {"head_dim": 128},
        {"head_dim_vo": 192},
        {"packed": False},
        {"uniform_packed_lengths": True},
        {"mask_type": "variable_window"},
        {"window_left": 128},
        {"qk_dtype": torch.bfloat16},
        {"pv_dtype": torch.bfloat16},
        {"qk_dtype": torch.float32, "pv_dtype": torch.float32},
        {"output_dtype": torch.float8_e4m3fn},
    ],
)
def test_unproved_geometry_keeps_original_buckets(override: dict) -> None:
    assert _capacities(3, 5, **(_GEOMETRY | override)) == (4, 8)
