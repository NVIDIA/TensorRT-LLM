# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the attention/m3_qkv_insert entry: the generic horizontal producer on an FP8 main cache.

Q, index-Q and every written K / V / index-K cache row are held within one E4M3 step of the fp32 reference.
"""

import zlib

import pytest
import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.attention.m3_qkv_insert import (  # noqa: E501
    m3_qkv_insert as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402, F401

assert torch.cuda.is_available(), "m3_qkv_insert requires a CUDA device"

PAGES, PAGE, HEADS, KV, INDEX, D, ROTARY = 256, 128, 16, 1, 1, 128, 64
MAX_POSITIONS = 262144
EPS = 1e-6


def _within_fp8(out: torch.Tensor, ref: torch.Tensor, name: str) -> None:
    # One E4M3 step (3 mantissa bits) of the value, or its smallest subnormal step.
    err = (out.float() - ref).abs()
    assert bool((err <= ref.abs() * 2.0**-3 + 2.0**-9).all()), (name, float(err.max()))


def _args(tokens: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    packed = torch.randn(
        tokens, (HEADS + 2 * KV + INDEX + 1) * D, device="cuda", generator=g
    ).bfloat16()
    kv = torch.zeros(PAGES, 2, KV, PAGE, D, dtype=torch.float8_e4m3fn, device="cuda")
    index_k = torch.zeros(PAGES, 1, PAGE, D, dtype=torch.float8_e4m3fn, device="cuda")
    slots = torch.randperm(PAGES * PAGE, device="cuda", generator=g)[:tokens].int()
    weights = [(torch.randn(D, device="cuda", generator=g) * 0.3).bfloat16() for _ in range(4)]
    rotary = torch.randn(MAX_POSITIONS, 2, ROTARY // 2, device="cuda", generator=g)
    positions = torch.randint(
        0, MAX_POSITIONS, (tokens,), device="cuda", dtype=torch.int32, generator=g
    )
    return (
        packed,
        kv,
        None,
        index_k,
        slots,
        None,
        HEADS,
        KV,
        INDEX,
        D,
        ROTARY,
        EPS,
        *weights,
        rotary,
        positions,
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    args = _args(cell.spec["m"], zlib.crc32(cell.why.encode()))
    certifying(op)
    ref = op.reference(*args)
    q, index_q = op(*args)
    torch.cuda.synchronize()
    _within_fp8(q.view(ref["q"].shape), ref["q"], "q")
    _within_fp8(index_q.view(ref["index_q"].shape), ref["index_q"], "index_q")
    kv, index_k, slots = args[1], args[3], args[4]
    for t in range(cell.spec["m"]):
        page, within = divmod(int(slots[t]), PAGE)
        _within_fp8(kv[page, 0, :, within], ref["k"][t], f"k token {t}")
        _within_fp8(kv[page, 1, :, within], ref["v"][t], f"v token {t}")
        _within_fp8(index_k[page, 0, within], ref["index_k"][t], f"index_k token {t}")
