# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the target's attention/k3_draft_rope op (sm_100).

Every cell runs on three copies of the same q / k: through ``__call__`` (with ``is_valid`` interposed), through the
stock ``flashinfer_apply_rope_with_cos_sin_cache_inplace`` call the built-in DFlash drafter makes (the op must give
its bits), and through ``reference`` (within one bf16 spacing). A cell with 0 query heads is the drafter's context K:
the caller passes a one-head scratch q, as the built-in drafter does.

The tables are built the way the built-in drafter builds them, from ``RopeParams``: "plain" is NeoX RoPE with theta
1e4 and head 64; "yarn" adds a YaRN scaling of factor 16 over 65536 original positions (a qwen3-like drafter config).
Both stop at ``MAX_POS`` positions, which ``RopeParams.from_config`` takes from ``max_position_embeddings``.
"""

import importlib
import types
import zlib

import pytest
import torch

assert torch.cuda.is_available(), "k3_draft_rope requires a CUDA device"

if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
    pytest.skip("k3_draft_rope is certified on sm_100 and sm_103", allow_module_level=True)

_rope = importlib.import_module(
    "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl."
    "kimi_k3_mxfp4__sm_100__tp16_moetp4ep4.ops.attention.k3_draft_rope"
)
k3_draft_rope = _rope.k3_draft_rope

if not _rope.available():
    pytest.skip("flashinfer's RoPE kernel is not registered in this build", allow_module_level=True)

from tensorrt_llm._torch.attention.backends.interface import RopeParams  # noqa: E402
from tensorrt_llm._torch.custom_ops import (  # noqa: E402
    flashinfer_apply_rope_with_cos_sin_cache_inplace as stock_rope,
)

__extra_import_path__ = ["../../.."]
from _validating import assert_within_ulp, validating  # noqa: E402

D = 64
MAX_POS = 4096
YARN = {
    "rope_type": "yarn",
    "factor": 16.0,
    "original_max_position_embeddings": 65536,
    "rope_theta": 10000.0,
}


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _table(kind: str) -> torch.Tensor:
    """[MAX_POS, D] fp32, ``[cos | sin]`` halves, as the built-in drafter builds it from its RopeParams."""
    cfg = types.SimpleNamespace(
        model_type="qwen3",
        hidden_size=64 * D,
        num_attention_heads=64,
        head_dim=D,
        max_position_embeddings=MAX_POS,
        rope_theta=10000.0,
        rope_scaling=dict(YARN) if kind == "yarn" else None,
    )
    params = RopeParams.from_config(cfg)
    assert params.max_positions == MAX_POS and params.dim == D
    cos_sin = params.create_rope_const_params(interleave=False)[1]
    table = cos_sin.reshape(MAX_POS, 2, -1).view(MAX_POS, -1).float().contiguous()
    assert table.shape == (MAX_POS, D)
    return table


_TABLES: dict = {}


def _get_table(kind: str) -> torch.Tensor:
    if kind not in _TABLES:
        _TABLES[kind] = _table(kind)
    return _TABLES[kind]


def _run_all(positions, q, k, table, is_neox):
    """The entry, the stock call and the reference, each on its own copy; returns the three (q, k) pairs."""
    q_op, k_op = q.clone(), k.clone()
    with validating(k3_draft_rope):
        assert k3_draft_rope(positions, q_op, k_op, D, table, is_neox) is None
    q_st, k_st = q.clone(), k.clone()
    stock_rope(positions, q_st, k_st, D, table, is_neox)
    q_ref, k_ref = q.clone(), k.clone()
    k3_draft_rope.reference(positions, q_ref, k_ref, D, table, is_neox)
    return (q_op, k_op), (q_st, k_st), (q_ref, k_ref)


def _gate(got, stock, want) -> None:
    for name, a, s, r in zip("qk", got, stock, want):
        assert torch.equal(_bits(a), _bits(s)), (
            f"{name}: the entry differs from the stock flashinfer call"
        )
        # Both sides rotate in fp32 and round once to bf16; fma contraction moves the fp32 value by a few ulps, which
        # flips the bf16 rounding of an element by at most one spacing (<= 2^-7 of the row's largest magnitude, i.e.
        # 2 units of 2^-8). Such flips need the fp32 value within a few ulps of a rounding boundary, so they are rare
        # and the relative RMS stays far under one unit.
        assert_within_ulp(a, r, element_ulp=2.0, rms_ulp=1.0)


def _inputs(seed: int, heads: int, kv: int, rows: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    positions = torch.randint(0, MAX_POS, (rows,), generator=gen, device="cuda", dtype=torch.int32)
    # heads == 0: the context-K call, with a one-head scratch q as the built-in drafter passes.
    q = torch.randn(rows, max(heads, 1) * D, generator=gen, device="cuda").bfloat16()
    k = torch.randn(rows, kv * D, generator=gen, device="cuda").bfloat16()
    return positions, q, k


@pytest.mark.parametrize("cell", k3_draft_rope.CELLS, ids=[c.why for c in k3_draft_rope.CELLS])
def test_k3_draft_rope(cell) -> None:
    spec = cell.spec
    seed = zlib.crc32(cell.why.encode()) % 2**31
    positions, q, k = _inputs(seed, spec["heads"], spec["kv"], spec["rows"])
    table = _get_table(spec["table"])
    got, stock, want = _run_all(positions, q, k, table, True)
    _gate(got, stock, want)
    assert not torch.equal(_bits(got[1]), _bits(k)), "k was not rotated"


@pytest.mark.parametrize("table", ["plain", "yarn"])
def test_gptj_layout(table) -> None:
    """``is_neox=False`` (pairs of adjacent elements), which the docstring states and no cell drives."""
    positions, q, k = _inputs(11, 4, 1, 56)
    got, stock, want = _run_all(positions, q, k, _get_table(table), False)
    _gate(got, stock, want)


_OUT_OF_CONTRACT = {
    "bf16 table": lambda p, q, k, t: (p, q, k, t.bfloat16()),
    "table width != head_size": lambda p, q, k, t: (p, q, k, t[:, : D // 2]),
    "int64 positions": lambda p, q, k, t: (p.long(), q, k, t),
    "rows differ": lambda p, q, k, t: (p, q[:-1], k, t),
    "strided head dim": lambda p, q, k, t: (p, q, k.repeat(1, 2)[:, ::2], t),
}


@pytest.mark.parametrize("case", list(_OUT_OF_CONTRACT))
def test_rejects_out_of_contract(case) -> None:
    """The guard refuses the inputs it lists (it raises before the kernel runs)."""
    positions, q, k = _inputs(3, 4, 1, 8)
    args = _OUT_OF_CONTRACT[case](positions, q, k, _get_table("plain"))
    with validating(k3_draft_rope), pytest.raises(AssertionError):
        k3_draft_rope(args[0], args[1], args[2], D, args[3], True)
