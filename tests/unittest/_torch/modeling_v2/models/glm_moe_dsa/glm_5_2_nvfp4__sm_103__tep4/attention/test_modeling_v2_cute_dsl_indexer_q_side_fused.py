# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_indexer_q_side_fused entry."""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL kernel is built for sm_103", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.attention.cute_dsl_indexer_q_side_fused import (  # noqa: E402, E501
    build,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.attention.cute_dsl_indexer_q_side_fused import (  # noqa: E402, E501
    cute_dsl_indexer_q_side_fused as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

#: The target's indexer geometry and its folded softmax scale.
HEADS, HEAD_DIM, ROPE_DIM = 32, 128, 64
WEIGHT_SCALE = HEAD_DIM**-0.5 * HEADS**-0.5
THETA = 8.0e6  # the checkpoint's `rope_theta`

_COMPILED: dict = {}


def _compiled(is_neox: bool):
    if is_neox not in _COMPILED:
        _COMPILED[is_neox] = build(torch.cuda.current_device(), is_neox=is_neox)
    return _COMPILED[is_neox]


def _rope_table(positions: int) -> torch.Tensor:
    """`[positions, 64]` fp32 `[cos | sin]`, built in fp64 and rounded once, as the target does."""
    half = ROPE_DIM // 2
    inv = 1.0 / (THETA ** (torch.arange(half, dtype=torch.float64) * 2.0 / ROPE_DIM))
    ang = torch.arange(positions, dtype=torch.float64)[:, None] * inv[None, :]
    return torch.cat([ang.cos(), ang.sin()], dim=1).float().cuda().contiguous()


def _build(spec, seed):
    """`(q, position_ids, cos_sin, wq, weight_scale, q_fp8_out, weights_out)` for one cell.

    Both outputs start poisoned, so a row the kernel skipped cannot pass.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    tokens, max_pos = spec["tokens"], spec["max_pos"]
    q = (torch.randn(tokens, HEADS, HEAD_DIM, generator=g, device="cuda") * 2.0).to(torch.bfloat16)
    if spec["positions"] == "arange":
        pos = torch.arange(tokens, device="cuda") % max_pos
    elif spec["positions"] == "tail":
        pos = max_pos - 1 - torch.randint(0, 64, (tokens,), generator=g, device="cuda")
    else:
        pos = torch.randint(0, max_pos, (tokens,), generator=g, device="cuda")
    wq = torch.randn(tokens, HEADS, generator=g, device="cuda")
    q_fp8_out = torch.full((tokens, HEADS, HEAD_DIM), 0x7F, dtype=torch.uint8, device="cuda").view(
        torch.float8_e4m3fn
    )
    weights_out = torch.full((tokens, HEADS), float("nan"), device="cuda")
    return (
        q,
        pos.to(torch.int32),
        _rope_table(max_pos),
        wq,
        WEIGHT_SCALE,
        q_fp8_out,
        weights_out,
    )


def _compare(out, ref) -> None:
    """Bit-exact codes and weights.

    The reference reproduces every rounding the kernel makes: the rope FMA
    (fp64 rounded once to fp32), the bf16 store of the rotated half, the
    power-of-two scale from exponent bits, an exact multiply by its inverse,
    the e4m3 cast, and two fp32 products for the weights. Nothing is left to
    accumulation order, so the band is zero; a wrong rope mode or table moves
    most codes.
    """
    (q_fp8, weights), (ref_fp8, ref_weights) = out, ref
    codes, want = q_fp8.view(torch.uint8), ref_fp8.view(torch.uint8)
    bad = codes != want
    assert not bool(bad.any()), (
        f"{int(bad.sum())} of {bad.numel()} e4m3 codes differ; first at "
        f"{tuple(int(i) for i in bad.nonzero()[0])}"
    )
    wbad = weights.view(torch.int32) != ref_weights.view(torch.int32)
    assert not bool(wbad.any()), f"{int(wbad.sum())} of {wbad.numel()} weights differ"


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    kw = dict(is_neox=cell.spec["is_neox"], compiled=_compiled(cell.spec["is_neox"]))
    q_before = args[0].clone()
    with validating(op):
        out = op(*args, **kw)
    assert out[0] is args[5] and out[1] is args[6], "the op writes the caller's buffers"
    assert torch.equal(args[0], q_before), "q must not be written"
    _compare(out, op.reference(*args, **kw))


def test_a_fresh_launch_overwrites_every_row() -> None:
    """Two calls into the same buffers: the second result is the second input's."""
    spec = dict(tokens=5, positions="spread", max_pos=1024, is_neox=False)
    first, second = _build(spec, seed=31), _build(spec, seed=32)
    kw = dict(is_neox=False, compiled=_compiled(False))
    out_fp8, out_w = first[5], first[6]
    op(*first, **kw)
    op(*second[:5], out_fp8, out_w, **kw)
    _compare((out_fp8, out_w), op.reference(*second, **kw))


def test_guard_refuses_a_rope_mode_the_launcher_was_not_built_for() -> None:
    args = _build(dict(tokens=4, positions="spread", max_pos=256, is_neox=False), seed=33)
    with pytest.raises(AssertionError, match="compiled for is_neox"):
        with validating(op):
            op(*args, is_neox=True, compiled=_compiled(False))


def test_guard_refuses_a_position_past_the_rope_table() -> None:
    q, pos, table, wq, c, q_out, w_out = _build(
        dict(tokens=4, positions="spread", max_pos=256, is_neox=False), seed=34
    )
    pos[2] = table.shape[0]
    with pytest.raises(AssertionError, match="rope table"):
        with validating(op):
            op(q, pos, table, wq, c, q_out, w_out, is_neox=False, compiled=_compiled(False))
