# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the attention/m3_producer entry: the pre-wait-staged NVFP4 QKV + indexer producer.

Every cell is held to the native producer (``minimax_m3_nvfp4_qkv_indexer_norm_rope_kv_insert``) bit for bit on q,
index-q and every K / V / scale / index-K cache byte, and to the fp32 torch reference: the FP8 outputs within one E4M3
step, the decoded NVFP4 K / V within one E2M1 step of their block scale.
"""

import zlib

import pytest
import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8.ops.attention.m3_producer import (  # noqa: E501
    FP4_BLOCK,
    HEADS_Q,
    PAGE,
    decode_nvfp4_token,
    scale_index,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8.ops.attention.m3_producer import (  # noqa: E501
    m3_producer as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402, F401

assert torch.cuda.is_available(), "m3_producer requires a CUDA device"

PAGES = 1024
MAX_POSITIONS = 262144
EPS = 1e-6
KV_QUANT = (1.0, 0.37, 0.61)


def _caches():
    return (
        torch.zeros(PAGES, 2, 1, PAGE, 64, dtype=torch.uint8, device="cuda"),
        torch.zeros(PAGES, 2, 1, PAGE, 8, dtype=torch.uint8, device="cuda"),
        torch.zeros(PAGES, 1, PAGE, 128, dtype=torch.float8_e4m3fn, device="cuda"),
    )


def _inputs(tokens: int, scale: float, bad_slots: bool, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    rotary = torch.randn(MAX_POSITIONS, 2, 32, device="cuda", generator=g)
    weights = [(torch.randn(128, device="cuda", generator=g) * 0.3).bfloat16() for _ in range(4)]
    packed = (
        torch.randn(tokens, (HEADS_Q + 4) * 128, device="cuda", generator=g) * scale
    ).bfloat16()
    slots = torch.randperm(PAGES * PAGE, device="cuda", generator=g)[:tokens].int()
    if bad_slots:
        slots[1] = -1
        slots[2] = PAGES * PAGE + 5
    positions = torch.randint(
        0, MAX_POSITIONS, (tokens,), device="cuda", dtype=torch.int32, generator=g
    )
    quant = torch.tensor(KV_QUANT, device="cuda")
    return packed, slots, quant, weights, rotary, positions


def _args(inputs, caches, early: bool = True):
    packed, slots, quant, weights, rotary, positions = inputs
    return (packed, *caches, slots, quant, EPS, *weights, rotary, positions, HEADS_Q, True, early)


def _native(inputs, caches):
    packed, slots, quant, weights, rotary, positions = inputs
    return torch.ops.trtllm.minimax_m3_nvfp4_qkv_indexer_norm_rope_kv_insert(
        packed, *caches, slots, quant, HEADS_Q, 1, 1, 128, 64, EPS, *weights, rotary, positions
    )


def _within_fp8(out: torch.Tensor, ref: torch.Tensor, name: str) -> None:
    # One E4M3 step (3 mantissa bits) of the value, or its smallest subnormal step.
    err = (out.float() - ref).abs()
    assert bool((err <= ref.abs() * 2.0**-3 + 2.0**-9).all()), (name, float(err.max()))


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    spec = cell.spec
    early = spec.get("early", True)
    inputs = _inputs(
        spec["tokens"], spec["scale"], spec.get("bad_slots", False), zlib.crc32(cell.why.encode())
    )
    stock_caches, caches = _caches(), _caches()
    stock = _native(inputs, stock_caches)
    args = _args(inputs, caches, early)
    certifying(op)
    ref = op.reference(*args)
    out = op(*args)
    torch.cuda.synchronize()
    names = ("q", "index_q", "kv_data", "kv_scale", "index_k")
    for name, s, o in zip(names, (*stock, *stock_caches), (*out, *caches)):
        assert torch.equal(s.view(torch.uint8), o.view(torch.uint8)), (
            f"{name} differs from the native producer"
        )

    _within_fp8(out[0], ref["q"], "q")
    _within_fp8(out[1][:, 0], ref["index_q"], "index_q")
    kv_data, kv_scale, index_k = caches
    slots = inputs[1]
    checked = 0
    for t in range(spec["tokens"]):
        slot = int(slots[t])
        if slot < 0 or slot >= PAGES * PAGE:
            continue
        page, within = divmod(slot, PAGE)
        _within_fp8(index_k[page, 0, within], ref["index_k"][t], f"index_k token {t}")
        for plane, name in ((0, "k"), (1, "v")):
            scales = kv_scale[page, plane, 0].reshape(-1)[scale_index(within, plane).cuda()]
            dq = decode_nvfp4_token(kv_data[page, plane, 0, within], scales, KV_QUANT[1 + plane])
            r = ref[name][t].bfloat16().float()
            # Round to nearest E2M1 under the block's E4M3-rounded scale s (in value units s / quant_scale): at most
            # half the widest E2M1 step (one unit), plus the clipping at 6 units when s rounded down; a scale that
            # underflows E4M3 zeroes the block. A slack of one E4M3 step covers the kernel's approximate reciprocals.
            qs = KV_QUANT[1 + plane]
            amax = r.view(-1, FP4_BLOCK).abs().amax(dim=1)
            unit = (qs * amax / 6.0).to(torch.float8_e4m3fn).float() / qs
            bound = torch.where(
                unit > 0, 1.15 * unit + (amax - 6.0 * unit / 1.15).clamp_min(0.0), amax
            )
            bound = bound.repeat_interleave(FP4_BLOCK) + 1e-12
            assert bool(((dq - r).abs() <= bound).all()), (name, t)
        checked += 1
    assert checked >= spec["tokens"] - 2


def test_early_trigger_does_not_change_the_bits() -> None:
    inputs = _inputs(48, 1.0, False, 7)
    a, b = _caches(), _caches()
    out_a = op(*_args(inputs, a, early=True))
    out_b = op(*_args(inputs, b, early=False))
    for x, y in zip((*out_a, *a), (*out_b, *b)):
        assert torch.equal(x.view(torch.uint8), y.view(torch.uint8))


def test_guard_refuses_a_repeated_slot() -> None:
    inputs = list(_inputs(4, 1.0, False, 3))
    inputs[1] = inputs[1].clone()
    inputs[1][1] = inputs[1][0]
    with pytest.raises(AssertionError, match="must not repeat a slot"):
        with validating(op):
            op(*_args(tuple(inputs), _caches()))


def test_op_refuses_an_operand_off_the_device() -> None:
    inputs = list(_inputs(4, 1.0, False, 9))
    inputs[5] = inputs[5].cpu()  # position ids on the host
    with pytest.raises(RuntimeError, match="packed's CUDA device"):
        op(*_args(tuple(inputs), _caches()))


def test_op_refuses_an_unaligned_index_k_page_stride() -> None:
    """The index-K rows are written as 32-bit words: a page stride that is not a multiple of 4 is refused."""
    kv_data, kv_scale, _ = _caches()
    backing = torch.zeros(PAGES * (128 * 128 + 1), dtype=torch.float8_e4m3fn, device="cuda")
    index_k = backing.as_strided((PAGES, 1, PAGE, 128), (128 * 128 + 1, 128 * 128, 128, 1))
    with pytest.raises(RuntimeError, match="index_k"):
        op(*_args(_inputs(4, 1.0, False, 13), (kv_data, kv_scale, index_k)))


def test_op_refuses_unaligned_weights_or_rope_table() -> None:
    """The norm weights are read as 8-byte words and the RoPE rows as float4: offset views are refused."""
    packed, slots, quant, weights, rotary, positions = _inputs(4, 1.0, False, 17)
    offset_weight = torch.zeros(129, dtype=torch.bfloat16, device="cuda")[1:]
    with pytest.raises(RuntimeError, match="norm weights"):
        op(
            *_args(
                (packed, slots, quant, [offset_weight, *weights[1:]], rotary, positions), _caches()
            )
        )
    flat = torch.zeros(rotary.numel() + 1, device="cuda")
    offset_rope = flat[1:].view(rotary.shape)
    with pytest.raises(RuntimeError, match="rotary_cos_sin"):
        op(*_args((packed, slots, quant, weights, offset_rope, positions), _caches()))
