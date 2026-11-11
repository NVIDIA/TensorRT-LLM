# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_mla_rope_append_prefill entry.

The pool is synthetic: `[pages, layers, 32 tokens, 576]` bytes, filled with a
sentinel, addressed the way the KV block array is (block offset = page *
layers, and the layer's base `layer * 32 * 576` bytes in). Request `b` owns a
random set of pages covering its cached prefix and its new rows. All four
outputs (fused query, latent, fp8 query, the whole pool) are compared bit for
bit, so a stray write anywhere in the pool fails too.
"""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL kernel needs an sm_103 GPU", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.attention import (  # noqa: E402, E501
    cute_dsl_mla_rope_append_prefill as entry,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

op = entry.cute_dsl_mla_rope_append_prefill
HEADS, LATENT, ROPE, QK = entry.NUM_HEADS, entry.LATENT_DIM, entry.ROPE_DIM, entry.QK_HEAD_DIM
PAGE = 32
SENTINEL = 0xA5
#: The checkpoint's rope base (`rope_theta`).
THETA = 8_000_000.0


def _rope_table(positions: int) -> torch.Tensor:
    """The target's `rotary_cos_sin`: `[1, positions * 64 * 2]` fp32, pair `p` of row
    `pos` at `[pos, p] = (cos, sin)`, the first 32 pairs the 32 frequencies."""
    half = ROPE // 2
    inv = 1.0 / (THETA ** (torch.arange(half, dtype=torch.float64) * 2.0 / ROPE))
    ang = torch.arange(positions, dtype=torch.float64)[:, None] * inv[None, :]
    table = torch.empty(positions, ROPE, 2, dtype=torch.float64)
    table[:, :half, 0], table[:, half:, 0] = ang.cos(), ang.cos()
    table[:, :half, 1], table[:, half:, 1] = ang.sin(), ang.sin()
    return table.reshape(1, -1).float().cuda()


def _build(spec, seed):
    """The op's arguments for one context step."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    cpu = torch.Generator().manual_seed(seed)
    new, cached, layers = spec["new"], spec["cached"], spec["layers"]
    pages = [-(-(c + n) // PAGE) for c, n in zip(cached, new)]
    num_pages = sum(pages) + 1
    order = torch.randperm(num_pages, generator=cpu)
    offsets = torch.full((len(new), 2, max(pages) + 1), -1, dtype=torch.int32)
    used = 0
    for b, n in enumerate(pages):
        offsets[b, :, :n] = (order[used : used + n] * layers).to(torch.int32)
        used += n
    t = sum(new)
    q = torch.randn(t, HEADS, QK, generator=g, device="cuda").to(torch.bfloat16)
    cu = torch.tensor([0] + torch.tensor(new).cumsum(0).tolist(), dtype=torch.int32)
    return dict(
        # The absorbed half random, the rope tail garbage the kernel must overwrite.
        fused_q=torch.randn(t, HEADS, LATENT, generator=g, device="cuda").to(torch.bfloat16),
        q_pe=q[..., QK - ROPE :],
        latent_cache=torch.randn(t, LATENT, generator=g, device="cuda").to(torch.bfloat16),
        quant_q_buffer=torch.full(
            (t, HEADS, LATENT), SENTINEL, dtype=torch.uint8, device="cuda"
        ).view(torch.float8_e4m3fn),
        kv_pool=torch.full(
            (num_pages * layers * PAGE * LATENT,), SENTINEL, dtype=torch.uint8, device="cuda"
        ),
        layer_in_pool=spec["layer"],
        kv_cache_block_offsets=offsets.cuda(),
        cu_seqlens=cu.cuda(),
        cached_lens=torch.tensor(cached, dtype=torch.int32, device="cuda"),
        rotary_cos_sin=_rope_table(max(c + n for c, n in zip(cached, new)) + 3),
        tokens_per_block=PAGE,
    )


@pytest.fixture(scope="module")
def compiled():
    """The handle the target builds at its first forward."""
    return entry.compile(tokens_per_block=PAGE, device_index=torch.cuda.current_device())


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, compiled) -> None:
    """Exact on every byte. The reference reproduces each rounding the kernel makes:
    the rotation's two fp32 products, one of them fused (formed exactly in fp64 and
    rounded once, which is an fp32 FMA), then bf16 round-to-nearest-even; every e4m3
    value one nearest, saturating conversion of a bf16 value; everything else a
    copy. There is no accumulation to reorder, so nothing to give a band to."""
    args = _build(cell.spec, zlib.crc32(cell.why.encode()) % 2**31)
    ref = op.reference(**{k: v.clone() if torch.is_tensor(v) else v for k, v in args.items()},
                       compiled=compiled)  # fmt: skip
    with validating(op):
        got = op(**args, compiled=compiled)
    torch.cuda.synchronize()
    assert got[0] is args["fused_q"] and got[3] is args["kv_pool"]
    names = ("fused_q", "latent_cache", "quant_q_buffer", "kv_pool")
    for name, out, want in zip(names, got, ref):
        out = out.view(torch.uint8) if out.dtype == torch.float8_e4m3fn else out
        bad = int((out.reshape(-1) != want.reshape(-1)).sum())
        assert bad == 0, f"{name}: {bad} elements differ from the reference"


def test_the_op_refuses_a_layout_it_did_not_compile(compiled) -> None:
    """A strided latent selects another specialization, which would compile here."""
    args = _build(dict(new=[8], cached=[0], layers=1, layer=0), seed=1)
    wide = torch.zeros(8, 2624, dtype=torch.bfloat16, device="cuda")
    args["latent_cache"] = wide[:, 2624 - LATENT :]
    with pytest.raises(ValueError, match="packed"):
        op(**args, compiled=compiled)


_GUARD = dict(new=[5, 20], cached=[30, 0], layers=2, layer=1)


def _handle() -> "entry.CompiledMlaRopeAppendPrefill":
    return entry.CompiledMlaRopeAppendPrefill(torch.cuda.current_device(), PAGE, None)


def test_is_valid_admits_a_well_formed_call() -> None:
    op.is_valid(**_build(_GUARD, 2), compiled=_handle())


def test_is_valid_refuses_a_prefix_sum_short_of_the_rows() -> None:
    """The kernel puts rows past `cu[B]` in the last request, at positions past it."""
    args = _build(_GUARD, 3)
    args["cu_seqlens"][-1] -= 1
    with pytest.raises(AssertionError, match="cu_seqlens must rise"):
        op.is_valid(**args, compiled=_handle())


def test_is_valid_refuses_a_block_past_the_offsets() -> None:
    """The kernel skips that row's append and says nothing."""
    args = _build(_GUARD, 4)
    args["kv_cache_block_offsets"] = args["kv_cache_block_offsets"][:, :, :1].contiguous()
    with pytest.raises(AssertionError, match="past the offsets' width"):
        op.is_valid(**args, compiled=_handle())


def test_is_valid_refuses_a_position_past_the_rope_table() -> None:
    """The table is read with no bound."""
    args = _build(_GUARD, 5)
    args["rotary_cos_sin"] = args["rotary_cos_sin"][:, : 20 * ROPE * 2]
    with pytest.raises(AssertionError, match="past the rope table"):
        op.is_valid(**args, compiled=_handle())
