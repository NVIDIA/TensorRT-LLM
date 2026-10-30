# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_fp8_paged_mqa_logits entry."""

import math
import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL kernel is built for sm_103", allow_module_level=True)

from tensorrt_llm import deep_gemm  # noqa: E402
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.attention.cute_dsl_fp8_paged_mqa_logits import (  # noqa: E402, E501
    cute_dsl_fp8_paged_mqa_logits as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

E4M3_MAX = 448.0
#: The schedule block the DSA indexer builds its decode schedule with.
SCHEDULE_BLOCK_KV = 64

# Gate, in units of the row's largest |logit|. See `_compare`.
ELEMENT_TOL = 2.0**-12
RMS_TOL = 2.0**-14


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """2**-12 of the row's largest |logit| per element, 2**-14 relative RMS,
    over the causal part of each row only.

    Same arithmetic as `fp8_mqa_logits`: exact fp8 products, then fp32 sums
    over 128 dims and 32 heads, so kernel and reference differ by a few
    fp32 ulps of the row scale; a wrong formula moves logits by O(1).
    Columns past the causal end hold whatever the shared buffer held and are
    not compared: NaN in `ref` means unspecified.
    """
    assert out.dtype == ref.dtype == torch.float32, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    live = ~torch.isnan(ref)
    assert bool(torch.isfinite(out[live]).all()), "a causal entry is not finite"
    o = torch.where(live, out, torch.zeros_like(out)).double()
    r = torch.where(live, ref, torch.zeros_like(ref)).double()
    row_scale = r.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
    err = (o - r).abs() / row_scale
    assert float(err.max()) <= ELEMENT_TOL, (
        f"max error {float(err.max()):.3e} of the row scale exceeds {ELEMENT_TOL:.3e}"
    )
    rms = float(((o - r) ** 2).sum().sqrt() / (r**2).sum().sqrt().clamp_min(1e-30))
    assert rms <= RMS_TOL, f"relative RMS {rms:.3e} exceeds {RMS_TOL:.3e}"


def _schedule(context_lens: torch.Tensor) -> torch.Tensor:
    """The decode schedule, built the way the DSA indexer builds it."""
    return deep_gemm.get_paged_mqa_logits_metadata(
        context_lens.view(-1, 1), SCHEDULE_BLOCK_KV, deep_gemm.get_num_sms()
    )


def _build(spec, seed):
    """`(q, kv_fused, weights, context_lens, block_table, schedule, max_context_len)`.

    Every key is quantized per token to e4m3 with an fp32 scale and written to
    a random page, data first and scales after, as the indexer cache stores
    them. Unused block-table entries point at page 0.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    cpu = torch.Generator().manual_seed(seed)
    heads, dim, tpb = spec["heads"], spec["head_dim"], spec["tokens_per_block"]
    next_n = spec["next_n"]
    lens = spec["context_lens"]
    batch = len(lens)
    pages = [math.ceil(n / tpb) for n in lens]
    num_pages = sum(pages) + 1
    order = torch.randperm(num_pages, generator=cpu).to(torch.int32)
    table = torch.zeros(batch, max(pages), dtype=torch.int32)
    used = 0
    for b, n in enumerate(pages):
        table[b, :n] = order[used : used + n]
        used += n

    flat = torch.zeros(num_pages, tpb * (dim + 4), dtype=torch.uint8, device="cuda")
    keys = torch.randn(num_pages, tpb, dim, generator=g, device="cuda")
    scale = (keys.abs().amax(-1) / E4M3_MAX).clamp_min(1e-12)
    codes = (keys / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    flat[:, : tpb * dim] = codes.view(torch.uint8).reshape(num_pages, -1)
    flat[:, tpb * dim :] = scale.contiguous().view(torch.uint8).reshape(num_pages, -1)
    kv_fused = flat.view(num_pages, tpb, 1, dim + 4)

    q = torch.randn(batch, next_n, heads, dim, generator=g, device="cuda").to(torch.float8_e4m3fn)
    w = torch.randn(batch * next_n, heads, generator=g, device="cuda") * (dim**-0.5 * heads**-0.5)
    context_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    return (
        q,
        kv_fused,
        w.contiguous(),
        context_lens,
        table.cuda(),
        _schedule(context_lens),
        spec["max_context_len"],
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    cache_before = args[1].clone()
    with validating(op):
        out = op(*args)
    assert torch.equal(args[1], cache_before), "scoring modified the cache"
    batch, next_n = args[0].shape[:2]
    assert tuple(out.shape) == (batch * next_n, args[6])
    _compare(out, op.reference(*args))


def test_the_result_is_a_shared_buffer() -> None:
    """Two calls of one shape return the same storage, and the second call
    overwrites the first result. A caller that keeps a result across calls
    reads the later scores without an error."""
    spec = dict(
        heads=32,
        head_dim=128,
        tokens_per_block=32,
        next_n=1,
        context_lens=[500, 300],
        max_context_len=1024,
    )
    first_args = _build(spec, seed=21)
    second_args = _build(spec, seed=22)
    first = op(*first_args)
    _compare(first, op.reference(*first_args))
    second = op(*second_args)
    assert first.data_ptr() == second.data_ptr()
    _compare(first, op.reference(*second_args))


def test_mtp_rows_have_their_own_causal_end() -> None:
    """At MTP 2 the first token of a sequence ends one key before the second.

    The decode top-k reads each row only up to its own causal end, so that is
    the part the entry certifies; the last key is defined for the second row
    only.
    """
    spec = dict(
        heads=32,
        head_dim=128,
        tokens_per_block=32,
        next_n=2,
        context_lens=[200],
        max_context_len=256,
    )
    args = _build(spec, seed=23)
    out = op(*args)
    ref = op.reference(*args)
    assert torch.isnan(ref[0, 199]) and not torch.isnan(ref[1, 199])
    _compare(out, ref)


def _ok():
    spec = dict(
        heads=32,
        head_dim=128,
        tokens_per_block=32,
        next_n=1,
        context_lens=[100, 64],
        max_context_len=256,
    )
    return _build(spec, seed=25)


def test_guard_refuses_a_schedule_for_another_sm_count() -> None:
    q, kv, w, lens, table, schedule, max_len = _ok()
    for rows in (schedule.shape[0] - 1, schedule.shape[0] + 1):
        bad = torch.zeros(rows, 2, dtype=torch.int32, device="cuda")
        with pytest.raises(AssertionError, match="SM count \\+ 1 rows"):
            with validating(op):
                op(q, kv, w, lens, table, bad, max_len)


def test_guard_refuses_a_context_past_max_context_len() -> None:
    q, kv, w, lens, table, schedule, _ = _ok()
    with pytest.raises(AssertionError, match="exceeds max_context_len"):
        with validating(op):
            op(q, kv, w, lens, table, schedule, 64)
