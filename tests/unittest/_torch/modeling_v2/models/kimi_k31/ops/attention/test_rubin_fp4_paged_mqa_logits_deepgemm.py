# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of the rubin_fp4_paged_mqa_logits_deepgemm entry (FP4 decode / verify logits, DeepGEMM).

The cache and data are the CuTe DSL entry's; the context lengths are the 2-D [B, n] form and the
schedule is built from them.
"""

import zlib

import pytest
import torch

# l0_gb300 collects this directory: any GPU but sm_107 skips the module before a family import.
if torch.cuda.is_available() and torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("certified on sm_107 only", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_fp4_paged_mqa_logits_deepgemm import (  # noqa: E402,E501
    rubin_fp4_paged_mqa_logits_deepgemm as op,
)

__extra_import_path__ = ["../../../..", "."]
import _rubin_fp4_inputs as fp4  # noqa: E402

if torch.cuda.is_available():  # the GPU cells' guard interposer: a CPU run needs neither
    from _validating import certifying, validating  # noqa: E402,F401 -- path above; fixture by name

cpu = pytest.mark.cpu_only
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="drives the kernel on sm_107")


def _schedule(lens_2d: torch.Tensor) -> torch.Tensor:
    """DeepGEMM's schedule over 2-D context lengths, as the DSA metadata builds it."""
    from tensorrt_llm.deep_gemm import get_paged_mqa_logits_metadata

    num_sms = torch.cuda.get_device_properties(lens_2d.device).multi_processor_count
    return get_paged_mqa_logits_metadata(lens_2d, fp4.SCHEDULE_BLOCK_KV, num_sms)


def _build(spec, seed: int, device="cuda") -> dict:
    gen = torch.Generator().manual_seed(seed)
    batch, next_n = spec["batch"], spec["next_n"]
    lo, hi = spec["context_lens"]
    lens = torch.randint(lo, hi + 1, (batch,), generator=gen, dtype=torch.int32)
    if spec.get("zero_every"):
        lens[:: spec["zero_every"]] = 0
    kv_cache, table, _ = fp4.paged_cache(
        lens.tolist(), spec["tokens_per_block"], spec["data"], gen, spec["page_stride_pad"], device
    )
    rows = batch * next_n
    q, sf_q = fp4.fp4_rows(rows * fp4.HEADS, spec["data"], gen, fp4.QUERY_EXPONENTS)
    weights = fp4.weights(rows, spec["data"], gen)
    lens_2d = lens[:, None].expand(batch, next_n).contiguous().to(device)
    schedule = _schedule(lens_2d) if device == "cuda" else torch.zeros((2, 2), dtype=torch.int32)
    return dict(
        q=q.view(batch, next_n, fp4.HEADS, fp4.DATA).to(device),
        sf_q=sf_q.view(batch, next_n, fp4.HEADS).to(device),
        kv_cache=kv_cache,
        weights=weights.to(device),
        context_lens=lens_2d,
        block_table=table,
        schedule_meta=schedule,
        max_context_len=spec["max_context_len"],
    )


def _small(batch, next_n, lens, page=16) -> dict:
    spec = dict(
        batch=batch,
        next_n=next_n,
        tokens_per_block=page,
        context_lens=lens,
        page_stride_pad=0,
        max_context_len=160,
        data="int",
    )
    return _build(spec, seed=batch * 10 + next_n, device="cpu")


@cpu
@pytest.mark.parametrize(
    "batch,next_n,lens,keyless",
    [
        (2, 1, (40, 150), False),  # plain decode
        (3, 4, (1, 130), False),  # a verify window, a context shorter than the window
        (2, 2, (40, 150), True),  # a helix8 rank holds no key of request 0: its rows stay -inf
    ],
)
def test_reference_is_the_causal_weighted_relu_sum(batch, next_n, lens, keyless):
    args = _small(batch, next_n, lens)
    if keyless:
        args["context_lens"][0] = 0
    ref = op.reference(**args)
    assert ref.dtype == torch.float32 and ref.shape == (batch * next_n, 160)
    assert torch.equal(ref, fp4.paged_logits(args, args["context_lens"]))
    if keyless:
        assert bool(torch.isneginf(ref[:next_n]).all())


@cpu
def test_guard_admits_a_well_formed_call():
    args = _small(3, 4, (60, 130))
    op.is_valid(**args)


@cpu
@pytest.mark.parametrize(
    "change,match",
    [
        (lambda a: dict(a, q=torch.zeros(*a["q"].shape[:3], 128, dtype=torch.uint8)), "q must"),
        (lambda a: dict(a, sf_q=a["sf_q"].view(torch.float32)), "sf_q"),
        (lambda a: dict(a, kv_cache=torch.zeros(9, 16, 1, 132, dtype=torch.uint8)), "68"),
        (lambda a: dict(a, context_lens=a["context_lens"][:, 0].contiguous()), "context_lens"),
        (lambda a: dict(a, max_context_len=50), "max_context_len"),
        (lambda a: dict(a, block_table=a["block_table"][:, :-1]), "block_table"),
        (lambda a: dict(a, weights=a["weights"][:, :16].contiguous()), "one row per query row"),
    ],
    ids=[
        "fp8-wide q",
        "fp32 sf_q",
        "fp8 cache",
        "1-D lengths",
        "context past max",
        "short table",
        "weights",
    ],
)
def test_guard_refuses(change, match):
    args = _small(3, 4, (60, 130))
    with pytest.raises(AssertionError, match=match):
        op.is_valid(**change(args))


@gpu
@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args = _build(cell.spec, zlib.crc32(cell.why.encode()) % 2**31)
    out = op(**args)
    again = op(**args)
    assert out.data_ptr() != again.data_ptr(), "each call returns a fresh allocation"
    ref = op.reference(**args)
    torch.cuda.synchronize()
    causal = torch.isfinite(ref)
    assert out.dtype == torch.float32 and out.shape == ref.shape
    assert torch.equal(out[causal], again[causal]), "two identical calls disagree"
    if cell.spec["data"] == "int":
        assert torch.equal(out[causal], ref[causal])
    else:
        fp4.assert_paged_band(out, ref, args, args["context_lens"])


@gpu
def test_the_cute_dsl_entry_gives_the_same_bits_on_exact_data():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_fp4_paged_mqa_logits import (  # noqa: E501
        rubin_fp4_paged_mqa_logits as dsl,
    )

    spec = next(c.spec for c in op.CELLS if c.spec["next_n"] == 4)
    args = _build(spec, seed=23)
    batch, next_n = args["q"].shape[:2]
    with validating(op):
        out = op(**args)
    aligned = -(-spec["max_context_len"] // 256) * 256
    buf = torch.empty(batch * next_n, aligned, dtype=torch.float32, device="cuda")
    with validating(dsl):
        other = dsl(
            args["q"],
            args["sf_q"],
            args["kv_cache"],
            args["weights"],
            args["context_lens"][:, -1].contiguous(),
            args["block_table"],
            _schedule(args["context_lens"][:, -1:].contiguous()),
            spec["max_context_len"],
            1,
            None,
            True,
            buf[:, : spec["max_context_len"]],
        )
    ref = op.reference(**args)
    torch.cuda.synchronize()
    causal = torch.isfinite(ref)
    assert torch.equal(out[causal], other[causal])
