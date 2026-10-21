# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the m3_msa_attention entry: one layer's MSA backend step against its float64 reference.

Every cell runs on the runtime's own objects (``_m3_attn_runtime_tp4``): a hybrid NVFP4 ``MiniMaxM3KVCacheManagerV2``
with the requests allocated in it, ``MiniMaxM3MsaSparseAttentionMetadata`` filled by its ``prepare()``, and the layer's
``MiniMaxM3MsaSparseAttention`` built as the target builds it. Earlier rows (and, on a sparse layer, the step's own
rows, which the target's producer writes before the call) go into the cache through the in-tree scatters, at the
slots the manager assigned. The sparse layers' index-K rows are planted so every block's score is 1/8 or more apart
from every other's: the top-k selection the reference recomputes is then unambiguous.

The cells are the target's calls (``_attention_core`` in modeling.py): a dense layer with live bf16 K / V in a
prefill chunk and live FP8 K / V in a decode step (the dense layers sit in the hybrid cache's FP8 half), a sparse
layer whose producer already wrote the caches (``k = v = None``) in a prefill chunk beyond 16 blocks and in a mixed
step, both with the NVFP4 scales the target passes.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.attention.m3_msa_attention import (  # noqa: E501
    m3_msa_attention as op,
)

__extra_import_path__ = ["../../../..", "."]
import _m3_attn_runtime_tp4 as rt  # noqa: E402 — needs the path declared above
from _validating import ULP, assert_within_ulp, certifying  # noqa: E402, F401

assert torch.cuda.is_available(), "m3_msa_attention requires a CUDA device"

# Bands, in BF16 ulps (2^-8) of each token's [heads x 128] row's largest magnitude (element) and of the relative RMS,
# per kernel the step reaches. The reference reads the same quantized cache rows (and casts Q to E4M3 where the FP8
# FMHA does), so what is left is each FMHA's own arithmetic, measured on GB300 over these cells (max element / RMS):
# * fmha_sm100 over the FP8 pool (dense context rows): 10.1 / 6.1. It runs E4M3 x E4M3 and its softmax
#   probabilities are rounded before the PV product.
# * trtllm-gen over the FP8 pool (dense generation rows): 7.0 / 6.1. It rounds the probabilities to E4M3 (scaled by
#   448 against the running maximum): emulating that brings a single-tile request to 0.6 RMS.
# * the NVFP4 CSR prefill (sparse context rows): 34.4 / 11.8 at 4096 rows, 26.8 / 9.1 at 512. It stages the
#   reconstructed K / V in E4M3 (the in-tree A/B test, test_minimax_m3_sparse_attn_decode, notes it and holds it
#   to 3e-2).
# * the Triton NVFP4 sparse decode (sparse generation rows): 1.0 / 0.25 -- the band of its own entry
#   (m3_sparse_decode): exact operands, FP16 probabilities, BF16 rounding.
# Each band leaves about 1.5x the measured maximum; a shifted or wrongly selected reference is far outside them.
_FP8_CONTEXT = dict(element_ulp=16.0, rms_ulp=9.0)
_FP8_DECODE = dict(element_ulp=12.0, rms_ulp=9.0)
_NVFP4_CONTEXT = dict(element_ulp=48.0, rms_ulp=16.0)
_NVFP4_DECODE = dict(element_ulp=2.0, rms_ulp=0.5)
_BANDS = {
    ("dense", "context"): (_FP8_CONTEXT, None),
    ("dense", "generation"): (None, _FP8_DECODE),
    ("sparse", "context"): (_NVFP4_CONTEXT, None),
    ("sparse", "mixed"): (_NVFP4_CONTEXT, _NVFP4_DECODE),
}


def _assert_in_bands(output, ref, bands, context_rows: int) -> None:
    """Gate the context rows and the generation rows each on their own kernel's band."""
    context_band, decode_band = bands
    if context_rows:
        assert_within_ulp(output[:context_rows], ref[:context_rows], **context_band)
    if context_rows < output.shape[0]:
        assert_within_ulp(output[context_rows:], ref[context_rows:], **decode_band)


# (request rows, cached tokens per request, new tokens per request, context rows) per cell.
_SHAPES = {
    ("dense", "context"): ((384,), (1024,), 1),
    ("dense", "generation"): ((127, 128, 1000, 2500), (1, 1, 1, 1), 0),
    ("sparse", "context"): ((0,), (4096,), 1),
    ("sparse", "mixed"): ((0, 2999, 4500), (512, 1, 1), 1),
}


def _ulp_stats(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    """The two quantities ``assert_within_ulp`` gates, in BF16 ulps."""
    o, r = out.float(), ref.float()
    scale = r.abs().amax(dim=1, keepdim=True).clamp_min(1e-9)
    element = ((o - r).abs() / scale).max().item() / ULP[torch.bfloat16]
    rms = ((o - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt()).item() / ULP[torch.bfloat16]
    return element, rms


def _shift(x: torch.Tensor) -> torch.Tensor:
    """``x``'s rows moved down by one (byte-wise, so E4M3 rows move too)."""
    return x.view(torch.uint8).roll(1, 0).view(x.dtype) if x.element_size() == 1 else x.roll(1, 0)


def _shape(cell):
    spec = cell.spec
    cached, qlens, contexts = _SHAPES[spec["layer"], spec["phase"]]
    if spec["phase"] == "context":
        assert sum(qlens) == spec["tokens"]
    elif spec["phase"] == "generation":
        assert len(qlens) == spec["reqs"]
    else:
        assert sum(qlens[:contexts]) == spec["tokens"] and len(qlens) - contexts == spec["reqs"]
    return cached, qlens, contexts


def _run_dense(cell, gen, manager, backend, metadata):
    cached, qlens, contexts = _shape(cell)
    rids = list(range(len(qlens)))
    rt.allocate(manager, rids, qlens, cached)
    for rid, c in zip(rids, cached):
        if c:
            hist = torch.randn(2, c, rt.HEAD_DIM, generator=gen, device="cuda").to(
                torch.float8_e4m3fn
            )
            rt.write_fp8_rows(manager, rt.DENSE, rid, torch.arange(c), hist[0], hist[1])
    rt.prepare(metadata, rids, qlens, cached, contexts)
    total = sum(qlens)
    # A prefill chunk arrives in BF16 (the hybrid cache's dense layers take FP8 Q / K / V only on decode steps).
    dtype = torch.bfloat16 if contexts else torch.float8_e4m3fn
    q = torch.randn(total, rt.HEADS * rt.HEAD_DIM, generator=gen, device="cuda").to(dtype)
    k = torch.randn(total, rt.HEAD_DIM, generator=gen, device="cuda").to(dtype)
    v = torch.randn(total, rt.HEAD_DIM, generator=gen, device="cuda").to(dtype)
    # The NVFP4 target passes its layer scales on every layer; an FP8 layer's write and FMHA take none.
    inv = torch.tensor([1.0, 0.5, 2.0], device="cuda")
    fwd = 1.0 / inv
    return (q, k, v, None, inv, fwd), dict(rids=rids, cached=cached, qlens=qlens, contexts=contexts)


def _run_sparse(cell, gen, manager, backend, metadata):
    cached, qlens, contexts = _shape(cell)
    rids = list(range(len(qlens)))
    rt.allocate(manager, rids, qlens, cached)
    u0, u1 = rt.index_directions(gen)
    lengths = [c + n for c, n in zip(cached, qlens)]
    keys = [torch.randn(n, rt.HEAD_DIM, generator=gen, device="cuda") for n in lengths]
    values = [torch.randn(n, rt.HEAD_DIM, generator=gen, device="cuda") for n in lengths]
    quant = rt.nvfp4_quant(torch.cat(keys), torch.cat(values))
    for rid, n, kk, vv in zip(rids, lengths, keys, values):
        # Every row, the step's own included: the producer writes a sparse layer's caches before the call.
        idx_k = rt.planted_index_keys(n, u0, u1, gen)
        rt.write_nvfp4_rows(
            manager,
            metadata,
            rt.SPARSE,
            rid,
            torch.arange(n),
            kk.to(torch.bfloat16),
            vv.to(torch.bfloat16),
            idx_k,
            quant,
        )
    rt.prepare(metadata, rids, qlens, cached, contexts)
    total = sum(qlens)
    q = torch.randn(total, rt.HEADS * rt.HEAD_DIM, generator=gen, device="cuda").to(
        torch.float8_e4m3fn
    )
    # Query tokens alternate between the two planted directions, so neighbouring tokens select different blocks.
    idx_q = torch.stack([u0, u1])[torch.arange(total, device="cuda") % 2].to(torch.float8_e4m3fn)
    return (q, None, None, idx_q, quant, 1.0 / quant), dict(
        rids=rids, cached=cached, qlens=qlens, contexts=contexts
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    spec = cell.spec
    sparse = spec["layer"] == "sparse"
    gen = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    manager = rt.make_manager(draft=False)
    try:
        backend = rt.make_msa_backend(rt.SPARSE if sparse else rt.DENSE)
        metadata = rt.make_msa_metadata(manager)
        run = _run_sparse if sparse else _run_dense
        (q, k, v, idx_q, inv, fwd), step = run(cell, gen, manager, backend, metadata)
        output = torch.full(
            (q.shape[0], rt.HEADS * rt.HEAD_DIM), float("nan"), dtype=torch.bfloat16, device="cuda"
        )
        certifying(op)
        result = op(backend, q, k, v, idx_q, metadata, output, sparse, inv, fwd)
        assert result is output
        ref = op.reference(backend, q, k, v, idx_q, metadata, output, sparse, inv, fwd)
        assert torch.isfinite(output.float()).all(), "every live row is written"
        bands = _BANDS[spec["layer"], spec["phase"]]
        context_rows = sum(step["qlens"][: step["contexts"]])
        for name, rows in (
            ("context", slice(0, context_rows)),
            ("generation", slice(context_rows, None)),
        ):
            if output[rows].numel():
                element, rms = _ulp_stats(output[rows], ref[rows])
                print(
                    f"\n[{cell.why}] {name} rows: max element {element:.2f} ulp, rms {rms:.3f} ulp"
                )
        _assert_in_bands(output, ref, bands, context_rows)

        if k is not None:
            # The call's own cache write: the live K / V rows land, as a plain E4M3 cast, at their positions.
            pool = manager.get_buffers(rt.DENSE, "HND")
            first = 0
            for rid, c, n in zip(step["rids"], step["cached"], step["qlens"]):
                slots = rt.flat_slots(manager, rt.DENSE, rid, torch.arange(c, c + n)).long()
                for role, src in ((0, k), (1, v)):
                    got = pool[slots // rt.PAGE, role, 0, slots % rt.PAGE]
                    want = src[first : first + n].to(torch.float8_e4m3fn)
                    assert torch.equal(got.view(torch.uint8), want.view(torch.uint8))
                first += n

        # The band discriminates: a reference whose query rows are shifted by one token is outside it.
        shifted = op.reference(backend, _shift(q), k, v, idx_q, metadata, output, sparse, inv, fwd)
        with pytest.raises(AssertionError):
            _assert_in_bands(output, shifted, bands, context_rows)
        if sparse:
            # And the selection is load-bearing: attending every causal block instead is outside it too.
            dense = op.reference(backend, q, k, v, idx_q, metadata, output, False, inv, fwd)
            with pytest.raises(AssertionError):
                _assert_in_bands(output, dense, bands, context_rows)
    finally:
        manager.shutdown()
