# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the m3_draft_attention entry: the Eagle3 drafter layer's attention against its float64 reference.

Every cell runs on the runtime's own objects (``_m3_attn_runtime_tp4``): the target's hybrid NVFP4
``MiniMaxM3KVCacheManagerV2`` with its one-model Eagle3 draft layer, the requests allocated in it, drafter
``TrtllmAttentionMetadata`` over the manager's FP8 draft view (the manager the runtime's draft swap installs) filled
by its ``prepare()``, and the drafter's ``TrtllmAttention`` built as ``M3Eagle3Draft`` builds it (fused NeoX RoPE at
the drafter checkpoint's theta, FP8 KV). Earlier rows go into the draft layer's cache through the in-tree scatter at
the slots the manager assigned; the call rotates and appends the step's own.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.attention.m3_draft_attention import (  # noqa: E501
    m3_draft_attention as op,
)

__extra_import_path__ = ["../../../..", "."]
import _m3_attn_runtime_tp4 as rt  # noqa: E402 — needs the path declared above
from _validating import ULP, assert_within_ulp, certifying  # noqa: E402, F401

assert torch.cuda.is_available(), "m3_draft_attention requires a CUDA device"

# Band, in BF16 ulps (2^-8) of each token's [heads x 128] row's largest magnitude (element) and of the relative RMS.
# The reference reads the same FP8 cache rows the call attends, so what is left is the FP8 FMHA's own arithmetic:
# it takes the rotated Q in E4M3 (rounding the reference's Q to E4M3 takes the RMS gap from ~9 to ~7 ulps) and rounds
# the softmax probabilities before the PV product. Measured on GB300 over these cells (max element / RMS): 13.6 / 8.6
# (4 x 4 rows), 9.9 / 7.2 (8 x 1), 20.4 / 7.7 (512 context rows). The band leaves about 1.5x; a reference shifted
# by one row is far outside it.
_ELEMENT_ULP = 30.0
_RMS_ULP = 12.0

# Cached tokens per request of each cell; the new rows per request come from the cell.
_CACHED = {
    (4, 4): (200, 129, 1000, 64),
    (8, 1): (5, 127, 128, 300, 1000, 2047, 64, 900),
    (1, None): (0,),
}
#: Rows past the live ones in the fused QKV, as a padded step hands them over.
_PAD = 3


def _ulp_stats(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    o, r = out.float(), ref.float()
    scale = r.abs().amax(dim=1, keepdim=True).clamp_min(1e-9)
    element = ((o - r).abs() / scale).max().item() / ULP[torch.bfloat16]
    rms = ((o - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt()).item() / ULP[torch.bfloat16]
    return element, rms


def _rope(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """NeoX RoPE over the whole head at the drafter's theta, float64."""
    half = rt.HEAD_DIM // 2
    inv_freq = rt.DRAFT_ROPE_THETA ** (
        -torch.arange(0, rt.HEAD_DIM, 2, dtype=torch.float64, device=x.device) / rt.HEAD_DIM
    )
    angles = positions.to(torch.float64)[:, None] * inv_freq[None, :]
    x = x.to(torch.float64)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat(
        [x1 * angles.cos() - x2 * angles.sin(), x2 * angles.cos() + x1 * angles.sin()], -1
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    spec = cell.spec
    context = spec.get("context")
    cached = _CACHED[spec["reqs"], spec.get("qlen")]
    qlens = (context,) if context else (spec["qlen"],) * spec["reqs"]
    assert len(cached) == spec["reqs"]
    contexts = 1 if context else 0
    rids = list(range(spec["reqs"]))
    gen = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    manager = rt.make_manager(draft=True)
    try:
        backend = rt.make_draft_backend()
        metadata = rt.make_draft_metadata(manager)
        rt.allocate(manager, rids, qlens, cached)
        for rid, c in zip(rids, cached):
            if c:
                hist = torch.randn(2, c, rt.HEAD_DIM, generator=gen, device="cuda")
                hist = hist.to(torch.float8_e4m3fn)
                rt.write_fp8_rows(manager, rt.DRAFT, rid, torch.arange(c), hist[0], hist[1])
        rt.prepare(metadata, rids, qlens, cached, contexts)
        total = sum(qlens)
        assert int(metadata.num_tokens) == total
        width = (rt.HEADS + 2 * rt.KV_HEADS) * rt.HEAD_DIM
        qkv = torch.randn(total + _PAD, width, generator=gen, device="cuda").to(torch.bfloat16)

        certifying(op)
        out = op(backend, qkv, metadata)
        ref = op.reference(backend, qkv, metadata)
        assert out.shape == (total, rt.HEADS * rt.HEAD_DIM) and out.dtype == torch.bfloat16
        assert torch.isfinite(out.float()).all()
        element, rms = _ulp_stats(out, ref)
        print(f"\n[{cell.why}] max element {element:.2f} ulp, rms {rms:.3f} ulp")
        assert_within_ulp(out, ref, element_ulp=_ELEMENT_ULP, rms_ulp=_RMS_ULP)

        # The call's append: each live row's rotated K and its V at the row's position, E4M3 (unit KV scale), to
        # within one E4M3 step of the float64 rotation (the kernel rotates in FP32).
        pool = manager.get_buffers(rt.DRAFT, "HND")
        q_size = rt.HEADS * rt.HEAD_DIM
        first = 0
        for rid, c, n in zip(rids, cached, qlens):
            positions = torch.arange(c, c + n, device="cuda")
            slots = rt.flat_slots(manager, rt.DRAFT, rid, positions).long()
            rows = qkv[first : first + n]
            want_k = _rope(rows[:, q_size : q_size + rt.HEAD_DIM].float(), positions)
            want_v = rows[:, q_size + rt.HEAD_DIM :].to(torch.float8_e4m3fn)
            got_k = pool[slots // rt.PAGE, 0, 0, slots % rt.PAGE].to(torch.float64)
            got_v = pool[slots // rt.PAGE, 1, 0, slots % rt.PAGE]
            step = want_k.abs().clamp_min(2.0**-6) * 2.0**-3
            assert bool(((got_k - want_k).abs() <= step).all()), "K is appended rotated, in E4M3"
            assert torch.equal(got_v.view(torch.uint8), want_v.view(torch.uint8))
            first += n

        # The band discriminates: a reference whose rows are shifted by one token is outside it.
        shifted = op.reference(backend, qkv.roll(1, 0), metadata)
        with pytest.raises(AssertionError):
            assert_within_ulp(out, shifted, element_ulp=_ELEMENT_ULP, rms_ulp=_RMS_ULP)
    finally:
        manager.shutdown()
