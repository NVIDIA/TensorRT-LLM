# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the glm5_next fp8 tp4 entry sparse_mla_decode."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import true_fp32_matmul
from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4.ops.attention.sparse_mla_decode import (  # noqa: E501
    WORKSPACE_BYTES,
    counter_bytes,
    sparse_mla_decode,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 100, reason="the glm5_next fp8 tp4 entries are certified on sm_100"
)

DEVICE = "cuda"
TPB, KPOOL, SELECT_K, WIDTH, DIM = 32, 4, 512, 2112, 512
# A coalesced latent pool: the layer's 32 rows of a slot are the second of three subpages.
SLOTS, SUBPAGES, BASE_ROW = 512, 3, 32
# The cache's dequantization scale and the model's softmax scale, qk_nope_head_dim ** -0.5.
KV_SCALE, SOFTMAX_SCALE = 0.75, 256**-0.5


def _lengths(spec: dict) -> torch.Tensor:
    """Each row's selection length: the members of its ``min(512, visible // 4)`` pools plus
    its tail. Generation cells: request ``b``'s KV length after the step is ``kv - b % 4`` and
    its rows see ``kv_b - T + 1 .. kv_b``; context cells: row ``r`` sees ``r + 1``."""
    if "context" in spec:
        lo, hi = spec["rows"]
        visible = torch.arange(lo + 1, hi + 1, device=DEVICE)
    else:
        kv = spec["kv"] - torch.arange(spec["requests"], device=DEVICE) % 4
        steps = torch.arange(spec["tokens"], device=DEVICE)
        visible = (kv[:, None] - spec["tokens"] + 1 + steps).reshape(-1)
    return ((visible // KPOOL).clamp(max=SELECT_K) * KPOOL + visible % KPOOL).to(torch.int32)


def _args(spec: dict, seed: int):
    """The op's arguments: the rows of the coalesced FP8 pool (every row outside the layer's
    subpage is NaN, except row 0, the dummy id), each row's distinct selected layer rows
    packed in front of -1 padding, E4M3 queries, zeroed buffers, and the module path's scales
    (a host float for bmm1, the device dequantization scale for bmm2). ``empty_rows`` trailing
    rows selected nothing: one dummy id 0, length 1."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    lengths = _lengths(spec)
    rows, heads = lengths.shape[0], spec["heads"]
    empty = spec.get("empty_rows", 0)
    storage = torch.full((SLOTS, SUBPAGES, TPB, DIM), float("nan"), device=DEVICE)
    storage[:, 1] = torch.randn(SLOTS, TPB, DIM, generator=g, device=DEVICE)
    storage[0, 0] = torch.randn(TPB, DIM, generator=g, device=DEVICE)
    kv_rows = storage.to(torch.float8_e4m3fn).view(-1, 1, DIM)
    layer = BASE_ROW + torch.arange(SLOTS, device=DEVICE)[:, None] * SUBPAGES * TPB
    layer = (layer + torch.arange(TPB, device=DEVICE)).reshape(-1)
    pick = torch.rand(rows, layer.numel(), generator=g, device=DEVICE)
    pick = pick.topk(int(lengths.max()), dim=1).indices
    packed = torch.full((rows, WIDTH), -1, dtype=torch.int32, device=DEVICE)
    packed[:, : pick.shape[1]] = layer[pick].to(torch.int32)
    col = torch.arange(WIDTH, device=DEVICE)
    packed = torch.where(col < lengths[:, None], packed, -1)
    if empty:
        packed[-empty:] = -1
        packed[-empty:, 0] = 0
        lengths[-empty:] = 1
    query = torch.randn(rows, heads, DIM, generator=g, device=DEVICE).to(torch.float8_e4m3fn)
    sm_count = torch.cuda.get_device_properties(DEVICE).multi_processor_count
    workspace = torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device=DEVICE)
    counter = torch.zeros(counter_bytes(rows, heads, sm_count), dtype=torch.uint8, device=DEVICE)
    bmm2 = torch.tensor([KV_SCALE], dtype=torch.float32, device=DEVICE)
    return query, kv_rows, packed, lengths, workspace, counter, KV_SCALE * SOFTMAX_SCALE, bmm2


def _sigma(query, kv_rows, packed, lengths, ref, bmm1, bmm2):
    """Standard deviation of each output's error from the E4M3 rounding of the probabilities.

    The kernel rounds ``p_j = exp(s_j - m) <= 1`` (``m`` a running maximum, never above the
    row's) to E4M3 for the value product: at most half a step, ``2**-4 p_j`` above ``2**-6``
    and ``2**-10`` below, modelled as uniform (variance ``h**2 / 3``). In units of the
    normalized ``pi_j = p_j / l`` that is ``h_j = max(2**-4 pi_j, 2**-10 / l)``. The output
    ``o_d = sum_j pi_j v_jd`` then errs with variance ``sum_j h_j**2 v_jd**2 / 3``, plus
    ``o_d**2 sum_j h_j**2 / 3`` should the normalizer use the rounded probabilities.
    """
    rows, heads, dim = query.shape
    latent = kv_rows.reshape(-1, dim).view(torch.uint8)
    col = torch.arange(packed.shape[1], device=DEVICE)
    var = torch.empty(rows, heads, dim, device=DEVICE)
    for r0 in range(0, rows, 16):
        part = slice(r0, min(r0 + 16, rows))
        selected = col[None, :] < lengths[part, None].long()
        values = latent[packed[part].long().clamp(min=0)].view(kv_rows.dtype).float()
        values = values.masked_fill(~selected[:, :, None], 0.0)
        with true_fp32_matmul():
            logits = torch.einsum("rhd,rwd->rhw", query[part].float(), values) * bmm1
        logits = logits.masked_fill(~selected[:, None, :], float("-inf"))
        p = (logits - logits.amax(dim=-1, keepdim=True)).exp()
        norm = p.sum(dim=-1, keepdim=True)
        h = torch.maximum(2**-4 * p / norm, 2**-10 / norm).masked_fill(~selected[:, None, :], 0)
        with true_fp32_matmul():
            spread = torch.einsum("rhw,rwd->rhd", h**2, (values * bmm2) ** 2)
        out = ref[part].float()
        var[part] = (spread + out**2 * (h**2).sum(dim=-1, keepdim=True)) / 3
    return var.sqrt()


@pytest.mark.parametrize(
    "cell", sparse_mla_decode.CELLS, ids=[c.why[:44] for c in sparse_mla_decode.CELLS]
)
def test_sparse_mla_decode_matches_reference(cell, certifying):  # noqa: F811
    """Every output within ``8 sigma + 2**-7 |ref|`` of the FP32 reference: ``sigma`` from the
    kernel's E4M3 probabilities (``_sigma``; the scores, the softmax and the accumulations are
    FP32 on exact E4M3 operands), ``2**-7 |ref|`` for the two sides rounding to BF16 apart.
    Across the cells' ~10**8 outputs a Gaussian tail past 8 sigma is below 1e-7. The relative
    RMS error of the whole cell stays under ``2**-4`` (the rounding alone predicts about
    ``2**-4 / sqrt(3)``). The kernel hands its multi-CTA counter back zeroed, and no NaN
    outside the layer's rows is read."""
    certifying(sparse_mla_decode)
    args = _args(cell.spec, zlib.crc32(cell.why.encode()) % 2**31)
    out = sparse_mla_decode(*args)
    ref = sparse_mla_decode.reference(*args)
    query, kv_rows, packed, lengths, workspace, counter, bmm1, bmm2 = args
    assert out.shape == ref.shape == query.shape and out.dtype == torch.bfloat16
    assert not bool(counter.any()), "the kernel leaves its counter at zero for the next launch"
    assert bool(torch.isfinite(out).all())
    sigma = _sigma(query, kv_rows, packed, lengths, ref, bmm1, bmm2)
    err = (out.float() - ref.float()).abs()
    band = 8 * sigma + 2**-7 * ref.float().abs()
    worst = float((err / band.clamp_min(1e-30)).max())
    assert bool((err <= band).all()), f"worst error {worst:.3f} of the band"
    rel_rms = float(err.square().mean().sqrt() / ref.float().square().mean().sqrt())
    assert rel_rms < 2**-4, f"relative RMS error {rel_rms:.3e}"


def test_guard_refuses_a_short_workspace():
    """The kernel's split-KV scratch is the rest of the workspace: a short one is overrun."""
    args = list(_args(dict(heads=16, requests=1, tokens=4, kv=13312), seed=0))
    args[4] = torch.zeros(1 << 20, dtype=torch.uint8, device=DEVICE)
    with pytest.raises(AssertionError, match="workspace"):
        with validating(sparse_mla_decode):
            sparse_mla_decode(*args)
