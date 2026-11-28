# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``k3_mla_decode_fp8`` entry (the CuTe DSL FP8 MLA decode on a paged FP8 latent cache, with
its per-request split plan) on its ``.CELLS``: the output against the entry's FP32 reference within
the error of the FP8 probabilities the PV MMA takes, and the Helix softmax stats against the
reference's row max and row sum."""

import zlib

import pytest
import torch

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, certifying  # noqa: E402, F401 — fixture by name

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 3), (10, 7)),
    reason="certified on sm_103 and sm_107",
)

LATENT, ROPE, PAGE = 512, 64, 128
#: The engine's request capacity the ULL deployment passes (``max_num_requests``); it fixes the
#: workspace layout.
MAX_BATCH_SIZE = 8


def _mod():
    import importlib

    return importlib.import_module(
        "tensorrt_llm._torch.modeling_v2.models.kimi_k3_vl.ops.attention.k3_mla_decode_fp8"
    )


def _entry():
    return _mod().k3_mla_decode_fp8


def _inputs(spec, gen):
    """The call's arguments, laid out as ``mla.fp8_mla_decode`` hands them over: views of one FP8
    query buffer ``[B, S_q, H, 576]``, of one page pool ``[pages, 128, 576]`` and of the BF16
    output ``[B, S_q, H, 512]``; the page table a transposed ``[B, max_pages]`` view."""
    mod = _mod()
    heads, batch, s_q = spec["heads"], spec["requests"], spec["seq_len_q"]
    kv_lens = spec["kv_lens"]
    assert len(kv_lens) == batch and min(kv_lens) >= s_q
    dev = "cuda"
    pages_per_req = [-(-n // PAGE) for n in kv_lens]
    num_pages = sum(pages_per_req) + 3
    pool = (0.5 * torch.randn(num_pages, PAGE, LATENT + ROPE, device=dev, generator=gen)).to(
        torch.float8_e4m3fn
    )
    q = (0.5 * torch.randn(batch, s_q, heads, LATENT + ROPE, device=dev, generator=gen)).to(
        torch.float8_e4m3fn
    )
    order = torch.randperm(num_pages, device=dev, generator=gen).to(torch.int32)
    table = torch.zeros(batch, max(pages_per_req) + 1, dtype=torch.int32, device=dev)
    used = 0
    for b, n in enumerate(pages_per_req):
        table[b, :n] = order[used : used + n]
        used += n
    cache_seqs = torch.tensor(kv_lens, dtype=torch.int32, device=dev)
    kv_bounds = stats = None
    if spec["helix"]:
        # Per-token rank-local bounds of a verify group, inside the kernel's admitted range
        # [max(0, K - S_q), K] (``mla_decode_fp8.py``'s own check): causal, every odd token one shorter.
        bounds = [n - (s_q - 1 - t) - (t % 2) for n in kv_lens for t in range(s_q)]
        kv_bounds = torch.tensor(bounds, dtype=torch.int32, device=dev)
        stats = torch.full((batch * s_q, heads, 2), float("nan"), device=dev)
    plan = None
    if spec["plan"]:
        rows, waves = mod.split_plan_geometry(heads, s_q)
        plan = torch.empty(batch, dtype=torch.int32, device=dev)
        mod.launch_split_plan(
            cache_seqs, plan, num_requests=batch, rows_per_request=rows, wave_clusters=waves
        )
    out = torch.full((batch, s_q, heads, LATENT), float("nan"), dtype=torch.bfloat16, device=dev)
    workspace = torch.empty(
        mod.workspace_bytes(heads, s_q, LATENT, max(batch, MAX_BATCH_SIZE)),
        dtype=torch.uint8,
        device=dev,
    )
    args = (
        q[..., :LATENT].permute(2, 3, 1, 0),
        q[..., LATENT:].permute(2, 3, 1, 0),
        pool[..., :LATENT].permute(1, 2, 0),
        pool[..., LATENT:].permute(1, 2, 0),
        table.transpose(0, 1),
        cache_seqs,
        out.permute(2, 3, 1, 0),
        workspace,
        heads,
        s_q,
        PAGE,
        torch.tensor([(LATENT + ROPE) ** -0.5], device=dev),
        torch.tensor([1.0], device=dev),
        MAX_BATCH_SIZE,
        stats,
        kv_bounds,
        plan,
    )
    return args, out, stats


@pytest.mark.parametrize("cell", _entry().CELLS, ids=[c.why[:40] for c in _entry().CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811 — fixture by name
    op = _entry()
    certifying(op)
    gen = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    args, out, stats = _inputs(cell.spec, gen)
    op(*args)
    torch.cuda.synchronize()
    ref_o, ref_stats = op.reference(*args)
    assert torch.isfinite(out).all(), cell.why
    # The PV MMA takes the probabilities in FP8 E4M3 (relative rounding up to 2^-4, about 2^-5.6
    # RMS), so each output element is off by about 2^-5.6 of its typical magnitude with random
    # sign: below 2^-3 of the row's largest element and 2^-4 in relative RMS, with margin. A wrong
    # page, bound or split merge is off by the output's own size.
    assert_within_ulp(
        out.view(-1, LATENT),
        ref_o.permute(3, 2, 0, 1).reshape(-1, LATENT),
        element_ulp=2,
        rms_ulp=1,
        ulp=2.0**-4,
    )
    if stats is not None:
        assert torch.isfinite(stats).all(), cell.why
        # The Helix combine reads a (max, sum) pair only through max + ln(sum): the split-KV
        # reduction writes (natural-log LSE, 1), a single split (row max, row sum). Compare that
        # log-sum-exp: the same FP32 logits of exact FP8 products, at worst the sum of the
        # FP8-rounded probabilities (2^-4 each, ln(1 + 2^-4) < 2^-4).
        assert (stats[..., 1] > 0).all(), cell.why
        lse = stats[..., 0] + torch.log(stats[..., 1])
        ref_lse = ref_stats[..., 0] + torch.log(ref_stats[..., 1])
        torch.testing.assert_close(lse, ref_lse, rtol=0.0, atol=2.0**-4)
