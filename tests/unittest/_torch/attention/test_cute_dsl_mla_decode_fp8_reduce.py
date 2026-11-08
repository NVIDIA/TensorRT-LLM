# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The FP8 MLA decode's unrolled split-KV reduction sums the splits in the same order."""

import math

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE


def _sm100f() -> bool:
    if not torch.cuda.is_available() or not IS_CUTLASS_DSL_AVAILABLE:
        return False
    from tensorrt_llm._utils import is_sm_100f

    return is_sm_100f()


pytestmark = pytest.mark.skipif(not _sm100f(), reason="CuTe DSL FP8 MLA decode needs SM100f")

HEADS, SEQ_Q, PAGE, KV_LORA, ROPE = 64, 8, 128, 512, 64
LATENT = KV_LORA + ROPE
# Context lengths of three requests: 87, 6 and 24 KV tiles of 128.
CONTEXT = (11000, 700, 3000)


def _runner_cls():
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner

    return CuteDSLNVMlaDecodeBlackwellRunner


def _operands(seed):
    g = torch.Generator().manual_seed(seed)
    batch = len(CONTEXT)
    lengths = torch.tensor(CONTEXT) + SEQ_Q
    blocks = math.ceil(int(lengths.max()) / PAGE)
    num_pages = batch * blocks
    page_rows = torch.randperm(num_pages, generator=g).view(batch, blocks)
    q = torch.randn(batch, SEQ_Q, HEADS, LATENT, generator=g) * 0.1
    cache = torch.randn(num_pages, PAGE, LATENT, generator=g) * 0.1
    q, cache = q.cuda().to(torch.float8_e4m3fn), cache.cuda().to(torch.float8_e4m3fn)
    cache_seqs = lengths.cuda().to(torch.int32)
    return dict(
        q_latent=q[..., :KV_LORA].permute(2, 3, 1, 0),
        q_rope=q[..., KV_LORA:].permute(2, 3, 1, 0),
        c_latent=cache[..., :KV_LORA].permute(1, 2, 0),
        c_rope=cache[..., KV_LORA:].permute(1, 2, 0),
        page_table=page_rows.cuda().to(torch.int32).t(),
        cache_seqs=cache_seqs,
        kv_bounds=cache_seqs.repeat_interleave(SEQ_Q),
        scales=(
            torch.tensor([1.0 / math.sqrt(192)], device="cuda"),
            torch.tensor([1.0], device="cuda"),
        ),
    )


def _decode(ops, plan):
    import cutlass

    batch = ops["cache_seqs"].shape[0]
    out = torch.full((batch, SEQ_Q, HEADS, KV_LORA), float("nan"), device="cuda").bfloat16()
    stats = torch.full((batch * SEQ_Q, HEADS, 2), float("nan"), device="cuda")
    workspace = torch.zeros(
        _runner_cls().get_max_padded_workspace_size(HEADS, SEQ_Q, KV_LORA, batch, cutlass.Float32),
        dtype=torch.uint8,
        device="cuda",
    )
    torch.ops.trtllm.cute_dsl_mla_decode_fp8_blackwell(
        ops["q_latent"],
        ops["q_rope"],
        ops["c_latent"],
        ops["c_rope"],
        ops["page_table"],
        ops["cache_seqs"],
        out.permute(2, 3, 1, 0),
        workspace,
        HEADS,
        SEQ_Q,
        PAGE,
        *ops["scales"],
        batch,
        stats,
        ops["kv_bounds"],
        plan,
    )
    return out, stats


@pytest.mark.parametrize("split_kv,plan", [(7, None), (32, [22, 3, 9]), (32, [32, 1, 13])])
def test_unrolled_split_kv_reduction_is_bitwise(monkeypatch, split_kv, plan):
    from tensorrt_llm._torch.attention.kernels.blackwell.mla import mla_decode_fp8

    runner = _runner_cls()
    torch.zeros(1, device="cuda")
    runner._max_split_kv_rows.cache_clear()
    monkeypatch.setattr(
        runner,
        "default_tactic",
        lambda self, batch_size: ((128, 128), (128, 256), split_kv, False),
    )
    monkeypatch.setattr(runner, "kernel_cache", {})
    ops = _operands(split_kv)
    plan_t = None if plan is None else torch.tensor(plan, dtype=torch.int32, device="cuda")
    results = {}
    for unroll in (1, mla_decode_fp8.REDUCE_UNROLL):
        monkeypatch.setattr(mla_decode_fp8, "REDUCE_UNROLL", unroll)
        runner.kernel_cache.clear()
        results[unroll] = _decode(ops, plan_t)
    torch.cuda.synchronize()
    ref_out, ref_stats = results[1]
    assert not torch.isnan(ref_out.float()).any()
    out, stats = results[max(results)]
    assert torch.equal(out.view(torch.int16), ref_out.view(torch.int16))
    assert torch.equal(stats.view(torch.int32), ref_stats.view(torch.int32))
    runner._max_split_kv_rows.cache_clear()
