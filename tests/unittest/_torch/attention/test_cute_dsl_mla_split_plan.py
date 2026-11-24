# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-request split-KV plan of the CuTe DSL MLA decode: policy, kernel vs mirror, kernel output."""

import math
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not IS_CUTLASS_DSL_AVAILABLE
    or get_sm_version() not in (100, 103, 107),
    reason="CuTe DSL MLA decode runs on SM100/SM103/SM107 only (see CuteDslMlaFmha)",
)

# One SM107 wave: 212 SMs, one 2-CTA cluster per SM pair -> 106 clusters.
WAVE = 106


def _plan_mod():
    import tensorrt_llm._torch.attention.backends.fmha.cute_dsl_mla as mod

    return mod


@pytest.mark.parametrize(
    "kv_lens,rows,expected",
    [
        # helix16 bs8 MTP7 (8 rows per request): 64 clusters fill the wave at one task each.
        ([14208] * 8, 8, [1] * 8),
        # helix16 bs4: 32 rows -> three pieces per request (96 clusters, one wave).
        ([14208] * 4, 8, [3] * 4),
        # helix16 bs2: six pieces of 19 tiles.
        ([14208] * 2, 8, [6] * 2),
        # helix16 bs16: already over one wave, a second piece would cost a third wave.
        ([14208] * 16, 8, [1] * 16),
        # One long request alone (H128, single token): the whole grid.
        ([65536], 1, [32]),
        # 64K x 8 (64 rows): three pieces fill two waves better than one 512-tile pass.
        ([65536] * 8, 8, [3] * 8),
        # Short requests never get pieces below the minimum tile count.
        ([512] * 4, 8, [1] * 4),
        # DEP skew: the 256K request takes eight 256-tile pieces, the 32K ones stay whole.
        ([32768] * 15 + [262144], 8, [1] * 15 + [8]),
    ],
)
def test_plan_mla_split_kv_policy(kv_lens: list[int], rows: int, expected: list[int]) -> None:
    plan = _plan_mod().plan_mla_split_kv(kv_lens, rows_per_request=rows, wave_clusters=WAVE)
    assert plan == expected


@pytest.mark.parametrize("piece_cost", [None, 0, 30])
def test_plan_kernel_matches_host_mirror(piece_cost) -> None:
    """Every piece cost (default: the target decode's), and every block size the launch picks
    for the batch (one warp of lanes up to the request cap)."""
    mod = _plan_mod()
    generator = torch.Generator().manual_seed(3)
    cost = {} if piece_cost is None else {"piece_cost": piece_cost}
    for rows in (1, 4, 8):
        for batch in (1, 3, 8, 16, 37, 128, 300, 1024):
            kv = torch.randint(1, 300_000, (batch,), generator=generator)
            kv[0] = 262144  # a long one
            kv_lens = kv.tolist()
            expected = mod.plan_mla_split_kv(
                kv_lens, rows_per_request=rows, wave_clusters=WAVE, **cost
            )
            cache_seqs = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
            plan = torch.full((batch + 3,), -5, dtype=torch.int32, device="cuda")
            mod.launch_mla_split_kv_plan(
                cache_seqs,
                plan,
                num_requests=batch,
                rows_per_request=rows,
                wave_clusters=WAVE,
                **cost,
            )
            torch.cuda.synchronize()
            assert plan[:batch].tolist() == expected, (rows, batch, kv_lens[:4])
            assert (plan[batch:] == -5).all()


def test_split_kv_plan_is_launched_once_per_step(monkeypatch) -> None:
    mod = _plan_mod()
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner

    monkeypatch.setattr(
        CuteDSLNVMlaDecodeBlackwellRunner, "_get_max_active_blocks", classmethod(lambda cls: 212)
    )
    launches = []
    monkeypatch.setattr(
        mod, "launch_mla_split_kv_plan", lambda *args, **kwargs: launches.append(kwargs)
    )
    meta = SimpleNamespace(max_num_requests=16, _cute_dsl_mla_split_plan_key=None)
    cache_seqs = torch.full((8,), 14208, dtype=torch.int32, device="cuda")
    first = mod.CuteDslMlaFmha._split_kv_plan(meta, cache_seqs, 8, 8, 96)
    assert first.numel() == 8 and meta._cute_dsl_mla_split_plan.numel() == 16
    # 8 tokens x 96 heads pack into six 128-row query tiles per request.
    assert launches == [dict(num_requests=8, rows_per_request=6, wave_clusters=106)]
    # The other layers of the step reuse the plan.
    second = mod.CuteDslMlaFmha._split_kv_plan(meta, cache_seqs, 8, 8, 96)
    assert second.data_ptr() == first.data_ptr() and len(launches) == 1
    # A new step (metadata invalidation) plans again.
    meta._cute_dsl_mla_split_plan_key = None
    mod.CuteDslMlaFmha._split_kv_plan(meta, cache_seqs, 8, 8, 96)
    assert len(launches) == 2
    # Beyond the plan's request capacity: uniform split.
    assert mod.CuteDslMlaFmha._split_kv_plan(meta, cache_seqs, 2000, 8, 96) is None


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("empty_request", [False, True])
def test_runner_variable_split_matches_uniform_split(
    input_dtype: torch.dtype, empty_request: bool
) -> None:
    """Per-request splits under a 16-slot grid reproduce the uniform-split output and Helix stats.

    Covers requests planned onto one split (stored directly by the split
    kernel), onto several, onto more splits than KV tiles, and a request
    without local KV (the reduction's canonical empty result).
    """
    import cutlass

    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner

    torch.manual_seed(5)
    device = torch.device("cuda")
    batch_size, seq_len_q, num_heads = 4, 2, 96
    latent_dim, rope_dim, page_size = 512, 64, 64
    kv_lens = [4096, 0 if empty_request else 300, 9000, 1500]
    blocks_per_sequence = math.ceil(max(kv_lens) / page_size)
    num_pages = batch_size * blocks_per_sequence
    softmax_scale = 1.0 / math.sqrt(latent_dim + rope_dim)
    q_storage = (
        torch.randn(batch_size, seq_len_q, num_heads, latent_dim + rope_dim, device=device) * 0.1
    ).to(input_dtype)
    cache_storage = (
        torch.randn(num_pages, page_size, latent_dim + rope_dim, device=device) * 0.1
    ).to(input_dtype)
    q_latent = q_storage[..., :latent_dim].permute(2, 3, 1, 0)
    q_rope = q_storage[..., latent_dim:].permute(2, 3, 1, 0)
    c_latent = cache_storage[..., :latent_dim].permute(1, 2, 0)
    c_rope = cache_storage[..., latent_dim:].permute(1, 2, 0)
    page_table = (
        torch.arange(num_pages, dtype=torch.int32, device=device)
        .view(batch_size, blocks_per_sequence)
        .transpose(0, 1)
    )
    cache_seqs = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    kv_bounds = torch.tensor(
        [max(k - (seq_len_q - 1 - q), 0) for k in kv_lens for q in range(seq_len_q)],
        dtype=torch.int32,
        device=device,
    )
    cutlass_dtype = cutlass.Float8E4M3FN if input_dtype == torch.float8_e4m3fn else cutlass.BFloat16
    runner = CuteDSLNVMlaDecodeBlackwellRunner(
        in_dtype=cutlass_dtype,
        num_heads=num_heads,
        seq_len_q=seq_len_q,
        page_size=page_size,
        max_batch_size=batch_size,
        emit_softmax_stats=True,
    )
    workspace = torch.empty(
        runner.get_max_padded_workspace_size(
            num_heads, seq_len_q, latent_dim, batch_size, cutlass.Float32
        ),
        device=device,
        dtype=torch.uint8,
    )

    def run(block_split_kvs, split_kv):
        output_storage = torch.zeros(
            batch_size, seq_len_q, num_heads, latent_dim, device=device, dtype=torch.bfloat16
        )
        stats = torch.zeros(batch_size * seq_len_q, num_heads, 2, device=device)
        inputs = [
            q_latent,
            q_rope,
            c_latent,
            c_rope,
            page_table,
            cache_seqs,
            output_storage.permute(2, 3, 1, 0),
            workspace,
            stats,
            kv_bounds,
        ]
        kwargs = {"softmax_scale": softmax_scale, "output_scale": 1.0}
        if input_dtype == torch.float8_e4m3fn:
            inputs.extend(
                [torch.tensor([softmax_scale], device=device), torch.tensor([1.0], device=device)]
            )
            kwargs = {}
        if block_split_kvs is not None:
            inputs.append(torch.tensor(block_split_kvs, dtype=torch.int32, device=device))
        runner.forward(inputs, tactic=((128, 128), (128, 256), split_kv, False), **kwargs)
        torch.cuda.synchronize()
        return output_storage.float(), stats

    ref_out, ref_stats = run(None, 1)
    for plan, split_kv in (
        ([3, 1, 7, 2], 16),
        ([16, 16, 16, 16], 16),
        ([40, 1, 2, 1], 8),
        ([1, 1, 1, 1], 16),
    ):
        out, stats = run(plan, split_kv)
        tol = 2e-2 if input_dtype == torch.float8_e4m3fn else 1e-2
        torch.testing.assert_close(out, ref_out, rtol=tol, atol=tol)
        log_partition = stats[:, :, 0] + stats[:, :, 1].log()
        ref_log_partition = ref_stats[:, :, 0] + ref_stats[:, :, 1].log()
        finite = torch.isfinite(ref_log_partition)
        assert torch.equal(torch.isfinite(log_partition), finite)
        torch.testing.assert_close(
            log_partition[finite], ref_log_partition[finite], rtol=1e-4, atol=1e-4
        )


@pytest.mark.parametrize("split_kv,plan", [(7, None), (32, [22, 3, 9]), (32, [32, 1, 13])])
def test_fp8_reduce_unroll_is_bitwise(monkeypatch, split_kv: int, plan) -> None:
    """TRTLLM_K3_DRAFT3_REDUCE_UNROLL: the FP8 split-KV reduction gives the same bits for every
    unroll, uniform and planned splits, eager and under CUDA graph, and the unrolled kernels are
    the ones that ran."""
    import cutlass

    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner

    torch.manual_seed(11)
    device = torch.device("cuda")
    num_heads, seq_len_q, page_size, max_batch_size = 64, 8, 128, 32
    latent_dim, rope_dim = 512, 64
    kv_lens = [5508, 358, 1508]
    batch_size = len(kv_lens)
    blocks_per_sequence = math.ceil(max(kv_lens) / page_size)
    num_pages = batch_size * blocks_per_sequence
    softmax_scale = 1.0 / math.sqrt(192)
    q_storage = (
        torch.randn(batch_size, seq_len_q, num_heads, latent_dim + rope_dim, device=device) * 0.1
    ).to(torch.float8_e4m3fn)
    cache_storage = (
        torch.randn(num_pages, page_size, latent_dim + rope_dim, device=device) * 0.1
    ).to(torch.float8_e4m3fn)
    page_table = (
        torch.randperm(num_pages, device=device)
        .to(torch.int32)
        .view(batch_size, blocks_per_sequence)
        .transpose(0, 1)
    )
    cache_seqs = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    runner = CuteDSLNVMlaDecodeBlackwellRunner(
        in_dtype=cutlass.Float8E4M3FN,
        num_heads=num_heads,
        seq_len_q=seq_len_q,
        page_size=page_size,
        max_batch_size=max_batch_size,
        emit_softmax_stats=True,
    )
    workspace = torch.zeros(
        runner.get_max_padded_workspace_size(
            num_heads, seq_len_q, latent_dim, max_batch_size, cutlass.Float32
        ),
        device=device,
        dtype=torch.uint8,
    )
    inputs = [
        q_storage[..., :latent_dim].permute(2, 3, 1, 0),
        q_storage[..., latent_dim:].permute(2, 3, 1, 0),
        cache_storage[..., :latent_dim].permute(1, 2, 0),
        cache_storage[..., latent_dim:].permute(1, 2, 0),
        page_table,
        cache_seqs,
    ]
    tail = [
        cache_seqs.repeat_interleave(seq_len_q),
        torch.tensor([softmax_scale], device=device),
        torch.tensor([1.0], device=device),
    ]
    if plan is not None:
        tail.append(torch.tensor(plan, dtype=torch.int32, device=device))

    def launch(out, stats):
        runner.forward(
            inputs + [out.permute(2, 3, 1, 0), workspace, stats] + tail,
            tactic=((128, 128), (128, 256), split_kv, False),
        )

    def fresh():
        return (
            torch.full(
                (batch_size, seq_len_q, num_heads, latent_dim),
                float("nan"),
                device=device,
                dtype=torch.bfloat16,
            ),
            torch.full((batch_size * seq_len_q, num_heads, 2), float("nan"), device=device),
        )

    # The unroll each launch compiled in (called only when the split workspace is used).
    applied = []
    resolve = CuteDSLNVMlaDecodeBlackwellRunner._kernel_reduce_unroll.__func__

    def record(cls, kernel_class):
        applied.append(resolve(cls, kernel_class))
        return applied[-1]

    monkeypatch.setattr(CuteDSLNVMlaDecodeBlackwellRunner, "_kernel_reduce_unroll", classmethod(record))

    results = {}
    for unroll in ("1", "4", "8"):
        monkeypatch.setenv("TRTLLM_K3_DRAFT3_REDUCE_UNROLL", unroll)
        results[unroll] = fresh()
        launch(*results[unroll])
    torch.cuda.synchronize()
    assert applied == [1, 4, 8]
    ref_out, ref_stats = results["1"]
    assert not torch.isnan(ref_out.float()).any()
    for unroll in ("4", "8"):
        out, stats = results[unroll]
        assert torch.equal(out.view(torch.int16), ref_out.view(torch.int16)), unroll
        assert torch.equal(stats.view(torch.int32), ref_stats.view(torch.int32)), unroll

    monkeypatch.setenv("TRTLLM_K3_DRAFT3_REDUCE_UNROLL", "8")
    captured = fresh()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(*captured)
    for t in captured:
        t.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured[0].view(torch.int16), ref_out.view(torch.int16))
    assert torch.equal(captured[1].view(torch.int32), ref_stats.view(torch.int32))
