# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 fused Helix tail on one GPU (``P = 1``, a local receive buffer).

With a ``W_UV`` that selects 128 latent columns the V up-projection is exact, so the output is
bitwise ``helix_post_process_native`` of the packed split reduction's rows and statistics (the
fold and the combine bitwise). With a random ``W_UV`` the output agrees with the production path
(reduction, BMM, combine) to BF16 rounding of the V up-projection. Both eagerly over both buffer
halves and replayed in a CUDA graph. The optional MXFP8 o_proj operand is bitwise the low-M
GEMM's quantization of ``bf16(out * gate)``.
"""

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="requires an SM100-or-newer CUDA device",
)

NUM_HEADS = 96
SM_SCALE = 0.1
OUTPUT_SCALE = 0.25


def _modules():
    from tensorrt_llm._torch.attention.backends.fp4_mla import fp4_mla_packed_reduce as reduce_mod
    from tensorrt_llm._torch.kernels.blackwell import k3_helix_tail as tail_mod

    return reduce_mod, tail_mod


def _split_case(reduce_mod, kv_tiles, splits, wave_tasks, query_len, seed):
    """Random decode split state of ``len(kv_tiles)`` requests: each split request's first task is
    empty, unplanned slots hold garbage, rows of single-task requests are stored in ``rows``."""
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    requests = len(kv_tiles)
    valid_m = NUM_HEADS * query_len
    physical_m = (valid_m + 127) // 128 * 128
    tasks = splits * requests
    kv_lens = torch.tensor([t * 256 - 3 for t in kv_tiles], dtype=torch.int32, device="cuda")
    plan = torch.empty(
        reduce_mod.packed_plan_length(requests, tasks), dtype=torch.int32, device="cuda"
    )
    reduce_mod.launch_packed_plan(
        kv_lens,
        plan,
        num_requests=requests,
        num_splits=splits,
        wave_tasks=wave_tasks,
        max_tiles_per_task=max(kv_tiles),
        query_len=query_len,
    )
    begins = plan[2 : 3 + requests].tolist()
    partial = torch.randn((tasks, physical_m, 512), device="cuda", generator=gen)
    maxima = torch.randn((tasks, physical_m), device="cuda", generator=gen) * 12
    sums = torch.rand((tasks, physical_m), device="cuda", generator=gen) * 16 + 0.5
    for begin, end in zip(begins, begins[1:]):
        if end - begin > 1:
            sums[begin] = 0
    maxima.masked_fill_(sums == 0, -torch.inf)
    partial.masked_fill_((sums == 0).unsqueeze(-1), torch.nan)
    sums[begins[-1] :] = 3.0
    maxima[begins[-1] :] = 1e9
    partial[begins[-1] :] = torch.nan
    rows = torch.randn((requests * valid_m, 512), device="cuda", generator=gen).to(torch.bfloat16)
    return dict(
        partial=partial,
        maxima=maxima,
        sums=sums,
        plan=plan,
        q_scale=torch.tensor(1.25, device="cuda"),
        kv_scale=torch.tensor(0.75, device="cuda"),
        rows=rows,
        requests=requests,
        splits=splits,
        valid_m=valid_m,
        physical_m=physical_m,
    )


def _reduce(reduce_mod, case):
    """The packed split reduction of ``case``: (latent rows, statistics)."""
    rows = case["rows"].clone()
    stats = torch.empty((rows.shape[0], 2), device="cuda")
    reduce_mod.reduce_packed_splits(
        case["partial"],
        case["maxima"],
        case["sums"],
        rows,
        case["q_scale"],
        case["kv_scale"],
        num_requests=case["requests"],
        num_splits=case["splits"],
        valid_m=case["valid_m"],
        physical_m=case["physical_m"],
        sm_scale=SM_SCALE,
        output_scale=OUTPUT_SCALE,
        plan=case["plan"],
        stats=stats,
        sm_count=torch.cuda.get_device_properties(0).multi_processor_count,
    )
    return rows, stats


def _partials(reduce_mod, case):
    return reduce_mod.PackedSplitPartials(
        case["partial"],
        case["maxima"],
        case["sums"],
        case["plan"],
        case["q_scale"],
        case["kv_scale"],
        num_requests=case["requests"],
        valid_m=case["valid_m"],
        physical_m=case["physical_m"],
        sm_scale=SM_SCALE,
        output_scale=OUTPUT_SCALE,
    )


def _selector(first_column):
    """``W_UV`` whose output column ``n`` of every head is latent column ``first_column + n``."""
    w_uv = torch.zeros((NUM_HEADS, 128, 512), dtype=torch.bfloat16, device="cuda")
    w_uv[:, torch.arange(128), first_column + torch.arange(128)] = 1.0
    return w_uv


def _random_w(seed):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    w = torch.randn((NUM_HEADS, 128, 512), device="cuda", generator=gen) * 0.05
    return w.to(torch.bfloat16)


class _Workspace:
    """One rank's receive buffer (two halves) and launch counter."""

    def __init__(self, half_bytes: int):
        self.half_bytes = half_bytes
        self.buffer = torch.full((2 * half_bytes // 4,), -1, dtype=torch.int32, device="cuda")
        self.state = torch.zeros(1, dtype=torch.int64, device="cuda")

    def launch(self, tail_mod, rows, stats, w_uv, out, fold, mx_out=None):
        tail_mod.k3_helix_tail(
            rows,
            stats,
            w_uv,
            out,
            fold=fold,
            zero_kv_mask=None,
            cp_rank=0,
            cp_size=1,
            ws_addr=self.buffer.data_ptr(),
            rank_stride=self.buffer.numel() * 4,
            half_bytes=self.half_bytes,
            state=self.state,
            max_ctas=torch.cuda.get_device_properties(0).multi_processor_count,
            use_pdl=True,
            mx_out=mx_out,
        )


def _combine(values, stats):
    """``helix_post_process_native`` of one source: ``values`` ``[T * H, 128]``."""
    num_tokens = values.shape[0] // NUM_HEADS
    out = torch.ops.trtllm.helix_post_process_native(
        values.view(num_tokens, 1, NUM_HEADS, 128),
        stats.view(num_tokens, 1, NUM_HEADS, 2),
        1.0,
        1,
    )
    return out.reshape(num_tokens, NUM_HEADS * 128)


def _bmm(latent, w_uv):
    num_tokens = latent.shape[0] // NUM_HEADS
    out = torch.empty((num_tokens, NUM_HEADS, 128), dtype=torch.bfloat16, device="cuda")
    torch.ops.trtllm.bmm_out(
        latent.view(num_tokens, NUM_HEADS, 512).transpose(0, 1),
        w_uv.transpose(1, 2),
        out.transpose(0, 1),
    )
    return out.view(-1, 128)


def _check_bitwise(out, latent_ref, stats_ref, first_column):
    values = latent_ref[:, first_column : first_column + 128]
    expected = _combine(values.contiguous(), stats_ref)
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))


def _check_close(out, latent_ref, stats_ref, w_uv):
    """Within BF16 rounding of the production path: the V up-projections differ by up to two ulp
    (another FP32 summation order), and the combine is a convex sum of them."""
    num_tokens = out.shape[0]
    prod = _combine(_bmm(latent_ref, w_uv), stats_ref).float().view(num_tokens, NUM_HEADS, 128)
    got = out.float().view(num_tokens, NUM_HEADS, 128)
    finite = torch.isfinite(prod) & torch.isfinite(got)
    assert torch.equal(finite, torch.isfinite(prod))
    scale = torch.where(finite, prod, 0.0).abs().amax(-1, keepdim=True)
    err = torch.where(finite, (got - prod).abs(), 0.0)
    assert bool((err <= scale * 2.0**-6 + 1e-6).all()), float(err.max())


def test_log2_e():
    from tensorrt_llm._torch.attention.backends.fp4_mla.fp4_mla_cutedsl_mufu16 import LOG2_E

    assert math.isclose(LOG2_E, math.log2(math.e))


CASES = [
    # (kv tiles per request, splits, wave tasks, query length)
    ([86], 8, 106, 8),  # n1, 21.9k tokens: 8 tasks
    ([1], 8, 106, 8),  # one tile: a single task, rows stored by the decode
    ([86, 60], 8, 106, 8),  # n2
    ([1, 79, 3, 176], 6, 17, 8),  # skewed, single-task requests between multi-task ones
    ([40] * 8, 8, 106, 8),  # n8
    ([20] * 16, 2, 106, 8),  # n16
    ([5, 9, 3], 4, 17, 5),  # tokens not a multiple of the 16-token tile
    ([30, 2, 12], 4, 17, 1),  # one token per request
]


@pytest.mark.parametrize("kv_tiles,splits,wave_tasks,query_len", CASES)
def test_helix_tail_fold_matches_reduction(kv_tiles, splits, wave_tasks, query_len):
    reduce_mod, tail_mod = _modules()
    case = _split_case(reduce_mod, kv_tiles, splits, wave_tasks, query_len, seed=11)
    latent_ref, stats_ref = _reduce(reduce_mod, case)
    num_tokens = case["requests"] * query_len
    ws = _Workspace(half_bytes=num_tokens * tail_mod.record_bytes(NUM_HEADS, 128) + 4096)
    out = torch.empty((num_tokens, NUM_HEADS * 128), dtype=torch.bfloat16, device="cuda")
    fold = _partials(reduce_mod, case)
    # Every 128-column quarter of the folded rows, over both buffer halves; each launch re-arms
    # what it read.
    for first_column in (0, 128, 256, 384):
        out.fill_(torch.nan)
        ws.launch(tail_mod, case["rows"], None, _selector(first_column), out, fold)
        torch.cuda.synchronize()
        _check_bitwise(out, latent_ref, stats_ref, first_column)
    w_uv = _random_w(5)
    ws.launch(tail_mod, case["rows"], None, w_uv, out, fold)
    torch.cuda.synchronize()
    _check_close(out, latent_ref, stats_ref, w_uv)
    assert bool((ws.buffer == -1).all()), "the receive buffer was not re-armed"


def test_helix_tail_direct_rows_and_graph_replay():
    """Rows stored directly (no split partials), several token counts in one CUDA graph."""
    _, tail_mod = _modules()
    w_select = _selector(256)
    w_uv = _random_w(3)
    token_counts = (8, 16, 37, 64)
    ws = _Workspace(half_bytes=max(token_counts) * tail_mod.record_bytes(NUM_HEADS, 128) + 4096)

    def inputs(num_tokens, seed):
        g = torch.Generator(device="cuda")
        g.manual_seed(seed)
        rows = torch.randn((num_tokens * NUM_HEADS, 512), device="cuda", generator=g)
        stats = torch.empty((num_tokens * NUM_HEADS, 2), device="cuda")
        stats[:, 0] = torch.randn(num_tokens * NUM_HEADS, device="cuda", generator=g) * 3
        stats[:, 1] = torch.rand(num_tokens * NUM_HEADS, device="cuda", generator=g) * 50 + 1
        return rows.to(torch.bfloat16), stats

    cases = [inputs(n, 100 + n) for n in token_counts]
    outs = [
        [torch.empty((n, NUM_HEADS * 128), dtype=torch.bfloat16, device="cuda") for _ in range(2)]
        for n in token_counts
    ]

    def launch_all():
        for (rows, stats), (out_select, out_random) in zip(cases, outs):
            ws.launch(tail_mod, rows, stats, w_select, out_select, None)
            ws.launch(tail_mod, rows, stats, w_uv, out_random, None)

    def check_all():
        for (rows, stats), (out_select, out_random) in zip(cases, outs):
            _check_bitwise(out_select, rows, stats, 256)
            _check_close(out_random, rows, stats, w_uv)

    launch_all()
    torch.cuda.synchronize()
    check_all()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch_all()
    for rep in range(3):
        for i, (rows, stats) in enumerate(cases):
            new_rows, new_stats = inputs(token_counts[i], 1000 * rep + i)
            rows.copy_(new_rows)
            stats.copy_(new_stats)
        for pair in outs:
            for out in pair:
                out.fill_(torch.nan)
        graph.replay()
        torch.cuda.synchronize()
        check_all()
    assert bool((ws.buffer == -1).all())


def _direct_inputs(num_tokens, seed):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    rows = torch.randn((num_tokens * NUM_HEADS, 512), device="cuda", generator=g)
    stats = torch.empty((num_tokens * NUM_HEADS, 2), device="cuda")
    stats[:, 0] = torch.randn(num_tokens * NUM_HEADS, device="cuda", generator=g) * 3
    stats[:, 1] = torch.rand(num_tokens * NUM_HEADS, device="cuda", generator=g) * 50 + 1
    return rows.to(torch.bfloat16), stats


@pytest.mark.parametrize("extra_tokens", [0, 183])
def test_helix_tail_tokens_beyond_sm_count(extra_tokens):
    """At least as many tokens as SMs: the grid stays within one CTA per SM (every CTA polls
    after its pushes, so a non-resident sender would hang the launch)."""
    _, tail_mod = _modules()
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    num_tokens = sm_count + extra_tokens
    for tokens in range(1, num_tokens + 1):
        assert tail_mod._grid(NUM_HEADS, tokens, sm_count)[1] <= sm_count
    rows, stats = _direct_inputs(num_tokens, num_tokens)
    ws = _Workspace(half_bytes=num_tokens * tail_mod.record_bytes(NUM_HEADS, 128) + 4096)
    out = torch.full((num_tokens, NUM_HEADS * 128), torch.nan, dtype=torch.bfloat16, device="cuda")
    ws.launch(tail_mod, rows, stats, _selector(128), out, None)
    torch.cuda.synchronize()
    _check_bitwise(out, rows, stats, 128)
    assert bool((ws.buffer == -1).all())


@pytest.mark.parametrize("num_tokens", [8, 37, 64, 130])
def test_helix_tail_mxfp8_operand(num_tokens):
    """The receivers' MXFP8 form of ``bf16(out * gate)`` (R128c4 scales) is bitwise the low-M
    GEMM's quantization of it, with a row-strided gate."""
    from tensorrt_llm._torch.kernels.blackwell import fp8_block_scale_low_m_gemm as low_m

    _, tail_mod = _modules()
    gen = torch.Generator(device="cuda")
    gen.manual_seed(num_tokens)
    w_uv = _random_w(num_tokens)
    rows = torch.randn((num_tokens * NUM_HEADS, 512), device="cuda", generator=gen)
    rows = rows.to(torch.bfloat16)
    stats = torch.empty((num_tokens * NUM_HEADS, 2), device="cuda")
    stats[:, 0] = torch.randn(num_tokens * NUM_HEADS, device="cuda", generator=gen) * 3
    stats[:, 1] = torch.rand(num_tokens * NUM_HEADS, device="cuda", generator=gen) * 50 + 1
    k = NUM_HEADS * 128
    gate_storage = torch.randn((num_tokens, k + 64), device="cuda", generator=gen).sigmoid()
    gate = gate_storage.to(torch.bfloat16)[:, :k]
    xq = torch.empty((num_tokens, k), dtype=torch.float8_e4m3fn, device="cuda")
    xs = torch.empty((-(-num_tokens // 128) * 128 * (k // 32),), dtype=torch.uint8, device="cuda")
    ws = _Workspace(half_bytes=num_tokens * tail_mod.record_bytes(NUM_HEADS, 128) + 4096)
    out = torch.empty((num_tokens, k), dtype=torch.bfloat16, device="cuda")
    ws.launch(tail_mod, rows, stats, w_uv, out, None, mx_out=(gate, xq, xs))
    torch.cuda.synchronize()
    _check_close(out, rows, stats, w_uv)
    ref_q, ref_s = low_m.mxfp8_quantize_reference(out * gate)
    assert torch.equal(xq.view(torch.uint8), ref_q.view(torch.uint8))
    r = torch.arange(num_tokens, device="cuda").unsqueeze(1)
    c = torch.arange(k // 32, device="cuda").unsqueeze(0)
    offsets = ((r // 128) * (k // 128) + c // 4) * 512 + (r % 32) * 16 + (r % 128) // 32 * 4 + c % 4
    assert torch.equal(xs[offsets], ref_s)
