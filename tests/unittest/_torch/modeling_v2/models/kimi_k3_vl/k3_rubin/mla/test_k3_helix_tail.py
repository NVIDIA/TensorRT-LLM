# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Kimi-K3 fused Helix tail (``k3_rubin/kernels/k3_helix_tail``) on one GPU (``P = 1``, a local
receive buffer).

The fold is bitwise the packed split reduction of the FP4 MLA backend (latent rows and statistics,
FP32 or normalized FP16 split partials, any task count, under the rule, wave and makespan task
plans), the V up-projection is within one BF16 ulp of an FP64 reference (and of the production
BMM), and the combine is bitwise ``helix_post_process_native`` of what the senders produced, eager
and replayed in a CUDA graph whose launches alternate the receive buffer halves.
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
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.k3_rubin.kernels import (
        k3_helix_tail as tail_mod,
    )
    from tensorrt_llm._torch.attention.backends.fp4_mla import fp4_mla_packed_reduce as reduce_mod

    return reduce_mod, tail_mod


def _plan_kwargs(plan_kind, physical_m):
    """``launch_packed_plan`` options of a task plan: ``rule`` (tasks in request order), ``wave``
    (simulated waves, full tasks first; the default with K3's decode kernel module) or ``lpt``
    (the makespan plan, ``TRTLLM_FP4_MLA_LPT_PLAN``, with the geometry the decode derives)."""
    if plan_kind == "rule":
        return {}
    if plan_kind == "wave":
        return {"wave_plan": True}
    assert plan_kind == "lpt"
    return {
        "lpt_m_tiles": physical_m // 128,
        "lpt_wave_clusters": max(1, torch.cuda.get_device_properties(0).multi_processor_count // 2),
    }


def _split_case(
    reduce_mod,
    kv_tiles,
    splits,
    wave_tasks,
    query_len,
    seed,
    partial_dtype=torch.float32,
    plan_kind="wave",
):
    """Random decode split state of ``len(kv_tiles)`` requests (the reduction test's recipe):
    each request's first task is empty, unplanned slots hold garbage, rows of single-task
    requests are stored directly in ``rows``; FP16 partials are normalized split outputs."""
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
        **_plan_kwargs(plan_kind, physical_m),
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
    partial = partial.to(partial_dtype)
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


def _fold_inputs(tail_mod, case):
    return tail_mod.FoldInputs(
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


class _Workspace:
    """One rank's receive buffer (two halves) and launch counter."""

    def __init__(self, half_bytes: int):
        self.half_bytes = half_bytes
        self.buffer = torch.full((2 * half_bytes // 4,), -1, dtype=torch.int32, device="cuda")
        self.state = torch.zeros(8, dtype=torch.int64, device="cuda")

    def launch(self, tail_mod, rows, stats, w_uv, out, fold, debug, mx_out=None, ull2=False):
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
            debug=debug,
            ull2=ull2,
        )


def _debug_buffers(num_rows):
    return (
        torch.empty((num_rows, 512), dtype=torch.bfloat16, device="cuda"),
        torch.empty((num_rows, 128), dtype=torch.bfloat16, device="cuda"),
        torch.empty((num_rows, 2), dtype=torch.float32, device="cuda"),
    )


def _check(tail_mod, latent_ref, stats_ref, w_uv, out, debug):
    num_tokens = out.shape[0]
    latent, values, stats = debug
    # Fold: bitwise the reduction's rows and statistics.
    assert torch.equal(latent.view(torch.int16), latent_ref.view(torch.int16))
    assert torch.equal(stats.view(torch.int32), stats_ref.view(torch.int32))
    # V up-projection: the FP64 product of the same BF16 rows, rounded once to BF16, up to the
    # FP32 summation error (relative to the sum of |terms|, which dominates near zero).
    ref = tail_mod.reference_v_up(latent_ref, w_uv)
    lat = latent_ref.view(-1, NUM_HEADS, 512).double()
    exact = torch.einsum("tjk,jnk->tjn", lat, w_uv.double()).reshape(-1, 128)
    terms = torch.einsum("tjk,jnk->tjn", lat.abs(), w_uv.double().abs()).reshape(-1, 128)
    err = (values.double() - exact).abs()
    tol = tail_mod.bf16_ulp(ref).double() + terms * 2.0**-20
    assert bool((err <= tol).all()), float((err - tol).max())
    # And within two ulp of the production BMM's BF16 result (cuBLAS sums in another order).
    bmm = torch.empty((num_tokens, NUM_HEADS, 128), dtype=torch.bfloat16, device="cuda")
    torch.ops.trtllm.bmm_out(
        latent_ref.view(num_tokens, NUM_HEADS, 512).transpose(0, 1),
        w_uv.transpose(1, 2),
        bmm.transpose(0, 1),
    )
    diff = (values.double() - bmm.view(-1, 128).double()).abs()
    tol = 2 * tail_mod.bf16_ulp(bmm.view(-1, 128)).double() + terms * 2.0**-19
    assert bool((diff <= tol).all())
    # Combine of the single source: bitwise helix_post_process_native of the sent values.
    combined = torch.ops.trtllm.helix_post_process_native(
        values.view(num_tokens, 1, NUM_HEADS, 128),
        stats.view(num_tokens, 1, NUM_HEADS, 2),
        1.0,
        1,
    )
    assert torch.equal(out.view(torch.int16), combined.reshape(num_tokens, -1).view(torch.int16))


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
    ([86], 16, 106, 8),  # n1, split 16: more tasks than one load round
    ([90, 10], 16, 106, 8),  # n2, split 16
]


@pytest.mark.parametrize("plan_kind", ["wave", "rule", "lpt"])
@pytest.mark.parametrize(
    "partial_dtype,ull2",
    [(torch.float32, False), (torch.float32, True), (torch.float16, True)],
    ids=["fp32", "fp32-ull2", "fp16-ull2"],
)
@pytest.mark.parametrize("kv_tiles,splits,wave_tasks,query_len", CASES)
def test_helix_tail_fold_matches_reduction(
    kv_tiles, splits, wave_tasks, query_len, partial_dtype, ull2, plan_kind
):
    reduce_mod, tail_mod = _modules()
    case = _split_case(
        reduce_mod,
        kv_tiles,
        splits,
        wave_tasks,
        query_len,
        seed=11,
        partial_dtype=partial_dtype,
        plan_kind=plan_kind,
    )
    tasks = case["plan"][2 : 3 + case["requests"]].diff()
    if splits == 16 and plan_kind == "wave":
        assert int(tasks.max()) > 8, "the case must exceed one fold load round"
    latent_ref, stats_ref = _reduce(reduce_mod, case)
    gen = torch.Generator(device="cuda")
    gen.manual_seed(5)
    w_uv = (torch.randn((NUM_HEADS, 128, 512), device="cuda", generator=gen) * 0.05).to(
        torch.bfloat16
    )
    num_tokens = case["requests"] * query_len
    ws = _Workspace(half_bytes=num_tokens * tail_mod.record_bytes(NUM_HEADS, 128) + 4096)
    out = torch.empty((num_tokens, NUM_HEADS * 128), dtype=torch.bfloat16, device="cuda")
    fold = _fold_inputs(tail_mod, case)
    # Two launches use both buffer halves; each re-arms what it read.
    for _ in range(2):
        debug = _debug_buffers(num_tokens * NUM_HEADS)
        out.fill_(torch.nan)
        ws.launch(tail_mod, case["rows"], None, w_uv, out, fold, debug, ull2=ull2)
        torch.cuda.synchronize()
        _check(tail_mod, latent_ref, stats_ref, w_uv, out, debug)
    assert bool((ws.buffer == -1).all()), "the receive buffer was not re-armed"


@pytest.mark.parametrize("ull2", [False, True], ids=["default", "ull2"])
def test_helix_tail_direct_rows_and_graph_replay(ull2):
    """Rows stored directly (no split partials), several token counts in one CUDA graph."""
    _, tail_mod = _modules()
    gen = torch.Generator(device="cuda")
    gen.manual_seed(3)
    w_uv = (torch.randn((NUM_HEADS, 128, 512), device="cuda", generator=gen) * 0.05).to(
        torch.bfloat16
    )
    token_counts = (8, 16, 37, 64)
    ws = _Workspace(half_bytes=max(token_counts) * tail_mod.record_bytes(NUM_HEADS, 128) + 4096)

    def inputs(num_tokens, seed):
        g = torch.Generator(device="cuda")
        g.manual_seed(seed)
        rows = torch.randn((num_tokens * NUM_HEADS, 512), device="cuda", generator=g).to(
            torch.bfloat16
        )
        stats = torch.empty((num_tokens * NUM_HEADS, 2), device="cuda")
        stats[:, 0] = torch.randn(num_tokens * NUM_HEADS, device="cuda", generator=g) * 3
        stats[:, 1] = torch.rand(num_tokens * NUM_HEADS, device="cuda", generator=g) * 50 + 1
        return rows, stats

    cases = [inputs(n, 100 + n) for n in token_counts]
    outs = [
        torch.empty((n, NUM_HEADS * 128), dtype=torch.bfloat16, device="cuda") for n in token_counts
    ]
    debugs = [_debug_buffers(n * NUM_HEADS) for n in token_counts]

    def launch_all():
        for (rows, stats), out, debug in zip(cases, outs, debugs):
            ws.launch(tail_mod, rows, stats, w_uv, out, None, debug, ull2=ull2)

    launch_all()
    torch.cuda.synchronize()
    for (rows, stats), out, debug in zip(cases, outs, debugs):
        _check(tail_mod, rows, stats, w_uv, out, debug)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch_all()
    for rep in range(3):
        for i, (rows, stats) in enumerate(cases):
            new_rows, new_stats = inputs(token_counts[i], 1000 * rep + i)
            rows.copy_(new_rows)
            stats.copy_(new_stats)
        for out in outs:
            out.fill_(torch.nan)
        graph.replay()
        torch.cuda.synchronize()
        for (rows, stats), out, debug in zip(cases, outs, debugs):
            _check(tail_mod, rows, stats, w_uv, out, debug)
    assert bool((ws.buffer == -1).all())
    assert math.isfinite(float(outs[0].float().abs().max()))


@pytest.mark.parametrize("num_tokens", [8, 37, 64])
def test_helix_tail_mxfp8_operand(num_tokens):
    """``TRTLLM_K3_OPROJ_PC``: the receivers' MXFP8 form of ``bf16(out * gate)`` is bitwise the
    gated low-M GEMM's quantization of it, with a row-strided gate."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.k3_rubin.kernels import (
        k3_fp8w_lowm_mxfp8 as mx,
    )

    _, tail_mod = _modules()
    gen = torch.Generator(device="cuda")
    gen.manual_seed(num_tokens)
    w_uv = (torch.randn((NUM_HEADS, 128, 512), device="cuda", generator=gen) * 0.05).to(
        torch.bfloat16
    )
    rows = torch.randn((num_tokens * NUM_HEADS, 512), device="cuda", generator=gen).to(
        torch.bfloat16
    )
    stats = torch.empty((num_tokens * NUM_HEADS, 2), device="cuda")
    stats[:, 0] = torch.randn(num_tokens * NUM_HEADS, device="cuda", generator=gen) * 3
    stats[:, 1] = torch.rand(num_tokens * NUM_HEADS, device="cuda", generator=gen) * 50 + 1
    k = NUM_HEADS * 128
    gate_storage = torch.sigmoid(
        torch.randn((num_tokens, k + 64), device="cuda", generator=gen)
    ).to(torch.bfloat16)
    gate = gate_storage[:, :k]
    xq = torch.empty((num_tokens, k), dtype=torch.float8_e4m3fn, device="cuda")
    xs = torch.empty((num_tokens, k // 32), dtype=torch.uint8, device="cuda")
    ws = _Workspace(half_bytes=num_tokens * tail_mod.record_bytes(NUM_HEADS, 128) + 4096)
    out = torch.empty((num_tokens, k), dtype=torch.bfloat16, device="cuda")
    debug = _debug_buffers(num_tokens * NUM_HEADS)
    ws.launch(tail_mod, rows, stats, w_uv, out, None, debug, mx_out=(gate, xq, xs))
    torch.cuda.synchronize()
    _check(tail_mod, rows, stats, w_uv, out, debug)
    ref_q, ref_s = mx.mxfp8_quantize_reference(out * gate)
    assert torch.equal(xq.view(torch.uint8), ref_q.view(torch.uint8))
    assert torch.equal(xs, ref_s)


def test_tail_output_buffer_is_the_generation_output():
    """The fused tail writes ``forward_absorption_generation``'s output in place: it is given that
    output viewed as ``[T, H / P * v_head_dim]`` (the same storage, so the base's copy into
    ``output`` is skipped), and nothing it could not write as is (no running call, another dtype,
    size or layout)."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.k3_rubin.mla.kimi_k3_mla_attention import (
        KimiK3MLAAttention,
    )

    attn = KimiK3MLAAttention.__new__(KimiK3MLAAttention)
    attn.num_heads_tp_cp = 12
    attn.v_head_dim = 128
    num_tokens, cols = 90, 12 * 128

    attn._gen_output_buffer = None
    assert attn._tail_output_buffer(num_tokens) is None

    storage = torch.empty((num_tokens + 6, cols), dtype=torch.bfloat16, device="cuda")
    output = storage[:num_tokens]
    attn._gen_output_buffer = output
    buf = attn._tail_output_buffer(num_tokens)
    assert buf is not None and buf.shape == (num_tokens, cols)
    assert buf.data_ptr() == output.data_ptr()

    assert attn._tail_output_buffer(num_tokens - 6) is None
    attn._gen_output_buffer = output.float()
    assert attn._tail_output_buffer(num_tokens) is None
    wide = torch.empty((num_tokens, cols + 64), dtype=torch.bfloat16, device="cuda")
    attn._gen_output_buffer = wide[:, :cols]
    assert attn._tail_output_buffer(num_tokens) is None
