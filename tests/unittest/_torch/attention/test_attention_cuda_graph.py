# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUDA-graph reuse test for the attention backends.

Production captures an attention graph once (at warmup, for a padded batch) and
then *reuses* it across decode steps where only the input *values* change — the
cuda-graph metadata holds pre-allocated buffers whose addresses stay fixed, and
seq_lens are refreshed via ``copy_`` (never reallocated). The kernel must read
the live data from those static buffers on every replay.

This test exercises exactly that contract at the backend level (no full model,
no CUDAGraphRunner): capture a graph for a fixed decode config, then replay it
several times with fresh q/k/v copied into the static buffer, asserting each
replay matches an eager (non-graph) ``run_backend`` run of the same inputs.

We compare graph-vs-eager for the *same* backend (the graph must not change the
math), so the tolerance is tight.
"""

import pytest
import torch
from attention_test_harness import (
    ATOL,
    RTOL,
    BackendCase,
    _build_kv_cache_manager,
    generate_inputs,
    run_backend,
)
from kv_cache_utils import fill_kv_cache_logical
from utils.util import getSMVersion

from tensorrt_llm._torch.attention_backend.interface import PredefinedAttentionMask
from tensorrt_llm._torch.attention_backend.utils import create_attention, get_attention_backend
from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE
from tensorrt_llm._torch.metadata import KVCacheParams

# Uniform decode batch (cuda graphs require a fixed batch/shape at capture).
_CASE = BackendCase(
    num_heads=8,
    num_kv_heads=2,
    head_dim=128,
    seq_lens=[1, 1, 1, 1],
    num_cached_tokens=[48, 48, 48, 48],
    num_contexts=0,
    page_size=64,
)

WARMUP_STEPS = 2


def _cuda_graph_metadata(AttentionCls, case, mgr):
    """Build cuda-graph metadata (is_cuda_graph=True, pre-allocated buffers)."""
    md = AttentionCls.Metadata(
        num_contexts=0,
        kv_cache_params=KVCacheParams(
            use_cache=True, num_cached_tokens_per_seq=case.num_cached_tokens
        ),
        seq_lens=torch.tensor(case.seq_lens, dtype=torch.int),
        max_num_requests=case.num_seqs,
        max_num_tokens=8192,
        kv_cache_manager=mgr,
        request_ids=list(range(case.num_seqs)),
        prompt_lens=case.token_nums,
    )
    cg = md.create_cuda_graph_metadata(case.num_seqs)
    # Refresh the live state into the pre-allocated buffers, then prepare()
    # populates block offsets / kv_lens into those fixed-address tensors.
    cg.seq_lens = torch.tensor(case.seq_lens, dtype=torch.int)
    cg.num_contexts = 0
    cg.prepare()
    return cg


def _backends():
    backends = ["TRTLLM"]
    if IS_FLASHINFER_AVAILABLE:
        backends.append("FLASHINFER")
    return backends


@pytest.mark.parametrize("backend", _backends())
def test_cuda_graph_decode_reuse(backend):
    """Capture one decode graph, replay with fresh data, match eager each time."""
    if getSMVersion() < 80:
        pytest.skip("CUDA graphs for attention require sm>=80")

    case = _CASE
    H, Hkv, D = case.num_heads, case.num_kv_heads, case.head_dim
    request_ids = list(range(case.num_seqs))
    compute_dtype = case.compute_dtype

    # Cached prefix is shared by the graph and every eager comparison.
    base = generate_inputs(case, seed=0)

    AttentionCls = get_attention_backend(backend)
    mgr = _build_kv_cache_manager(case, backend, compute_dtype)
    mgr.add_dummy_requests(request_ids, case.token_nums)
    layout = "NHD" if backend == "VANILLA" else "HND"
    fill_kv_cache_logical(mgr, 0, request_ids, base["cached_k"], base["cached_v"], kv_layout=layout)

    attn = create_attention(backend, layer_idx=0, num_heads=H, head_dim=D, num_kv_heads=Hkv)
    cg_md = _cuda_graph_metadata(AttentionCls, case, mgr)
    mask = PredefinedAttentionMask.CAUSAL

    # Static input buffers: TRTLLM consumes fused QKV; FlashInfer takes q/k/v.
    use_fused = AttentionCls.support_fused_qkv()
    if use_fused:
        static_q = torch.zeros(case.nnz_q, (H + 2 * Hkv) * D, device="cuda", dtype=compute_dtype)
        static_k = static_v = None
    else:
        static_q = torch.zeros(case.nnz_q, H * D, device="cuda", dtype=compute_dtype)
        static_k = torch.zeros(case.nnz_kv, Hkv * D, device="cuda", dtype=compute_dtype)
        static_v = torch.zeros(case.nnz_kv, Hkv * D, device="cuda", dtype=compute_dtype)

    def _set_inputs(ins):
        if use_fused:
            static_q.copy_(torch.cat([ins["q"], ins["new_k"], ins["new_v"]], dim=-1))
        else:
            static_q.copy_(ins["q"])
            static_k.copy_(ins["new_k"])
            static_v.copy_(ins["new_v"])

    def _forward():
        out = attn.forward(static_q, static_k, static_v, cg_md, attention_mask=mask)
        return out[0] if isinstance(out, tuple) else out

    # Prime inputs, then warm up on a side stream. The warmup also runs the
    # backend's metadata-dependent host work (e.g. FlashInfer plan()) OUTSIDE
    # capture -- plan() host-syncs and cannot run while the stream is capturing.
    _set_inputs(generate_inputs(case, seed=100))
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(WARMUP_STEPS):
            _forward()
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = _forward()

    try:
        # Replay several times with fresh inputs; each must match an eager run.
        for step in range(3):
            ins = generate_inputs(case, seed=100 + step)
            _set_inputs(ins)
            graph.replay()
            torch.cuda.synchronize()
            graph_result = graph_out[: case.nnz_q].clone()

            # Eager: same cached prefix (base) + this step's new tokens.
            eager_inputs = dict(
                q=ins["q"],
                new_k=ins["new_k"],
                new_v=ins["new_v"],
                cached_k=base["cached_k"],
                cached_v=base["cached_v"],
            )
            eager_out = run_backend(case, backend, eager_inputs, kv_dtype=compute_dtype)

            torch.testing.assert_close(
                graph_result,
                eager_out,
                atol=ATOL,
                rtol=RTOL,
                msg=lambda m: f"graph replay step {step} != eager\n{m}",
            )
    finally:
        mgr.shutdown()
