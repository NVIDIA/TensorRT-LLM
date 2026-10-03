# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the k3_mla_attn_vb_out catalog entry (and its k3_mla_attn / k3_mla_attn_out forms).

The op reads the paged latent cache through addressing tensors the runtime derives from a KVCacheManager and a
prepared TrtllmAttentionMetadata, and keeps its partials and arrival counters in a caller-owned K3MlaAttnWorkspace.
The test builds that state for real: an MLA (SELFKONLY, kv_factor 1) KVCacheManager of three layers at Kimi K3's
latent width (576 = 512 + 64) and page size (64), every layer's rows written through the manager's own block ids,
a prepared TrtllmAttentionMetadata of R generation requests of T tokens, one TRTLLM MLA attention object per layer
(k3_mla_decode_view gives the op's pool, row stride, page table, layer slot and lengths) and workspaces made by
K3MlaAttnWorkspace.create. The reference reads each request's rows back through the manager's block ids, not
through the op's addressing.

Covered, at Kimi K3's per-rank shapes (6 heads at TP16, 24 at TP4; latent 512, rope 64, v_head 128), in both
launch modes (up to CLUSTER_WAVE = 7 clusters of 16 CTAs; more take the no-cluster mode, which uses the workspace's
arrival counters):

1. Cells: the gated output against a float64 reference (attention output rounded to bf16, v_b in float64, the gate
   applied as bf16(y * s)); the plain v_b output, k3_mla_attn_out and k3_mla_attn on the same step.
2. Call sequences: layers x decode steps on one shared workspace eagerly, then the same steps captured once as a
   CUDA graph and replayed with rewritten inputs and advanced lengths; two workspaces interleaved across layers.
   Every output is bit-identical to the same call on a new workspace, and each workspace's counters account for
   exactly its own no-cluster launches (16 per launch per request and head group).
3. Negative control: a workspace laid out for another head-group count, or of another dtype or size, is refused
   with ValueError before any launch, and neither the output nor the workspace changes.
"""

from typing import List

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.k3_mla_attn_vb_out import (
    k3_mla_attn,
    k3_mla_attn_out,
    k3_mla_attn_vb_out,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.k3_mla_attn_workspace import (
    K3MlaAttnWorkspace,
)
from tensorrt_llm._torch.attention.backends.fmha.cute_dsl_mla import k3_mla_decode_view
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.attention.backends.utils import create_attention
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping


def _is_sm100() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    # sm_100 exactly: the receipts' architecture (a missing receipt reads as unknown).
    return (major, minor) == (10, 0)


pytestmark = pytest.mark.skipif(not _is_sm100(), reason="certified on SM100 only")

H, NOPE, LAT, PE, QL, V = 6, 128, 512, 64, 1536, 128
DQK, PAGE = LAT + PE, 64
SCALE = 1.0 / (NOPE + PE) ** 0.5
GATE_COL0 = QL + DQK  # the gate's columns in the fused projection's rows
LAYERS = 3
MAX_REQUESTS = 8
# Request lengths before the first step, assigned in turn: rows crossing a page boundary (579, 1989), > 2048 rows
# (several tiles per CTA), a short context, exactly one page.
LENGTHS = (1100, 64 * 9 + 3, 2049, 127, 4100, 64 * 31 + 5, 300, 64)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _no_cluster(num_requests: int, heads: int) -> bool:
    """Whether a call of R requests takes the no-cluster mode: more 16-CTA clusters than co-reside, all of whose CTAs
    fit on the SMs at once."""
    from tensorrt_llm._torch.cute_dsl_kernels.k3_mla import k3_mla_attn_kernel as kernel

    clusters = num_requests * heads // H
    sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    return clusters > kernel.CLUSTER_WAVE and clusters * kernel.CLUSTER <= sms


class _MlaCache:
    """Real op state: an MLA (SELFKONLY) paged KV cache manager of LAYERS layers holding R requests that grow by T
    tokens per decode step, every row they will reach written through the manager's block ids, a
    TrtllmAttentionMetadata over it and one TRTLLM MLA attention object per layer for k3_mla_decode_view."""

    def __init__(self, num_requests: int, tokens: int, steps: int, heads: int, seed: int):
        self.tokens, self.heads = tokens, heads
        self.request_ids = list(range(num_requests))
        self.cached = [LENGTHS[(i + seed) % len(LENGTHS)] for i in range(num_requests)]
        final = [n + steps * tokens for n in self.cached]
        pages = sum((n + PAGE - 1) // PAGE for n in final)
        self.mgr = KVCacheManager(
            KvCacheConfig(max_tokens=(pages + 16) * PAGE, enable_block_reuse=False),
            CacheType.SELFKONLY,
            num_layers=LAYERS,
            num_kv_heads=1,
            head_dim=DQK,
            tokens_per_block=PAGE,
            max_seq_len=((max(final) + PAGE - 1) // PAGE + 1) * PAGE,
            max_batch_size=MAX_REQUESTS,
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            dtype=DataType.BF16,
        )
        self.mgr.add_dummy_requests(self.request_ids, token_nums=final)
        self.attn = [
            create_attention(
                "TRTLLM",
                layer_idx=layer,
                num_heads=heads,
                head_dim=DQK,
                num_kv_heads=1,
                is_mla_enable=True,
                q_lora_rank=QL,
                kv_lora_rank=LAT,
                qk_nope_head_dim=NOPE,
                qk_rope_head_dim=PE,
                v_head_dim=V,
            )
            for layer in range(LAYERS)
        ]
        gen = torch.Generator(device="cuda").manual_seed(seed)
        for layer in range(LAYERS):
            pool = self.mgr.get_buffers(layer)
            assert pool.dtype == torch.bfloat16 and tuple(pool.shape[1:]) == (1, PAGE, 1, DQK)
            for blocks, length in zip(self._blocks(layer), final):
                for p in range((length + PAGE - 1) // PAGE):
                    rows = torch.randn(PAGE, DQK, generator=gen, device="cuda") * 0.5
                    pool[blocks[p], 0, :, 0] = rows.bfloat16()

    def _blocks(self, layer: int) -> List[List[int]]:
        return self.mgr.get_batch_cache_indices(self.request_ids, layer)

    def metadata(self) -> TrtllmAttentionMetadata:
        return TrtllmAttentionMetadata(
            max_num_requests=MAX_REQUESTS, max_num_tokens=8192, kv_cache_manager=self.mgr
        )

    def prepare(self, md: TrtllmAttentionMetadata) -> None:
        """The metadata of the next decode step: every request's cached rows plus its T new tokens."""
        md.seq_lens = torch.tensor([self.tokens] * len(self.request_ids), dtype=torch.int)
        md.num_contexts = 0
        md.request_ids = self.request_ids
        md.prompt_lens = [c + self.tokens for c in self.cached]
        md.kv_cache_params = KVCacheParams(
            use_cache=True, num_cached_tokens_per_seq=list(self.cached)
        )
        md.prepare()

    def view(self, md: TrtllmAttentionMetadata, layer: int) -> dict:
        view = k3_mla_decode_view(self.attn[layer], md, len(self.request_ids) * self.tokens)
        assert isinstance(view, dict), view
        assert view["row_stride"] == DQK and view["page_offset"] == self.mgr.layer_offsets[layer]
        assert view["softmax_scale"] == pytest.approx(SCALE)
        return view

    def advance(self) -> None:
        self.cached = [c + self.tokens for c in self.cached]

    def reference(self, layer: int, q: torch.Tensor) -> torch.Tensor:
        """float64 attention of each request's T tokens over its rows [0, L) read through the manager's block ids;
        token t sees rows <= L - T + t. [M, heads, 512]."""
        pool, blocks, outs = self.mgr.get_buffers(layer), self._blocks(layer), []
        t = self.tokens
        for i, cached in enumerate(self.cached):
            length = cached + t
            pages = [blocks[i][p] for p in range((length + PAGE - 1) // PAGE)]
            kv = pool[pages, 0, :, 0].reshape(-1, DQK)[:length].double()
            qi = q[i * t : (i + 1) * t].view(t, self.heads, DQK).double()
            s = torch.einsum("thd,ld->thl", qi, kv) * SCALE
            limit = length - t + torch.arange(t, device="cuda")
            hidden = torch.arange(length, device="cuda")[None, :] > limit[:, None]
            s = s.masked_fill(hidden[:, None, :], float("-inf"))
            outs.append(torch.einsum("thl,ld->thd", torch.softmax(s, dim=-1), kv[:, :LAT]))
        return torch.cat(outs)

    def shutdown(self) -> None:
        self.mgr.shutdown()


def _inputs(seed: int, m: int, heads: int):
    """q (fused_q rows), the v_b weight and the gate (sigmoid values at GATE_COL0 + 128 h)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q = (torch.randn(m, heads * DQK, generator=gen, device="cuda") * 0.5).bfloat16()
    w_vb = (torch.randn(heads, V, LAT, generator=gen, device="cuda") * 0.05).bfloat16()
    gate = torch.rand(m, GATE_COL0 + heads * V, generator=gen, device="cuda").bfloat16()
    return q, w_vb, gate


def _vb(view: dict, q, w_vb, gate, workspace: K3MlaAttnWorkspace) -> torch.Tensor:
    m, heads = q.shape[0], q.shape[1] // DQK
    out = torch.empty(m, heads * V, dtype=torch.bfloat16, device="cuda")
    k3_mla_attn_vb_out(
        q,
        view["pool"],
        view["row_stride"],
        view["page_table"],
        view["page_offset"],
        view["seq_len"],
        view["softmax_scale"],
        w_vb,
        out,
        workspace,
        gate,
        GATE_COL0,
    )
    return out


def _vb_reference(cache: _MlaCache, layer: int, q, w_vb, gate) -> torch.Tensor:
    m, heads = q.shape[0], q.shape[1] // DQK
    o = cache.reference(layer, q).bfloat16().double()
    y = torch.einsum("thc,hvc->thv", o, w_vb.double()).reshape(m, heads * V)
    if gate is None:
        return y
    return y.bfloat16().double() * gate[:, GATE_COL0:].double()


def _max_rel(a, b, tokens: int) -> float:
    """max over requests of max |a - b| / max |b| (per request, so a short context is not hidden by a long one)."""
    err = 0.0
    for i in range(a.shape[0] // tokens):
        ai, bi = (
            a[i * tokens : (i + 1) * tokens].double(),
            b[i * tokens : (i + 1) * tokens].double(),
        )
        err = max(err, (ai - bi).abs().max().item() / max(bi.abs().max().item(), 1e-6))
    return err


def _counters(workspace: K3MlaAttnWorkspace) -> torch.Tensor:
    """The no-cluster arrival counters, [8 requests, groups, 2] (the (m, l) exchange's and the drain's)."""
    from tensorrt_llm._torch.cute_dsl_kernels.k3_mla import k3_mla_attn_kernel as kernel

    g = workspace.groups
    slots = kernel.MAX_REQUESTS * g * kernel.CLUSTER
    # fp16 elements before the counters: the partial slots, then the fp32 (m, l) exchange.
    start = slots * kernel.WS_SLOT_ELEMS + slots * kernel.ROWS * 4
    words = workspace.buffer[start:].view(torch.int32)
    return words.view(kernel.MAX_REQUESTS, g, 2, kernel.CTR_STRIDE)[..., 0].clone()


def _check_counters(workspace: K3MlaAttnWorkspace, num_requests: int, launches: int) -> None:
    """After `launches` no-cluster launches of R requests, every counter of those requests' head groups is
    16 x launches and every other counter is 0."""
    ctrs = _counters(workspace)
    assert bool((ctrs[:num_requests] == 16 * launches).all()), ctrs[:num_requests]
    assert bool((ctrs[num_requests:] == 0).all())


# (R, T, heads): T 1 (decode without speculation) and 8 (a DSpark verify step); the cluster and no-cluster modes;
# the TP4 head count (4 head groups: several clusters per request).
CASES = [(3, 8, H), (8, 1, H), (2, 8, 4 * H)]
CASE_IDS = [f"{r}x{t}-h{h}" for r, t, h in CASES]
EAGER_STEPS, GRAPH_STEPS = 2, 2


@pytest.mark.parametrize("num_requests,tokens,heads", CASES, ids=CASE_IDS)
def test_cells(num_requests, tokens, heads):
    """One step of layer 0: the gated output within 1e-2 (per request) of the float64 reference and rerun-identical;
    the plain v_b output within 1e-2 of its reference, and the gated one bit-identical to bf16(plain * s);
    k3_mla_attn_out within 1e-2 of the attention reference and k3_mla_attn bit-identical to it."""
    seed = 17 * num_requests + tokens + heads
    cache = _MlaCache(num_requests, tokens, 1, heads, seed)
    m = num_requests * tokens
    try:
        md = cache.metadata()
        cache.prepare(md)
        view = cache.view(md, 0)
        workspace = K3MlaAttnWorkspace.create(torch.device("cuda"), heads // H)
        q, w_vb, gate = _inputs(seed, m, heads)
        y = _vb(view, q, w_vb, gate, workspace)
        again = _vb(view, q, w_vb, gate, workspace)
        plain = _vb(view, q, w_vb, None, workspace)
        o = torch.empty(m, heads * LAT, dtype=torch.bfloat16, device="cuda")
        k3_mla_attn_out(
            q, view["pool"], DQK, view["page_table"], view["page_offset"], view["seq_len"],
            view["softmax_scale"], o, workspace,
        )  # fmt: skip
        assert view["page_offset"] == 0  # layer 0: k3_mla_attn takes no page offset
        o_new = k3_mla_attn(
            q,
            view["pool"],
            DQK,
            view["page_table"],
            view["seq_len"],
            view["softmax_scale"],
            workspace,
        )
        torch.cuda.synchronize()
        assert _max_rel(y, _vb_reference(cache, 0, q, w_vb, gate), tokens) <= 1e-2
        assert _max_rel(plain, _vb_reference(cache, 0, q, w_vb, None), tokens) <= 1e-2
        assert torch.equal(_bits(y), _bits(again))
        assert torch.equal(_bits(y), _bits(plain * gate[:, GATE_COL0:]))
        ref_o = cache.reference(0, q).reshape(m, heads * LAT)
        assert _max_rel(o, ref_o, tokens) <= 1e-2
        assert torch.equal(_bits(o_new), _bits(o))
    finally:
        cache.shutdown()


@pytest.mark.parametrize("num_requests,tokens,heads", CASES, ids=CASE_IDS)
def test_layers_by_steps_eager_then_graph(num_requests, tokens, heads):
    """LAYERS layers x EAGER_STEPS decode steps on one shared workspace, eagerly; then one step's LAYERS calls
    captured as a CUDA graph on the same workspace and replayed for GRAPH_STEPS steps with rewritten q and the
    metadata prepared for each step. Every output within 1e-2 of the reference and bit-identical to the same call
    on a new workspace; in the no-cluster mode the shared workspace's counters end at 16 x its launches."""
    seed = 23 * num_requests + tokens + heads
    groups = heads // H
    cache = _MlaCache(num_requests, tokens, EAGER_STEPS + GRAPH_STEPS, heads, seed)
    m = num_requests * tokens
    shared = K3MlaAttnWorkspace.create(torch.device("cuda"), groups)
    launches = 0
    try:
        weights = [_inputs(seed + layer, m, heads)[1:] for layer in range(LAYERS)]
        md = cache.metadata()
        for step in range(EAGER_STEPS):
            cache.prepare(md)
            for layer in range(LAYERS):
                q = _inputs(1000 * step + 10 * layer + seed, m, heads)[0]
                view = cache.view(md, layer)
                y = _vb(view, q, *weights[layer], shared)
                launches += 1
                fresh = _vb(view, q, *weights[layer], K3MlaAttnWorkspace.create("cuda", groups))
                ref = _vb_reference(cache, layer, q, *weights[layer])
                torch.cuda.synchronize()
                assert torch.equal(_bits(y), _bits(fresh)), (step, layer)
                assert _max_rel(y, ref, tokens) <= 1e-2, (step, layer)
            cache.advance()

        graph_md = cache.metadata().create_cuda_graph_metadata(
            num_requests, max_draft_tokens=tokens - 1
        )
        cache.prepare(graph_md)
        views = [cache.view(graph_md, layer) for layer in range(LAYERS)]
        static_q = [
            torch.zeros(m, heads * DQK, dtype=torch.bfloat16, device="cuda") for _ in range(LAYERS)
        ]
        static_y = [
            torch.empty(m, heads * V, dtype=torch.bfloat16, device="cuda") for _ in range(LAYERS)
        ]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for layer in range(LAYERS):
                w_vb, gate = weights[layer]
                k3_mla_attn_vb_out(
                    static_q[layer], views[layer]["pool"], DQK, views[layer]["page_table"],
                    views[layer]["page_offset"], views[layer]["seq_len"], SCALE, w_vb, static_y[layer],
                    shared, gate, GATE_COL0,
                )  # fmt: skip
        for step in range(EAGER_STEPS, EAGER_STEPS + GRAPH_STEPS):
            if step > EAGER_STEPS:
                cache.prepare(graph_md)
            qs = [_inputs(1000 * step + 10 * layer + seed, m, heads)[0] for layer in range(LAYERS)]
            for layer in range(LAYERS):
                static_q[layer].copy_(qs[layer])
            graph.replay()
            launches += LAYERS
            for layer in range(LAYERS):
                fresh = _vb(views[layer], qs[layer], *weights[layer],
                            K3MlaAttnWorkspace.create("cuda", groups))  # fmt: skip
                ref = _vb_reference(cache, layer, qs[layer], *weights[layer])
                torch.cuda.synchronize()
                assert torch.equal(_bits(static_y[layer]), _bits(fresh)), (step, layer)
                assert _max_rel(static_y[layer], ref, tokens) <= 1e-2, (step, layer)
            cache.advance()
        torch.cuda.synchronize()
        _check_counters(shared, num_requests, launches if _no_cluster(num_requests, heads) else 0)
    finally:
        cache.shutdown()


def test_two_workspaces_interleaved():
    """Two workspaces with the layers' calls alternating between them over 3 no-cluster steps (8 requests at TP16):
    every output bit-identical to the same call on a new workspace, and each workspace's counters at 16 x its own
    launches."""
    num_requests, tokens = 8, 1
    assert _no_cluster(num_requests, H)
    cache = _MlaCache(num_requests, tokens, 3, H, 11)
    pair = [K3MlaAttnWorkspace.create(torch.device("cuda"), 1) for _ in range(2)]
    launches = [0, 0]
    try:
        weights = _inputs(12, num_requests, H)[1:]
        md = cache.metadata()
        for step in range(3):
            cache.prepare(md)
            for layer in range(LAYERS):
                k = (step * LAYERS + layer) % 2
                q = _inputs(100 * step + layer, num_requests, H)[0]
                view = cache.view(md, layer)
                y = _vb(view, q, *weights, pair[k])
                launches[k] += 1
                fresh = _vb(view, q, *weights, K3MlaAttnWorkspace.create("cuda", 1))
                torch.cuda.synchronize()
                assert torch.equal(_bits(y), _bits(fresh)), (step, layer)
            cache.advance()
        for workspace, n in zip(pair, launches):
            _check_counters(workspace, num_requests, n)
    finally:
        cache.shutdown()


def test_workspace_of_another_shape_is_refused():
    """Negative control. A TP16 call (one head group) given a workspace laid out for four head groups, an fp32
    tensor of the same size, or a buffer 8 elements short raises ValueError before any launch: neither the output
    nor the workspace changes."""
    cache = _MlaCache(2, 8, 1, H, 13)
    try:
        md = cache.metadata()
        cache.prepare(md)
        view = cache.view(md, 0)
        q, w_vb, gate = _inputs(14, 16, H)
        tp4 = K3MlaAttnWorkspace.create(torch.device("cuda"), 4)
        right = K3MlaAttnWorkspace.create(torch.device("cuda"), 1)
        out = torch.zeros(16, H * V, dtype=torch.bfloat16, device="cuda")
        before = (_bits(tp4.buffer).clone(), _bits(right.buffer).clone())
        for bad in (
            tp4,
            K3MlaAttnWorkspace(buffer=right.buffer.float(), groups=1),
            K3MlaAttnWorkspace(buffer=right.buffer[:-8], groups=1),
        ):
            with pytest.raises(ValueError, match="workspace"):
                k3_mla_attn_vb_out(
                    q, view["pool"], DQK, view["page_table"], view["page_offset"], view["seq_len"], SCALE,
                    w_vb, out, bad, gate, GATE_COL0,
                )  # fmt: skip
        torch.cuda.synchronize()
        assert not bool(out.any())
        assert torch.equal(_bits(tp4.buffer), before[0]) and torch.equal(
            _bits(right.buffer), before[1]
        )
    finally:
        cache.shutdown()


def test_create_refuses_capture():
    """A workspace is made eagerly: K3MlaAttnWorkspace.create raises under CUDA-graph capture."""
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="capture"):
        with torch.cuda.graph(graph):
            K3MlaAttnWorkspace.create(torch.device("cuda"), 1)
