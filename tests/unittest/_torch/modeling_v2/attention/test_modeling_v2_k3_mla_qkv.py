# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the k3_mla_qkv catalog entry (and its k3_mla_q / k3_mla_qkv_out forms).

The op writes the paged latent cache through addressing tensors the runtime derives from a KVCacheManager and a
prepared TrtllmAttentionMetadata. The test builds that state for real: an MLA (SELFKONLY, kv_factor 1) KVCacheManager
of three layers at Kimi K3's latent width (576 = 512 + 64) and page size (64), a prepared TrtllmAttentionMetadata of R
generation requests of T tokens, and one TRTLLM MLA attention object per layer, whose k3_mla_decode_view gives the
op's pool, row stride, page table, layer slot (page offset) and lengths. Every pool of the manager starts filled with
a sentinel; the expected image of each layer is kept from the manager's own block ids (get_batch_cache_indices), not
from the op's addressing, and compared with the whole pool after every step.

Covered, at Kimi K3's per-rank shapes (6 heads at TP16, 24 at TP4; q_lora 1536, latent 512, rope 64):

1. Cells: fused_q against a reference with the model's bf16 roundings, the cache rows against an fp32 reference
   (rope columns bit-exact), k3_mla_q's and k3_mla_qkv_out's results bit-identical to k3_mla_qkv's.
2. Call sequences on real cache objects: layers x decode steps eagerly, then the same steps captured once as a CUDA
   graph and replayed with rewritten inputs and advanced lengths; two managers' calls interleaved.
3. Negative control: a call given another layer's slot. Nothing raises; it overwrites that layer's rows at the
   step's positions and leaves its own slot unwritten, which the pool comparison sees.
"""

from typing import Dict

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.k3_mla_qkv import (
    k3_mla_q,
    k3_mla_qkv,
    k3_mla_qkv_out,
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
    return major * 10 + minor in (100, 103)


pytestmark = pytest.mark.skipif(not _is_sm100(), reason="needs an SM100 / SM103 GPU")

H, NOPE, PE, QK, LAT, QL, V = 6, 128, 64, 192, 512, 1536, 128
DQK, PAGE = LAT + PE, 64
EPS = KV_EPS = 1e-6
LAYERS = 3
MAX_REQUESTS = 8
SENTINEL = 0x7F7F  # bf16 bits of the largest finite value: no cache row of the test holds it
# Request lengths before the first step, assigned in turn: rows crossing a page boundary (579, 1989), > 2048 rows,
# a short context, exactly one page.
LENGTHS = (1100, 64 * 9 + 3, 2049, 127, 4100, 64 * 31 + 5, 300, 64)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _weights(seed: int, heads: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    w_qa = (1.0 + 0.1 * torch.randn(QL, generator=gen, device="cuda")).bfloat16()
    w_qb = (torch.randn(heads * QK, QL, generator=gen, device="cuda") * 0.03).bfloat16()
    w_kb = (torch.randn(heads, LAT, NOPE, generator=gen, device="cuda") * 0.08).bfloat16()
    w_kv = (1.0 + 0.1 * torch.randn(LAT, generator=gen, device="cuda")).bfloat16()
    return w_qa, w_qb, w_kb, w_kv


def _ag(seed: int, m: int, heads: int) -> torch.Tensor:
    """The fused projection's rows: [q_a 1536 | kv_a latent 512 | rope 64 | gate heads * 128]."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(m, QL + DQK + heads * V, generator=gen, device="cuda") * 0.7).bfloat16()


def _reference_q(ag, w_qa, w_qb, w_kb):
    """The norm in fp32, the GEMMs in float64, bf16 rounding where the model rounds (norm output, q_b output, q_abs)."""
    m, heads = ag.shape[0], w_kb.shape[0]
    x = ag[:, :QL].float()
    qn = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w_qa.float()).bfloat16()
    q = (qn.double() @ w_qb.double().t()).bfloat16().view(m, heads, QK)
    q_abs = torch.einsum("thd,hcd->thc", q[..., :NOPE].double(), w_kb.double()).bfloat16()
    return torch.cat([q_abs, q[..., NOPE:]], dim=-1).reshape(m, heads * DQK)


def _reference_kv(ag, w_kv):
    """The cache rows in fp32 ((x * rrms) * w, one bf16 rounding); the rope columns copied."""
    x = ag[:, QL : QL + LAT].float()
    r = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + KV_EPS)
    return torch.cat([((x * r) * w_kv.float()).bfloat16(), ag[:, QL + LAT : QL + DQK]], dim=-1)


def _max_rel(a, b) -> float:
    return (a.float() - b.float()).abs().max().item() / b.float().abs().max().item()


class _MlaCache:
    """Real op state: an MLA (SELFKONLY) paged KV cache manager of LAYERS layers holding R requests that grow by T
    tokens per decode step, a TrtllmAttentionMetadata over it, one TRTLLM MLA attention object per layer for
    k3_mla_decode_view, and the expected image of every layer's pool."""

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
        self.image: Dict[int, torch.Tensor] = {}
        for layer in range(LAYERS):
            pool = self.mgr.get_buffers(layer)
            assert pool.dtype == torch.bfloat16 and tuple(pool.shape[1:]) == (1, PAGE, 1, DQK)
            pool.view(torch.int16).fill_(SENTINEL)
            self.image[layer] = torch.full(pool.shape, SENTINEL, dtype=torch.int16, device="cuda")
        self.blocks = {
            layer: self.mgr.get_batch_cache_indices(self.request_ids, layer)
            for layer in range(LAYERS)
        }

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
        return view

    def record(self, layer: int, rows: torch.Tensor) -> None:
        """Expect `rows` (the step's cache rows, request-major) at each token's position of `layer`."""
        for i in range(len(self.request_ids)):
            for u in range(self.tokens):
                pos = self.cached[i] + u
                block = self.blocks[layer][i][pos // PAGE]
                self.image[layer][block, 0, pos % PAGE, 0] = _bits(rows[i * self.tokens + u])

    def advance(self) -> None:
        self.cached = [c + self.tokens for c in self.cached]

    def pool_matches(self, layer: int) -> bool:
        return torch.equal(self.mgr.get_buffers(layer).view(torch.int16), self.image[layer])

    def shutdown(self) -> None:
        self.mgr.shutdown()


def _call(view: dict, ag, weights) -> torch.Tensor:
    w_qa, w_qb, w_kb, w_kv = weights
    return k3_mla_qkv(
        ag,
        w_qa,
        EPS,
        w_qb,
        w_kb,
        w_kv,
        KV_EPS,
        view["pool"],
        view["row_stride"],
        view["page_table"],
        view["page_offset"],
        view["seq_len"],
    )


def _dense(ag, weights):
    """fused_q and the step's cache rows from the dense form: the bits k3_mla_qkv stores."""
    w_qa, w_qb, w_kb, w_kv = weights
    rows = torch.empty(ag.shape[0], DQK, dtype=torch.bfloat16, device="cuda")
    y = k3_mla_qkv_out(ag, w_qa, EPS, w_qb, w_kb, w_kv, KV_EPS, rows)
    return y, rows


def _check_cell(y, rows, ag, weights) -> None:
    """fused_q and the stored rows against the references; the q-only form's fused_q bit-identical."""
    w_qa, w_qb, w_kb, w_kv = weights
    m = ag.shape[0]
    y_q = k3_mla_q(ag, w_qa, EPS, w_qb, w_kb)
    ref_q = _reference_q(ag, w_qa, w_qb, w_kb).view(m, -1, DQK)
    ref_kv = _reference_kv(ag, w_kv)
    torch.cuda.synchronize()
    for part in (slice(0, LAT), slice(LAT, DQK)):
        assert _max_rel(y.view(m, -1, DQK)[..., part], ref_q[..., part]) <= 2e-2
    assert _max_rel(rows[:, :LAT], ref_kv[:, :LAT]) <= 1e-2
    assert torch.equal(_bits(rows[:, LAT:]), _bits(ag[:, QL + LAT : QL + DQK]))
    assert torch.equal(_bits(y), _bits(y_q))


# (R, T, heads): T 1 (decode without speculation) and 8 (a DSpark verify step), up to 8 requests, the TP4 head count.
CASES = [(3, 1, H), (8, 1, H), (2, 8, H), (8, 8, H), (2, 8, 4 * H)]
CASE_IDS = [f"{r}x{t}-h{h}" for r, t, h in CASES]
EAGER_STEPS, GRAPH_STEPS = 2, 2


@pytest.mark.parametrize("num_requests,tokens,heads", CASES, ids=CASE_IDS)
def test_layers_by_steps_eager_then_graph(num_requests, tokens, heads):
    """LAYERS layers x EAGER_STEPS decode steps called eagerly, then one step captured as a CUDA graph and replayed
    for GRAPH_STEPS steps with rewritten inputs and the metadata prepared for each step: after every step each
    layer's whole pool equals its expected image (the step's rows at the manager's blocks, nothing else written),
    and every call's outputs pass the cell checks."""
    seed = 31 * num_requests + tokens + heads
    weights = [_weights(seed + layer, heads) for layer in range(LAYERS)]
    cache = _MlaCache(num_requests, tokens, EAGER_STEPS + GRAPH_STEPS, heads, seed)
    m = num_requests * tokens
    try:
        md = cache.metadata()
        for step in range(EAGER_STEPS):
            cache.prepare(md)
            for layer in range(LAYERS):
                ag = _ag(1000 * step + 10 * layer + seed, m, heads)
                y = _call(cache.view(md, layer), ag, weights[layer])
                y_dense, rows = _dense(ag, weights[layer])
                _check_cell(y, rows, ag, weights[layer])
                assert torch.equal(_bits(y), _bits(y_dense))
                cache.record(layer, rows)
            for layer in range(LAYERS):
                assert cache.pool_matches(layer), (step, layer)
            cache.advance()

        graph_md = cache.metadata().create_cuda_graph_metadata(
            num_requests, max_draft_tokens=tokens - 1
        )
        cache.prepare(graph_md)
        views = [cache.view(graph_md, layer) for layer in range(LAYERS)]
        static_ag = [torch.zeros(m, QL + DQK + heads * V, dtype=torch.bfloat16, device="cuda")
                     for _ in range(LAYERS)]  # fmt: skip
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_y = [
                _call(views[layer], static_ag[layer], weights[layer]) for layer in range(LAYERS)
            ]
        for step in range(EAGER_STEPS, EAGER_STEPS + GRAPH_STEPS):
            if step > EAGER_STEPS:
                cache.prepare(graph_md)
            inputs = [_ag(1000 * step + 10 * layer + seed, m, heads) for layer in range(LAYERS)]
            for layer in range(LAYERS):
                static_ag[layer].copy_(inputs[layer])
            graph.replay()
            for layer in range(LAYERS):
                y_dense, rows = _dense(inputs[layer], weights[layer])
                torch.cuda.synchronize()
                assert torch.equal(_bits(static_y[layer]), _bits(y_dense)), (step, layer)
                cache.record(layer, rows)
            for layer in range(LAYERS):
                assert cache.pool_matches(layer), (step, layer)
            cache.advance()
    finally:
        cache.shutdown()


def test_two_managers_interleaved():
    """Two managers (two models' caches) with their calls alternating step by step and layer by layer: each pool
    holds exactly its own rows."""
    caches = [_MlaCache(3, 8, 3, H, seed) for seed in (5, 6)]
    weights = _weights(77, H)
    try:
        mds = [cache.metadata() for cache in caches]
        for step in range(3):
            for cache, md in zip(caches, mds):
                cache.prepare(md)
            for layer in range(LAYERS):
                for k, (cache, md) in enumerate(zip(caches, mds)):
                    ag = _ag(100 * step + 10 * layer + k, 24, H)
                    _call(cache.view(md, layer), ag, weights)
                    cache.record(layer, _dense(ag, weights)[1])
            for cache in caches:
                for layer in range(LAYERS):
                    assert cache.pool_matches(layer), (step, layer)
                cache.advance()
    finally:
        for cache in caches:
            cache.shutdown()


def test_wrong_layer_slot_overwrites_its_neighbour():
    """Negative control. Layer 1's call given layer 0's slot (page offset): nothing raises; layer 0's rows at the
    step's positions now hold layer 1's values and layer 1's slot is unwritten. The op trusts the page offset, and the
    pool comparison detects the misplacement."""
    cache = _MlaCache(3, 8, 1, H, 9)
    weights = _weights(78, H)
    try:
        md = cache.metadata()
        cache.prepare(md)
        ag0, ag1 = _ag(1, 24, H), _ag(2, 24, H)
        _call(cache.view(md, 0), ag0, weights)
        cache.record(0, _dense(ag0, weights)[1])
        assert cache.pool_matches(0) and cache.pool_matches(1)
        wrong = dict(cache.view(md, 1), page_offset=cache.view(md, 0)["page_offset"])
        _call(wrong, ag1, weights)
        torch.cuda.synchronize()
        assert not cache.pool_matches(0)
        assert cache.pool_matches(1)  # nothing written into layer 1's slot
        cache.record(0, _dense(ag1, weights)[1])
        assert cache.pool_matches(0)  # layer 0's step rows are now layer 1's
    finally:
        cache.shutdown()


def test_rejects_out_of_contract():
    """More than 64 tokens, or page-table rows / lengths that do not describe R requests of the call's tokens, raise
    ValueError before any launch; the pool is left as it was."""
    cache = _MlaCache(2, 4, 1, H, 3)
    weights = _weights(79, H)
    try:
        md = cache.metadata()
        cache.prepare(md)
        view = cache.view(md, 0)
        with pytest.raises(ValueError):  # 7 tokens over 2 requests
            _call(view, _ag(4, 7, H), weights)
        with pytest.raises(ValueError):  # 1 page-table row, 2 lengths
            _call(dict(view, page_table=view["page_table"][:1]), _ag(4, 8, H), weights)
        with pytest.raises(ValueError):  # an int64 page table
            _call(dict(view, page_table=view["page_table"].long()), _ag(4, 8, H), weights)
        with pytest.raises(ValueError):  # 65 tokens
            k3_mla_q(_ag(5, 65, H), weights[0], EPS, weights[1], weights[2])
        torch.cuda.synchronize()
        assert all(cache.pool_matches(layer) for layer in range(LAYERS))
    finally:
        cache.shutdown()
