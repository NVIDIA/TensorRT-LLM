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

import copy
import functools
import weakref
from types import SimpleNamespace
from typing import Optional

import backend_case
import pytest
import torch
from backend_case import BackendCase, run_case
from kv_cache_utils import fill_kv_cache_logical

import tensorrt_llm
from tensorrt_llm._torch.attention.backends.fmha import gqa_kv_replica_fused as F
from tensorrt_llm._torch.attention.backends.fmha.gqa_decode import (
    FoldInputs,
    GqaDecodeFmha,
    GqaDecodeSetup,
    GqaKvReplicaFmha,
    GqaKvReplicaFusedFmha,
    GqaSegmentedFmha,
    GqaSplitKvFmha,
    KvAppendPlan,
)
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention, TrtllmAttentionMetadata
from tensorrt_llm._torch.attention.backends.utils import create_attention
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_qwen3 import Qwen3Attention
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

transformers = pytest.importorskip("transformers")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not IS_CUTLASS_DSL_AVAILABLE
    or get_sm_version() not in (100, 103),
    reason="the GQA decode kernels need an SM100 / SM103 GPU with the CuTe DSL",
)

HIDDEN, HEADS, HEAD_DIM, PAGE = 512, 4, 256, 64
CP = 4


@pytest.mark.parametrize(
    "num_heads,num_kv_heads,seq_lens,num_cached_tokens,library",
    [
        # split-KV decode: one KV head, up to four requests
        (4, 1, [1, 1, 1], [300, 7000, 65], "GqaSplitKvFmha"),
        # ~40K-token contexts: more than two tiles per CTA, so the K / V ring phases wrap
        (4, 1, [1, 1], [40000, 38500], "GqaSplitKvFmha"),
        (4, 1, [1, 1, 1, 1], [41000, 300, 40000, 39000], "GqaSplitKvFmha"),
        # segmented: two KV heads, up to two requests
        (32, 2, [1, 1], [5000, 130], "GqaSegmentedFmha"),
        # neither: trtllm-gen
        (32, 2, [1, 1, 1], [5000, 130, 70], None),
    ],
)
def test_gqa_decode_matches_golden(
    monkeypatch: pytest.MonkeyPatch,
    num_heads: int,
    num_kv_heads: int,
    seq_lens: list,
    num_cached_tokens: list,
    library: Optional[str],
) -> None:
    monkeypatch.setattr(
        backend_case,
        "create_attention",
        functools.partial(backend_case.create_attention, gqa_decode=GqaDecodeSetup()),
    )
    monkeypatch.setattr(backend_case, "BACKENDS_UNDER_TEST", ("TRTLLM",))
    taken = []
    for cls in (GqaSplitKvFmha, GqaSegmentedFmha):

        def spy(self, *args, _run=cls._run_dedicated, _name=cls.__name__, **kwargs):
            taken.append((_name, _run(self, *args, **kwargs)))
            return taken[-1][1]

        monkeypatch.setattr(cls, "_run_dedicated", spy)
    run_case(
        BackendCase(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=HEAD_DIM,
            seq_lens=seq_lens,
            num_cached_tokens=num_cached_tokens,
            num_contexts=0,
            dtype="bfloat16",
            kv_dtype="fp8",
            page_size=PAGE,
            kv_layout="HND",
        )
    )
    if library is None:
        assert taken == []
    else:
        assert taken and all(t == (library, True) for t in taken)


def _qwen3_next_attention(use_gqa_decode_fmha: bool) -> Qwen3Attention:
    config = transformers.Qwen3NextConfig(
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        num_key_value_heads=1,
        head_dim=HEAD_DIM,
        partial_rotary_factor=0.25,
        max_position_embeddings=65536,
        torch_dtype=torch.bfloat16,
    )
    config.partial_rotary_factor = 0.25
    model_config = ModelConfig(
        pretrained_config=config,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8),
    )
    attn = Qwen3Attention(
        model_config,
        layer_idx=0,
        attn_output_gate=True,
        use_gemma_rms_norm=True,
        use_gqa_decode_fmha=use_gqa_decode_fmha,
    ).cuda()
    attn._fuse_qk_norm_rope_gate = True
    generator = torch.Generator(device="cuda").manual_seed(0)
    with torch.no_grad():
        for param in attn.parameters():
            param.copy_(torch.randn(param.shape, generator=generator, device="cuda") * 0.05)
    return attn


def _generation_step(use_gqa_decode_fmha: bool, cached: list, q_len: int):
    attn = _qwen3_next_attention(use_gqa_decode_fmha)
    num_seqs = len(cached)
    request_ids = list(range(num_seqs))
    token_nums = [c + q_len for c in cached]
    pages = (max(token_nums) + PAGE - 1) // PAGE
    manager = KVCacheManagerV2(
        KvCacheConfig(max_tokens=num_seqs * pages * PAGE),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=1,
        num_kv_heads=1,
        head_dim=HEAD_DIM,
        tokens_per_block=PAGE,
        max_seq_len=pages * PAGE,
        max_batch_size=num_seqs,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=tensorrt_llm.bindings.DataType.FP8,
    )
    manager.add_dummy_requests(request_ids, token_nums)
    generator = torch.Generator(device="cuda").manual_seed(1)
    cached_kv = [
        torch.randn(2, c, 1, HEAD_DIM, generator=generator, device="cuda", dtype=torch.bfloat16)
        for c in cached
    ]
    fill_kv_cache_logical(
        manager,
        0,
        request_ids,
        [kv[0] for kv in cached_kv],
        [kv[1] for kv in cached_kv],
        kv_layout="HND",
    )
    metadata = TrtllmAttentionMetadata(
        num_contexts=0,
        kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=cached),
        seq_lens=torch.tensor([q_len] * num_seqs, dtype=torch.int),
        max_num_requests=num_seqs,
        max_num_tokens=num_seqs * q_len,
        kv_cache_manager=manager,
        request_ids=request_ids,
        prompt_lens=token_nums,
    )
    metadata.prepare()
    hidden = torch.randn(num_seqs * q_len, HIDDEN, generator=generator, device="cuda").to(
        torch.bfloat16
    )
    positions = torch.tensor(
        [c + t for c in cached for t in range(q_len)], dtype=torch.int32, device="cuda"
    ).view(1, -1)
    with torch.no_grad():
        output = attn(positions, hidden, metadata)
    torch.cuda.synchronize()
    cache = manager.get_buffers(0, kv_layout="HND").clone()
    manager.shutdown()
    return output, cache


@pytest.mark.parametrize(
    "cached,q_len,fmha_libs,path",
    [
        # up to four requests without a KV-replica group: the split-KV decode kernel runs the prologue (fold)
        ([5000], 7, None, "fold"),
        ([300, 7000, 65], 1, None, "fold"),
        # ~40K-token contexts: more than two tiles per CTA, so the K / V ring phases wrap
        ([40000, 38500], 7, None, "fold"),
        ([41000, 300, 40000, 39000], 7, None, "fold"),
        # a deferred prologue reaching another GQA decode library: it runs the prologue kernel itself
        ([5000], 7, "-gqa_split_kv", "append"),
        # more: the module's prologue appends the FP8 K / V, trtllm-gen decodes its prepared query
        ([900, 3000, 4100, 70, 8000], 7, None, "append"),
    ],
)
def test_qwen3_next_generation_step_matches_stock(
    monkeypatch: pytest.MonkeyPatch, cached: list, q_len: int, fmha_libs: Optional[str], path: str
) -> None:
    if fmha_libs is not None:
        monkeypatch.setenv("TLLM_FMHA_LIBS", fmha_libs)
    taken = {"fold": [], "append": []}
    run_folded = GqaSplitKvFmha._run_folded
    plan_matches = GqaDecodeFmha._plan_matches

    def spy_folded(self, *args, **kwargs):
        taken["fold"].append(run_folded(self, *args, **kwargs))
        return taken["fold"][-1]

    def spy_appended(*args, **kwargs):
        result = plan_matches(*args, **kwargs)
        taken["append"].append(result)
        return result

    monkeypatch.setattr(GqaSplitKvFmha, "_run_folded", spy_folded)
    monkeypatch.setattr(GqaDecodeFmha, "_plan_matches", staticmethod(spy_appended))

    reference, reference_cache = _generation_step(False, cached, q_len)
    assert taken == {"fold": [], "append": []}
    output, cache = _generation_step(True, cached, q_len)
    assert taken[path] == [True]
    # The FP8 K / V of the new tokens are the bytes the C++ generation preprocess writes.
    assert torch.equal(cache.view(torch.uint8), reference_cache.view(torch.uint8))
    # FP8 P in the split-KV kernel vs trtllm-gen: a few percent of the row max per element
    assert ((output.float() - reference.float()).norm() / reference.float().norm()).item() < 0.05


def _kv_replica_fused_partial_ref(q_all, kv_pool, block_tables, seq_lens, q_len, cp_rank, bmm1):
    """FP32 partials of this rank's pages ((j // PAGE) % CP == cp_rank) in the alltoall_merge layout: out_o
    [CP, rows, HQ, D] normalized by this rank's sum (0 if no visible key), out_lse base-2 (-inf if none)."""
    cp, rows, hq, d = q_all.shape
    b = rows // q_len
    dev = q_all.device
    own = torch.arange(cp_rank, block_tables.shape[2], CP, device=dev)
    k = kv_pool[block_tables[:b, 0].index_select(1, own).long(), 0].float()  # [b, n_own, PAGE, D]
    v = kv_pool[block_tables[:b, 1].index_select(1, own).long(), 0].float()
    pos = own[:, None] * PAGE + torch.arange(PAGE, device=dev)[None, :]
    # queries [b, q_len, G, D], group head g = s * hq + h
    q = q_all.float().reshape(cp, b, q_len, hq, d).permute(1, 2, 0, 3, 4).reshape(b, q_len, -1, d)
    s = torch.einsum("btgd,bnpd->btgnp", q, k) * bmm1
    limit = seq_lens[:b].long()[:, None] - q_len + torch.arange(q_len, device=dev)[None, :] + 1
    vis = pos[None, None] < limit[:, :, None, None]
    s = s.masked_fill(~vis[:, :, None], float("-inf"))
    m = s.amax(dim=(3, 4), keepdim=True)
    p = torch.exp(s - torch.where(torch.isfinite(m), m, torch.zeros_like(m)))
    den = p.sum(dim=(3, 4))
    o = torch.einsum("btgnp,bnpd->btgd", p, v) / den.clamp_min(1e-30)[..., None]
    o = torch.where((den > 0)[..., None], o, torch.zeros_like(o))
    lse = torch.where(
        den > 0,
        m.squeeze(-1).squeeze(-1) * 1.4426950408889634 + torch.log2(den.clamp_min(1e-30)),
        torch.full_like(den, float("-inf")),
    )
    out_o = o.reshape(b, q_len, cp, hq, d).permute(2, 0, 1, 3, 4).reshape(cp, rows, hq, d)
    out_lse = lse.reshape(b, q_len, cp, hq).permute(2, 0, 1, 3).reshape(cp, rows, hq)
    return out_o, out_lse


@pytest.mark.parametrize("q_len", [7, 1])
@pytest.mark.parametrize(
    "lens",
    [
        [7, 9000, 33000, 64, 130, 21000, 1, 4000],
        # wide plan (> 32 requests): the longest request in the upper half, tiny and empty-prefix requests
        [1 + (i * 7919) % 50000 for i in range(40)] + [120000],
        # two-slot kernel (> 64 requests): the longest request last; uniform long contexts
        [1 + (i * 7919) % 50000 for i in range(100)] + [120000],
        [65536] * 76,
    ],
    ids=["balanced", "wide", "two_slot", "two_slot_uniform"],
)
def test_kv_replica_fused_partial_matches_reference(q_len: int, lens: list) -> None:
    _check_kv_replica_fused_partial(q_len, lens)


@pytest.mark.parametrize("q_len", [7, 1])
def test_kv_replica_fused_two_slot_plan_up_to_64_requests(q_len: int) -> None:
    """At most 64 requests take the two-slot plan where a whole-CTA plan would leave CTAs idle; its partials match
    the reference as the plan of larger batches does."""
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    n_ctas = F._num_ctas(sms)
    batches = [b for b in range(F._LANE_PLAN_MAX_BATCH + 1, 65) if F._plan(b, n_ctas) == "two"]
    if not batches:
        pytest.skip(f"no batch of at most 64 requests takes the two-slot plan on {sms} SMs")
    b = batches[-1]
    _check_kv_replica_fused_partial(q_len, [1 + (i * 7919) % 50000 for i in range(b - 1)] + [90000])


@pytest.mark.parametrize("q_len", [7, 1])
@pytest.mark.parametrize(
    "lens",
    [
        [7, 9000, 33000, 64, 130, 21000, 1, 4000],
        [1 + (i * 7919) % 50000 for i in range(40)] + [120000],
        [1 + (i * 7919) % 50000 for i in range(100)] + [120000],
    ],
    ids=["lane", "wide", "two_slot"],
)
def test_kv_replica_fused_partial_softmax_v2_is_bitwise(q_len: int, lens: list) -> None:
    """softmax_v2 (row maxima by tcgen05.ld.red.max) and the default softmax write the same bits, on every plan."""
    device = torch.device("cuda", 0)
    if not F.ld_red_max_supported(device):
        pytest.skip("softmax_v2 needs tcgen05.ld.red.max")
    lens = [max(L, q_len) for L in lens]
    b = len(lens)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    if not F.kv_replica_fused_supported(q_len, CP, HEADS, b, sms):
        pytest.skip("shape outside the fused KV-replica split partial kernel's limits")
    state = F.kv_replica_fused_state(device, sms, b)
    assert state.softmax_v2
    v1 = copy.copy(state)
    v1.softmax_v2 = False
    g = torch.Generator(device="cuda").manual_seed(b * 10 + q_len + 1)
    max_pages = max((max(lens) + PAGE - 1) // PAGE, CP)
    nblocks = b * 2 * max_pages + 8
    kv = (torch.randn(nblocks, 1, PAGE, HEAD_DIM, device="cuda", generator=g) * 0.6).to(
        torch.float8_e4m3fn
    )
    tables = torch.randperm(nblocks, device="cuda", generator=g)[: b * 2 * max_pages]
    tables = tables.to(torch.int32).view(b, 2, max_pages).contiguous()
    seq = torch.tensor(lens, dtype=torch.int32, device="cuda")
    q_all = (
        torch.randn(b * q_len, CP, HEADS, HEAD_DIM, device="cuda", generator=g)
        .to(torch.float8_e4m3fn)
        .permute(1, 0, 2, 3)
    )
    for rank in range(CP):
        outs = []
        for st in (state, v1):
            o = torch.full(
                (CP, b * q_len, HEADS, HEAD_DIM), float("nan"), dtype=torch.bfloat16, device="cuda"
            )
            lse = torch.full(
                (CP, b * q_len, HEADS), float("nan"), dtype=torch.float32, device="cuda"
            )
            F.kv_replica_fused_partial(
                q_all, kv, tables, seq, q_len, rank, CP, 1.0 / 16, 1.0, o, lse, st
            )
            outs.append((o, lse))
        torch.cuda.synchronize()
        (o2, lse2), (o1, lse1) = outs
        assert torch.equal(o2.view(torch.int16), o1.view(torch.int16)), rank
        assert torch.equal(lse2.view(torch.int32), lse1.view(torch.int32)), rank


def _check_kv_replica_fused_partial(q_len: int, lens: list) -> None:
    hq, d = HEADS, HEAD_DIM
    lens = [max(L, q_len) for L in lens]
    b = len(lens)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    if not F.kv_replica_fused_supported(q_len, CP, hq, b, sms):
        pytest.skip("shape outside the fused KV-replica split partial kernel's limits")
    state = F.kv_replica_fused_state(torch.device("cuda", 0), sms, b)
    g = torch.Generator(device="cuda").manual_seed(b * 10 + q_len)
    max_pages = max((max(lens) + PAGE - 1) // PAGE, CP)
    nblocks = b * 2 * max_pages + 8
    kv = (torch.randn(nblocks, 1, PAGE, d, device="cuda", generator=g) * 0.6).to(
        torch.float8_e4m3fn
    )
    tables = torch.randperm(nblocks, device="cuda", generator=g)[: b * 2 * max_pages]
    tables = tables.to(torch.int32).view(b, 2, max_pages).contiguous()
    seq = torch.tensor(lens, dtype=torch.int32, device="cuda")
    q_all = (
        torch.randn(b * q_len, CP, hq, d, device="cuda", generator=g)
        .to(torch.float8_e4m3fn)
        .permute(1, 0, 2, 3)
    )
    for rank in range(CP):
        o = torch.full((CP, b * q_len, hq, d), float("nan"), dtype=torch.bfloat16, device="cuda")
        lse = torch.full((CP, b * q_len, hq), float("nan"), dtype=torch.float32, device="cuda")
        F.kv_replica_fused_partial(
            q_all, kv, tables, seq, q_len, rank, CP, 1.0 / 16, 1.0, o, lse, state
        )
        ref_o, ref_lse = _kv_replica_fused_partial_ref(
            q_all, kv, tables, seq, q_len, rank, 1.0 / 16
        )
        torch.cuda.synchronize()
        # FP8 P / FP16 partials: max error ~5 % of the row max, mean ~1e-3
        of, rf = o.float(), ref_o.bfloat16().float()
        assert not torch.isnan(of).any() and not torch.isnan(lse).any()
        rel = ((of - rf).abs() / rf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-3)).amax()
        assert rel.item() < 0.07, (rank, rel.item())
        assert (of - rf).abs().mean().item() < 2e-3
        fin = torch.isfinite(ref_lse)
        assert torch.equal(torch.isfinite(lse), fin)
        assert torch.equal(torch.isneginf(lse), torch.isneginf(ref_lse))
        if fin.any():
            assert (lse[fin] - ref_lse[fin]).abs().amax().item() < 0.05


# KV group of two TP ranks with eight local query heads each: Qwen3.5-397B at TP4.
KV_REPLICAS, LOCAL_HEADS = 2, 8


@pytest.mark.parametrize(
    "max_batch,tokens_per_gen_step,softmax_bytes",
    [
        # batch 512: 1024 FMHA requests of 16 folded heads (NVBug 7046451)
        (512, 1, 34603008),
        # with three draft tokens: 512 requests of 64 folded heads
        (512, 4, 68157440),
    ],
)
def test_kv_replica_admission_respects_counter_capacity() -> None:
    """A real library admits eight-token calls at capacity, but not ten-token calls; ordinary decode admits both."""
    max_batch = 512
    setup = GqaDecodeSetup(
        kv_replicas=KV_REPLICAS,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        tokens_per_gen_step=10,
    )
    layer = _gqa_layer(LOCAL_HEADS, setup)
    library = next(lib for lib in layer._fmha_manager.fmha_libs if type(lib) is GqaKvReplicaFmha)
    ordinary = next(lib for lib in layer._fmha_manager.fmha_libs if type(lib) is GqaDecodeFmha)
    manager = KVCacheManagerV2(
        KvCacheConfig(max_tokens=max_batch * PAGE),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=1,
        num_kv_heads=1,
        head_dim=HEAD_DIM,
        tokens_per_block=PAGE,
        max_seq_len=PAGE,
        max_batch_size=max_batch,
        mapping=setup.mapping,
        dtype=tensorrt_llm.bindings.DataType.FP8,
    )
    request_ids = list(range(max_batch))
    try:
        manager.add_dummy_requests(request_ids, [11] * max_batch)
        for q_len, admitted in ((8, True), (10, False)):
            metadata = TrtllmAttentionMetadata(
                num_contexts=0,
                kv_cache_params=KVCacheParams(
                    use_cache=True, num_cached_tokens_per_seq=[1] * max_batch
                ),
                seq_lens=torch.full((max_batch,), q_len, dtype=torch.int32),
                max_num_requests=max_batch,
                max_num_tokens=max_batch * setup.tokens_per_gen_step,
                kv_cache_manager=manager,
                request_ids=request_ids,
                prompt_lens=[q_len + 1] * max_batch,
            )
            metadata.prepare()
            qkv = torch.empty(
                max_batch * q_len,
                (LOCAL_HEADS + 2) * HEAD_DIM,
                dtype=torch.bfloat16,
                device="cuda",
            )
            forward_args = AttentionForwardArgs(
                output=torch.empty(
                    max_batch * q_len,
                    LOCAL_HEADS * HEAD_DIM,
                    dtype=torch.float8_e4m3fn,
                    device="cuda",
                ),
                attention_input_type=AttentionInputType.generation_only,
                is_fused_qkv=True,
            )
            assert ordinary.is_supported(qkv, None, None, metadata, forward_args)
            assert library.is_supported(qkv, None, None, metadata, forward_args) is admitted
    finally:
        manager.shutdown()


def _gqa_layer(num_heads: int, setup: GqaDecodeSetup) -> TrtllmAttention:
    """An FP8-KV TRTLLM layer with one KV head and the GQA decode libraries."""
    return create_attention(
        "TRTLLM",
        layer_idx=0,
        num_heads=num_heads,
        head_dim=HEAD_DIM,
        num_kv_heads=1,
        quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8),
        gqa_decode=setup,
    )


@pytest.mark.parametrize(
    "num_heads,num_kv_heads,tokens_per_gen_step,splits",
    [
        (32, 2, 7, 8),  # Qwen3.5-397B under attention DP, MTP draft length 6
        (32, 2, 9, 0),  # 144 rows: more than one Q tile
        (8, 2, 1, 0),  # 8 splits x 2 KV heads exceed the counters of 8 heads
    ],
)
def test_min_kv_splits_need_one_q_tile_and_the_counters(
    num_heads: int, num_kv_heads: int, tokens_per_gen_step: int, splits: int
) -> None:
    assert GqaDecodeFmha._min_kv_splits_for(num_heads, num_kv_heads, tokens_per_gen_step) == splits


def test_min_kv_split_stops_above_the_largest_graph_batch() -> None:
    """Batches above the largest CUDA-graph decode batch keep the kernel selection's split (no planned SM count), so
    the forced split's multi-CTA scratch is bounded by that batch; so do batches the device's SMs already split."""
    kv_heads, sms = 2, 148
    fmha = SimpleNamespace(
        _min_kv_splits=8,
        _multi_processor_count=sms,
        _min_kv_split_batch=16,
        attn=SimpleNamespace(num_kv_heads=kv_heads),
    )
    batch = fmha._min_kv_split_batch
    assert GqaDecodeFmha._min_kv_split_sm_count(fmha, batch) == 8 * kv_heads * batch
    assert GqaDecodeFmha._min_kv_split_sm_count(fmha, batch + 1) == 0
    assert 8 * kv_heads * 8 <= sms
    assert GqaDecodeFmha._min_kv_split_sm_count(fmha, 8) == 0


def test_min_kv_split_batch_is_the_largest_captured_graph_batch() -> None:
    """ModelConfig.cuda_graph_batch_sizes (whose largest size bounds the forced split's scratch) holds the decode graph
    sizes the engine captures: none above max_batch_size or max_num_tokens // tokens per request (plan_cuda_graphs)."""
    from tensorrt_llm._torch.pyexecutor.engine.cuda_graph import plan_cuda_graphs
    from tensorrt_llm._torch.pyexecutor.model_loader import captured_cuda_graph_batch_sizes
    from tensorrt_llm.llmapi.llm_args import CudaGraphConfig

    spec = SimpleNamespace(tokens_per_gen_step=7)
    for graphs, max_batch_size, max_num_tokens, largest in (
        (CudaGraphConfig(), 32, 8192, 32),  # default sizes up to 128
        (CudaGraphConfig(max_batch_size=512), 64, 8192, 64),
        (CudaGraphConfig(max_batch_size=64), 128, 336, 32),  # 336 tokens: 48 requests of 7
        (CudaGraphConfig(batch_sizes=[1, 2, 40], enable_padding=True), 32, 8192, 32),
    ):
        args = SimpleNamespace(cuda_graph_config=graphs, max_batch_size=max_batch_size)
        sizes = captured_cuda_graph_batch_sizes(args, max_num_tokens, spec)
        plan = plan_cuda_graphs(
            graphs,
            None,
            max_batch_size=max_batch_size,
            max_num_tokens=max_num_tokens,
            max_seq_len=4096,
            decode_tokens_per_request=7,
        )
        assert list(sizes) == plan.batch_sizes and max(sizes) == largest, (graphs, sizes)
        assert max(graphs.batch_sizes) > largest
    no_graphs = SimpleNamespace(cuda_graph_config=None, max_batch_size=32)
    assert captured_cuda_graph_batch_sizes(no_graphs, 8192, spec) == ()


def test_rope_gather_grid_residency_comes_from_the_compiled_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fused q all-gather takes a grid only while all of it stays resident: at most min(occupancy of the compiled
    kernel, GATHER_MAX_PROGRAMS_PER_SM) programs per SM. At a TEP8 Qwen3.5 layer (4 q heads, 1 KV head, group of 4)
    the occupancy is above the cap, so the cap decides the token bound. Every member of the group compiles the same
    kernel, so every member takes the same bound."""
    from tensorrt_llm._torch.attention.kernels import fused_qk_norm_rope_gate as K

    hq, hkv, d, rot, q_len, group = 4, 1, HEAD_DIM, 64, 7, 4
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    occupancy = []
    kernels = []
    resident = K._resident_programs_per_sm

    def spy(kernel):
        kernels.append(kernel)
        occupancy.append(resident(kernel))
        return occupancy[-1]

    monkeypatch.setattr(K, "_resident_programs_per_sm", spy)

    def resident_tokens(num_tokens: int, grank: int = 0) -> int:
        b = num_tokens // q_len
        dev = "cuda"
        plan = KvAppendPlan(
            0,
            num_tokens,
            b,
            q_len,
            torch.zeros(8, hkv, PAGE, d, dtype=torch.float8_e4m3fn, device=dev),
            torch.zeros(b, 2, 4, dtype=torch.int32, device=dev),
            torch.full((b,), q_len, dtype=torch.int32, device=dev),
            torch.ones(1, dtype=torch.float32, device=dev),
            torch.ones(1, dtype=torch.float32, device=dev),
            1.0 / 16,
        )
        return K.rope_gather_max_tokens(
            torch.zeros(num_tokens, (2 * hq + 2 * hkv) * d, dtype=torch.bfloat16, device=dev),
            torch.zeros(d, dtype=torch.bfloat16, device=dev),
            torch.zeros(d, dtype=torch.bfloat16, device=dev),
            torch.zeros(4096, 2, rot // 2, dtype=torch.float32, device=dev),
            torch.zeros(num_tokens, dtype=torch.int32, device=dev),
            1e-6,
            hq,
            hkv,
            d,
            rot,
            None,
            plan,
            (
                torch.zeros(group, dtype=torch.int64, device=dev),
                torch.zeros(num_tokens * hq, dtype=torch.int32, device=dev),
                grank,
                group,
                1 << 16,
                True,
            ),
        )

    bound = K.GATHER_MAX_PROGRAMS_PER_SM * sms // (hq + 2 * hkv)
    assert resident_tokens(q_len) == bound
    assert resident_tokens(16 * q_len) == bound
    assert occupancy and min(occupancy) >= K.GATHER_MAX_PROGRAMS_PER_SM
    kernels.clear()
    assert [resident_tokens(q_len, grank) for grank in range(group)] == [bound] * group
    assert all(k is kernels[0] for k in kernels)


@pytest.mark.parametrize("folded", [False, True])
def test_rope_gather_launches_only_with_a_resident_grid(
    monkeypatch: pytest.MonkeyPatch, folded: bool
) -> None:
    """GqaKvReplicaFusedFmha's prologue (the module's unfolded one, and the folded call's fallback) all-gathers q in
    the RoPE kernel only for token counts within rope_gather_max_tokens, taken once per shape; otherwise, and
    without the plan's gather arguments, the plain kernel runs and plan.q_gathered stays None."""
    from tensorrt_llm._torch.attention.kernels import fused_qk_norm_rope_gate as K

    bound = 21
    calls = []

    def max_tokens(qkv, *args):
        calls.append(("bound", qkv.shape[0]))
        return bound

    def gather(qkv, *args):
        calls.append(("gather", qkv.shape[0]))
        return qkv, qkv, qkv, qkv, qkv

    def plain(qkv, *args):
        calls.append(("plain", qkv.shape[0]))
        return qkv, qkv, qkv, qkv

    monkeypatch.setattr(K, "rope_gather_max_tokens", max_tokens)
    monkeypatch.setattr(K, "fused_qkv_rope_gate_fp8_append_gather", gather)
    monkeypatch.setattr(K, "fused_qkv_rope_gate_fp8_append", plain)

    class _Attn:
        num_heads, num_kv_heads, head_dim = HEADS, 1, HEAD_DIM

    attn = _Attn()
    lib = GqaKvReplicaFusedFmha.__new__(GqaKvReplicaFusedFmha)
    lib._attn_ref = weakref.ref(attn)
    lib._rope_gather_resident = {}
    x = torch.zeros(1)

    def prologue(num_tokens: int, offered: bool) -> Optional[torch.Tensor]:
        qkv = torch.zeros(num_tokens, 8)
        plan = KvAppendPlan(0, num_tokens, num_tokens // 7, 7, x, x, x, x, x, 1.0)
        plan.rope_gather = (x, x, 0, CP, 1, True) if offered else None
        f = FoldInputs(qkv, x, x, x, x, 1e-6, 64, None)
        if folded:
            plan.fold = f
            params = SimpleNamespace(num_tokens=num_tokens, fwd=SimpleNamespace(output_gate=None))
            lib._run_fold_prologue(params, plan)
            assert params.qkv_input.shape[0] == num_tokens
        else:
            lib.run_prologue(plan, f.qkv, x, x, x, x, 1e-6, 64, None)
        return plan.q_gathered

    assert prologue(21, True) is not None
    assert prologue(28, True) is None
    assert prologue(21, True) is not None
    assert prologue(14, False) is None
    assert calls == [
        ("bound", 21),
        ("gather", 21),
        ("bound", 28),
        ("plain", 28),
        ("gather", 21),
        ("plain", 14),
    ]


@pytest.mark.parametrize("q_len, armed", [(1, True), (7, True), (8, True), (9, False), (16, False)])
def test_rope_gather_is_armed_only_where_the_fused_library_is_selected(
    monkeypatch: pytest.MonkeyPatch, q_len: int, armed: bool
) -> None:
    """kv_append_plan arms the RoPE kernel's q all-gather on the selection-cache cell bound that _is_supported uses:
    q_len 9..16 share the cell up to 31 > MAX_Q_LEN, which another library serves, so no gather runs there."""
    from tensorrt_llm._torch.attention.backends.fmha.manager import generation_cell_bound

    class _Attn:
        num_heads, num_kv_heads, head_dim = 4, 1, F.HEAD_DIM

    attn = _Attn()
    lib = GqaKvReplicaFusedFmha.__new__(GqaKvReplicaFusedFmha)
    lib._attn_ref = weakref.ref(attn)
    lib._kv_replicas = 2
    lib._multi_processor_count = 148
    x = torch.zeros(1)
    batch = 2
    plan = KvAppendPlan(0, batch * q_len, batch, q_len, x, x, x, x, x, 1.0)
    monkeypatch.setattr(GqaDecodeFmha, "kv_append_plan", lambda self, metadata, num_tokens: plan)
    monkeypatch.setattr(
        GqaKvReplicaFusedFmha, "_rope_gather", lambda self, attn, num_tokens: "gather"
    )
    metadata = SimpleNamespace(tokens_per_block=F.PAGE)
    got = lib.kv_append_plan(metadata, batch * q_len)
    assert (got.rope_gather == "gather") is armed
    # The library's selection (_is_supported) at this call's cell agrees.
    assert lib._takes(*generation_cell_bound(batch, q_len), metadata) is armed
