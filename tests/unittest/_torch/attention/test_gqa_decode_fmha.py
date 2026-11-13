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

import functools
from types import SimpleNamespace
from typing import Optional

import backend_case
import pytest
import torch
from backend_case import BackendCase, run_case
from kv_cache_utils import fill_kv_cache_logical

import tensorrt_llm
from tensorrt_llm._torch.attention.backends.fmha import gqa_decode
from tensorrt_llm._torch.attention.backends.fmha import gqa_kv_replica_fused as F
from tensorrt_llm._torch.attention.backends.fmha.flashinfer_trtllm_gen import (
    FlashInferTrtllmGenFmha,
    _get_generation_workspace_layout,
)
from tensorrt_llm._torch.attention.backends.fmha.gqa_decode import (
    GqaDecodeFmha,
    GqaDecodeSetup,
    GqaKvReplicaFmha,
    GqaSegmentedFmha,
    GqaSplitKvFmha,
)
from tensorrt_llm._torch.attention.backends.fmha.gqa_kv_replica import kv_replica_partial_merge_free
from tensorrt_llm._torch.attention.backends.fmha.phased import FmhaParams
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
BMM1_SCALE = HEAD_DIM**-0.5
# FP8 P and FP16 partials against the FP32 reference: largest element error relative to the row's largest value,
# mean absolute error, base-2 LSE error.
PARTIAL_MAX_REL_ERR, PARTIAL_MEAN_ABS_ERR, LSE_ABS_ERR = 0.07, 2e-3, 0.05
MIB = 1 << 20


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
    "cached,q_len,fmha_libs,path,segmented",
    [
        # up to four requests without a KV-replica group: the split-KV decode kernel runs the prologue (fold)
        ([5000], 7, None, "fold", False),
        ([300, 7000, 65], 1, None, "fold", False),
        # ~40K-token contexts: more than two tiles per CTA, so the K / V ring phases wrap
        ([40000, 38500], 7, None, "fold", False),
        ([41000, 300, 40000, 39000], 7, None, "fold", False),
        # a deferred prologue reaching another GQA decode library: it runs the prologue kernel itself, then the
        # segmented decode with the output gate (seven tokens per step on a layer without a speculative config)
        ([5000], 7, "-gqa_split_kv", "append", True),
        # more: the module's prologue appends the FP8 K / V, trtllm-gen decodes
        ([900, 3000, 4100, 70, 8000], 7, None, "append", False),
    ],
)
def test_qwen3_next_generation_step_matches_stock(
    monkeypatch: pytest.MonkeyPatch,
    cached: list,
    q_len: int,
    fmha_libs: Optional[str],
    path: str,
    segmented: bool,
) -> None:
    if fmha_libs is not None:
        monkeypatch.setenv("TLLM_FMHA_LIBS", fmha_libs)
    taken = {"fold": [], "append": [], "segmented": []}
    run_folded = GqaSplitKvFmha._run_folded
    appended_preprocess = GqaDecodeFmha._appended_preprocess
    run_segmented = GqaSegmentedFmha._run_dedicated

    def spy_segmented(self, *args, **kwargs):
        taken["segmented"].append(run_segmented(self, *args, **kwargs))
        return taken["segmented"][-1]

    def spy_folded(self, *args, **kwargs):
        taken["fold"].append(run_folded(self, *args, **kwargs))
        return taken["fold"][-1]

    def spy_appended(self, *args, **kwargs):
        result = appended_preprocess(self, *args, **kwargs)
        taken["append"].append(result is not None)
        return result

    monkeypatch.setattr(GqaSplitKvFmha, "_run_folded", spy_folded)
    monkeypatch.setattr(GqaDecodeFmha, "_appended_preprocess", spy_appended)
    monkeypatch.setattr(GqaSegmentedFmha, "_run_dedicated", spy_segmented)

    reference, reference_cache = _generation_step(False, cached, q_len)
    assert taken == {"fold": [], "append": [], "segmented": []}
    output, cache = _generation_step(True, cached, q_len)
    assert taken[path] == [True]
    assert taken["segmented"] == ([True] if segmented else [])
    # The FP8 K / V of the new tokens are the bytes the C++ generation preprocess writes.
    assert torch.equal(cache.view(torch.uint8), reference_cache.view(torch.uint8))
    # FP8 P in the split-KV kernel vs trtllm-gen: a few percent of the row max per element
    assert ((output.float() - reference.float()).norm() / reference.float().norm()).item() < 0.05


def _kv_replica_partial_ref(
    q_all: torch.Tensor,
    kv_pool: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    q_len: int,
    cp_rank: int,
    page: int,
    with_draft_block: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 attention of every head of the KV group over member cp_rank's pages ((j // page) % cp == cp_rank).

    Args:
        q_all: [cp, b * q_len, hq, D] FP8 queries of the group's members.
        kv_pool: [blocks, 1, page, D] FP8; block_tables: [b, 2, max_pages]; seq_lens: [b] KV lengths with the new
            tokens.
        with_draft_block: token t sees the keys before position L - q_len + t + 1 (the fused partial kernel); else
            only the prefix [0, L - q_len) (the segmented partials, whose draft block is a separate partial).

    Returns:
        o [cp, b * q_len, hq, D], normalized by this member's sum (0 where it sees no key), and its base-2 lse
        [cp, b * q_len, hq] (-inf where it sees no key).
    """
    cp, rows, hq, d = q_all.shape
    b = rows // q_len
    dev = q_all.device
    own = torch.arange(cp_rank, block_tables.shape[2], cp, device=dev)
    k = kv_pool[block_tables[:b, 0].index_select(1, own).long(), 0].float()  # [b, n_own, page, D]
    v = kv_pool[block_tables[:b, 1].index_select(1, own).long(), 0].float()
    pos = own[:, None] * page + torch.arange(page, device=dev)[None, :]
    # queries [b, q_len, G, D], group head g = s * hq + h
    q = q_all.float().reshape(cp, b, q_len, hq, d).permute(1, 2, 0, 3, 4).reshape(b, q_len, -1, d)
    s = torch.einsum("btgd,bnpd->btgnp", q, k) * BMM1_SCALE
    limit = (seq_lens[:b].long() - q_len)[:, None].expand(b, q_len)
    if with_draft_block:
        limit = limit + torch.arange(q_len, device=dev)[None, :] + 1
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


def _assert_partial_matches(o: torch.Tensor, ref_o: torch.Tensor) -> None:
    rf = ref_o.bfloat16().float()
    rel = ((o - rf).abs() / rf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-3)).amax()
    assert rel.item() < PARTIAL_MAX_REL_ERR, rel.item()
    assert (o - rf).abs().mean().item() < PARTIAL_MEAN_ABS_ERR


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
            q_all, kv, tables, seq, q_len, rank, CP, BMM1_SCALE, 1.0, o, lse, state
        )
        ref_o, ref_lse = _kv_replica_partial_ref(q_all, kv, tables, seq, q_len, rank, PAGE, True)
        torch.cuda.synchronize()
        assert not torch.isnan(o).any() and not torch.isnan(lse).any()
        _assert_partial_matches(o.float(), ref_o)
        fin = torch.isfinite(ref_lse)
        assert torch.equal(torch.isfinite(lse), fin)
        assert torch.equal(torch.isneginf(lse), torch.isneginf(ref_lse))
        if fin.any():
            assert (lse[fin] - ref_lse[fin]).abs().amax().item() < LSE_ABS_ERR


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
def test_kv_replica_segment_attention_takes_softmax_stats(
    max_batch: int, tokens_per_gen_step: int, softmax_bytes: int
) -> None:
    """The segment kernel needs its softmax statistics in addition to the ordinary FMHA workspace."""
    page = 32
    max_kv_len = 1 << 18  # the 256K-token window of the failing configuration
    group_heads = KV_REPLICAS * LOCAL_HEADS
    # The largest call: max_batch requests of tokens_per_gen_step tokens.
    b, q_len = max_batch, tokens_per_gen_step
    segments = GqaKvReplicaFmha._num_segments(b, q_len)
    num_segment_requests = b * segments
    num_folded_heads = q_len * group_heads
    dev = torch.device("cuda", 0)
    g = torch.Generator(device=dev).manual_seed(b + q_len)
    lens = torch.randint(q_len + 1, q_len + 300, (b,), device=dev, generator=g).to(torch.int32)
    max_pages = (int(lens.max()) + page - 1) // page
    nblocks = b * 2 * max_pages + 8
    kv = (torch.randn(nblocks, 1, page, HEAD_DIM, device=dev, generator=g) * 0.6).to(
        torch.float8_e4m3fn
    )
    tables = torch.randperm(nblocks, device=dev, generator=g)[: b * 2 * max_pages]
    tables = tables.to(torch.int32).view(b, 2, max_pages).contiguous()
    q_all = (
        torch.randn(b * q_len, KV_REPLICAS, LOCAL_HEADS, HEAD_DIM, device=dev, generator=g)
        .to(torch.float8_e4m3fn)
        .permute(1, 0, 2, 3)
    )
    counter = torch.zeros(
        num_segment_requests * num_folded_heads + 1024, dtype=torch.int32, device=dev
    )

    def run(workspace: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return kv_replica_partial_merge_free(
            q_all,
            kv,
            tables,
            lens,
            max_kv_len,
            q_len,
            KV_REPLICAS,
            0,
            segments,
            BMM1_SCALE,
            1.0,
            workspace,
            counter,
            False,
        )

    fmha_bytes = int(
        _get_generation_workspace_layout(
            torch.bfloat16, b, b * q_len, LOCAL_HEADS, HEAD_DIM, 1, 64
        )["trtllm_gen_workspace_size"]
    )
    # FlashInfer's message carries the size it takes for the stats: the formula of segment_attention_softmax_bytes.
    with pytest.raises(
        RuntimeError, match=rf"trtllm_gen_softmax_workspace with size {softmax_bytes} "
    ):
        run(torch.empty(fmha_bytes, dtype=torch.uint8, device=dev))
    o, lse, _, _, valid = run(
        torch.empty(fmha_bytes + softmax_bytes, dtype=torch.uint8, device=dev)
    )
    torch.cuda.synchronize()

    # LSE merge of member 0's segments vs the reference over its pages of the prefix
    seg_valid = valid[0].bool()[None, :, :, None]  # [1, S, rows, 1]
    seg_lse = torch.where(seg_valid, lse, float("-inf"))
    m = seg_lse.amax(dim=1)
    w = torch.exp2(seg_lse - m[:, None])
    merged_o = (w[..., None] * o.float()).sum(dim=1) / w.sum(dim=1)[..., None]
    merged_lse = m + torch.log2(w.sum(dim=1))
    ref_o, ref_lse = _kv_replica_partial_ref(q_all, kv, tables, lens, q_len, 0, page, False)
    _assert_partial_matches(merged_o, ref_o)
    assert (merged_lse - ref_lse).abs().amax().item() < LSE_ABS_ERR


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


def test_segment_stats_bytes_follow_the_admitted_calls() -> None:
    """The workspace extension of a layer: the KV-replica calls _takes admits (capped by the counter buffer), and
    the segmented calls of any q_len _is_supported admits."""
    setup = functools.partial(GqaDecodeSetup, kv_replicas=KV_REPLICAS)
    tp4_layer = SimpleNamespace(
        num_heads=LOCAL_HEADS, num_kv_heads=1, gqa_decode=setup(tokens_per_gen_step=10)
    )
    # ten tokens per step: 512 x 10 x 16 pairs exceed the 512 x 128 counters, so _takes admits no such call
    assert GqaKvReplicaFmha._largest_segment_stats_bytes(tp4_layer, 512) == 2048 * 512 * 128 + MIB
    assert not _takes_kv_replica(tp4_layer, 512, 10) and _takes_kv_replica(tp4_layer, 512, 8)
    # two requests x 16 segments x (q_len 8 x 8 heads = 64 rows on the KV head)
    assert GqaSegmentedFmha._largest_segment_stats_bytes(tp4_layer, 512) == 2048 * 32 * 64 + MIB
    # one rank with both KV heads: q_len 4 x 16 heads per KV head
    tp1_layer = SimpleNamespace(num_heads=32, num_kv_heads=2, gqa_decode=GqaDecodeSetup())
    assert GqaSegmentedFmha._largest_segment_stats_bytes(tp1_layer, 512) == 2048 * 32 * 128 + MIB
    assert GqaSegmentedFmha._largest_segment_stats_bytes(tp1_layer, 1) == 2048 * 16 * 128 + MIB


def _takes_kv_replica(layer: SimpleNamespace, batch: int, q_len: int) -> bool:
    library = SimpleNamespace(
        attn=layer,
        _kv_replicas=layer.gqa_decode.kv_replicas,
        _largest_segment_rows=GqaKvReplicaFmha._largest_segment_rows,
        _counter_slots=GqaKvReplicaFmha._counter_slots,
    )
    return GqaKvReplicaFmha._takes(library, batch, q_len, SimpleNamespace(max_num_requests=512))


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


def _generation_call(
    layer: TrtllmAttention, max_num_requests: int, num_requests: int, q_len: int
) -> SimpleNamespace:
    """Inputs of prepare_workspace and _run_dedicated for a generation-only call on an empty attention workspace."""
    dev = torch.device("cuda", 0)
    num_tokens = num_requests * q_len
    output = torch.empty(num_tokens, layer.num_heads * HEAD_DIM, dtype=torch.bfloat16, device=dev)
    fwd = AttentionForwardArgs(
        output=output, attention_input_type=AttentionInputType.generation_only
    )
    metadata = SimpleNamespace(
        max_num_requests=max_num_requests,
        max_num_sequences=max_num_requests,
        max_context_length=1,
        num_ctx_tokens=0,
        num_contexts=0,
        is_cuda_graph=False,
    )
    workspace = torch.empty(0, dtype=torch.int8, device=dev)
    return SimpleNamespace(
        qkv=torch.empty(num_tokens, 1, dtype=torch.bfloat16, device=dev),
        q=torch.empty(num_tokens, layer.num_heads, HEAD_DIM, dtype=torch.float8_e4m3fn, device=dev),
        metadata=metadata,
        fwd=fwd,
        workspace=workspace,
        params=FmhaParams(
            attn=layer,
            workspace=workspace,
            input_seq_length=q_len,
            num_tokens=num_tokens,
            num_requests=num_requests,
            fwd=fwd,
            meta=metadata,
            sequence_lengths=None,
            output=output,
        ),
    )


def _fmha_slice(
    layer: TrtllmAttention, call: SimpleNamespace, workspace: torch.Tensor
) -> torch.Tensor:
    """The trtllm-gen FMHA workspace of the call: the last region of its generation layout."""
    params = call.params
    layout = _get_generation_workspace_layout(
        torch.bfloat16,
        params.num_requests,
        params.num_tokens,
        layer.num_heads,
        HEAD_DIM,
        layer.num_kv_heads,
        layer.rope_dim,
    )
    offset, size = layout["trtllm_gen_workspace_offset"], layout["trtllm_gen_workspace_size"]
    assert offset + size == layout["total_size"]
    return workspace.view(torch.uint8).narrow(0, int(offset), int(size))


@pytest.mark.parametrize(
    "tokens_per_gen_step,q_len,stats_bytes",
    [(1, 1, 34603008), (4, 4, 68157440), (10, 8, 135266304)],
)
def test_kv_replica_library_runs_on_the_extended_fmha_workspace(
    monkeypatch: pytest.MonkeyPatch, tokens_per_gen_step: int, q_len: int, stats_bytes: int
) -> None:
    """The layer's first generation call sizes the attention workspace for its largest segment attention call, and
    GqaKvReplicaFmha passes the FMHA slice extended by the call's softmax stats; a workspace that ends at the
    trtllm-gen layout sends the call to trtllm-gen."""
    max_batch = 512
    setup = GqaDecodeSetup(
        kv_replicas=KV_REPLICAS,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        tokens_per_gen_step=tokens_per_gen_step,
    )
    layer = _gqa_layer(LOCAL_HEADS, setup)
    monkeypatch.setattr(
        "tensorrt_llm._torch.distributed.ll_exchange.get_tp_group_exchange",
        lambda *args: SimpleNamespace(max_words=1 << 40),
    )
    workspaces = []
    monkeypatch.setattr(
        gqa_decode, "kv_replica_decode", lambda *args, **kwargs: workspaces.append(args[10])
    )

    segmented = GqaSegmentedFmha(layer)
    library = GqaKvReplicaFmha(layer)

    # The attention JIT warmup: one request, taken by the segmented library, sizes for the KV-replica calls too.
    warmup = _generation_call(layer, max_batch, 1, q_len)
    workspace = warmup.workspace
    segmented.prepare_workspace(warmup.qkv, None, None, warmup.metadata, warmup.fwd, workspace)
    sized = (workspace.data_ptr(), workspace.numel())

    call = _generation_call(layer, max_batch, max_batch, q_len)
    call.params.workspace = workspace
    library.prepare_workspace(call.qkv, None, None, call.metadata, call.fwd, workspace)
    # nothing left to grow at the first full batch
    assert (workspace.data_ptr(), workspace.numel()) == sized
    fmha_workspace = _fmha_slice(layer, call, workspace)
    assert library._run_dedicated(call.params, call.q, None, None, 0, 1.0, 1.0, fmha_workspace)
    assert stats_bytes > fmha_workspace.numel() > 0
    (taken,) = workspaces
    assert taken.data_ptr() == fmha_workspace.data_ptr()
    assert taken.numel() == fmha_workspace.numel() + stats_bytes
    assert taken.data_ptr() + taken.numel() <= workspace.data_ptr() + workspace.numel()

    layout_only = torch.empty(
        fmha_workspace.data_ptr() - workspace.data_ptr() + fmha_workspace.numel(),
        dtype=torch.int8,
        device=workspace.device,
    )
    call.params.workspace = layout_only
    assert not library._run_dedicated(
        call.params, call.q, None, None, 0, 1.0, 1.0, _fmha_slice(layer, call, layout_only)
    )
    assert len(workspaces) == 1


def test_segmented_library_takes_any_supported_q_len(monkeypatch: pytest.MonkeyPatch) -> None:
    """GqaSegmentedFmha's workspace extension follows its support limits, not the layer's tokens per step: a
    seven-token call of a layer without a speculative config runs on the extended FMHA workspace."""
    layer = _gqa_layer(HEADS, GqaDecodeSetup())
    workspaces = []
    monkeypatch.setattr(
        gqa_decode, "segmented_decode", lambda *args, **kwargs: workspaces.append(args[10])
    )
    library = GqaSegmentedFmha(layer)
    q_len = 7
    call = _generation_call(layer, 1, 1, q_len)
    library.prepare_workspace(call.qkv, None, None, call.metadata, call.fwd, call.workspace)
    fmha_workspace = _fmha_slice(layer, call, call.workspace)
    assert library._run_dedicated(call.params, call.q, None, None, 0, 1.0, 1.0, fmha_workspace)
    (taken,) = workspaces
    assert taken.data_ptr() == fmha_workspace.data_ptr()
    # 16 segments of q_len x HEADS folded heads
    assert taken.numel() == fmha_workspace.numel() + 2048 * 16 * q_len * HEADS + MIB


def test_disabled_segment_libraries_do_not_extend_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disabling both LSE-using libraries leaves the ordinary decode workspace unchanged."""
    monkeypatch.setenv("TLLM_FMHA_LIBS", "gqa_decode,fallback")
    layer = _gqa_layer(LOCAL_HEADS, GqaDecodeSetup(kv_replicas=KV_REPLICAS, tokens_per_gen_step=1))
    library = next(lib for lib in layer._fmha_manager.fmha_libs if type(lib) is GqaDecodeFmha)
    call = _generation_call(layer, 512, 1, 1)
    reference = torch.empty_like(call.workspace)
    FlashInferTrtllmGenFmha(layer).prepare_workspace(
        call.qkv, None, None, call.metadata, call.fwd, reference
    )
    library.prepare_workspace(call.qkv, None, None, call.metadata, call.fwd, call.workspace)
    assert call.workspace.numel() == reference.numel()
