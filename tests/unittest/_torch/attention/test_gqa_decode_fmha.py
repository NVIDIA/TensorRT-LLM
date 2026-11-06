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
from typing import Optional

import backend_case
import pytest
import torch
from backend_case import BackendCase, run_case
from kv_cache_utils import fill_kv_cache_logical

import tensorrt_llm
from tensorrt_llm._torch.attention.backends.fmha import gqa_kv_replica_fused as F
from tensorrt_llm._torch.attention.backends.fmha.gqa_decode import (
    GqaDecodeFmha,
    GqaDecodeSetup,
    GqaSegmentedFmha,
    GqaSplitKvFmha,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
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
        # more: the module's prologue appends the FP8 K / V, trtllm-gen decodes
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
    appended_preprocess = GqaDecodeFmha._appended_preprocess

    def spy_folded(self, *args, **kwargs):
        taken["fold"].append(run_folded(self, *args, **kwargs))
        return taken["fold"][-1]

    def spy_appended(self, *args, **kwargs):
        result = appended_preprocess(self, *args, **kwargs)
        taken["append"].append(result is not None)
        return result

    monkeypatch.setattr(GqaSplitKvFmha, "_run_folded", spy_folded)
    monkeypatch.setattr(GqaDecodeFmha, "_appended_preprocess", spy_appended)

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
