# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The per-rank pool layout this target binds at tp4, tep4, dep4 and dep2, against the stock
manager.

`pool.pool_geometry` derives one rank's pool shapes from the text config and the rank's `Mapping`.
These tests hold it to `LAYOUT` (what `core` binds at each member), then build the stock
`Qwen4ExpHybridCacheManager` the way the engine does -- `get_kv_cache_manager_cls` and
`_create_kv_cache_manager` over a `ModelConfig`, with the core runtime's cache configuration
(bf16 recurrent state, block reuse off, KV cache manager V2, which the engine resolves `auto` to
for this model) -- on the first and the last rank of each member, one GPU, and take every pool
view through `pool.take_pools`, which checks each against the geometry. Ranks 0 and 3 hold
different KV and GDN heads at tp4 and tep4; the shapes are the same, the heads are not. dep2, the
layout of both disaggregated-serving engines, binds dep4's geometry on both of its ranks, so the
transceiver moves pools of one layout between identical layouts.
"""

import contextlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp_for_conditional_generation.hf_fc694b5_orig__sm_103__unified import (  # noqa: E501
    pool as P,
)
from tensorrt_llm._torch.configs import Qwen4ExpTextConfig
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import duplicate_kv_weight
from tensorrt_llm._torch.modules.qwen4_exp.cache_manager import Qwen4ExpHybridCacheManager
from tensorrt_llm._torch.pyexecutor._util import _create_kv_cache_manager, get_kv_cache_manager_cls
from tensorrt_llm._torch.pyexecutor.model_loader import validate_and_set_mamba_ssm_cache_dtype
from tensorrt_llm.bindings import DataType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MTPDecodingConfig, QSASparseAttentionConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

NUM_LAYERS = 48

# Qwen3.8-Flash-Next's text config: the fields the manager and the geometry read, and what the
# QSA lowering and the config class need beside them.
_TEXT = dict(
    architectures=["Qwen4ExpForCausalLM"],
    dtype="bfloat16",
    full_attention_interval=4,
    hc_count=4,
    hc_lowrank=320,
    head_dim=256,
    heads_per_ngram=8,
    hidden_size=2560,
    indexer_budget=2048,
    indexer_compress_ratio=4,
    indexer_head_dim=128,
    indexer_kv_heads=1,
    indexer_n_heads=4,
    layer_types=["full_attention" if i % 4 == 3 else "linear_attention" for i in range(NUM_LAYERS)],
    linear_conv_kernel_dim=4,
    linear_key_head_dim=128,
    linear_num_key_heads=16,
    linear_num_value_heads=48,
    linear_value_head_dim=128,
    make_ngram_vocab_size_divisible_by=128,
    mamba_ssm_dtype="float32",
    max_position_embeddings=262144,
    moe_intermediate_size=640,
    mtp=dict(
        hybrid=True,
        layer_types=["full_attention"],
        mtp_use_hidden_state_from_layer=None,
        num_hidden_layers=1,
        rope_theta=10000000,
    ),
    mtp_num_hidden_layers=1,
    mtp_use_dedicated_embeddings=False,
    ngram_size=3,
    ngram_vocab_size_base=20000000,
    num_attention_heads=24,
    num_experts=512,
    num_experts_per_tok=10,
    num_hidden_layers=NUM_LAYERS,
    num_key_value_heads=2,
    partial_rotary_factor=0.25,
    ple_conv_kernel_size=4,
    ple_embed_dim=2560,
    ple_layer_ids=[2],
    rms_norm_eps=1e-06,
    rope_parameters=dict(
        mrope_interleaved=True,
        mrope_section=[11, 11, 10],
        partial_rotary_factor=0.25,
        rope_theta=10000000,
        rope_type="default",
    ),
    shared_expert_intermediate_size=640,
    split_ngram_parts=128,
    tie_word_embeddings=False,
    vocab_size=248320,
)

ATTN_LAYERS = [i for i in range(NUM_LAYERS) if i % 4 == 3]
GDN_LAYERS = [i for i in range(NUM_LAYERS) if i % 4 != 3]
PLE_LAYER = 1  # `ple_layer_ids: [2]` is one-based
MTP_LAYER = NUM_LAYERS
TOKENS_PER_BLOCK = 32  # `kv_cache_config.tokens_per_block`'s default, which the runtime keeps

#: Each member's mapping: attention TP4 with the experts split by intermediate (tp4) or by expert
#: (tep4), and attention DP with expert parallelism (dep4). The expert split moves no pool.
MEMBERS = {
    "tp4": dict(world_size=4, tp_size=4, moe_tp_size=4, moe_ep_size=1, enable_attention_dp=False),
    "tep4": dict(world_size=4, tp_size=4, moe_tp_size=1, moe_ep_size=4, enable_attention_dp=False),
    "dep4": dict(world_size=4, tp_size=4, moe_tp_size=1, moe_ep_size=4, enable_attention_dp=True),
    "dep2": dict(world_size=2, tp_size=2, moe_tp_size=1, moe_ep_size=2, enable_attention_dp=True),
}

#: What `core` binds on each member and rank: KV heads and the first one's global index, GDN key
#: heads and the first, GDN value heads and the first, and the conv channels ([q | k | v]).
LAYOUT = {
    ("tp4", 0): dict(heads_kv=1, kv_head_start=0, num_k_heads=4, k_head_start=0, num_v_heads=12,
                     v_head_start=0, conv_dim=2560),
    ("tp4", 3): dict(heads_kv=1, kv_head_start=1, num_k_heads=4, k_head_start=12, num_v_heads=12,
                     v_head_start=36, conv_dim=2560),
    ("tep4", 0): dict(heads_kv=1, kv_head_start=0, num_k_heads=4, k_head_start=0, num_v_heads=12,
                      v_head_start=0, conv_dim=2560),
    ("tep4", 3): dict(heads_kv=1, kv_head_start=1, num_k_heads=4, k_head_start=12, num_v_heads=12,
                      v_head_start=36, conv_dim=2560),
    ("dep4", 0): dict(heads_kv=2, kv_head_start=0, num_k_heads=16, k_head_start=0, num_v_heads=48,
                      v_head_start=0, conv_dim=10240),
    ("dep4", 3): dict(heads_kv=2, kv_head_start=0, num_k_heads=16, k_head_start=0, num_v_heads=48,
                      v_head_start=0, conv_dim=10240),
    ("dep2", 0): dict(heads_kv=2, kv_head_start=0, num_k_heads=16, k_head_start=0, num_v_heads=48,
                      v_head_start=0, conv_dim=10240),
    ("dep2", 1): dict(heads_kv=2, kv_head_start=0, num_k_heads=16, k_head_start=0, num_v_heads=48,
                      v_head_start=0, conv_dim=10240),
}  # fmt: skip

#: The same on every member and rank: the page size, the head dims, the replicated QSA index and
#: PLE pools, and the activation dtype the conv, index and PLE pools hold.
COMMON = dict(
    tokens_per_block=TOKENS_PER_BLOCK,
    head_dim=256,
    index_kv_heads=1,
    index_head_dim=128,
    conv_kernel=4,
    head_k_dim=128,
    head_v_dim=128,
    stream=4 * 2560,
    ple_state_len=(4 - 1) * 3,
    ngram_context_len=3 - 1,
    dtype=torch.bfloat16,
)

#: The K/V pool's storage per resolved KV-cache algorithm: (pool dtype, innermost extent, the
#: manager's binding dtype).
KV_STORAGE = {
    None: (torch.bfloat16, 256, DataType.BF16),
    QuantAlgo.FP8: (torch.float8_e4m3fn, 256, DataType.FP8),
    QuantAlgo.NVFP4: (torch.uint8, 128, DataType.NVFP4),
}
KV_IDS = {None: "kv_bf16", QuantAlgo.FP8: "kv_fp8", QuantAlgo.NVFP4: "kv_nvfp4"}

_needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the manager allocates its pools on the GPU"
)


def _mapping(member: str, rank: int) -> Mapping:
    return Mapping(rank=rank, gpus_per_node=4, pp_size=1, **MEMBERS[member])


def _geometry(member: str, rank: int, kv_cache_quant_algo=None) -> P.PoolGeometry:
    return P.pool_geometry(
        Qwen4ExpTextConfig(**_TEXT),
        _mapping(member, rank),
        tokens_per_block=TOKENS_PER_BLOCK,
        kv_cache_quant_algo=kv_cache_quant_algo,
        dtype=torch.bfloat16,
    )


# The first and the last rank of each member.
_MEMBER_RANKS = [
    (member, rank) for member, m in MEMBERS.items() for rank in (0, m["world_size"] - 1)
]
_MEMBER_RANK_IDS = [f"{member}-rank{rank}" for member, rank in _MEMBER_RANKS]
# dep2 serves without speculation (routing declines a `speculative_config` there).
_MTP_MEMBER_RANKS = [(member, rank) for member, rank in _MEMBER_RANKS if member != "dep2"]
_MTP_MEMBER_RANK_IDS = [f"{member}-rank{rank}" for member, rank in _MTP_MEMBER_RANKS]


@pytest.mark.parametrize("member, rank", _MEMBER_RANKS, ids=_MEMBER_RANK_IDS)
def test_the_geometry_is_the_layout_core_binds(member, rank):
    g = _geometry(member, rank)
    expected = dict(COMMON, **LAYOUT[member, rank], kv_dtype=torch.bfloat16, kv_width=256)
    assert g._asdict() == expected
    assert g.conv_sections == (
        g.num_k_heads * 128,
        g.num_k_heads * 128,
        g.num_v_heads * 128,
    )


@pytest.mark.parametrize("algo", list(KV_STORAGE), ids=list(KV_IDS.values()))
def test_the_kv_storage_follows_the_resolved_algorithm(algo):
    g = _geometry("dep4", 0, algo)
    assert (g.kv_dtype, g.kv_width) == KV_STORAGE[algo][:2]
    assert g.kv_block_scales == (algo == QuantAlgo.NVFP4)


def test_another_kv_algorithm_is_refused():
    with pytest.raises(ValueError, match="kv_cache_quant_algo"):
        _geometry("dep4", 0, QuantAlgo.INT8)


@pytest.mark.parametrize("member, rank", _MEMBER_RANKS, ids=_MEMBER_RANK_IDS)
def test_a_rank_holds_the_kv_head_the_weight_split_gives_it(member, rank):
    """With more attention-TP ranks than KV heads the stock loader repeats each head
    (`duplicate_kv_weight`) before the even row split, so the K/V a rank writes into its pages
    is that of global head `kv_head_start`; under attention DP a rank holds all of them."""
    g = _geometry(member, rank)
    heads, rows_per_head = 2, 4
    weight = torch.arange(heads * rows_per_head, dtype=torch.float32).reshape(-1, 1)
    tp = P.attention_tp_size(_mapping(member, rank))
    local = duplicate_kv_weight(weight, heads, tp).chunk(tp)[rank if tp > 1 else 0]
    start, stop = g.kv_head_start * rows_per_head, (g.kv_head_start + g.heads_kv) * rows_per_head
    assert torch.equal(local, weight[start:stop])


@contextlib.contextmanager
def _manager(member: str, rank: int, kv_cache_quant_algo=None, spec_config=None):
    """The trunk manager the engine builds on `rank` of `member`, through the engine's own class
    choice and factory."""
    mapping = _mapping(member, rank)
    model_config = ModelConfig(
        pretrained_config=Qwen4ExpTextConfig(**_TEXT),
        mapping=mapping,
        quant_config=QuantConfig(
            quant_algo=QuantAlgo.MIXED_PRECISION, kv_cache_quant_algo=kv_cache_quant_algo
        ),
        sparse_attention_config=QSASparseAttentionConfig(),
        spec_config=spec_config,
    )
    # What the loader does with `kv_cache_config.mamba_ssm_cache_dtype: bfloat16`.
    validate_and_set_mamba_ssm_cache_dtype(model_config, "bfloat16")
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=False,
        mamba_ssm_cache_dtype="bfloat16",
        max_tokens=4096,
        use_kv_cache_manager_v2=True,
    )
    cls = get_kv_cache_manager_cls(model_config, kv_cache_config)
    cm = _create_kv_cache_manager(
        None,
        cls,
        mapping,
        kv_cache_config,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=2048,
        max_batch_size=4,
        spec_config=spec_config,
        sparse_attention_config=model_config.sparse_attention_config,
        max_num_tokens=2048,
        max_beam_width=1,
        kv_connector_manager=None,
        model_config=model_config,
        dtype=torch.bfloat16,
        max_cuda_graph_batch_size=4,
    )
    try:
        yield cm
    finally:
        cm.shutdown()


@_needs_gpu
@pytest.mark.parametrize("algo", list(KV_STORAGE), ids=list(KV_IDS.values()))
@pytest.mark.parametrize("member, rank", _MEMBER_RANKS, ids=_MEMBER_RANK_IDS)
def test_the_stock_manager_serves_the_layout(member, rank, algo):
    """Every trunk pool view of the stock manager passes the contract at this rank's geometry,
    and the manager's own per-rank numbers are the geometry's."""
    g = _geometry(member, rank, algo)
    with _manager(member, rank, algo) as cm:
        assert type(cm) is Qwen4ExpHybridCacheManager
        assert cm.dtype == KV_STORAGE[algo][2]
        assert cm.ssm_state_dtype == torch.bfloat16
        pools = P.take_pools(cm, g, ATTN_LAYERS, GDN_LAYERS, PLE_LAYER)

        assert {cm.num_kv_heads_per_layer[cm.layer_offsets[i]] for i in ATTN_LAYERS} == {g.heads_kv}
        assert (cm.qsa_index_kv_heads, cm.qsa_index_dim) == (g.index_kv_heads, g.index_head_dim)
        assert list(cm.conv_state_shape) == [g.conv_dim, g.conv_kernel - 1]
        assert list(cm.ssm_state_shape) == [g.num_v_heads, g.head_v_dim, g.head_k_dim]
        # Each attention layer has a pool offset of its own.
        assert len(set(pools.pool_idx.values())) == len(ATTN_LAYERS)
        # One slot space for every QSA view of the layers, and one for the recurrent and PLE
        # state.
        slots = {view.shape[0] for view in (pools.position_cache, *pools.index_cache.values())}
        slots |= {planes.k.shape[0] for planes in pools.kv.values()}
        assert len(slots) == 1
        assert {t.shape[0] for t in (*pools.conv.values(), *pools.ssm.values())} == {
            pools.ple_state[0].shape[0]
        }


@_needs_gpu
@pytest.mark.parametrize("member, rank", _MTP_MEMBER_RANKS, ids=_MTP_MEMBER_RANK_IDS)
def test_the_mtp_verify_buffers_hold_this_ranks_heads(monkeypatch, member, rank):
    """With MTP (three drafts, the GDN cached replay) the verify buffers -- the per-step conv
    windows and every GDN layer's replay history -- are this rank's heads. Under attention TP the
    draft layer has a separate draft KV cache; under attention DP it is in the trunk's manager,
    with the trunk's per-rank pool shapes."""
    monkeypatch.delenv("TRTLLM_USE_GDN_REPLAY", raising=False)
    g = _geometry(member, rank)
    spec = MTPDecodingConfig(max_draft_len=3, num_nextn_predict_layers=1)
    with _manager(member, rank, spec_config=spec) as cm:
        pools = P.take_pools(cm, g, ATTN_LAYERS, GDN_LAYERS, PLE_LAYER)
        assert cm.is_speculative()
        ssm0 = pools.ssm[GDN_LAYERS[0]]
        caches = {i: cm.mamba_layer_cache(i) for i in GDN_LAYERS}
        for i, cache in caches.items():
            P.check_step_buffers(
                g, cache.intermediate_conv_window, pools.conv[i], cm.intermediate_state_indices
            )
            assert cache.intermediate_ssm is None, "the cached replay keeps no per-step state"
        meta = cm.get_replay_state_update_metadata()
        assert meta is not None
        P.check_replay_state(
            meta.replay_history_size,
            meta.replay_step_width,
            meta.prev_num_accepted_tokens,
            meta.cache_buf_idx,
            ssm0,
        )
        assert meta.replay_step_width == spec.max_draft_len + 1
        history, slots = meta.replay_history_size, ssm0.shape[0]
        for i, cache in caches.items():
            P.check_replay_history(g, i, history, slots, cache.old_x, cache.old_B, cache.old_dt)

        draft_in_trunk = MTP_LAYER in cm.layer_offsets
        assert draft_in_trunk == MEMBERS[member]["enable_attention_dp"]
        if draft_in_trunk:
            P.kv_planes(cm, g, MTP_LAYER, cm.get_index_k_buffer(MTP_LAYER))
