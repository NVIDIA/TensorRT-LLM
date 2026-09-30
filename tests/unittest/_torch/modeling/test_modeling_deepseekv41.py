# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 configuration, loading, replay and quantization tests.

Checkpoint-dependent tests use LLM_MODELS_ROOT; GPU and MPI requirements are
marked on individual tests. Loader coverage uses real checkpoint headers with
meta tensors; constructor checks use a small topology without loading weights.
"""

import dataclasses
import inspect
import json
import os
import struct
import subprocess
import sys
import textwrap
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from _torch.attention.sparse.csa2._utils import _FakeMetadata
from torch import nn
from transformers import AutoConfig
from utils.util import skip_blackwell_geforce

from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
from tensorrt_llm._torch.configs import deepseek_v41 as v41_config
from tensorrt_llm._torch.configs.deepseek_v41 import (
    DeepseekV41Config,
    DeepseekV41QuantLayout,
    DeepseekV41QuantRole,
    DeepseekV41TextConfig,
    DeepseekV41VisionConfig,
    assert_weight_layout,
    build_layer_descriptors,
    derive_block,
    layout_for_role,
    parse_quantization_layout,
    quant_role_for_weight_key,
)
from tensorrt_llm._torch.distributed import AllReduceParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models import modeling_deepseekv4 as v4
from tensorrt_llm._torch.models import modeling_deepseekv41 as v41
from tensorrt_llm._torch.models.modeling_deepseekv4 import (
    DeepseekV4Attention,
    DeepseekV4Gate,
    weight_dequant,
)
from tensorrt_llm._torch.models.modeling_deepseekv41 import (
    DeepseekV41Attention,
    DeepseekV41ForCausalLM,
    DeepseekV41Model,
)
from tensorrt_llm._torch.modules.mhc.hyper_connection import HCState
from tensorrt_llm.llmapi.llm_args import CSA2SparseAttentionConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo

__extra_import_path__ = ["~/tests/integration"]
from defs.deepseek_v41_serving import model_kwargs as serving_model_kwargs

# Configuration and checkpoint topology

# The released DeepSeek-V4.1-Flash topology: 40 decoder layers + 3 DSpark stages.
# [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0]
RELEASE_RATIOS = [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0]
RELEASE_KV_SOURCES = [2, 8, 14, 20]
RELEASE_INDEX_SOURCES = [2, 8, 14, 20, 24, 28, 32, 36]
RELEASE_ENGRAM_LAYERS = [1, 14]


def release_text_config(**overrides) -> DeepseekV41TextConfig:
    kwargs = dict(
        compress_ratios=RELEASE_RATIOS,
        kv_source_layer_ids=RELEASE_KV_SOURCES,
        index_source_layer_ids=RELEASE_INDEX_SOURCES,
        engram_layer_ids=RELEASE_ENGRAM_LAYERS,
        engram_num_embeddings=[384006168, 384016682],
    )
    kwargs.update(overrides)
    return DeepseekV41TextConfig(**kwargs)


# ---------------------------------------------------------------------------
# Per-layer descriptor
# ---------------------------------------------------------------------------


def test_all_43_ratio_entries_classified():
    """Classify every released decoder and DSpark layer, including unpooled ratio 1."""
    descriptors = release_text_config().layer_descriptors
    assert len(descriptors) == 43

    for d in descriptors:
        if d.layer_idx in (0, 1) or d.layer_idx >= 40:
            # Pure sliding window: no long-range path, no YaRN, base theta.
            assert d.compress_ratio == 0, d
            assert d.has_long_range is False, d
            assert d.pools_kv is False, d
            assert d.pool_factor == 1, d
            assert d.rope_theta == 10000, d
            assert d.yarn_enabled is False, d
        elif 2 <= d.layer_idx <= 19:
            # Indexed and pooled 2:1.
            assert d.compress_ratio == 2, d
            assert d.has_long_range is True, d
            assert d.pools_kv is True, d
            assert d.pool_factor == 2, d
            assert d.rope_theta == 160000, d
            assert d.yarn_enabled is True, d
        else:
            # 20-39: indexed but UNPOOLED. Ratio 1 is not "no compression" --
            # compress_len is the full sequence, so these layers retrieve over
            # all history and take YaRN with the compressed theta.
            assert 20 <= d.layer_idx <= 39, d
            assert d.compress_ratio == 1, d
            assert d.has_long_range is True, d
            assert d.pools_kv is False, d
            assert d.pool_factor == 1, d
            assert d.rope_theta == 160000, d
            assert d.yarn_enabled is True, d

        # Every layer keeps its sliding window; the indexed path is additive.
        assert d.window_size == 128, d
        assert d.kind == ("mtp" if d.layer_idx >= 40 else "decoder"), d


def test_kv_and_index_source_asymmetry():
    """Four KV sources but eight index sources; only KV sources own a compressor."""
    descriptors = release_text_config().layer_descriptors

    kv_sources = [d.layer_idx for d in descriptors if d.is_kv_source]
    index_sources = [d.layer_idx for d in descriptors if d.is_index_source]
    assert kv_sources == RELEASE_KV_SOURCES
    assert index_sources == RELEASE_INDEX_SOURCES

    # Only the four KV sources own a compressor / indexer key projection --
    # allocating either per compressing layer would over-allocate ~10x.
    assert [d.layer_idx for d in descriptors if d.owns_compressor] == RELEASE_KV_SOURCES
    assert [d.layer_idx for d in descriptors if d.owns_indexer_wk] == RELEASE_KV_SOURCES

    # The extra four index sources contribute queries but own no key projection.
    extra = set(index_sources) - set(kv_sources)
    assert extra == {24, 28, 32, 36}
    for d in descriptors:
        if d.layer_idx in extra:
            assert d.is_index_source and not d.owns_indexer_wk


def test_compressor_dtype_follows_pooling_not_membership():
    """Pooling compressors run fp32; the ratio-1 compressor runs bf16 with no gate."""
    by_idx = {d.layer_idx: d for d in release_text_config().layer_descriptors}
    # Layers 2, 8, 14 are ratio 2 -> pooled -> fp32.
    for i in (2, 8, 14):
        assert by_idx[i].pools_kv is True
        assert by_idx[i].compressor_wkv_dtype == "fp32"
    # Layer 20 is ratio 1 -> unpooled -> bf16 (and it has no wgate in the ckpt).
    assert by_idx[20].pools_kv is False
    assert by_idx[20].compressor_wkv_dtype == "bf16"
    # Non-source layers own no compressor at all.
    assert by_idx[3].compressor_wkv_dtype is None


def test_source_layer_routing_is_most_recent_upstream():
    """A consumer layer reads the most recent source's published cache."""
    by_idx = {d.layer_idx: d for d in release_text_config().layer_descriptors}
    # Consumers between sources read the preceding source.
    assert by_idx[3].kv_source_layer_idx == 2
    assert by_idx[9].kv_source_layer_idx == 8
    assert by_idx[21].kv_source_layer_idx == 20
    assert by_idx[39].kv_source_layer_idx == 20
    # A source layer points at itself.
    assert by_idx[14].kv_source_layer_idx == 14
    # Pure-SWA layers have no long-range path, so no source.
    assert by_idx[0].kv_source_layer_idx is None
    assert by_idx[42].kv_source_layer_idx is None
    # Index sources advance independently of KV sources.
    assert by_idx[30].index_source_layer_idx == 28
    assert by_idx[39].index_source_layer_idx == 36


def test_two_level_candidate_topk_wiring():
    """Layer 20 publishes candidates; only *later* index sources consume them."""
    descriptors = release_text_config().layer_descriptors
    producers = [d.layer_idx for d in descriptors if d.is_candidate_source]
    consumers = [d.layer_idx for d in descriptors if d.consumes_candidates]
    assert producers == [20]
    # 24/28/32/36 mask their own scores with ~candidates before their own top-k.
    # The earlier index sources (2, 8, 14) and layer 20 itself do not.
    assert consumers == [24, 28, 32, 36]


def test_absent_v4_keys_are_not_defaulted():
    """Absent keys are instructions: V4 defaults here would build a wrong net."""
    cfg = release_text_config()
    # No dense FFN prefix and no dense intermediate size -- all layers are MoE.
    assert not hasattr(cfg, "first_k_dense_replace")
    assert not hasattr(cfg, "intermediate_size")
    # V4's spelling of the hash-routing count stays absent; ours is pinned off
    # instead of absent, for the reason in the next test.
    assert not hasattr(cfg, "num_hash_layers")


def test_v4_router_knobs_are_pinned_to_their_identity_setting():
    """The shared V4 gate must use global top-k without hash or group routing."""
    cfg = release_text_config()
    # Hash routing off: the checkpoint has no `tid2eid` table and every
    # `ffn.gate` ships a real `bias`, so all 40 layers route by top-k.
    assert cfg.n_hash_layers == 0
    # One group holding every expert, and that one group selected == global
    # top-k, which is what the reference gate does.
    assert cfg.n_group == 1
    assert cfg.topk_group == 1
    # Pinned by this class, not accepted from a caller: a checkpoint that really
    # does declare group routing must not be able to land on the degenerate
    # setting by omission.
    params = inspect.signature(DeepseekV41TextConfig.__init__).parameters
    assert "n_hash_layers" not in params
    assert "n_group" not in params
    assert "topk_group" not in params


def test_release_dimensions_and_eps():
    cfg = release_text_config()
    assert cfg.hidden_size == 5120
    assert cfg.q_lora_rank == 1280
    assert cfg.n_routed_experts == 384
    assert cfg.moe_intermediate_size == 2304
    assert cfg.index_n_heads == 32
    assert cfg.num_hidden_layers == 40
    assert cfg.head_dim == 512
    assert cfg.qk_rope_head_dim == 64
    assert cfg.qk_nope_head_dim == 448  # derived: 512 - 64
    assert cfg.num_key_value_heads == 1  # all 64 heads share one 512-dim latent
    # TRT-LLM's MLA splits that one 512-wide latent into `kv_lora_rank`
    # un-rotated lanes + `qk_rope_head_dim` rotated ones, so `kv_lora_rank` is
    # 448 and NOT the latent width. Same pair of numbers V4 pins literally.
    assert cfg.kv_lora_rank == 448
    assert cfg.kv_lora_rank + cfg.qk_rope_head_dim == cfg.head_dim
    # ... while the value MLA reads out of that latent is the whole thing.
    assert cfg.v_head_dim == 512
    assert cfg.o_groups == 8
    assert cfg.o_lora_rank == 1024
    assert cfg.vocab_size == 129280
    assert cfg.max_position_embeddings == 1048576
    # 1e-20, down from V4's 1e-6, and load-bearing.
    assert cfg.rms_norm_eps == 1e-20
    # The mHC epsilon is separate and stays at 1e-6.
    assert cfg.hc_eps == 1e-6
    assert cfg.hc_mult == 4
    assert cfg.hc_sinkhorn_iters == 20
    assert [d.layer_idx for d in cfg.layer_descriptors if d.has_engram] == [1, 14]
    assert cfg.engram_num_embeddings_for_layer(1) == 384006168
    assert cfg.engram_num_embeddings_for_layer(14) == 384016682
    assert cfg.engram_pad_token_id == 2
    with pytest.raises(ValueError, match="does not host an Engram table"):
        cfg.engram_num_embeddings_for_layer(2)


def test_dspark_config():
    cfg = release_text_config()
    assert cfg.num_nextn_predict_layers == 3
    assert cfg.dspark_target_layer_ids == []  # not set by the fixture
    cfg = release_text_config(dspark_target_layer_ids=[37, 38, 39])
    assert cfg.dspark_target_layer_ids == [37, 38, 39]
    # Each DSpark stage runs a SMALLER 128-expert top-3 MoE.
    assert cfg.dspark_n_routed_experts == 128
    assert cfg.dspark_num_experts_per_tok == 3
    assert cfg.dspark_markov_rank == 256
    assert cfg.dspark_noise_token_id == 128799


def test_bytes_per_compressed_entry_agrees_with_the_allocator():
    """Capacity accounting uses the actual packed GLOBAL record width."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import row_bytes

    cfg = release_text_config()
    packed_record = row_bytes(cfg.head_dim, "main") + row_bytes(cfg.index_head_dim, "index")
    assert cfg.bytes_per_compressed_entry("fp4", "fp4") == packed_record == 356
    assert cfg.kv_bytes_per_token("fp4", "fp4") == sum(
        packed_record / descriptor.pool_factor
        for descriptor in cfg.layer_descriptors
        if descriptor.is_kv_source
    )
    assert cfg.kv_bytes_per_token("fp4", "fp4") == pytest.approx(890.0)


def test_descriptor_rejects_inconsistent_source_lists():
    """A source layer with ratio 0 is a config contradiction, not a warning."""
    with pytest.raises(ValueError, match="listed as a kv/index source"):
        build_layer_descriptors(
            compress_ratios=[0, 0, 2, 2],
            num_hidden_layers=4,
            rope_theta=10000,
            compress_rope_theta=160000,
            window_size=128,
            kv_source_layer_ids=[0],  # ratio 0 -> cannot own a compressor
            index_source_layer_ids=[2],
        )
    with pytest.raises(ValueError, match="outside the"):
        build_layer_descriptors(
            compress_ratios=[0, 2],
            num_hidden_layers=2,
            rope_theta=10000,
            compress_rope_theta=160000,
            window_size=128,
            kv_source_layer_ids=[7],
            index_source_layer_ids=[],
        )


def test_descriptor_refuses_empty_ratio_list():
    """Never synthesize a default ratio list -- it changes attention semantics."""
    with pytest.raises(ValueError, match="requires the per-layer"):
        build_layer_descriptors(
            compress_ratios=[],
            num_hidden_layers=0,
            rope_theta=10000,
            compress_rope_theta=160000,
            window_size=128,
            kv_source_layer_ids=[],
            index_source_layer_ids=[],
        )


def test_v4_ratio_classifiers_must_not_be_reused_for_v41():
    """V4 sentinels miss V4.1 long-range layers; pooling is the only shared predicate."""
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import (
        DEEPSEEK_V4_SPARSE_RATIO,
        is_compress_layer,
        is_sparse_layer,
    )

    descriptors = release_text_config().layer_descriptors

    # V4's sparse sentinel appears nowhere in V4.1, so `is_sparse_layer` declares
    # *every* layer non-sparse -- the whole indexed long-range path disappears with
    # no error anywhere.
    assert DEEPSEEK_V4_SPARSE_RATIO == 4
    assert sum(is_sparse_layer(d.compress_ratio) for d in descriptors) == 0
    assert sum(d.has_long_range for d in descriptors) == 38

    # `is_compress_layer` is `ratio > 1`, so it finds the 18 pooled layers and
    # misses all 20 ratio-1 layers that retrieve over full history unpooled.
    v4_compress = {d.layer_idx for d in descriptors if is_compress_layer(d.compress_ratio)}
    ours_long_range = {d.layer_idx for d in descriptors if d.has_long_range}
    assert v4_compress == set(range(2, 20))
    assert ours_long_range - v4_compress == set(range(20, 40))
    # It does agree with our narrower `pools_kv`, which is what it actually means.
    assert v4_compress == {d.layer_idx for d in descriptors if d.pools_kv}


# ---------------------------------------------------------------------------
# Quantization layout
# ---------------------------------------------------------------------------


RELEASE_QUANT_CONFIG = {
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "weight_block_size": [32, 32],
    "scale_fmt": "ue8m0",
    "expert_dtype": "fp4",
}


def test_three_distinct_quant_layouts():
    """``quantization_config`` advertises one block; the checkpoint uses three."""
    layout = parse_quantization_layout(RELEASE_QUANT_CONFIG)

    expert = layout[DeepseekV41QuantRole.EXPERT]
    engram = layout[DeepseekV41QuantRole.ENGRAM_EMBED]
    dense = layout[DeepseekV41QuantRole.DENSE]

    # Routed experts: OCP MXFP4 -- 32-element blocks along K only, e8m0 scales,
    # two elements per int8 container.
    assert (expert.weight_dtype, expert.scale_dtype) == ("mxfp4_e2m1", "float8_e8m0")
    assert expert.element_block == (1, 32)
    assert (expert.storage_dtype, expert.pack_factor, expert.packed_dim) == ("int8", 2, 1)
    # Engram tables: fp8 e4m3 with one e8m0 scale per 32 columns, unpacked.
    assert (engram.weight_dtype, engram.element_block, engram.pack_factor) == (
        "float8_e4m3fn",
        (1, 32),
        1,
    )
    # Dense: fp8 e4m3 blockwise 32x32 -- the only role the config's block fits.
    assert (dense.weight_dtype, dense.element_block, dense.pack_factor) == (
        "float8_e4m3fn",
        (32, 32),
        1,
    )

    # The advertised block is right for exactly one of the three roles.
    advertised = tuple(RELEASE_QUANT_CONFIG["weight_block_size"])
    assert sum(1 for e in layout.values() if e.element_block == advertised) == 1


def test_expert_stored_block_is_16_and_that_is_the_trap():
    """MXFP4 stores 32 logical elements in 16 bytes per UE8M0 scale."""
    expert = layout_for_role(DeepseekV41QuantRole.EXPERT)
    assert expert.element_block == (1, 32)  # semantics: 32 elements per scale
    assert expert.stored_block == (1, 16)  # on disk: 16 int8 containers per scale
    # ... and the real header shapes derive the stored block, not the element one.
    assert derive_block((2304, 2560), (2304, 160)) == expert.stored_block
    assert expert.logical_shape((2304, 2560)) == (2304, 5120)
    # NVFP4 would also have block 16 on disk but e4m3 scales and a global scale.
    assert expert.scale_dtype == "float8_e8m0"


def test_unpacked_roles_have_no_stored_vs_element_gap():
    for role in (DeepseekV41QuantRole.DENSE, DeepseekV41QuantRole.ENGRAM_EMBED):
        entry = layout_for_role(role)
        assert entry.stored_block == entry.element_block
        assert entry.logical_shape((7, 11)) == (7, 11)


@pytest.mark.cpu_only
def test_dense_fp8_policy_and_checkpoint_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    """The dense FP8 policy and checkpoint remapper agree on weights and scales."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.weights import expand_mxfp8_scales

    checkpoint_config = dict(RELEASE_QUANT_CONFIG, modules_to_not_convert=["*engram*"])
    quant = ModelConfig._build_deepseek_v41_quant_config(checkpoint_config)
    assert quant.quant_algo == QuantAlgo.FP8_BLOCK_SCALES
    assert quant.group_size == 128
    assert {"*engram*", "*kv_b_proj*", "*k_b_proj*", "*eh_proj"} <= set(quant.exclude_modules)
    assert checkpoint_config["weight_block_size"] == [32, 32]

    native_stems = {
        "attn.wq_a": "self_attn.q_a_proj",
        "attn.wq_b": "self_attn.q_b_proj",
        "attn.indexer.wq_b": "self_attn.indexer.wq_b",
        "attn.wkv": "self_attn.kv_a_proj_with_mqa",
        "attn.wo_b": "self_attn.o_b_proj",
    }
    requant_stems = {
        "attn.wo_a": "self_attn.o_a_proj",
        "ffn.shared_experts.w1": "mlp.shared_experts.gate_proj",
        "ffn.shared_experts.w2": "mlp.shared_experts.down_proj",
        "ffn.shared_experts.w3": "mlp.shared_experts.up_proj",
    }
    values = torch.ones((128, 128))
    values[0, 0] = 448
    weight = values.to(torch.float8_e4m3fn)
    scale_bytes = torch.full((4, 4), 127, dtype=torch.uint8)
    scale_bytes[1, 1] = 117  # 2**-10 would underflow after merging with the 448 tile.
    scale = scale_bytes.view(torch.float8_e8m0fnu)
    dequantized = torch.ones_like(weight, dtype=torch.bfloat16)
    dequant = Mock(return_value=dequantized)
    monkeypatch.setattr(v41, "_dequantize_block32", dequant)
    forwarded = v41._remap_deepseek_v41_checkpoint_keys(
        {
            f"layers.0.{stem}.{suffix}": tensor
            for stem in (*native_stems, *requant_stems)
            for suffix, tensor in (("weight", weight), ("scale", scale))
        },
        num_hidden_layers=1,
    )
    for stem in native_stems.values():
        q = forwarded[f"model.layers.0.{stem}.weight"]
        s = forwarded[f"model.layers.0.{stem}.weight_scale_inv"]
        assert q is weight and s is scale
        expanded = expand_mxfp8_scales(s, tuple(q.shape), 32)
        restored = q.float() * torch.exp2(expanded.float() - 127).repeat_interleave(32, 1)
        assert restored[0, 0].item() == 448
        assert restored[32, 32].item() == 2**-10
    for stem in requant_stems.values():
        weight_key = stem if stem.endswith("o_a_proj") else f"{stem}.weight"
        q = forwarded[f"model.layers.0.{weight_key}"]
        s = forwarded[f"model.layers.0.{stem}.weight_scale_inv"]
        assert q.dtype == torch.float8_e4m3fn
        assert s.dtype == torch.float32 and s.shape == (1, 1)
        torch.testing.assert_close(q.float() * s, dequantized.float(), rtol=0, atol=0)
    assert dequant.call_count == len(requant_stems)
    assert forwarded.census.requantized == len(requant_stems)
    assert forwarded.census.folded == 0


def test_quant_config_rejects_unknown_formats():
    for bad in (
        {"quant_method": "awq"},
        {"scale_fmt": "e4m3"},
        {"expert_dtype": "int4"},
        {"weight_block_size": [32, 32, 32]},
    ):
        with pytest.raises(ValueError):
            parse_quantization_layout(dict(RELEASE_QUANT_CONFIG, **bad))


def test_dense_block_override_never_leaks_to_the_other_roles():
    layout = parse_quantization_layout(dict(RELEASE_QUANT_CONFIG, weight_block_size=[128, 128]))
    assert layout[DeepseekV41QuantRole.DENSE].element_block == (128, 128)
    # The two 1x32 roles are not derived from weight_block_size and must not move.
    assert layout[DeepseekV41QuantRole.EXPERT].element_block == (1, 32)
    assert layout[DeepseekV41QuantRole.ENGRAM_EMBED].element_block == (1, 32)


def test_quant_layout_rejects_incoherent_packing():
    with pytest.raises(ValueError, match="requires packed_dim"):
        DeepseekV41QuantLayout("a", "b", "c", (1, 32), pack_factor=2)
    with pytest.raises(ValueError, match="not divisible by"):
        DeepseekV41QuantLayout("a", "b", "c", (1, 32), pack_factor=2, packed_dim=0)
    with pytest.raises(ValueError, match="pack_factor must be"):
        DeepseekV41QuantLayout("a", "b", "c", (1, 32), pack_factor=0)


def test_quant_role_classification():
    """Routed experts are MXFP4; shared experts next door are not."""
    assert (
        quant_role_for_weight_key("layers.5.ffn.experts.17.w1.weight")
        == DeepseekV41QuantRole.EXPERT
    )
    # The DSpark stages' experts are routed experts too.
    assert quant_role_for_weight_key("mtp.0.ffn.experts.3.w2.weight") == DeepseekV41QuantRole.EXPERT
    # `shared_experts` is plain fp8 32x32 -- it must NOT be read as MXFP4.
    assert (
        quant_role_for_weight_key("layers.5.ffn.shared_experts.w1.weight")
        == DeepseekV41QuantRole.DENSE
    )
    assert (
        quant_role_for_weight_key("layers.1.engram.embed.weight")
        == DeepseekV41QuantRole.ENGRAM_EMBED
    )
    assert quant_role_for_weight_key("layers.5.attn.wq_b.weight") == DeepseekV41QuantRole.DENSE


def test_quant_role_classification_spans_both_naming_conventions():
    """Classify checkpoint ffn and module mlp names without confusing shared experts."""
    for key in (
        "layers.5.mlp.experts.17.w1.weight",
        "model.layers.5.mlp.experts.0.w3.weight",
        "ffn.experts.2.w2.weight",  # leading-anchored, no dotted prefix
    ):
        assert quant_role_for_weight_key(key) == DeepseekV41QuantRole.EXPERT, key

    # `experts.` must be followed by an index: the shared-expert block and a bare
    # `experts` attribute are both DENSE, under either spelling.
    for key in (
        "layers.5.mlp.shared_experts.w1.weight",
        "layers.5.ffn.experts.gate.weight",
        "layers.5.mlp.experts_bias",
    ):
        assert quant_role_for_weight_key(key) == DeepseekV41QuantRole.DENSE, key

    # The Engram role covers the scale as well as the weight -- callers classify a
    # stem, and both members of the pair have to land on the same layout.
    for key in (
        "layers.1.engram.embed.weight",
        "layers.14.engram.embed.scale",
        "layers.1.engram.embed",
    ):
        assert quant_role_for_weight_key(key) == DeepseekV41QuantRole.ENGRAM_EMBED, key
    # ... but a neighbouring Engram tensor is not the embedding table.
    assert (
        quant_role_for_weight_key("layers.1.engram.embed_norm.weight") == DeepseekV41QuantRole.DENSE
    )


def test_load_assert_accepts_every_dtype_spelling_a_loader_can_hand_it():
    """Accept safetensors, torch-name, and torch-dtype spellings consistently."""
    expert = ("layers.0.ffn.experts.0.w1.weight", (2304, 2560), (2304, 160))
    dense = ("layers.0.attn.wkv.weight", (512, 5120), (16, 160))
    for dtypes, (key, wsh, ssh) in (
        (("I8", "int8", "torch.int8"), expert),
        (("F8_E4M3", "float8_e4m3fn", "torch.float8_e4m3fn"), dense),
    ):
        for dtype in dtypes:
            assert_weight_layout(key, wsh, ssh, storage_dtype=dtype)

    # Normalization must not turn into "accept anything": a genuinely wrong dtype
    # still has to raise, under every spelling.
    for dtype in ("BF16", "bfloat16", "torch.bfloat16"):
        with pytest.raises(ValueError, match="expects on-disk dtype"):
            assert_weight_layout(*expert, storage_dtype=dtype)


def test_layout_for_role_rejects_unknown_role():
    with pytest.raises(ValueError, match="Unknown DeepSeek-V4.1 quant role"):
        layout_for_role("nope")


def test_derive_block_rejects_incompatible_shapes():
    with pytest.raises(ValueError, match="different ranks"):
        derive_block((32, 32), (32,))
    with pytest.raises(ValueError, match="not an integer multiple"):
        derive_block((32, 33), (1, 32))


# Representative checkpoint (key, weight shape, scale shape, dtype) layouts.
# The full release-census test covers every tensor; these samples cover shape classes.
RELEASE_QUANTIZED_TENSORS = [
    ("layers.6.attn.wq_a.weight", (1280, 5120), (40, 160), "float8_e4m3fn"),
    ("layers.6.attn.wq_b.weight", (32768, 1280), (1024, 40), "float8_e4m3fn"),
    ("layers.6.attn.wkv.weight", (512, 5120), (16, 160), "float8_e4m3fn"),
    ("layers.6.attn.wo_a.weight", (8192, 4096), (256, 128), "float8_e4m3fn"),
    ("layers.6.attn.wo_b.weight", (5120, 8192), (160, 256), "float8_e4m3fn"),
    ("layers.2.attn.indexer.wq_b.weight", (4096, 1280), (128, 40), "float8_e4m3fn"),
    ("layers.6.ffn.shared_experts.w1.weight", (2304, 5120), (72, 160), "float8_e4m3fn"),
    ("layers.6.ffn.shared_experts.w2.weight", (5120, 2304), (160, 72), "float8_e4m3fn"),
    ("layers.6.ffn.experts.0.w1.weight", (2304, 2560), (2304, 160), "int8"),
    ("layers.6.ffn.experts.0.w2.weight", (5120, 1152), (5120, 72), "int8"),
    ("layers.6.ffn.experts.0.w3.weight", (2304, 2560), (2304, 160), "int8"),
    ("layers.1.engram.embed.weight", (384006168, 256), (384006168, 8), "float8_e4m3fn"),
    ("layers.14.engram.embed.weight", (384016682, 256), (384016682, 8), "float8_e4m3fn"),
    ("mtp.0.main_proj.weight", (5120, 15360), (160, 480), "float8_e4m3fn"),
    ("mtp.1.ffn.experts.7.w1.weight", (2304, 2560), (2304, 160), "int8"),
    ("mtp.2.ffn.shared_experts.w3.weight", (2304, 5120), (72, 160), "float8_e4m3fn"),
]


def test_release_shape_sample_passes_the_load_assert():
    """Validate all sampled checkpoint shape classes, not an exhaustive tensor census."""
    for key, wshape, sshape, dtype in RELEASE_QUANTIZED_TENSORS:
        layout = assert_weight_layout(
            key, wshape, sshape, dtype, quantization_config=RELEASE_QUANT_CONFIG
        )
        assert layout.stored_block == derive_block(wshape, sshape), key


def test_load_assert_raises_on_a_corrupted_expectation():
    """Demonstrate the assert is load-bearing, not decorative."""
    # An expert tensor whose scale implies block 32 on disk -- i.e. someone wrote
    # it as if MXFP4 were unpacked. Must raise, and must say so in MXFP4 terms.
    with pytest.raises(ValueError, match=r"expects on-disk block \(1, 16\)"):
        assert_weight_layout("layers.6.ffn.experts.0.w1.weight", (2304, 2560), (2304, 80), "int8")
    # A dense tensor read with the experts' 1x32 block.
    with pytest.raises(ValueError, match=r"expects on-disk block \(32, 32\)"):
        assert_weight_layout(
            "layers.6.attn.wq_b.weight", (32768, 1280), (32768, 40), "float8_e4m3fn"
        )
    # Right block, wrong container dtype: an expert stored unpacked as fp8 while
    # quantization_config still says fp4.
    with pytest.raises(ValueError, match="expects on-disk dtype 'int8'"):
        assert_weight_layout(
            "layers.6.ffn.experts.0.w1.weight",
            (2304, 2560),
            (2304, 160),
            "float8_e4m3fn",
        )


def test_modeling_pack_factor_is_what_rules_nvfp4_out():
    """Account for nibble packing when distinguishing MXFP4 from NVFP4."""
    naive = derive_block((2304, 2560), (2304, 160))
    nvfp4_like = DeepseekV41QuantLayout(
        weight_dtype="nvfp4_e2m1",
        storage_dtype="int8",
        scale_dtype="float8_e4m3fn",  # NVFP4 uses e4m3 scales, not e8m0
        element_block=(1, 16),
        pack_factor=2,
        packed_dim=1,
    )
    expert = layout_for_role(DeepseekV41QuantRole.EXPERT)

    # The coincidence: the naive derivation equals NVFP4's element block.
    assert naive == (1, 16) == nvfp4_like.element_block
    # The discrimination: NVFP4 would store 8 containers per scale, MXFP4 16.
    assert nvfp4_like.stored_block == (1, 8)
    assert expert.stored_block == (1, 16) == naive
    assert naive != nvfp4_like.stored_block
    # And the scale dtype is a second, independent discriminator.
    assert expert.scale_dtype == "float8_e8m0" != nvfp4_like.scale_dtype


# ---------------------------------------------------------------------------
# Composite config
# ---------------------------------------------------------------------------


def test_composite_config_rebuilds_nested_dicts():
    """Sub-configs arrive as dicts from AutoConfig and must become real classes."""
    cfg = DeepseekV41Config(
        text_config={
            "compress_ratios": RELEASE_RATIOS,
            "kv_source_layer_ids": RELEASE_KV_SOURCES,
            "index_source_layer_ids": RELEASE_INDEX_SOURCES,
            "engram_layer_ids": RELEASE_ENGRAM_LAYERS,
        },
        vision_config={"num_hidden_layers": 32, "hidden_size": 1024},
        quantization_config=RELEASE_QUANT_CONFIG,
    )
    assert isinstance(cfg.text_config, DeepseekV41TextConfig)
    assert isinstance(cfg.vision_config, DeepseekV41VisionConfig)
    assert cfg.model_type == "deepseek_v41"
    # Language-model fields are forwarded so flat-config call sites keep working.
    assert cfg.hidden_size == 5120
    assert cfg.n_routed_experts == 384
    assert len(cfg.layer_descriptors) == 43
    # The top-level quantization_config drives the per-role layout.
    assert cfg.quantization_layout[DeepseekV41QuantRole.EXPERT].element_block == (1, 32)
    # ... and must reach the text tower, which is where the weight loader reads it.
    # V4.1 publishes it at the top level only, so a sub-config built from
    # `text_config` verbatim has no quantization policy at all and every weight
    # then loads as unquantized bf16 with nothing raising.
    assert cfg.text_config.quantization_config == RELEASE_QUANT_CONFIG
    assert cfg.text_config.quantization_layout == cfg.quantization_layout
    # Unknown attributes still raise rather than silently returning None.
    with pytest.raises(AttributeError):
        _ = cfg.definitely_not_a_field


def test_composite_config_refuses_to_drop_the_quantization_policy(monkeypatch):
    """Reject a composite whose text tower loses the top-level quantization policy."""
    explicit = {"quant_method": "fp8", "scale_fmt": "ue8m0", "weight_block_size": [32, 32]}
    cfg = DeepseekV41Config(
        text_config={"quantization_config": explicit},
        quantization_config=RELEASE_QUANT_CONFIG,
    )
    assert cfg.text_config.quantization_config == explicit

    # An unquantized checkpoint stays unquantized on both levels.
    bare = DeepseekV41Config()
    assert getattr(bare, "quantization_config", None) is None
    assert getattr(bare.text_config, "quantization_config", None) is None

    class _Dropper(DeepseekV41TextConfig):
        """A text config that silently discards the key, as a refactor might."""

        def __init__(self, **kwargs):
            kwargs.pop("quantization_config", None)
            super().__init__(**kwargs)

    # Swap the class through `sub_configs` rather than by subclassing the composite:
    # transformers 5.x replaces `__init__` on any config subclass that does not
    # define its own, so a subclass here would never run the code under test.
    monkeypatch.setitem(DeepseekV41Config.sub_configs, "text_config", _Dropper)
    with pytest.raises(RuntimeError, match="did not survive into text_config"):
        DeepseekV41Config(text_config={}, quantization_config=RELEASE_QUANT_CONFIG)


def test_vision_config_derived_geometry():
    """Each number here is confirmed by a checkpoint tensor shape."""
    v = DeepseekV41VisionConfig()
    assert v.head_dim == 64  # wqkv (3072, 1024) / 16 heads
    assert v.rope_dim == 32  # 2D RoPE splits head_dim in half
    # patch_embed.proj.weight is (1024, 588): a Linear, not a Conv2d.
    assert v.patch_input_dim == 588  # 3 * 14**2
    assert v.downsample_ratio == 3
    # aligner.w1.weight is (5120, 9216) = hidden * downsample_ratio**2.
    assert v.hidden_size * v.downsample_ratio**2 == 9216
    # mlp.w1 is (5632, 1024) = 2 * intermediate: gate and up are fused in w1,
    # and mlp.w2 is (1024, 2816), so the tower is gated with only two tensors.
    assert 2 * v.intermediate_size == 5632
    # The vision tower keeps eps 1e-6; only the text stack moved to 1e-20.
    assert v.rms_norm_eps == 1e-6


def test_layer_plan_dump_is_gate_shaped():
    """The dump must carry every field the layer_plan gate compares."""
    dump = release_text_config().layer_plan_dump()
    assert dump["num_ratio_entries"] == 43
    assert len(dump["layers"]) == 43
    required = {
        "has_long_range",
        "pools_kv",
        "pool_factor",
        "rope_theta",
        "yarn_enabled",
        "window_size",
        "is_kv_source",
        "is_index_source",
        "owns_compressor",
        "owns_indexer_wk",
        "has_engram",
        "compressor_wkv_dtype",
    }
    for row in dump["layers"]:
        assert required <= set(row), required - set(row)
    # Round-trips through JSON, since that is how the gate consumes it.
    assert json.loads(json.dumps(dump))["layers"][20]["rope_theta"] == 160000


def test_descriptors_are_cached_and_immutable():
    cfg = release_text_config()
    assert cfg.layer_descriptors is cfg.layer_descriptors
    with pytest.raises(Exception):
        cfg.layer_descriptors[0].has_long_range = True


@pytest.mark.parametrize(
    "name,value",
    [
        ("main_kv_dtype", "fp4"),
        ("main_kv_dtype", "auto"),
        ("kv_source_layer_ids", None),
        ("index_source_layer_ids", []),
    ],
)
def test_csa2_rejects_retired_precision_and_source_overrides(name, value):
    from pydantic import ValidationError

    from tensorrt_llm.llmapi.llm_args import CSA2SparseAttentionConfig

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CSA2SparseAttentionConfig(**{name: value})


# ---------------------------------------------------------------------------
# The released checkpoint, through the real loader
#
# Real checkpoint loading also checks registration and composite-field forwarding.
# ---------------------------------------------------------------------------

RELEASE_CHECKPOINT = (
    Path(os.environ.get("LLM_MODELS_ROOT", "/code/llm-models")) / "DeepSeek-V4.1-Flash"
)

requires_release_checkpoint = pytest.mark.skipif(
    not (RELEASE_CHECKPOINT / "config.json").is_file(),
    reason=f"release checkpoint not present at {RELEASE_CHECKPOINT}",
)


@requires_release_checkpoint
@pytest.mark.cpu_only
def test_serving_baseline_uses_checkpoint_csa2_defaults() -> None:
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

    args = TorchLlmArgs(model=str(RELEASE_CHECKPOINT), **serving_model_kwargs())
    assert isinstance(args.sparse_attention_config, CSA2SparseAttentionConfig)
    assert args.sparse_attention_config.use_fp8_staging is True
    model_config = ModelConfig.from_pretrained(
        RELEASE_CHECKPOINT,
        sparse_attention_config=args.sparse_attention_config,
        attn_backend=args.attn_backend,
    )
    sparse = model_config.sparse_attention_config
    assert isinstance(sparse, CSA2SparseAttentionConfig)
    params = sparse.to_sparse_params(pretrained_config=model_config.pretrained_config)
    assert params.use_fp8_staging is True
    assert params.layout.kv_source_layer_ids == tuple(RELEASE_KV_SOURCES)
    assert params.layout.index_source_layer_ids == tuple(RELEASE_INDEX_SOURCES)


@requires_release_checkpoint
@pytest.mark.parametrize("materialize_defaults", [False, True])
def test_csa2_default_survives_model_config_rebuild(materialize_defaults):
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import CSA2SparseAttentionConfig

    requested = CSA2SparseAttentionConfig()
    if materialize_defaults:
        requested = CSA2SparseAttentionConfig.model_validate(requested.model_dump())
    model_config = ModelConfig.from_pretrained(
        RELEASE_CHECKPOINT, sparse_attention_config=requested
    )
    sparse = model_config.sparse_attention_config
    assert isinstance(sparse, CSA2SparseAttentionConfig)
    layout = sparse.to_sparse_params(pretrained_config=model_config.pretrained_config).layout
    assert layout.compress_ratios == tuple(RELEASE_RATIOS)
    assert layout.kv_source_layer_ids == tuple(RELEASE_KV_SOURCES)
    assert layout.index_source_layer_ids == tuple(RELEASE_INDEX_SOURCES)


@requires_release_checkpoint
@pytest.mark.parametrize("materialize_defaults", [False, True])
def test_csa2_explicit_default_preserves_weight_owners(materialize_defaults):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Mode
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import CSA2SparseAttentionConfig

    baseline = ModelConfig.from_pretrained(RELEASE_CHECKPOINT)
    requested = CSA2SparseAttentionConfig()
    if materialize_defaults:
        requested = CSA2SparseAttentionConfig.model_validate(requested.model_dump())
    explicit = ModelConfig.from_pretrained(RELEASE_CHECKPOINT, sparse_attention_config=requested)
    expected = baseline.sparse_attention_config.to_sparse_params(
        pretrained_config=baseline.pretrained_config
    )
    actual = explicit.sparse_attention_config.to_sparse_params(
        pretrained_config=explicit.pretrained_config
    )
    assert asdict(actual) == asdict(expected)
    plans = [actual.layout.layer(i) for i in range(len(RELEASE_RATIOS))]
    assert [p.layer_idx for p in plans if p.mode == CSA2Mode.FULL] == RELEASE_KV_SOURCES
    assert [
        p.layer_idx for p in plans if p.mode in (CSA2Mode.FULL, CSA2Mode.REINDEX)
    ] == RELEASE_INDEX_SOURCES


@requires_release_checkpoint
def test_release_config_json_is_the_shape_the_loader_assumes():
    """Pin the three structural facts the loader branch depends on."""
    raw = json.loads((RELEASE_CHECKPOINT / "config.json").read_text())
    assert raw["model_type"] == "deepseek_v41"
    assert raw["architectures"] == ["DeepseekV41ForCausalLM"]
    # Nested towers, and `quantization_config` at the top level *only*.
    assert isinstance(raw["text_config"], dict) and raw["text_config"]
    assert isinstance(raw["vision_config"], dict) and raw["vision_config"]
    assert "quantization_config" not in raw["text_config"]
    assert raw["quantization_config"]["quant_method"] == "fp8"
    assert raw["quantization_config"]["scale_fmt"] == "ue8m0"
    # `dtype` is top-level only, so the text tower needs it resolved explicitly.
    assert "dtype" not in raw["text_config"] and "torch_dtype" not in raw["text_config"]
    # The inline dict is the only quantization policy here. `model_config.py` keeps V4.1
    # out of the `hf_quant_config.json` (modelopt-format) gate on that basis; if a future
    # release ships one, this fails and that decision gets revisited rather than silently
    # taking a branch that was never exercised.
    assert not (RELEASE_CHECKPOINT / "hf_quant_config.json").exists()


@requires_release_checkpoint
def test_release_config_exposes_every_text_field_off_the_composite():
    """Forward every text field on the bare composite before ModelConfig mirrors it."""
    raw = json.loads((RELEASE_CHECKPOINT / "config.json").read_text())
    config = DeepseekV41Config.from_pretrained(RELEASE_CHECKPOINT)
    # `AutoConfig` must land on the same class -- that is the path production takes.
    assert type(AutoConfig.from_pretrained(RELEASE_CHECKPOINT)) is type(config)
    assert isinstance(config, DeepseekV41Config)

    unreachable = [key for key in raw["text_config"] if not hasattr(config, key)]
    assert not unreachable, f"{len(unreachable)} text_config field(s) unreachable: {unreachable}"

    # `model_type` is the one field both levels legitimately own and disagree on:
    # the composite is `deepseek_v41` (what CONFIG_MAPPING keys off) and the text
    # tower is `deepseek_v41_text`. Forwarding must not shadow the composite's own,
    # so its value is checked separately rather than against the sub-config.
    assert config.model_type == raw["model_type"] == "deepseek_v41"
    assert config.text_config.model_type == raw["text_config"]["model_type"]

    for key, value in raw["text_config"].items():
        if key == "model_type":
            continue
        expected = getattr(config.text_config, key)
        if isinstance(expected, (list, tuple)):
            assert list(getattr(config, key)) == list(expected), key
        else:
            assert getattr(config, key) == expected, key

    assert config.swiglu_limit == raw["text_config"]["swiglu_limit"] is not None
    assert config.routed_scaling_factor == raw["text_config"]["routed_scaling_factor"]
    assert config.hc_sinkhorn_iters == raw["text_config"]["hc_sinkhorn_iters"]
    # `engram_vocab_size` is the one forwarded field whose *type* the config
    # deliberately changes, so it is checked against both meanings rather than
    # against the raw scalar. config.json publishes the scalar per-bucket target
    # the prime search starts from; `EngramConfig.engram_vocab_size` is indexed
    # per n-gram order (`vocab_size_per_ngram[ngram - 2]`, engram.py:262). The
    # scalar survives under `engram_bucket_vocab_size` and the forwarded field is
    # its broadcast over the `engram_max_ngram_size - 1` orders. Asserting the raw
    # scalar here would pin the wrong one of the two.
    assert config.engram_bucket_vocab_size == raw["text_config"]["engram_vocab_size"]
    assert config.engram_vocab_size == [raw["text_config"]["engram_vocab_size"]] * (
        raw["text_config"]["engram_max_ngram_size"] - 1
    )
    assert config.engram_max_ngram_size == raw["text_config"]["engram_max_ngram_size"]

    # Delegation, not duplication: the composite must not have copied the text
    # fields into its own __dict__, or the two levels can diverge under mutation
    # and `to_dict()` serializes each field twice.
    copied = sorted(set(raw["text_config"]) & set(vars(config)) - {"model_type"})
    assert not copied, f"text fields copied onto the composite: {copied}"

    # And a name that is in neither config still raises rather than reading None.
    with pytest.raises(AttributeError):
        _ = config.definitely_not_a_field


@requires_release_checkpoint
def test_release_config_layer_plan_matches_the_hand_written_fixture():
    """Match the fixture layer plan to the real checkpoint configuration."""
    from tensorrt_llm._torch.model_config import ModelConfig

    pretrained_config = ModelConfig.from_pretrained(RELEASE_CHECKPOINT).pretrained_config
    assert pretrained_config.layer_plan_dump() == release_text_config().layer_plan_dump()


@requires_release_checkpoint
def test_release_checkpoint_loads_in_a_fresh_interpreter():
    """Load the real checkpoint without relying on this test module to register it."""
    child = textwrap.dedent(
        f"""
        import json, sys
        assert "tensorrt_llm" not in sys.modules

        import transformers
        try:
            transformers.AutoConfig.from_pretrained({str(RELEASE_CHECKPOINT)!r})
            transformers_alone = "loaded"
        except Exception as exc:
            transformers_alone = type(exc).__name__

        from tensorrt_llm._torch.model_config import ModelConfig
        pc = ModelConfig.from_pretrained({str(RELEASE_CHECKPOINT)!r}).pretrained_config
        print("@@" + json.dumps({{
            "transformers_alone": transformers_alone,
            "cls": type(pc).__name__,
            "architectures": pc.architectures,
            "text_architectures": pc.text_config.architectures,
            "text_quantization_config": pc.text_config.quantization_config,
            "text_dtype": str(pc.text_config.torch_dtype),
            "dtype": str(pc.torch_dtype),
            "num_descriptors": len(pc.layer_descriptors),
            "hidden_size": pc.hidden_size,
        }}))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", child], capture_output=True, text=True, timeout=900
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    payload = json.loads(
        next(line for line in completed.stdout.splitlines() if line.startswith("@@"))[2:]
    )

    assert payload["cls"] == "DeepseekV41Config"
    assert payload["architectures"] == ["DeepseekV41ForConditionalGeneration"]
    assert payload["text_architectures"] == ["DeepseekV41ForCausalLM"]
    # The gate: V4.1's quantization_config is top level only, so a text tower with
    # an empty one means 475 GiB would load as unquantized bf16 with nothing raising.
    assert payload["text_quantization_config"]["quant_method"] == "fp8"
    assert payload["text_quantization_config"]["scale_fmt"] == "ue8m0"
    assert payload["text_quantization_config"]["expert_dtype"] == "fp4"
    # `dtype` is top level only; an unresolved text dtype is a None.itemsize crash
    # in the KV-cache byte sizing rather than a wrong answer.
    assert payload["dtype"] == payload["text_dtype"] == "torch.bfloat16"
    assert payload["num_descriptors"] == 43
    assert payload["hidden_size"] == 5120
    # If transformers ever learns `deepseek_v41` this stops being a discriminator;
    # say so out loud rather than letting the test quietly weaken.
    assert payload["transformers_alone"] in ("ValueError", "loaded")


@requires_release_checkpoint
@pytest.mark.parametrize(
    "name",
    [
        "layers.0.attn.wkv.weight",  # attention projection
        "layers.0.ffn.shared_experts.w1.weight",  # shared-expert FFN
        "layers.1.engram.wkv.weight",  # Engram projection
    ],
)
def test_dense_fp8_widens_to_bf16_without_loss(name):
    """BF16 widening must preserve E4M3 values multiplied by power-of-two UE8M0 scales."""
    import torch
    from safetensors import safe_open

    weight_map = json.loads((RELEASE_CHECKPOINT / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    scale_name = name.removesuffix(".weight") + ".scale"

    tensors = {}
    for key in (name, scale_name):
        with safe_open(RELEASE_CHECKPOINT / weight_map[key], framework="pt") as handle:
            tensors[key] = handle.get_tensor(key)
    weight, scale = tensors[name], tensors[scale_name]

    assert weight.dtype is torch.float8_e4m3fn
    assert scale.dtype is torch.float8_e8m0fnu

    layout = parse_quantization_layout(
        json.loads((RELEASE_CHECKPOINT / "config.json").read_text())["quantization_config"]
    )[DeepseekV41QuantRole.DENSE]
    block = tuple(layout.element_block)
    assert block == (32, 32)
    assert tuple(a // b for a, b in zip(weight.shape, scale.shape)) == block
    assert tuple(a % b for a, b in zip(weight.shape, scale.shape)) == (0, 0)

    scale_f32 = scale.to(torch.float32)
    # Exponent-only means every scale is exactly 2**k; anything else and the exactness
    # argument below does not hold.
    log2 = torch.log2(scale_f32[scale_f32 > 0])
    assert torch.equal(log2, log2.round())

    expanded = scale_f32.repeat_interleave(block[0], 0).repeat_interleave(block[1], 1)
    reference = weight.to(torch.float32) * expanded
    via_bf16 = weight.to(torch.bfloat16) * expanded.to(torch.bfloat16)

    assert torch.isfinite(reference).all()
    assert (reference - via_bf16.to(torch.float32)).abs().max().item() == 0.0
    assert (reference - reference.to(torch.bfloat16).to(torch.float32)).abs().max().item() == 0.0

    # Control: the zero above is a measurement, not an identity. Nudge the scale off a
    # power of two -- what an fp32-scale checkpoint looks like -- and bf16 must lose bits.
    nudged = expanded * 1.0000001
    lossy_ref = weight.to(torch.float32) * nudged
    lossy_bf16 = weight.to(torch.bfloat16) * nudged.to(torch.bfloat16)
    assert (lossy_ref - lossy_bf16.to(torch.float32)).abs().max().item() > 0.0


# Weight loading and coverage

RELEASE_TENSOR_COUNT = 96085
RELEASE_SHARDS = 48

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the attention modules are built on the GPU"
)

# safetensors dtype tag -> torch dtype name. e2m1 has no torch dtype, which is
# why the checkpoint stores packed experts as ``I8``.
_SAFETENSORS_DTYPES = {
    "F8_E4M3": "float8_e4m3fn",
    "F8_E8M0": "float8_e8m0fnu",
    "I8": "int8",
    "U8": "uint8",
    "BF16": "bfloat16",
    "F16": "float16",
    "F32": "float32",
}

# Only size-bearing dims, and only enough to construct 40 real layers on one GPU.
# No per-layer role is touched, so layer i keeps its real compress ratio, its real
# pooling flags and its real Engram membership -- which is what decides how many
# tensors each layer claims.
_REDUCTIONS = {
    "vocab_size": 1024,
    "n_routed_experts": 8,
    "moe_intermediate_size": 256,
    "engram_vocab_size": 4096,
}


def _reduce(before, value):
    if isinstance(before, (list, tuple)):
        return type(before)(value for _ in before)
    return value


@pytest.fixture(scope="module")
def release_meta_weights():
    """Every released tensor as a ``meta`` tensor of its true shape and dtype."""
    weights = {}
    shards = sorted(RELEASE_CHECKPOINT.glob("model-*-of-*.safetensors"))
    assert len(shards) == RELEASE_SHARDS, f"expected {RELEASE_SHARDS} shards, saw {len(shards)}"
    for shard in shards:
        with open(shard, "rb") as handle:
            (length,) = struct.unpack("<Q", handle.read(8))
            header = json.loads(handle.read(length))
        for key, entry in header.items():
            if key == "__metadata__":
                continue
            dtype = getattr(torch, _SAFETENSORS_DTYPES[entry["dtype"]])
            weights[key] = torch.empty(tuple(entry["shape"]), dtype=dtype, device="meta")
    assert len(weights) == RELEASE_TENSOR_COUNT
    return weights


def _build_release_model(*, bounded_replay_on_generation=False):
    """Construct the release topology with reduced vocabulary and expert counts."""
    # ModelConfig freezes itself at the end of from_pretrained, so both the mapping
    # and the sequence ceiling have to be handed in rather than assigned afterwards.
    # The model refuses to construct without a ceiling it can serve exactly, because
    # the two-level candidate prefilter is not implemented; the ceiling is
    # ``candidate_topk_blocks * candidate_block_size``, read off the checkpoint so
    # this test keeps agreeing with whatever the release publishes.
    raw = json.loads((RELEASE_CHECKPOINT / "config.json").read_text())
    raw_text = raw.get("text_config", raw)
    inert_up_to = int(raw_text.get("candidate_topk_blocks", 0) or 0) * int(
        raw_text.get("candidate_block_size", 0) or 0
    )
    assert inert_up_to > 0, "the release publishes a two-level candidate prefilter"
    model_config = ModelConfig.from_pretrained(
        str(RELEASE_CHECKPOINT),
        sparse_attention_config=CSA2SparseAttentionConfig(),
        mapping=Mapping(world_size=1, tp_size=1, pp_size=1, rank=0),
        max_seq_len=inert_up_to,
    )
    model_config.extra_attrs["bounded_replay_on_generation"] = bounded_replay_on_generation
    text = getattr(model_config.pretrained_config, "text_config", model_config.pretrained_config)
    for name, value in _REDUCTIONS.items():
        setattr(text, name, _reduce(getattr(text, name), value))
    text.__dict__.pop("_layer_descriptors", None)

    # Construct on the host: a ``torch.device("cuda")`` default-device context
    # breaks YaRN RoPE table creation, which calls ``.numpy()``.
    torch.cuda.set_device(0)
    return v41.DeepseekV41ForCausalLM(model_config)


@pytest.fixture(scope="module")
def release_model():
    """``_build_release_model()``, paid once for the whole file."""
    return _build_release_model()


def _audited_loader(model, weights, monkeypatch):
    """Run the loader's audits over ``weights``, returning the loader.

    Numerics are stubbed (see the module docstring) and the ``attn_sink``
    parameters V4's walk *assigns* rather than declares are stood up, because the
    coverage audit runs after the walk in a real load but before it here.

    Where they are stood up matters, and getting it wrong is why a real 8-way load
    failed the coverage audit while this file was green: the checkpoint keys them
    on the attention module (``self_attn.attn_sink``) but both of the walk's
    branches register them on the inner implementation, so the parameter is
    ``self_attn.mqa.attn_sink``. Creating them under the checkpoint's own name
    instead makes the audit's ``_V41_RELOCATED_PARAMS`` rewrite look unnecessary
    and hides its absence. So this mirrors the walk rather than the key.
    """
    monkeypatch.setattr(
        v41,
        "_dequantize_block32",
        lambda weight, scale: torch.empty(tuple(weight.shape), dtype=torch.bfloat16, device="meta"),
    )

    loader = v41.DeepseekV41WeightLoader(model)
    forwarded = loader.remap_checkpoint_keys(
        dict(weights),
        num_hidden_layers=model.config.num_hidden_layers,
        kv_lora_rank=model.config.kv_lora_rank,
    )
    for name, module in model.named_modules():
        key = f"{name}.attn_sink"
        if key not in forwarded:
            continue
        # The walk relocates the sinks onto `mqa`; refuse to invent a home for
        # them rather than quietly falling back to the parent and re-hiding the
        # mismatch this helper exists to expose.
        target = module if isinstance(module, v41.DeepseekV41Attention) else None
        assert target is not None, f"{name} holds an attn_sink key but has no mqa child"
        # `attn_sink` is *preset to None* by the attention module's __init__ so that
        # forward can skip the sinks when the checkpoint has none -- so `hasattr` is
        # already true here and testing it would silently create nothing.
        if getattr(target, "attn_sink", None) is None:
            target.attn_sink = torch.nn.Parameter(
                torch.empty(tuple(forwarded[key].shape), dtype=torch.float32, device="meta"),
                requires_grad=False,
            )
        # The grouped projection loader materializes this registered buffer;
        # coverage runs after loading, while this fixture skips the numeric load.
        scale_key = f"{name}.o_a_proj.weight_scale_inv"
        if scale_key in forwarded and target.o_a_proj_scale is None:
            target.o_a_proj_scale = torch.empty_like(forwarded[scale_key], device="meta")
    loader.assert_load_complete()
    return loader


# --------------------------------------------------------------------------- #
# Name rules. These are pure, so they are also the cheapest place to pin down
# *why* a key is claimed -- the release fixtures below only prove that it is.
# --------------------------------------------------------------------------- #


def test_structural_key_rewrites_are_exactly_the_walk_s_two_renames():
    """Flat mHC names become structured; a fused sub-module takes the fused name."""
    assert v41._v41_structural_key("model.layers.3.hc_attn_fn") == "model.layers.3.hc_attn.fn"
    assert v41._v41_structural_key("model.layers.3.hc_ffn_base") == "model.layers.3.hc_ffn.base"
    assert v41._v41_structural_key("model.layers.3.hc_head_scale") == "model.layers.3.hc_head.scale"

    # ``_fn`` only splits off a real mHC stem, never an arbitrary module.
    assert v41._v41_structural_key("model.layers.3.mlp.act_fn") == "model.layers.3.mlp.act_fn"

    for source in ("gate_proj", "up_proj"):
        assert (
            v41._v41_structural_key(f"model.layers.0.mlp.shared_experts.{source}.weight")
            == "model.layers.0.mlp.shared_experts.gate_up_proj.weight"
        )
    assert (
        v41._v41_structural_key("model.layers.0.self_attn.q_a_proj.weight")
        == "model.layers.0.self_attn.wq_a.weight"
    )

    # `attn_sink` is *not* rewritten even though the walk relocates it: `mqa` is a
    # plain attribute rather than a sub-module, so no name reaches the destination
    # and `_v41_load_coverage` resolves it as an attribute instead. Rewriting here
    # would produce a name that can never match and re-hide the mismatch.
    assert (
        v41._v41_structural_key("model.layers.0.self_attn.attn_sink")
        == "model.layers.0.self_attn.attn_sink"
    )

    # A name that is already the model's is left alone.
    assert (
        v41._v41_structural_key("model.layers.0.self_attn.kv_a_proj_with_mqa.weight")
        == "model.layers.0.self_attn.wkv.weight"
    )


@pytest.mark.parametrize("load_vision_bias", [False, True])
def test_vision_bias_is_loaded_only_when_enabled(load_vision_bias: bool) -> None:
    bias = torch.tensor([0.25, -0.5, 1.0, -2.0])
    vision_bias = torch.tensor([-9.0, 3.0, 4.0, 2.0])
    forwarded = v41._remap_deepseek_v41_checkpoint_keys(
        {"layers.7.ffn.gate.bias": bias, "layers.7.ffn.gate.bias_vl": vision_bias},
        num_hidden_layers=40,
        load_vision_bias=load_vision_bias,
    )
    expected = {"model.layers.7.mlp.gate.e_score_correction_bias": bias}
    if load_vision_bias:
        expected["model.layers.7.mlp.gate.e_score_correction_bias_vl"] = vision_bias
    assert set(forwarded) == set(expected)
    for key, value in expected.items():
        torch.testing.assert_close(forwarded[key], value, rtol=0, atol=0)
    assert forwarded.census.ignored_count == int(not load_vision_bias)


@pytest.mark.parametrize("has_vl_bias", [False, True])
def test_the_gate_builds_the_vision_bias_only_when_asked(has_vl_bias):
    """Loading optional vision bias must leave the text-routing bias unchanged."""
    gate = DeepseekV4Gate(
        hidden_size=8,
        num_experts=4,
        top_k=2,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.5,
        is_hashed=False,
        dtype=torch.bfloat16,
        has_vl_bias=has_vl_bias,
    )
    assert (gate.e_score_correction_bias_vl is not None) is has_vl_bias

    bias = torch.tensor([0.25, -0.5, 1.0, -2.0])
    weights = {"weight": torch.zeros(4, 8, dtype=torch.bfloat16), "e_score_correction_bias": bias}
    if has_vl_bias:
        weights["e_score_correction_bias_vl"] = torch.tensor([-9.0, -9.0, -9.0, -9.0])

    gate.load_weights([weights])
    assert torch.equal(gate.e_score_correction_bias, bias.to(torch.float32))
    assert torch.equal(gate.routing_method.e_score_correction_bias, bias.to(torch.float32))
    if has_vl_bias:
        assert torch.equal(gate.e_score_correction_bias_vl, weights["e_score_correction_bias_vl"])


def test_keeping_one_mtp_layer_ignores_only_the_rest():
    assert v41._v41_ignore_reason("mtp.0.eh_proj.weight", keep_mtp_layers=1) is None
    kept = v41._v41_ignore_reason("mtp.1.eh_proj.weight", keep_mtp_layers=1)
    assert kept is not None and kept[0] == "mtp.<i>.* for i >= 1"


def test_census_verify_rejects_a_count_that_does_not_close():
    census = v41.DeepseekV41LoadCensus(total=10, folded=2, forwarded=3)
    census.ignore("a", ("pattern", "because"))
    with pytest.raises(ValueError, match="does not close"):
        census.verify()

    census.forwarded = 7
    census.verify()
    assert census.consumed == 9 and census.ignored_count == 1
    rendered = census.render()
    assert "consumed + ignored == 10" in rendered and "because" in rendered


def test_forwarded_weights_separates_forwarded_from_synthesized_keys():
    forwarded = v41._V41ForwardedWeights({"a": 1}, census=v41.DeepseekV41LoadCensus(total=1))
    forwarded["b"] = 2
    forwarded.update({"c": 3, "a": 4})
    assert forwarded.forwarded_keys == frozenset({"a"})
    assert forwarded.synthesized_keys == {"b", "c"}
    assert forwarded.all_keys == {"a", "b", "c"}


@pytest.mark.parametrize("precompute", [False, True])
def test_context_only_checkpoint_filter_never_reads_decoder_tensors(precompute):
    """Discard decoder tensors before lazy reads, including their quantization scales."""
    prefix = {
        "embed.weight": "model.embed_tokens.weight",
        "layers.19.attn_norm.weight": "model.layers.19.input_layernorm.weight",
        "layers.19.ffn.gate.weight": "model.layers.19.mlp.gate.weight",
    }
    boundary = {
        "layers.20.attn_norm.weight": "model.layers.20.input_layernorm.weight",
        "layers.20.attn.compressor.wkv.weight": "model.layers.20.self_attn.compressor.wkv.weight",
        "layers.20.attn.compressor.norm.weight": "model.layers.20.self_attn.compressor.norm.weight",
        "layers.20.attn.indexer.wk.weight": "model.layers.20.self_attn.indexer.wk.weight",
        "layers.20.attn.indexer.k_norm.weight": "model.layers.20.self_attn.indexer.k_norm.weight",
    }
    decoder = {
        "head.weight",
        "norm.weight",
        "layers.20.attn.wq_a.weight",
        "layers.20.attn.wq_a.scale",
        "layers.20.attn.indexer.wq_b.weight",
        "layers.20.attn.indexer.wq_b.scale",
        "layers.20.attn.indexer.weights_proj.weight",
        "layers.20.attn.attn_sink",
        "layers.20.ffn_norm.weight",
        "layers.20.ffn.experts.0.w1.weight",
        "layers.20.ffn.experts.0.w1.scale",
        "layers.20.hc_attn_fn",
        "layers.20.hc_ffn_fn",
        "layers.21.attn_norm.weight",
        "layers.39.ffn.experts.0.w2.weight",
        "layers.39.ffn.experts.0.w2.scale",
    }
    retained = prefix | (boundary if precompute else {})
    omitted = decoder | (set() if precompute else set(boundary))
    tensor = torch.ones(1, 1, dtype=torch.bfloat16)

    class UnreadableDecoderWeights(dict):
        def __getitem__(self, key):
            assert key not in omitted, f"read omitted decoder tensor {key}"
            return super().__getitem__(key)

        def items(self):
            for key in self:
                yield key, self[key]

    weights = UnreadableDecoderWeights({key: tensor for key in set(retained) | omitted})
    forwarded = v41._remap_deepseek_v41_checkpoint_keys(
        weights,
        num_hidden_layers=40,
        context_only_split=20,
        context_only_precompute=precompute,
    )
    assert set(forwarded) == set(retained.values())
    assert all(value is tensor for value in forwarded.values())
    ignored = {key for names in forwarded.census.ignored.values() for key in names}
    assert ignored == omitted
    assert forwarded.census.forwarded == len(retained)
    assert forwarded.census.ignored_count == len(omitted)
    assert forwarded.census.checked_pairs == 0
    forwarded.census.verify()


def test_context_only_checkpoint_filter_preserves_default_full_load():
    weights = {
        "embed.weight": torch.ones(1),
        "head.weight": torch.ones(1),
        "norm.weight": torch.ones(1),
        "layers.20.attn.wq_a.weight": torch.ones(1),
        "layers.39.ffn.gate.weight": torch.ones(1),
    }
    forwarded = v41._remap_deepseek_v41_checkpoint_keys(weights, num_hidden_layers=40)
    assert set(forwarded) == {
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.norm.weight",
        "model.layers.20.self_attn.q_a_proj.weight",
        "model.layers.39.mlp.gate.weight",
    }
    assert forwarded.census.forwarded == len(weights)
    assert forwarded.census.ignored_count == 0


# --------------------------------------------------------------------------- #
# The release checkpoint.
# --------------------------------------------------------------------------- #


@requires_cuda
@requires_release_checkpoint
def test_release_census_closes_and_coverage_is_empty(
    release_model, release_meta_weights, monkeypatch
):
    """The audit accepts the released checkpoint, and accounts for all of it."""
    loader = _audited_loader(release_model, release_meta_weights, monkeypatch)
    census = loader.census

    assert census.total == RELEASE_TENSOR_COUNT
    assert census.consumed + census.ignored_count == RELEASE_TENSOR_COUNT
    assert census.consumed > 0 and census.ignored_count > 0

    # Every ignored group carries both a pattern and a reason, and nothing is
    # ignored by accident: the multimodal tower, its three separator embeddings, the
    # per-token vision router bias, and the unbuilt MTP layers. Why the vision bias
    # is a *separate* group rather than folded into the tower, and why dropping it
    # is provably lossless, is
    # `test_the_per_token_gate_bias_is_dropped_and_the_drop_stays_lossless`.
    assert {pattern for pattern, _ in census.ignored} == {
        "vision.* | aligner.*",
        "image_start | image_end | image_newline",
        "layers.<i>.ffn.gate.bias_vl",
        "mtp.*",
    }
    for pattern, why in census.ignored:
        assert why.strip(), pattern

    # The vision-bias group must contain exactly the checkpoint's `bias_vl` tensors
    # and nothing else. Asserting the membership and not just the pattern is what
    # makes this catch the failure mode that matters: a too-greedy match that
    # swallowed `ffn.gate.bias` as well would leave the pattern set identical and
    # close the census just as neatly, while silently stripping the bias that
    # routing actually reads. The expected set is derived from the checkpoint rather
    # than hardcoded, so it keeps agreeing with whatever the release ships.
    vl_group = next(
        names
        for (pattern, _), names in census.ignored.items()
        if pattern == "layers.<i>.ffn.gate.bias_vl"
    )
    expected_vl = {k for k in release_meta_weights if k.endswith("ffn.gate.bias_vl")}
    assert set(vl_group) == expected_vl, (
        f"the vision-bias drop claimed {len(vl_group)} tensors, the checkpoint ships "
        f"{len(expected_vl)}; symmetric difference "
        f"{set(vl_group) ^ expected_vl}"
    )
    assert expected_vl, "the release is expected to ship a per-token vision router bias"
    # The static bias is never swept up with it.
    all_ignored = {name for names in census.ignored.values() for name in names}
    swept = {k for k in all_ignored if k.endswith("ffn.gate.bias") and not k.startswith("mtp.")}
    assert not swept, f"the bias routing reads was ignored for {sorted(swept)[:5]}"

    # The layout check is the expensive audit; prove it actually ran, on more
    # pairs than the dense tail alone.
    assert census.checked_pairs > 40000, census.checked_pairs
    assert census.requantized > 0 and census.forwarded >= 2 * census.requantized

    unexpected, unfed = v41._v41_load_coverage(release_model, loader._forwarded.all_keys)
    assert unexpected == [] and unfed == []


@requires_cuda
@requires_release_checkpoint
def test_context_only_release_census_covers_every_retained_parameter(
    release_meta_weights, monkeypatch
):
    monkeypatch.setenv("TRTLLM_DISAGG_ROLE", "context")
    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "1")
    model = _build_release_model(bounded_replay_on_generation=True)
    assert model.model.disagg_context_only
    assert model.model.decoder_replay_split == 20
    assert len(model.model.layers) == model.config.num_hidden_layers == 40
    loader = _audited_loader(model, release_meta_weights, monkeypatch)
    census = loader.census
    assert census.total == RELEASE_TENSOR_COUNT
    census.verify()
    assert v41._v41_load_coverage(model, loader._forwarded.all_keys) == ([], [])
    ignored = {name for names in census.ignored.values() for name in names}
    assert {"head.weight", "norm.weight"} <= ignored
    assert all(
        key in ignored
        for key in release_meta_weights
        if key.startswith("layers.") and int(key.split(".")[1]) > 20
    )
    assert "layers.19.attn_norm.weight" not in ignored
    assert "layers.20.attn_norm.weight" not in ignored
    assert "layers.20.attn.compressor.wkv.weight" not in ignored
    assert "layers.20.attn.indexer.wk.weight" not in ignored
    assert "layers.20.attn.indexer.k_norm.weight" not in ignored


@requires_cuda
@requires_release_checkpoint
def test_a_sink_the_walk_failed_to_deliver_is_not_silently_accepted(
    release_model, release_meta_weights, monkeypatch
):
    """Reject an attention sink claimed by name but not delivered to its attribute."""
    loader = _audited_loader(release_model, release_meta_weights, monkeypatch)
    sinks = [
        module
        for name, module in release_model.named_modules()
        if f"{name}.attn_sink" in loader._forwarded
    ]
    assert sinks, "no attention module claims an attn_sink key"
    assert all(getattr(inner, "attn_sink", None) is not None for inner in sinks)

    # Undone by the next `_audited_loader` call, which repopulates any sink it finds
    # cleared -- but restore anyway so an assertion failure here cannot leak state
    # into the rest of the module-scoped fixture's lifetime.
    saved = [inner.attn_sink for inner in sinks]
    for inner in sinks:
        inner.attn_sink = None
    try:
        with pytest.raises(ValueError, match="no module would load"):
            loader.assert_load_complete()
        unexpected, _ = v41._v41_load_coverage(release_model, loader._forwarded.all_keys)
        assert len(unexpected) == len(sinks)
        assert all(key.endswith(".attn_sink") for key in unexpected), unexpected[:4]
    finally:
        for inner, sink in zip(sinks, saved):
            inner.attn_sink = sink


def _quant_config(model):
    """The ``quantization_config`` dict the loader will actually consult."""
    pretrained = model.model_config.pretrained_config
    text = getattr(pretrained, "text_config", pretrained)
    return text.quantization_config


@requires_cuda
@requires_release_checkpoint
@pytest.mark.parametrize(
    "role,block,stored",
    [
        # 16 is what the experts *store* (two e2m1 per byte); reading it as the
        # element block is the silent 2x scale-stride bug this check exists for.
        (DeepseekV41QuantRole.EXPERT, (1, 16), (1, 8)),
        (DeepseekV41QuantRole.EXPERT, (1, 64), (1, 32)),
        (DeepseekV41QuantRole.ENGRAM_EMBED, (1, 16), (1, 16)),
    ],
)
def test_loader_refuses_a_wrong_built_in_block_extent(
    release_model, release_meta_weights, monkeypatch, role, block, stored
):
    """Corrupt one role's built-in expectation; the load must fail loudly, not warn."""
    expected = v41_config._EXPECTED_LAYOUT[role]
    assert expected.element_block != block
    corrupted = dataclasses.replace(expected, element_block=block)
    # ``stored_block`` is derived, so corrupting the element extent moves the
    # on-disk expectation with it -- which is the whole point: the two are only
    # equal for the unpacked roles.
    assert corrupted.stored_block == stored
    monkeypatch.setitem(v41_config._EXPECTED_LAYOUT, role, corrupted)

    with pytest.raises(ValueError) as excinfo:
        _audited_loader(release_model, release_meta_weights, monkeypatch)

    message = str(excinfo.value)
    assert f"role {role!r}" in message
    assert f"expects on-disk block {stored}" in message
    assert "derives" in message
    if expected.pack_factor > 1:
        # Only the packed role can hide a stored/element gap, so only it has to
        # spell both extents out in the failure.
        assert f"Elements per scale are {block}" in message


@requires_cuda
@requires_release_checkpoint
@pytest.mark.parametrize("block", [(128, 128), (1, 32), (64, 64)])
def test_loader_refuses_a_wrong_published_dense_block(
    release_model, release_meta_weights, monkeypatch, block
):
    """Dense is the one role the checkpoint itself gets to specify.

    ``quantization_config.weight_block_size`` overrides the built-in dense entry,
    so corrupting the built-in cannot move the dense check while the checkpoint
    publishes a block -- this is the vector that can, and 128x128 is specifically
    V4's block, i.e. the value a V4-shaped assumption would supply.
    """
    quant = _quant_config(release_model)
    assert tuple(quant["weight_block_size"]) == (32, 32)
    monkeypatch.setitem(quant, "weight_block_size", list(block))

    with pytest.raises(ValueError) as excinfo:
        _audited_loader(release_model, release_meta_weights, monkeypatch)
    message = str(excinfo.value)
    assert f"role {DeepseekV41QuantRole.DENSE!r}" in message
    assert f"expects on-disk block {block}" in message


@requires_cuda
@requires_release_checkpoint
def test_the_built_in_dense_block_is_the_fallback_and_is_itself_checked(
    release_model, release_meta_weights, monkeypatch
):
    """Neither dense expectation source is unchecked.

    Drop the published block and the built-in takes over -- and it agrees with the
    checkpoint, so the load still passes. Corrupt the built-in *then*, and the load
    fails. So a checkpoint that publishes nothing is still audited.
    """
    quant = _quant_config(release_model)
    monkeypatch.delitem(quant, "weight_block_size")
    loader = _audited_loader(release_model, release_meta_weights, monkeypatch)
    assert loader.census.checked_pairs > 40000

    role = DeepseekV41QuantRole.DENSE
    monkeypatch.setitem(
        v41_config._EXPECTED_LAYOUT,
        role,
        dataclasses.replace(v41_config._EXPECTED_LAYOUT[role], element_block=(128, 128)),
    )
    with pytest.raises(ValueError, match=r"expects on-disk block \(128, 128\)"):
        _audited_loader(release_model, release_meta_weights, monkeypatch)


@requires_cuda
@requires_release_checkpoint
# Raw checkpoint names, not model names: the checkpoint ships DeepSeek's own
# ``layers.N.attn.w*`` spelling and the remap renames it.
@pytest.mark.parametrize("dropped", ["attn.wq_b", "attn.wo_b", "hc_attn_fn"])
def test_loader_refuses_to_leave_a_parameter_unfilled(
    release_model, release_meta_weights, monkeypatch, dropped
):
    """Hide a checkpoint tensor; the parameter it fed must be reported by name."""
    weights = {k: v for k, v in release_meta_weights.items() if dropped not in k}
    assert len(weights) < len(release_meta_weights)

    with pytest.raises(ValueError) as excinfo:
        _audited_loader(release_model, weights, monkeypatch)
    assert "no checkpoint tensor would fill" in str(excinfo.value)


@requires_cuda
@requires_release_checkpoint
def test_loader_refuses_a_tensor_no_module_would_load(
    release_model, release_meta_weights, monkeypatch
):
    """A key that is neither ignorable nor addressable must be named, not dropped."""
    weights = dict(release_meta_weights)
    weights["layers.0.attn.wq_c.weight"] = torch.empty(8, dtype=torch.bfloat16, device="meta")

    with pytest.raises(ValueError) as excinfo:
        _audited_loader(release_model, weights, monkeypatch)
    message = str(excinfo.value)
    assert "no module would load" in message
    assert "wq_c" in message or "q_c_proj" in message
    # The census rides along with the failure, so a rejection is diagnosable.
    assert "raw checkpoint tensors" in message


@requires_cuda
@requires_release_checkpoint
def test_v41_every_fp8_consumer_receives_its_scale(
    release_model, release_meta_weights, monkeypatch
):
    """Every FP8 consumer receives one scale per weight, including fused modules."""
    model = release_model
    assert model.model_config.quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES

    monkeypatch.setattr(
        v41,
        "_dequantize_block32",
        lambda weight, scale: torch.empty(tuple(weight.shape), dtype=torch.bfloat16, device="meta"),
    )
    loader = v41.DeepseekV41WeightLoader(model)
    forwarded = loader.remap_checkpoint_keys(
        dict(release_meta_weights),
        num_hidden_layers=model.config.num_hidden_layers,
        kv_lora_rank=model.config.kv_lora_rank,
    )
    keys = set(forwarded.all_keys)

    # Who consumes a key is decided the same way the coverage audit decides it, so
    # this test and that one cannot disagree about module ownership.
    modules = dict(model.named_modules())
    consumers = {
        name
        for name, mod in modules.items()
        if name and hasattr(mod, "load_weights") and not isinstance(mod, v41.DeepseekV41Attention)
    }

    def consumer_of(key):
        parts = key.split(".")
        for i in range(len(parts) - 1, 0, -1):
            prefix = ".".join(parts[:i])
            if prefix in consumers:
                return prefix
        return None

    weights_per, scales_per = {}, {}
    for key in keys:
        key = v41._v41_structural_key(key)
        owner = consumer_of(key)
        if owner is None:
            continue
        if key.endswith(".weight"):
            weights_per[owner] = weights_per.get(owner, 0) + 1
        elif key.endswith((".weight_scale_inv", ".weight_scale")) or (
            key.endswith(".scale") and isinstance(modules[owner], v41.EngramFp8Projection)
        ):
            # The specialized Engram loader consumes the checkpoint's raw
            # 32x32 UE8M0 scale name, then expands it to native MXFP8 metadata.
            scales_per[owner] = scales_per.get(owner, 0) + 1

    needs_scale = [
        name
        for name, mod in modules.items()
        if getattr(getattr(mod, "weight", None), "dtype", None) is torch.float8_e4m3fn
        and getattr(mod, "weight_scale", None) is not None
    ]
    assert needs_scale, "the model built no FP8 block-scaled modules"

    starved = []
    for name in needs_scale:
        owner = name if name in consumers else (consumer_of(f"{name}.weight") or name)
        n_w, n_s = weights_per.get(owner, 0), scales_per.get(owner, 0)
        if n_s < n_w:
            starved.append(f"{name} (consumer {owner}: {n_w} weight(s), {n_s} scale(s))")

    assert not starved, (
        f"{len(starved)} of {len(needs_scale)} fp8 module(s) allocate a weight_scale that "
        f"no forwarded key fills. Each will cast its bf16 weight into fp8 unscaled and "
        f"multiply by uninitialized memory, and neither the load census nor `Linear` will "
        f"say so. Preserve the missing weight/scale pair in the consumer's native layout.\n  "
        + "\n  ".join(sorted(starved)[:12])
    )


def _compressor_only_attention_for_load_test():
    """Real adapter/donor loading without constructing attention kernels."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.compressor import CSA2Compressor

    module = v41.DeepseekV41Attention.__new__(v41.DeepseekV41Attention)
    torch.nn.Module.__init__(module)
    module.compressor = CSA2Compressor(4, 128, 2, 1e-20)
    with torch.no_grad():
        for parameter in module.compressor.parameters():
            parameter.zero_()
    module.num_groups = module.o_lora_rank = 1
    module.mapping = Mapping()
    module.attention_mapping = module.mapping
    module.projection_quantization = "mxfp8"
    return module


def test_csa2_loader_fuses_shared_compressor_value_then_gate():
    module = _compressor_only_attention_for_load_test()
    values = torch.arange(128 * 4, dtype=torch.float32).reshape(128, 4).bfloat16()
    gates = -values - 17
    remapped = v41._remap_deepseek_v41_checkpoint_keys(
        {
            "layers.0.attn.compressor.wkv.weight": values,
            "layers.0.attn.compressor.wgate.weight": gates,
            "layers.0.attn.compressor.norm.weight": torch.ones(128, dtype=torch.bfloat16),
        },
        num_hidden_layers=1,
    )
    prefix = "model.layers.0.self_attn."
    assert prefix + "compressor.wkv_gate.weight" in remapped
    module.load_weights([{key.removeprefix(prefix): value for key, value in remapped.items()}])
    torch.testing.assert_close(
        module.compressor.wkv_gate.weight, torch.cat((values, gates)), atol=0, rtol=0
    )
    root = torch.nn.Module()
    root.model = torch.nn.Module()
    layer = torch.nn.Module()
    layer.self_attn = module
    root.model.layers = torch.nn.ModuleList([layer])
    keys = list(remapped)
    assert v41._v41_load_coverage(root, keys) == ([], [])
    module.compressor.wkv_gate = None
    unexpected, _ = v41._v41_load_coverage(root, keys)
    assert unexpected == ["model.layers.0.self_attn.compressor.wkv_gate.weight"]


@pytest.mark.parametrize("malformation", ["short", "transposed"])
def test_csa2_loader_rejects_malformed_fused_compressor(malformation):
    module = _compressor_only_attention_for_load_test()
    before = module.compressor.wkv_gate.weight.detach().clone()
    fused = torch.zeros((256, 4), dtype=torch.float32)
    weights = {
        "compressor.wkv_gate.weight": fused,
        "compressor.norm.weight": torch.ones(128, dtype=torch.bfloat16),
    }
    if malformation == "short":
        weights["compressor.wkv_gate.weight"] = fused[:-1]
    else:
        weights["compressor.wkv_gate.weight"] = fused.T
    with pytest.raises(ValueError, match="compressor"):
        module.load_weights([weights])
    torch.testing.assert_close(module.compressor.wkv_gate.weight, before, atol=0, rtol=0)


@pytest.mark.parametrize("case", ["raw-consumable", "remapped", "skipped"])
def test_csa2_subtrees_load_once_before_shared_walk(monkeypatch, case):
    from types import SimpleNamespace

    from tensorrt_llm._torch.models.checkpoints.base_weight_loader import ConsumableWeightsDict

    loaded = []

    class Attention(torch.nn.Module):
        def load_weights(self, weights):
            loaded.append(weights[0])

    model = torch.nn.Module()
    model.self_attn = Attention()
    model.draft_model = torch.nn.Module()
    model.draft_model.self_attn = Attention()
    model.config = SimpleNamespace(num_hidden_layers=1, kv_lora_rank=4)
    model.model_config = SimpleNamespace(pretrained_config=model.config)
    monkeypatch.setattr(v41, "DeepseekV41Attention", Attention)
    loader = v41.DeepseekV41WeightLoader(model)
    tensors = {
        "self_attn.weight": torch.tensor([3.0]),
        "draft_model.self_attn.weight": torch.tensor([5.0]),
        "other.weight": torch.tensor([7.0]),
    }
    remaps, delegated = [], []

    def remap(weights, **kwargs):
        remaps.append(kwargs)
        return dict(tensors)

    def shared_walk(self, weights, skip_modules):
        delegated.append((dict(weights.items()), skip_modules))
        assert "self_attn" in skip_modules

    monkeypatch.setattr(loader, "remap_checkpoint_keys", remap)
    monkeypatch.setattr(v41.DeepseekV4WeightLoader, "_load_weights_impl", shared_walk)
    weights = (
        ConsumableWeightsDict({"embed.weight": torch.tensor([1.0])})
        if case == "raw-consumable"
        else dict(tensors)
    )
    skipped = ["self_attn"] if case == "skipped" else []
    loader._load_weights_impl(weights, skipped)
    assert len(delegated) == 1
    assert len(remaps) == int(case == "raw-consumable")
    assert len(loaded) == int(case != "skipped")
    if loaded:
        assert set(loaded[0]) == {"weight"}
        torch.testing.assert_close(loaded[0]["weight"], tensors["self_attn.weight"])
    remaining, _ = delegated[0]
    assert "other.weight" in remaining and "draft_model.self_attn.weight" in remaining
    if case == "raw-consumable":
        assert not weights
        assert "self_attn.weight" not in remaining


# Bounded decoder replay


class _ReplayMetadata(_FakeMetadata):
    _get_csa2_buffer = CSA2TrtllmMetadata._get_csa2_buffer
    _copy_host = CSA2TrtllmMetadata._copy_host
    _copy_csa2_tensor = CSA2TrtllmMetadata._copy_csa2_tensor
    get_adp_token_counts = CSA2TrtllmMetadata.get_adp_token_counts
    set_adp_token_counts = CSA2TrtllmMetadata.set_adp_token_counts
    decoder_replay_plan = None
    padded_num_tokens = None

    def __init__(self, lengths, num_contexts):
        super().__init__(lengths, num_contexts)
        self.all_rank_num_tokens = None
        self.kv_cache_manager.enable_block_reuse = False
        self.csa2_precomputed_kv_layers = set()
        self.csa2_replay_query_rows = None


def _make_policy_case() -> tuple[DeepseekV41Model, SimpleNamespace]:
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Mode

    window = DeepseekV41TextConfig().sliding_window
    num_layers, last_source = 40, 20
    source = SimpleNamespace(
        self_attn=SimpleNamespace(
            compressor=object(),
            index_wk=object(),
            index_wq_b=object(),
            layer=SimpleNamespace(mode=CSA2Mode.FULL, compress_ratio=1, kv_source=last_source),
        )
    )
    layers = [None] * last_source + [source]
    layers.extend(
        SimpleNamespace(
            self_attn=SimpleNamespace(
                compressor=None,
                layer=SimpleNamespace(mode=CSA2Mode.REUSE, compress_ratio=1, kv_source=last_source),
            )
        )
        for _ in range(num_layers - last_source - 1)
    )
    model = DeepseekV41Model.__new__(DeepseekV41Model)
    nn.Module.__init__(model)
    model.disagg_context_only = False
    model.disagg_remote_tail_replay = False
    model.ced_kv_precompute = False
    model.num_hidden_layers = num_layers
    model.layers = layers
    model.engram_layer_ids = []
    layout = SimpleNamespace(kv_source_layer_ids=[last_source], window_size=window)
    config = SimpleNamespace(
        sparse_attention_config=SimpleNamespace(
            to_sparse_params=lambda **kwargs: SimpleNamespace(layout=layout)
        ),
        mapping=SimpleNamespace(pp_size=1, cp_size=1, enable_attention_dp=False),
        spec_config=None,
        extra_attrs={"bounded_replay_on_generation": True},
        pretrained_config=SimpleNamespace(sliding_window=window),
    )
    return model, config


@pytest.mark.parametrize("attention_dp", [False, True])
@pytest.mark.parametrize("remote_tail", [False, True])
@pytest.mark.parametrize(
    "role,allowed",
    [
        ("generation", True),
        ("context", True),
        (None, False),
        ("unknown", False),
    ],
)
def test_attention_dp_replay_policy_requires_uniform_layer_schedule(
    monkeypatch: pytest.MonkeyPatch,
    attention_dp: bool,
    remote_tail: bool,
    role: str | None,
    allowed: bool,
) -> None:
    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "1")
    if role is None:
        monkeypatch.delenv("TRTLLM_DISAGG_ROLE", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_DISAGG_ROLE", role)
    model, config = _make_policy_case()
    config.mapping.enable_attention_dp = attention_dp
    config.extra_attrs["bounded_replay_on_generation"] = remote_tail

    expected = (20, 128) if not attention_dp or not remote_tail or allowed else (None, 0)
    assert model._resolve_replay_policy(config) == expected
    assert model.ced_kv_precompute is (expected[0] == 20)


@pytest.mark.parametrize(
    "unsupported",
    [
        "bounded-off",
        "pipeline",
        "context-parallel",
        "speculative",
        "late-engram",
        "decoder-kv-owner",
    ],
)
def test_remote_tail_attention_dp_policy_keeps_unsupported_paths_disabled(
    monkeypatch: pytest.MonkeyPatch,
    unsupported: str,
) -> None:
    monkeypatch.setenv("TRTLLM_DISAGG_ROLE", "generation")
    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "1")
    model, config = _make_policy_case()
    config.mapping.enable_attention_dp = True
    if unsupported == "bounded-off":
        monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "0")
    elif unsupported == "pipeline":
        config.mapping.pp_size = 2
    elif unsupported == "context-parallel":
        config.mapping.cp_size = 2
    elif unsupported == "speculative":
        config.spec_config = SimpleNamespace(
            spec_dec_mode=SimpleNamespace(is_dspark=lambda: True),
            draft_is_embedded_in_target=True,
            target_layer_ids=[10, 20],
        )
    elif unsupported == "late-engram":
        model.engram_layer_ids = [21]
    elif unsupported == "decoder-kv-owner":
        model.layers[21].self_attn.compressor = object()

    assert model._resolve_replay_policy(config) == (None, 0)
    assert not model.ced_kv_precompute


@pytest.mark.parametrize("all_token_states_required", [False, True])
@pytest.mark.parametrize("remote_mode", [None, "source", "destination"])
def test_remote_tail_attention_dp_never_enables_local_row_compaction(
    monkeypatch: pytest.MonkeyPatch,
    all_token_states_required: bool,
    remote_mode: str | None,
) -> None:
    monkeypatch.setenv("TRTLLM_DISAGG_ROLE", "generation")
    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "1")
    model, config = _make_policy_case()
    model.model_config = config
    model.disagg_remote_tail_replay = True
    config.mapping.enable_attention_dp = True
    model.decoder_replay_split, model.decoder_replay_window = model._resolve_replay_policy(config)
    assert model.decoder_replay_split == 20
    metadata = _ReplayMetadata([1024, 1], num_contexts=1)
    metadata.mapping = config.mapping
    metadata.csa2_remote_tail_mode = remote_mode
    saved_lengths = metadata.seq_lens.clone()

    assert model._plan_bounded_replay(metadata, all_token_states_required) == (None, None)
    torch.testing.assert_close(metadata.seq_lens, saved_lengths)
    assert metadata.csa2_replay_query_rows is None
    assert metadata.prepare_calls == 0


@pytest.mark.parametrize("private_decoder", [False, True])
@pytest.mark.parametrize("pass_requests", [False, True])
@pytest.mark.parametrize("all_rows", [False, True])
def test_replay_uses_pool_policy_independently_of_requests(
    monkeypatch, private_decoder, pass_requests, all_rows
):
    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "1")
    model, config = _make_policy_case()
    model.decoder_replay_split, model.decoder_replay_window = model._resolve_replay_policy(config)
    model._decoder_replay_observed = False
    model._decoder_replay_reuse_warning_emitted = False
    metadata = _ReplayMetadata([257, 6], num_contexts=1)
    metadata.kv_cache_manager.enable_block_reuse = True
    metadata.kv_cache_manager.has_private_swa_suffix = Mock(return_value=private_decoder)
    requests = [SimpleNamespace(py_ced_replay=None)] if pass_requests else None
    split, plan = model._plan_bounded_replay(metadata, all_rows, requests)
    metadata.kv_cache_manager.has_private_swa_suffix.assert_called_once_with(20)
    if private_decoder:
        assert split == 20
        assert plan.replay_seq_lens.tolist() == ([257, 6] if all_rows else [128, 6])
        assert plan.swa_floors == ([0, 0] if all_rows else [129, 0])
    else:
        assert split is None and plan is None


@pytest.mark.parametrize(
    "mode",
    [
        "compact",
        "final_window",
        "all_empty",
        "full_states",
        "late_full_states",
        "late_padding",
        "unprepared",
        "padded",
        "prefill_graph",
        "decode_graph",
    ],
)
@pytest.mark.parametrize("speculative", [False, True])
def test_adp_replay_counts_follow_final_execution(monkeypatch, mode, speculative):
    from tensorrt_llm._torch import utils
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import decoder_replay as replay
    from tensorrt_llm._torch.distributed import Distributed
    from tensorrt_llm._torch.pyexecutor.engine.runners.common import get_all_rank_num_tokens

    model, config = _make_policy_case()
    model.decoder_replay_split = 20
    model.decoder_replay_window = 128
    model._decoder_replay_observed = False
    model._decoder_replay_reuse_warning_emitted = False
    metadata = _ReplayMetadata([257], num_contexts=1)
    metadata.mapping = Mapping(world_size=2, tp_size=2, rank=0, enable_attention_dp=True)
    metadata.kv_cache_manager.has_private_swa_suffix = Mock(return_value=True)
    if mode in ("final_window", "all_empty"):
        metadata.decoder_context_ends = (1024,)
    local_rows = 512 if mode == "padded" else 257
    metadata.padded_num_tokens = 512 if mode == "padded" else None
    metadata.is_cuda_graph = mode == "decode_graph"
    monkeypatch.setattr(replay, "get_per_request_prefill_cuda_graph_flag", lambda: False)
    monkeypatch.setattr(utils, "get_per_request_prefill_cuda_graph_flag", lambda: False)
    # No distributed access is permitted inside plan preparation or consumption.
    monkeypatch.setattr(Distributed, "get", Mock(side_effect=AssertionError("forward collective")))
    if mode != "unprepared":
        model.prepare_adp_inputs(metadata, all_token_states_required=mode == "full_states")
    decoder_rows = (
        0
        if mode in ("final_window", "all_empty")
        else 128
        if mode in ("compact", "late_full_states", "late_padding", "prefill_graph")
        else local_rows
    )
    peer_rows = 0 if mode == "all_empty" else 513 if mode in ("unprepared", "decode_graph") else 128
    if speculative:
        from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine

        gather = Mock(
            return_value=torch.tensor(
                [[local_rows, decoder_rows, 6, 1, 1], [513, peer_rows, 9, 2, 2]]
            )
        )
        engine = SimpleNamespace(
            enable_attention_dp=True,
            mapping=metadata.mapping,
            dist=SimpleNamespace(tp_cp_allgather_int64=gather),
        )
        spec = SimpleNamespace(dp_num_tokens=lambda: 6, seq_lens=[6], num_generations=1)
        metadata.all_rank_num_tokens, spec_counts = (
            PyTorchModelEngine._get_all_rank_num_tokens_and_spec_counts(engine, metadata, spec)
        )
        assert spec_counts == [[6, 9], [1, 2], [1, 2]]
        gather.assert_called_once_with([local_rows, decoder_rows, 6, 1, 1])
    else:
        gather = Mock(return_value=torch.tensor([[local_rows, decoder_rows], [513, peer_rows]]))
        metadata.all_rank_num_tokens = get_all_rank_num_tokens(
            metadata,
            enable_attention_dp=True,
            mapping=metadata.mapping,
            dist=SimpleNamespace(tp_allgather_int64=gather),
        )
        gather.assert_called_once_with([local_rows, decoder_rows])
    if mode == "late_full_states":
        with pytest.raises(ValueError, match="Full token states must be requested"):
            model._plan_bounded_replay(metadata, True)
        assert metadata.all_rank_num_tokens == [257, 513]
        return
    if mode == "late_padding":
        metadata.padded_num_tokens = 512
        with pytest.raises(ValueError, match="ADP input rows changed"):
            model._plan_bounded_replay(metadata, False)
        return
    if mode == "prefill_graph":
        metadata.all_rank_num_tokens = [1024, 1024]
        metadata.padded_num_tokens = 1024
        monkeypatch.setattr(utils, "get_per_request_prefill_cuda_graph_flag", lambda: True)
    saved_counts = metadata.all_rank_num_tokens
    split, plan = model._plan_bounded_replay(metadata, mode in ("full_states", "unprepared"))
    if mode in ("prefill_graph", "decode_graph", "unprepared"):
        assert (split, plan) == (None, None)
        assert metadata.all_rank_num_tokens is saved_counts
        return
    assert split == 20
    assert plan is metadata.decoder_replay_plan
    assert plan.replay_all_rank_num_tokens == [decoder_rows, peer_rows]
    assert plan.replays_local_tokens is (mode in ("compact", "final_window", "all_empty"))
    if not plan.replays_local_tokens:
        hidden = torch.randn(local_rows, 8)
        assert replay.gather_replayed_rows(hidden, plan) is hidden
        assert replay.scatter_replayed_rows(hidden, plan) is hidden


@pytest.mark.parametrize(
    "chunks", [(513,), (128, 128, 128, 128, 1), (127, 129, 127, 130), (1, 127)]
)
@pytest.mark.parametrize("window", [64, 128, 256])
def test_final_window_replay_covers_prompt_suffix_once(chunks, window):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.decoder_replay import (
        plan_decoder_replay,
    )

    prompt_end = sum(chunks)
    req = SimpleNamespace(prompt_len=prompt_end, py_ced_replay=None)
    selected = []
    offset = 0
    for length in chunks:
        metadata = _ReplayMetadata([length], num_contexts=1)
        metadata.kv_cache_params.num_cached_tokens_per_seq = [offset]
        plan = plan_decoder_replay(metadata, window, [req], private_decoder=True)
        selected.extend(offset + row for row in plan.rows.tolist())
        assert plan.swa_floors == [max(0, prompt_end - window)]
        assert plan.replay_num_cached[0] + plan.replay_seq_lens[0] == offset + length
        offset += length
    assert selected == list(range(prompt_end))[-window:]


@pytest.mark.parametrize("full_output", [False, True])
def test_final_window_replay_freezes_endpoints_and_preserves_mixed_consumers(full_output):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.decoder_replay import (
        enter_decoder_replay,
        exit_decoder_replay,
        plan_decoder_replay,
    )

    metadata = _ReplayMetadata([10, 3, 6], num_contexts=2)
    metadata.kv_cache_params.num_cached_tokens_per_seq = [0, 3, 100]
    metadata.kv_lens_cuda = torch.tensor([10, 6, 103])  # Live speculative endpoint is corrected.
    reqs = [
        SimpleNamespace(prompt_len=50, py_ced_replay=None, py_return_context_logits=full_output),
        SimpleNamespace(prompt_len=6, py_ced_replay=None),
    ]
    CSA2TrtllmMetadata.prepare_context_replay(metadata, reqs)
    reqs[0].prompt_len = 10  # Later scheduling must not change this forward's selection.
    plan = plan_decoder_replay(metadata, 8, reqs, private_decoder=True)
    assert plan.replay_seq_lens.tolist() == ([10, 3, 6] if full_output else [0, 3, 6])
    assert plan.swa_floors == ([0, 0, 0] if full_output else [42, 0, 0])
    assert plan.rows.tolist() == list(range(0 if full_output else 10, 19))
    enter_decoder_replay(metadata, plan)
    assert metadata.csa2_positions[-6:].tolist() == list(range(97, 103))
    exit_decoder_replay(metadata, plan)
    assert metadata.seq_lens.tolist() == [10, 3, 6]
    assert metadata.kv_cache_params.num_cached_tokens_per_seq == [0, 3, 100]


@pytest.mark.parametrize("peer_tokens", [0, 128])
def test_final_window_zero_queries_still_publish_global_and_restore_outputs(peer_tokens):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.decoder_replay import (
        plan_decoder_replay,
    )

    model, _ = _make_policy_case()
    model.decoder_replay_split = 20
    model.ced_kv_precompute = True
    layer, state = _make_boundary(129)
    model.layers[20] = layer
    metadata = _ReplayMetadata([129], num_contexts=1)
    metadata.decoder_context_ends = (512,)
    plan = plan_decoder_replay(metadata, 128)
    plan.saved_all_rank_num_tokens = [129, 257]
    plan.replay_all_rank_num_tokens = [0, peer_tokens]
    positions = torch.arange(129)
    _, _, replay_state = model._enter_bounded_replay(plan, metadata, positions, positions, state)
    assert replay_state.residual.shape == (0, 4, 64)
    layer.self_attn.prepare_global_cache.assert_called_once()
    assert layer.self_attn.prepare_global_cache.call_args.args[0].shape[0] == 129
    assert metadata.csa2_precomputed_kv_layers == {20}
    assert metadata.prepare_calls == 0
    assert metadata.all_rank_num_tokens == [0, peer_tokens]
    restored = model._exit_bounded_replay(plan, metadata, replay_state.residual[:, 0])
    assert restored.shape == (129, 64)
    assert not restored.count_nonzero()
    assert metadata.csa2_decoder_capture_lens == (0,)
    assert metadata.all_rank_num_tokens == [129, 257]
    assert metadata.seq_lens.tolist() == [129]
    assert metadata.kv_cache_params.num_cached_tokens_per_seq == [0]


def test_final_window_prefill_graph_capability_keeps_intermediate_history():
    from tensorrt_llm._torch.attention.backends.sparse.csa2.decoder_replay import (
        plan_decoder_replay,
    )

    metadata = _ReplayMetadata([257], num_contexts=1)
    reqs = [SimpleNamespace(prompt_len=2048, py_ced_replay=None)]
    CSA2TrtllmMetadata.prepare_context_replay(metadata, reqs, allow_final_window=False)
    plan = plan_decoder_replay(metadata, 128, reqs, private_decoder=True)
    assert plan.replay_seq_lens.tolist() == [128]
    assert plan.rows.tolist() == list(range(129, 257))
    assert plan.swa_floors == [129]


@pytest.mark.parametrize(
    "attention_dp,counts,collective",
    [(False, None, False), (True, [0, 0], False), (True, [0, 128], True)],
)
def test_empty_decoder_layer_participates_only_in_required_collectives(
    attention_dp, counts, collective
):
    layer = v41.DeepseekV41DecoderLayer.__new__(v41.DeepseekV41DecoderLayer)
    nn.Module.__init__(layer)
    layer.mapping = SimpleNamespace(enable_attention_dp=attention_dp)
    layer.forward_MoE = Mock()
    state = HCState.resolved(torch.empty(0, 4, 64), pre_mix=torch.empty(0, 4, 1))
    result = layer(
        position_ids=torch.empty(1, 0, dtype=torch.int32),
        hc_state=state,
        attn_metadata=SimpleNamespace(all_rank_num_tokens=counts),
        input_ids=torch.empty(0, dtype=torch.int32),
    )
    assert result.residual is state.residual
    assert layer.forward_MoE.call_count == int(collective)
    if collective:
        assert layer.forward_MoE.call_args.kwargs["hidden_states"].shape == (0, 64)


def test_empty_decoder_moe_runs_expert_communication_without_local_gemms():
    from tensorrt_llm._torch.models.modeling_deepseekv4 import DeepseekV4MoE
    from tensorrt_llm._torch.utils import EventType

    module = DeepseekV4MoE.__new__(DeepseekV4MoE)
    nn.Module.__init__(module)
    module.use_dp = True
    module.mapping = SimpleNamespace(tp_size=2)
    module.gate = DeepseekV4Gate(
        hidden_size=5120,
        num_experts=256,
        top_k=8,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        is_hashed=False,
        dtype=torch.bfloat16,
    )
    hidden = torch.empty(0, 5120, dtype=torch.bfloat16)
    module.experts = Mock(return_value=hidden.clone())
    module.shared_experts = Mock(side_effect=AssertionError("empty shared expert GEMM"))
    module.event_dict = {EventType.Main: None, EventType.MoeShared: None}
    module.aux_stream = None
    with patch(
        "torch.ops.trtllm.dsv3_router_gemm_op", side_effect=AssertionError("empty router GEMM")
    ):
        result = module(hidden, all_rank_num_tokens=[0, 7])
    assert result.shape == hidden.shape
    assert result.dtype == hidden.dtype
    module.experts.assert_called_once()
    assert module.experts.call_args.args[1].shape == (0, 256)
    assert module.experts.call_args.args[1].dtype == torch.float32
    assert module.experts.call_args.kwargs["all_rank_num_tokens"] == [0, 7]


def _make_boundary(tokens: int) -> tuple[SimpleNamespace, HCState]:
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Mode

    residual = torch.randn(tokens, 4, 64, dtype=torch.bfloat16)
    pre = torch.randn(tokens, 4, 1)
    norm = torch.nn.RMSNorm(64, eps=1e-6, dtype=torch.bfloat16)
    norm.variance_epsilon = norm.eps

    def publish(values, metadata):
        assert values.shape[0] == int(metadata.seq_lens.sum())
        metadata.csa2_precomputed_kv_layers.add(20)

    attention = SimpleNamespace(
        layer=SimpleNamespace(kv_source=20, compress_ratio=1, mode=CSA2Mode.FULL),
        prepare_global_cache=Mock(side_effect=publish),
    )
    return SimpleNamespace(
        layer_idx=20,
        engram=None,
        input_layernorm=norm,
        self_attn=attention,
        _decoder_global_input=lambda state: norm(v41.mHC.collapse(state.residual, state.pre_mix)),
    ), HCState.resolved(residual, pre_mix=pre)


def test_model_boundary_preserves_full_cache_and_compacts_query_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", raising=False)
    torch.manual_seed(4120)
    lengths = [129, 7, 6]
    layer, state = _make_boundary(sum(lengths))
    metadata = _ReplayMetadata(lengths, num_contexts=2)
    model, config = _make_policy_case()
    policy = DeepseekV41Model._resolve_replay_policy(model, config)
    assert policy == (20, 128)
    assert model.ced_kv_precompute
    model.decoder_replay_split, model.decoder_replay_window = policy
    model._decoder_replay_observed = False
    model._decoder_replay_reuse_warning_emitted = False
    model.layers[20] = layer
    split, plan = DeepseekV41Model._plan_bounded_replay(
        model, metadata, all_token_states_required=False
    )
    assert split == 20
    assert plan is not None
    assert plan.rows.tolist() == list(range(1, sum(lengths)))
    positions = torch.arange(sum(lengths))
    ids = positions + 100
    new_positions, new_ids, new_state = model._enter_bounded_replay(
        plan, metadata, positions, ids, state
    )
    assert metadata.csa2_precomputed_kv_layers == {20}
    assert metadata.csa2_replay_query_rows is plan.rows
    writer = layer.self_attn.prepare_global_cache
    writer.assert_called_once()
    actual, passed_metadata = writer.call_args.args
    collapsed = state.residual[:, 0].float() * state.pre_mix[:, 0]
    for head in range(1, 4):
        collapsed += state.residual[:, head].float() * state.pre_mix[:, head]
    expected = layer.input_layernorm(collapsed.to(state.residual.dtype))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.shape[0] == sum(lengths) and passed_metadata is metadata
    torch.testing.assert_close(new_positions, positions[plan.rows])
    torch.testing.assert_close(new_ids, ids[plan.rows])
    torch.testing.assert_close(new_state.residual, state.residual[plan.rows])
    torch.testing.assert_close(new_state.pre_mix, state.pre_mix[plan.rows])
    assert metadata.seq_lens.tolist() == [128, 7, 6]
    assert metadata.kv_cache_params.num_cached_tokens_per_seq == [1, 0, 0]
    # A later request must not see the cache-only marker from this forward.
    restored = model._exit_bounded_replay(plan, metadata, new_state.residual)
    assert restored.shape == state.residual.shape
    assert not restored[0].any()
    torch.testing.assert_close(restored[plan.rows], new_state.residual)
    assert metadata.seq_lens.tolist() == lengths
    assert metadata.kv_cache_params.num_cached_tokens_per_seq == [0, 0, 0]
    assert metadata.csa2_replay_query_rows is None
    assert not metadata.csa2_precomputed_kv_layers
    metadata.prepare()
    assert not metadata.csa2_precomputed_kv_layers
    assert not metadata.csa2_indices
    assert metadata.seq_lens.tolist() == lengths
    assert metadata.kv_cache_params.num_cached_tokens_per_seq == [0, 0, 0]
    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "0")
    assert DeepseekV41Model._resolve_replay_policy(model, config) == (None, 0)


@pytest.mark.parametrize("ced_precompute", [False, True])
@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("context_only,attention_dp", [(False, False), (True, False), (True, True)])
def test_remote_tail_source_reuses_boundary_collapse(
    monkeypatch: pytest.MonkeyPatch,
    ced_precompute: bool,
    deferred: bool,
    context_only: bool,
    attention_dp: bool,
) -> None:
    """Keep cache writes and source outputs exact with one collapse per forward."""
    torch.manual_seed(4121)
    model = DeepseekV41Model.__new__(DeepseekV41Model)
    nn.Module.__init__(model)
    model.model_config = SimpleNamespace(
        mapping=Mapping(world_size=4, tp_size=4, enable_attention_dp=attention_dp)
    )
    model.use_engram = False
    model.hc_mult = 4
    model.num_hidden_layers = 2
    model.decoder_replay_split = 1
    model.disagg_remote_tail_replay = True
    model.disagg_context_only = context_only
    model.ced_kv_precompute = ced_precompute
    collapse = Mock(wraps=v41.mHC.collapse)
    finalize = Mock(wraps=model._finalize_hc_state)
    monkeypatch.setattr(v41.mHC, "collapse", collapse)
    monkeypatch.setattr(model, "_finalize_hc_state", finalize)

    # Change both the token count and values between requests. No tensor from
    # the first boundary may leak into a later forward on the same model.
    for lengths, num_contexts in (([129, 3], 2), ([5], 1), ([1], 0)):
        layer, resolved = _make_boundary(sum(lengths))
        metadata = _ReplayMetadata(lengths, num_contexts=num_contexts)
        metadata.mapping = model.model_config.mapping
        metadata.csa2_remote_tail_mode = None if context_only else "source"
        metadata.begin_model_forward = Mock()
        post_mapping = Mock(return_value=resolved.residual)
        state = resolved
        if deferred:
            state = HCState.deferred(
                residual=resolved.residual - 1,
                post_mix=torch.ones(sum(lengths), 4, 1),
                comb_mix=torch.eye(4).expand(sum(lengths), 4, 4),
                x_prev=torch.ones(sum(lengths), 64, dtype=torch.bfloat16),
                pre_mix=resolved.pre_mix,
            )
        encoder = Mock(return_value=state)
        encoder.hc_attn = SimpleNamespace(post_mapping=post_mapping)
        model.layers = [encoder, layer]
        residual_before = state.residual.clone()
        pre_before = state.pre_mix.clone()

        result = model.forward(
            metadata, inputs_embeds=torch.zeros(sum(lengths), 64, dtype=torch.bfloat16)
        )

        expected = resolved.residual[:, 0].float() * resolved.pre_mix[:, 0]
        for stream in range(1, 4):
            expected += resolved.residual[:, stream].float() * resolved.pre_mix[:, stream]
        expected = expected.to(resolved.residual.dtype)
        torch.testing.assert_close(result, expected, atol=0, rtol=0)
        torch.testing.assert_close(state.residual, residual_before, atol=0, rtol=0)
        torch.testing.assert_close(state.pre_mix, pre_before, atol=0, rtol=0)
        collapse.assert_called_once()
        assert finalize.call_count == int(not ced_precompute)
        assert post_mapping.call_count == int(deferred)
        if deferred:
            post_args = post_mapping.call_args.kwargs
            assert post_args["x"] is state.x_prev
            assert post_args["residual"] is state.residual
            assert post_args["post_layer_mix"] is state.post_mix
            assert post_args["comb_res_mix"] is state.comb_mix
        encoder.assert_called_once()
        metadata.begin_model_forward.assert_called_once()
        writer = layer.self_attn.prepare_global_cache
        if ced_precompute:
            writer.assert_called_once()
            actual, passed_metadata = writer.call_args.args
            torch.testing.assert_close(actual, layer.input_layernorm(expected), atol=0, rtol=0)
            assert passed_metadata is metadata
            assert metadata.csa2_precomputed_kv_layers == {20}
        else:
            writer.assert_not_called()
            assert not metadata.csa2_precomputed_kv_layers
        collapse.reset_mock()
        finalize.reset_mock()


@pytest.mark.parametrize("attention_dp", [False, True])
def test_remote_tail_destination_keeps_decoder_state(attention_dp: bool) -> None:
    """Only decoder SWA visibility changes; query rows and generation history stay intact."""
    model, config = _make_policy_case()
    config.mapping.enable_attention_dp = attention_dp
    layer, state = _make_boundary(129)
    model.layers[20] = layer
    model.decoder_replay_split = 20
    model.ced_kv_precompute = True
    metadata = _ReplayMetadata([128, 1], num_contexts=1)
    metadata.mapping = config.mapping
    metadata.num_generations = 1
    metadata.kv_cache_params.num_cached_tokens_per_seq = [1024, 63]
    metadata.prepare()
    metadata.csa2_remote_tail_mode = "destination"
    metadata.csa2_remote_tail_starts = [1024]
    saved_lengths = metadata.seq_lens.clone()
    saved_positions = metadata.csa2_positions.clone()
    metadata._ensure_swa_slots()  # As the encoder's first layer does.
    saved_indices = metadata.csa2_swa_indices[0].clone()
    saved_residual = state.residual.clone()
    saved_pre_mix = state.pre_mix.clone()

    result, decoder_state = model._enter_remote_tail_boundary(metadata, state)

    assert result is None
    assert decoder_state is state
    layer.self_attn.prepare_global_cache.assert_called_once()
    assert metadata.csa2_precomputed_kv_layers == {20}
    torch.testing.assert_close(state.residual, saved_residual, atol=0, rtol=0)
    torch.testing.assert_close(state.pre_mix, saved_pre_mix, atol=0, rtol=0)
    torch.testing.assert_close(metadata.seq_lens, saved_lengths)
    torch.testing.assert_close(metadata.csa2_positions, saved_positions)
    assert metadata.kv_cache_params.num_cached_tokens_per_seq == [1024, 63]
    assert metadata.csa2_replay_query_rows is None
    assert metadata.prepare_calls == 1
    expected_indices = saved_indices.clone()
    expected_indices[:128].masked_fill_(expected_indices[:128] < 1024, -1)
    metadata._ensure_swa_slots()  # As the first decoder layer does.
    torch.testing.assert_close(metadata.csa2_swa_indices[0], expected_indices)


@pytest.mark.parametrize(
    "remote_mode,is_cuda_graph", [(None, False), (None, True), ("destination", False)]
)
def test_remote_tail_attention_dp_generation_keeps_all_layers(
    monkeypatch: pytest.MonkeyPatch,
    remote_mode: str | None,
    is_cuda_graph: bool,
) -> None:
    """Decode/dummy ranks and tail ranks participate in the same layer collectives."""
    model, _ = _make_policy_case()
    model.model_config = SimpleNamespace(
        mapping=Mapping(world_size=4, tp_size=4, enable_attention_dp=True)
    )
    model.use_engram = False
    model.hc_mult = 4
    model.decoder_replay_split = 20
    model.decoder_replay_window = 128
    model.disagg_remote_tail_replay = True
    model.disagg_context_only = False
    num_tokens = 128 if remote_mode == "destination" else 1
    boundary, state = _make_boundary(num_tokens)
    model.layers = [Mock(return_value=state) for _ in range(model.num_hidden_layers)]
    model.layers[20] = Mock(
        return_value=state,
        layer_idx=boundary.layer_idx,
        engram=boundary.engram,
        self_attn=boundary.self_attn,
        input_layernorm=boundary.input_layernorm,
        _decoder_global_input=boundary._decoder_global_input,
    )
    model.ced_kv_precompute = True
    metadata = _ReplayMetadata([num_tokens], num_contexts=int(remote_mode == "destination"))
    metadata.mapping = model.model_config.mapping
    metadata.is_cuda_graph = is_cuda_graph
    metadata.csa2_remote_tail_mode = remote_mode
    metadata.csa2_remote_tail_starts = [1024] if remote_mode == "destination" else []
    metadata.begin_model_forward = Mock()
    enter_decoder = Mock()
    monkeypatch.setattr(v41, "enter_remote_tail_decoder", enter_decoder)

    result = model.forward(
        metadata,
        inputs_embeds=torch.zeros(num_tokens, 64, dtype=torch.bfloat16),
        all_token_states_required=False,
    )

    assert result.shape == (num_tokens, 64)
    for layer in model.layers:
        layer.assert_called_once()
        assert layer.call_args.kwargs["hc_state"].residual.shape[0] == num_tokens
    assert enter_decoder.call_count == int(remote_mode == "destination")
    assert boundary.self_attn.prepare_global_cache.call_count == int(remote_mode == "destination")
    assert metadata.csa2_replay_query_rows is None
    metadata.begin_model_forward.assert_called_once()


def test_context_decoder_skipping_requires_bounded_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRTLLM_DISAGG_ROLE", "context")
    assert not v41_config.disagg_context_decoder_skipping_enabled(False)

    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "1")
    assert v41_config.disagg_context_decoder_skipping_enabled(True)

    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "0")
    assert not v41_config.disagg_context_decoder_skipping_enabled(True)


@pytest.mark.parametrize(
    "role,remote,bounded,expected",
    [
        ("context", "1", "1", True),
        ("context", "0", "1", False),
        ("context", "1", "0", False),
        ("generation", "1", "1", False),
        (None, "1", "1", False),
    ],
)
def test_context_decoder_skipping_requires_explicit_context_role(
    monkeypatch, role, remote, bounded, expected
):
    for name, value in (
        ("TRTLLM_DISAGG_ROLE", role),
        ("TRTLLM_V41_DECODER_BOUNDED_REPLAY", bounded),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    assert v41_config.disagg_context_decoder_skipping_enabled(remote == "1") is expected


def test_context_only_logits_do_not_call_omitted_output_head():
    processor = v41._ContextOnlyLogitsProcessor(vocab_size=37)
    metadata = SimpleNamespace(seq_lens_cuda=torch.tensor([6, 2, 9], dtype=torch.int32))
    hidden_states = torch.full((17, 64), torch.nan, dtype=torch.bfloat16)
    head = v41._OmittedContextDecoder()
    logits = processor(hidden_states, head, metadata)
    assert logits.shape == (3, 37)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()
    assert torch.count_nonzero(logits) == 0
    with pytest.raises(ValueError, match="context logits"):
        processor(hidden_states, head, metadata, return_context_logits=True)
    with pytest.raises(RuntimeError, match="context-only worker"):
        head(hidden_states)


@pytest.mark.parametrize("mode", [None, "source", "destination"])
def test_context_only_forward_routes_warmup_to_source_and_refuses_destination(monkeypatch, mode):
    model = DeepseekV41Model.__new__(DeepseekV41Model)
    nn.Module.__init__(model)
    model.disagg_context_only = True
    metadata = SimpleNamespace(
        csa2_remote_tail_mode=mode, begin_model_forward=Mock(), all_rank_num_tokens=None
    )
    result = object()

    def source_forward(self, passed_metadata):
        assert self is model
        assert passed_metadata is metadata
        assert metadata.csa2_remote_tail_mode == "source"
        return result

    monkeypatch.setattr(v41.DeepseekV4Model, "forward", source_forward)
    if mode == "destination":
        with pytest.raises(ValueError, match="remote-tail destination"):
            model.forward(metadata)
    else:
        assert model.forward(metadata) is result
    metadata.begin_model_forward.assert_called_once()


def test_context_only_model_rejects_speculative_decoding_before_construction(monkeypatch):
    monkeypatch.setenv("TRTLLM_DISAGG_ROLE", "context")
    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "1")
    config = SimpleNamespace(
        mapping=Mapping(),
        extra_attrs={"bounded_replay_on_generation": True},
        spec_config=SimpleNamespace(spec_dec_mode=SimpleNamespace(is_dspark=lambda: True)),
    )
    with pytest.raises(ValueError, match="speculative decoding to be disabled"):
        DeepseekV41ForCausalLM(config)


# Forward execution and tensor parallelism

# The small topology covers SWA, pooled owners, reuse and reindex layers. `num_hidden_layers` equals
# `len(compress_ratios)` because MTP is not constructed for V4.1 yet, so an MTP
# tail here would only add unverified descriptor rows.
V41_TINY_RATIOS = [0, 0, 2, 2, 2, 1, 1, 1]
V41_TINY_KV_SOURCES = [2, 5]
V41_TINY_INDEX_SOURCES = [2, 5, 7]

# Attention geometry is the released one -- 512-wide latent head, 64 of it rope --
# because the compressor and indexer kernels are specialized on those widths.
# Everything that only costs memory (vocab, expert width, hidden size) is shrunk.
# `n_routed_experts` stays at the released 384: the routing kernel is specialized
# on the expert count, so shrinking it would test a topology V4.1 never runs.
V41_TINY_TEXT_CONFIG = {
    "vocab_size": 1024,
    "hidden_size": 2048,
    "moe_intermediate_size": 128,
    "num_hidden_layers": len(V41_TINY_RATIOS),
    "num_attention_heads": 16,
    "num_key_value_heads": 1,
    "head_dim": 512,
    "qk_rope_head_dim": 64,
    "q_lora_rank": 512,
    "o_lora_rank": 512,
    "o_groups": 8,
    "max_position_embeddings": 65536,
    "n_routed_experts": 384,
    "n_shared_experts": 1,
    "num_experts_per_tok": 6,
    "compress_ratios": V41_TINY_RATIOS,
    "kv_source_layer_ids": V41_TINY_KV_SOURCES,
    "index_source_layer_ids": V41_TINY_INDEX_SOURCES,
    "index_n_heads": 16,
    "index_head_dim": 128,
    "index_topk": 512,
    "sliding_window": 128,
    # The two-level candidate prefilter is a long-context-only path (it is a
    # mathematical no-op until a layer has more compressed entries than
    # `index_topk`), and layer 20 does not exist here. `None` turns it off
    # explicitly rather than leaving an out-of-range layer id in the descriptors.
    "candidate_source_layer_id": None,
    # Engram is off: its tables are sized by prime search over a 16M vocab and
    # nothing in the sparse-attention path depends on them.
    "engram_layer_ids": [],
    "engram_num_embeddings": [],
    # V4.1's three MTP layers are heterogeneous and not constructed yet;
    # `DeepseekV41ForCausalLM` refuses a `spec_config` outright.
    "num_nextn_predict_layers": 0,
    "dspark_target_layer_ids": [],
    "rope_scaling": {
        "rope_type": "yarn",
        "factor": 4.0,
        "beta_fast": 32,
        "beta_slow": 1,
        "original_max_position_embeddings": 65536,
    },
}


def _tiny_v41_model_config():
    """Reduced topology for constructor-only checks, retaining FP8 dense weights."""
    config = DeepseekV41Config(text_config=deepcopy(V41_TINY_TEXT_CONFIG))
    config.dtype = torch.bfloat16
    config.tie_word_embeddings = False
    config.mapping = Mapping(world_size=1, tp_size=1, rank=0)

    sparse_attn_config = CSA2SparseAttentionConfig()
    config.sparse_attention_config = sparse_attn_config

    model_config = ModelConfig(
        pretrained_config=config,
        sparse_attention_config=sparse_attn_config,
        attn_backend="TRTLLM",
        quant_config=ModelConfig._build_deepseek_v41_quant_config(RELEASE_QUANT_CONFIG),
    )
    return model_config, sparse_attn_config


def _build_context_only_inventory_model(
    *, stale_cache_limit=None, bounded_replay_on_generation=True
):
    """Construct the real layer classes with small, non-forwarded expert tensors."""
    config, _ = _tiny_v41_model_config()
    config.extra_attrs["bounded_replay_on_generation"] = bounded_replay_on_generation
    # Direct construction bypasses from_pretrained's architecture-aware backend
    # selection; CUTLASS cannot construct FP8 block-scale experts on SM100/SM103.
    config.moe_backend = ModelConfig.resolve_moe_backend(
        "AUTO", "DeepseekV41ForCausalLM", config.quant_config
    )
    text = config.pretrained_config.text_config
    text.hidden_size = 512
    text.q_lora_rank = 128
    text.o_lora_rank = 128
    text.n_routed_experts = 8
    text.max_position_embeddings = 2048
    if stale_cache_limit is not None:
        config.extra_attrs["csa2_context_swa_layer_limit"] = stale_cache_limit
    return DeepseekV41ForCausalLM(config)


@requires_cuda
def test_context_only_model_keeps_exact_prefix_and_boundary_parameter_inventory(
    monkeypatch,
):
    monkeypatch.setenv("TRTLLM_DISAGG_ROLE", "generation")
    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "1")
    full_model = _build_context_only_inventory_model()
    full_inventory = {
        name: (tuple(parameter.shape), parameter.dtype, parameter.numel())
        for name, parameter in full_model.named_parameters()
    }
    split = full_model.model.decoder_replay_split
    assert split == 5
    assert not full_model.model.disagg_context_only
    assert "csa2_context_swa_layer_limit" not in full_model.model_config.extra_attrs
    del full_model

    monkeypatch.setenv("TRTLLM_DISAGG_ROLE", "context")
    model = _build_context_only_inventory_model()
    assert model.model.disagg_context_only
    assert model.model_config.extra_attrs["csa2_context_swa_layer_limit"] == split
    assert model.model.ced_kv_precompute
    assert len(model.model.layers) == model.config.num_hidden_layers == len(V41_TINY_RATIOS)
    retained_prefixes = (
        "model.embed_tokens.",
        *(f"model.layers.{index}." for index in range(split)),
        f"model.layers.{split}.input_layernorm.",
        f"model.layers.{split}.self_attn.compressor.",
        f"model.layers.{split}.self_attn.index_wk.",
        f"model.layers.{split}.self_attn.index_k_norm.",
    )
    expected = {
        name: details
        for name, details in full_inventory.items()
        if name.startswith(retained_prefixes)
    }
    actual = {
        name: (tuple(parameter.shape), parameter.dtype, parameter.numel())
        for name, parameter in model.named_parameters()
    }
    assert actual == expected
    assert sum(details[2] for details in actual.values()) < sum(
        details[2] for details in full_inventory.values()
    )
    assert not list(model.model.norm.parameters())
    assert not list(model.lm_head.parameters())
    assert all(not list(layer.parameters()) for layer in model.model.layers[split + 1 :])
    boundary = model.model.layers[split]
    assert boundary.self_attn.compressor is not None
    assert boundary.self_attn.index_wk is not None
    assert isinstance(boundary.self_attn.index_k_norm, nn.Module)
    assert hasattr(boundary.self_attn, "rotary_emb")
    assert not hasattr(boundary.self_attn, "wq_a")
    assert not hasattr(boundary, "mlp")
    assert len(model._post_load_decoder_layers()) == split
    description = model.model.describe_layers()
    assert description["num_constructed"] == split
    assert description["not_constructed"] == list(range(split, len(V41_TINY_RATIOS)))


@requires_cuda
@pytest.mark.parametrize(
    "role,remote,bounded",
    [
        ("generation", "1", "1"),
        (None, "1", "1"),
        ("context", "0", "1"),
        ("context", "1", "0"),
    ],
)
def test_context_only_inactive_modes_preserve_all_model_weights(monkeypatch, role, remote, bounded):
    if role is None:
        monkeypatch.delenv("TRTLLM_DISAGG_ROLE", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_DISAGG_ROLE", role)
    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", bounded)
    if bounded == "0" and remote == "1":
        with pytest.raises(ValueError, match="requires supported decoder bounded replay"):
            _build_context_only_inventory_model(
                stale_cache_limit=1, bounded_replay_on_generation=True
            )
        return
    model = _build_context_only_inventory_model(
        stale_cache_limit=1, bounded_replay_on_generation=remote == "1"
    )
    assert not model.model.disagg_context_only
    assert "csa2_context_swa_layer_limit" not in model.model_config.extra_attrs
    assert len(model.model.layers) == len(V41_TINY_RATIOS)
    assert all(list(layer.parameters()) for layer in model.model.layers)
    assert list(model.model.norm.parameters())
    assert list(model.lm_head.parameters())
    assert len(model._post_load_decoder_layers()) == len(V41_TINY_RATIOS)
    assert model.model.describe_layers()["num_constructed"] == len(V41_TINY_RATIOS)


def _assert_v41_module_topology(model) -> None:
    """Check compressor/indexer ownership and shared-key routing for every tiny-model layer."""
    for layer_idx, ratio in enumerate(V41_TINY_RATIOS):
        attn = model.model.layers[layer_idx].self_attn
        expect_compressor = layer_idx in V41_TINY_KV_SOURCES
        expect_indexer = layer_idx in V41_TINY_INDEX_SOURCES
        assert (attn.compressor is not None) == expect_compressor, (
            f"layer {layer_idx} (ratio {ratio}) compressor presence is wrong"
        )
        assert hasattr(attn, "index_wq_b") == expect_indexer, (
            f"layer {layer_idx} (ratio {ratio}) indexer presence is wrong"
        )
        if expect_indexer:
            # `owns_index_keys` is what decides whether the indexer projects its
            # own keys from a latent or reads a published set. Layer 7 is the
            # only one here where it differs from `is_index_source`.
            assert hasattr(attn, "index_wk") == expect_compressor
            expected_source = max(i for i in V41_TINY_KV_SOURCES if i <= layer_idx)
            assert attn.layer.kv_source == expected_source, (
                f"layer {layer_idx} scores against the wrong index-key cache"
            )


# V4.1 omits V4's per-head query normalization.


@pytest.mark.skip_less_device_memory(40000)
@skip_blackwell_geforce
def test_v41_attention_topology_has_no_per_head_query_norm() -> None:
    model_config, _ = _tiny_v41_model_config()
    model_config.moe_backend = ModelConfig.resolve_moe_backend(
        "AUTO", "DeepseekV41ForCausalLM", model_config.quant_config
    )
    model = DeepseekV41ForCausalLM(model_config).to(torch.device("cuda"))
    _assert_v41_module_topology(model)
    model.model.describe_layers()
    offenders = [
        idx
        for idx, layer in enumerate(model.model.layers)
        if getattr(layer.self_attn, "q_b_layernorm", None) is not None
    ]
    assert not offenders, (
        f"layers {offenders} built a per-head query norm; V4.1 removed it, and an "
        f"inherited one costs ~0.52x on every query"
    )
    attn = model.model.layers[0].self_attn
    assert (attn.qk_head_dim, attn.qk_nope_head_dim, attn.qk_rope_head_dim) == (512, 448, 64)
    assert attn.kv_norm.weight.shape == (512,)
    assert attn.wq_b.out_features == model_config.pretrained_config.num_attention_heads * 512
    del model
    torch.cuda.empty_cache()

    assert DeepseekV41Attention.q_b_norm_enabled is False
    # V4 keeps it: the flag defaults to the base class's behaviour so that a
    # subclass which forgets to state it stays V4-correct.
    assert DeepseekV4Attention.q_b_norm_enabled is True

    assert not issubclass(DeepseekV41Attention, DeepseekV4Attention)


@pytest.mark.parametrize("strategy", ["AUTO", "NCCL"])
def test_attention_adapter_preserves_model_allreduce_strategy(strategy):
    from tensorrt_llm._torch.distributed import AllReduceStrategy
    from tensorrt_llm._utils import mpi_rank, mpi_world_size

    if mpi_world_size() != 2:
        pytest.skip("Requires two MPI ranks")
    torch.cuda.set_device(mpi_rank())
    original, sparse = _tiny_v41_model_config()
    requested = AllReduceStrategy.AUTO if strategy == "AUTO" else AllReduceStrategy.NCCL
    config = ModelConfig(
        pretrained_config=original.pretrained_config,
        sparse_attention_config=sparse,
        mapping=Mapping(world_size=2, rank=mpi_rank(), tp_size=2),
        attn_backend="TRTLLM",
        quant_config=original.quant_config,
        allreduce_strategy=requested,
    )
    attention = DeepseekV41Attention(config, layer_idx=0)
    assert attention.o_b_proj.all_reduce is not None
    assert attention.o_b_proj.all_reduce.strategy == requested
    assert not attention.o_b_proj.use_fused_gemm_allreduce


def test_model_boundary_resets_routing_and_preserves_layer_guard(monkeypatch):
    """Test only the model boundary with real routing and a controlled body."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout
    from tensorrt_llm._torch.models.modeling_deepseekv4 import DeepseekV4Model
    from tensorrt_llm._torch.models.modeling_deepseekv41 import DeepseekV41Model

    metadata = CSA2TrtllmMetadata.__new__(CSA2TrtllmMetadata)
    metadata.reset_routing()
    layout = CSA2Layout((1, 1), (0,), (0,))
    model = DeepseekV41Model.__new__(DeepseekV41Model)
    torch.nn.Module.__init__(model)
    model.disagg_context_only = False
    argument = object()
    outputs = []

    def body(self, attn_metadata, value, *, check_value):
        assert self is model and attn_metadata is metadata
        assert value is argument and check_value is argument
        assert not metadata.csa2_indices and not metadata.csa2_candidates
        assert not metadata.csa2_precomputed_kv_layers
        assert metadata.csa2_replay_query_rows is None
        metadata.enter_layer(layout.layer(0))
        producer = torch.tensor([len(outputs)])
        metadata.csa2_indices[0] = producer
        metadata.csa2_candidates[0] = producer
        metadata.enter_layer(layout.layer(1))
        assert metadata.csa2_indices[0] is producer
        with pytest.raises(ValueError, match="reordered layers"):
            metadata.enter_layer(layout.layer(0))
        outputs.append(producer)
        return producer

    monkeypatch.setattr(DeepseekV4Model, "forward", body)
    for _ in range(3):
        old_indices = metadata.csa2_indices
        metadata.csa2_precomputed_kv_layers = {0}
        metadata.csa2_replay_query_rows = torch.tensor([1])
        result = model.forward(metadata, argument, check_value=argument)
        assert metadata.csa2_indices is not old_indices
        assert result is outputs[-1]
    assert [output.item() for output in outputs] == [0, 1, 2]


# Dense FP8 requantization

_BLOCK = v41._V41_REQUANT_BLOCK
_E4M3_MAX = v41._E4M3_MAX
_SUBNORMAL_STEP = 2.0**v41._E4M3_SUBNORMAL_FLOOR_EXP


# Synthetic weights have a heavier subnormal tail than checkpoint weights.
_MAX_INEXACT_FRACTION = 1e-2


def _reconstruct(q: torch.Tensor, scale: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
    """Reconstruct 128x128 block scales on CPU, including partial edge tiles."""
    m, n = q.shape
    expanded = scale.repeat_interleave(_BLOCK, dim=0).repeat_interleave(_BLOCK, dim=1)
    return (q.float() * expanded[:m, :n]).to(dtype)


def _fp8_32_weight(tiles_m: int, tiles_n: int, spread: float, seed: int = 0) -> torch.Tensor:
    """BF16 reference from E4M3 values with per-32x32-block power-of-two scales."""
    g = torch.Generator().manual_seed(seed)
    m, n = tiles_m * _BLOCK, tiles_n * _BLOCK
    base = torch.randn(m, n, generator=g)

    blocks = base.reshape(m // 32, 32, n // 32, 32)
    amax = blocks.abs().amax(dim=(1, 3), keepdim=True)
    # Keep amax below 448 so non-power-of-two scale regressions remain observable.
    base = (blocks / amax * _E4M3_MAX * 0.93).reshape(m, n)
    q = base.to(torch.float8_e4m3fn).float()

    nb_m, nb_n = m // 32, n // 32
    exps = torch.arange(nb_m * nb_n, dtype=torch.float32) % (spread + 1)
    exps = exps.reshape(nb_m, nb_n) - spread  # keep magnitudes modest
    s32 = torch.exp2(exps).repeat_interleave(32, dim=0).repeat_interleave(32, dim=1)
    return (q * s32).to(torch.bfloat16)


@pytest.mark.parametrize("spread", [0.0, 7.0])
def test_dense_fp8_requantization_preserves_dtype_and_error_bound(spread: float) -> None:
    """Requantization preserves exact values or only flushes E4M3 subnormals."""
    w = _fp8_32_weight(2, 3, spread)
    q, s = v41._requantize_block128(w)
    assert q.dtype == torch.float8_e4m3fn
    assert s.dtype == torch.float32
    assert torch.isfinite(q.float()).all()
    assert q.float().abs().max() <= _E4M3_MAX
    assert torch.equal(torch.log2(s), torch.log2(s).round())
    back = _reconstruct(q, s)
    if spread == 0:
        torch.testing.assert_close(back, w, rtol=0, atol=0)
    assert int((back != w).sum()) / w.numel() <= _MAX_INEXACT_FRACTION
    err = (back.float() - w.float()).abs()
    step = s.repeat_interleave(_BLOCK, dim=0).repeat_interleave(_BLOCK, dim=1)
    step = step[: w.shape[0], : w.shape[1]] * _SUBNORMAL_STEP
    assert torch.all(err <= step)


def test_zero_tile_is_handled() -> None:
    """An all-zero tile has amax == 0, where ``log2`` is ``-inf``."""
    w = _fp8_32_weight(2, 2, 0.0)
    w[:_BLOCK, :_BLOCK] = 0.0
    q, s = v41._requantize_block128(w)
    assert torch.isfinite(s).all(), f"non-finite scale from a zero tile: {s}"
    assert s[0, 0] == 1.0
    assert torch.equal(_reconstruct(q, s), w)


def test_partial_edge_tiles() -> None:
    """Pad both dimensions without changing scale reconstruction."""
    m, n = _BLOCK + 7, _BLOCK + 1
    w = torch.randn(m, n).to(torch.float8_e4m3fn).to(torch.bfloat16)
    q, s = v41._requantize_block128(w)
    assert q.shape == (m, n)
    assert s.shape == ((m + _BLOCK - 1) // _BLOCK, (n + _BLOCK - 1) // _BLOCK)
    assert torch.isfinite(s).all()
    assert torch.equal(_reconstruct(q, s), w)


def test_rejects_non_2d() -> None:
    with pytest.raises(ValueError, match="2-D"):
        v41._requantize_block128(torch.zeros(4, 4, 4))


def test_pow2_tile_scale_runs_on_meta_tensors() -> None:
    """Keep scale construction data-independent for meta-tensor loader audits."""
    s = v41._pow2_tile_scale(torch.zeros(4, 4, device="meta"))
    assert s.device.type == "meta" and s.shape == (4, 4)


# --------------------------------------------------------------------------------
# Load equivalence: the emitted pair, read back by the loader's own consumer.
# --------------------------------------------------------------------------------


def _safetensors_index() -> dict:
    with open(RELEASE_CHECKPOINT / "model.safetensors.index.json") as f:
        return json.load(f)["weight_map"]


def _read_tensor(shard: str, name: str) -> torch.Tensor:
    """One tensor out of one shard, without a safetensors dependency in the test."""
    from safetensors.torch import load_file

    return load_file(RELEASE_CHECKPOINT / shard)[name]


# Only grouped output and shared experts use block-128 requantization.
_EQUIVALENCE_STEMS = (
    "layers.1.attn.wo_a",
    "layers.1.ffn.shared_experts.w1",
    "layers.1.ffn.shared_experts.w2",
)


@requires_release_checkpoint
@requires_cuda
@pytest.mark.parametrize("stem", _EQUIVALENCE_STEMS)
def test_requantized_pair_matches_checkpoint_reference(stem: str) -> None:
    """The real loader reconstructs FP8 within the checkpoint's subnormal bound."""
    index = _safetensors_index()
    weight = _read_tensor(index[f"{stem}.weight"], f"{stem}.weight")
    scale = _read_tensor(index[f"{stem}.scale"], f"{stem}.scale")

    reference = v41._dequantize_block32(weight, scale)
    q, s = v41._requantize_dense_stem(weight, scale)

    assert q.dtype == torch.float8_e4m3fn
    assert q.shape == reference.shape
    assert s.shape == (
        (q.shape[0] + _BLOCK - 1) // _BLOCK,
        (q.shape[1] + _BLOCK - 1) // _BLOCK,
    )

    back = (
        weight_dequant(q.contiguous().cuda(), s.float().contiguous().cuda(), block_size=_BLOCK)
        .to(torch.bfloat16)
        .cpu()
    )

    err = (back.float() - reference.float()).abs()
    bad = int((back != reference).sum())
    if bad:
        step = s.repeat_interleave(_BLOCK, dim=0).repeat_interleave(_BLOCK, dim=1)
        step = step[: q.shape[0], : q.shape[1]] * _SUBNORMAL_STEP
        worst = err.max().item()
        assert torch.all(err <= step), (
            f"{stem}: {bad} of {reference.numel()} elements differ and the worst ({worst:.3e}) "
            f"exceeds one e4m3 subnormal step of its tile scale. That is not subnormal flush -- "
            f"suspect the scale convention (weight_scale_inv holds the multiplier, not its "
            f"reciprocal) or the 128 block extent"
        )
        assert bad / reference.numel() <= _MAX_INEXACT_FRACTION, (
            f"{stem}: {bad / reference.numel():.2e} of elements deviate, far above the "
            f"4.97e-04 worst case measured across the checkpoint"
        )


# Decoder reduction contracts


class _Norm(nn.Module):
    def __init__(self, *, hidden_size: int, eps: float, dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype), requires_grad=False)
        self.variance_epsilon = eps

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        values = hidden.float()
        return (
            values
            * torch.rsqrt(values.square().mean(-1, keepdim=True) + self.variance_epsilon)
            * self.weight.float()
        ).to(hidden.dtype)


class _Attention(nn.Module):
    def __init__(self, config: SimpleNamespace, *, reduce_output: bool, **kwargs: object) -> None:
        super().__init__()
        self.reduce_output = reduce_output
        self.calls: list[bool] = []

    def forward(
        self, *, hidden_states: torch.Tensor, all_reduce_params: AllReduceParams, **kwargs: object
    ) -> torch.Tensor:
        self.calls.append(all_reduce_params.enable_allreduce)
        assert all_reduce_params.enable_allreduce == self.reduce_output
        return hidden_states + 0.25


class _MoE(nn.Module):
    def __init__(self, **kwargs: object) -> None:
        super().__init__()
        self.calls: list[tuple[torch.Tensor, bool, bool]] = []

    def forward(
        self,
        hidden: torch.Tensor,
        hidden_fp4: None,
        *,
        final_all_reduce_params: AllReduceParams,
        do_finalize: bool,
        **kwargs: object,
    ) -> torch.Tensor:
        assert hidden_fp4 is None
        self.calls.append((hidden.clone(), final_all_reduce_params.enable_allreduce, do_finalize))
        return hidden + 1


class _MHC(nn.Module):
    def hc_coeffs(self, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mult = residual.shape[1]
        pre = torch.zeros((residual.shape[0], mult, 1), device=residual.device)
        pre[:, 0] = 1
        post = torch.ones_like(pre)
        comb = torch.eye(mult, device=residual.device).expand(residual.shape[0], -1, -1)
        return pre, post, comb

    @staticmethod
    def _norm(hidden: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
        # The production fused boundary folds the next RMSNorm into layer_input.
        if weight is None:
            return hidden
        values = hidden.float()
        return (
            values * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps) * weight.float()
        ).to(hidden.dtype)

    def pre_mapping_lagged(
        self,
        residual: torch.Tensor,
        pre: torch.Tensor,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        post = torch.ones_like(pre)
        comb = torch.eye(residual.shape[1], device=residual.device).expand(
            residual.shape[0], -1, -1
        )
        return pre, post, comb, self._norm(residual[:, 0, :], norm_weight, norm_eps)

    def post_mapping(
        self,
        *,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        return residual + x.unsqueeze(1)

    def fused_hc_lagged(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        pre: torch.Tensor,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        updated = self.post_mapping(x=x, residual=residual, post_layer_mix=post, comb_res_mix=comb)
        pre_own, post_own, comb_own = self.hc_coeffs(updated)
        return (
            updated,
            pre_own,
            post_own,
            comb_own,
            self._norm(updated[:, 0], norm_weight, norm_eps),
        )


def _build_layer(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    attention_dp: bool,
    eager_fusion: bool,
    hidden_size: int = 8,
) -> tuple[v41.DeepseekV41DecoderLayer, Mock]:
    """Run the production parent constructor, replacing hardware-heavy components."""
    monkeypatch.setenv("TRTLLM_DEEPSEEK_EAGER_FUSION_DISABLED", "0" if eager_fusion else "1")
    monkeypatch.setattr(v4, "can_access_peer", lambda mapping: False)
    monkeypatch.setattr(v4, "RMSNorm", _Norm)
    monkeypatch.setattr(v4, "DeepseekV4MoE", _MoE)
    monkeypatch.setattr(v4, "MoEAllReduce", lambda mapping: None)
    allreduce = Mock(side_effect=AssertionError("replicated mHC output must not be all-reduced"))
    monkeypatch.setattr(v4, "AllReduce", lambda **kwargs: allreduce)
    monkeypatch.setattr(v4, "_resolve_enable_fused_hc", lambda config: False)
    monkeypatch.setattr(v41.DeepseekV41DecoderLayer, "attention_cls", _Attention)
    monkeypatch.setattr(v41.DeepseekV41DecoderLayer, "_make_mhc", lambda self: _MHC())
    quant = SimpleNamespace(
        layer_quant_mode=SimpleNamespace(has_nvfp4=lambda: False), quant_algo=None
    )
    monkeypatch.setattr(
        v41.DeepseekV41DecoderLayer,
        "_get_decoder_layer_quant_config",
        lambda self, model_config, layer_idx: quant,
    )
    mapping = Mapping(world_size=tp_size, tp_size=tp_size, rank=0, enable_attention_dp=attention_dp)
    config = SimpleNamespace(
        hidden_size=hidden_size,
        moe_intermediate_size=16,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=1,
        rms_norm_eps=0.25,
        torch_dtype=torch.bfloat16,
    )
    model_config = SimpleNamespace(
        pretrained_config=config, mapping=mapping, allreduce_strategy=None
    )
    layer = v41.DeepseekV41DecoderLayer(model_config, layer_idx=0, aux_stream_dict={})
    return layer, allreduce


@pytest.mark.parametrize("tp_size,attention_dp", [(1, False), (4, False), (4, True)])
@pytest.mark.parametrize("input_normalized", [False, True])
def test_forward_moe_does_not_reduce_replicated_input(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    attention_dp: bool,
    input_normalized: bool,
) -> None:
    layer, allreduce = _build_layer(monkeypatch, tp_size, attention_dp, eager_fusion=True)
    requires_attention_reduction = tp_size > 1 and not attention_dp
    assert layer.self_attn.reduce_output == requires_attention_reduction
    assert layer.disable_attn_allreduce == (not requires_attention_reduction)
    assert layer.fusion_config.PRE_MOE_FUSION is False
    assert layer.fusion_config.POST_MOE_FUSION is False
    hidden = torch.arange(24, dtype=torch.bfloat16).reshape(3, 8) / 16
    metadata = SimpleNamespace(all_rank_num_tokens=[3] * tp_size)
    expected = (
        hidden.float() * torch.rsqrt(hidden.float().square().mean(-1, keepdim=True) + 0.25)
    ).to(torch.bfloat16)
    norm = Mock(wraps=layer.post_attention_layernorm.forward)
    monkeypatch.setattr(layer.post_attention_layernorm, "forward", norm)
    if input_normalized:
        output = layer.forward_MoE(expected, metadata, hidden_states_normalized=True)
    else:
        output = layer.forward_MoE(hidden, metadata)
    torch.testing.assert_close(output, expected + 1, rtol=0, atol=0)
    assert norm.call_count == (0 if input_normalized else 1)
    assert len(layer.mlp.calls) == 1
    moe_input, final_reduce, do_finalize = layer.mlp.calls[0]
    torch.testing.assert_close(moe_input, expected, rtol=0, atol=0)
    assert final_reduce == (tp_size > 1)
    assert do_finalize is True
    allreduce.assert_not_called()


@pytest.mark.parametrize("input_normalized", [False, True])
def test_forward_moe_pre_fusion_requires_unnormalized_input(
    monkeypatch: pytest.MonkeyPatch, input_normalized: bool
) -> None:
    layer, allreduce = _build_layer(monkeypatch, tp_size=4, attention_dp=False, eager_fusion=True)
    # Exercise the inherited V4 collective path, which V4.1 normally disables.
    layer.fusion_config.PRE_MOE_FUSION = True
    hidden = torch.arange(24, dtype=torch.bfloat16).reshape(3, 8) / 16
    normalized = layer.post_attention_layernorm(hidden)
    allreduce.side_effect = None
    allreduce.return_value = normalized
    norm = Mock(side_effect=AssertionError("the fused allreduce must apply the norm"))
    monkeypatch.setattr(layer.post_attention_layernorm, "forward", norm)
    metadata = SimpleNamespace(all_rank_num_tokens=[3] * 4)

    if input_normalized:
        with pytest.raises(ValueError, match="Pre-normalized MoE input"):
            layer.forward_MoE(normalized, metadata, hidden_states_normalized=True)
        allreduce.assert_not_called()
        assert layer.mlp.calls == []
    else:
        output = layer.forward_MoE(hidden, metadata)
        torch.testing.assert_close(output, normalized + 1, rtol=0, atol=0)
        allreduce.assert_called_once()
        assert allreduce.call_args.args[0] is hidden
        reduce_params = allreduce.call_args.kwargs["all_reduce_params"]
        assert reduce_params.norm_weight is layer.post_attention_layernorm.weight
        assert reduce_params.eps == layer.post_attention_layernorm.variance_epsilon
        torch.testing.assert_close(layer.mlp.calls[0][0], normalized, rtol=0, atol=0)
    norm.assert_not_called()


@pytest.mark.parametrize("tp_size,attention_dp", [(1, False), (4, False), (4, True)])
def test_decoder_forwards_attention_reduction_contract(
    monkeypatch: pytest.MonkeyPatch, tp_size: int, attention_dp: bool
) -> None:
    layer, allreduce = _build_layer(monkeypatch, tp_size, attention_dp, eager_fusion=True)
    forward_moe = Mock(wraps=layer.forward_MoE)
    monkeypatch.setattr(layer, "forward_MoE", forward_moe)
    residual = torch.ones((3, 4, 8), dtype=torch.bfloat16)
    pre = torch.full((3, 4, 1), 0.25)
    image_mask = torch.tensor([False, True, False])
    metadata = SimpleNamespace(all_rank_num_tokens=[3] * tp_size)
    output = layer(
        torch.arange(3), HCState.resolved(residual, pre_mix=pre), metadata, image_mask=image_mask
    )
    assert output.residual.shape == residual.shape
    assert forward_moe.call_args.kwargs["hidden_states_normalized"] is True
    assert forward_moe.call_args.kwargs["image_mask"] is image_mask
    assert layer.self_attn.calls == [tp_size > 1 and not attention_dp]
    assert len(layer.mlp.calls) == 1
    allreduce.assert_not_called()


@pytest.mark.parametrize("precompute", [False, True])
def test_context_only_hf_weight_ownership_preserves_encoder_and_global_keys(precompute):
    kept = {
        "model.embed_tokens.weight",
        "model.layers.19.input_layernorm.weight",
        "model.layers.19.self_attn.q_a_proj.weight",
    }
    boundary = {
        "model.layers.20.input_layernorm.weight",
        "model.layers.20.self_attn.compressor.wkv.weight",
        "model.layers.20.self_attn.compressor.norm.weight",
        "model.layers.20.self_attn.indexer.wk.weight",
        "model.layers.20.self_attn.indexer.k_norm.weight",
    }
    omitted = {
        "lm_head.weight",
        "model.norm.weight",
        "model.layers.20.self_attn.q_a_proj.weight",
        "model.layers.20.self_attn.indexer.wq_b.weight_scale_inv",
        "model.layers.20.mlp.experts.0.gate_proj.weight",
        "model.layers.21.input_layernorm.weight",
        "model.layers.40.shared_head.norm.weight",
    }
    for key in kept | boundary | omitted:
        reason = v41._v41_ignore_reason(
            key, context_only_split=20, context_only_precompute=precompute
        )
        assert (reason is None) == (key in kept or precompute and key in boundary)


def test_context_only_hf_loader_filters_before_reading_and_audits_coverage(monkeypatch):
    model = nn.Module()
    model.model = nn.Module()
    model.model.disagg_context_only = True
    model.model.decoder_replay_split = 1
    model.model.ced_kv_precompute = True
    model.model.embed_tokens = nn.Embedding(3, 2)
    encoder = nn.Module()
    encoder.input_layernorm = nn.Linear(2, 2, bias=False)
    boundary = nn.Module()
    boundary.input_layernorm = nn.Linear(2, 2, bias=False)
    boundary.self_attn = nn.Module()
    boundary.self_attn.compressor = nn.Module()
    boundary.self_attn.compressor.wkv = nn.Linear(2, 2, bias=False)
    boundary.self_attn.index_wk = nn.Linear(2, 2, bias=False)
    boundary.self_attn.index_k_norm = nn.Module()
    boundary.self_attn.index_k_norm.weight = nn.Parameter(torch.ones(2))
    model.model.layers = nn.ModuleList([encoder, boundary, v41._OmittedContextDecoder()])
    model.config = SimpleNamespace(num_hidden_layers=3, kv_lora_rank=2)
    model.model_config = SimpleNamespace(pretrained_config=model.config)
    retained = {
        "model.embed_tokens.weight": torch.ones(3, 2),
        "model.layers.0.input_layernorm.weight": torch.ones(2, 2),
        "model.layers.1.input_layernorm.weight": torch.ones(2, 2),
        "model.layers.1.self_attn.compressor.wkv.weight": torch.ones(2, 2),
        "model.layers.1.self_attn.indexer.wk.weight": torch.ones(2, 2),
        "model.layers.1.self_attn.indexer.k_norm.weight": torch.ones(2),
    }
    omitted = {
        "model.layers.1.self_attn.q_a_proj.weight",
        "model.layers.1.mlp.experts.0.gate_proj.weight",
        "model.layers.2.input_layernorm.weight",
        "lm_head.weight",
        "model.norm.weight",
    }

    class UnreadableDecoderWeights(dict):
        def __getitem__(self, key):
            assert key not in omitted, f"Read omitted decoder tensor: {key}"
            return super().__getitem__(key)

    reads = []

    class LazyTensor:
        def __init__(self, name, tensor):
            self.name, self.tensor = name, tensor

        def __getitem__(self, index):
            assert index == slice(None)
            reads.append(self.name)
            return self.tensor

    weights = UnreadableDecoderWeights(
        {name: LazyTensor(name, tensor) for name, tensor in retained.items()}
        | dict.fromkeys(omitted, object())
    )
    loaded = {}

    def load_retained(self, passed_weights, skip_modules):
        loaded.update(passed_weights)

    monkeypatch.setattr(v41.DeepseekV4WeightLoader, "_load_weights_impl", load_retained)
    loader = v41.DeepseekV41WeightLoader(model)
    loader._load_weights_impl(weights)
    loader.assert_load_complete()
    assert loaded.keys() == retained.keys()
    assert sorted(reads) == sorted(retained)
    assert all(isinstance(value, torch.Tensor) for value in loaded.values())
    assert loader.census.total == len(retained) + len(omitted)
    assert loader.census.forwarded == len(retained)
    assert loader.census.ignored_count == len(omitted)
    loader.census.verify()


def test_hf_draft_loader_without_target_body_keeps_existing_weight_path(monkeypatch):
    draft = nn.Module()
    draft.config = SimpleNamespace()
    draft.model_config = SimpleNamespace(pretrained_config=draft.config)
    weights = {"mtp_layers.0.main_proj.weight": torch.ones(2, 2)}
    forwarded = Mock()
    monkeypatch.setattr(v41.DeepseekV4WeightLoader, "_load_weights_impl", forwarded)
    loader = v41.DeepseekV41WeightLoader(draft)
    loader._load_weights_impl(weights)
    forwarded.assert_called_once_with(weights, skip_modules=[])
    assert loader.census is None
