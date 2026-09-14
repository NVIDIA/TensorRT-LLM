# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM configuration, loading and vision ownership regressions without checkpoint files."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch.distributed import AllReduce
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.hf.glm5_next_weight_mapper import (
    Disposition,
    audit_glm5_next_checkpoint,
)
from tensorrt_llm._torch.models.modeling_glm5_next import (
    Glm5NextForCausalLM,
    Glm5NextLinearAttention,
    Glm5NextSparseAttention,
)
from tensorrt_llm._torch.models.modeling_glm5_next_vision import Glm5NextVisionModelBase
from tensorrt_llm._torch.pyexecutor.config_utils import get_glm5_next_layer_masks
from tensorrt_llm.mapping import Mapping


def _config():
    config = PretrainedConfig(
        model_type="glm5_next",
        text_config=PretrainedConfig(
            model_type="glm5_next_text",
            dtype="bfloat16",
            rms_norm_eps=1e-05,
            num_attention_heads=64,
            q_lora_rank=1536,
            kv_lora_rank=512,
            qk_nope_head_dim=256,
            qk_rope_head_dim=0,
            v_head_dim=256,
            index_n_heads=32,
            index_head_dim=128,
            index_topk=2048,
            index_kpool=4,
            index_kpool_always_select_tail=True,
            hidden_size=256,
            num_hidden_layers=2,
            mlp_layer_types=["dense", "dense"],
            first_k_dense_replace=2,
            num_nextn_predict_layers=1,
            max_position_embeddings=1024,
            linear_attn_config={
                "num_heads": 64,
                "head_dim": 128,
                "short_conv_kernel_size": 4,
                "gate_lower_bound": -5.0,
            },
        ),
        vision_config=PretrainedConfig(
            attention_bias=True,
            rms_norm_eps=1e-05,
            swiglu_limit=7.0,
            in_channels=3,
            depth=1,
            hidden_size=256,
            num_heads=4,
            intermediate_size=512,
            out_hidden_size=256,
            projection_intermediate_size=512,
            patch_size=2,
            temporal_patch_size=2,
            spatial_merge_size=2,
        ),
    )
    # The shared Transformers version predates this layer name. Attach the
    # synthetic schedule after its constructor validates the base config.
    config.text_config.layer_types = ["linear_attention", "deepseek_sparse_attention"]
    return config


@pytest.mark.cpu_only
def test_layer_masks_accept_composite_and_text_configs():
    config = _config()
    expected = ([False, True], [True, False])
    assert get_glm5_next_layer_masks(config) == expected
    assert get_glm5_next_layer_masks(config.text_config) == expected
    config.text_config.layer_types.pop()
    with pytest.raises(ValueError, match="num_hidden_layers"):
        get_glm5_next_layer_masks(config)
    config.text_config.layer_types = ["linear_attention", "unknown"]
    with pytest.raises(ValueError, match="must be exactly one"):
        get_glm5_next_layer_masks(config)


@pytest.mark.cpu_only
def test_checkpoint_routes_vision_and_optional_mtp_separately():
    config = _config()
    keys = [
        "model.visual.patch_embed.proj.weight",
        "model.language_model.layers.0.self_attn.A_log",
        "model.language_model.layers.2.eh_proj.weight",
        "lm_head.weight",
        "unexpected.weight",
    ]
    plain = audit_glm5_next_checkpoint(keys, config)
    mtp = audit_glm5_next_checkpoint(keys, config, num_mtp_layers=1)
    assert plain.disposition[keys[0]] == Disposition.IGNORED
    assert plain.disposition[keys[2]] == Disposition.IGNORED
    assert mtp.destinations["model.layers.2.eh_proj.weight"] == keys[2]
    assert mtp.destinations["model.layers.0.self_attn.A_log"] == keys[1]
    assert mtp.unresolved == ["unexpected.weight"]
    for invalid in (-1, 2):
        with pytest.raises(ValueError, match=f"cannot load {invalid} MTP layers"):
            audit_glm5_next_checkpoint(keys, config, num_mtp_layers=invalid)


@pytest.mark.cpu_only
@pytest.mark.parametrize("route", ["valid", "py_mamba", "manager_preference", "v1", "cpp", "ucx"])
def test_cache_manager_routing_guards(route, monkeypatch):
    from tensorrt_llm._torch.attention.backends.sparse.glm_kpool import Glm5NextCacheManager
    from tensorrt_llm._torch.pyexecutor._util import get_kv_cache_manager_cls
    from tensorrt_llm.llmapi import CacheTransceiverConfig, KvCacheConfig

    monkeypatch.delenv("TRTLLM_USE_PY_MAMBA", raising=False)
    monkeypatch.delenv("TLLM_MAMBA_MANAGER_PREFERENCE", raising=False)
    if route == "py_mamba":
        monkeypatch.setenv("TRTLLM_USE_PY_MAMBA", "1")
    if route == "manager_preference":
        monkeypatch.setenv("TLLM_MAMBA_MANAGER_PREFERENCE", "MIXED")
    config = ModelConfig(pretrained_config=_config())
    kv = KvCacheConfig(use_kv_cache_manager_v2=route != "v1")
    transceiver = CacheTransceiverConfig(
        backend="UCX" if route == "ucx" else "NIXL",
        transceiver_runtime="CPP" if route == "cpp" else "PYTHON",
    )
    if route == "valid":
        assert get_kv_cache_manager_cls(config, kv) is Glm5NextCacheManager
        assert get_kv_cache_manager_cls(config, kv, True, transceiver) is Glm5NextCacheManager
    else:
        with pytest.raises(ValueError, match="glm5_next"):
            get_kv_cache_manager_cls(config, kv, True, transceiver)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("deferred", [False, True])
def test_encoder_only_factory_materializes_attention_weights(deferred):
    from tensorrt_llm._torch.models.modeling_auto import AutoModelForCausalLM
    from tensorrt_llm._torch.models.modeling_glm5_next_vision import Glm5NextVisionModel

    config = _config()
    config.architectures = ["Glm5NextForConditionalGeneration"]
    model_config = ModelConfig(
        pretrained_config=config, mm_encoder_only=True, skip_create_weights_in_init=deferred
    )
    with torch.device("meta"):
        encoder = AutoModelForCausalLM.from_config(model_config)
    assert isinstance(encoder, Glm5NextVisionModelBase)
    assert isinstance(encoder.visual, Glm5NextVisionModel)
    for block in encoder.visual.blocks:
        for projection in (block.attn.qkv_proj, block.attn.o_proj, block.mlp.down_proj):
            assert projection._weights_created
            assert projection.weight is not None


@pytest.mark.cpu_only
def test_attention_dp_speculation_rejected_before_model_construction():
    config = SimpleNamespace(
        pretrained_config=_config(),
        mapping=Mapping(world_size=4, tp_size=4, enable_attention_dp=True),
        spec_config=object(),
    )
    with (
        patch(
            "tensorrt_llm._torch.models.modeling_glm5_next.Glm5NextModel",
            side_effect=AssertionError("must reject before constructing layers"),
        ),
        pytest.raises(ValueError, match="does not support attention DP"),
    ):
        Glm5NextForCausalLM(config)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_kda_shards_preserve_fp32_gate_parameters():
    config = _config().text_config
    full_bias = torch.arange(64 * 128, dtype=torch.float32)
    full_log = torch.arange(64, dtype=torch.float32)
    biases, logs = [], []
    with patch("tensorrt_llm._torch.modules.kimi_kda.kimi_kda_mixer.AllReduce"):
        for rank in range(4):
            layer = Glm5NextLinearAttention(
                config, 0, mapping=Mapping(world_size=4, tp_size=4, rank=rank)
            )
            assert not layer.use_full_rank_gate
            assert layer.A_log.shape == (16,)
            assert layer.dt_bias.shape == (16 * 128,)
            assert layer.A_log.dtype == layer.dt_bias.dtype == torch.float32
            biases.append(layer.shard_checkpoint_tensor("dt_bias", full_bias))
            logs.append(layer.shard_checkpoint_tensor("A_log", full_log))
    torch.testing.assert_close(torch.cat(biases), full_bias)
    torch.testing.assert_close(torch.cat(logs), full_log)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_vision_attention_dp_is_local_and_matches_single_rank():
    config = _config()
    reference = Glm5NextVisionModelBase(ModelConfig(pretrained_config=config)).cuda().eval()
    reference.visual.setup_attn_metadata(max_num_tokens=128)
    torch.manual_seed(7)
    for parameter in reference.parameters():
        parameter.normal_(0, 0.05)

    # Rank 0 has no image; the other ranks see different token counts and
    # pixels. Any TP collective on the encoder path is an error, even when
    # another rank has no encoder work at all.
    def local_allreduce(**kwargs):
        assert kwargs["mapping"].tp_size == 1, "vision must not reduce across DP ranks"
        return AllReduce(**kwargs)

    with patch("tensorrt_llm._torch.distributed.AllReduce", side_effect=local_allreduce):
        for rank, width in enumerate((0, 4, 6, 8)):
            mapping = Mapping(world_size=4, tp_size=4, rank=rank, enable_attention_dp=True)
            local = (
                Glm5NextVisionModelBase(ModelConfig(pretrained_config=config, mapping=mapping))
                .cuda()
                .eval()
            )
            local.load_state_dict(reference.state_dict())
            local.visual.setup_attn_metadata(max_num_tokens=128)
            if not width:
                continue
            pixels = torch.randn(4 * width, 3 * 2 * 2 * 2, device="cuda", dtype=torch.bfloat16)
            grid = torch.tensor([[1, 4, width]])
            expected = reference.encode_batched(pixels, grid)
            actual = local.encode_batched(pixels, grid)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("temporal_groups", [1, 3], ids=["image", "video"])
@torch.inference_mode()
def test_vision_weights_and_features_match_hf(temporal_groups):
    hf_module = pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    Glm5NextVisionModel = hf_module.Glm5NextVisionModel

    from tensorrt_llm._torch.models.modeling_glm5_next_vision import _flatten_video_grid_thw

    torch.manual_seed(11)
    config = _config()
    config.vision_config = hf_module.Glm5NextVisionConfig.from_dict(config.vision_config.to_dict())
    hf = Glm5NextVisionModel(config.vision_config).to(device="cuda", dtype=torch.bfloat16).eval()
    runtime = Glm5NextVisionModelBase(ModelConfig(pretrained_config=config)).cuda().eval()
    runtime.visual.setup_attn_metadata(max_num_tokens=128)
    weights = {"model.visual." + name: tensor for name, tensor in hf.state_dict().items()}
    runtime.load_weights(weights)
    pixels = torch.randn(
        temporal_groups * 4 * 6, 3 * 2 * 2 * 2, device="cuda", dtype=torch.bfloat16
    )
    grid = torch.tensor([[temporal_groups, 4, 6]])
    flat_grid = _flatten_video_grid_thw(grid)
    expected = hf(pixels, grid_thw=flat_grid.cuda()).pooler_output
    actual = runtime.encode_batched(pixels, flat_grid)
    assert actual.shape == (temporal_groups * 6, config.vision_config.out_hidden_size)
    assert torch.isfinite(actual).all()
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().flatten(), expected.float().flatten(), dim=0
    )
    relative_l2 = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(
        1e-12
    )
    assert cosine > 0.999
    assert relative_l2 < 0.03


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] != 10,
    reason="requires Blackwell sparse MLA",
)
@torch.no_grad()
def test_sparse_prefill_continuation_preserves_partial_pools():
    """A chunk boundary inside a k-pool preserves both cached state and attention outputs."""
    torch.manual_seed(17)
    layer = Glm5NextSparseAttention(_config().text_config, 1).cuda().eval()
    for parameter in layer.parameters():
        parameter.normal_(0, 0.05)
    hidden = torch.randn(13, 256, device="cuda", dtype=torch.bfloat16)
    hidden[:5] += 1
    tables = torch.zeros(1, 256, device="cuda", dtype=torch.long)
    tables[0, :2] = torch.tensor([2, 0], device="cuda")

    def run(lengths, discard_prefix=False):
        latent = torch.zeros(4, 8, 1, layer.kv_lora_rank, device="cuda", dtype=torch.bfloat16)
        index = torch.zeros(
            4, 8, 1, layer.indexer.cache_state_dim, device="cuda", dtype=torch.bfloat16
        )
        manager = SimpleNamespace(
            tokens_per_block=8,
            get_latent_state_buffer=lambda _: latent,
            get_index_state_buffer=lambda _: index,
        )
        outputs = []
        cached = 0
        for length in lengths:
            if cached and discard_prefix:
                latent.zero_()
                index.zero_()
            metadata = SimpleNamespace(
                kv_cache_manager=manager,
                seq_lens=torch.tensor([length]),
                num_contexts=1,
                kv_lens_cuda=torch.tensor([cached + length], device="cuda", dtype=torch.int32),
                mamba_metadata=SimpleNamespace(glm_block_tables=tables),
            )
            outputs.append(
                layer.forward_prefill(
                    hidden[cached : cached + length], [0, length], [cached], metadata
                )
            )
            cached += length
        return torch.cat(outputs), latent, index

    reference, latent, index = run([13])
    actual, chunked_latent, chunked_index = run([5, 8])
    torch.testing.assert_close(chunked_latent, latent, rtol=0.02, atol=1e-3)
    torch.testing.assert_close(chunked_index, index, rtol=0.02, atol=1e-3)
    relative_l2 = (actual.float() - reference.float()).norm() / reference.float().norm()
    assert relative_l2 < 0.03
    # Negative control: this gate must detect losing the previous chunk.
    broken, _, _ = run([5, 8], discard_prefix=True)
    broken_l2 = (broken.float() - reference.float()).norm() / reference.float().norm()
    assert broken_l2 > 0.03
