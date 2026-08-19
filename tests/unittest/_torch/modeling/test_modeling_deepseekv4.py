# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
import json
import struct
import weakref
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from transformers import PretrainedConfig
from utils.util import getSMVersion, skip_blackwell_geforce, skip_pre_blackwell

# from utils.util import default_dtype
import tensorrt_llm
from tensorrt_llm._torch.attention.mla import MLA
from tensorrt_llm._torch.attention_backend.fmha import FallbackFmha, FlashInferSparseMlaFmha
from tensorrt_llm._torch.attention_backend.interface import (
    AttentionForwardArgs,
    PositionalEmbeddingParams,
    RopeParams,
)
from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4 import (
    DeepseekV4CacheManager,
    DeepseekV4Indexer,
    DeepseekV4TrtllmAttention,
    DeepseekV4TrtllmAttentionMetadata,
)
from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4.compressor import Compressor
from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4.module import (
    _fused_q_rope_specs,
    _is_fused_kv_norm_enabled,
    _is_fused_prologue_active,
    _is_fused_q_fp8_quant_enabled,
)
from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention
from tensorrt_llm._torch.configs.deepseekv4 import DeepseekV4Config
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_deepseekv4 import (
    DeepseekV4DecoderLayer,
    DeepseekV4ForCausalLM,
    DeepseekV4Gate,
    DeepseekV4MTP,
    DeepseekV4WeightLoader,
    _copy_deepseek_v4_fused_a_weight_scale,
    _deepseek_v4_pos_embd_params,
    _normalize_deepseek_v4_nvfp4_mixed_precision_config,
    _remap_deepseek_v4_checkpoint_keys,
    _resolve_enable_fused_hc,
)
from tensorrt_llm._torch.modules.linear import TensorParallelMode
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.utils import AuxStreamType, model_extra_attrs
from tensorrt_llm.functional import PositionEmbeddingType, RotaryScalingType
from tensorrt_llm.llmapi.llm_args import (
    DeepSeekV4SparseAttentionConfig,
    KvCacheConfig,
    MTPDecodingConfig,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

DEEPSEEK_V4_TINY_CONFIG = {
    "architectures": ["DeepseekV4ForCausalLM"],
    "model_type": "deepseek_v4",
    "hidden_size": 4096,
    "num_attention_heads": 64,
    "num_key_value_heads": 1,
    "qk_nope_head_dim": 448,
    "qk_rope_head_dim": 64,
    "v_head_dim": 512,
    "q_lora_rank": 1024,
    "kv_lora_rank": 448,
    "o_groups": 8,
    "o_lora_rank": 1024,
    "max_position_embeddings": 65536,
    "rms_norm_eps": 1e-6,
    "dtype": "bfloat16",
    "vocab_size": 129280,
    "num_hidden_layers": 7,
    "n_hash_layers": 3,
    "moe_intermediate_size": 2048,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "num_experts_per_tok": 6,
    "n_group": 1,
    "topk_group": 1,
    "routed_scaling_factor": 1.5,
    "score_func": "sqrtsoftplus",
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-6,
    "compress_rope_theta": 40000.0,
    "rope_theta": 10000.0,
    "rope_scaling": {
        "type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 65536,
        "beta_fast": 32,
        "beta_slow": 1,
    },
    "quantization_config": {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "scale_fmt": "ue8m0",
        "weight_block_size": [128, 128],
    },
}


def _write_safetensors_header(path, tensor_name, dtype, shape):
    header = {
        tensor_name: {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [0, 0],
        }
    }
    payload = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(payload)) + payload)


def test_deepseek_v4_config_aliases():
    config = DeepseekV4Config(
        num_hash_layers=5, sliding_window=256, head_dim=128, score_func="sigmoid", swiglu_limit=9.0
    )

    assert config.model_type == "deepseek_v4"
    assert config.n_hash_layers == 5
    assert config.window_size == 256
    assert config.v_head_dim == 128
    assert config.scoring_func == "sigmoid"
    assert config.swiglu_limit == 9.0


def test_deepseek_v4_fused_hc_default_enabled(monkeypatch):
    monkeypatch.delenv("TRTLLM_MHC_ENABLE_FUSED_HC", raising=False)
    config = PretrainedConfig()

    assert _resolve_enable_fused_hc(config) is True

    config.enable_fused_hc = False
    assert _resolve_enable_fused_hc(config) is False

    monkeypatch.setenv("TRTLLM_MHC_ENABLE_FUSED_HC", "1")
    assert _resolve_enable_fused_hc(config) is True

    monkeypatch.setenv("TRTLLM_MHC_ENABLE_FUSED_HC", "0")
    assert _resolve_enable_fused_hc(config) is False


def test_deepseek_v4_kv_cache_defaults_and_v2_preference(monkeypatch):
    monkeypatch.setattr(
        "tensorrt_llm._torch.models.modeling_deepseekv4.get_sm_version", lambda: 100
    )
    defaults = DeepseekV4ForCausalLM.get_model_defaults(None)

    assert defaults == {
        "kv_cache_config": {
            "tokens_per_block": 128,
            "enable_swa_scratch_reuse": True,
        }
    }
    assert DeepseekV4ForCausalLM.get_preferred_kv_cache_manager_version() == "V2"


def test_deepseek_v4_hopper_uses_fp8_ds_mla_defaults(monkeypatch) -> None:
    monkeypatch.setattr("tensorrt_llm._torch.models.modeling_deepseekv4.get_sm_version", lambda: 90)

    defaults = DeepseekV4ForCausalLM.get_model_defaults(None)

    assert defaults == {
        "kv_cache_config": {
            "dtype": "fp8_ds_mla",
            "tokens_per_block": 128,
            "enable_swa_scratch_reuse": True,
        }
    }


def test_deepseek_v4_fp8_ds_mla_uses_256_token_blocks(monkeypatch) -> None:
    monkeypatch.setattr(
        "tensorrt_llm._torch.models.modeling_deepseekv4.get_sm_version", lambda: 100
    )

    class LlmArgs:
        kv_cache_config = KvCacheConfig(dtype="fp8_ds_mla")

    defaults = DeepseekV4ForCausalLM.get_model_defaults(LlmArgs())

    assert defaults == {
        "kv_cache_config": {
            "tokens_per_block": 256,
            "enable_swa_scratch_reuse": True,
        }
    }


def test_deepseek_v4_weight_remap_for_mxfp4_routed_experts():
    weights = {
        "layers.0.ffn.experts.0.w1.weight": torch.tensor([[-1, 2], [3, -4]], dtype=torch.int8),
        "layers.0.ffn.experts.0.w1.scale": torch.tensor([1, 2], dtype=torch.int8),
    }

    remapped = _remap_deepseek_v4_checkpoint_keys(weights, num_hidden_layers=1, kv_lora_rank=448)

    assert remapped["model.layers.0.mlp.experts.0.w1.weight"].dtype == torch.uint8
    canonical_scale = remapped["model.layers.0.mlp.experts.0.w1.weight_scale"]
    legacy_scale = remapped["model.layers.0.mlp.experts.0.w1.weight_scale_inv"]
    assert canonical_scale.dtype == torch.uint8
    assert legacy_scale is canonical_scale


def test_deepseek_v4_weight_remap_for_fp8_routed_experts():
    weights = {
        "layers.0.ffn.experts.0.w1.weight": torch.zeros((2, 2), dtype=torch.float32),
        "layers.0.ffn.experts.0.w1.scale": torch.ones((2, 2), dtype=torch.float32),
    }

    remapped = _remap_deepseek_v4_checkpoint_keys(weights, num_hidden_layers=1, kv_lora_rank=448)

    assert "model.layers.0.mlp.experts.0.w1.weight_scale_inv" in remapped
    assert "model.layers.0.mlp.experts.0.w1.weight_scale" not in remapped


def test_deepseek_v4_eplb_weight_loader_pages_out_each_moe_layer(monkeypatch):
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    layer = model.model.layers[0]
    layer.foo = torch.nn.Module()
    layer.foo.weight = torch.nn.Parameter(torch.zeros(2))
    layer.mlp = torch.nn.Module()
    layer.mlp.experts = torch.nn.Module()
    layer.mlp.experts.backend = torch.nn.Module()
    model.config = SimpleNamespace(
        q_lora_rank=1,
        num_attention_heads=1,
        qk_nope_head_dim=1,
        v_head_dim=1,
        kv_lora_rank=1,
        num_hidden_layers=1,
        num_nextn_predict_layers=0,
    )
    model.model_config = SimpleNamespace(
        mapping=SimpleNamespace(
            tp_rank=0,
            tp_size=1,
            cp_rank=0,
            cp_size=1,
            enable_attention_dp=False,
        ),
        moe_load_balancer=object(),
    )
    weights = {"model.layers.0.foo.weight": torch.tensor([1.0, 2.0])}
    synchronize_calls = []
    pageout_calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: synchronize_calls.append(None))
    monkeypatch.setattr(
        "tensorrt_llm._torch.models.modeling_deepseekv4.pageout_file_backed_regions",
        lambda path_substring, mode: pageout_calls.append((path_substring, mode)),
    )

    DeepseekV4WeightLoader(model)._load_weights_impl(weights)

    assert len(synchronize_calls) == 1
    assert pageout_calls == [(".safetensors", "dontneed")]


def test_deepseek_v4_fused_a_weight_scale_rebuilds_fp8_shape():
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(torch.empty((2048, 7168), dtype=torch.float8_e4m3fn))
    module.weight_scale = torch.nn.Parameter(torch.empty((16, 16), dtype=torch.float32))
    module.rebuild_tensor_metadata = {}
    fused_a = torch.empty((2048, 7168), dtype=torch.float8_e4m3fn)
    fused_a_scale = torch.ones((16, 56), dtype=torch.float32)

    _copy_deepseek_v4_fused_a_weight_scale(module, fused_a, fused_a_scale)

    assert module.weight_scale.shape == fused_a_scale.shape
    assert torch.equal(module.weight_scale, fused_a_scale)


def test_deepseek_v4_fused_a_weight_scale_keeps_oversized_slice():
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(torch.empty((2176, 7168), dtype=torch.float8_e4m3fn))
    module.weight_scale = torch.nn.Parameter(torch.zeros((17, 56), dtype=torch.float32))
    module.rebuild_tensor_metadata = {}
    fused_a = torch.empty((2048, 7168), dtype=torch.float8_e4m3fn)
    fused_a_scale = torch.ones((16, 56), dtype=torch.float32)

    _copy_deepseek_v4_fused_a_weight_scale(module, fused_a, fused_a_scale)

    assert module.weight_scale.shape == (17, 56)
    assert torch.equal(module.weight_scale[:16], fused_a_scale)
    assert torch.equal(module.weight_scale[16], torch.zeros(56))


def test_deepseek_v4_kv_norm_keeps_full_head_dim():
    weights = {
        "layers.0.attn.kv_norm.weight": torch.arange(512, dtype=torch.float32),
    }

    remapped = _remap_deepseek_v4_checkpoint_keys(weights, num_hidden_layers=1, kv_lora_rank=448)

    tensor = remapped["model.layers.0.self_attn.kv_a_layernorm.weight"]
    assert tensor.shape == (512,)
    assert tensor[-1].item() == 511


def test_deepseek_v4_gate_bias_maps_to_score_correction_bias():
    weights = {
        "layers.0.ffn.gate.bias": torch.arange(4, dtype=torch.float32),
    }

    remapped = _remap_deepseek_v4_checkpoint_keys(weights, num_hidden_layers=1, kv_lora_rank=448)

    assert torch.equal(
        remapped["model.layers.0.mlp.gate.e_score_correction_bias"],
        weights["layers.0.ffn.gate.bias"],
    )


def test_deepseek_v4_gate_uses_fp32_reference_linear():
    if not torch.cuda.is_available():
        pytest.skip("dsv3_router_gemm_op requires CUDA")

    device = torch.device("cuda")
    gate = DeepseekV4Gate(
        hidden_size=4,
        num_experts=3,
        top_k=2,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        is_hashed=False,
        dtype=torch.bfloat16,
        moe_backend="TRTLLM",
    ).to(device)
    hidden_states = torch.tensor([[1.0, -2.0, 3.0, -4.0]], dtype=torch.bfloat16, device=device)
    weight = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [-1.0, 1.0, -1.0, 1.0], [0.5, -0.5, 0.25, -0.25]],
        dtype=torch.bfloat16,
        device=device,
    )
    gate.weight.copy_(weight)

    logits = gate(hidden_states)

    assert gate.e_score_correction_bias.dtype == torch.float32
    assert logits.dtype == torch.float32
    assert torch.equal(logits, torch.nn.functional.linear(hidden_states.float(), weight.float()))


def test_deepseek_v4_attn_sink_remap():
    weights = {
        "layers.0.attn.attn_sink": torch.arange(4, dtype=torch.float32),
    }

    remapped = _remap_deepseek_v4_checkpoint_keys(weights, num_hidden_layers=1, kv_lora_rank=448)

    assert torch.equal(
        remapped["model.layers.0.self_attn.attn_sink"], weights["layers.0.attn.attn_sink"]
    )


def test_deepseek_v4_flat_hc_weight_remap():
    weights = {
        "layers.0.hc_attn_fn": torch.tensor([1.0]),
        "layers.0.hc_ffn_scale": torch.tensor([2.0]),
        "hc_head_base": torch.tensor([3.0]),
    }

    remapped = _remap_deepseek_v4_checkpoint_keys(weights, num_hidden_layers=1, kv_lora_rank=448)

    assert torch.equal(remapped["model.layers.0.hc_attn_fn"], weights["layers.0.hc_attn_fn"])
    assert torch.equal(remapped["model.layers.0.hc_ffn_scale"], weights["layers.0.hc_ffn_scale"])
    assert torch.equal(remapped["model.hc_head_base"], weights["hc_head_base"])


def test_deepseek_v4_o_a_proj_scale_remap():
    weights = {
        "layers.0.attn.wo_a.weight": torch.zeros((8, 8), dtype=torch.float8_e4m3fn),
        "layers.0.attn.wo_a.scale": torch.ones((1, 1), dtype=torch.float32),
    }

    remapped = _remap_deepseek_v4_checkpoint_keys(weights, num_hidden_layers=1, kv_lora_rank=448)

    assert "model.layers.0.self_attn.o_a_proj" in remapped
    assert "model.layers.0.self_attn.o_a_proj.weight_scale_inv" in remapped


def test_deepseek_v4_q_b_layernorm_matches_per_head_reference():
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    if not torch.cuda.is_available():
        pytest.skip("RMSNorm fast paths require CUDA")

    eps = 1e-6
    num_heads = 2
    head_dim = 4
    device = torch.device("cuda")
    norm = RMSNorm(
        hidden_size=head_dim, eps=eps, dtype=torch.bfloat16, device=device, has_weights=False
    )
    hidden_states = torch.arange(1, 17, dtype=torch.bfloat16, device=device).reshape(2, 8)

    output = norm(hidden_states.view(-1, head_dim)).view_as(hidden_states)

    ref = hidden_states.view(2, num_heads, head_dim)
    ref = ref * torch.rsqrt(ref.square().float().mean(dim=-1, keepdim=True) + eps).to(ref.dtype)
    torch.testing.assert_close(output, ref.reshape(2, 8), rtol=1e-2, atol=2e-2)
    assert list(norm.named_parameters()) == []


def test_deepseek_v4_q_b_layernorm_differs_from_joint_flat_rms():
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    if not torch.cuda.is_available():
        pytest.skip("RMSNorm fast paths require CUDA")

    eps = 1e-6
    head_dim = 4
    device = torch.device("cuda")
    head_scales = torch.tensor([1.0, 10.0, 0.1, 1.0], dtype=torch.bfloat16, device=device)
    num_heads = head_scales.numel()
    base = torch.arange(1, 1 + 2 * num_heads * head_dim, dtype=torch.bfloat16, device=device).view(
        2, num_heads, head_dim
    )
    hidden_states = (base * head_scales.view(1, num_heads, 1)).reshape(2, num_heads * head_dim)

    per_head_norm = RMSNorm(
        hidden_size=head_dim, eps=eps, dtype=torch.bfloat16, device=device, has_weights=False
    )
    per_head = per_head_norm(hidden_states.view(-1, head_dim)).view_as(hidden_states)

    joint_norm = RMSNorm(
        hidden_size=num_heads * head_dim,
        eps=eps,
        dtype=torch.bfloat16,
        device=device,
        has_weights=False,
    )
    joint = joint_norm(hidden_states)

    assert not torch.allclose(per_head, joint, atol=0.1)


def test_deepseek_v4_mla_builds_both_norms_at_the_v4_widths():
    """The two tests above pin the norm maths; this pins how MLA wires them up.

    `kv_a_layernorm` spans the WHOLE 512-wide latent, RoPE tail included -- unlike
    V3/V3.2, where it is `kv_lora_rank` wide and the tail bypasses it. The fused KV
    kernel applies the norm itself over that full row and
    `_is_fused_kv_norm_enabled` gates on the weight width, so narrowing it would
    silently disable the fusion rather than fail.
    """
    if not torch.cuda.is_available():
        pytest.skip("MLA construction requires CUDA")

    cfg = DeepseekV4Config(**deepcopy(DEEPSEEK_V4_TINY_CONFIG))
    model_config = ModelConfig(
        pretrained_config=cfg,
        sparse_attention_config=DeepSeekV4SparseAttentionConfig(
            index_n_heads=32, index_head_dim=128, index_topk=512
        ),
    )
    if getSMVersion() in (120, 121):
        # SM120/SM121 only support DeepSeek-V4 sparse MLA through the FlashInfer
        # fp8_ds_mla path; construction raises ValueError with any other dtype.
        model_config.extra_attrs["kv_cache_dtype"] = "fp8_ds_mla"
    mla = MLA(
        hidden_size=cfg.hidden_size,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=1,
        qk_nope_head_dim=cfg.qk_nope_head_dim,
        qk_rope_head_dim=cfg.qk_rope_head_dim,
        v_head_dim=cfg.v_head_dim,
        q_lora_rank=cfg.q_lora_rank,
        kv_lora_rank=cfg.kv_lora_rank,
        predicted_tokens_per_seq=1,
        max_position_embeddings=cfg.max_position_embeddings,
        bias=False,
        pos_embd_params=_deepseek_v4_pos_embd_params(cfg, model_config, 0),
        layer_idx=0,
        dtype=torch.bfloat16,
        config=model_config,
        num_groups=cfg.o_groups,
        o_lora_rank=cfg.o_lora_rank,
    ).to("cuda")

    latent_width = cfg.kv_lora_rank + cfg.qk_rope_head_dim
    assert mla.kv_a_layernorm.weight.shape == (latent_width,)
    assert mla.q_b_layernorm.weight.shape == (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim,)
    assert list(mla.q_b_layernorm.named_parameters()) == []
    # `kv_b_proj` is absent because DeepSeekV4Hooks.need_absorption is False, which
    # is what proves the 512 above came from the V4 hook rather than from MLA's own
    # `kv_lora_rank`-wide construction.
    assert not hasattr(mla, "kv_b_proj")

    # The RoPE tail is inside the norm: perturbing it moves the normalized nope
    # segment, which a 448-wide V3-style norm would leave untouched.
    latent = torch.randn(4, latent_width, dtype=torch.bfloat16, device="cuda")
    perturbed = latent.clone()
    perturbed[:, cfg.kv_lora_rank :] *= 4.0
    assert not torch.allclose(
        mla.kv_a_layernorm(latent)[:, : cfg.kv_lora_rank],
        mla.kv_a_layernorm(perturbed)[:, : cfg.kv_lora_rank],
    )


def test_deepseek_v4_compressor_rotate_and_indexer_rope_contracts():
    assert inspect.signature(Compressor).parameters["rotate_activation"].default is False

    indexer_init = inspect.getsource(DeepseekV4Indexer.__init__)
    assert "is_neox=False" in indexer_init
    assert "rotate_activation=HAS_FAST_HADAMARD" in indexer_init

    attention_init = inspect.getsource(DeepseekV4TrtllmAttention.__init__)
    assert "rotate_activation=False" in attention_init


def test_deepseek_v4_attention_forward_injects_attn_sink(monkeypatch):
    captured = {}

    def fake_forward(self, *args, **kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(TrtllmAttention, "forward", fake_forward)
    monkeypatch.setattr(
        DeepseekV4TrtllmAttention,
        "_prepare_sparse_forward_args",
        lambda self, metadata, forward_args: None,
    )
    attn = object.__new__(DeepseekV4TrtllmAttention)
    sink = torch.ones(4, dtype=torch.float32)
    attn.attn_sink = torch.nn.Parameter(sink, requires_grad=False)

    metadata = object()
    assert DeepseekV4TrtllmAttention.forward(attn, "q", None, None, metadata) == "ok"
    assert "attention_sinks" not in captured
    assert captured["forward_args"].attention_sinks.data_ptr() == sink.data_ptr()

    captured.clear()
    forward_args = AttentionForwardArgs()
    assert (
        DeepseekV4TrtllmAttention.forward(
            attn, "q", None, None, metadata, forward_args=forward_args
        )
        == "ok"
    )
    assert "attention_sinks" not in captured
    assert captured["forward_args"].attention_sinks.data_ptr() == sink.data_ptr()
    assert forward_args.attention_sinks is None


def test_deepseek_v4_moe_auto_backend_on_blackwell(monkeypatch):
    monkeypatch.setattr("tensorrt_llm._torch.model_config.get_sm_version", lambda: 100)

    assert ModelConfig.resolve_moe_backend("AUTO", "DeepseekV4ForCausalLM") == "TRTLLM"


def test_deepseek_v4_nvfp4_mixed_precision_config():
    config = DeepseekV4Config()
    config.quantization_config = {
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
        "modules_to_not_convert": ["lm_head"],
    }
    mixed_quant_config = QuantConfig(
        quant_algo=QuantAlgo.MIXED_PRECISION,
        group_size=16,
        exclude_modules=["*.attn.*", "*.ffn.shared_experts.*", "head", "mtp.*"],
    )
    mixed_quant_config.mamba_ssm_cache_dtype = torch.bfloat16
    assert not mixed_quant_config.layer_quant_mode.has_fp8_block_scales()
    experts_quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=16)
    model_config = ModelConfig(
        pretrained_config=config,
        quant_config=mixed_quant_config,
        quant_config_dict={"model.layers.0.mlp.experts": experts_quant_config},
    )
    model_config._frozen = True

    normalized_config = _normalize_deepseek_v4_nvfp4_mixed_precision_config(model_config)

    assert normalized_config is model_config
    assert mixed_quant_config.quant_algo == QuantAlgo.MIXED_PRECISION
    assert normalized_config.quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES
    assert normalized_config.quant_config.layer_quant_mode.has_fp8_block_scales()
    assert normalized_config.quant_config.group_size == 128
    assert normalized_config.quant_config.mamba_ssm_cache_dtype == torch.bfloat16
    assert normalized_config.quant_config.exclude_modules == [
        "lm_head",
        "*kv_b_proj*",
        "*k_b_proj*",
        "*eh_proj*",
    ]
    assert (
        normalized_config.quant_config_dict["model.layers.0.mlp.experts"].quant_algo
        == QuantAlgo.NVFP4
    )


def test_deepseek_v4_routed_moe_quant_config_from_mxfp4_header(tmp_path, monkeypatch):
    monkeypatch.setattr("tensorrt_llm._torch.model_config.get_sm_version", lambda: 100)
    tensor_name = "layers.0.ffn.experts.0.w1.weight"
    shard_name = "model-00001-of-00001.safetensors"
    header = {
        tensor_name: {
            "dtype": "I8",
            "shape": [2, 2],
            "data_offsets": [0, 0],
        },
    }
    payload = json.dumps(header).encode("utf-8")
    (tmp_path / shard_name).write_bytes(struct.pack("<Q", len(payload)) + payload)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    tensor_name: shard_name,
                }
            }
        )
    )
    config = DeepseekV4Config(num_hidden_layers=2)

    layer_quant_config = ModelConfig._set_deepseek_v4_routed_moe_quant_config(
        config, str(tmp_path), "TRTLLM", None
    )

    quant_config = layer_quant_config["model.layers.0.mlp.experts"]
    assert layer_quant_config["model.layers.1.mlp.experts"].quant_algo == quant_config.quant_algo
    assert quant_config.quant_algo == QuantAlgo.W4A8_MXFP4_MXFP8
    assert quant_config.group_size == 32


def test_deepseek_v4_routed_moe_quant_config_covers_mtp_layers(tmp_path, monkeypatch):
    monkeypatch.setattr("tensorrt_llm._torch.model_config.get_sm_version", lambda: 100)
    tensor_name = "layers.0.ffn.experts.0.w1.weight"
    shard_name = "model-00001-of-00001.safetensors"
    _write_safetensors_header(tmp_path / shard_name, tensor_name, "I8", [2, 2])
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    tensor_name: shard_name,
                }
            }
        )
    )

    class MTPMode:
        @staticmethod
        def is_mtp_one_model():
            return True

    class MTPConfig:
        spec_dec_mode = MTPMode()
        num_nextn_predict_layers = 3

    layer_quant_config = ModelConfig._set_deepseek_v4_routed_moe_quant_config(
        DeepseekV4Config(num_hidden_layers=2), str(tmp_path), "TRTLLM", None, MTPConfig()
    )

    quant_algo = layer_quant_config["model.layers.0.mlp.experts"].quant_algo
    for layer_idx in range(1, 5):
        assert layer_quant_config[f"model.layers.{layer_idx}.mlp.experts"].quant_algo == quant_algo


def test_deepseek_v4_mtp_projection_uses_fp8_quant_config(monkeypatch):
    def fake_decoder_layer_init(self, model_config, *_args, **_kwargs):
        torch.nn.Module.__init__(self)
        self.model_config = model_config
        self.config = model_config.pretrained_config

    monkeypatch.setattr(DeepseekV4DecoderLayer, "__init__", fake_decoder_layer_init)
    monkeypatch.setattr(torch.cuda, "Event", lambda: object())
    monkeypatch.setattr(
        "tensorrt_llm._torch.distributed.AllReduce", lambda *args, **kwargs: object()
    )

    config = DeepseekV4Config(hidden_size=512, hc_mult=2)
    config.torch_dtype = torch.bfloat16
    quant_config = QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES)
    model_config = ModelConfig(
        pretrained_config=config,
        mapping=Mapping(world_size=4, rank=2, tp_size=4),
        quant_config=quant_config,
    )

    mtp_layer = DeepseekV4MTP(
        model_config,
        layer_idx=config.num_hidden_layers,
        aux_stream_dict={AuxStreamType.MoeShared: object()},
    )

    assert mtp_layer.e_proj.quant_config is quant_config
    assert mtp_layer.h_proj.quant_config is quant_config
    assert mtp_layer.e_proj.tp_mode == TensorParallelMode.ROW
    assert mtp_layer.h_proj.tp_mode == TensorParallelMode.ROW
    assert mtp_layer.e_proj.in_features == config.hidden_size // 4
    assert mtp_layer.h_proj.in_features == config.hidden_size // 4
    assert mtp_layer.e_proj.out_features == config.hidden_size
    assert mtp_layer.h_proj.out_features == config.hidden_size
    assert mtp_layer.e_proj.reduce_output is True
    assert mtp_layer.h_proj.reduce_output is True
    assert mtp_layer.e_proj.weight.dtype is torch.float8_e4m3fn
    assert mtp_layer.h_proj.weight.dtype is torch.float8_e4m3fn
    assert hasattr(mtp_layer.e_proj, "weight_scale")
    assert hasattr(mtp_layer.h_proj, "weight_scale")


def test_deepseek_v4_routed_moe_quant_config_ignores_fp8_header(tmp_path):
    tensor_name = "layers.0.ffn.experts.0.w1.weight"
    shard_name = "model-00001-of-00001.safetensors"
    _write_safetensors_header(tmp_path / shard_name, tensor_name, "F8_E4M3", [2, 2])
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {tensor_name: shard_name}})
    )
    existing = {"existing": object()}

    layer_quant_config = ModelConfig._set_deepseek_v4_routed_moe_quant_config(
        DeepseekV4Config(), str(tmp_path), "TRTLLM", existing
    )

    assert layer_quant_config is existing


def test_deepseek_v4_rope_params_follow_layer_compress_ratio():
    config = DeepseekV4Config(
        compress_ratios=[0, 4],
        rope_theta=10000.0,
        compress_rope_theta=160000.0,
        rope_scaling={
            "type": "yarn",
            "factor": 16.0,
            "original_max_position_embeddings": 65536,
            "beta_fast": 32,
            "beta_slow": 1,
        },
    )
    sparse_attn_config = DeepSeekV4SparseAttentionConfig(
        compress_ratios=[1, 4, 1],
        window_size=128,
    )
    model_config = ModelConfig(pretrained_config=config, sparse_attention_config=sparse_attn_config)

    dense_rope = _deepseek_v4_pos_embd_params(config, model_config, 0)
    compressed_rope = _deepseek_v4_pos_embd_params(config, model_config, 1)
    active_config_rope = _deepseek_v4_pos_embd_params(config, model_config, 2)

    assert dense_rope.type == PositionEmbeddingType.rope_gptj
    assert dense_rope.rope.scale_type == RotaryScalingType.none
    assert dense_rope.rope.theta == 10000.0
    assert dense_rope.rope.scale == 1.0
    assert compressed_rope.type == PositionEmbeddingType.yarn
    assert compressed_rope.rope.scale_type == RotaryScalingType.yarn
    assert compressed_rope.rope.theta == 160000.0
    assert compressed_rope.rope.scale == 16.0
    assert compressed_rope.rope.mscale == 0.0
    assert compressed_rope.rope.mscale_all_dim == 0.0
    assert active_config_rope.type == PositionEmbeddingType.rope_gptj
    assert active_config_rope.rope.scale_type == RotaryScalingType.none
    assert active_config_rope.rope.theta == 10000.0


def test_deepseek_v4_sparse_ratios_prefer_checkpoint_defaults(tmp_path, monkeypatch):
    checkpoint_ratios = [128, 128, 4, 128, 4, 128, 0, 4]
    config = DeepseekV4Config(
        architectures=["DeepseekV4ForCausalLM"],
        num_hidden_layers=len(checkpoint_ratios),
        compress_ratios=checkpoint_ratios,
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.model_config.load_pretrained_config", lambda *args, **kwargs: config
    )
    sparse_attn_config = DeepSeekV4SparseAttentionConfig(
        compress_ratios=[1, 1, 4, 128, 4, 128, 4],
        q_split_threshold=2048,
        skip_indexer_for_short_seqs=False,
    )

    model_config = ModelConfig.from_pretrained(
        str(tmp_path),
        sparse_attention_config=sparse_attn_config,
        attn_backend="TRTLLM",
        moe_backend="TRTLLM",
    )

    assert model_config.sparse_attention_config.compress_ratios == [128, 128, 4, 128, 4, 128, 1, 4]
    # V4 sparse MLA hardcodes window_size==128 (FMHA kernel TileSizeKV; see
    # the runtime assertion in deepseek_v4.py:DeepseekV4TrtllmAttentionMetadata
    # __post_init__), so this is the only legal value here.
    assert model_config.sparse_attention_config.window_size == 128


def test_deepseek_v4_model_config_defaults_to_fp4_indexer(tmp_path, monkeypatch):
    checkpoint_ratios = [128, 128, 4, 128, 4, 128, 0, 4]
    config = DeepseekV4Config(
        architectures=["DeepseekV4ForCausalLM"],
        num_hidden_layers=len(checkpoint_ratios),
        compress_ratios=checkpoint_ratios,
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.model_config.load_pretrained_config", lambda *args, **kwargs: config
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr("tensorrt_llm._utils.get_sm_version", lambda: 100)

    model_config = ModelConfig.from_pretrained(
        str(tmp_path),
        attn_backend="TRTLLM",
        moe_backend="TRTLLM",
    )

    assert model_config.sparse_attention_config.indexer_k_dtype == "fp4"


def test_deepseek_v4_model_config_defaults_to_fp8_before_blackwell(tmp_path, monkeypatch):
    checkpoint_ratios = [128, 128, 4, 128, 4, 128, 0, 4]
    config = DeepseekV4Config(
        architectures=["DeepseekV4ForCausalLM"],
        num_hidden_layers=len(checkpoint_ratios),
        compress_ratios=checkpoint_ratios,
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.model_config.load_pretrained_config", lambda *args, **kwargs: config
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr("tensorrt_llm._utils.get_sm_version", lambda: 90)

    model_config = ModelConfig.from_pretrained(
        str(tmp_path),
        attn_backend="TRTLLM",
        moe_backend="TRTLLM",
    )

    assert model_config.sparse_attention_config.indexer_k_dtype == "fp8"


def test_deepseek_v4_sparse_ratios_keep_checkpoint_length_without_mtp(tmp_path, monkeypatch):
    checkpoint_ratios = [128, 128] + [4, 128] * 29 + [0, 4]
    config = DeepseekV4Config(
        architectures=["DeepseekV4ForCausalLM"],
        num_hidden_layers=61,
        compress_ratios=checkpoint_ratios,
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.model_config.load_pretrained_config", lambda *args, **kwargs: config
    )
    sparse_attn_config = DeepSeekV4SparseAttentionConfig(
        compress_ratios=[1, 1, 4, 128, 4, 128, 4],
    )

    model_config = ModelConfig.from_pretrained(
        str(tmp_path),
        sparse_attention_config=sparse_attn_config,
        attn_backend="TRTLLM",
        moe_backend="TRTLLM",
    )

    assert len(model_config.sparse_attention_config.compress_ratios) == len(checkpoint_ratios)
    assert model_config.sparse_attention_config.compress_ratios[:-2] == (checkpoint_ratios[:-2])
    assert model_config.sparse_attention_config.compress_ratios[-2:] == [1, 4]


def test_deepseek_v4_sparse_ratios_keep_explicit_override(tmp_path, monkeypatch):
    checkpoint_ratios = [128, 128, 4, 128, 4, 128, 0, 4]
    explicit_ratios = [1, 4, 1, 4, 1, 4, 1, 4]
    config = DeepseekV4Config(
        architectures=["DeepseekV4ForCausalLM"],
        num_hidden_layers=len(checkpoint_ratios),
        compress_ratios=checkpoint_ratios,
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.model_config.load_pretrained_config", lambda *args, **kwargs: config
    )
    sparse_attn_config = DeepSeekV4SparseAttentionConfig(
        compress_ratios=explicit_ratios,
    )

    model_config = ModelConfig.from_pretrained(
        str(tmp_path),
        sparse_attention_config=sparse_attn_config,
        attn_backend="TRTLLM",
        moe_backend="TRTLLM",
    )

    assert model_config.sparse_attention_config.compress_ratios == explicit_ratios


def test_deepseek_v4_sparse_ratios_resolve_mtp_layers_from_checkpoint(tmp_path, monkeypatch):
    checkpoint_ratios = [128, 128]
    config = DeepseekV4Config(
        architectures=["DeepseekV4ForCausalLM"],
        num_hidden_layers=len(checkpoint_ratios),
        num_nextn_predict_layers=1,
        compress_ratios=checkpoint_ratios,
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.model_config.load_pretrained_config", lambda *args, **kwargs: config
    )

    spec_config = MTPDecodingConfig(max_draft_len=1)
    model_config = ModelConfig.from_pretrained(
        str(tmp_path),
        attn_backend="TRTLLM",
        moe_backend="TRTLLM",
        spec_config=spec_config,
    )

    assert spec_config.num_nextn_predict_layers == 1
    assert model_config.sparse_attention_config.compress_ratios == [128, 128, 1]


@pytest.mark.parametrize(
    "kv_cache_dtype,tokens_per_block,binding_dtype",
    [
        pytest.param(
            "auto",
            128,
            tensorrt_llm.bindings.DataType.BF16,
            marks=skip_blackwell_geforce,
            id="bf16-kv",
        ),
        pytest.param(
            "fp8_ds_mla",
            256,
            tensorrt_llm.bindings.DataType.FP8,
            marks=pytest.mark.skipif(
                getSMVersion() not in (120, 121),
                reason="FlashInfer sparse MLA requires SM 120 or SM 121",
            ),
            id="fp8-ds-mla",
        ),
    ],
)
def test_deepseek_v4_sanity(
    kv_cache_dtype: str,
    tokens_per_block: int,
    binding_dtype: tensorrt_llm.bindings.DataType,
) -> None:
    config_dict = deepcopy(DEEPSEEK_V4_TINY_CONFIG)
    if kv_cache_dtype == "fp8_ds_mla":
        # Sparse MLA coverage does not depend on the MoE intermediate width.
        # Preserve 256 experts because the real routing kernel requires that
        # topology, but keep its weights small enough for an RTX Pro 6000D.
        config_dict["moe_intermediate_size"] = 128
        config_dict["vocab_size"] = 1024
    config = DeepseekV4Config(**config_dict)
    config.dtype = torch.bfloat16
    config.mapping = Mapping(world_size=1, tp_size=1, rank=0)
    config.tie_word_embeddings = False

    vocab_size = config.vocab_size
    max_batch_size = 4

    sparse_attn_config = DeepSeekV4SparseAttentionConfig(
        index_n_heads=64,
        index_head_dim=128,
        window_size=128,
        compress_ratios=[1, 1, 4, 128, 4, 128, 4],
        index_topk=512,
    )
    config.sparse_attention_config = sparse_attn_config

    device = torch.device("cuda")
    # with default_dtype(config.dtype):
    quant_config = QuantConfig(
        kv_cache_quant_algo=QuantAlgo.FP8 if kv_cache_dtype == "fp8_ds_mla" else None
    )
    model_config = ModelConfig(
        pretrained_config=config,
        sparse_attention_config=sparse_attn_config,
        attn_backend="TRTLLM",
        quant_config=quant_config,
    )
    model_config.extra_attrs["kv_cache_dtype"] = kv_cache_dtype
    model = DeepseekV4ForCausalLM(model_config).to(device)
    assert not model.model.layers[0].fusion_config.POST_MOE_FUSION
    fmha_libs = model.model.layers[0].self_attn.mqa._fmha_manager.fmha_libs
    if kv_cache_dtype == "fp8_ds_mla":
        assert any(isinstance(fmha, FlashInferSparseMlaFmha) for fmha in fmha_libs)
        assert not any(isinstance(fmha, FallbackFmha) for fmha in fmha_libs)
    else:
        assert any(isinstance(fmha, FallbackFmha) for fmha in fmha_libs)

    context_sequence_length = [3, 2, 5]
    num_contexts = len(context_sequence_length)
    sequence_length = context_sequence_length + [1, 1]

    # Total tokens = sum(sequence_length) = 3+2+5+1+1 = 12
    input_ids = torch.randint(
        0, vocab_size, (sum(sequence_length),), dtype=torch.int32, device=device
    )
    past_seen_tokens = [0, 0, 0, 62, 75]
    request_ids = list(range(len(sequence_length)))
    token_nums = (torch.tensor(past_seen_tokens) + torch.tensor(sequence_length)).tolist()
    prompt_lens = token_nums[:num_contexts] + past_seen_tokens[num_contexts:]
    max_new_tokens = 1024
    required_blocks = sum(
        (token_num + max_new_tokens + tokens_per_block - 1) // tokens_per_block
        for token_num in token_nums
    )
    num_blocks = max(10, required_blocks)
    head_dim = config.v_head_dim
    num_layers = config.num_hidden_layers
    max_seq_len = num_blocks * tokens_per_block
    batch_size = len(sequence_length)

    mapping = config.mapping
    kv_cache_config = KvCacheConfig(
        dtype=kv_cache_dtype,
        enable_block_reuse=False,
        max_tokens=num_blocks * tokens_per_block,
        event_buffer_max_size=0,
    )

    kv_cache_manager = DeepseekV4CacheManager(
        kv_cache_config=kv_cache_config,
        kv_cache_type=tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
        num_layers=num_layers,
        num_kv_heads=1,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
        mapping=mapping,
        dtype=binding_dtype,
        compressor_dtype=tensorrt_llm.bindings.DataType.FLOAT,
        vocab_size=vocab_size,
        max_num_tokens=max_seq_len * max_batch_size,
        sparse_attn_config=sparse_attn_config,
        model_config=model_config,
    )
    # Register request IDs in KV cache via prepare_context / resize_context
    reqs = []
    for i, req_id in enumerate(request_ids):
        req = LlmRequest(
            request_id=req_id,
            max_new_tokens=max_new_tokens,
            input_tokens=list(range(token_nums[i])),
            sampling_config=SamplingConfig(),
            is_streaming=False,
        )
        success = kv_cache_manager.prepare_context(req)
        assert success, f"Failed to prepare context for request {req_id}"
        if i < num_contexts:
            success = kv_cache_manager.resize_context(req, req.context_chunk_size)
        else:
            # Warm-cache setup for a generation request: simulate
            # past_seen_tokens[i] worth of history without running forward.
            # Reach into kv_cache.resize directly because resize_context no
            # longer exposes a history_length override (production callers
            # use prepare_disagg_gen_init or update_resources to advance it).
            kv_cache = kv_cache_manager.kv_cache_map[req.py_request_id]
            kv_cache.enable_swa_scratch_reuse = False
            target = (
                req.context_current_position + token_nums[i] + kv_cache_manager.num_extra_kv_tokens
            )
            capacity = max(kv_cache.capacity, target)
            success = kv_cache.resize(capacity, past_seen_tokens[i])
        assert success, f"Failed to resize context for request {req_id}"
        reqs.append(req)

    attn_metadata = DeepseekV4TrtllmAttentionMetadata(
        seq_lens=torch.tensor(sequence_length, dtype=torch.int32),
        num_contexts=num_contexts,
        max_num_requests=len(sequence_length),
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=past_seen_tokens,
        ),
        kv_cache_manager=kv_cache_manager,
        request_ids=request_ids,
        prompt_lens=prompt_lens,
        max_num_tokens=8192,
        mapping=mapping,
        sparse_attention_config=sparse_attn_config,
    )

    position_ids = []
    seq_lens = []
    for i, tokens in enumerate(past_seen_tokens):
        seq_len = context_sequence_length[i] if i < len(context_sequence_length) else 1
        position_id = torch.arange(tokens, tokens + seq_len, device=input_ids.device)
        position_ids.append(position_id)
        seq_lens.append(seq_len)

    position_ids = torch.cat(position_ids).unsqueeze(0).to(torch.int32)

    extra_attrs = model_config.extra_attrs
    extra_attrs["attention_metadata"] = weakref.ref(attn_metadata)
    with torch.inference_mode(), model_extra_attrs(extra_attrs):
        scheduled_batch = ScheduledRequests()
        scheduled_batch.context_requests_last_chunk = reqs[:num_contexts]
        scheduled_batch.generation_requests = reqs[num_contexts:]
        kv_cache_manager.prepare_resources(scheduled_batch)
        attn_metadata.prepare()

        logits = model.forward(
            input_ids=input_ids, position_ids=position_ids, attn_metadata=attn_metadata
        )

        for req in reqs[:num_contexts]:
            req.context_current_position = seq_lens[req.py_request_id]
        for req in reqs:
            req.add_new_token(seq_lens[req.py_request_id], 0)
        kv_cache_manager.update_context_resources(scheduled_batch)
        kv_cache_manager.update_resources(scheduled_batch)
    assert len(past_seen_tokens) == logits.shape[0]

    extra_attrs["attention_metadata"] = weakref.ref(attn_metadata)
    with torch.inference_mode(), model_extra_attrs(extra_attrs):
        seq_lens = [seq_len + 1 for seq_len in seq_lens]
        scheduled_batch = ScheduledRequests()
        scheduled_batch.generation_requests = reqs
        for req in reqs:
            assert kv_cache_manager.try_allocate_generation(req)
        kv_cache_manager.prepare_resources(scheduled_batch)
        attn_metadata.prepare()
        logits = model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attn_metadata=attn_metadata,
            return_context_logits=True,
        )
        for req in reqs:
            req.add_new_token(seq_lens[req.py_request_id], 0)
        kv_cache_manager.update_resources(scheduled_batch)
    assert input_ids.shape == logits.shape[:-1]

    for req in reqs:
        kv_cache_manager.free_resources(req)
    kv_cache_manager.shutdown()


# ---------------------------------------------------------------------------
# Dispatch coverage for the MLA prologue fusions: which path is taken, not
# the numerics. Merged from test_mla_dsv4_fusion_dispatch.py.
# ---------------------------------------------------------------------------


# DSv4-Pro latent geometry: 448 nope + 64 rope = a 512-wide head.
KV_LORA_RANK = 448
QK_ROPE_HEAD_DIM = 64


class _FakeAttention(nn.Module):
    """Stands in for the attention backend; only the fusion inputs matter."""

    def __init__(self, has_fp8_kv_cache: bool):
        super().__init__()
        self.has_fp8_kv_cache = has_fp8_kv_cache
        self.rotary_cos_sin = torch.zeros(8, dtype=torch.float32)

    def support_fused_rope(self) -> bool:
        return True

    def update_quant_config(self, _quant_config: object) -> None:
        pass

    def _ensure_rope_table_size(self, _max_seq_len: int) -> None:
        pass


def _make_mla(
    *,
    has_fp8_kv_cache: bool,
    dsv4_geometry: bool = True,
    kv_lora_rank: int = KV_LORA_RANK,
    qk_rope_head_dim: int = QK_ROPE_HEAD_DIM,
) -> MLA:
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    config = ModelConfig(skip_create_weights_in_init=True)
    position_embedding = PositionalEmbeddingParams(
        type=PositionEmbeddingType.rope_gpt_neox,
        rope=RopeParams(dim=QK_ROPE_HEAD_DIM, max_positions=8192),
    )
    with patch(
        "tensorrt_llm._torch.attention.mla.create_attention",
        side_effect=lambda *a, **kw: _FakeAttention(has_fp8_kv_cache),
    ):
        mla = MLA(
            hidden_size=64,
            num_attention_heads=2,
            num_key_value_heads=1,
            qk_nope_head_dim=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=kv_lora_rank,
            q_lora_rank=32,
            kv_lora_rank=kv_lora_rank,
            predicted_tokens_per_seq=1,
            max_position_embeddings=8192,
            bias=False,
            pos_embd_params=position_embedding,
            layer_idx=0,
            dtype=torch.bfloat16,
            config=config,
        )
    # DSv4 widens kv_a_layernorm to the whole 512 latent, and
    # `_is_fused_kv_norm_enabled` checks that width. Do NOT fabricate attributes the
    # real module lacks: the predicates live in the DSv4 sparse module, so reaching
    # them already implies DSv4 and only the geometry is actually checked.
    if dsv4_geometry:
        mla.kv_a_layernorm = RMSNorm(
            hidden_size=kv_lora_rank + qk_rope_head_dim, dtype=torch.bfloat16, eps=1e-6
        )
    return mla


def _make_metadata(
    *, num_ctx_tokens: int, num_tokens: int, num_seqs: int, num_contexts: int = 0
) -> SimpleNamespace:
    """Only the five attributes `_fused_q_rope_specs` reads."""
    # Production returns `mla_ctx_cu_q_seqlens[:num_contexts + 1]`, so match that length.
    cu_ctx = torch.zeros(num_contexts + 1, dtype=torch.int32)
    return SimpleNamespace(
        kv_lens_cuda_runtime=torch.arange(num_seqs, dtype=torch.int32),
        num_ctx_tokens=num_ctx_tokens,
        num_tokens=num_tokens,
        max_seq_len=8192,
        mla_prepare_ctx_cu_seqlens=lambda: cu_ctx,
    )


@pytest.mark.parametrize("has_fp8_kv_cache", [True, False])
def test_fusions_require_fp8_kv_cache(has_fp8_kv_cache: bool) -> None:
    """Both predicates hang off the KV-cache dtype.

    A bf16 configuration must report the fusions off. Without this the
    backend-level suites, which build a bf16 cache, look like coverage while
    never entering a fused path.
    """
    mla = _make_mla(has_fp8_kv_cache=has_fp8_kv_cache)
    assert _is_fused_kv_norm_enabled(mla, num_generations=1) is has_fp8_kv_cache
    assert _is_fused_q_fp8_quant_enabled(mla, num_generations=1, num_contexts=0) is has_fp8_kv_cache


def test_kv_norm_fusion_needs_the_full_width_weight() -> None:
    """The KV kernels norm the whole 512 latent, so a 448-wide weight must bail.

    This is the guard against an out-of-bounds read, not a style check: the kernel
    indexes `kv_norm_weight` across `K_DIM + ROPE_DIM` regardless of its length.
    """
    mla = _make_mla(has_fp8_kv_cache=True, dsv4_geometry=False)
    assert mla.kv_a_layernorm.weight.shape[0] == KV_LORA_RANK
    assert _is_fused_kv_norm_enabled(mla, num_generations=1) is False


def test_rope_specs_mixed_batch_splits_by_phase() -> None:
    """The regression this file exists for, and the only per-phase spec test.

    Pure-context and pure-generation cases were dropped: mutation attribution
    showed the generation-only test killed no mutant, and every mutant the
    context-only test killed is also killed here. A mixed batch exercises both
    position rules at once, so it strictly dominates them.

    A mixed batch needs both position rules, so it gets one spec per phase. When
    this returned nothing the fused path silently fell back to
    `applyMLARopeAndAssignQKVKernel*` and no test noticed.
    """
    # 2 context sequences (96 tokens) + 3 generation sequences (3 tokens).
    metadata = _make_metadata(num_ctx_tokens=96, num_tokens=99, num_seqs=5, num_contexts=2)
    mla = _make_mla(has_fp8_kv_cache=True)

    cos_sin, specs = _fused_q_rope_specs(mla, metadata, num_contexts=2, num_generations=3)

    assert cos_sin is not None
    assert len(specs) == 2, "mixed batch must not fall back to a single launch"

    (
        (ctx_rows, ctx_cache_lens, ctx_seq_len, ctx_cu),
        (
            gen_rows,
            gen_cache_lens,
            gen_seq_len,
            gen_cu,
        ),
    ) = specs

    # Context first, generation second, disjoint and covering every row exactly once.
    assert ctx_rows == slice(0, 96)
    assert gen_rows == slice(96, 99)
    assert ctx_rows.stop == gen_rows.start

    assert ctx_seq_len == 0 and ctx_cu is not None
    assert gen_seq_len == 1 and gen_cu is None

    # Each half sees only its own sequences' cache lengths.
    assert ctx_cache_lens.shape[0] == 2
    assert gen_cache_lens.shape[0] == 3


def test_kv_norm_fusion_is_coupled_to_the_q_rope_fold() -> None:
    """The two fusions must move together.

    The KV fusion hands the un-fused RoPE kernels the RAW latent, so their Q
    region would read it un-normalized. That is only safe because the fused Q
    path takes the Q side over entirely -- enabling one without the other is a
    silent correctness bug, not a slower path.
    """
    mla = _make_mla(has_fp8_kv_cache=True)
    metadata = _make_metadata(num_ctx_tokens=96, num_tokens=96, num_seqs=3, num_contexts=3)
    metadata.mla_prepare_ctx_cu_seqlens = None  # forces the Q fold off

    _cos_sin, specs = _fused_q_rope_specs(mla, metadata, num_contexts=3, num_generations=0)
    assert not specs

    # The KV predicate on its own still says yes -- so the coupling, not a shared
    # precondition, is what has to turn the fusion off.
    assert _is_fused_kv_norm_enabled(mla, num_generations=0) is True

    # `forward_impl_with_deepseek_v4` assigns `_fused_kv_norm_active` from exactly
    # this call, so asserting on it here is asserting on the shipped decision.
    assert (
        _is_fused_prologue_active(mla, num_contexts=3, num_generations=0, rope_specs=specs) is False
    ), "kv-norm fusion must not engage when the Q RoPE fold is unavailable"

    # ...and it does engage once the specs exist, so the False above is the coupling
    # talking and not a predicate that is off for some unrelated reason.
    assert (
        _is_fused_prologue_active(
            mla, num_contexts=3, num_generations=0, rope_specs=[("dummy", None, 0, None)]
        )
        is True
    )


@pytest.mark.parametrize(
    "kv_lora_rank,qk_rope_head_dim",
    [(512, 64), (448, 128)],
    ids=["lora512", "rope128"],
)
def test_fusions_require_the_448_64_latent(kv_lora_rank: int, qk_rope_head_dim: int) -> None:
    """The kernels hard-code the latent row in template constants.

    `mlaKvNormRopeQuant*Kernel` is instantiated at K_DIM=448 / ROPE_DIM=64, so a
    model whose latent is shaped differently must not reach it -- the kernel would
    stride the wrong row width. Every other fixture here builds DSv4 geometry, so
    without this case the guard is unreachable and deleting it breaks no test.
    """
    mla = _make_mla(
        has_fp8_kv_cache=True,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
    )
    assert _is_fused_kv_norm_enabled(mla, num_generations=1) is False


# ---------------------------------------------------------------------------
# Numerics for the Q RoPE fold in `deepseek_v4_q_norm_fused_fp8`.
# Merged from test_deepseek_v4_q_norm_fused_rope.py.
# ---------------------------------------------------------------------------


# DSv4-Pro Q geometry: 448 nope + 64 rope per head.
HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = HEAD_DIM - NOPE_DIM
EPS = 1e-6
QUANT_SCALE = 0.5
MAX_POSITIONS = 512


def _make_cos_sin(device: torch.device) -> torch.Tensor:
    """Rope table in the layout every MLA kernel here indexes.

    The pointer is `float2 const*` strided by ROPE_DIM per position, so a row is
    ROPE_DIM float2 entries even though only the first ROPE_DIM/2 -- one per
    rotated pair -- are ever read. Filling the unused tail with NaN keeps a
    stride mistake from silently reading plausible numbers.
    """
    table = torch.full((MAX_POSITIONS, ROPE_DIM, 2), float("nan"), dtype=torch.float32)
    # Angles are decorrelated across positions on purpose. A smooth table (e.g.
    # linspace) makes neighbouring positions nearly identical, and an off-by-one
    # in the position arithmetic then lands inside any sane tolerance.
    generator = torch.Generator().manual_seed(1234)
    angles = torch.rand((MAX_POSITIONS, ROPE_DIM // 2), generator=generator) * (2 * torch.pi)
    table[:, : ROPE_DIM // 2, 0] = torch.cos(angles)
    table[:, : ROPE_DIM // 2, 1] = torch.sin(angles)
    return table.to(device)


def _reference(
    q: torch.Tensor, cos_sin: torch.Tensor, positions: torch.Tensor, num_heads: int
) -> torch.Tensor:
    """RMS-norm over the whole 512-wide head, rotate the tail, scale for FP8."""
    num_tokens = q.shape[0]
    x = q.view(num_tokens, num_heads, HEAD_DIM).float()
    inv_rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
    normed = x * inv_rms

    nope = normed[..., :NOPE_DIM] * QUANT_SCALE

    # GPT-J interleave: pair j is elements (2j, 2j+1) of the rope tail.
    rope = normed[..., NOPE_DIM:]
    even, odd = rope[..., 0::2], rope[..., 1::2]
    coef = cos_sin[positions][:, : ROPE_DIM // 2, :]  # [tokens, 32, 2]
    cos = coef[..., 0].unsqueeze(1)  # broadcast over heads
    sin = coef[..., 1].unsqueeze(1)
    rotated = torch.empty_like(rope)
    rotated[..., 0::2] = cos * even - sin * odd
    rotated[..., 1::2] = cos * odd + sin * even

    return torch.cat([nope, rotated * QUANT_SCALE], dim=-1)


def _run_op(q, num_heads, cos_sin, cache_seq_lens, seq_len, cu_q_seqlens):
    num_tokens = q.shape[0]
    quant_q = torch.zeros(
        (num_tokens, num_heads * HEAD_DIM), dtype=torch.float8_e4m3fn, device=q.device
    )
    # Sentinel: the fold must leave q_pe untouched, because the rope tail is
    # supposed to land in quant_q instead.
    q_pe = torch.full((num_tokens, num_heads * ROPE_DIM), 7.0, dtype=q.dtype, device=q.device)
    quant_scale = torch.tensor([QUANT_SCALE], dtype=torch.float32, device=q.device)

    torch.ops.trtllm.deepseek_v4_q_norm_fused_fp8(
        q,
        quant_q,
        q_pe,
        num_heads,
        HEAD_DIM,
        NOPE_DIM,
        EPS,
        quant_scale,
        cos_sin,
        cache_seq_lens,
        seq_len,
        cu_q_seqlens,
    )
    return quant_q, q_pe


def _assert_matches(quant_q, q_pe, reference, num_heads):
    """Compare in FP8, not in float.

    A relative tolerance on the dequantized values has to be at least one e4m3
    step (~13%) to absorb rounding, and that is wide enough to swallow real bugs
    -- normalizing over 448 dims instead of 512 is only a 6.9% shift. So quantize
    the reference the same way and require the codes to agree. The kernel folds
    inv_rms and the quant scale into a single multiply where the reference uses
    two, so a few values sit on the other side of a rounding boundary; those get
    a small budget, capped at one FP8 step each.
    """
    num_tokens = quant_q.shape[0]
    got = quant_q.view(num_tokens, num_heads, HEAD_DIM).float()
    expected = reference.to(torch.float8_e4m3fn).float()

    differing = got != expected
    frac = differing.float().mean().item()
    assert frac < 0.01, f"{frac:.4%} of FP8 codes differ from the reference"

    if differing.any():
        scale = torch.maximum(got.abs(), expected.abs()).clamp_min(1e-6)
        worst = ((got - expected).abs() / scale)[differing].max().item()
        assert worst < 0.13, f"a differing code is off by more than one FP8 step ({worst:.3f})"

    assert torch.all(q_pe == 7.0), (
        "q_pe was written; the rope tail must go to quant_q on the fused path"
    )


@skip_pre_blackwell
@pytest.mark.parametrize(
    "num_heads,seq_len",
    [(4, 2), (6, 3)],
    ids=["heads4_seqlen2_pow2", "heads6_seqlen3_divide"],
)
def test_fused_rope_generation_positions(num_heads: int, seq_len: int) -> None:
    """Uniform query length: position = cache_len[batch] - seq_len + local_token.

    The parameters flip both power-of-two shortcuts the kernel takes (`row /
    num_heads` and `token / seq_len` become shift/mask only when the host says
    the divisor is a power of two), so the ids name which arithmetic runs.
    """
    device = torch.device("cuda")
    torch.manual_seed(0)
    num_seqs = 3
    num_tokens = num_seqs * seq_len

    q = torch.randn((num_tokens, num_heads * HEAD_DIM), dtype=torch.bfloat16, device=device)
    cos_sin = _make_cos_sin(device)
    cache_seq_lens = torch.tensor([16, 40, 71], dtype=torch.int32, device=device)

    quant_q, q_pe = _run_op(q, num_heads, cos_sin, cache_seq_lens, seq_len, None)

    token = torch.arange(num_tokens, device=device)
    positions = cache_seq_lens[token // seq_len] - seq_len + (token % seq_len)
    reference = _reference(q, cos_sin, positions.long(), num_heads)
    _assert_matches(quant_q, q_pe, reference, num_heads)


@skip_pre_blackwell
@pytest.mark.parametrize(
    "num_heads,cached_offset",
    [(4, 0), (6, 5)],
    ids=["heads4_fresh_prefill", "heads6_chunked_prefill"],
)
def test_fused_rope_context_positions(num_heads: int, cached_offset: int) -> None:
    """Ragged: position = local_token + (cache_len[seq] - current_seq_len).

    `cached_offset > 0` is the chunked-prefill / block-reuse case, where part of
    the sequence is already in the KV cache and this chunk's first token is not
    at position 0. Every other test in the suite pins it to zero by disabling
    block reuse, so the second parameter is the only thing that walks that term.
    """
    device = torch.device("cuda")
    torch.manual_seed(0)
    seq_lens = [5, 3, 7]
    num_tokens = sum(seq_lens)

    cu_q_seqlens = torch.tensor(
        [0, *torch.tensor(seq_lens).cumsum(0).tolist()], dtype=torch.int32, device=device
    )
    cache_seq_lens = torch.tensor(
        [s + cached_offset for s in seq_lens], dtype=torch.int32, device=device
    )
    q = torch.randn((num_tokens, num_heads * HEAD_DIM), dtype=torch.bfloat16, device=device)
    cos_sin = _make_cos_sin(device)

    quant_q, q_pe = _run_op(q, num_heads, cos_sin, cache_seq_lens, 0, cu_q_seqlens)

    positions = torch.cat(
        [torch.arange(length, device=device) + cached_offset for length in seq_lens]
    )
    reference = _reference(q, cos_sin, positions.long(), num_heads)
    _assert_matches(quant_q, q_pe, reference, num_heads)
