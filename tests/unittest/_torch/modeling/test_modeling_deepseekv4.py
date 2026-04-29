import inspect
import json
import struct
import weakref
from copy import deepcopy

import pytest
import torch
from transformers import PretrainedConfig

# from utils.util import default_dtype
import tensorrt_llm
from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4.cache_manager import (
    DeepseekV4CacheManager,
)
from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4.compressor import Compressor
from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4.deepseek_v4 import (
    DeepseekV4Indexer,
    DeepseekV4TrtllmAttention,
    DeepseekV4TrtllmAttentionMetadata,
)
from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention
from tensorrt_llm._torch.configs.deepseekv4 import DeepseekV4Config
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_deepseekv4 import (
    DeepseekV4DecoderLayer,
    DeepseekV4ForCausalLM,
    DeepseekV4Gate,
    DeepseekV4MoE,
    DeepseekV4MTP,
    _deepseek_v4_pos_embd_params,
    _remap_deepseek_v4_checkpoint_keys,
    _resolve_enable_fused_hc,
)
from tensorrt_llm._torch.modules.linear import TensorParallelMode
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.utils import AuxStreamType, model_extra_attrs
from tensorrt_llm.functional import PositionEmbeddingType, RotaryScalingType
from tensorrt_llm.llmapi.llm_args import DeepSeekV4SparseAttentionConfig, KvCacheConfig
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


def test_deepseek_v4_model_defaults_keep_tokens_per_block():
    class LlmArgs:
        pass

    defaults = DeepseekV4ForCausalLM.get_model_defaults(LlmArgs())

    assert defaults == {"kv_cache_config": {"tokens_per_block": 128}}


def test_deepseek_v4_weight_remap_for_mxfp4_routed_experts():
    weights = {
        "layers.0.ffn.experts.0.w1.weight": torch.tensor([[-1, 2], [3, -4]], dtype=torch.int8),
        "layers.0.ffn.experts.0.w1.scale": torch.tensor([1, 2], dtype=torch.int8),
    }

    remapped = _remap_deepseek_v4_checkpoint_keys(weights, num_hidden_layers=1, kv_lora_rank=448)

    assert remapped["model.layers.0.mlp.experts.0.w1.weight"].dtype == torch.uint8
    assert remapped["model.layers.0.mlp.experts.0.w1.weight_scale"].dtype == torch.uint8


def test_deepseek_v4_weight_remap_for_fp8_routed_experts():
    weights = {
        "layers.0.ffn.experts.0.w1.weight": torch.zeros((2, 2), dtype=torch.float32),
        "layers.0.ffn.experts.0.w1.scale": torch.ones((2, 2), dtype=torch.float32),
    }

    remapped = _remap_deepseek_v4_checkpoint_keys(weights, num_hidden_layers=1, kv_lora_rank=448)

    assert "model.layers.0.mlp.experts.0.w1.weight_scale_inv" in remapped
    assert "model.layers.0.mlp.experts.0.w1.weight_scale" not in remapped


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
    )
    hidden_states = torch.tensor([[1.0, -2.0, 3.0, -4.0]], dtype=torch.bfloat16)
    weight = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [-1.0, 1.0, -1.0, 1.0], [0.5, -0.5, 0.25, -0.25]],
        dtype=torch.bfloat16,
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


def test_deepseek_v4_mla_q_b_layernorm_init_and_forward_shape():
    from tensorrt_llm._torch.modules.attention import MLA

    init_src = inspect.getsource(MLA.__init__)
    forward_src = inspect.getsource(MLA.forward_impl_with_deepseek_v4)

    assert "self.q_b_layernorm = RMSNorm(hidden_size=self.qk_head_dim" in init_src
    assert "has_weights=False" in init_src
    assert "kv_a_layernorm_hidden_size = (" in init_src
    assert "self.kv_lora_rank + self.qk_rope_head_dim" in init_src
    assert "self.kv_a_layernorm = RMSNorm(hidden_size=kv_a_layernorm_hidden_size" in init_src
    assert "self.q_b_layernorm(q.view(-1, self.qk_head_dim)).view_as(q)" in forward_src


def test_deepseek_v4_compressor_rotate_and_indexer_rope_contracts():
    assert inspect.signature(Compressor).parameters["rotate_activation"].default is False

    indexer_init = inspect.getsource(DeepseekV4Indexer.__init__)
    assert "is_neox=False" in indexer_init
    assert "rotate_activation=True" in indexer_init

    attention_init = inspect.getsource(DeepseekV4TrtllmAttention.__init__)
    assert "rotate_activation=False" in attention_init


def test_deepseek_v4_moe_swiglu_limit_applies_to_routed_and_shared_experts():
    moe_init = inspect.getsource(DeepseekV4MoE.__init__)
    assert "moe_swiglu_limit = None" in moe_init
    assert "supports_swiglu_limit = False" in moe_init
    assert "mode.has_w4a8_mxfp4_mxfp8()" in moe_init
    assert "swiglu_limit=moe_swiglu_limit" in moe_init

    shared_expert_block = moe_init.split("self.shared_experts = GatedMLP", 1)[1].split(
        "self.allreduce", 1
    )[0]
    assert "swiglu_limit=swiglu_limit" in shared_expert_block


def test_deepseek_v4_attention_forward_injects_attn_sink(monkeypatch):
    captured = {}

    def fake_forward(self, *args, **kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(TrtllmAttention, "forward", fake_forward)
    attn = object.__new__(DeepseekV4TrtllmAttention)
    sink = torch.ones(4, dtype=torch.float32)
    attn.attn_sink = torch.nn.Parameter(sink, requires_grad=False)

    assert DeepseekV4TrtllmAttention.forward(attn, "q") == "ok"
    assert captured["attention_sinks"].data_ptr() == sink.data_ptr()


def test_deepseek_v4_moe_auto_backend_on_blackwell(monkeypatch):
    monkeypatch.setattr("tensorrt_llm._torch.model_config.get_sm_version", lambda: 100)

    assert ModelConfig.resolve_moe_backend("AUTO", "DeepseekV4ForCausalLM") == "TRTLLM"


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

def test_deepseek_v4_sparse_ratios_prefer_checkpoint_defaults(
        tmp_path, monkeypatch):
    checkpoint_ratios = [128, 128, 4, 128, 4, 128, 0, 4]
    config = DeepseekV4Config(
        architectures=["DeepseekV4ForCausalLM"],
        num_hidden_layers=len(checkpoint_ratios),
        compress_ratios=checkpoint_ratios,
        sliding_window=256,
    )
    monkeypatch.setattr("tensorrt_llm._torch.model_config.load_pretrained_config",
                        lambda *args, **kwargs: config)
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

    assert model_config.sparse_attention_config.compress_ratios == [
        128, 128, 4, 128, 4, 128, 1, 4
    ]
    assert model_config.sparse_attention_config.window_size == 256


def test_deepseek_v4_sparse_ratios_keep_checkpoint_length_without_mtp(
        tmp_path, monkeypatch):
    checkpoint_ratios = [128, 128] + [4, 128] * 29 + [0, 4]
    config = DeepseekV4Config(
        architectures=["DeepseekV4ForCausalLM"],
        num_hidden_layers=61,
        compress_ratios=checkpoint_ratios,
    )
    monkeypatch.setattr("tensorrt_llm._torch.model_config.load_pretrained_config",
                        lambda *args, **kwargs: config)
    sparse_attn_config = DeepSeekV4SparseAttentionConfig(
        compress_ratios=[1, 1, 4, 128, 4, 128, 4],
    )

    model_config = ModelConfig.from_pretrained(
        str(tmp_path),
        sparse_attention_config=sparse_attn_config,
        attn_backend="TRTLLM",
        moe_backend="TRTLLM",
    )

    assert len(model_config.sparse_attention_config.compress_ratios) == len(
        checkpoint_ratios)
    assert model_config.sparse_attention_config.compress_ratios[:-2] == (
        checkpoint_ratios[:-2])
    assert model_config.sparse_attention_config.compress_ratios[-2:] == [1, 4]


def test_deepseek_v4_sparse_ratios_keep_explicit_override(
        tmp_path, monkeypatch):
    checkpoint_ratios = [128, 128, 4, 128, 4, 128, 0, 4]
    explicit_ratios = [1, 4, 1, 4, 1, 4, 1, 4]
    config = DeepseekV4Config(
        architectures=["DeepseekV4ForCausalLM"],
        num_hidden_layers=len(checkpoint_ratios),
        compress_ratios=checkpoint_ratios,
    )
    monkeypatch.setattr("tensorrt_llm._torch.model_config.load_pretrained_config",
                        lambda *args, **kwargs: config)
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


def test_deepseek_v4_sanity():
    config_dict = deepcopy(DEEPSEEK_V4_TINY_CONFIG)
    config = PretrainedConfig(**config_dict)
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
    model_config = ModelConfig(
        pretrained_config=config, sparse_attention_config=sparse_attn_config, attn_backend="TRTLLM"
    )
    model = DeepseekV4ForCausalLM(model_config).to(device)
    assert not model.model.layers[0].fusion_config.POST_MOE_FUSION

    context_sequence_length = [3, 2, 5]
    sequence_length = context_sequence_length + [1, 1]

    # Total tokens = sum(sequence_length) = 3+2+5+1+1 = 12
    input_ids = torch.randint(
        0, vocab_size, (sum(sequence_length),), dtype=torch.int32, device=device
    )
    past_seen_tokens = [0, 0, 0, 62, 75]
    request_ids = list(range(len(sequence_length)))
    token_nums = (torch.tensor(past_seen_tokens) + torch.tensor(sequence_length)).tolist()
    prompt_lens = token_nums[:3] + past_seen_tokens[3:]
    tokens_per_block = 128  # DeepSeek-V4 requirement
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

    if config.dtype == torch.half:
        kv_cache_dtype = tensorrt_llm.bindings.DataType.HALF
    elif config.dtype == torch.bfloat16:
        kv_cache_dtype = tensorrt_llm.bindings.DataType.BF16
    else:
        raise ValueError("Invalid dtype")
    mapping = config.mapping
    kv_cache_config = KvCacheConfig(max_tokens=num_blocks * tokens_per_block)
    kv_cache_config.max_util_for_resume = 0.1

    kv_cache_manager = DeepseekV4CacheManager(
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=False,
            max_tokens=num_blocks * tokens_per_block,
            event_buffer_max_size=0,
        ),
        kv_cache_type=tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
        num_layers=num_layers,
        num_kv_heads=1,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
        mapping=mapping,
        dtype=kv_cache_dtype,
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
        # Allocate enough capacity for context tokens plus generation headroom
        success = kv_cache_manager.resize_context(req, token_nums[i] + max_new_tokens)
        assert success, f"Failed to resize context for request {req_id}"
        reqs.append(req)

    attn_metadata = DeepseekV4TrtllmAttentionMetadata(
        seq_lens=torch.tensor(sequence_length, dtype=torch.int32),
        num_contexts=len(context_sequence_length),
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
        scheduled_batch.context_requests_last_chunk = reqs
        kv_cache_manager.prepare_resources(scheduled_batch)
        attn_metadata.prepare()

        logits = model.forward(
            input_ids=input_ids, position_ids=position_ids, attn_metadata=attn_metadata
        )

        for req in reqs:
            req.context_current_position = seq_lens[req.py_request_id]
            req.add_new_token(seq_lens[req.py_request_id], 0)
        kv_cache_manager.update_resources(scheduled_batch)
    assert len(past_seen_tokens) == logits.shape[0]

    extra_attrs["attention_metadata"] = weakref.ref(attn_metadata)
    with torch.inference_mode(), model_extra_attrs(extra_attrs):
        seq_lens = [seq_len + 1 for seq_len in seq_lens]
        scheduled_batch = ScheduledRequests()
        scheduled_batch.generation_requests = reqs
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
