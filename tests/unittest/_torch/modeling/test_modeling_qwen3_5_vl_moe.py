# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import List, Optional

import torch
import transformers
from test_modeling_multimodal import MultimodalScenario, TestModelingMultimodal
from transformers import Qwen3_5MoeForConditionalGeneration as HFQwen3_5MoeForConditionalGeneration
from utils.llm_data import llm_models_root
from utils.util import skip_pre_hopper

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models import Qwen3_5MoeVLModel
from tensorrt_llm._torch.models.checkpoints.auto_mapper import AutoCheckpointMapper
from tensorrt_llm._torch.models.checkpoints.hf.qwen3_5_weight_mapper import Qwen3_5MoeHfWeightMapper
from tensorrt_llm._torch.models.modeling_auto import AutoModelForCausalLM
from tensorrt_llm._torch.models.modeling_qwen3_5 import _normalize_qwen35_moe_vl_config
from tensorrt_llm._torch.pyexecutor.config_utils import (
    extract_mamba_kv_cache_params,
    load_pretrained_config,
)
from tensorrt_llm._torch.pyexecutor.model_loader import validate_and_set_mamba_ssm_cache_dtype
from tensorrt_llm.inputs import ContentFormat
from tensorrt_llm.inputs.registry import MULTIMODAL_PLACEHOLDER_REGISTRY


def _write_qwen35_moe_vl_config(tmp_path: Path) -> Path:
    config = {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "image_token_id": 248056,
        "model_type": "qwen3_5_moe",
        "text_config": {
            "attention_bias": False,
            "attention_dropout": 0.0,
            "bos_token_id": 151643,
            "dtype": "bfloat16",
            "eos_token_id": 151645,
            "full_attention_interval": 4,
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 2048,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
            "mamba_ssm_dtype": "float32",
            "max_position_embeddings": 262144,
            "mlp_only_layers": [],
            "model_type": "qwen3_5_moe_text",
            "moe_intermediate_size": 512,
            "norm_topk_prob": True,
            "num_attention_heads": 32,
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "num_hidden_layers": 2,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-6,
            "shared_expert_intermediate_size": 512,
            "rope_parameters": {
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.25,
                "rope_theta": 1000000.0,
                "rope_type": "default",
            },
            "use_cache": True,
            "vocab_size": 151936,
        },
        "tie_word_embeddings": False,
        "video_token_id": 248057,
        "vision_config": {
            "deepstack_visual_indexes": [8, 16, 24],
            "depth": 27,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 1152,
            "in_channels": 3,
            "intermediate_size": 4304,
            "model_type": "qwen3_5_moe",
            "num_heads": 16,
            "num_position_embeddings": 2304,
            "out_hidden_size": 2048,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
        "vision_end_token_id": 248054,
        "vision_start_token_id": 248053,
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return tmp_path


def test_qwen35_moe_vl_config_preserves_vlm_architecture(
    tmp_path: Path,
) -> None:
    config = load_pretrained_config(str(_write_qwen35_moe_vl_config(tmp_path)))

    assert isinstance(config, transformers.Qwen3_5MoeConfig)
    assert config.architectures == ["Qwen3_5MoeForConditionalGeneration"]
    assert config.text_config.architectures == ["Qwen3_5MoeForCausalLM"]
    assert config.text_config.num_experts == 128
    assert config.text_config.intermediate_size == 4608
    assert config.text_config.rope_theta == 1000000.0
    assert config.text_config.partial_rotary_factor == 0.25
    assert config.text_config.rope_scaling["type"] == "mrope"
    assert config.text_config.rope_scaling["mrope_section"] == [11, 11, 10]
    assert config.text_config.mamba_ssm_dtype == "float32"
    assert config.get_text_config() is config.text_config


def test_qwen35_moe_vl_resolves_mamba_ssm_cache_dtype(
    tmp_path: Path,
) -> None:
    config = load_pretrained_config(str(_write_qwen35_moe_vl_config(tmp_path)))
    model_config = ModelConfig(pretrained_config=config)

    validate_and_set_mamba_ssm_cache_dtype(model_config, "auto")
    assert model_config.quant_config.mamba_ssm_cache_dtype is torch.float32

    mamba_params = extract_mamba_kv_cache_params(
        config.text_config,
        quant_config=model_config.quant_config,
    )
    assert mamba_params.dtype is torch.bfloat16
    assert mamba_params.mamba_ssm_cache_dtype is torch.float32


def test_qwen35_moe_vl_resolves_model_and_mapper(tmp_path: Path) -> None:
    config = load_pretrained_config(str(_write_qwen35_moe_vl_config(tmp_path)))
    model_config = ModelConfig(pretrained_config=config)

    assert AutoModelForCausalLM._resolve_class(model_config) is Qwen3_5MoeVLModel
    assert isinstance(
        AutoCheckpointMapper.get("HF", "Qwen3_5MoeForConditionalGeneration"),
        Qwen3_5MoeHfWeightMapper,
    )


def test_qwen35_moe_vl_placeholder_metadata_registered() -> None:
    metadata = MULTIMODAL_PLACEHOLDER_REGISTRY.get_placeholder_metadata("qwen3_5_moe")

    assert metadata.placeholder_map == {
        "image": "<|vision_start|><|image_pad|><|vision_end|>",
        "video": "<|vision_start|><|video_pad|><|vision_end|>",
    }
    assert metadata.placeholders_separator == ""
    assert metadata.content_format is ContentFormat.STRING


# --- Layered parity test scaffold -------------------------------------------
#
# Tiny synthetic config used by TestQwen3_5MoeVL below. Same architecture as
# the real Qwen/Qwen3.5-35B-A3B checkpoint but with much smaller dimensions
# where possible.
#
# Shapes that have to match real Qwen3.5 (can't shrink without breaking
# things downstream):
#
#   - head_dim=256, partial_rotary_factor=0.25 --> rotary tensor width is
#     `head_dim * 0.25 / 2 = 32`, which equals `sum(mrope_section)`.
#     A smaller head_dim (e.g. 128) yields a 16-wide tensor that mRoPE
#     can't split with section sum 32.
#   - num_attention_heads=16, num_key_value_heads=2 match the real
#     model's 8:1 GQA layout; Q proj is 2048 --> 4096, K/V are 2048 --> 512.
#   - Vision deepstack indices [8, 16, 24] match the real config, and
#     depth=27 is the smallest value that hosts those indices. Disabling
#     deepstack (indices=[], depth=2) produces fewer vision embeddings
#     than the HF processor reserves placeholder tokens for, which
#     breaks `fuse_input_embeds`.
#   - vocab_size=248320 matches the real Qwen3.5 tokenizer. The
#     tokenizer (loaded via _name_or_path) emits special-token ids in
#     the 248k range; `fuse_input_embeds` uses `vocab_size` as the
#     OOV threshold to identify image-pad tokens. A smaller vocab_size
#     would misclassify regular chat-template specials as mm tokens
#     and trip the placeholder/embedding count check.
#
# Shapes that can be shrunk for tests:
#
#   - num_hidden_layers: 2 (vs 40+).
#   - num_experts: 128 (vs 256). Still moderate so MoE routing runs.
#   - full_attention_interval=2 with 2 LM layers yields the pattern
#     [linear_attention, full_attention] — one of each kind, exercising
#     both the regular KV cache and the Mamba SSM/conv state via the
#     base-class dispatch.
#
# `_name_or_path` points at the real checkpoint dir so the test can load
# the tokenizer/processor (only the processor; not the full model weights).
QWEN3_5_VL_MOE_PARITY_CONFIG = {
    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    "image_token_id": 248056,
    "model_type": "qwen3_5_moe",
    "text_config": {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "dtype": "bfloat16",
        "eos_token_id": 151645,
        "full_attention_interval": 2,
        "head_dim": 256,
        "hidden_act": "silu",
        "hidden_size": 2048,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_value_head_dim": 128,
        "mamba_ssm_dtype": "float32",
        "max_position_embeddings": 8192,
        "mlp_only_layers": [],
        "model_type": "qwen3_5_moe_text",
        "moe_intermediate_size": 512,
        "norm_topk_prob": True,
        "num_attention_heads": 16,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "shared_expert_intermediate_size": 512,
        "rope_parameters": {
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 1000000.0,
            "rope_type": "default",
        },
        "use_cache": True,
        "vocab_size": 248320,
    },
    "tie_word_embeddings": False,
    "video_token_id": 248057,
    "vision_config": {
        "deepstack_visual_indexes": [8, 16, 24],
        "depth": 27,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1152,
        "in_channels": 3,
        "initializer_range": 0.02,
        "intermediate_size": 4304,
        "model_type": "qwen3_5_moe",
        "num_heads": 16,
        "num_position_embeddings": 2304,
        "out_hidden_size": 2048,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
    "vision_end_token_id": 248054,
    "vision_start_token_id": 248053,
    "_name_or_path": str(os.path.join(llm_models_root(), "Qwen3.5-35B-A3B")),
}


@skip_pre_hopper
class TestQwen3_5MoeVL(TestModelingMultimodal):
    """Forward-parity test for Qwen3.5-MoE-VL against HuggingFace.

    Tiny-synthetic-config parity test in the same shape as
    `TestQwen3VLMoe` / `TestQwen2_5VL`: both stacks are constructed
    from `QWEN3_5_VL_MOE_PARITY_CONFIG` (2 LM layers, 1 linear + 1 full
    attention, 128 experts, 2 vision layers), HF weights are copied
    into TRT-LLM via `Qwen3_5MoeHfWeightMapper`, then `test_all`
    sweeps the default `MultimodalScenario`s comparing last-position
    logits at context + generation phases.

    Two-config design:
      - `self.hf_config` is the raw `Qwen3_5MoeConfig.from_dict(...)`
        result. HF model construction sees the native HF schema
        (`rope_parameters` intact with `rope_type`,
        `moe_intermediate_size`, …).
      - TRT-LLM gets a deep-copied + normalized version via the
        `create_trtllm_model` override below. That copy goes through
        `_normalize_qwen35_moe_vl_config` exactly the same way
        production `load_pretrained_config` does, so the Qwen3Next
        runtime sees the flat aliases it expects
        (`intermediate_size`, `rope_theta`, `rope_scaling`, …).

    Keeping the two configs separate means the production normalizer
    doesn't need to be HF-safe — production only ever constructs the
    TRT-LLM model from a normalized config, and the test mirrors that
    boundary explicitly. The hybrid-cache path is handled by the base
    class's `init_kv_cache_manager` dispatch on
    `is_qwen3_hybrid` / `is_nemotron_hybrid`.
    """

    def get_model_config(self):
        return QWEN3_5_VL_MOE_PARITY_CONFIG

    def get_trtllm_model_class(self):
        return Qwen3_5MoeVLModel

    def get_hf_model_class(self):
        return HFQwen3_5MoeForConditionalGeneration

    def get_weight_mapper_class(self):
        return Qwen3_5MoeHfWeightMapper

    def get_model_type(self):
        return "qwen3_5_moe"

    def get_model_config_class(self):
        return transformers.Qwen3_5MoeConfig

    def create_trtllm_model(
        self,
        load_weights: bool = False,
        hf_model_state_dict: Optional[dict] = None,
        **kwargs,
    ):
        """Build the TRT-LLM model from a *normalized copy* of `self.hf_config`.

        Mirrors the base-class body but swaps in
        `_normalize_qwen35_moe_vl_config(trtllm_config)` before
        wrapping in `ModelConfig`. `self.hf_config` itself stays
        raw so the HF model that the base class builds in `setUp`
        sees native HF schema.
        """
        trtllm_config = deepcopy(self.hf_config)
        _normalize_qwen35_moe_vl_config(trtllm_config)

        model_config = ModelConfig(pretrained_config=trtllm_config)
        model_class = self.get_trtllm_model_class()
        model = model_class(model_config, **kwargs).to("cuda")

        if load_weights:
            weight_mapper = self.get_weight_mapper_class()()
            weight_mapper.init_model_and_config(model, trtllm_config)
            model.load_weights(hf_model_state_dict, weight_mapper)

            for module in model.modules():
                if hasattr(module, "post_load_weights") and not getattr(
                    module, "_weights_removed", False
                ):
                    module.post_load_weights()

        return model, model_config

    def _dummy_request_kwargs(self, scenario):
        """Qwen3.5-VL uses mRoPE; the cache manager needs the mRoPE
        position-id buffer allocated at dummy-request time."""
        return {"use_mrope": True}

    def get_tolerance(self):
        """Tighten `rtol` to `0.1` (4x tighter than the base 0.4
        default) while keeping `atol` at `0.4` to absorb single-logit
        tail outliers seen on `multiple_image` / `video`.
        """
        return 0.4, 0.1

    def get_trtllm_inputs(
        self,
        input_ids,
        multimodal_params_list,
        is_gen: bool = False,
        num_cached_tokens_per_seq: Optional[List[int]] = None,
        total_prompt_len: Optional[int] = None,
    ):
        """Override position_ids with mRoPE position IDs from the
        multimodal params. Same pattern as `TestQwen3VLMoe` — the
        VLM wrapper feeds mRoPE-shaped position IDs to the decoder,
        not the simple range-based default the base class produces.
        """
        trtllm_inputs = super().get_trtllm_inputs(
            input_ids,
            multimodal_params_list,
            is_gen,
            num_cached_tokens_per_seq,
            total_prompt_len=total_prompt_len,
        )

        if is_gen:
            mrope_gen_position_ids = []
            for multimodal_param in multimodal_params_list:
                mrope_gen_position_ids.append(
                    multimodal_param.multimodal_data["mrope_config"]["mrope_position_deltas"]
                )
            mrope_gen_position_ids = torch.cat(mrope_gen_position_ids, dim=-1).to(self.device)
            trtllm_inputs["position_ids"] = (
                (trtllm_inputs["position_ids"] + mrope_gen_position_ids).expand(3, -1, 1).cuda()
            )
            gen_multimodal_params_list = []
            for multimodal_param in multimodal_params_list:
                multimodal_param.strip_for_generation()
                multimodal_param.to_device(
                    "multimodal_data",
                    self.device,
                    pin_memory=True,
                    target_keywords=["mrope_config.mrope_position_deltas"],
                )
                gen_multimodal_params_list.append(multimodal_param)
            trtllm_inputs["multimodal_params"] = gen_multimodal_params_list
        else:
            mrope_position_ids = []
            for multimodal_param in multimodal_params_list:
                mrope_position_ids.append(
                    multimodal_param.multimodal_data["mrope_config"]["mrope_position_ids"]
                )
            position_ids = torch.cat(mrope_position_ids, dim=-1).cuda()
            trtllm_inputs["position_ids"] = position_ids

        return trtllm_inputs

    def get_scenarios(self) -> List[MultimodalScenario]:
        """Modality-sanity sweep (image / multiple_image / video).

        These three catch differences in placeholder counts and the
        multimodal-cumsum path between single-image, multi-image, and
        video inputs.

        CUDA-graph capture is intentionally not exercised here. The
        standard `attn_metadata.create_cuda_graph_metadata` path only
        addresses attention metadata; the Mamba SSM state buffer of the
        hybrid (Mamba + attention) cache is not threaded through, so
        replayed logits diverge from the HF reference. Adding that path
        is dedicated harness work and tracked separately.
        """
        return [
            MultimodalScenario(
                modality="image",
                use_cuda_graph=False,
                chunked_prefill=False,
                kv_cache_reuse=False,
            ),
            MultimodalScenario(
                modality="multiple_image",
                use_cuda_graph=False,
                chunked_prefill=False,
                kv_cache_reuse=False,
            ),
            MultimodalScenario(
                modality="video",
                use_cuda_graph=False,
                chunked_prefill=False,
                kv_cache_reuse=False,
            ),
        ]

    def test_construction_and_weight_loading_smoke(self):
        """Smoke test: setUp built HF + TRT-LLM models and copied HF
        weights into TRT-LLM via the weight mapper. Detailed
        assertions on the normalizer's outputs live in the routing
        tests above (e.g. `test_qwen35_moe_vl_config_preserves_vlm_architecture`)
        — this one just confirms construction reached the end without
        exception.
        """
        self.assertIsNotNone(self.hf_model)
        self.assertIsNotNone(self.trtllm_model)
        self.assertIsNotNone(self.model_config)
