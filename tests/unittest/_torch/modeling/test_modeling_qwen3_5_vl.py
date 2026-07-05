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
from transformers import Qwen3_5ForConditionalGeneration as HFQwen3_5ForConditionalGeneration
from utils.llm_data import llm_models_root
from utils.util import skip_pre_hopper

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models import Qwen3_5VLModel
from tensorrt_llm._torch.models.checkpoints.auto_mapper import AutoCheckpointMapper
from tensorrt_llm._torch.models.checkpoints.hf.qwen3_5_weight_mapper import Qwen3_5MoeHfWeightMapper
from tensorrt_llm._torch.models.modeling_auto import AutoModelForCausalLM
from tensorrt_llm._torch.models.modeling_qwen3_5 import _normalize_qwen35_vl_config
from tensorrt_llm._torch.pyexecutor.config_utils import (
    extract_mamba_kv_cache_params,
    load_pretrained_config,
)
from tensorrt_llm._torch.pyexecutor.model_loader import validate_and_set_mamba_ssm_cache_dtype
from tensorrt_llm.inputs import ContentFormat
from tensorrt_llm.inputs.registry import MULTIMODAL_PLACEHOLDER_REGISTRY

# Dense sibling of test_modeling_qwen3_5_vl_moe.py. The dense Qwen3.5-VL
# (Qwen/Qwen3.5-27B, arch Qwen3_5ForConditionalGeneration, model_type qwen3_5)
# reuses the same hybrid Qwen3Next runtime as the MoE variant, differing only in
# the feed-forward block (GatedMLP instead of SparseMoeBlock). This file mirrors
# the MoE routing + parity tests with a dense synthetic config.
#
# Two dense-specific differences from the MoE config:
#   - No MoE fields (num_experts / moe_intermediate_size / ...); a native
#     `intermediate_size` is present and must be preserved by the normalizer
#     (the Qwen3Next-alias synthesis is a no-op for dense).
#   - `deepstack_visual_indexes: []` matches the real dense checkpoint (as it
#     does for the MoE one — the Qwen3.5 family dropped Qwen3-VL's deepstack).
#     `use_deepstack` stays truthy via `hasattr`, but
#     `deepstack_num_level == 0` makes the split a no-op.
#   - `attn_output_gate: true` mirrors the real config and matches TRT-LLM's
#     Qwen3NextAttention, which hardcodes output gating.


def _write_qwen35_dense_vl_config(tmp_path: Path) -> Path:
    config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "image_token_id": 248056,
        "model_type": "qwen3_5",
        "text_config": {
            "attention_bias": False,
            "attention_dropout": 0.0,
            "attn_output_gate": True,
            "dtype": "bfloat16",
            "eos_token_id": 248044,
            "full_attention_interval": 4,
            "head_dim": 256,
            "hidden_act": "silu",
            "hidden_size": 2048,
            "intermediate_size": 2048,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
            "mamba_ssm_dtype": "float32",
            "max_position_embeddings": 262144,
            "mlp_only_layers": [],
            "model_type": "qwen3_5_text",
            "num_attention_heads": 16,
            "num_hidden_layers": 2,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000.0,
                "rope_type": "default",
            },
            "use_cache": True,
            "vocab_size": 248320,
        },
        "tie_word_embeddings": False,
        "video_token_id": 248057,
        "vision_config": {
            "deepstack_visual_indexes": [],
            "depth": 2,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 1152,
            "in_channels": 3,
            "intermediate_size": 4304,
            "model_type": "qwen3_5",
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


def test_qwen35_dense_vl_config_preserves_vlm_architecture(
    tmp_path: Path,
) -> None:
    config = load_pretrained_config(str(_write_qwen35_dense_vl_config(tmp_path)))

    assert isinstance(config, transformers.Qwen3_5Config)
    assert config.architectures == ["Qwen3_5ForConditionalGeneration"]
    assert config.text_config.architectures == ["Qwen3_5ForCausalLM"]
    # Dense: native intermediate_size is preserved (no MoE synthesis).
    assert config.text_config.intermediate_size == 2048
    assert getattr(config.text_config, "num_experts", 0) in (0, None)
    # Qwen3.5 family contract: deepstack stays disabled (empty) — both real
    # checkpoints (dense 27B and MoE 35B-A3B) publish []. Normalization must
    # not synthesize indices; the placeholder path depends on this staying
    # empty (see re-greening item D in the MoE testing-notes doc).
    assert config.vision_config.deepstack_visual_indexes == []
    assert config.text_config.rope_theta == 10000000.0
    assert config.text_config.partial_rotary_factor == 0.25
    assert config.text_config.rope_scaling["type"] == "mrope"
    assert config.text_config.rope_scaling["mrope_section"] == [11, 11, 10]
    # mrope_interleaved must survive normalization: the fused QK-norm-RoPE op
    # gates the mRoPE path on it, and without it position_ids gets flattened
    # to 3*num_tokens and mismatches the QKV token count.
    assert config.text_config.rope_scaling["mrope_interleaved"] is True
    assert config.text_config.mamba_ssm_dtype == "float32"
    assert config.get_text_config() is config.text_config


def test_qwen35_dense_vl_resolves_mamba_ssm_cache_dtype(
    tmp_path: Path,
) -> None:
    config = load_pretrained_config(str(_write_qwen35_dense_vl_config(tmp_path)))
    model_config = ModelConfig(pretrained_config=config)

    validate_and_set_mamba_ssm_cache_dtype(model_config, "auto")
    assert model_config.quant_config.mamba_ssm_cache_dtype is torch.float32

    mamba_params = extract_mamba_kv_cache_params(
        config.text_config,
        quant_config=model_config.quant_config,
    )
    assert mamba_params.dtype is torch.bfloat16
    assert mamba_params.mamba_ssm_cache_dtype is torch.float32


def test_qwen35_dense_vl_resolves_model_and_mapper(tmp_path: Path) -> None:
    config = load_pretrained_config(str(_write_qwen35_dense_vl_config(tmp_path)))
    model_config = ModelConfig(pretrained_config=config)

    assert AutoModelForCausalLM._resolve_class(model_config) is Qwen3_5VLModel
    assert isinstance(
        AutoCheckpointMapper.get("HF", "Qwen3_5ForConditionalGeneration"),
        Qwen3_5MoeHfWeightMapper,
    )


def test_qwen35_dense_vl_placeholder_metadata_registered() -> None:
    metadata = MULTIMODAL_PLACEHOLDER_REGISTRY.get_placeholder_metadata("qwen3_5")

    assert metadata.placeholder_map == {
        "image": "<|vision_start|><|image_pad|><|vision_end|>",
        "video": "<|vision_start|><|video_pad|><|vision_end|>",
    }
    assert metadata.placeholders_separator == ""
    assert metadata.content_format is ContentFormat.STRING


# --- Layered parity test scaffold -------------------------------------------
#
# Tiny synthetic config used by TestQwen3_5VL below. Same architecture as the
# real Qwen/Qwen3.5-27B checkpoint but with much smaller dimensions. The shape
# constraints are identical to the MoE parity config (see
# test_modeling_qwen3_5_vl_moe.py) except:
#
#   - dense MLP: native `intermediate_size` (no MoE fields), so Qwen3NextModel
#     selects GatedMLP for the feed-forward layers.
#   - `deepstack_visual_indexes=[]` matches the real dense checkpoint (same
#     as the MoE parity config — the Qwen3.5 family dropped deepstack);
#     `depth=2` keeps the tower tiny since nothing pins its depth, and the
#     config exercises the `deepstack_num_level == 0` no-op path.
#
# `_name_or_path` points at the real checkpoint dir so the test can load the
# tokenizer/processor (only the processor; not the full model weights).
QWEN3_5_VL_DENSE_PARITY_CONFIG = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "image_token_id": 248056,
    "model_type": "qwen3_5",
    "text_config": {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attn_output_gate": True,
        "dtype": "bfloat16",
        "eos_token_id": 248044,
        "full_attention_interval": 2,
        "head_dim": 256,
        "hidden_act": "silu",
        "hidden_size": 2048,
        "intermediate_size": 2048,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_value_head_dim": 128,
        "mamba_ssm_dtype": "float32",
        "max_position_embeddings": 8192,
        "mlp_only_layers": [],
        "model_type": "qwen3_5_text",
        "num_attention_heads": 16,
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000.0,
            "rope_type": "default",
        },
        "use_cache": True,
        "vocab_size": 248320,
    },
    "tie_word_embeddings": False,
    "video_token_id": 248057,
    "vision_config": {
        "deepstack_visual_indexes": [],
        "depth": 2,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1152,
        "in_channels": 3,
        "initializer_range": 0.02,
        "intermediate_size": 4304,
        "model_type": "qwen3_5",
        "num_heads": 16,
        "num_position_embeddings": 2304,
        "out_hidden_size": 2048,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
    "vision_end_token_id": 248054,
    "vision_start_token_id": 248053,
    "_name_or_path": str(os.path.join(llm_models_root(), "Qwen3.5-27B")),
}


@skip_pre_hopper
class TestQwen3_5VL(TestModelingMultimodal):
    """Forward-parity test for dense Qwen3.5-VL against HuggingFace.

    Dense sibling of `TestQwen3_5MoeVL`: both stacks are constructed from
    `QWEN3_5_VL_DENSE_PARITY_CONFIG` (2 LM layers, 1 linear + 1 full attention,
    dense GatedMLP, 2 vision layers, deepstack disabled), HF weights are copied
    into TRT-LLM via `Qwen3_5MoeHfWeightMapper`, then `test_all` sweeps the
    default `MultimodalScenario`s comparing last-position logits.

    Two-config design (same as the MoE test): `self.hf_config` stays raw HF
    schema; TRT-LLM gets a deep-copied + normalized copy via the
    `get_trtllm_pretrained_config` override, mirroring production
    `load_pretrained_config`
    (`_normalize_qwen35_vl_config(..., inner_arch="Qwen3_5ForCausalLM")`).
    """

    def get_model_config(self):
        return QWEN3_5_VL_DENSE_PARITY_CONFIG

    def get_trtllm_model_class(self):
        return Qwen3_5VLModel

    def get_hf_model_class(self):
        return HFQwen3_5ForConditionalGeneration

    def get_weight_mapper_class(self):
        return Qwen3_5MoeHfWeightMapper

    def get_model_type(self):
        return "qwen3_5"

    def get_model_config_class(self):
        return transformers.Qwen3_5Config

    def get_trtllm_pretrained_config(self) -> transformers.PretrainedConfig:
        """Return a normalized config copy for TRT-LLM model construction.

        Mirrors the MoE test but passes `inner_arch="Qwen3_5ForCausalLM"` to the
        shared normalizer so the dense text decoder is selected.
        """
        trtllm_config = deepcopy(self.hf_config)
        _normalize_qwen35_vl_config(trtllm_config, inner_arch="Qwen3_5ForCausalLM")
        return trtllm_config

    def _dummy_request_kwargs(self, scenario):
        """Qwen3.5-VL uses mRoPE; the cache manager needs the mRoPE
        position-id buffer allocated at dummy-request time."""
        return {"use_mrope": True}

    def get_tolerance(self):
        """Tighten `rtol` to `0.1` (4x tighter than the base 0.4 default)
        while keeping `atol` at `0.4` to absorb single-logit tail outliers.
        Same band as the MoE parity test.
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
        """Override position_ids with mRoPE position IDs from the multimodal
        params. Identical to the MoE test — the VLM wrapper feeds mRoPE-shaped
        position IDs to the decoder, not the base class's range-based default.
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
                (trtllm_inputs["position_ids"] + mrope_gen_position_ids)
                .expand(3, -1, 1)
                .to(self.device)
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
            # Cached-mRoPE read slots (added in #11943): the decode path reads
            # per-request deltas from the cache by seq slot.
            trtllm_inputs["mrope_delta_read_seq_slots"] = torch.arange(
                len(multimodal_params_list), device=self.device, dtype=torch.long
            )
        else:
            # Mrope position ids. For chunked prefill / KV cache reuse we must
            # mirror production `PyTorchModelEngine` and slice each request's
            # full `mrope_position_ids` to the current chunk's token range —
            # the fused QK-norm-RoPE op requires position_ids tokens to match
            # the QKV token count.
            chunk_len = input_ids.shape[-1]
            if num_cached_tokens_per_seq is None:
                begin_offsets = [0] * len(multimodal_params_list)
            elif isinstance(num_cached_tokens_per_seq, int):
                begin_offsets = [num_cached_tokens_per_seq] * len(multimodal_params_list)
            else:
                begin_offsets = list(num_cached_tokens_per_seq)
            mrope_position_ids = []
            for multimodal_param, begin in zip(multimodal_params_list, begin_offsets):
                full_mrope = multimodal_param.multimodal_data["mrope_config"]["mrope_position_ids"]
                mrope_position_ids.append(full_mrope[:, :, begin : begin + chunk_len])
            position_ids = torch.cat(mrope_position_ids, dim=-1).to(self.device)
            trtllm_inputs["position_ids"] = position_ids
            # Cached-mRoPE write slots (added in #11943): the context path
            # writes per-request deltas into the cache by seq slot.
            trtllm_inputs["mrope_delta_write_seq_slots"] = torch.arange(
                len(multimodal_params_list), device=self.device, dtype=torch.long
            )

        return trtllm_inputs

    def get_scenarios(self) -> List[MultimodalScenario]:
        """Modality-sanity sweep (image / multiple_image / video).

        CUDA-graph capture is intentionally not exercised — the hybrid
        (Mamba + attention) cache's SSM state buffer isn't threaded through the
        harness graph-capture path. Same limitation as the MoE parity test.
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
        """Smoke test: setUp built HF + TRT-LLM models and copied HF weights
        into TRT-LLM via the weight mapper. Detailed assertions on the
        normalizer's outputs live in the routing tests above — this one just
        confirms construction reached the end without exception.
        """
        self.assertIsNotNone(self.hf_model)
        self.assertIsNotNone(self.trtllm_model)
        self.assertIsNotNone(self.model_config)
