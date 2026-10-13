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
"""Embedded DeepSeek-V4.1 DSpark drafter.

The worker owns separate per-stage sliding-window context KV. It seeds those
windows from target layer-entry captures and back-fills all accepted positions;
the draft block attends non-causally to its own positions. V4.1 differs from V4
in the lagged mHC input mix, absence of an HC head and query-head RMSNorm, smaller
draft expert bank, and the checkpoint's 32-by-32 FP8 dense scale granularity.
"""

import copy
import re
from typing import TYPE_CHECKING

import torch

from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantAlgo

from ...models.modeling_utils import QuantConfig
from ..model_config import ModelConfig
from ..utils import AuxStreamType
from .modeling_deepseekv41 import (
    DeepseekV41DecoderLayer,
    DeepseekV41WeightLoader,
    _remap_deepseek_v41_checkpoint_keys,
    _v41_load_coverage,
    _v41_tensor_dtype,
)
from .modeling_dspark import DSv4DSparkBlock, DSv4DSparkDraftModel, DSv4DSparkForCausalLM

if TYPE_CHECKING:
    from ...llmapi.llm_args import DeepSeekV4SparseAttentionConfig

_STAGE_KEY = re.compile(r"^mtp\.(\d+)\.(.+)$")


def _validate_quantized_pairs(weights: dict[str, torch.Tensor]) -> None:
    """Reject quantized draft weights without scales, and orphan scales.

    Inspect dtype metadata only so lazy checkpoint tensors are not materialized.
    In particular, an FP8 main_proj without its scale must not reach a BF16
    Linear as an unscaled cast: parameter coverage alone cannot detect that.
    """
    quantized_dtypes = {
        "torch.float8_e4m3fn",
        "float8_e4m3fn",
        "F8_E4M3",
        "F8_E4M3FN",
        "torch.int8",
        "int8",
        "I8",
        "torch.uint8",
        "uint8",
        "U8",
    }
    missing = []
    for name, value in weights.items():
        if name.endswith(".scale"):
            sibling = name.removesuffix(".scale") + ".weight"
        elif name.endswith(".weight") and _v41_tensor_dtype(value) in quantized_dtypes:
            sibling = name.removesuffix(".weight") + ".scale"
        else:
            continue
        if sibling not in weights:
            missing.append(f"{name} has no {sibling}")
    if missing:
        raise ValueError(
            f"V4.1 DSpark checkpoint has incomplete quantized weight/scale pairs: {missing[:8]}"
        )


def remap_dspark_v41_draft_keys(
    weights: dict[str, torch.Tensor],
    num_stages: int,
    kv_lora_rank: int,
    quantization_config: dict | None = None,
) -> dict[str, torch.Tensor]:
    """Load every draft stage using the target's checked V4.1 quantization layout.

    Exposing each ``mtp.N`` stage as a regular layer to the shared pre-pass keeps
    dense FP8 conversion, packed expert scales, and mHC names identical to the
    target loader. The resulting keys are draft-local; no target tensor is loaded
    or changed. Stage-0 ``main_proj`` stays BF16 because it is outside the target
    dense-GEMM requantization allow-list.
    """
    stage_weights = {}
    seen_stages = set()
    for name, weight in weights.items():
        match = _STAGE_KEY.match(name)
        if match is None:
            continue
        stage = int(match.group(1))
        if stage >= num_stages:
            continue
        seen_stages.add(stage)
        rest = match.group(2)
        rest = rest.replace("markov_head.embed.", "markov_head.markov_w1.")
        rest = rest.replace("markov_head.head.", "markov_head.markov_w2.")
        stage_weights[f"layers.{stage}.{rest}"] = weight
    missing = sorted(set(range(num_stages)) - seen_stages)
    if missing:
        raise ValueError(f"V4.1 DSpark checkpoint is missing draft stages {missing}")
    _validate_quantized_pairs(stage_weights)
    remapped = _remap_deepseek_v41_checkpoint_keys(
        stage_weights,
        num_hidden_layers=num_stages,
        kv_lora_rank=kv_lora_rank,
        quantization_config=quantization_config,
    )
    return {
        name.replace("model.layers.", "mtp_layers.", 1): weight for name, weight in remapped.items()
    }


class DSv41DSparkBlock(DSv4DSparkBlock, DeepseekV41DecoderLayer):
    """V4.1 decoder numerics with DSpark's stage-0 and final-stage heads."""

    uses_hc_head = False

    @staticmethod
    def _capture_quant_config(model_config: ModelConfig) -> QuantConfig:
        # main_proj has checkpoint block-32 scales and is intentionally
        # dequantized by the shared V4.1 pre-pass, even in block-128 FP8 mode.
        return QuantConfig()


class DSv41DSparkDraftModel(DSv4DSparkDraftModel):
    """Three lagged-pre draft stages, each with its own captured-context KV."""

    block_cls = DSv41DSparkBlock
    uses_lagged_pre = True
    checkpoint_fp8_block_size = 32
    # Reference DSparkBlock.forward_head passes collapsed x to confidence,
    # while only the shared vocabulary projection consumes norm(x).
    confidence_uses_normalized_hidden = False

    def __init__(
        self,
        model_config: ModelConfig,
        aux_stream_dict: dict[AuxStreamType, torch.cuda.Stream] | None,
        num_stages: int | None = None,
        block_size: int | None = None,
        draft_moe_backend: str | None = None,
    ) -> None:
        draft_config = copy.copy(model_config)
        config = copy.deepcopy(model_config.pretrained_config)
        config.n_routed_experts = config.dspark_n_routed_experts or config.n_routed_experts
        config.num_experts_per_tok = config.dspark_num_experts_per_tok or config.num_experts_per_tok
        if config.n_routed_experts < config.num_experts_per_tok:
            raise ValueError("V4.1 DSpark top-k exceeds its draft expert count")
        if not config.dspark_target_layer_ids:
            raise ValueError("V4.1 DSpark requires checkpoint dspark_target_layer_ids")
        # The draft is SWA-only, with no encoder/decoder KV or index sharing.
        # Its attention module is instantiated to load weights; runtime attention
        # uses worker-owned windows, independently of the target cache manager.
        config.engram_layer_ids = []
        config.engram_config = None
        config.window_size = config.sliding_window
        config.candidate_source_layer_id = None
        # CSA2 lowers checkpoint-owned geometry. Keep the draft layout on
        # this private HF config copy, independent of the target's sources.
        spec_config = getattr(model_config, "spec_config", None)
        stage_count = int(
            num_stages
            if num_stages is not None
            else getattr(spec_config, "num_draft_layers", None)
            or getattr(config, "n_mtp_layers", None)
            or config.num_nextn_predict_layers
        )
        config.compress_ratios = [0] * stage_count
        config.kv_source_layer_ids = []
        config.index_source_layer_ids = []
        # No candidate source consumes this positive capacity value.
        config.candidate_topk_blocks = max(1, config.candidate_topk_blocks)
        # The released HF config is composite. Model consumers read forwarded
        # outer attributes, while CSA2 lowers text_config directly. Keep both
        # views of this private copy on the same draft-only geometry.
        text_config = getattr(config, "text_config", None)
        if text_config is not None:
            for name in (
                "n_routed_experts",
                "num_experts_per_tok",
                "engram_layer_ids",
                "engram_config",
                "window_size",
                "candidate_source_layer_id",
                "compress_ratios",
                "kv_source_layer_ids",
                "index_source_layer_ids",
                "candidate_topk_blocks",
            ):
                setattr(text_config, name, copy.deepcopy(getattr(config, name)))
        # A target may have materialized these cached descriptors before its
        # embedded drafter is copied; they must be derived from the new layout.
        config.__dict__.pop("layer_descriptors", None)
        if text_config is not None:
            text_config.__dict__.pop("layer_descriptors", None)
        object.__setattr__(draft_config, "pretrained_config", config)
        super().__init__(
            draft_config,
            aux_stream_dict,
            num_stages,
            block_size,
            draft_moe_backend=draft_moe_backend,
        )
        self._attn_params["q_b_norm_enabled"] = False

    @staticmethod
    def _draft_quant_config_dict(
        model_config: ModelConfig, base: int, num_stages: int
    ) -> dict[str, QuantConfig] | None:
        quant_configs = DSv4DSparkDraftModel._draft_quant_config_dict(
            model_config, base, num_stages
        )
        checkpoint_dir = getattr(model_config.pretrained_config, "_name_or_path", None)
        if not checkpoint_dir:
            return quant_configs

        # Embedded MTP experts can use a different format from the main experts.
        draft_quant_configs = dict(quant_configs or model_config.quant_config_dict or {})
        for stage in range(num_stages):
            tensor_info = ModelConfig._get_safetensors_header_for_tensor(
                checkpoint_dir, f"mtp.{stage}.ffn.experts.0.w1.weight"
            )
            if tensor_info is None:
                continue
            dtype = tensor_info.get("dtype")
            if dtype == "I8" and len(tensor_info.get("shape", [])) == 2:
                quant_algo = ModelConfig.get_mxfp4_quant_algo(model_config.moe_backend)
                group_size = 32
            elif dtype == "U8":
                quant_algo = QuantAlgo.NVFP4
                group_size = 16
            else:
                continue
            key = f"model.layers.{base + stage}.mlp.experts"
            source = draft_quant_configs.get(key)
            values = source.model_dump() if source is not None else {}
            values.update(quant_algo=quant_algo, group_size=group_size)
            draft_quant_configs[key] = QuantConfig(**values)
        return draft_quant_configs or None

    @staticmethod
    def _draft_sparse_config(
        model_config: ModelConfig, base: int, num_stages: int
    ) -> "DeepSeekV4SparseAttentionConfig | None":
        sparse_config = model_config.sparse_attention_config
        if sparse_config is None:
            return None
        if sparse_config.algorithm == "csa2":
            return None
        updates = {"compress_ratios": [0] * num_stages}
        for name in ("kv_source_layer_ids", "index_source_layer_ids"):
            if name in type(sparse_config).model_fields:
                updates[name] = []
        return sparse_config.model_copy(update=updates)


class DSv41DSparkForCausalLM(DSv4DSparkForCausalLM):
    """V4.1 wrapper using the existing embedded DSpark worker and verifier."""

    draft_model_cls = DSv41DSparkDraftModel

    def load_weights(self, weights: dict[str, torch.Tensor], weight_mapper=None, **kwargs) -> None:
        remapped = remap_dspark_v41_draft_keys(
            weights,
            self.num_stages,
            self.config.kv_lora_rank,
            getattr(self.config, "quantization_config", None),
        )
        last = self.dspark_model.mtp_layers[-1]
        required = [
            "mtp_layers.0.main_proj.weight",
            "mtp_layers.0.main_norm.weight",
            f"mtp_layers.{self.num_stages - 1}.norm.weight",
        ]
        if last.markov_head is not None:
            required.extend(
                f"mtp_layers.{self.num_stages - 1}.markov_head.markov_w{i}.weight" for i in (1, 2)
            )
        missing = [name for name in required if name not in remapped]
        if missing:
            raise ValueError(f"V4.1 DSpark checkpoint is missing draft head weights: {missing}")
        confidence_prefix = f"mtp_layers.{self.num_stages - 1}.confidence_head."
        if not any(name.startswith(confidence_prefix) for name in remapped):
            last.confidence_head = None
        DeepseekV41WeightLoader(self.dspark_model).load_weights(remapped)
        unexpected, unfed = _v41_load_coverage(self.dspark_model, remapped)
        if unexpected or unfed:
            raise ValueError(
                "DeepSeek-V4.1 DSpark weight load is incomplete: "
                f"unconsumed keys {unexpected[:8]}; unfed parameters {unfed[:8]}"
            )
        self.dspark_model.post_load_weights()
        self.dspark_model.cache_attn_weights_from_state_dict(weights)
        logger.info(f"[DSpark V4.1] loaded {self.num_stages} lagged-pre draft stages")


__all__ = [
    "DSv41DSparkBlock",
    "DSv41DSparkDraftModel",
    "DSv41DSparkForCausalLM",
    "remap_dspark_v41_draft_keys",
]
