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
import math
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


# MegaMoE serves NVFP4 experts only: a draft on MEGAMOE_CUTEDSL re-encodes the checkpoint's MXFP4
# routed experts (E2M1 codes, one E8M0 scale per 32 elements) as NVFP4 at load. The weights stay exact: an
# E8M0 scale is a power of two and E4M3 holds every power of two from 2**-9 to 2**8, so with one
# weight_scale_2 per operand at 2**(e_max - 135) each block keeps its value while the operand's
# non-zero blocks span at most 17 binades. The static NVFP4 activation scales, which the checkpoint
# has only for the target, are bounds from the model rather than calibration data: the MoE input is
# an RMSNorm output, |x| <= sqrt(hidden) * max|norm weight|; the FC2 input is the clamped SwiGLU
# times a routing weight, |a| <= limit * silu(limit) * routed_scaling_factor.
_E2M1_MAX = 6.0
_E4M3_MAX = 448.0
# E4M3 holds 2**d exactly for d in [-9, 8]; an operand's largest block scale becomes 2**8.
_TOP_EXP = 8
_MIN_EXP = -9
_DRAFT_EXPERT_KEY = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$")


def _draft_experts_as_nvfp4(draft_moe_backend: str | None) -> bool:
    """Whether the draft's MXFP4 routed experts load as NVFP4 (an explicit MegaMoE draft)."""
    return draft_moe_backend == "MEGAMOE_CUTEDSL"


def _materialize(value) -> torch.Tensor:
    return value if isinstance(value, torch.Tensor) else value[:]


def _exponents(scale: torch.Tensor) -> torch.Tensor:
    return scale.contiguous().view(torch.uint8).to(torch.int32)


def _nonzero_blocks(weight: torch.Tensor) -> torch.Tensor:
    """Blocks of 32 codes (16 bytes) holding at least one non-zero E2M1 magnitude."""
    packed = weight.contiguous().view(torch.uint8)
    rows, nbytes = packed.shape
    return ((packed & 0x77) != 0).reshape(rows, nbytes // 16, 16).any(dim=-1)


def _operand_exponent_range(weights: list, exponents: list) -> tuple:
    """(min, max) E8M0 exponent of an operand's blocks, over the non-zero ones if all do not fit."""
    lo = min(int(e.min()) for e in exponents)
    hi = max(int(e.max()) for e in exponents)
    if hi - lo <= _TOP_EXP - _MIN_EXP:
        return lo, hi
    live = [e[_nonzero_blocks(w)] for w, e in zip(weights, exponents)]
    live = [e for e in live if e.numel()]
    if not live:
        return 127, 127
    return min(int(e.min()) for e in live), max(int(e.max()) for e in live)


def _pow2_to_e4m3(exp: torch.Tensor) -> torch.Tensor:
    """E4M3 bytes of 2**exp for integer exp in [-9, 8] (subnormal below -6)."""
    normal = torch.bitwise_left_shift((exp + 7).clamp(min=0), 3)
    subnormal = torch.bitwise_left_shift(torch.ones_like(exp), (exp + 9).clamp(min=0, max=2))
    return torch.where(exp >= -6, normal, subnormal).to(torch.uint8)


def _mxfp4_to_nvfp4(weight: torch.Tensor, exponents: torch.Tensor, e_max: int) -> tuple:
    """[N, K/2] MXFP4 codes, [N, K/32] E8M0 exponents -> NVFP4 codes, [N, K/16] E4M3 scales."""
    exp = exponents - e_max + _TOP_EXP
    if bool((exp < _MIN_EXP).any()) and bool((exp[_nonzero_blocks(weight)] < _MIN_EXP).any()):
        raise ValueError(
            "MXFP4 block exponents span more binades than E4M3 holds; "
            "the NVFP4 re-encoding would not be exact"
        )
    # Blocks of zero codes keep their value under any scale.
    exp = exp.clamp(min=_MIN_EXP, max=_TOP_EXP)
    scale = _pow2_to_e4m3(exp).repeat_interleave(2, dim=-1).view(torch.float8_e4m3fn)
    return weight.contiguous().view(torch.uint8), scale


def _draft_activation_amax(
    norm_weight: torch.Tensor,
    hidden_size: int,
    swiglu_limit: float,
    routed_scaling_factor: float,
) -> tuple:
    """Bounds of the MoE input (FC1) and the routing-weighted SwiGLU output (FC2)."""
    fc1_amax = math.sqrt(hidden_size) * float(norm_weight.float().abs().max())
    silu_limit = swiglu_limit / (1.0 + math.exp(-swiglu_limit))
    fc2_amax = swiglu_limit * silu_limit * routed_scaling_factor
    return fc1_amax, fc2_amax


def convert_draft_experts_to_nvfp4(
    stage_weights: dict,
    hidden_size: int,
    swiglu_limit: float,
    routed_scaling_factor: float,
) -> dict:
    """Replace the MXFP4 ``layers.<s>.ffn.experts.<e>.w{1,2,3}.{weight,scale}`` entries of a draft
    weight map by NVFP4 ``weight`` / ``weight_scale`` / ``weight_scale_2`` / ``input_scale``."""
    if swiglu_limit is None:
        raise ValueError("NVFP4 DSpark draft experts need a SwiGLU limit to bound the FC2 input")
    groups: dict = {}
    out = {}
    for name, value in stage_weights.items():
        match = _DRAFT_EXPERT_KEY.match(name)
        if match is None:
            out[name] = value
            continue
        stage, expert, proj, kind = (
            int(match.group(1)),
            int(match.group(2)),
            match.group(3),
            match.group(4),
        )
        groups.setdefault((stage, expert), {}).setdefault(proj, {})[kind] = value
    amax = {}
    for stage in sorted({stage for stage, _ in groups}):
        amax[stage] = _draft_activation_amax(
            _materialize(stage_weights[f"layers.{stage}.ffn_norm.weight"]),
            hidden_size,
            swiglu_limit,
            routed_scaling_factor,
        )
    max_span = 0
    for (stage, expert), projs in sorted(groups.items()):
        if set(projs) != {"w1", "w2", "w3"} or any(
            set(p) != {"weight", "scale"} for p in projs.values()
        ):
            raise ValueError(
                f"draft stage {stage} expert {expert}: incomplete MXFP4 expert tensors"
            )
        weights = {p: _materialize(kv["weight"]) for p, kv in projs.items()}
        exponents = {p: _exponents(_materialize(kv["scale"])) for p, kv in projs.items()}
        # The gate and up projections are fused into FC1, so they share one weight_scale_2.
        for operand, operand_amax in ((("w1", "w3"), amax[stage][0]), (("w2",), amax[stage][1])):
            lo, hi = _operand_exponent_range(
                [weights[p] for p in operand], [exponents[p] for p in operand]
            )
            max_span = max(max_span, hi - lo)
            weight_scale_2 = torch.tensor(2.0 ** (hi - 127 - _TOP_EXP), dtype=torch.float32)
            input_scale = torch.tensor(operand_amax / (_E4M3_MAX * _E2M1_MAX), dtype=torch.float32)
            for p in operand:
                codes, scale = _mxfp4_to_nvfp4(weights[p], exponents[p], hi)
                prefix = f"layers.{stage}.ffn.experts.{expert}.{p}"
                out[f"{prefix}.weight"] = codes
                out[f"{prefix}.weight_scale"] = scale
                out[f"{prefix}.weight_scale_2"] = weight_scale_2.clone()
                out[f"{prefix}.input_scale"] = input_scale.clone()
    if groups:
        bounds = ", ".join(f"{s}: {a1:.2f} / {a2:.2f}" for s, (a1, a2) in sorted(amax.items()))
        logger.info(
            f"DSpark: {len(groups)} draft experts re-encoded MXFP4 -> NVFP4 (largest E8M0 span "
            f"{max_span} of {_TOP_EXP - _MIN_EXP}); static FC1 / FC2 activation amax by stage "
            f"{bounds}"
        )
    return out


def remap_dspark_v41_draft_keys(
    weights: dict[str, torch.Tensor],
    num_stages: int,
    kv_lora_rank: int,
    quantization_config: dict | None = None,
    pretrained_config=None,
    nvfp4_experts: bool = False,
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
    if nvfp4_experts:
        stage_weights = convert_draft_experts_to_nvfp4(
            stage_weights,
            pretrained_config.hidden_size,
            getattr(pretrained_config, "swiglu_limit", None),
            pretrained_config.routed_scaling_factor,
        )
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
    # V4.1 binds BF16 attention projections; the FP8 backend is rejected.
    supports_fp8_attn_projections = False
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
        self.nvfp4_experts = _draft_experts_as_nvfp4(draft_moe_backend)

    @staticmethod
    def _draft_quant_config_dict(
        model_config: ModelConfig, base: int, num_stages: int, draft_moe_backend: str | None = None
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
            mxfp4 = dtype == "I8" and len(tensor_info.get("shape", [])) == 2
            if mxfp4 and _draft_experts_as_nvfp4(draft_moe_backend):
                # remap_dspark_v41_draft_keys re-encodes these MXFP4 experts as NVFP4.
                quant_algo = QuantAlgo.NVFP4
                group_size = 16
            elif mxfp4:
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
            pretrained_config=self.config,
            nvfp4_experts=self.dspark_model.nvfp4_experts,
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
        self._prepare_confidence_head_for_load(remapped)
        DeepseekV41WeightLoader(self.dspark_model).load_weights(remapped)
        unexpected, unfed = _v41_load_coverage(self.dspark_model, remapped)
        if unexpected or unfed:
            raise ValueError(
                "DeepSeek-V4.1 DSpark weight load is incomplete: "
                f"unconsumed keys {unexpected[:8]}; unfed parameters {unfed[:8]}"
            )
        self.dspark_model.post_load_weights(weights)
        logger.info(f"[DSpark V4.1] loaded {self.num_stages} lagged-pre draft stages")


__all__ = [
    "DSv41DSparkBlock",
    "DSv41DSparkDraftModel",
    "DSv41DSparkForCausalLM",
    "remap_dspark_v41_draft_keys",
]
