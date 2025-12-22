# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Supported models and feature support matrices (stdlib-only)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Mapping, Optional, Sequence, Tuple


class SupportStatus(str, Enum):
    YES = "Yes"
    NO = "No"
    UNTESTED = "Untested"
    NA = "N/A"


class Feature(str, Enum):
    OVERLAP_SCHEDULER = "Overlap Scheduler"
    CUDA_GRAPH = "CUDA Graph"
    ATTENTION_DP = "Attention Data Parallelism"
    DISAGGREGATED_SERVING = "Disaggregated Serving"
    CHUNKED_PREFILL = "Chunked Prefill"
    MTP = "MTP"
    EAGLE3_ONE_MODEL_ENGINE = "EAGLE-3(One Model Engine)"
    EAGLE3_TWO_MODEL_ENGINE = "EAGLE-3(Two Model Engine)"
    TORCH_SAMPLER = "Torch Sampler"
    TLLM_CPP_SAMPLER = "TLLM C++ Sampler"
    KV_CACHE_REUSE = "KV Cache Reuse"
    SLIDING_WINDOW_ATTENTION = "Sliding Window Attention"
    LOGITS_POST_PROCESSOR = "Logits Post Processor"
    GUIDED_DECODING = "Guided Decoding"

    EPD_DISAGG_SERVING = "EPD Disaggregated Serving"
    MODALITY = "Modality"


@dataclass(frozen=True)
class SupportedModel:
    architecture: str
    model: str
    huggingface_example: str


@dataclass(frozen=True)
class FeatureCell:
    status: Optional[SupportStatus] = None
    footnote: Optional[str] = None
    text: Optional[str] = None

    def render(self) -> str:
        if self.text is not None:
            return self.text
        if self.status is None:
            return ""
        out = self.status.value
        if self.footnote:
            out = f"{out} {self.footnote}"
        return out


SUPPORTED_MODELS_PYTORCH: Tuple[SupportedModel, ...] = (
    SupportedModel(
        architecture="BertForSequenceClassification",
        model="BERT-based",
        huggingface_example="textattack/bert-base-uncased-yelp-polarity",
    ),
    SupportedModel(
        architecture="DeciLMForCausalLM",
        model="Nemotron",
        huggingface_example="nvidia/Llama-3_1-Nemotron-51B-Instruct",
    ),
    SupportedModel(
        architecture="DeepseekV3ForCausalLM",
        model="DeepSeek-V3",
        huggingface_example="deepseek-ai/DeepSeek-V3",
    ),
    SupportedModel(
        architecture="DeepseekV32ForCausalLM",
        model="DeepSeek-V3.2",
        huggingface_example="deepseek-ai/DeepSeek-V3.2",
    ),
    SupportedModel(
        architecture="Exaone4ForCausalLM",
        model="EXAONE 4.0",
        huggingface_example="LGAI-EXAONE/EXAONE-4.0-32B",
    ),
    SupportedModel(
        architecture="Gemma3ForCausalLM",
        model="Gemma 3",
        huggingface_example="google/gemma-3-1b-it",
    ),
    SupportedModel(
        architecture="GptOssForCausalLM",
        model="GPT-OSS",
        huggingface_example="openai/gpt-oss-120b",
    ),
    SupportedModel(
        architecture="LlamaForCausalLM",
        model="Llama 3.1, Llama 3, Llama 2, LLaMA",
        huggingface_example="meta-llama/Meta-Llama-3.1-70B",
    ),
    SupportedModel(
        architecture="Llama4ForConditionalGeneration",
        model="Llama 4",
        huggingface_example="meta-llama/Llama-4-Scout-17B-16E-Instruct",
    ),
    SupportedModel(
        architecture="MiniMaxM2ForCausalLM",
        model="MiniMax M2/M2.1",
        huggingface_example="MiniMaxAI/MiniMax-M2",
    ),
    SupportedModel(
        architecture="MistralForCausalLM",
        model="Mistral",
        huggingface_example="mistralai/Mistral-7B-v0.1",
    ),
    SupportedModel(
        architecture="MixtralForCausalLM",
        model="Mixtral",
        huggingface_example="mistralai/Mixtral-8x7B-v0.1",
    ),
    SupportedModel(
        architecture="MllamaForConditionalGeneration",
        model="Llama 3.2",
        huggingface_example="meta-llama/Llama-3.2-11B-Vision",
    ),
    SupportedModel(
        architecture="NemotronForCausalLM",
        model="Nemotron-3, Nemotron-4, Minitron",
        huggingface_example="nvidia/Minitron-8B-Base",
    ),
    SupportedModel(
        architecture="NemotronHForCausalLM",
        model="Nemotron-3-Nano",
        huggingface_example="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
    ),
    SupportedModel(
        architecture="NemotronNASForCausalLM",
        model="NemotronNAS",
        huggingface_example="nvidia/Llama-3_3-Nemotron-Super-49B-v1",
    ),
    SupportedModel(
        architecture="Phi3ForCausalLM",
        model="Phi-4",
        huggingface_example="microsoft/Phi-4",
    ),
    SupportedModel(
        architecture="Qwen2ForCausalLM",
        model="QwQ, Qwen2",
        huggingface_example="Qwen/Qwen2-7B-Instruct",
    ),
    SupportedModel(
        architecture="Qwen2ForProcessRewardModel",
        model="Qwen2-based",
        huggingface_example="Qwen/Qwen2.5-Math-PRM-7B",
    ),
    SupportedModel(
        architecture="Qwen2ForRewardModel",
        model="Qwen2-based",
        huggingface_example="Qwen/Qwen2.5-Math-RM-72B",
    ),
    SupportedModel(
        architecture="Qwen3ForCausalLM",
        model="Qwen3",
        huggingface_example="Qwen/Qwen3-8B",
    ),
    SupportedModel(
        architecture="Qwen3MoeForCausalLM",
        model="Qwen3MoE",
        huggingface_example="Qwen/Qwen3-30B-A3B",
    ),
    SupportedModel(
        architecture="Qwen3NextForCausalLM",
        model="Qwen3Next",
        huggingface_example="Qwen/Qwen3-Next-80B-A3B-Thinking",
    ),
)
KEY_MODEL_FEATURES: Tuple[Feature, ...] = (
    Feature.OVERLAP_SCHEDULER,
    Feature.CUDA_GRAPH,
    Feature.ATTENTION_DP,
    Feature.DISAGGREGATED_SERVING,
    Feature.CHUNKED_PREFILL,
    Feature.MTP,
    Feature.EAGLE3_ONE_MODEL_ENGINE,
    Feature.EAGLE3_TWO_MODEL_ENGINE,
    Feature.TORCH_SAMPLER,
    Feature.TLLM_CPP_SAMPLER,
    Feature.KV_CACHE_REUSE,
    Feature.SLIDING_WINDOW_ATTENTION,
    Feature.LOGITS_POST_PROCESSOR,
    Feature.GUIDED_DECODING,
)

KEY_MODEL_MATRIX: Mapping[str, Mapping[Feature, FeatureCell]] = {
    "DeepseekV3ForCausalLM": {
        Feature.OVERLAP_SCHEDULER: FeatureCell(status=SupportStatus.YES),
        Feature.CUDA_GRAPH: FeatureCell(status=SupportStatus.YES),
        Feature.ATTENTION_DP: FeatureCell(status=SupportStatus.YES),
        Feature.DISAGGREGATED_SERVING: FeatureCell(status=SupportStatus.YES),
        Feature.CHUNKED_PREFILL: FeatureCell(status=SupportStatus.YES, footnote="[^1]"),
        Feature.MTP: FeatureCell(status=SupportStatus.YES),
        Feature.EAGLE3_ONE_MODEL_ENGINE: FeatureCell(status=SupportStatus.NO),
        Feature.EAGLE3_TWO_MODEL_ENGINE: FeatureCell(status=SupportStatus.NO),
        Feature.TORCH_SAMPLER: FeatureCell(status=SupportStatus.YES),
        Feature.TLLM_CPP_SAMPLER: FeatureCell(status=SupportStatus.YES),
        Feature.KV_CACHE_REUSE: FeatureCell(status=SupportStatus.YES, footnote="[^2]"),
        Feature.SLIDING_WINDOW_ATTENTION: FeatureCell(status=SupportStatus.NA),
        Feature.LOGITS_POST_PROCESSOR: FeatureCell(status=SupportStatus.YES),
        Feature.GUIDED_DECODING: FeatureCell(status=SupportStatus.YES),
    },
    "DeepseekV32ForCausalLM": {
        Feature.OVERLAP_SCHEDULER: FeatureCell(status=SupportStatus.YES),
        Feature.CUDA_GRAPH: FeatureCell(status=SupportStatus.YES),
        Feature.ATTENTION_DP: FeatureCell(status=SupportStatus.YES),
        Feature.DISAGGREGATED_SERVING: FeatureCell(status=SupportStatus.YES),
        Feature.CHUNKED_PREFILL: FeatureCell(status=SupportStatus.YES),
        Feature.MTP: FeatureCell(status=SupportStatus.YES),
        Feature.EAGLE3_ONE_MODEL_ENGINE: FeatureCell(status=SupportStatus.NO),
        Feature.EAGLE3_TWO_MODEL_ENGINE: FeatureCell(status=SupportStatus.NO),
        Feature.TORCH_SAMPLER: FeatureCell(status=SupportStatus.YES),
        Feature.TLLM_CPP_SAMPLER: FeatureCell(status=SupportStatus.YES),
        Feature.KV_CACHE_REUSE: FeatureCell(status=SupportStatus.YES),
        Feature.SLIDING_WINDOW_ATTENTION: FeatureCell(status=SupportStatus.NA),
        Feature.LOGITS_POST_PROCESSOR: FeatureCell(status=SupportStatus.YES),
        Feature.GUIDED_DECODING: FeatureCell(status=SupportStatus.YES),
    },
    "Qwen3MoeForCausalLM": {
        Feature.OVERLAP_SCHEDULER: FeatureCell(status=SupportStatus.YES),
        Feature.CUDA_GRAPH: FeatureCell(status=SupportStatus.YES),
        Feature.ATTENTION_DP: FeatureCell(status=SupportStatus.YES),
        Feature.DISAGGREGATED_SERVING: FeatureCell(status=SupportStatus.YES),
        Feature.CHUNKED_PREFILL: FeatureCell(status=SupportStatus.YES),
        Feature.MTP: FeatureCell(status=SupportStatus.NO),
        Feature.EAGLE3_ONE_MODEL_ENGINE: FeatureCell(status=SupportStatus.YES),
        Feature.EAGLE3_TWO_MODEL_ENGINE: FeatureCell(status=SupportStatus.YES),
        Feature.TORCH_SAMPLER: FeatureCell(status=SupportStatus.YES),
        Feature.TLLM_CPP_SAMPLER: FeatureCell(status=SupportStatus.YES),
        Feature.KV_CACHE_REUSE: FeatureCell(status=SupportStatus.YES),
        Feature.SLIDING_WINDOW_ATTENTION: FeatureCell(status=SupportStatus.NA),
        Feature.LOGITS_POST_PROCESSOR: FeatureCell(status=SupportStatus.YES),
        Feature.GUIDED_DECODING: FeatureCell(status=SupportStatus.YES),
    },
    "Qwen3NextForCausalLM": {
        Feature.OVERLAP_SCHEDULER: FeatureCell(status=SupportStatus.YES),
        Feature.CUDA_GRAPH: FeatureCell(status=SupportStatus.YES),
        Feature.ATTENTION_DP: FeatureCell(status=SupportStatus.NO),
        Feature.DISAGGREGATED_SERVING: FeatureCell(status=SupportStatus.UNTESTED),
        Feature.CHUNKED_PREFILL: FeatureCell(status=SupportStatus.YES),
        Feature.MTP: FeatureCell(status=SupportStatus.NO),
        Feature.EAGLE3_ONE_MODEL_ENGINE: FeatureCell(status=SupportStatus.NO),
        Feature.EAGLE3_TWO_MODEL_ENGINE: FeatureCell(status=SupportStatus.NO),
        Feature.TORCH_SAMPLER: FeatureCell(status=SupportStatus.YES),
        Feature.TLLM_CPP_SAMPLER: FeatureCell(status=SupportStatus.YES),
        Feature.KV_CACHE_REUSE: FeatureCell(status=SupportStatus.NO),
        Feature.SLIDING_WINDOW_ATTENTION: FeatureCell(status=SupportStatus.NO),
        Feature.LOGITS_POST_PROCESSOR: FeatureCell(status=SupportStatus.UNTESTED),
        Feature.GUIDED_DECODING: FeatureCell(status=SupportStatus.UNTESTED),
    },
    "Llama4ForConditionalGeneration": {
        Feature.OVERLAP_SCHEDULER: FeatureCell(status=SupportStatus.YES),
        Feature.CUDA_GRAPH: FeatureCell(status=SupportStatus.YES),
        Feature.ATTENTION_DP: FeatureCell(status=SupportStatus.YES),
        Feature.DISAGGREGATED_SERVING: FeatureCell(status=SupportStatus.YES),
        Feature.CHUNKED_PREFILL: FeatureCell(status=SupportStatus.YES),
        Feature.MTP: FeatureCell(status=SupportStatus.NO),
        Feature.EAGLE3_ONE_MODEL_ENGINE: FeatureCell(status=SupportStatus.YES),
        Feature.EAGLE3_TWO_MODEL_ENGINE: FeatureCell(status=SupportStatus.YES),
        Feature.TORCH_SAMPLER: FeatureCell(status=SupportStatus.YES),
        Feature.TLLM_CPP_SAMPLER: FeatureCell(status=SupportStatus.YES),
        Feature.KV_CACHE_REUSE: FeatureCell(status=SupportStatus.UNTESTED),
        Feature.SLIDING_WINDOW_ATTENTION: FeatureCell(status=SupportStatus.NA),
        Feature.LOGITS_POST_PROCESSOR: FeatureCell(status=SupportStatus.YES),
        Feature.GUIDED_DECODING: FeatureCell(status=SupportStatus.YES),
    },
    "GptOssForCausalLM": {
        Feature.OVERLAP_SCHEDULER: FeatureCell(status=SupportStatus.YES),
        Feature.CUDA_GRAPH: FeatureCell(status=SupportStatus.YES),
        Feature.ATTENTION_DP: FeatureCell(status=SupportStatus.YES),
        Feature.DISAGGREGATED_SERVING: FeatureCell(status=SupportStatus.YES),
        Feature.CHUNKED_PREFILL: FeatureCell(status=SupportStatus.YES),
        Feature.MTP: FeatureCell(status=SupportStatus.NO),
        Feature.EAGLE3_ONE_MODEL_ENGINE: FeatureCell(status=SupportStatus.YES),
        Feature.EAGLE3_TWO_MODEL_ENGINE: FeatureCell(status=SupportStatus.YES, footnote="[^4]"),
        Feature.TORCH_SAMPLER: FeatureCell(status=SupportStatus.YES),
        Feature.TLLM_CPP_SAMPLER: FeatureCell(status=SupportStatus.YES),
        Feature.KV_CACHE_REUSE: FeatureCell(status=SupportStatus.YES),
        Feature.SLIDING_WINDOW_ATTENTION: FeatureCell(status=SupportStatus.NA),
        Feature.LOGITS_POST_PROCESSOR: FeatureCell(status=SupportStatus.YES),
        Feature.GUIDED_DECODING: FeatureCell(status=SupportStatus.YES),
    },
}

KEY_MODEL_ARCH_ORDER: Tuple[str, ...] = (
    "DeepseekV3ForCausalLM",
    "DeepseekV32ForCausalLM",
    "Qwen3MoeForCausalLM",
    "Qwen3NextForCausalLM",
    "Llama4ForConditionalGeneration",
    "GptOssForCausalLM",
)

KEY_MODEL_FOOTNOTES: Tuple[str, ...] = (
    "[^1]: Chunked Prefill for MLA can only be enabled on SM100/SM103.",
    "[^2]: KV cache reuse for MLA can only be enabled on SM90/SM100/SM103 and in BF16/FP8 KV cache dtype.",
    "[^3]: Qwen3-Next-80B-A3B exhibits relatively low accuracy on the SciCode-AA-v2 benchmark.",
    "[^4]: Overlap scheduler isn't supported when using EAGLE-3(Two Model Engine) for GPT-OSS.",
)

# Architecture-level footnotes (attached to architecture name, not feature cells)
KEY_MODEL_ARCH_FOOTNOTES: Mapping[str, str] = {
    "Qwen3NextForCausalLM": "[^3]",
}


MULTIMODAL_FEATURES: Tuple[Feature, ...] = (
    Feature.OVERLAP_SCHEDULER,
    Feature.CUDA_GRAPH,
    Feature.CHUNKED_PREFILL,
    Feature.TORCH_SAMPLER,
    Feature.TLLM_CPP_SAMPLER,
    Feature.KV_CACHE_REUSE,
    Feature.LOGITS_POST_PROCESSOR,
    Feature.EPD_DISAGG_SERVING,
    Feature.MODALITY,
)

MULTIMODAL_MATRIX: Mapping[str, Mapping[Feature, FeatureCell]] = {
    "Gemma3ForConditionalGeneration": {
        Feature.OVERLAP_SCHEDULER: FeatureCell(status=SupportStatus.YES),
        Feature.CUDA_GRAPH: FeatureCell(status=SupportStatus.YES),
        Feature.CHUNKED_PREFILL: FeatureCell(status=SupportStatus.NA),
        Feature.TORCH_SAMPLER: FeatureCell(status=SupportStatus.YES),
        Feature.TLLM_CPP_SAMPLER: FeatureCell(status=SupportStatus.YES),
        Feature.KV_CACHE_REUSE: FeatureCell(status=SupportStatus.NA),
        Feature.LOGITS_POST_PROCESSOR: FeatureCell(status=SupportStatus.YES),
        Feature.EPD_DISAGG_SERVING: FeatureCell(status=SupportStatus.NO),
        Feature.MODALITY: FeatureCell(text="L + I"),
    },
    "HCXVisionForCausalLM": {
        Feature.OVERLAP_SCHEDULER: FeatureCell(status=SupportStatus.YES),
        Feature.CUDA_GRAPH: FeatureCell(status=SupportStatus.YES),
        Feature.CHUNKED_PREFILL: FeatureCell(status=SupportStatus.NO),
        Feature.TORCH_SAMPLER: FeatureCell(status=SupportStatus.YES),
        Feature.TLLM_CPP_SAMPLER: FeatureCell(status=SupportStatus.YES),
        Feature.KV_CACHE_REUSE: FeatureCell(status=SupportStatus.YES),
        Feature.LOGITS_POST_PROCESSOR: FeatureCell(status=SupportStatus.YES),
        Feature.EPD_DISAGG_SERVING: FeatureCell(status=SupportStatus.NO),
        Feature.MODALITY: FeatureCell(text="L + I"),
    },
    "LlavaLlamaModel": {
        Feature.OVERLAP_SCHEDULER: FeatureCell(status=SupportStatus.YES),
        Feature.CUDA_GRAPH: FeatureCell(status=SupportStatus.YES),
        Feature.CHUNKED_PREFILL: FeatureCell(status=SupportStatus.NO),
        Feature.TORCH_SAMPLER: FeatureCell(status=SupportStatus.YES),
        Feature.TLLM_CPP_SAMPLER: FeatureCell(status=SupportStatus.YES),
        Feature.KV_CACHE_REUSE: FeatureCell(status=SupportStatus.NO),
        Feature.LOGITS_POST_PROCESSOR: FeatureCell(status=SupportStatus.YES),
        Feature.EPD_DISAGG_SERVING: FeatureCell(status=SupportStatus.NO),
        Feature.MODALITY: FeatureCell(text="L + I + V"),
    },
    "LlavaNextForConditionalGeneration": {
        Feature.OVERLAP_SCHEDULER: FeatureCell(status=SupportStatus.YES),
        Feature.CUDA_GRAPH: FeatureCell(status=SupportStatus.YES),
        Feature.CHUNKED_PREFILL: FeatureCell(status=SupportStatus.YES),
        Feature.TORCH_SAMPLER: FeatureCell(status=SupportStatus.YES),
        Feature.TLLM_CPP_SAMPLER: FeatureCell(status=SupportStatus.YES),
        Feature.KV_CACHE_REUSE: FeatureCell(status=SupportStatus.YES),
        Feature.LOGITS_POST_PROCESSOR: FeatureCell(status=SupportStatus.YES),
        Feature.EPD_DISAGG_SERVING: FeatureCell(status=SupportStatus.YES),
        Feature.MODALITY: FeatureCell(text="L + I"),
    },
    "Llama4ForConditionalGeneration": {
        Feature.OVERLAP_SCHEDULER: FeatureCell(status=SupportStatus.YES),
        Feature.CUDA_GRAPH: FeatureCell(status=SupportStatus.YES),
        Feature.CHUNKED_PREFILL: FeatureCell(status=SupportStatus.NO),
        Feature.TORCH_SAMPLER: FeatureCell(status=SupportStatus.YES),
        Feature.TLLM_CPP_SAMPLER: FeatureCell(status=SupportStatus.YES),
        Feature.KV_CACHE_REUSE: FeatureCell(status=SupportStatus.NO),
        Feature.LOGITS_POST_PROCESSOR: FeatureCell(status=SupportStatus.YES),
        Feature.EPD_DISAGG_SERVING: FeatureCell(status=SupportStatus.NO),
        Feature.MODALITY: FeatureCell(text="L + I"),
    },
    "Mistral3ForConditionalGeneration": {
        Feature.OVERLAP_SCHEDULER: FeatureCell(status=SupportStatus.YES),
        Feature.CUDA_GRAPH: FeatureCell(status=SupportStatus.YES),
        Feature.CHUNKED_PREFILL: FeatureCell(status=SupportStatus.YES),
        Feature.TORCH_SAMPLER: FeatureCell(status=SupportStatus.YES),
        Feature.TLLM_CPP_SAMPLER: FeatureCell(status=SupportStatus.YES),
        Feature.KV_CACHE_REUSE: FeatureCell(status=SupportStatus.YES),
        Feature.LOGITS_POST_PROCESSOR: FeatureCell(status=SupportStatus.YES),
        Feature.EPD_DISAGG_SERVING: FeatureCell(status=SupportStatus.NO),
        Feature.MODALITY: FeatureCell(text="L + I"),
    },
    "NemotronH_Nano_VL_V2": {
        Feature.OVERLAP_SCHEDULER: FeatureCell(status=SupportStatus.YES),
        Feature.CUDA_GRAPH: FeatureCell(status=SupportStatus.YES),
        Feature.CHUNKED_PREFILL: FeatureCell(status=SupportStatus.YES),
        Feature.TORCH_SAMPLER: FeatureCell(status=SupportStatus.YES),
        Feature.TLLM_CPP_SAMPLER: FeatureCell(status=SupportStatus.YES),
        Feature.KV_CACHE_REUSE: FeatureCell(status=SupportStatus.NA),
        Feature.LOGITS_POST_PROCESSOR: FeatureCell(status=SupportStatus.YES),
        Feature.EPD_DISAGG_SERVING: FeatureCell(status=SupportStatus.NO),
        Feature.MODALITY: FeatureCell(text="L + I + V"),
    },
    "Phi4MMForCausalLM": {
        Feature.OVERLAP_SCHEDULER: FeatureCell(status=SupportStatus.YES),
        Feature.CUDA_GRAPH: FeatureCell(status=SupportStatus.YES),
        Feature.CHUNKED_PREFILL: FeatureCell(status=SupportStatus.YES),
        Feature.TORCH_SAMPLER: FeatureCell(status=SupportStatus.YES),
        Feature.TLLM_CPP_SAMPLER: FeatureCell(status=SupportStatus.YES),
        Feature.KV_CACHE_REUSE: FeatureCell(status=SupportStatus.YES),
        Feature.LOGITS_POST_PROCESSOR: FeatureCell(status=SupportStatus.YES),
        Feature.EPD_DISAGG_SERVING: FeatureCell(status=SupportStatus.NO),
        Feature.MODALITY: FeatureCell(text="L + I + A"),
    },
    "Qwen2VLForConditionalGeneration": {
        Feature.OVERLAP_SCHEDULER: FeatureCell(status=SupportStatus.YES),
        Feature.CUDA_GRAPH: FeatureCell(status=SupportStatus.YES),
        Feature.CHUNKED_PREFILL: FeatureCell(status=SupportStatus.YES),
        Feature.TORCH_SAMPLER: FeatureCell(status=SupportStatus.YES),
        Feature.TLLM_CPP_SAMPLER: FeatureCell(status=SupportStatus.YES),
        Feature.KV_CACHE_REUSE: FeatureCell(status=SupportStatus.YES),
        Feature.LOGITS_POST_PROCESSOR: FeatureCell(status=SupportStatus.YES),
        Feature.EPD_DISAGG_SERVING: FeatureCell(status=SupportStatus.NO),
        Feature.MODALITY: FeatureCell(text="L + I + V"),
    },
    "Qwen2_5_VLForConditionalGeneration": {
        Feature.OVERLAP_SCHEDULER: FeatureCell(status=SupportStatus.YES),
        Feature.CUDA_GRAPH: FeatureCell(status=SupportStatus.YES),
        Feature.CHUNKED_PREFILL: FeatureCell(status=SupportStatus.YES),
        Feature.TORCH_SAMPLER: FeatureCell(status=SupportStatus.YES),
        Feature.TLLM_CPP_SAMPLER: FeatureCell(status=SupportStatus.YES),
        Feature.KV_CACHE_REUSE: FeatureCell(status=SupportStatus.YES),
        Feature.LOGITS_POST_PROCESSOR: FeatureCell(status=SupportStatus.YES),
        Feature.EPD_DISAGG_SERVING: FeatureCell(status=SupportStatus.NO),
        Feature.MODALITY: FeatureCell(text="L + I + V"),
    },
}

MULTIMODAL_ARCH_ORDER: Tuple[str, ...] = (
    "Gemma3ForConditionalGeneration",
    "HCXVisionForCausalLM",
    "LlavaLlamaModel",
    "LlavaNextForConditionalGeneration",
    "Llama4ForConditionalGeneration",
    "Mistral3ForConditionalGeneration",
    "NemotronH_Nano_VL_V2",
    "Phi4MMForCausalLM",
    "Qwen2VLForConditionalGeneration",
    "Qwen2_5_VLForConditionalGeneration",
)


def get_cell(architecture: str, feature: Feature) -> Optional[FeatureCell]:
    row = MULTIMODAL_MATRIX.get(architecture)
    if row is not None:
        cell = row.get(feature)
        if cell is not None:
            return cell
    row = KEY_MODEL_MATRIX.get(architecture)
    if row is not None:
        cell = row.get(feature)
        if cell is not None:
            return cell
    return None


def get_status(architecture: str, feature: Feature) -> Optional[SupportStatus]:
    cell = get_cell(architecture, feature)
    if cell is None:
        return None
    return cell.status


def _render_md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _row(cells: Sequence[str]) -> str:
        padded = [cells[i].ljust(widths[i]) for i in range(len(widths))]
        return "| " + " | ".join(padded) + " |"

    def _sep() -> str:
        return _row(["-" * w for w in widths])

    lines: List[str] = []
    lines.append(_row(headers))
    lines.append(_sep())
    for row in rows:
        lines.append(_row(row))
    return "\n".join(lines)


def render_supported_models_markdown() -> str:
    """Render the full `docs/source/models/supported-models.md` content."""
    out: List[str] = []
    out.append("(support-matrix)=")
    out.append("# Supported Models")
    out.append("")
    out.append("The following is a table of supported models for the PyTorch backend:")
    out.append("")

    supported_rows = [
        [
            f"`{m.architecture}`",
            m.model,
            f"`{m.huggingface_example}`",
        ]
        for m in SUPPORTED_MODELS_PYTORCH
    ]
    out.append(
        _render_md_table(
            headers=["Architecture", "Model", "HuggingFace Example"],
            rows=supported_rows,
        )
    )
    out.append("")
    out.append("")
    out.append("## Model-Feature Support Matrix(Key Models)")
    out.append("")
    out.append(
        'Note: Support for other models may vary. Features marked "N/A" are not applicable to the model architecture.'
    )
    out.append("")

    key_headers = ["Model Architecture/Feature"] + [feature.value for feature in KEY_MODEL_FEATURES]
    key_rows: List[List[str]] = []
    for arch in KEY_MODEL_ARCH_ORDER:
        cells = KEY_MODEL_MATRIX.get(arch, {})
        arch_footnote = KEY_MODEL_ARCH_FOOTNOTES.get(arch, "")
        arch_cell = f"`{arch}` {arch_footnote}".rstrip() if arch_footnote else f"`{arch}`"
        key_rows.append(
            [arch_cell]
            + [cells.get(feature, FeatureCell()).render() for feature in KEY_MODEL_FEATURES]
        )
    out.append(_render_md_table(headers=key_headers, rows=key_rows))
    out.append("")
    for fn in KEY_MODEL_FOOTNOTES:
        out.append(fn)
    out.append("")
    out.append("")
    out.append("# Multimodal Feature Support Matrix (PyTorch Backend)")
    out.append("")

    mm_headers = ["Model Architecture/Feature"] + [feature.value for feature in MULTIMODAL_FEATURES]
    mm_rows: List[List[str]] = []
    for arch in MULTIMODAL_ARCH_ORDER:
        cells = MULTIMODAL_MATRIX.get(arch, {})
        mm_rows.append(
            [f"`{arch}`"]
            + [cells.get(feature, FeatureCell()).render() for feature in MULTIMODAL_FEATURES]
        )
    out.append(_render_md_table(headers=mm_headers, rows=mm_rows))
    out.append("")
    out.append("Note:")
    out.append("- L: Language")
    out.append("- I: Image")
    out.append("- V: Video")
    out.append("- A: Audio")
    out.append("")
    return "\n".join(out)
