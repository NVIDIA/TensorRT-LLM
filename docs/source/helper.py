import importlib.util
import logging
import os
import re
import sys
from dataclasses import dataclass
from itertools import chain, groupby
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

import pygit2

# =============================================================================
# Model Support Matrix - Documentation Data Structures
# =============================================================================
# These are used for generating supported-models.md documentation.
# Runtime code should use tensorrt_llm._torch.models.modeling_utils instead
# (get_status, is_feature_unsupported functions).


def _load_feature_types():
    """Load feature_types.py directly (stdlib-only, no torch dependency).

    This avoids importing tensorrt_llm which would trigger torch imports
    that aren't available in documentation build environments.
    """
    module_path = Path(
        __file__
    ).parent.parent.parent / "tensorrt_llm/_torch/models/feature_types.py"
    spec = importlib.util.spec_from_file_location("feature_types", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    # Required for dataclasses to work correctly in Python 3.14+
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# Import Feature and SupportStatus from the canonical source
_feature_types = _load_feature_types()
SupportStatus = _feature_types.SupportStatus
Feature = _feature_types.Feature


@dataclass(frozen=True)
class SupportedModel:
    """A supported model entry for documentation."""
    architecture: str
    model: str
    huggingface: str


@dataclass(frozen=True)
class FeatureCell:
    """A cell in the feature support matrix for documentation."""
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
        huggingface="textattack/bert-base-uncased-yelp-polarity",
    ),
    SupportedModel(
        architecture="DeciLMForCausalLM",
        model="Nemotron",
        huggingface="nvidia/Llama-3_1-Nemotron-51B-Instruct",
    ),
    SupportedModel(
        architecture="DeepseekV3ForCausalLM",
        model="DeepSeek-V3",
        huggingface="deepseek-ai/DeepSeek-V3",
    ),
    SupportedModel(
        architecture="DeepseekV32ForCausalLM",
        model="DeepSeek-V3.2",
        huggingface="deepseek-ai/DeepSeek-V3.2",
    ),
    SupportedModel(
        architecture="Exaone4ForCausalLM",
        model="EXAONE 4.0",
        huggingface="LGAI-EXAONE/EXAONE-4.0-32B",
    ),
    SupportedModel(
        architecture="Gemma3ForCausalLM",
        model="Gemma 3",
        huggingface="google/gemma-3-1b-it",
    ),
    SupportedModel(
        architecture="GptOssForCausalLM",
        model="GPT-OSS",
        huggingface="openai/gpt-oss-120b",
    ),
    SupportedModel(
        architecture="LlamaForCausalLM",
        model="Llama 3.1, Llama 3, Llama 2, LLaMA",
        huggingface="meta-llama/Meta-Llama-3.1-70B",
    ),
    SupportedModel(
        architecture="Llama4ForConditionalGeneration",
        model="Llama 4",
        huggingface="meta-llama/Llama-4-Scout-17B-16E-Instruct",
    ),
    SupportedModel(
        architecture="MiniMaxM2ForCausalLM",
        model="MiniMax M2/M2.1",
        huggingface="MiniMaxAI/MiniMax-M2",
    ),
    SupportedModel(
        architecture="MistralForCausalLM",
        model="Mistral",
        huggingface="mistralai/Mistral-7B-v0.1",
    ),
    SupportedModel(
        architecture="MixtralForCausalLM",
        model="Mixtral",
        huggingface="mistralai/Mixtral-8x7B-v0.1",
    ),
    SupportedModel(
        architecture="MllamaForConditionalGeneration",
        model="Llama 3.2",
        huggingface="meta-llama/Llama-3.2-11B-Vision",
    ),
    SupportedModel(
        architecture="NemotronForCausalLM",
        model="Nemotron-3, Nemotron-4, Minitron",
        huggingface="nvidia/Minitron-8B-Base",
    ),
    SupportedModel(
        architecture="NemotronHForCausalLM",
        model="Nemotron-3-Nano",
        huggingface="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
    ),
    SupportedModel(
        architecture="NemotronNASForCausalLM",
        model="NemotronNAS",
        huggingface="nvidia/Llama-3_3-Nemotron-Super-49B-v1",
    ),
    SupportedModel(
        architecture="Phi3ForCausalLM",
        model="Phi-4",
        huggingface="microsoft/Phi-4",
    ),
    SupportedModel(
        architecture="Qwen2ForCausalLM",
        model="QwQ, Qwen2",
        huggingface="Qwen/Qwen2-7B-Instruct",
    ),
    SupportedModel(
        architecture="Qwen2ForProcessRewardModel",
        model="Qwen2-based",
        huggingface="Qwen/Qwen2.5-Math-PRM-7B",
    ),
    SupportedModel(
        architecture="Qwen2ForRewardModel",
        model="Qwen2-based",
        huggingface="Qwen/Qwen2.5-Math-RM-72B",
    ),
    SupportedModel(
        architecture="Qwen3ForCausalLM",
        model="Qwen3",
        huggingface="Qwen/Qwen3-8B",
    ),
    SupportedModel(
        architecture="Qwen3MoeForCausalLM",
        model="Qwen3MoE",
        huggingface="Qwen/Qwen3-30B-A3B",
    ),
    SupportedModel(
        architecture="Qwen3NextForCausalLM",
        model="Qwen3Next",
        huggingface="Qwen/Qwen3-Next-80B-A3B-Thinking",
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
        Feature.OVERLAP_SCHEDULER:
        FeatureCell(status=SupportStatus.YES),
        Feature.CUDA_GRAPH:
        FeatureCell(status=SupportStatus.YES),
        Feature.ATTENTION_DP:
        FeatureCell(status=SupportStatus.YES),
        Feature.DISAGGREGATED_SERVING:
        FeatureCell(status=SupportStatus.YES),
        Feature.CHUNKED_PREFILL:
        FeatureCell(status=SupportStatus.YES, footnote="[^1]"),
        Feature.MTP:
        FeatureCell(status=SupportStatus.YES),
        Feature.EAGLE3_ONE_MODEL_ENGINE:
        FeatureCell(status=SupportStatus.NO),
        Feature.EAGLE3_TWO_MODEL_ENGINE:
        FeatureCell(status=SupportStatus.NO),
        Feature.TORCH_SAMPLER:
        FeatureCell(status=SupportStatus.YES),
        Feature.TLLM_CPP_SAMPLER:
        FeatureCell(status=SupportStatus.YES),
        Feature.KV_CACHE_REUSE:
        FeatureCell(status=SupportStatus.YES, footnote="[^2]"),
        Feature.SLIDING_WINDOW_ATTENTION:
        FeatureCell(status=SupportStatus.NA),
        Feature.LOGITS_POST_PROCESSOR:
        FeatureCell(status=SupportStatus.YES),
        Feature.GUIDED_DECODING:
        FeatureCell(status=SupportStatus.YES),
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
        Feature.DISAGGREGATED_SERVING:
        FeatureCell(status=SupportStatus.UNTESTED),
        Feature.CHUNKED_PREFILL: FeatureCell(status=SupportStatus.YES),
        Feature.MTP: FeatureCell(status=SupportStatus.NO),
        Feature.EAGLE3_ONE_MODEL_ENGINE: FeatureCell(status=SupportStatus.NO),
        Feature.EAGLE3_TWO_MODEL_ENGINE: FeatureCell(status=SupportStatus.NO),
        Feature.TORCH_SAMPLER: FeatureCell(status=SupportStatus.YES),
        Feature.TLLM_CPP_SAMPLER: FeatureCell(status=SupportStatus.YES),
        Feature.KV_CACHE_REUSE: FeatureCell(status=SupportStatus.NO),
        Feature.SLIDING_WINDOW_ATTENTION: FeatureCell(status=SupportStatus.NO),
        Feature.LOGITS_POST_PROCESSOR:
        FeatureCell(status=SupportStatus.UNTESTED),
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
        Feature.OVERLAP_SCHEDULER:
        FeatureCell(status=SupportStatus.YES),
        Feature.CUDA_GRAPH:
        FeatureCell(status=SupportStatus.YES),
        Feature.ATTENTION_DP:
        FeatureCell(status=SupportStatus.YES),
        Feature.DISAGGREGATED_SERVING:
        FeatureCell(status=SupportStatus.YES),
        Feature.CHUNKED_PREFILL:
        FeatureCell(status=SupportStatus.YES),
        Feature.MTP:
        FeatureCell(status=SupportStatus.NO),
        Feature.EAGLE3_ONE_MODEL_ENGINE:
        FeatureCell(status=SupportStatus.YES),
        Feature.EAGLE3_TWO_MODEL_ENGINE:
        FeatureCell(status=SupportStatus.YES, footnote="[^4]"),
        Feature.TORCH_SAMPLER:
        FeatureCell(status=SupportStatus.YES),
        Feature.TLLM_CPP_SAMPLER:
        FeatureCell(status=SupportStatus.YES),
        Feature.KV_CACHE_REUSE:
        FeatureCell(status=SupportStatus.YES),
        Feature.SLIDING_WINDOW_ATTENTION:
        FeatureCell(status=SupportStatus.NA),
        Feature.LOGITS_POST_PROCESSOR:
        FeatureCell(status=SupportStatus.YES),
        Feature.GUIDED_DECODING:
        FeatureCell(status=SupportStatus.YES),
    },
}

KEY_MODEL_FOOTNOTES: Tuple[str, ...] = (
    "[^1]: Chunked Prefill for MLA can only be enabled on SM100/SM103.",
    "[^2]: KV cache reuse for MLA can only be enabled on SM90/SM100/SM103 and in BF16/FP8 KV cache dtype.",
    "[^3]: Qwen3-Next-80B-A3B exhibits relatively low accuracy on the SciCode-AA-v2 benchmark.",
    "[^4]: Overlap scheduler isn't supported when using EAGLE-3(Two Model Engine) for GPT-OSS.",
)

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


def get_cell(architecture: str, feature: Feature) -> Optional[FeatureCell]:
    """Get a FeatureCell from the hardcoded matrices (for documentation)."""
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


def _render_md_table(headers: Sequence[str],
                     rows: Sequence[Sequence[str]]) -> str:
    """Render a markdown table."""
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
    out.append("<!-- Generated from docs/source/helper.py; do not edit. -->")
    out.append(
        "<!-- To regenerate: build docs, which runs generate_supported_models() -->"
    )
    out.append("# Supported Models")
    out.append("")
    out.append(
        "The following is a table of supported models for the PyTorch backend:")
    out.append("")

    supported_rows = [[
        f"`{m.architecture}`",
        m.model,
        f"`{m.huggingface}`",
    ] for m in SUPPORTED_MODELS_PYTORCH]
    out.append(
        _render_md_table(
            headers=["Architecture", "Model", "HuggingFace Example"],
            rows=supported_rows,
        ))
    out.append("")
    out.append("")
    out.append("## Model-Feature Support Matrix(Key Models)")
    out.append("")
    out.append(
        'Note: Support for other models may vary. Features marked "N/A" are not applicable to the model architecture.'
    )
    out.append("")

    key_headers = ["Model Architecture/Feature"
                   ] + [feature.description for feature in KEY_MODEL_FEATURES]
    key_rows: List[List[str]] = []
    for arch, cells in KEY_MODEL_MATRIX.items():
        arch_footnote = KEY_MODEL_ARCH_FOOTNOTES.get(arch, "")
        arch_cell = f"`{arch}` {arch_footnote}".rstrip(
        ) if arch_footnote else f"`{arch}`"
        key_rows.append([arch_cell] + [
            cells.get(feature, FeatureCell()).render()
            for feature in KEY_MODEL_FEATURES
        ])
    out.append(_render_md_table(headers=key_headers, rows=key_rows))
    out.append("")
    for fn in KEY_MODEL_FOOTNOTES:
        out.append(fn)
    out.append("")
    out.append("")
    out.append("# Multimodal Feature Support Matrix (PyTorch Backend)")
    out.append("")

    mm_headers = ["Model Architecture/Feature"
                  ] + [feature.description for feature in MULTIMODAL_FEATURES]
    mm_rows: List[List[str]] = []
    for arch, cells in MULTIMODAL_MATRIX.items():
        mm_rows.append([f"`{arch}`"] + [
            cells.get(feature, FeatureCell()).render()
            for feature in MULTIMODAL_FEATURES
        ])
    out.append(_render_md_table(headers=mm_headers, rows=mm_rows))
    out.append("")
    out.append("Note:")
    out.append("- L: Language")
    out.append("- I: Image")
    out.append("- V: Video")
    out.append("- A: Audio")
    return "\n".join(out)


# =============================================================================
# End of Model Support Matrix section
# =============================================================================


def underline(title: str, character: str = "=") -> str:
    return f"{title}\n{character * len(title)}"


def generate_title(filename: str) -> str:
    with open(filename) as f:
        # fine the first line that contains '###'
        for line in f:
            if '###' in line:
                title = line[3:].strip()
                break
        assert title is not None, f"No title found in {filename}"
    return underline(title)


@dataclass
class DocMeta:
    title: str
    order: int
    section: str
    filename: Path


def extract_meta_info(filename: str) -> Optional[DocMeta]:
    """Extract metadata from file following the pattern ### :[a-zA-Z_]+[0-9]* <value>"""
    metadata_pattern = re.compile(r'^### :([a-zA-Z_]+[0-9]*)\s+(.+)$')

    with open(filename) as f:
        metadata = DocMeta(title="",
                           order=0,
                           section="",
                           filename=Path(filename))

        for line in f:
            line = line.strip()
            match = metadata_pattern.match(line)
            if match:
                key = match.group(1).strip()
                value = match.group(2).strip()
                setattr(metadata, key, value)
            elif not line.startswith('###'):
                continue
        if metadata.title == "":
            return None
        return metadata


# NOTE: Update here to keep consistent with the examples
LLMAPI_SECTIONS = ["Basics", "Customization", "Slurm"]


def generate_examples():
    root_dir = Path(__file__).parent.parent.parent.resolve()
    ignore_list = {
        '__init__.py', 'quickstart_example.py', 'quickstart_advanced.py',
        'quickstart_multimodal.py', 'star_attention.py'
    }
    doc_dir = root_dir / "docs/source/examples"

    def collect_script_paths(examples_subdir: str) -> list[Path]:
        """Collect Python and shell script paths from an examples subdirectory."""
        script_dir = root_dir / f"examples/{examples_subdir}"
        script_paths = list(
            chain(script_dir.glob("*.py"), script_dir.glob("*.sh")))
        return [
            path for path in sorted(script_paths)
            if path.name not in ignore_list
        ]

    # Collect source paths for LLMAPI examples
    llmapi_script_paths = collect_script_paths("llm-api")
    llmapi_doc_paths = [
        doc_dir / f"{path.stem}.rst" for path in llmapi_script_paths
    ]
    repo = pygit2.Repository('.')
    commit_hash = str(repo.head.target)
    llmapi_script_base_url = f"https://github.com/NVIDIA/TensorRT-LLM/blob/{commit_hash}/examples/llm-api"

    # Collect source paths for trtllm-serve examples
    serve_script_paths = collect_script_paths("serve")
    serve_doc_paths = [
        doc_dir / f"{path.stem}.rst" for path in serve_script_paths
    ]
    serve_script_base_url = f"https://github.com/NVIDIA/TensorRT-LLM/blob/{commit_hash}/examples/serve"

    def _get_lines_without_metadata(filename: str) -> str:
        """Get line ranges that exclude metadata lines.
        Returns a string like "5-10,15-20" for use in :lines: directive.
        """
        with open(filename) as f:
            metadata_pattern = re.compile(r'^### :([a-zA-Z_]+[0-9]*)\s+(.+)$')
            all_lines = f.readlines()

        # Find line numbers that are NOT metadata (1-indexed)
        content_lines = []
        for line_num, line in enumerate(all_lines, 1):
            line_stripped = line.strip()
            # Include line if it's not empty and not metadata
            if not metadata_pattern.match(line_stripped):
                content_lines.append(line_num)

        if not content_lines:
            return ""  # No content lines found

        # Group consecutive line numbers into ranges
        ranges = []
        start = content_lines[0]
        end = start

        for line_num in content_lines[1:]:
            if line_num == end + 1:
                # Consecutive line, extend current range
                end = line_num
            else:
                # Gap found, close current range and start new one
                if start == end:
                    ranges.append(str(start))
                else:
                    ranges.append(f"{start}-{end}")
                start = line_num
                end = line_num

        # Add the final range
        if start == end:
            ranges.append(str(start))
        else:
            ranges.append(f"{start}-{end}")

        return ",".join(ranges)

    # Generate the example docs for each example script
    def write_scripts(base_url: str,
                      example_script_paths: list[Path],
                      doc_paths: list[Path],
                      extra_content="") -> list[DocMeta]:
        metas = []
        for script_path, doc_path in zip(example_script_paths, doc_paths):
            if script_path.name in ignore_list:
                logging.warning(f"Ignoring file: {script_path.name}")
                continue
            script_url = f"{base_url}/{script_path.name}"
            # Determine language based on file extension
            language = "python" if script_path.suffix == ".py" else "bash"

            # Make script_path relative to doc_path and call it include_path
            include_path = '../../..' / script_path.relative_to(root_dir)

            # Extract metadata from the script file
            if meta := extract_meta_info(str(script_path)):
                title = underline(meta.title)
            else:
                logging.warning(
                    f"No metadata found for {script_path.name}, using filename as title"
                )
                title = script_path.stem.replace('_', ' ').title()
                meta = DocMeta(title=title,
                               order=0,
                               section="",
                               filename=script_path)
                title = underline(title)
            metas.append(meta)

            # Get line ranges excluding metadata
            lines_without_metadata = _get_lines_without_metadata(
                str(script_path))

            # Build literalinclude directive
            literalinclude_lines = [f".. literalinclude:: {include_path}"]
            if lines_without_metadata:
                literalinclude_lines.append(
                    f"    :lines: {lines_without_metadata}")
            literalinclude_lines.extend(
                [f"    :language: {language}", f"    :linenos:"])

            content = (f"{title}\n"
                       f"{extra_content}"
                       f"Source {script_url}.\n\n"
                       f"{chr(10).join(literalinclude_lines)}\n")
            with open(doc_path, "w+") as f:
                logging.warning(f"Writing {doc_path}")
                f.write(content)

        return metas

    def write_index(metas: list[DocMeta], doc_template_path: Path,
                    doc_path: Path, example_name: str,
                    section_order: list[str]):
        """Write the index file for the examples.

        Args:
            metas: The metadata for the examples.
            doc_template_path: The path to the template file.
            doc_path: The path to the output file.
            example_name: The name of the examples.
            section_order: The order of sections to display.

        The template file is expected to have the following placeholders:
        - %EXAMPLE_DOCS%: The documentation for the examples.
        - %EXAMPLE_NAME%: The name of the examples.
        """
        with open(doc_template_path) as f:
            template_content = f.read()

        # Sort metadata by section order and example order
        sort_key = lambda x: (section_order.index(x.section)
                              if section_order and x.section in section_order
                              else 0, int(x.order))
        metas.sort(key=sort_key)

        content = []
        for section, group in groupby(metas, key=lambda x: x.section):
            if section_order and section not in section_order:
                raise ValueError(
                    f"Section '{section}' not in section_order {section_order}")

            group_list = list(group)
            content.extend([
                section, "_" * len(section), "", ".. toctree::",
                "   :maxdepth: 2", ""
            ])

            for meta in group_list:
                content.append(f"   {meta.filename.stem}")
            content.append("")

        example_docs = "\n".join(content)

        # Replace placeholders and write to file
        output_content = template_content.replace("%EXAMPLE_DOCS%",
                                                  example_docs).replace(
                                                      "%EXAMPLE_NAME%",
                                                      example_name)
        with open(doc_path, "w") as f:
            f.write(output_content)

    # Generate the toctree for LLMAPI example scripts
    llmapi_metas = write_scripts(llmapi_script_base_url, llmapi_script_paths,
                                 llmapi_doc_paths)
    write_index(metas=llmapi_metas,
                doc_template_path=doc_dir / "llm_examples_index.template.rst_",
                doc_path=doc_dir / "llm_api_examples.rst",
                example_name="LLM Examples",
                section_order=LLMAPI_SECTIONS)

    # Generate the toctree for trtllm-serve example scripts
    serve_extra_content = (
        "Refer to the `trtllm-serve documentation "
        "<https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve.html>`_ "
        "for starting a server.\n\n")
    serve_metas = write_scripts(serve_script_base_url, serve_script_paths,
                                serve_doc_paths, serve_extra_content)
    write_index(metas=serve_metas,
                doc_template_path=doc_dir / "llm_examples_index.template.rst_",
                doc_path=doc_dir / "trtllm_serve_examples.rst",
                example_name="Online Serving Examples",
                section_order=[])


def extract_all_and_eval(file_path):
    ''' Extract the __all__ variable from a Python file.
    This is a trick to make the CI happy even the tensorrt_llm lib is not available.
    NOTE: This requires the __all__ variable to be defined at the end of the file.
    '''
    with open(file_path, 'r') as file:
        content = file.read()

    lines = content.split('\n')
    filtered_line_begin = 0

    for i, line in enumerate(lines):
        if line.startswith("__all__"):
            filtered_line_begin = i
            break

    code_to_eval = '\n'.join(lines[filtered_line_begin:])

    local_vars = {}
    exec(code_to_eval, {}, local_vars)
    return local_vars


def get_pydantic_methods() -> list[str]:
    from pydantic import BaseModel

    class Dummy(BaseModel):
        pass

    methods = set(
        [method for method in dir(Dummy) if not method.startswith('_')])
    methods.discard("__init__")
    return list(methods)


def generate_llmapi():
    root_dir = Path(__file__).parent.parent.parent.resolve()

    # Set up destination paths
    doc_dir = root_dir / "docs/source/llm-api"
    doc_dir.mkdir(exist_ok=True)
    doc_path = doc_dir / "reference.rst"

    llmapi_all_file = root_dir / "tensorrt_llm/llmapi/__init__.py"
    public_classes_names = extract_all_and_eval(llmapi_all_file)['__all__']

    content = underline("API Reference", "-") + "\n\n"
    content += ".. note::\n"
    content += "    Since version 1.0, we have attached a status label to `LLM`, `LlmArgs` and `TorchLlmArgs` Classes.\n\n"
    content += "    1. :tag:`stable` - The item is stable and will keep consistent.\n"
    content += '    2. :tag:`prototype` - The item is a prototype and is subject to change.\n'
    content += '    3. :tag:`beta` - The item is in beta and approaching stability.\n'
    content += '    4. :tag:`deprecated` - The item is deprecated and will be removed in a future release.\n'
    content += "\n"

    for cls_name in public_classes_names:
        cls_name = cls_name.strip()
        options = [
            "    :members:",
            "    :undoc-members:",
            "    :show-inheritance:",
            "    :special-members: __init__",
            "    :member-order: groupwise",
        ]

        options.append("    :inherited-members:")
        if cls_name in ["TorchLlmArgs", "TrtLlmArgs"]:
            # exclude tons of methods from Pydantic
            options.append(
                f"    :exclude-members: {','.join(get_pydantic_methods())}")

        content += f".. autoclass:: tensorrt_llm.llmapi.{cls_name}\n"
        content += "\n".join(options) + "\n\n"
    with open(doc_path, "w+") as f:
        f.write(content)


def generate_supported_models():
    """Regenerate `docs/source/models/supported-models.md` from in-code matrix."""
    root_dir = Path(__file__).parent.parent.parent.resolve()
    doc_path = root_dir / "docs/source/models/supported-models.md"

    # Use local render_supported_models_markdown() from this file
    content = render_supported_models_markdown()

    prev = ""
    if doc_path.exists():
        with open(doc_path, "r") as f:
            prev = f.read()
    if prev != content:
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        logging.warning(f"Writing {doc_path}")
        with open(doc_path, "w") as f:
            f.write(content)


def update_version():
    """Replace the placeholder container version in all docs source files."""
    version_path = (Path(__file__).parent.parent.parent / "tensorrt_llm" /
                    "version.py").resolve()
    spec = importlib.util.spec_from_file_location("version_module",
                                                  version_path)
    version_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(version_module)
    version = version_module.__version__

    docs_source_dir = Path(__file__).parent.resolve()
    md_files = list(docs_source_dir.rglob("*.md"))

    # Default is to replace `release:x.y.z` placeholders; set to 0 to disable.
    if os.environ.get("TRTLLM_DOCS_REPLACE_CONTAINER_TAG", "1") != "1":
        return

    for file_path in md_files:
        with open(file_path, "r") as f:
            content = f.read()
        updated = content.replace(
            "nvcr.io/nvidia/tensorrt-llm/release:x.y.z",
            f"nvcr.io/nvidia/tensorrt-llm/release:{version}",
        )
        if updated != content:
            with open(file_path, "w") as f:
                f.write(updated)


if __name__ == "__main__":
    import os
    path = os.environ["TEKIT_ROOT"] + "/examples/llm-api/llm_inference.py"
    #print(extract_meta_info(path))
    generate_examples()
