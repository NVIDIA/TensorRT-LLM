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
"""Feature support types for model registration (stdlib-only, no dependencies)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict


class SupportStatus(str, Enum):
    """Feature support status for a model architecture."""

    YES = "Yes"
    NO = "No"
    UNTESTED = "Untested"
    NA = "N/A"


@dataclass(frozen=True)
class FeatureSpec:
    """Specification for a model feature.

    Attributes:
        id: Short identifier used in registration (e.g., "chunked_prefill")
        description: Human-readable name for documentation (e.g., "Chunked Prefill")
        has_llmapi_knob: True if this feature maps to an LLM API parameter
    """

    id: str
    description: str
    has_llmapi_knob: bool


# Define all features with their metadata
# Features with has_llmapi_knob=True map to LLM API parameters that can be auto-disabled
OVERLAP_SCHEDULER = FeatureSpec("overlap_scheduler", "Overlap Scheduler", has_llmapi_knob=True)
CUDA_GRAPH = FeatureSpec("cuda_graph", "CUDA Graph", has_llmapi_knob=True)
ATTENTION_DP = FeatureSpec("attention_dp", "Attention Data Parallelism", has_llmapi_knob=True)
DISAGGREGATED_SERVING = FeatureSpec(
    "disaggregated_serving", "Disaggregated Serving", has_llmapi_knob=False
)
CHUNKED_PREFILL = FeatureSpec("chunked_prefill", "Chunked Prefill", has_llmapi_knob=True)
MTP = FeatureSpec("mtp", "MTP", has_llmapi_knob=False)
EAGLE3_ONE_MODEL = FeatureSpec(
    "eagle3_one_model", "EAGLE-3(One Model Engine)", has_llmapi_knob=False
)
EAGLE3_TWO_MODEL = FeatureSpec(
    "eagle3_two_model", "EAGLE-3(Two Model Engine)", has_llmapi_knob=False
)
TORCH_SAMPLER = FeatureSpec("torch_sampler", "Torch Sampler", has_llmapi_knob=False)
TLLM_CPP_SAMPLER = FeatureSpec("tllm_cpp_sampler", "TLLM C++ Sampler", has_llmapi_knob=False)
KV_CACHE_REUSE = FeatureSpec("kv_cache_reuse", "KV Cache Reuse", has_llmapi_knob=True)
SLIDING_WINDOW = FeatureSpec("sliding_window", "Sliding Window Attention", has_llmapi_knob=False)
LOGITS_POST_PROCESSOR = FeatureSpec(
    "logits_post_processor", "Logits Post Processor", has_llmapi_knob=False
)
GUIDED_DECODING = FeatureSpec("guided_decoding", "Guided Decoding", has_llmapi_knob=True)
EPD_DISAGG_SERVING = FeatureSpec(
    "epd_disagg_serving", "EPD Disaggregated Serving", has_llmapi_knob=False
)
# Modality features (split from single MODALITY text field)
MODALITY_LANGUAGE = FeatureSpec("modality_language", "Modality: Language", has_llmapi_knob=False)
MODALITY_IMAGE = FeatureSpec("modality_image", "Modality: Image", has_llmapi_knob=False)
MODALITY_VIDEO = FeatureSpec("modality_video", "Modality: Video", has_llmapi_knob=False)
MODALITY_AUDIO = FeatureSpec("modality_audio", "Modality: Audio", has_llmapi_knob=False)

# Lookup by string id
ALL_FEATURES: Dict[str, FeatureSpec] = {
    f.id: f
    for f in [
        OVERLAP_SCHEDULER,
        CUDA_GRAPH,
        ATTENTION_DP,
        DISAGGREGATED_SERVING,
        CHUNKED_PREFILL,
        MTP,
        EAGLE3_ONE_MODEL,
        EAGLE3_TWO_MODEL,
        TORCH_SAMPLER,
        TLLM_CPP_SAMPLER,
        KV_CACHE_REUSE,
        SLIDING_WINDOW,
        LOGITS_POST_PROCESSOR,
        GUIDED_DECODING,
        EPD_DISAGG_SERVING,
        MODALITY_LANGUAGE,
        MODALITY_IMAGE,
        MODALITY_VIDEO,
        MODALITY_AUDIO,
    ]
}


def get_feature(feature_id: str) -> FeatureSpec:
    """Look up a FeatureSpec by its string id.

    Args:
        feature_id: The feature identifier (e.g., "chunked_prefill")

    Returns:
        The corresponding FeatureSpec

    Raises:
        KeyError: If feature_id is not found
    """
    return ALL_FEATURES[feature_id]


class Feature(str, Enum):
    """Features that can be enabled/disabled for model architectures.

    The enum value is the feature ID string used for registration.
    Use get_feature_spec() to get the full FeatureSpec with metadata.
    """

    OVERLAP_SCHEDULER = "overlap_scheduler"
    CUDA_GRAPH = "cuda_graph"
    ATTENTION_DP = "attention_dp"
    DISAGGREGATED_SERVING = "disaggregated_serving"
    CHUNKED_PREFILL = "chunked_prefill"
    MTP = "mtp"
    EAGLE3_ONE_MODEL_ENGINE = "eagle3_one_model"
    EAGLE3_TWO_MODEL_ENGINE = "eagle3_two_model"
    TORCH_SAMPLER = "torch_sampler"
    TLLM_CPP_SAMPLER = "tllm_cpp_sampler"
    KV_CACHE_REUSE = "kv_cache_reuse"
    SLIDING_WINDOW_ATTENTION = "sliding_window"
    LOGITS_POST_PROCESSOR = "logits_post_processor"
    GUIDED_DECODING = "guided_decoding"
    EPD_DISAGG_SERVING = "epd_disagg_serving"
    # Modality features
    MODALITY_LANGUAGE = "modality_language"
    MODALITY_IMAGE = "modality_image"
    MODALITY_VIDEO = "modality_video"
    MODALITY_AUDIO = "modality_audio"
    # Legacy modality field for doc rendering (text like "L + I + V")
    MODALITY = "modality"

    def get_spec(self) -> FeatureSpec:
        """Get the FeatureSpec for this feature."""
        return ALL_FEATURES[self.value]

    @property
    def description(self) -> str:
        """Human-readable description for documentation."""
        spec = ALL_FEATURES.get(self.value)
        if spec:
            return spec.description
        # For MODALITY (legacy), return a reasonable default
        return "Modality"

    @property
    def has_llmapi_knob(self) -> bool:
        """Whether this feature maps to an LLM API parameter."""
        spec = ALL_FEATURES.get(self.value)
        return spec.has_llmapi_knob if spec else False
