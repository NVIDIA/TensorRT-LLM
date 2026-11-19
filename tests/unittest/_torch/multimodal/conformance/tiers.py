# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multimodal onboarding requirements, in the order used by the onboarding guide."""

from tensorrt_llm._torch.models.multimodal import MultimodalFeature as F

FEATURE_TIERS = {
    "contract": (
        "P0",
        "Expose the language model and tower. Supply item order, row counts and input partitions.",
    ),
    "text_only": (
        "P0",
        "With `disable_mm_encoder`, omit the tower and refuse raw media and attached rows at admission.",
    ),
    F.ITEM_SCHEDULING: (
        "P0",
        "Encode whole items. Supply exact per-item costs, profiling inputs and encoder budgets.",
    ),
    F.ENCODER_CACHE: (
        "P0",
        "Return whole-item encoder rows with exact row counts and stable media hashes.",
    ),
    F.CHUNKED_PREFILL_MEDIA: ("P0", "Fuse only the media rows in the current prefill chunk."),
    F.KV_REUSE_MEDIA: ("P0", "Supply stable media hashes and token counts for prefix reuse."),
    F.LLM_BREAKABLE_CUDA_GRAPH: (
        "P0",
        "Preserve decoder graph support for media. Use a `DecoderModelForCausalLM`, keep encoder "
        "attention out of its layer registry and prepare extra graph inputs.",
    ),
    F.LLM_PIECEWISE_CUDA_GRAPH: (
        "P0",
        "Preserve the decoder's piecewise graph support. Scope compilation to the language model.",
    ),
    F.UINT8_MEDIA: ("P0", "Accept the loader's uint8 media and apply processor normalization."),
    F.DISAGG_P_TO_D: ("P0", "Preserve media hashes and export and restore mRoPE state when used."),
    F.ENCODER_DP: (
        "P0",
        "Build the tower from the encoder config so replicas can encode whole items.",
    ),
    "encoder_attention_backend": (
        "P0",
        "Declare supported attention backends per tower in the model class body.",
    ),
    F.ENCODER_TP: ("P1", "Shard tower weights under the encoder mapping."),
    F.ENCODER_CUDA_GRAPH: ("P1", "Provide a graph capture adapter for the tower blocks."),
    F.DISAGG_E_TO_P: (
        "P1",
        "Accept attached rows with a complete item layout. Becomes P0 when the generic handoff lands.",
    ),
}

UNTIERED = {
    "supports_vision_gpu_normalize": "Requires processor statistics and a compatible tensor dtype and layout.",
    "openengine_image_routing": "Requires separate qualification for image KV events and routing.",
}

# New models are not held to these yet, because what they check is not on main. A model
# that opts out of one of them in its class still needs an exception for its tier.
STRUCTURAL_ONLY = {
    F.LLM_BREAKABLE_CUDA_GRAPH: "The executor has no multimodal breakable graph path yet.",
    F.LLM_PIECEWISE_CUDA_GRAPH: "Only some wrappers scope compilation to the language model.",
    F.ENCODER_DP: "The encoder has no separate data parallel mapping yet.",
    F.ENCODER_TP: "The encoder has no separate tensor parallel mapping yet.",
    "base": "The shared multimodal base class is not available yet.",
    "fixture": "The tiny reference model and its fixtures are not available yet.",
}

# No class flag shows these on main, so a class is eligible unless it opts out. Their
# snapshot entries come from the reviewed support matrix and are never reported as stale;
# the check named here deletes them when it lands.
NOT_DERIVABLE = {
    F.ENCODER_CUDA_GRAPH: "The one-GPU encoder graph check.",
    F.CHUNKED_PREFILL_MEDIA: "The one-GPU chunked prefill check.",
    F.KV_REUSE_MEDIA: "The one-GPU KV reuse check.",
    F.DISAGG_P_TO_D: "The decode state export and restore check.",
    F.UINT8_MEDIA: "The uint8 versus float input check, with the uint8 loader default.",
}
