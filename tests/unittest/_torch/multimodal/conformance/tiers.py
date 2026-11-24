# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multimodal onboarding requirements, in the order used by the onboarding guide."""

from tensorrt_llm._torch.models.multimodal import MultimodalFeature as F

FEATURE_TIERS = {
    "contract": (
        "P0",
        "Expose the decoder and local towers. Preserve prompt order and each item's embedding "
        "count. Supply partitions when the input builder needs them.",
    ),
    "text_only": (
        "P0",
        "With the encoder disabled, skip tower construction and loading. V2 admission in #22450 "
        "checks the local tower, split-worker mode and request content.",
    ),
    F.UINT8_MEDIA: (
        "P0",
        "Honor loader defaults and processor normalization. #22334 supplies RGB PIL images and "
        "RGB uint8 NumPy video frames.",
    ),
    F.ENCODER_CACHE: (
        "P0",
        "Return complete item embeddings with exact counts and stable keys. Enable reuse with a "
        "positive cache budget and, in #22450, a local encoder.",
    ),
    F.CHUNKED_PREFILL_MEDIA: (
        "P0",
        "Count tokens per supported modality and fuse only the embeddings in the current chunk. "
        "Handle model-specific slicing when encoding changes lengths.",
    ),
    F.KV_REUSE_MEDIA: (
        "P0",
        "Supply token counts and placeholder geometry so the runtime can key media spans by "
        "content hash or UUID.",
    ),
    F.LLM_BREAKABLE_CUDA_GRAPH: (
        "P0",
        "Declare media replay on the wrapper. The engine checks the decoder, pipeline size and "
        "layer registry. Enforce after #21333.",
    ),
    F.LLM_PIECEWISE_CUDA_GRAPH: (
        "P0",
        "Use native decoder compilation or compile only the wrapped decoder. Enforce after #21333.",
    ),
    F.ENCODER_DP: (
        "P0",
        "Declare supports_mm_encoder_data_parallel and build replicas from the encoder config. "
        "Enforce after #17330.",
    ),
    F.ITEM_SCHEDULING: (
        "P1",
        "Declare item scheduling and supply costs, profiling inputs and budgets. Override "
        "attention capacity when geometry requires it. Moves to P0 when costs can be derived.",
    ),
    "supports_vision_gpu_normalize": (
        "P1",
        "Skip CPU rescaling and normalization on the GPU path. Preserve the processor "
        "statistics, input dtype and layout.",
    ),
    "encoder_attention_backend": (
        "P1",
        "Declare selectable backends per tower, or an empty set for fixed attention. Size "
        "metadata at setup and refresh it on each forward.",
    ),
    F.DISAGG_P_TO_D: (
        "P1",
        "Transfer positions and deltas for mRoPE models. Preserve media identity and spans when "
        "reusing KV across workers.",
    ),
    "mm_encoder_groups": (
        "P2",
        "Batch compatible items across requests. Moves to P1 when #21052 supplies the default "
        "partition-based builder.",
    ),
    F.ENCODER_CUDA_GRAPH: (
        "P2",
        "Capture tower blocks and check model accuracy. Moves to P1 when a default block "
        "adapter is available.",
    ),
    F.ENCODER_TP: ("P2", "Shard tower weights across the ranks of the encoder mapping."),
    F.DISAGG_E_TO_P: (
        "P2",
        "Run the standalone encoder and restore its embeddings, item layout and required "
        "position metadata on the prefill worker.",
    ),
    "image_routing": (
        "P2",
        "Validate image KV events and routing for the advertised OpenEngine configuration.",
    ),
}

# Deferred P0 enforcement for new models. CI still checks declarations and explicit opt-outs.
STRUCTURAL_ONLY = {
    F.LLM_BREAKABLE_CUDA_GRAPH: "The multimodal breakable graph path lands in #21333.",
    F.LLM_PIECEWISE_CUDA_GRAPH: "Enforce decoder-scoped compilation with breakable graphs after #21333.",
    F.ENCODER_DP: "Encoder data parallelism lands in #17330.",
    "base": "Require the shared V2 composite after #22450.",
    "fixture": "Require per-target behavior fixtures when the tiny reference model is available.",
}

# No class flag shows these on main, so a class is eligible unless it opts out. Their
# snapshot entries come from the reviewed support matrix and are never reported as stale;
# the check named here deletes them when it lands.
NOT_DERIVABLE = {
    F.CHUNKED_PREFILL_MEDIA: "The one-GPU chunked prefill check.",
    F.KV_REUSE_MEDIA: "The one-GPU KV reuse check.",
    F.DISAGG_P_TO_D: "The decode state export and restore check.",
    F.UINT8_MEDIA: "The uint8 versus float input check, with the uint8 loader default.",
}
