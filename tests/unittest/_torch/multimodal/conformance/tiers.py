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
        "With `disable_mm_encoder`, omit the tower and refuse raw media and attached rows at "
        "admission.",
    ),
    F.UINT8_MEDIA: (
        "P0",
        "Accept the loader's default uint8 arrays or PIL images and honor processor normalization.",
    ),
    F.ENCODER_CACHE: (
        "P0",
        "Return whole-item encoder rows with exact row counts and stable media hashes.",
    ),
    F.CHUNKED_PREFILL_MEDIA: ("P0", "Fuse only the media rows in the current prefill chunk."),
    F.KV_REUSE_MEDIA: ("P0", "Supply stable media hashes and token counts for prefix reuse."),
    F.LLM_BREAKABLE_CUDA_GRAPH: (
        "P0",
        "Keep encoder attention out of the decoder layer registry and prepare extra graph "
        "inputs. The check is structural until #21333 lands.",
    ),
    F.LLM_PIECEWISE_CUDA_GRAPH: (
        "P0",
        "Scope piecewise prefill compilation to the decoder. The check is structural until "
        "#21333 lands.",
    ),
    F.ENCODER_DP: (
        "P0",
        "Build the tower from the encoder config so replicas can encode whole items. The check "
        "is structural until #17330 lands.",
    ),
    F.ITEM_SCHEDULING: (
        "P1",
        "Supply exact item costs, profiling inputs and encoder budgets. Moves to P0 when the "
        "runtime derives costs from item metadata.",
    ),
    "supports_vision_gpu_normalize": (
        "P1",
        "Provide processor statistics and a compatible tensor dtype and layout for normalization.",
    ),
    "encoder_attention_backend": (
        "P1",
        "Declare each tower's backends in the model class body. Let the engine size its "
        "attention metadata.",
    ),
    F.DISAGG_P_TO_D: (
        "P1",
        "Preserve media identity and export and restore mRoPE state across workers when used.",
    ),
    "mm_encoder_groups": (
        "P2",
        "Batch compatible items across requests. Moves to P1 when #21052 supplies the default "
        "builder.",
    ),
    F.ENCODER_CUDA_GRAPH: (
        "P2",
        "Provide a graph capture adapter for the tower blocks. Moves to P1 when the default "
        "block adapter lands.",
    ),
    F.ENCODER_TP: ("P2", "Shard tower weights under the encoder mapping."),
    F.DISAGG_E_TO_P: ("P2", "Accept attached rows with a complete item layout."),
    "image_routing": ("P2", "Qualify image KV events and routing before opting in."),
}

# New models can omit these hooks until their runtime support lands. Explicit P0 opt-outs
# still need approval.
STRUCTURAL_ONLY = {
    F.LLM_BREAKABLE_CUDA_GRAPH: "The multimodal breakable graph path lands in #21333.",
    F.LLM_PIECEWISE_CUDA_GRAPH: "The multimodal piecewise graph path lands in #21333.",
    F.ENCODER_DP: "Encoder data parallelism lands in #17330.",
    "base": "The shared multimodal base class is not available yet.",
    "fixture": "The tiny reference model and its fixtures are not available yet.",
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
