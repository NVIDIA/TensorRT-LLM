# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Image input on the DeepSeek-V4.1 targets, which serve text only.

No target builds the image encoder. A checkpoint with a vision tower served with the encoder on is declined by
`route` (the built-in model serves it); served with `disable_mm_encoder`, it reaches a target, which then refuses a
context request carrying image input instead of answering it from placeholder embeddings of the image tokens. A
generation-only request of an image prompt is admitted: its prompt ran on the context server, and the image tokens in
its look-back break the Engram n-grams as they do there (each target's `_EngramHash` hashes the image token as the
pad id).
"""

from __future__ import annotations

_NO_ENCODER = (
    "this DeepSeek-V4.1 modeling_v2 target serves text only: no image encoder is built (disable_mm_encoder), so the "
    "image input would be embedded as placeholder tokens. Serve images with disable_mm_encoder off (modeling_v2 then "
    "uses the built-in model)."
)


def refuse_image_request(request) -> None:
    """A target's admission check (`validate_request`): a context request whose prompt still has to be encoded (image
    pixels or precomputed image embeddings) is refused, as a
    per-request error. A generation-only request is admitted."""
    if request.is_generation_only_request:
        return
    data = getattr(request, "py_multimodal_data", None) or {}
    if data.get("image") is not None or data.get("multimodal_embedding") is not None:
        raise ValueError(_NO_ENCODER)


def refuse_image_step(multimodal_params) -> None:
    """The forward-time form of `refuse_image_request`, for a step that carries image input anyway."""
    if multimodal_params and any(p is not None and p.has_content() for p in multimodal_params):
        raise ValueError(_NO_ENCODER)
