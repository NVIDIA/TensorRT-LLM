# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP_IDXK_PRODUCER_NVFP4: on an NVFP4 index-K layer (``indexer_kv_dtype="nvfp4"``) the built-in horizontal producer
(both main-KV kernels, FP8 and NVFP4) quantizes the index-K straight into the layer's NVFP4 index-K pages: each
element rounded to bf16, then the m31idxk writer's recipe, so the pages equal ``nvfp4_index_k.write`` on those bf16
rows. The E4M3 store into the cache manager's FP8 sink and the writer's pack from it (``write_from_sink``, a non-PDL
launch per layer) are skipped.

The index-only producer, and pages the producer cannot write in place, keep the sink and its pack; an FP8 index-K
layer is unchanged. ``TRTLLM_M31_DISABLE_DEP_IDXK_PRODUCER_NVFP4=1`` installs nothing and restores the sink + pack
on every layer. Read at import. The take is logged once per main-KV dtype at the first forward that takes it
(``[M3.1 DEP] IDXK_PRODUCER_NVFP4 taken: ...``).
"""

from __future__ import annotations

from tensorrt_llm._torch.models import modeling_minimaxm3 as _mm3

from .m31_switches import m31_disabled

DEP_IDXK_PRODUCER_NVFP4 = not m31_disabled("DEP_IDXK_PRODUCER_NVFP4")
OWNER = "M3.1 DEP"


def install() -> bool:
    """Switch the built-in horizontal producer to writing NVFP4 index-K pages itself; returns whether it now does.
    Process-wide, so call it before the first forward; installs nothing with
    TRTLLM_M31_DISABLE_DEP_IDXK_PRODUCER_NVFP4=1."""
    if not DEP_IDXK_PRODUCER_NVFP4:
        return False
    # The horizontal producer's NVFP4 index-K mode of the built-in model; a runtime without it installs nothing.
    set_mode = getattr(_mm3, "set_index_k_producer_nvfp4", None)
    get_mode = getattr(_mm3, "index_k_producer_nvfp4", None)
    if set_mode is None or get_mode is None:
        return False
    set_mode(True, owner=OWNER)
    return get_mode()


def status(installed: bool) -> str:
    """The item's state for the rank's engagement line."""
    if installed:
        return "on"
    if not DEP_IDXK_PRODUCER_NVFP4:
        return "off (TRTLLM_M31_DISABLE_DEP_IDXK_PRODUCER_NVFP4=1)"
    if not hasattr(_mm3, "set_index_k_producer_nvfp4"):
        return (
            "not installed (the built-in model has no NVFP4 index-K producer mode on this runtime)"
        )
    return "not installed"
