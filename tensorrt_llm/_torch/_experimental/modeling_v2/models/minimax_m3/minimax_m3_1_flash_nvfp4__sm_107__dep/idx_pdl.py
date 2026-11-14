# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP_IDX_PDL: the index-K decode reader starts while the horizontal producer still runs (exact).

On a pure-decode step whose NVFP4 index-K pages the horizontal producer writes itself (DEP_IDXK_PRODUCER_NVFP4), the
producer (minimaxM3Nvfp4QKVIndexerNormRopeKVInsertKernel) fires its PDL launch_dependents trigger right after its own
grid-dependency wait instead of at exit, so the Q8KV4 decode reader that follows it launches early; the reader then
L2-prefetches the first PREWAIT_PREFETCH_PAGES pages of each CTA's range before its griddepcontrol.wait. Bitwise the
same: the reader still reads every producer output (q, index-Q, the new index-K rows) after its wait, which waits for
the whole producer grid; the prefetches are non-binding hints. vLLM's reader also starts before its producer ends.

``TRTLLM_M31_DISABLE_DEP_IDX_PDL=1`` installs nothing (the trigger at exit, no prefetch). Read at import.
"""

from __future__ import annotations

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.msa.kernels import (
    nvfp4_index_k as _ik,
)
from tensorrt_llm._torch.models import modeling_minimaxm3 as _mm3

from .m31_switches import m31_disabled

DEP_IDX_PDL = not m31_disabled("DEP_IDX_PDL")
# Pages per reader CTA prefetched before the PDL wait.
PREWAIT_PREFETCH_PAGES = 6


def install() -> bool:
    """Switch on the producer's early trigger and the reader's pre-wait prefetch; returns whether they are on.
    Process-wide, so call it before the first forward; installs nothing with TRTLLM_M31_DISABLE_DEP_IDX_PDL=1."""
    if not DEP_IDX_PDL:
        return False
    # The horizontal producer's early-trigger switch of the built-in model; a runtime without it installs nothing.
    set_trigger = getattr(_mm3, "set_idx_pdl_early_trigger", None)
    get_trigger = getattr(_mm3, "idx_pdl_early_trigger", None)
    if set_trigger is None or get_trigger is None:
        return False
    set_trigger(True)
    _ik.set_decode_prewait_prefetch(PREWAIT_PREFETCH_PAGES)
    return get_trigger()


def status(installed: bool) -> str:
    """The item's state for the rank's engagement line."""
    if installed:
        return f"on (reader prefetch {_ik.decode_prewait_prefetch()} pages)"
    if not DEP_IDX_PDL:
        return "off (TRTLLM_M31_DISABLE_DEP_IDX_PDL=1)"
    if not hasattr(_mm3, "set_idx_pdl_early_trigger"):
        return "not installed (the built-in model has no producer early trigger on this runtime)"
    return "not installed"
