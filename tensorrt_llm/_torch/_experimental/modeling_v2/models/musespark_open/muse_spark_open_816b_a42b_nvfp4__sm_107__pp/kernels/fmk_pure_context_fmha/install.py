# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``fmk_pure_context_fmha`` on a stage: pure context steps' full-attention FMHA on the
multi-instance paged context FMHA, at the stage's 128 query / 16 KV heads.

The TEP target's library (``kernels/fmk_context_fmha/attention.FmkPureContextFmha``) goes just
before the thop fallback of every full-attention layer the stage owns; the stripped layers of
the other stages are never touched. It runs the same three steps as on a TEP4 rank: the thop
context preprocess (Q to E4M3, K / V appended to the paged cache), the host plan of the
step's lengths, and the CuTe DSL kernel. The kernel is compiled per (query, KV) head pair
and needs the query heads of one KV head to be a multiple of its 4 instances per CTA; 128 /
16 has the 8 query heads per KV head of a TEP4 rank's 32 / 4, so the same build serves it.
The plan is made at the stage's head counts (``stage_library.StagePlans``): 32 head groups of
4 per 128-row block, four times a TEP4 rank's work units, over the same per-CTA workspace
slots.

Numerics are the TEP library's: the same E4M3 Q / K / V and P, fp32 accumulation and BF16
output as the thop op's cubin, in another summation order and P rounding point (numeric, not
bitwise); the K / V appended to the cache are the preprocess's, bitwise the thop op's. A call
the library declines runs the thop op as before.

Construction-time, before any CUDA graph capture: the workspace is reserved and the kernel
compiled and warmed up at the stage's head counts (``stage_library.prepare``). Agreed over the
stages (``dispatch.agreed_install``): every stage checks its layers and builds the workspace,
the outcome is agreed, and only then are the libraries inserted.
"""

from __future__ import annotations

from typing import Optional, Tuple

from tensorrt_llm.logger import logger

from ... import dispatch as _dispatch
from . import stage_library as _stage


def install(text_model, mapping, compute_capability: Optional[Tuple[int, int]]) -> int:
    """Put the multi-instance context FMHA in front of the thop fallback of every full-attention
    layer of the stage; returns the number of layers changed.

    Collective over the stages (the outcome is agreed before any layer changes). Raises
    ``ValueError`` on every stage alike when some stage refuses (not sm107, a layer the kernel
    does not serve, head counts other than ``dispatch.FMK_STAGE_HEADS``, a stage without a
    full-attention layer, or a failing compile / warm-up).
    """
    layers = _dispatch.owned_layers(text_model)

    def build():
        if not any(not layer.self_attn.is_sliding for layer in layers):
            raise ValueError(
                f"{_dispatch.TAG} fmk_pure_context_fmha: stage {text_model._pp_stage} has no "
                "full-attention layer"
            )
        try:
            return _stage.prepare(layers, compute_capability)
        except ValueError:
            raise
        except Exception as error:  # noqa: BLE001 - the warm-up's failure must reach the agreement
            raise RuntimeError(
                f"{_dispatch.TAG} fmk_pure_context_fmha could not build its workspace or warm "
                f"up: {error!r}"
            ) from error

    prepared = _dispatch.agreed_install(mapping, build, (ValueError, RuntimeError))
    changed = prepared.commit()
    logger.info(
        f"{_dispatch.TAG} fmk_pure_context_fmha: pure context steps of stage "
        f"{text_model._pp_stage} run the multi-instance context FMHA on its {changed} "
        f"full-attention layers at {_dispatch.FMK_STAGE_HEADS[0]} / "
        f"{_dispatch.FMK_STAGE_HEADS[1]} heads ({prepared.num_ctas} CTAs, workspace "
        f"{prepared.workspace.nbytes / 2**20:.1f} MiB; {_stage._fmk.build_note()})"
    )
    return changed
