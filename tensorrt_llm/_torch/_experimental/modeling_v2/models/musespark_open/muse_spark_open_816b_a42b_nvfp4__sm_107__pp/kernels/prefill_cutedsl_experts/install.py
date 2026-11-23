# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``prefill_cutedsl_experts`` on a stage: the large expert calls on the Rubin CuTe DSL GEMMs.

The TEP target installs its op at sm107 tp 4 only (its ``serves``), on each rank's
64-expert window of a ring (``ring_buddy.eligible``). A stage at tp 1 / ep 1 holds every
expert of its layers, so this installer serves that layout instead: every NVFP4 MoE layer of
the stage whose leaf is the native TRTLLM-gen one over all ``EXPERTS`` experts (slot 0..255,
no external communication) gets the TEP target's expert-forward wrapper around the TEP
target's op built over the whole leaf (``PrefillCuteDslExperts(backend, 256, 8, 0)``). An
unfinalized call of ``MIN_ROWS..MAX_ROWS`` NVFP4 rows -- the context steps of at least 4096
tokens -- runs the op and returns the triple the TRTLLM-gen call returns; every other call
keeps the expert forward it had. Layers 0 and 61 keep the built-in leaf when it is the BF16
one, and take the op when the checkpoint's BF16 experts were quantized to NVFP4 at load
(``MUSESPARK_OPEN_L061_EXPERTS_NVFP4=1``).

The op takes any local expert count; a one-GPU sweep ran it unmodified at 256 experts and
offset 0, faster than four 64-expert windows. Rounding-different from the TRTLLM-gen call
(as on the TEP target); expert selection is unchanged.

Install after the weights are final and before any CUDA graph capture (the kernels compile
here). Agreed over the stages (``dispatch.agreed_install``): every stage checks its layers and
compiles the op, the outcome is agreed, and only then are the expert forwards wrapped.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from tensorrt_llm.logger import logger

from ... import dispatch as _dispatch
from ..._tep import module as _tep_module

#: The SM version this path is installed at: Rubin (sm107).
SERVED_SM = 107
#: The experts of every MoE layer and the experts each token routes to.
EXPERTS = 256
TOP_K = 8


def serves(compute_capability: Optional[Tuple[int, int]], mapping) -> bool:
    """Whether a stage on ``compute_capability`` with ``mapping`` installs this path."""
    return (
        _dispatch.sm_of(compute_capability) == SERVED_SM
        and int(mapping.tp_size) == 1
        and int(mapping.moe_ep_size) == 1
        and int(mapping.moe_tp_size) == 1
    )


def refusal(moe, mapping) -> Optional[str]:
    """Why this MoE layer cannot take the op over all of its experts, or None if it can."""
    ring_buddy = _tep_module("kernels.replicated_experts.ring_buddy")
    if int(mapping.tp_size) != 1 or int(mapping.moe_ep_size) != 1 or int(mapping.moe_tp_size) != 1:
        return "the stage is not tp 1 / ep 1"
    if getattr(moe.experts, "comm", None) is not None:
        return "the layer has an external MoE communication strategy"
    if ring_buddy.expert_format(moe) != "nvfp4":
        return "the experts are not the native NVFP4 TRTLLM-gen leaf"
    backend = ring_buddy._backend(moe)
    if (
        getattr(backend, "num_slots", None) != EXPERTS
        or getattr(backend, "expert_size_per_partition", None) != EXPERTS
        or getattr(backend, "slot_start", None) != 0
    ):
        return f"the leaf does not hold all {EXPERTS} experts from slot 0"
    if getattr(backend, "bias", False):
        return "the experts carry biases"
    for name in ("act_alpha", "act_beta", "act_clamp", "fc31_act_scale"):
        if getattr(backend, name, None) is not None:
            return f"the experts carry {name}"
    if not ring_buddy._is_plain_swiglu(backend):
        return "the experts are not plain SwiGLU"
    if getattr(backend, "fc2_input_scale", None) is None:
        return "the leaf has no FC2 input scale"
    return None


def install(text_model, mapping, compute_capability: Optional[Tuple[int, int]]) -> List[int]:
    """Wrap every eligible NVFP4 layer of the stage; returns the layer indices.

    Collective over the stages (the outcome is agreed before any layer is wrapped). Raises
    ValueError on every stage alike when some stage refuses (not sm107 tp 1, a stage without
    an eligible layer, or a failing compile).
    """

    def build():
        if not serves(compute_capability, mapping):
            raise ValueError(
                f"{_dispatch.TAG} prefill_cutedsl_experts: written for sm{SERVED_SM} at tp 1 / "
                f"ep 1, "
                f"not {compute_capability} at tp {mapping.tp_size} / ep {mapping.moe_ep_size}"
            )
        ring_buddy = _tep_module("kernels.replicated_experts.ring_buddy")
        op_module = _tep_module("ops.moe.prefill_cutedsl_experts")
        wrapper = _tep_module("kernels.prefill_cutedsl_experts.install")
        installed, skipped, ops = [], {}, []
        for layer in _dispatch.owned_layers(text_model):
            moe = layer.feed_forward
            reason = refusal(moe, mapping)
            if reason is not None:
                skipped[layer.layer_idx] = reason
                continue
            backend = ring_buddy._backend(moe)
            ops.append((moe, op_module.PrefillCuteDslExperts(backend, EXPERTS, TOP_K, 0)))
            installed.append(layer.layer_idx)
        if not installed:
            raise ValueError(
                f"{_dispatch.TAG} prefill_cutedsl_experts found no eligible NVFP4 layer on stage "
                f"{text_model._pp_stage}: {sorted(set(skipped.values()))}"
            )
        try:
            # One compile serves every layer (the kernels' shapes are runtime arguments).
            ops[0][1].prepare(wrapper.MAX_ROWS)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception as error:  # noqa: BLE001 - the compile's failure must reach the agreement
            raise RuntimeError(
                f"{_dispatch.TAG} prefill_cutedsl_experts could not compile: {error!r}"
            ) from error
        return wrapper, installed, skipped, ops

    wrapper, installed, skipped, ops = _dispatch.agreed_install(
        mapping, build, (ValueError, RuntimeError)
    )
    for moe, op in ops:
        wrapper._wrap(moe, op)
        moe._prefill_cutedsl_experts = op
    logger.info(
        f"{_dispatch.TAG} prefill_cutedsl_experts: {len(installed)} NVFP4 MoE layers of stage "
        f"{text_model._pp_stage} run their {wrapper.MIN_ROWS}..{wrapper.MAX_ROWS}-row expert "
        f"calls over all {EXPERTS} experts on the Rubin CuTe DSL GEMMs; built-in expert call "
        f"kept for layers {sorted(skipped)}: {sorted(set(skipped.values()))}"
    )
    return installed
