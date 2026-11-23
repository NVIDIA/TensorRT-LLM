# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The one place this target reaches into the musespark_open TEP target, and what it reaches.

A pipeline stage runs the TEP target's context kernels at the full (unsliced) shapes: the
same pre-attention prep op, the same MXFP8 projection copy, the same Rubin CuTe DSL prefill
expert op, the same join and finalize kernels of the sequence-parallel MoE (without its
collectives), and the same lever resolver and knob grammar (``MUSESPARK_OPEN_LEVER_*``). They
are imported from the TEP target rather than copied, so a fix there reaches both. Nothing
here changes the TEP target: its installers keep their own gates (tp 4 / tp 8), and this
target installs the ops through its own installers (``kernels/``).

Every other module of this target imports the TEP target through ``module`` only, and only
the modules named in ``MODULES``. ``levers`` is imported when this module is (the resolver
is pure: no torch, no device); every other module where it is installed.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

#: The TEP target's package.
TEP_PACKAGE = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open."
    "muse_spark_open_816b_a42b_nvfp4__sm_107__tep"
)

#: The TEP modules this target reads, relative to ``TEP_PACKAGE``, and what for.
MODULES = {
    "levers": "the lever resolver, its rules and the knob grammar",
    "dispatch": "the names of every TEP fast path, which the knob grammar knows",
    "attention": "the attention class whose prep a context step swaps for one op",
    "ops.attention.preattn_prep": "the pre-attention prep op (context_prep)",
    "kernels.context_fp8_dense.install": "the MXFP8 projection copy and its quant method",
    "ops.moe.prefill_cutedsl_experts": "the Rubin CuTe DSL prefill expert op",
    "kernels.prefill_cutedsl_experts.install": "the expert-forward wrapper of that op",
    "kernels.replicated_experts.ring_buddy": "the expert leaf predicates (format, SwiGLU)",
    "kernels.fmk_context_fmha.attention": "the multi-instance context FMHA library of pure "
    "context steps and its build constants (fmk_pure_context_fmha)",
    "ops.attention.context_fmha_fmk": "the multi-instance context FMHA's planner, workspace and "
    "kernel, which a stage plans and warms up at its own heads (fmk_pure_context_fmha)",
    "kernels.context_sequence_parallel.router_input": "the attention-side join that also "
    "writes the router's FP32 input (context_router_input)",
    "kernels.context_sequence_parallel.moe": "the BF16-storing finalize and the FP32 router "
    "input predicate of the sequence-parallel MoE (context_finalize_bf16, context_router_input)",
}


def module(name: str) -> ModuleType:
    """The TEP target's module ``name`` (a key of ``MODULES``).

    Raises:
        KeyError: for a module this target does not declare it reads.
    """
    if name not in MODULES:
        raise KeyError(f"the pp target does not read the TEP module {name!r}")
    return import_module(f"{TEP_PACKAGE}.{name}")


levers = module("levers")
