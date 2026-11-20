# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Switches of the MiniMax-M3.1 DEP generation target (each target keeps its own copy: targets share only the
catalog). The MegaMoE local arm, whose MoE identity the target then requires, is read with the MegaMoE backend's own
rule (``engagement.megamoe_local_requested``), not with these switches.

Every item is on by default; ``TRTLLM_M31_DISABLE_<ID>=1`` turns item ``<ID>`` off and restores the path it replaced
(``SWITCHES`` lists each id and what it restores). The environment is read once, at import, so every rank and every
step of a process sees the same setting (a step must not change its graph mid-run).

Values (case-insensitive, surrounding blanks ignored): ``1`` / ``true`` / ``on`` / ``yes`` set a switch; empty / ``0`` /
``false`` / ``off`` / ``no`` leave it unset; anything else raises at import, naming the variable, so a typo fails loudly
instead of silently turning an item on or off. ``m31_disabled`` raises on an id that is not in ``SWITCHES``; a set
variable whose id is not in ``SWITCHES`` is reported once at import and ignored.
"""

import os
from typing import Dict, FrozenSet

from tensorrt_llm.logger import logger

_TARGET = "DEP"
_PREFIX = "TRTLLM_M31_DISABLE_"
_SET = frozenset(("1", "true", "on", "yes"))
_UNSET = frozenset(("", "0", "false", "off", "no"))

#: The target's items: id -> what ``TRTLLM_M31_DISABLE_<id>=1`` restores.
SWITCHES: Dict[str, str] = {
    # Step-loop hooks (modeling.dep_runtime_hooks).
    "DEP_HOOKS": "the built-in model's step-loop hooks (no target hooks at all)",
    "DEP_STEP_HOOKS": "the engine's step loop: no step-loop group (ordering, timing and input path)",
    "DEP_STEP_INPUTS": "the engine's per-copy step inputs (no packed input path)",
    "DEP_DEFER_REUSE_COMMIT": "the KV-reuse commit of a finished receive before the step's forward",
    "HOSTTAIL": "the engine's host tail: no speculative input packing or side-stream sampler copy",
    "SKIP_ACCEPTED_UPLOAD": "the upload of the accepted-draft counts",
    "BATCH_CHANGED_ONLY": "the drafter batch-map copy on every step",
    "HOST_D2H_SIDE": "the sampler's host copy on the execution stream",
    "SMALL_H2D": "the speculative sampler's small host-to-device copies one by one",
    "PHILOX_PACK": "the Philox seed / offset uploads outside the input pack",
    "KV_LENS_ONCE": "the MSA KV lengths derived at every use",
    "PACK_PSCATTER": "the serial scatter of the step-input pack",
    "PACK_IN_GRAPH": "the step-input scatter outside the decode graph",
    "PACK_EARLY_COPY": "the step-input upload on the execution stream",
    "H2D_CHANGED_SPAN": "the whole packed table staged on a change",
    "BLOCKOFF_GUARD": "the drafter block-offset copy on every step",
    "BLOCKOFF_PATCH": "the whole drafter block-offset copy on a change (no host patch)",
    "DEP_STEP_NATIVE": "the engine's own step-glue lines (no native step launches)",
    "DEP_STEP_PROLOGUE": "the decode graph's step-input prologue as separate launches",
    "DEP_SPEC_OFFSETS": "the speculative step correction as separate launches",
    "DEP_MSA_KV_UPDATE": "the MSA KV-length correction as separate launches",
    "STEP_GATHER_FUSED": "the engine's previous-batch input gathers as separate launches",
    "DEP_MSA_POLICY": "the stock MSA metadata path (block-table copies, pool views, block-id tables)",
    "DEP_KERNEL_HOOKS": "the kernel hooks: the engine's MXFP8 GEMM tactics and the full sampler tail",
    "TAIL_BOOKKEEP": "the sampler tail's bookkeeping launches",
    "DEP_FIX_HOOKS": "the MSA cache and drafter pool without the target's fixes",
    "DEP_TACTIC_PIN": "each attention-DP rank's own tuned GEMM tactic (no cross-rank merge)",
    "DEP_TACTIC_BUNDLE": "a tuning run without the FlashInfer MXFP8 tactics in the autotuner cache",
    "DEP_DISAGG_DRAFT_KV": "an empty drafter window at the start of a disaggregated request",
    # Model items (modeling and the item modules).
    "DEP_DSPARK_GLUE": "the in-tree DSpark drafter glue (ops/dspark/dspark_glue_install installs nothing)",
    "DEP_GEMM_PROVIDER": "the small-M decode MXFP8 GEMMs on FlashInfer's pinned tactic",
    "DEP_GEMM_CUBLASLT": "the small-M decode MXFP8 GEMMs on split-K 2 / the pinned tactic instead of cuBLASLt",
    "DEP_GUMBEL_DRAFT": "the in-tree DSpark ancestral block sampler",
    "DEP_IDX_PDL": "the index-K producer's trigger at exit and no reader prefetch before its wait",
    "DEP_IDXK_PRODUCER_NVFP4": "the FP8 index-K sink and its NVFP4 pack on NVFP4 index-K layers",
    "DEP_LC2": "the in-tree Triton sparse decode attention",
    "DEP_LC2_PAD_SKIP": "LC2 reading KV for CUDA-graph padding rows",
    "DEP_LC2_PDL": "the LC2 split launches without PDL",
    "DEP_LOGITS_WIDEN": "the stock strided widening copy of the logits",
    "DEP_MARKOV_GLUE": "the in-tree DSpark ancestral draft glue",
    "DEP_MERGE_MXFP8": "o_proj quantizing the LC2 output itself",
    "DEP_DRAFTER_MXFP8": "the drafter's projections as BF16 GEMMs on the dequantized weights",
    "DEP_NORM_MXFP8": "qkv_proj quantizing the boundary norm's output itself",
    "DEP_OWNT": "LC2 on every call it takes (no single-pass OWN-T launch)",
    "DEP_PAD_FILL": "the spec sampler's full-tensor padding-row fills",
    "DEP_PROJ_CUTE": "the AutoTuner's tactic for qkv_proj and o_proj",
    "DEP_PROJ_PIN": "the engine's tactic choice for qkv_proj and o_proj",
    "DEP_REJ_MC_FI": "FlashInfer's chain speculative sampling",
    "DEP_ROUTER_OPERAND": "the gate's torch.cat of the replicated BF16 operand",
    "DEP_SE_SIDE": "the shared expert after the routed launch at 65..128 rows per rank",
    "DEP_SFP_MC_FI": "FlashInfer's sampling_from_probs",
    "DEP_SHARED_MXFP8": "the shared expert's GEMMs quantizing their activations themselves",
    "DEP_SHARED_GU_MXFP8": "the shared expert's gate_up quantizing its activation itself",
    "DEP_SHARED_DOWN_MXFP8": "the shared expert's down quantizing its activation itself",
    "DEP_SHARED_PIN": "the engine's tactic choice for the shared expert's GEMMs",
    "DEP_TAIL": "the MoE's own top-k reduce and the stock shared-expert add and boundary norm",
    "DEP_TAIL_NORM": "the top-k reduce folded into the shared-expert add only (form (a)), never into the norm",
    "MEGAMOE_PAD_MASK": "CUDA-graph padding rows routed to experts like real tokens",
    "MEGAMOE_LOCAL": "the MegaMoE backend without its local arm (read with the backend's rule, engagement.py)",
    "NATIVE_WIDEN_ADD": "the stock widening add of the draft logits and the Markov bias",
    "NORM_L2PF": "the boundary norm without the next qkv_proj weight prefetch",
    "SOFTMAX_MC": "FlashInfer's softmax in the spec sampler",
    "SOFTMAX_PATHB": "FlashInfer's softmax at 129..448 rows",
    "TOPP_DET": "FlashInfer's deterministic top-p renorm",
    "TOPP_SPLIT": "the top-p renorm's single launch at every row count",
    "TP2WIDE_LC2_SPLIT": "the LC2 cluster kernel at every row count (no split workspace)",
    # Read by the family DSpark drafter glue through DEP_DSPARK_GLUE (ops/dspark/dspark_glue_install).
    "DRAFTER_LEAN_GLUE": "the drafter's stock strided-view glue",
    "DRAFTER_FUSED_GLUE": "every fused drafter glue op (the DRAFTER_*_FUSED / _GRID items)",
    "DRAFTER_CTX_KV_STORE_FUSED": "the stock drafter context-KV store",
    "DRAFTER_BLOCK_TABLE_REFRESH_FUSED": "the stock drafter block-table refresh",
    "DRAFTER_BLOCK_TABLE_REFRESH_GRID": "the drafter block-table refresh on one CTA",
    "DRAFTER_INPUTS_FUSED": "the stock drafter inputs",
    "DRAFTER_CTX_ADVANCE_FUSED": "the stock drafter context advance",
    "DRAFTER_FORWARD_INDICES_FUSED": "the stock drafter forward indices",
    "ADV_PAR": "the drafter context advance on one CTA",
    "EX11": "the drafter's stock q / k norm + RoPE",
    "DRAFT_APPEND_FUSE": "the drafter's q / k norm + RoPE without the fused KV append",
    "STRICT_ACCEPT_FUSED": "the stock strict accept (no fused KV rewind)",
    "DRAFT_KV_REWIND_FUSED": "the stock drafter KV rewind",
    "TOPP_V6": "the top-p renorm's staged summation loop on every device",
    "DEP_OWNT_FASTPLAN": "OWN-T's main kernel loads the union's block weights (no plan-side derivation)",
}


def _is_set(name: str, value: str) -> bool:
    """Whether the switch variable ``name`` is set to ``value``; raises on a value of neither kind."""
    word = value.strip().lower()
    if word in _SET:
        return True
    if word in _UNSET:
        return False
    raise ValueError(
        f"{name}={value!r}: expected 1 / true / on / yes (set) or 0 / false / off / no / empty (unset)"
    )


# The item ids switched off in this process, read once at import.
_DISABLED: FrozenSet[str] = frozenset(
    name[len(_PREFIX) :]
    for name, value in os.environ.items()
    if name.startswith(_PREFIX) and _is_set(name, value)
)

# Set switches this registry does not list: one environment may serve several targets, the family modules and the
# engine, which read their own, so they are reported once, not refused.
_FOREIGN = sorted(_DISABLED - SWITCHES.keys())
if _FOREIGN:
    logger.warning(
        f"[M3.1 {_TARGET}] set switches this target does not read (the family, the engine or another target may): "
        + ", ".join(_PREFIX + item for item in _FOREIGN)
    )


def m31_disabled(item_id: str) -> bool:
    """Whether ``TRTLLM_M31_DISABLE_<item_id>`` turned the item off for this process. Raises ``KeyError`` for an id
    that is not in ``SWITCHES``."""
    if item_id not in SWITCHES:
        raise KeyError(
            f"{item_id!r} is not a switch of the {_TARGET} target (see m31_switches.SWITCHES)"
        )
    return item_id in _DISABLED


# Default-off items (opt-ins): ``TRTLLM_M31_ENABLE_<ID>=1`` turns item ``<ID>`` on. Each changes the target's numerics
# against the default path (not bitwise) or is not faster on every shape, so it stays off unless asked for. Same strict
# values as the kill switches, read once at import; an opt-in variable naming an id this target does not register is
# logged once and otherwise ignored.
_ENABLE_PREFIX = "TRTLLM_M31_ENABLE_"

# Opt-in id -> what TRTLLM_M31_ENABLE_<id>=1 turns on.
OPT_INS: Dict[str, str] = {
    "SLOT_IDS_CHANGED": (
        "the spec sampler's slot-id table uploaded only when it changes (RuntimeHooks spec_slot_ids_changed_only)"
    ),
}


def _read_enabled() -> FrozenSet[str]:
    """The registered opt-in ids switched on in this process; logs each unregistered opt-in variable once."""
    out = set()
    for name, value in os.environ.items():
        if not name.startswith(_ENABLE_PREFIX):
            continue
        item_id = name[len(_ENABLE_PREFIX) :]
        if item_id not in OPT_INS:
            logger.warning(f"{name} is not an opt-in of this MiniMax-M3.1 target; ignored")
            continue
        if _is_set(name, value):
            out.add(item_id)
    return frozenset(out)


# The opt-in ids switched on in this process, read once at import.
_ENABLED: FrozenSet[str] = _read_enabled()


def m31_enabled(item_id: str) -> bool:
    """Whether ``TRTLLM_M31_ENABLE_<item_id>`` turned the default-off item on for this process."""
    if item_id not in OPT_INS:
        raise KeyError(
            f"{item_id!r} is not a registered opt-in of this target (m31_switches.OPT_INS)"
        )
    return item_id in _ENABLED
