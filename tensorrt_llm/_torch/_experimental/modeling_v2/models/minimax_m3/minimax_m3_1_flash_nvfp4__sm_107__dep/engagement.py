# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Construction checks of the MiniMax-M3.1 DEP target: the deployment it serves and the MoE every layer runs.

Kept free of modeling imports so the checks run on CPU against stub layers.
"""

from __future__ import annotations

import collections
import os
from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Optional, Sequence

MOE_BACKEND = "MEGAMOE_CUTEDSL"
MEGAMOE_LOCAL_DISABLE_ENV = "TRTLLM_M31_DISABLE_MEGAMOE_LOCAL"
# The canonical MoEImplId every MoE layer must declare, by whether the MegaMoE local arm is requested:
# TrtllmCutedslMegaMoeNvfp4Impl when not, the MegaMoE local implementation when it is. Both are MOE_BACKEND members, so
# membership cannot tell them apart, and a local implementation that declines a layer resolves to the stock identity.
ARM_IDENTITIES = {
    False: "trtllm.cutedsl.mega_moe.nvfp4",
    True: "trtllm.cutedsl.mega_moe_local.nvfp4",
}
DEP_SIZES = (4, 8, 16)


def is_dep(m: Any) -> bool:
    """The MiniMax-M3.1 DEP generation topology: one attention-DP group over every rank, with the routed experts
    split over all of them by expert parallelism (DEP4 / DEP8 / DEP16)."""
    return bool(
        m.enable_attention_dp
        and m.world_size in DEP_SIZES
        and m.tp_size == m.world_size
        and m.moe_ep_size == m.world_size
        and m.moe_tp_size == 1
        and m.pp_size == 1
        and m.cp_size == 1
    )


@dataclass(frozen=True)
class MoeLayerIdentity:
    """What one MoE layer's experts run, as the identity log prints it."""

    impl: str
    identity: str
    bias: Any
    activation: str
    combine_format: Any


def deployment_mismatch(mapping: Any, moe_backend: str) -> Optional[str]:
    """Why this deployment is not one the DEP target serves, or None. The topology test is the one the route takes."""
    reasons = []
    if not is_dep(mapping):
        reasons.append(
            f"topology world {mapping.world_size} tp {mapping.tp_size} moe_ep {mapping.moe_ep_size} "
            f"moe_tp {mapping.moe_tp_size} attention_dp {mapping.enable_attention_dp} is not DEP"
        )
    if str(moe_backend).upper() != MOE_BACKEND:
        reasons.append(f"moe_backend {moe_backend}, not {MOE_BACKEND}")
    if getattr(mapping, "enable_lm_head_tp_in_adp", False):
        reasons.append(
            "enable_lm_head_tp_in_adp is set (the DSpark drafter refuses a vocab-sharded head under ADP)"
        )
    return "; ".join(reasons) or None


def _activation(backend: Any) -> str:
    activation = getattr(backend, "activation", None)
    if activation is None:
        return "unknown"
    constants = activation.constants()
    return f"{type(activation).__name__}(alpha={constants.alpha!r}, beta={constants.beta!r}, limit={constants.limit!r})"


def megamoe_local_requested(environ: Optional[Mapping[str, str]] = None) -> bool:
    """Whether the MegaMoE backend builds its local arm: on unless the disable switch is exactly ``1``. This must stay
    the backend's own rule (``local_plan.megamoe_local_requested``): any other reading makes the target require
    an identity the backend did not build and refuse at construction."""
    env = os.environ if environ is None else environ
    return env.get(MEGAMOE_LOCAL_DISABLE_ENV) != "1"


def arm_identity(environ: Optional[Mapping[str, str]] = None) -> str:
    """The MoE implementation identity this process's arm requires (``ARM_IDENTITIES``)."""
    return ARM_IDENTITIES[megamoe_local_requested(environ)]


def moe_engagement(
    layers: Sequence[Any], moe_layer_ids: Iterable[int], fused_comm: Any, identity: str
) -> List[MoeLayerIdentity]:
    """The identity of every MoE layer's experts; raises unless all of them are ``identity`` with fused comm.

    ``moe_layer_ids`` is the config's MoE layer set (``get_moe_layer_ids``): each of those layers must carry a MoE
    and no other layer may. The resolver serves a layer every ``MOE_BACKEND`` member rejects with its CutlassFusedMoE
    fallback, so a layer is matched by its implementation's declared identity, never by family membership.
    """
    expected = frozenset(moe_layer_ids)
    if not expected:
        raise RuntimeError("the MiniMax-M3.1 DEP target found no MoE layer in the config")
    identities, refused = [], []
    for idx in sorted(expected - set(range(len(layers)))):
        refused.append(f"layer {idx}: not built ({len(layers)} layers)")
    for idx, layer in enumerate(layers):
        moe = getattr(layer, "block_sparse_moe", None)
        if idx not in expected:
            if moe is not None:
                refused.append(f"layer {idx}: a MoE on a layer the config makes dense")
            continue
        if moe is None:
            refused.append(f"layer {idx}: no MoE on a layer the config makes MoE")
            continue
        backend = getattr(moe.experts, "backend", moe.experts)
        impl = type(backend)
        declared = str(getattr(getattr(impl, "descriptor", None), "identity", None))
        scheduler = getattr(backend, "scheduler_kind", None)
        if declared != identity or scheduler != fused_comm:
            refused.append(
                f"layer {idx}: {impl.__name__} (identity {declared}, scheduler {scheduler})"
            )
            continue
        identities.append(
            MoeLayerIdentity(
                impl=impl.__name__,
                identity=declared,
                bias=getattr(backend, "bias", None),
                activation=_activation(backend),
                combine_format=getattr(backend, "combine_format", None),
            )
        )
    if refused:
        arm = "requested" if identity == ARM_IDENTITIES[True] else "not requested"
        raise RuntimeError(
            f"the MiniMax-M3.1 DEP target requires every MoE layer on {identity} ({MOE_BACKEND}, "
            f"MegaMoE local {arm}) with fused communication; refused: " + "; ".join(refused)
        )
    return identities


def identity_lines(
    rank: int, world: int, identities: List[MoeLayerIdentity], kernel: str
) -> List[str]:
    """One log line per distinct layer identity, with its layer count."""
    counts = collections.Counter(identities)
    return [
        f"[M3.1 DEP] rank {rank}/{world} MoE {ident.identity} x{count} impl={ident.impl} kernel={kernel} "
        f"bias={ident.bias} act={ident.activation} combine={ident.combine_format}"
        for ident, count in counts.items()
    ]
