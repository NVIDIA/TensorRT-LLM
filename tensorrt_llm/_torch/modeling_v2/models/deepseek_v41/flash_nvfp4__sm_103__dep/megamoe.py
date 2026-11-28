# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MegaMoE (`moe_config.backend: MEGAMOE_CUTEDSL`) path of the dep target: what is decided at build time and the
per-layer module.

Selected when the engine's resolved `model_config.moe_backend` is `MEGAMOE_CUTEDSL`; any other backend keeps the
TRTLLM fp4 path of `modeling.py` (`_Moe`). This file assembles the family's entries (`ops/moe/megamoe_*`,
`ops/moe/cute_dsl_megamoe_nvfp4`; each module's docstring states its contract) and the shared catalog's `cublas_mm`:

* `resolve_megamoe_config` (pure python, runs in the constructor): the deployment facts from the engine's config
  (EP size, slot count, bucket cap, fused shared expert, combine format). A static placement only: see below.
* `build_megamoe` (collective, runs at the start of weight loading, every rank, same order): the EP process group,
  every launch bucket's symmetric regions, the top-k index buffers and the shared expert's MXFP8 buffers. Nothing
  collective happens in a forward or a CUDA-graph capture.
* `_MegaMoe` (one per layer): the router (sqrtsoftplus, bias-selected, `norm_topk_prob`, `routed_scaling_factor`),
  the per-layer slot table and the packed expert stacks (filled by `weights.load_megamoe_experts`), and the forward:
  gate GEMM, one route + slot lookup + NVFP4 quantize (+ MXFP8 of the shared expert's input) launch that writes the
  bucket, one fused launch.
  Every rank launches every layer, also with no rows.
* `prime`: every bucket's kernel is compiled and fenced once at build time with zero live rows (what the engine's
  `_prime_ladder` does at its first launch), so no first launch lands inside a graph capture or a serving step.

Placement: every layer starts from the placement the engine starts from (`initial_global_assignments` of the layer
when configured, else the default: rank `r` holds `num_slots // ep` consecutive experts from `r * E // ep`, wrapping;
with extra slots a replicated cover whose copies the slot table spreads a hot expert's rows over). With
`moe_config.load_balancer.warmup_placement` the layers join the engine's warmup placement controller as the engine's
MoE layers do (`ops/moe/megamoe_placement`): the routing launch counts the picks, the controller re-places experts at
step boundaries (moving the per-slot tensors in place) and the layer rewrites its slot table in place. The outputs are
the same function under any placement; only the rank balance changes. The online and the per-layer dynamic balancers
are not implemented: routing declines them (`limits.dep_refused`).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._torch.modeling_v2._router_index import NoTarget
from tensorrt_llm._torch.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.cute_dsl_megamoe_nvfp4 import (
    cute_dsl_megamoe_nvfp4,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.dsv4_route_nvfp4_quant import (
    dsv4_route_nvfp4_quant,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.dsv4_router_logits import (
    dsv4_router_logits,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.megamoe_expert_slots import (
    megamoe_check_placement,
    megamoe_default_placement,
    megamoe_slot_table,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.megamoe_placement import (
    megamoe_placement_register,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.megamoe_shared_expert import (
    megamoe_shared_activation_buffers,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.megamoe_stage_inputs import (
    megamoe_stage_inputs,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.megamoe_workspace import (
    megamoe_ep_process_group,
    megamoe_pick_bucket,
    megamoe_shared_workspace_bytes,
    megamoe_symm_regions,
    megamoe_token_buckets,
    megamoe_topk_idx_buffer,
)
from tensorrt_llm.logger import logger

BACKEND = "MEGAMOE_CUTEDSL"
_COMBINE_FORMAT = (
    "bf16"  # the kernel's combine wire format: the production one (the engine's default)
)
_SF_VEC = 16  # NVFP4 activation scale group
_SM_VERSION = 103


@dataclass(frozen=True)
class MegaMoeConfig:
    """What the engine's config fixes for the MegaMoE path (identical on every rank)."""

    ep_size: int
    ep_rank: int
    num_experts: int
    num_slots: int
    top_k: int
    hidden: int
    inter: int
    clamp: float  # `swiglu_limit`, in real units (not divided by the FC1 scalar as `fp4_block_scale_moe_runner` wants)
    combine_format: str
    fused_shared: bool
    max_tokens: int  # the per-rank cap the bucket ladder is built from
    initial_assignments: Optional[
        Dict[int, List[int]]
    ]  # layer -> slot -> expert, when the config carries them

    @property
    def local_slots(self) -> int:
        return self.num_slots // self.ep_size


def resolve_megamoe_config(model_config, cfg) -> Optional[MegaMoeConfig]:
    """The MegaMoE deployment of this engine, or None when the backend is not `MEGAMOE_CUTEDSL`.

    Raises `NoTarget` (on every rank alike, so `modeling_v2: auto` falls back to the built-in model) for a layout the
    kernel cannot serve or a setting this target does not implement, instead of running something else; routing
    declines the same settings first (`limits.dep_refused`).
    """
    if getattr(model_config, "moe_backend", None) != BACKEND:
        return None
    mapping = model_config.mapping
    if not (
        mapping.enable_attention_dp
        and mapping.moe_tp_size == 1
        and mapping.moe_ep_size == mapping.world_size
    ):
        raise NoTarget(
            f"{BACKEND} needs pure DEP (attention DP, moe_tp 1, moe_ep == world size: the kernel's barrier fence runs "
            f"on WORLD); got world_size={mapping.world_size} moe_tp={mapping.moe_tp_size} "
            f"moe_ep={mapping.moe_ep_size} attention_dp={mapping.enable_attention_dp}"
        )
    ep, experts = mapping.moe_ep_size, cfg.n_routed_experts
    balancer = getattr(model_config, "moe_load_balancer", None)
    mode = getattr(balancer, "mode", "standard")
    if mode != "standard":
        raise NoTarget(
            f"moe_config.load_balancer.mode={mode!r}: this target implements the static placement only"
        )
    num_slots = int(getattr(balancer, "num_slots", None) or experts)
    if num_slots % ep or num_slots < experts:
        raise NoTarget(
            f"load_balancer.num_slots {num_slots} must be a multiple of EP {ep} and >= {experts}"
        )
    if balancer is not None and getattr(balancer, "layer_updates_per_iter", 0) > 0:
        raise NoTarget(
            f"moe_config.load_balancer.layer_updates_per_iter={balancer.layer_updates_per_iter}: "
            "this target implements "
            f"the static placement and the warmup placement (warmup_placement), not the online balancer"
        )
    combine = _COMBINE_FORMAT
    hidden, inter = cfg.hidden_size, cfg.moe_intermediate_size
    fused = bool(getattr(model_config, "megamoe_fuse_shared_expert", False))
    if fused:
        # The stock backend's own conditions (`fused_shared_expert_count`) plus what the packing needs.
        why = []
        if getattr(cfg, "n_shared_experts", 0) != 1:
            why.append("not exactly one shared expert")
        if hidden % 512 or (2 * inter) % 512:
            why.append("H and 2*I must be multiples of 512")
        if why:
            logger.warning(
                f"modeling_v2 deepseek_v41: megamoe_fuse_shared_expert is off: {'; '.join(why)}"
            )
            fused = False
    # The engine's ladder cap: the MoE cap and the engine's per-rank token cap, whichever is smaller.
    cap = int(getattr(model_config, "moe_max_num_tokens", 0) or 0)
    engine_cap = int(getattr(model_config, "max_num_tokens", 0) or 0)
    if 0 < cap < engine_cap:
        # Stock's MegaMoE scheduler runs such a step as ceil(rows / moe_max_num_tokens) lockstep launches; this
        # target launches every step whole, in the smallest bucket of a ladder that ends at the cap, so a rank's
        # step past the cap would have no bucket. Refused here, on every rank alike, rather than at that step.
        raise NoTarget(
            f"{BACKEND}: moe_config.max_num_tokens {cap} is below the engine's per-rank max_num_tokens {engine_cap}; "
            f"this target launches each step whole (no MoE chunking), so leave moe_config.max_num_tokens unset or "
            f">= max_num_tokens"
        )
    if engine_cap > 0:
        cap = min(cap, engine_cap) if cap > 0 else engine_cap
    initial = getattr(balancer, "initial_global_assignments", None)
    return MegaMoeConfig(
        ep_size=ep,
        ep_rank=mapping.moe_ep_rank,
        num_experts=experts,
        num_slots=num_slots,
        top_k=cfg.num_experts_per_tok,
        hidden=hidden,
        inter=inter,
        clamp=float(cfg.swiglu_limit),
        combine_format=combine,
        fused_shared=fused,
        max_tokens=cap,
        initial_assignments={int(k): list(v) for k, v in initial.items()} if initial else None,
    )


def layer_assignments(mc: MegaMoeConfig, layer_idx: int) -> List[int]:
    """Slot -> expert of one layer: the configured `initial_global_assignments`, else the default placement."""
    if mc.initial_assignments is not None:
        assign = mc.initial_assignments[layer_idx]
        if len(assign) != mc.num_slots:
            raise ValueError(
                f"layer {layer_idx}: {len(assign)} assignments for {mc.num_slots} slots"
            )
    else:
        assign = megamoe_default_placement(mc.num_experts, mc.num_slots, mc.ep_size)
    megamoe_check_placement(assign, mc.num_experts, mc.ep_size)
    return list(assign)


class MegaMoeBuild:
    """Process-wide build state shared by every MoE layer (layers run one after another): the buckets, their
    symmetric regions, top-k index buffers and the shared expert's MXFP8 input buffers."""

    def __init__(self, mc: MegaMoeConfig, mapping, device):
        self.mc = mc
        pg = megamoe_ep_process_group(mapping)
        self.buckets = megamoe_token_buckets(mc.max_tokens, _SM_VERSION)
        self.regions, self.idx, self.x8 = {}, {}, {}
        # Collective and ordered: every rank walks the same buckets in the same order.
        for b in self.buckets:
            shared_bytes, _ = megamoe_shared_workspace_bytes(
                world_size=mc.ep_size,
                num_topk=mc.top_k,
                num_experts_per_rank=mc.local_slots,
                hidden_size=mc.hidden,
                intermediate_size=mc.inter,
                max_tokens_per_rank=b,
                gate_up_clamp=mc.clamp,
                combine_format=mc.combine_format,
            )
            self.regions[b] = megamoe_symm_regions(
                process_group=pg,
                world_size=mc.ep_size,
                rank=mc.ep_rank,
                hidden_size=mc.hidden,
                max_tokens_per_rank=b,
                num_topk=mc.top_k,
                combine_format=mc.combine_format,
                shared_workspace_bytes=shared_bytes,
            )
            self.idx[b] = megamoe_topk_idx_buffer(b, mc.top_k, device)
            if mc.fused_shared:
                self.x8[b] = megamoe_shared_activation_buffers(b, mc.hidden, device)
        logger.info(
            f"modeling_v2 deepseek_v41: MegaMoE built: ep={mc.ep_size} slots={mc.num_slots} "
            f"({mc.local_slots}/rank) buckets={self.buckets} combine={mc.combine_format} "
            f"fused_shared={mc.fused_shared}"
        )


class _MegaMoe(nn.Module):
    """Routed experts (and, fused, the shared expert) of one layer through the MegaMoE kernel.

    Replaces `_Moe` when the backend is `MEGAMOE_CUTEDSL`. With attention DP every rank passes its own rows and
    gets its own rows back: the kernel exchanges tokens itself, so there is no allgather / reduce-scatter here.
    """

    def __init__(self, cfg, mc: MegaMoeConfig, layer_idx: int):
        super().__init__()
        self.cfg, self.mc, self.layer_idx = cfg, mc, layer_idx
        self.topk = mc.top_k
        self.assignments = layer_assignments(mc, layer_idx)
        self.gate_weight = nn.Parameter(
            torch.empty(mc.num_experts, mc.hidden, dtype=torch.bfloat16), requires_grad=False
        )
        self.gate_bias = nn.Parameter(
            torch.empty(mc.num_experts, dtype=torch.float32), requires_grad=False
        )
        self.build: Optional[MegaMoeBuild] = None
        self.shared_pack: Optional[dict] = None
        # The engine's warmup placement (when configured) learns this layer: registered in layer order on every rank.
        self.placement = megamoe_placement_register(
            self,
            mc.num_experts,
            mc.top_k,
            mc.local_slots,
            self.assignments,
            self._placement_changed,
        )

    def _placement_changed(self, assignments: List[int]) -> None:
        """The warmup placement moved this layer's experts (per-slot
        tensors already in place): route to the new slots."""
        megamoe_check_placement(assignments, self.mc.num_experts, self.mc.ep_size)
        self.assignments = list(assignments)
        self.table.copy_(
            megamoe_slot_table(
                self.assignments, self.mc.num_experts, self.mc.ep_rank, self.table.device
            )
        )

    # ---- state attached at load (see weights.load_megamoe_experts / load_megamoe_shared) ----------------
    def set_experts(
        self, fc1, fc1_sf, fc2, fc2_sf, scalars: Dict[str, torch.Tensor], device
    ) -> None:
        for name, t in (("fc1", fc1), ("fc1_sf", fc1_sf), ("fc2", fc2), ("fc2_sf", fc2_sf)):
            self.register_buffer(name, t, persistent=False)
        for name in ("fc1_alpha", "fc2_alpha", "fc1_norm_const", "input_global_scale"):
            self.register_buffer(name, scalars[name], persistent=False)
        # One table per layer at a fixed address; rewritten in place when a controller moves experts.
        self.register_buffer(
            "table",
            megamoe_slot_table(self.assignments, self.mc.num_experts, self.mc.ep_rank, device),
            persistent=False,
        )

    def set_shared(self, pack: dict) -> None:
        for name in ("fc1_weight", "fc1_weight_sf", "fc2_weight", "fc2_weight_sf"):
            self.register_buffer(f"shared_{name}", pack[name], persistent=False)
        self.shared_pack = {
            "intermediate_size": pack["intermediate_size"],
            "gate_up_clamp": pack["gate_up_clamp"],
        }

    # ---- forward ----------------------------------------------------------------------------------------
    def _route(self, x: torch.Tensor):
        """The torch routing of the same rows (the oracle `dsv4_route_nvfp4_quant`
        is checked against; not on the hot path)."""
        logits = cublas_mm(x, self.gate_weight.t(), None, torch.float32)
        scores = torch.sqrt(F.softplus(logits))
        ids = torch.topk(scores + self.gate_bias, self.topk, dim=-1)[1]
        w = torch.gather(scores, 1, ids)
        w = w / (torch.sum(w, dim=-1, keepdim=True) + 1e-20)  # norm_topk_prob
        return (w * self.cfg.routed_scaling_factor).contiguous(), ids.to(torch.int32)

    def forward(self, x: torch.Tensor, all_rank_num_tokens: Sequence[int]) -> torch.Tensor:
        """x [n, H] bf16 -> [n, H] bf16 (routed experts, plus the shared expert when fused)."""
        assert all_rank_num_tokens is not None, (
            "MegaMoE picks its launch bucket from every rank's row count"
        )
        bucket = megamoe_pick_bucket(self.build.buckets, max(int(c) for c in all_rank_num_tokens))
        return self.run(x, bucket)

    def run(self, x: torch.Tensor, bucket: int) -> torch.Tensor:
        mc, bd = self.mc, self.build
        n, hidden, dev = x.shape[0], mc.hidden, x.device
        r = bd.regions[bucket]
        fused = self.shared_pack is not None
        if n > 0:
            # One launch routes, quantizes (NVFP4, and MXFP8 for the fused shared expert) and writes the bucket's inputs
            # (expert slots, weights, activation rows); rows past `n` get slot -1 and weight 0.
            x = x.contiguous()
            # fp32 K-chunk partials (one launch) at 65..1280 rows, which the route launch sums as it loads them
            logits = dsv4_router_logits(x, self.gate_weight, partials=True)
            dsv4_route_nvfp4_quant(
                logits, self.gate_bias, x, self.input_global_scale, self.topk, self.cfg.routed_scaling_factor,
                torch.float32,
                staged=(bd.idx[bucket], r.topk_weights, r.activation, r.activation_sf),
                slot_of_expert=self.table, replicas=self.table.numel() > mc.num_experts,
                mxfp8=bd.x8[bucket] if fused else None,
                counters=None if self.placement is None else self.placement.counters(),
            )  # fmt: skip
        else:  # an idle rank still takes part in every launch
            slots = torch.empty((0, self.topk), dtype=torch.int32, device=dev)
            weights = torch.empty((0, self.topk), dtype=torch.float32, device=dev)
            x_fp4 = torch.empty((0, hidden // 2), dtype=torch.uint8, device=dev)
            x_sf = torch.empty((0, hidden // _SF_VEC), dtype=torch.uint8, device=dev)
            megamoe_stage_inputs(
                slots,
                weights,
                bd.idx[bucket],
                r.topk_weights,
                x_fp4,
                x_sf,
                r.activation,
                r.activation_sf,
            )
        shared = {}
        if fused:
            shared = dict(
                shared_fc1_weight=self.shared_fc1_weight,
                shared_fc1_weight_sf=self.shared_fc1_weight_sf,
                shared_fc2_weight=self.shared_fc2_weight,
                shared_fc2_weight_sf=self.shared_fc2_weight_sf,
                shared_activation=bd.x8[bucket][0],
                shared_activation_sf=bd.x8[bucket][1],
                shared_intermediate_size=self.shared_pack["intermediate_size"],
                shared_gate_up_clamp=self.shared_pack["gate_up_clamp"],
            )
        return cute_dsl_megamoe_nvfp4(
            r.activation, r.activation_sf, bd.idx[bucket], r.topk_weights,
            self.fc1, self.fc1_sf, self.fc2, self.fc2_sf,
            self.fc1_alpha, self.fc2_alpha, self.fc1_norm_const,
            r.shared_workspace, r.peer_offsets,
            world_size=mc.ep_size, local_rank=mc.ep_rank, max_tokens_per_rank=bucket, num_tokens=n,
            gate_up_clamp=mc.clamp, combine_format=mc.combine_format, **shared,
        )  # fmt: skip


def attach_build(core) -> None:
    """Collective, first thing in weight loading: build the shared state and hand it to every layer."""
    build = MegaMoeBuild(core.mega_cfg, core.mapping, torch.device("cuda"))
    core.mega_build = build
    for layer in core.layers:
        layer.moe.build = build


def prime(core) -> None:
    """Compile and fence every bucket's kernel with zero live rows (every rank, build time, outside any capture).

    Every layer has the same kernel; the first layer's tensors stand for all. This is the engine's priming of its
    adaptive ladder, moved from the first serving launch to the end of loading.
    """
    layer0 = core.layers[0].moe
    empty = torch.empty((0, core.mega_cfg.hidden), dtype=torch.bfloat16, device="cuda")
    for b in core.mega_build.buckets:
        layer0.run(empty, b)
    torch.cuda.synchronize()
    logger.info(f"modeling_v2 deepseek_v41: MegaMoE primed buckets {core.mega_build.buckets}")
