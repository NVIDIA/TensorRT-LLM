# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-stage reads/writes contracts. Each read/write of a Stage must match a pattern of one of the stages it covers;
`ws.*` (its own arena carve-outs) is always allowed."""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase

_W = ("aw.*", "mw.*", "geo.*", "t")


@dataclass(frozen=True)
class Contract:
    slot: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]


def _c(slot, reads, writes):
    return Contract(slot, tuple(reads) + _W, tuple(writes))


# Sparse layers (L3-59): K1 / K2 / K3.
STAGES: dict[str, Contract] = {
    "boundary_in": _c(
        "K1",
        ["ffn_partial", "residual_in", "chain.main_md", "chain.idx_md"],
        ["residual_mid", "x_norm", "x_q"],
    ),
    "qkv": _c("K1", ["x_norm", "x_q"], ["chain.qkv"]),
    "producer": _c(
        "K1",
        ["chain.qkv", "chain.main_md", "chain.idx_md", "chain.kv_cache", "chain.index_cache"],
        ["chain.q", "chain.q_fp8", "chain.index_q", "chain.kv_cache", "chain.index_cache"],
    ),
    "scorer": _c("K1", ["chain.index_q", "chain.index_cache", "chain.idx_md"], ["chain.scores"]),
    "topk": _c("K1", ["chain.scores", "chain.idx_md", "chain.main_md"], ["chain.topk"]),
    "decode": _c(
        "K2",
        ["chain.q", "chain.q_fp8", "chain.kv_cache", "chain.topk", "chain.main_md"],
        ["chain.out"],
    ),
    "oproj": _c("K2", ["chain.out"], ["attn_partial"]),
    "boundary_post": _c(
        "K3", ["attn_partial", "residual_mid"], ["residual_out", "moe_in", "moe_in_q"]
    ),
    "router": _c("K3", ["moe_in"], ["route"]),
    "moe_quant": _c("K3", ["moe_in"], ["moe_in_q"]),
    "moe_experts": _c("K3", ["moe_in", "moe_in_q", "route"], ["ffn_out", "ffn_routed"]),
    "shared": _c("K3", ["moe_in", "moe_in_q"], ["ffn_shared", "ffn_out"]),
    "combine": _c("K3", ["ffn_routed", "ffn_shared", "route"], ["ffn_out"]),
    # Dense layers (L0-2): whole-layer or split; the L2 ffn_out IS the L3 ffn_partial (the plan binds one tensor).
    "dense_front": _c(
        "D",
        ["ffn_partial", "residual_in", "chain.*", "embed"],
        [
            "residual_mid",
            "x_norm",
            "x_q",
            "chain.qkv",
            "chain.q",
            "chain.q_fp8",
            "chain.kv_cache",
            "chain.index_cache",
            "chain.index_q",
        ],
    ),
    "dense_attn": _c("D", ["chain.*"], ["chain.out", "attn_partial"]),
    "dense_mlp": _c("D", ["attn_partial", "residual_mid"], ["residual_out", "ffn_out", "tap.*"]),
    # EAGLE3 pieces: blk is the drafter/runner view, fields under draft.* / target.*.
    "eagle_head": _c("E", ["draft.*", "target.*"], ["draft.*"]),
    "eagle_pass": _c("E", ["draft.*", "target.*", "tap.*"], ["draft.*"]),
    "eagle_combine": _c("E", ["draft.*", "tap.*"], ["draft.*"]),
    "eagle_glue": _c("E", ["draft.*", "target.*"], ["draft.*", "target.*"]),
    "eagle_verify": _c("E", ["draft.*", "target.*"], ["target.*"]),
    # DSpark drafter kernels: installed into vLLM's speculator by the plugin, outside the target chain.
    "draft_prep": _c("S", ["draft.*", "target.*"], ["draft.*"]),
    "draft_sample": _c("S", ["draft.*"], ["draft.*"]),
    "draft_boundary": _c("S", ["draft.*"], ["draft.*"]),
}

SPARSE_STAGES = (
    "boundary_in",
    "qkv",
    "producer",
    "scorer",
    "topk",
    "decode",
    "oproj",
    "boundary_post",
    "router",
    "moe_quant",
    "moe_experts",
    "shared",
    "combine",
)
DENSE_STAGES = ("dense_front", "dense_attn", "dense_mlp")
# M3.1 layers 0-2: the sparse attention slots, then the dense MLP (no router, no MoE)
M31D_STAGES = SPARSE_STAGES[:7] + ("dense_mlp",)
EAGLE_COMPONENTS = {
    "head": "eagle_head",
    "pass": "eagle_pass",
    "combine": "eagle_combine",
    "glue": "eagle_glue",
    "verify": "eagle_verify",
}
FAMILIES = (
    "front",
    "scorer",
    "topk",
    "decode",
    "oproj",
    "post",
    "router",
    "moe",
    "dense",
    "eagle",
    "drafter",
)


def allowed(stages, kind: str) -> tuple[str, ...]:
    out = ["ws.*"]
    for s in stages:
        c = STAGES[s]
        out.extend(c.reads if kind == "reads" else c.writes)
    return tuple(out)


def violations(stages, reads, writes) -> list[str]:
    bad = []
    for kind, fields in (("reads", reads), ("writes", writes)):
        pats = allowed(stages, kind)
        for f in fields:
            if not any(fnmatchcase(f, p) for p in pats):
                bad.append(f"{kind} {f!r} outside the contract of {tuple(stages)}")
    return bad


__all__ = [
    "Contract",
    "STAGES",
    "SPARSE_STAGES",
    "DENSE_STAGES",
    "M31D_STAGES",
    "EAGLE_COMPONENTS",
    "FAMILIES",
    "allowed",
    "violations",
]
