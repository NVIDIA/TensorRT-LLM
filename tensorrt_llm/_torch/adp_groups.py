# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Attention-DP groups with tensor-parallel attention inside each group (ADP x TP). Opt-in, off by default.

Under ``enable_attention_dp`` every rank is its own attention-DP rank and holds every attention head
(``Mapping.dp_size == tp_size``). ``TRTLLM_ADPTP_ATTN_TP=t`` (an integer, 1 < t <= tp_size, t dividing tp_size)
groups the tp ranks into ``tp_size // t`` groups of ``t`` consecutive ranks:

* the ranks of one group serve the same requests: the attention-DP router routes requests to groups, every rank of a
  group admits and schedules the same requests, and only the group's first rank returns their responses;
* inside a group, attention is tensor-parallel: each rank holds ``heads / t`` query heads and ``kv_heads / t`` KV heads
  (KV cache included), and the attention output projection all-reduces over the group;
* the routed experts stay expert-parallel over all ranks: each rank of a group dispatches a ``1 / t`` slice of the
  group's rows and the group all-gathers the MoE output, so every rank of a group holds the same bytes;
* every other module keeps attention DP's layout (replicated weights, no reduction).

With the switch unset (or 1) every helper here returns the stock attention-DP answer, so the stock paths are unchanged.
``t == tp_size`` (one group) is attention TP over every rank with the experts dispatched by row slices (the 1-node test
layout); ``t == 1`` is plain attention DP.

Only pure-Python logic lives here (no torch import), so the topology, routing and row-slice rules run in CPU tests.
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, TypeVar

ENV = "TRTLLM_ADPTP_ATTN_TP"
# Per-step check that the ranks of every group scheduled the same batch (one int64 pair per rank, piggybacked on the
# attention-DP batch-size all-gather). On by default in group mode; TRTLLM_ADPTP_CHECK=0 skips the signature.
CHECK_ENV = "TRTLLM_ADPTP_CHECK"
# The names the MiniMax-M3.1 port used; read only when the new name is unset (aliases).
LEGACY_ENV = "TRTLLM_M31_ADPTP_ATTN_TP"
LEGACY_CHECK_ENV = "TRTLLM_M31_ADPTP_CHECK"


def _env_value(env: Mapping[str, str], name: str, legacy: str) -> Tuple[str, str]:
    """(name, raw value) of the switch: ``name`` when set (non-empty), else its legacy alias."""
    raw = env.get(name, "").strip()
    if raw == "" and env.get(legacy, "").strip() != "":
        return legacy, env[legacy].strip()
    return name, raw

T = TypeVar("T")


def requested_attn_tp(environ: Optional[Mapping[str, str]] = None) -> int:
    """The attention-TP size the switch asks for (1 when unset or empty). Raises on a malformed value."""
    env = os.environ if environ is None else environ
    name, raw = _env_value(env, ENV, LEGACY_ENV)
    if raw == "":
        return 1
    try:
        value = int(raw)
    except ValueError as err:
        raise ValueError(f"{name}={raw!r} is not an integer") from err
    if value < 1:
        raise ValueError(f"{name}={value} must be >= 1")
    return value


def topology_mismatch(mapping: Any, attn_tp: int) -> Optional[str]:
    """Why ``mapping`` cannot be split into attention-TP groups of ``attn_tp`` ranks, or None."""
    reasons = []
    if not getattr(mapping, "enable_attention_dp", False):
        reasons.append("enable_attention_dp is off")
    tp = int(getattr(mapping, "tp_size", 1))
    if attn_tp > tp or tp % attn_tp != 0:
        reasons.append(f"attention TP {attn_tp} does not divide tp_size {tp}")
    if int(getattr(mapping, "pp_size", 1)) != 1:
        reasons.append(f"pp_size {mapping.pp_size} != 1")
    if int(getattr(mapping, "cp_size", 1)) != 1:
        reasons.append(f"cp_size {mapping.cp_size} != 1")
    if getattr(mapping, "enable_lm_head_tp_in_adp", False):
        reasons.append("enable_lm_head_tp_in_adp is set")
    return "; ".join(reasons) or None


def group_attn_tp(mapping: Any, environ: Optional[Mapping[str, str]] = None) -> int:
    """Ranks per attention group for ``mapping``: 1 (plain attention DP, or no attention DP) unless the switch asks
    for more and the mapping is attention DP. Raises when the switch is set and the mapping cannot be grouped."""
    if mapping is None or not getattr(mapping, "enable_attention_dp", False):
        return 1
    t = requested_attn_tp(environ)
    if t == 1:
        return 1
    why = topology_mismatch(mapping, t)
    if why is not None:
        raise ValueError(f"{ENV}={t}: cannot group this attention-DP mapping: {why}")
    return t


def is_grouped(mapping: Any, environ: Optional[Mapping[str, str]] = None) -> bool:
    return group_attn_tp(mapping, environ) > 1


def attn_tp_size(mapping: Any, environ: Optional[Mapping[str, str]] = None) -> int:
    """How many ways each rank's attention heads (and KV heads) are split: ``tp_size`` without attention DP, the group
    size under attention DP (1 unless grouped). Replaces the stock ``1 if enable_attention_dp else tp_size``."""
    if mapping is None:
        return 1
    if not getattr(mapping, "enable_attention_dp", False):
        return int(mapping.tp_size)
    return group_attn_tp(mapping, environ)


def attn_tp_rank(mapping: Any, environ: Optional[Mapping[str, str]] = None) -> int:
    """This rank's position in its attention-TP group (``tp_rank`` without attention DP, 0 under plain ADP)."""
    if mapping is None:
        return 0
    if not getattr(mapping, "enable_attention_dp", False):
        return int(mapping.tp_rank)
    return int(mapping.tp_rank) % group_attn_tp(mapping, environ)


def dp_size(mapping: Any, environ: Optional[Mapping[str, str]] = None) -> int:
    """Attention-DP ranks the router routes to: tp_size // group size under attention DP, else 1."""
    if mapping is None or not getattr(mapping, "enable_attention_dp", False):
        return 1
    return int(mapping.tp_size) // group_attn_tp(mapping, environ)


def dp_rank(mapping: Any, environ: Optional[Mapping[str, str]] = None) -> int:
    """This rank's attention-DP group (the index the router's output is read at)."""
    if mapping is None or not getattr(mapping, "enable_attention_dp", False):
        return 0
    return int(mapping.tp_rank) // group_attn_tp(mapping, environ)


def is_group_leader(mapping: Any, environ: Optional[Mapping[str, str]] = None) -> bool:
    """Whether this rank returns its group's responses (the group's first rank; every rank when not grouped)."""
    return attn_tp_rank(mapping, environ) == 0 if is_grouped(mapping, environ) else True


def attention_shard(rank: int, world_size: int, enable_attention_dp: bool,
                    environ: Optional[Mapping[str, str]] = None) -> Tuple[int, int]:
    """(shard index, shard count) of this rank's attention KV for keying content shared across workers (the Mooncake
    store namespaces): ``(rank, world_size)`` without attention DP, ``(0, 1)`` under plain attention DP (every rank
    holds every head), ``(rank % t, t)`` under attention-DP groups of ``t`` ranks (the group's TP shard, the same keys a
    TP-``t`` worker uses). Assumes the grouped layout's tp == world (pp 1, cp 1; ``topology_mismatch``)."""
    if not enable_attention_dp:
        return int(rank), int(world_size)
    t = requested_attn_tp(environ)
    if t <= 1:
        return 0, 1
    if t > world_size or world_size % t != 0:
        raise ValueError(f"{ENV}={t}: attention TP does not divide world size {world_size}")
    return int(rank) % t, t


def group_tp_ranks(group: int, attn_tp: int) -> List[int]:
    """The tp ranks of attention group ``group`` (consecutive, the attention module's ``pp``-as-DP mapping layout)."""
    return list(range(group * attn_tp, (group + 1) * attn_tp))


def collapse(per_rank: Sequence[T], attn_tp: int) -> List[T]:
    """One entry per group (its first rank's) from one entry per tp rank."""
    if attn_tp <= 1:
        return list(per_rank)
    if len(per_rank) % attn_tp != 0:
        raise ValueError(f"{len(per_rank)} rank entries do not split into groups of {attn_tp}")
    return [per_rank[g * attn_tp] for g in range(len(per_rank) // attn_tp)]


def route_states(per_rank_states: Sequence[T], attn_tp: int) -> List[T]:
    """The attention-DP router's input under groups: one rank state per group (its first rank's), re-keyed so that
    ``state.rank`` is the group index.

    The routers key their output dict by ``state.rank`` but fill it by the state's position (heap / explicit dp-rank
    index), and the executor reads it at the group index. A plain ``collapse`` keeps the leaders' tp ranks
    (0, t, 2t, ...), so with two or more groups the second group's assignment raised ``KeyError`` (or was dropped).
    Copies are returned; the gathered per-rank states stay untouched. ``attn_tp <= 1`` returns the stock list."""
    if attn_tp <= 1:
        return list(per_rank_states)
    out = []
    for group, state in enumerate(collapse(per_rank_states, attn_tp)):
        state = copy.copy(state)
        state.rank = group
        out.append(state)
    return out


def expand(per_group: Mapping[int, T], attn_tp: int, num_ranks: int) -> Dict[int, T]:
    """One entry per tp rank (its group's) from one entry per group."""
    if attn_tp <= 1:
        return dict(per_group)
    return {r: per_group[r // attn_tp] for r in range(num_ranks)}


def row_slice(rows: int, attn_tp: int, rank_in_group: int) -> Tuple[int, int, int]:
    """(start, stop, width) of the rows this rank of a group dispatches to the experts.

    Every rank of a group takes ``width = ceil(rows / attn_tp)`` rows, the last ones padded past ``rows``; the group
    all-gather of the ``width``-row outputs then holds the ``rows`` real rows first. ``stop`` is clipped to ``rows``
    (``stop - start`` real rows, possibly 0); a group with no rows has width 0.
    """
    if attn_tp < 1 or not 0 <= rank_in_group < attn_tp:
        raise ValueError(f"rank {rank_in_group} is not in a group of {attn_tp}")
    width = -(-rows // attn_tp) if rows > 0 else 0
    start = min(rows, rank_in_group * width)
    stop = min(rows, start + width)
    return start, stop, width


def sliced_rank_tokens(all_rank_num_tokens: Sequence[int], attn_tp: int) -> List[int]:
    """The per-rank token counts the expert dispatch sees when every group dispatches row slices.

    ``all_rank_num_tokens`` is the stock attention-DP list (one entry per rank; the ranks of a group carry the group's
    row count). Every rank of a group then dispatches ``ceil(rows / attn_tp)`` rows."""
    counts = list(all_rank_num_tokens)
    if attn_tp <= 1:
        return counts
    if len(counts) % attn_tp != 0:
        raise ValueError(f"{len(counts)} rank entries do not split into groups of {attn_tp}")
    out = []
    for g in range(len(counts) // attn_tp):
        rows = counts[g * attn_tp: (g + 1) * attn_tp]
        if len(set(rows)) != 1:
            raise ValueError(f"group {g} ranks report different row counts {rows}")
        width = row_slice(rows[0], attn_tp, 0)[2]
        out += [width] * attn_tp
    return out


def batch_signature(entries) -> int:
    """A process-independent int64 signature (FNV-1a over int64 words) of a scheduled batch: one tuple of ints per
    request, in order (the executor sends request id, token count and last token id, so a group whose ranks sampled
    different tokens is caught on the next step)."""
    acc = 1469598103934665603
    for entry in entries:
        for v in entry:
            acc = ((acc ^ (int(v) & 0xFFFFFFFFFFFFFFFF)) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return acc - (1 << 64) if acc >= (1 << 63) else acc


# ADP routers that route over the collapsed per-group states correctly (see router_mismatch).
# - DefaultADPRouter: heap over the group states.
# - ConversationAwareADPRouter: its conversation -> rank table, round-robin cursor and token load are filled from the
#   route states' positions, i.e. group indices under groups (no per-tp-rank table).
# - KVCacheAwareADPRouter: its per-rank prefix-match table is collapsed to one entry per group (the group's first rank:
#   every rank of a group admits the same requests, so the group's radix trees hold the same blocks) and its cold-start
#   warmup set spans the groups.
GROUP_SAFE_ROUTERS = frozenset(("DefaultADPRouter", "ConversationAwareADPRouter", "KVCacheAwareADPRouter"))


def router_mismatch(router_cls_name: str) -> Optional[str]:
    """Why an ADP router cannot serve attention-DP groups, or None."""
    if router_cls_name in GROUP_SAFE_ROUTERS:
        return None
    return (f"the {router_cls_name} keeps per-rank tables (prefix matches, warmup ranks) indexed by tp rank; "
            f"attention-DP groups support {sorted(GROUP_SAFE_ROUTERS)} only")


def check_enabled(environ: Optional[Mapping[str, str]] = None) -> bool:
    env = os.environ if environ is None else environ
    _, raw = _env_value(env, CHECK_ENV, LEGACY_CHECK_ENV)
    return raw != "0"


def diverged_groups(signatures: Sequence[int], attn_tp: int) -> List[int]:
    """The groups whose ranks reported different batch signatures."""
    if attn_tp <= 1:
        return []
    return [
        g
        for g in range(len(signatures) // attn_tp)
        if len(set(signatures[g * attn_tp: (g + 1) * attn_tp])) != 1
    ]
