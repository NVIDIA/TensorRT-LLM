# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The KNOBS the MoE hosts read, under descriptive names, each with the legacy name it replaces.

A knob is bound as M31_TP4C1_MOE_<NAME> or under its legacy alias (presets and harnesses still bind the aliases);
binding both to different values raises. Values are strings with env semantics; each read site keeps its own parse and
default. The hosts write only legacy names (the `written` knobs); binding one of those by its new name to a different
value fails at the next read.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...runtime.knobs import KNOBS, KnobConflict

PREFIX = "M31_TP4C1_MOE_"


@dataclass(frozen=True)
class Knob:
    name: str
    alias: str
    doc: str
    written: bool = (
        False  # the hosts also write it (by its alias), so a save/restore reads the alias too
    )

    @property
    def env(self) -> str:
        return PREFIX + self.name.upper()


TABLE = (
    # engine packages (bound by the moe family from its typed params)
    Knob("fused_engine_dir", "H4_DL_PORT", "fused engine package dir (kernels/moe/engine/fused)"),
    Knob(
        "expert_local_engine_dir", "H9_X3_PORT", "expert_local engine package dir, static FC1 scale"
    ),
    Knob(
        "expert_local_per_token_engine_dir",
        "PTC2_PORT",
        "expert_local engine package dir, per-token FC1 scale",
    ),
    # fused engine build
    Knob(
        "fused_shared_trigger_cfg",
        "H9_SHT_CFG",
        'json {"sh_trig": 0|1}: shared CTAs trigger PDL after their wait',
    ),
    Knob(
        "fused_tp4_engine_cfg",
        "TP4_MOE_OVER",
        'json engine keys merged into every TP4 build (default {"b2img": 1})',
    ),
    Knob(
        "fused_shared_ctas_le4_rows",
        "H4_NSH4",
        "shared-expert CTAs at <= 4 rows (TP2 48, TP4 24)",
        written=True,
    ),
    Knob("fused_shared_ctas_5_8_rows", "H4_NSH8", "shared-expert CTAs at 5..8 rows (24)"),
    Knob(
        "fused_push",
        "H4_PUSH",
        "1: the finalize pushes its rows into the next layer's all-reduce slots (default)",
    ),
    Knob("fused_engine_cfg_le4_rows", "H4_CFG4", "json engine keys at <= 4 rows"),
    Knob(
        "fused_engine_cfg_5_8_rows",
        "H4_CFG8",
        'json engine keys at 5..8 rows (TP2 default {"early0": 0})',
    ),
    Knob(
        "fused_allreduce_in",
        "H4_ARIN",
        "1: the engine reduces the attention partials in its prologue (unshipped)",
    ),
    Knob(
        "fused_phase_lag",
        "M3_MOE_LAG",
        "FC2 lag in groups behind FC1 (2), read at fused/host.py import",
    ),
    Knob(
        "fused_state_cfg_overlay",
        "M3_G1_CFG",
        "json applied last to every fused engine config (debug)",
    ),
    # expert_local engine build
    Knob("expert_local_push", "H7_PUSH", "1: the finalize pushes its rows (default)"),
    Knob(
        "expert_local_base_cfg",
        "H9_C2_BASECFG",
        "json replacing the shipped expert_local base config",
    ),
    Knob("expert_local_extra_cfg", "H7_XLX_CFG", "json engine keys over the base config"),
    Knob("expert_local_slot_order_cfg", "H9_X3_CFG", 'json over that (default {"xl_sord": 1})'),
    Knob(
        "expert_local_per_token_cfg",
        "PTC2_CFG",
        "json over the per-token engine config (debug keys)",
    ),
    Knob(
        "expert_local_phase_lag", "H7_M16_LAG", "FC2 lag (3), read at expert_local/host.py import"
    ),
    Knob(
        "expert_local_state_cfg_overlay",
        "M3_G1_CFG16",
        "json applied to every expert_local config (debug); the "
        'factories hold it at "{}" while they build',
        written=True,
    ),
    Knob(
        "expert_local_tdbg_host_mapped",
        "XLD_TDBG_MAPPED",
        "1: host-mapped tdbg, readable after a device fault",
    ),
    Knob("die_table_map", "XL_DIEMAP", "explicit SM -> die table (diagnostics)"),
    Knob("die_table_no_probe", "XL_NOPROBE", "1: no uGPU probe, die_table_map only (diagnostics)"),
    # shared
    Knob(
        "routed_inter_per_rank",
        "M3_G1_ITP",
        "routed intermediate per TP rank; set by the factories, read at engine host import",
        written=True,
    ),
    Knob(
        "debug_topk_ids",
        "M3MINI_DBG_TOPK",
        "1: copy the consumed top-4 ids to blk.dbg_topk_ids after every call",
    ),
    Knob(
        "hoist_verify",
        "H9_HOIST_VERIFY",
        "1: a learning arm compares its first table-fed output bitwise (default)",
    ),
    Knob("hoist_strict", "H9_HOIST_STRICT", "1: raise on that mismatch"),
    Knob(
        "hoist_debug",
        "H9_HOIST_DEBUG",
        "1: sync after every eager call and check routes and call time",
    ),
    Knob("router_rx_check", "H9_RA_CK", "router GEMV rx check mode (1; 2 with H9_RH_HANG_DBG)"),
    Knob("router_timed", "H9_RA_TIMED", "1: the router GEMV stamps its timeline"),
)
BY_NAME = {k.name: k for k in TABLE}
BY_ALIAS = {k.alias: k for k in TABLE}


def knob(name: str, default: str | None = None) -> str | None:
    """The value bound under the descriptive name or its alias; default when neither is bound."""
    k = BY_NAME[name]
    new, old = KNOBS.get(k.env), KNOBS.get(k.alias)
    if new is not None and old is not None and new != old:
        raise KnobConflict(f"{k.env}={new!r} and its alias {k.alias}={old!r} differ")
    v = new if new is not None else old
    return default if v is None else v


def require(name: str) -> str:
    """knob(name), raising KeyError(alias) when unbound (the legacy KNOBS[alias] behaviour)."""
    v = knob(name)
    if v is None:
        raise KeyError(BY_NAME[name].alias)
    return v


__all__ = ["BY_ALIAS", "BY_NAME", "Knob", "PREFIX", "TABLE", "knob", "require"]
