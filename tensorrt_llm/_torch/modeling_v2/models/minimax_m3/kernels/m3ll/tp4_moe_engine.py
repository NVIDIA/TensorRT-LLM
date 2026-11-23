# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP4_MOE_ENGINE: the reference fused MoE engine on a target's own m3_moe plan.

The engine is the low-latency reference's fused engine (``moe_engine_kernel.py``, vendored byte for byte). Its kernel
is a superset of the targets' m3_moe lineage: with its arithmetic keys at the m3_moe values (``router_vllm`` 0: the
router GEMV in m3_moe's order; ``silu_vllm`` 0: the shared expert's SiLU in m3_moe's form) and the multicast push of
set A (``ar_g3`` / ``ar_v3`` 0: the ``ar_push_desc`` layout the targets' boundary all-reduce and front land), every
routed and shared sum runs in m3_moe's order. :func:`engine_config` therefore starts from the target's m3_moe state
config and keeps its CTA split (routed CTAs, 24 shared CTAs) and its phase plan (the FC1 split-unit boundaries), and
adds the reference's scheduling keys only:

* B2LAG (rows > 4): FC2 runs six groups behind FC1 (``lag`` 6) on the kernel's 8-deep ``fc1_done`` ring (``fc1d``
  8 >= LAG + 2 and >= the 8 groups of an 8-row call: every group its own barrier, no phase aliasing; the idle-CTA hang
  of lag > 3 on m3_moe's 4-slot ring cannot occur), two FC2 B buffers (``b2buf`` 2), 6 ring stages. The plan stays
  m3_moe's (lag 2), as the reference ships it (``fused_die_local_host._dev_b2lag``).
* the FC1 -> FC2 hand-off through global B images (``b2img``) with the FC2 scale atoms by warp 6 (``sf2_w6``);
* rolled FC2 k-tile loops (``roll_fc2``);
* the shared CTAs' PDL trigger after their own wait and the routed CTAs' release once routing is done
  (``sh_trig`` / ``early_trig`` / ``late_trigger``) where m3_moe releases at teardown or at the producer tail.

Not taken, because they change a summation order (so the engine would no longer be bit for bit m3_moe's):

* die-local roles (``dl`` / ``sharded``): the per-die plan (``make_phase_plan_dl``) moves the FC1 split-unit
  boundaries, and a split unit's FC1 sum is head + tail of per-quarter sums (moe_engine_kernel.py:4569-4606) where an
  unsplit unit sums its quarters once;
* 48 shared CTAs (the reference's ``nsh`` 48): 16 intermediate rows per shared CTA split gate_up's K into 4 slices
  (``SH_SPLIT``, moe_engine_kernel.py:219) where 24 CTAs split it into 2.

Widths: the engine kernel takes the routed / shared intermediate as trace-time keys (``i_tp`` / ``sh_i``, 768 at TP4,
1536 at TP2) and the host module sizes its buffers from a module constant, so
:func:`host_module` gives one host copy per width (the TP2 target, TP2WC_MOE_ENGINE, builds at 1536; TP4 keeps the
module imported normally, unchanged).

The m3_moe keys the reference kernel does not have are dropped (``OURS_ONLY_KEYS``: producer-tail release, the pushed
call's dead ``out`` store skip, the heavy-CTA tail release, the 2 s spin trap, the lag guard override, E0@m4); all
are scheduling or diagnostics.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

TP4_MOE_ENGINE_ATTR = "m31_tp4_moe_engine"
SHARED_CTAS = 24  # capacity: fixed: m3_moe's shared-expert CTAs (bitwise needs the same split)

# m3_moe config keys the reference kernel has no reader for (scheduling / diagnostics only)
OURS_ONLY_KEYS = ("prod_trigger", "out_skip", "tail_trigger", "hang_ms", "lag_unsafe", "e0_m4")
# the reference kernel's arithmetic keys, at m3_moe's arithmetic
ARITH_KEYS = {"router_vllm": 0, "silu_vllm": 0}
# keys that would change a summation order or the push layout: never set here
NOT_TAKEN_KEYS = (
    "dl",
    "sharded",
    "nsh",
    "sh_sf_off",
    "sh_full_stride",
    "ar_g3",
    "ar_v3",
    "ar_in",
    "ext_route",
    "b1_x4",
    "fc1_per_token",
    "pt5",
    "pw2",
    "pw2f",
    # the m3_moe MOE_HANDOFF keys (tagged-record hand-off scheduling the reference kernel does not have): the engine is
    # built on a state without them
    "vpoll",
    "epi_vskip",
    "hvpf",
)
B2LAG_MIN_TOKENS = 5  # capacity: fixed: the reference's 5..8-row B2LAG / B2IMG configs
B2LAG_LAG = 6
B2LAG_FC1D = 8
B2LAG_B2BUF = 2
B2LAG_STAGES = (6, 5)  # ring stages tried in order (the shared-memory budget decides)
HIDDEN = 6144
STAGE_BYTES_SFA_CP = (
    128 * 128 + 4 * 512
)  # one 18 KB ring stage (A tile + FC1 scale atoms) with sfa_cp

HOST_INTER_DEFAULT = (
    768  # capacity: fixed: the width the host module imported normally is built for (TP4)
)
HOST_INTERS = (
    768,
    1536,
)  # capacity: fixed: the kernel's admitted widths that a target builds (TP4, TP2)

_lock = threading.Lock()
_engaged: set = set()
_hosts: Dict[int, object] = {}


def host_module(inter: int = HOST_INTER_DEFAULT):
    """The engine host module for routed / shared intermediate ``inter`` per rank: the package's module at 768, else
    one private copy per width with ``M3_HOST_INTER`` preset before it executes (moe_engine_host vendoring edit 6), so
    its buffer shapes, shared-expert views and CTA tables are the width's. Raises ValueError outside HOST_INTERS."""
    inter = int(inter)
    if inter not in HOST_INTERS:
        raise ValueError(
            f"m3ll moe engine: no host for routed intermediate {inter} (built: {HOST_INTERS})"
        )
    if inter == HOST_INTER_DEFAULT:
        from . import moe_engine_host as host

        return host
    with _lock:
        mod = _hosts.get(inter)
        if mod is None:
            path = Path(__file__).resolve().parent / "moe_engine_host.py"
            name = f"{__package__}.moe_engine_host_i{inter}"
            spec = importlib.util.spec_from_file_location(name, str(path))
            mod = importlib.util.module_from_spec(spec)
            mod.M3_HOST_INTER = inter
            sys.modules[name] = mod
            try:
                spec.loader.exec_module(mod)
            except BaseException:
                sys.modules.pop(name, None)
                raise
            if int(mod.INTER) != inter or int(mod.SHARED_INTER) != inter:
                raise RuntimeError(
                    f"m3ll moe engine: host copy {name} built INTER {mod.INTER}, not {inter}"
                )
            _hosts[inter] = mod
        return mod


def default_fc1d(lag: int) -> int:
    """The kernel's default fc1_done ring depth for ``lag`` (moe_engine_kernel.py FC1D)."""
    return max(4, 1 << (lag + 1).bit_length())


def check_ring(config: dict) -> None:
    """The fc1_done ring of ``config`` cannot alias a phase: a power of two > LAG + 1 (the kernel's own assert) and,
    past
    m3_moe's lag, at least one slot per group of a call (NG_CAP = m_max), so every group has its own barrier."""
    lag = int(config.get("lag", 2))
    m = int(config["m_max"])
    depth = int(config.get("fc1d", default_fc1d(lag)))
    if depth & (depth - 1) or depth <= lag + 1 or depth > 32:
        raise ValueError(
            f"tp4 moe engine: fc1_done depth {depth} must be a power of two > LAG + 1 = {lag + 1}"
        )
    if lag > 2 and depth < m:
        raise ValueError(
            f"tp4 moe engine: lag {lag} needs fc1_done depth >= {m} groups (got {depth}): a CTA without FC1 work in "
            "a group can otherwise see one slot complete twice"
        )


def engine_config(
    ours: dict,
    *,
    b2lag: bool = True,
    early_trig: bool = True,
    stages: Optional[int] = None,
    ours_xstage: Optional[bool] = None,
) -> dict:
    """The reference engine's trace-time config for a target m3_moe state config ``ours`` (see the module docstring).

    ``ours_xstage``: whether the m3_moe module stages the hidden rows in its ring (its XSTAGE after its own fit rule:
    m3_moe turns XSTAGE off when M x 6144 x 2 bytes exceed the ring, e.g. 8 rows on its 5-stage ring; the router
    GEMV then reads them from global, same math). The reference kernel asserts instead of falling back, so ``xstage`` is
    set to 0 explicitly wherever the ring cannot hold the rows (B2LAG off keeps m3_moe's 5 stages at 8 rows; the
    B2LAG stage fallback 5) or m3_moe runs without it on the same ring."""
    bad = [k for k in NOT_TAKEN_KEYS if ours.get(k)]
    if bad:
        raise ValueError(
            f"tp4 moe engine: the m3_moe config carries {bad}, which this port never takes"
        )
    if not ours.get("fused_router") or not ours.get("shared"):
        raise ValueError("tp4 moe engine: needs a fused-router m3_moe state with the shared expert")
    cfg = {k: v for k, v in ours.items() if k not in OURS_ONLY_KEYS}
    cfg.update(ARITH_KEYS)
    cfg["roll_fc2"] = 1
    m = int(cfg["m_max"])
    if m >= B2LAG_MIN_TOKENS:
        cfg.update({"b2img": 1, "sfa_cp": 1, "sf2_w6": 1})
        if b2lag:
            cfg.update({"lag": B2LAG_LAG, "fc1d": B2LAG_FC1D, "b2buf": B2LAG_B2BUF})
            cfg["stages"] = B2LAG_STAGES[0] if stages is None else stages
        else:
            cfg["b2buf"] = 1
            cfg.pop("lag", None)
            if stages is not None:
                cfg["stages"] = stages
        if early_trig and int(cfg.get("late_trigger", 1)):
            cfg.update({"late_trigger": 1, "early_trig": 1, "sh_trig": 1})
    kept_ring = "stages" not in cfg or cfg.get("stages") == ours.get("stages")
    if "stages" in cfg and m * HIDDEN * 2 > int(cfg["stages"]) * STAGE_BYTES_SFA_CP:
        cfg["xstage"] = (
            0  # the rows do not fit the ring: read from global (m3_moe's own fallback, same math)
        )
    elif kept_ring and ours_xstage is not None and not ours_xstage:
        cfg["xstage"] = 0  # m3_moe's ring, m3_moe's choice
    check_ring(cfg)
    return cfg


def state_inter(ours_state) -> int:
    """The routed intermediate per rank ``ours_state`` (an m3_moe state) is built for; its shared expert must match."""
    cfg = ours_state.config
    inter = int(cfg.get("i_tp", HOST_INTER_DEFAULT))
    if cfg.get("shared") and int(cfg.get("sh_i", HOST_INTER_DEFAULT)) != inter:
        raise ValueError(
            f"m3ll moe engine: shared intermediate {cfg.get('sh_i')} != routed {inter} (the host ties them)"
        )
    return inter


def build_state(device, ours_state, *, b2lag: bool = True, early_trig: bool = True, die_tab=None):
    """The reference engine's state on ``ours_state``'s (an m3_moe ``M3MoeState``) CTA split and phase plan, at its
    width (:func:`host_module`). Tries the B2LAG ring stages in order; raises ValueError when no config fits.

    ``die_tab`` (DIE_LOCAL, ``die_local.prepare``): an int32 ``%smid`` -> role table, a bijection of the
    SMs onto the routed + shared roles; the kernel reads its role from it (trace key ``dl_tab``, vendoring edit 8) and
    the state's plan buffer carries it after the unchanged plan. The plan and every sum stay m3_moe's."""
    host = host_module(state_inter(ours_state))
    tab = None
    if die_tab is not None:
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        roles = int(ours_state.num_ctas) + SHARED_CTAS
        tab = die_tab.to(device=device, dtype=torch.int32).reshape(-1)
        if (
            roles != sms
            or tab.numel() != sms
            or not torch.equal(tab.sort().values.cpu(), torch.arange(roles, dtype=torch.int32))
        ):
            raise ValueError(
                f"tp4 moe engine: die table is not a bijection of {sms} SMs onto {roles} roles"
            )

    if getattr(ours_state.mod, "NSH", SHARED_CTAS) != SHARED_CTAS:
        raise ValueError(
            f"tp4 moe engine: m3_moe runs {ours_state.mod.NSH} shared CTAs, the port keeps {SHARED_CTAS}"
        )
    m = int(ours_state.m)
    tries = B2LAG_STAGES if (b2lag and m >= B2LAG_MIN_TOKENS) else (None,)
    errors = []
    for stages in tries:
        cfg = engine_config(
            ours_state.config,
            b2lag=b2lag,
            early_trig=early_trig,
            stages=stages,
            ours_xstage=bool(getattr(ours_state.mod, "XSTAGE", True)),
        )
        if tab is not None:
            cfg.update({"dl_tab": 1, "dl_tab_off": int(ours_state.plan.numel())})
        try:
            state = host.M3MoeState(
                device,
                num_tokens=m,
                inter=int(host.INTER),
                num_ctas=ours_state.num_ctas,
                shared=True,
                n_shared=SHARED_CTAS,
                base_cfg=cfg,
                plan=(ours_state.plan, int(ours_state.config["nb2_cap"])),
            )
        except ValueError as err:  # the shared-memory / TMEM budget of this ring depth
            errors.append(f"stages {stages}: {err}")
            continue
        if int(state.mod.FC1D) <= int(state.mod.LAG) + 1 or (
            int(state.mod.LAG) > 2 and int(state.mod.FC1D) < m
        ):
            raise ValueError(f"tp4 moe engine: module ring {state.mod.FC1D} at lag {state.mod.LAG}")
        if tab is not None:
            if not getattr(state.mod, "DL_TAB", False) or int(state.plan.numel()) != int(
                state.mod.DL_TAB_OFF
            ):
                raise ValueError(
                    "tp4 moe engine: the kernel module did not take dl_tab after the plan"
                )
            state.plan = torch.cat([state.plan.reshape(-1), tab])
        return state
    raise ValueError("tp4 moe engine: no config fits: " + "; ".join(errors))


def layer(state, *tensors, **kwargs):
    """One layer on ``state`` (``moe_engine_host.M3MoeState.layer``), tagged as this port's engine."""
    lay = state.layer(*tensors, **kwargs)
    setattr(lay, TP4_MOE_ENGINE_ATTR, True)
    return lay


def is_engine(engine) -> bool:
    """Whether ``engine`` (a layer a target's engine lookup returned) is this port's."""
    return bool(getattr(engine, TP4_MOE_ENGINE_ATTR, False))


def workspace_bytes(state) -> int:
    """Device bytes a state allocates (plan + call workspace); the layers add no weight copies (the expert buffers
    are read in place; the SFA_BULK image is shared with m3_moe)."""
    total = 0
    for name in (
        "plan",
        "partials",
        "hbuf32",
        "epochs",
        "route_dbg",
        "tdbg",
        "lgbuf",
        "_dummy",
        "_ar_off",
        "img32",
        "cnt",
        "hx",
        "dnx",
    ):
        t = getattr(state, name, None)
        if isinstance(t, torch.Tensor):
            total += t.numel() * t.element_size()
    return total


def log_engaged(
    target: str,
    rank: int,
    device,
    rows: int,
    state,
    ours_state=None,
    tag: str = "M31_TP4_MOE_ENGINE",
) -> None:
    """One engagement line per (target, rank, rows, lag, early trigger) on stderr: ``[<tag>] engaged ...``
    (a kill switch that changes the build prints its own line). The TP2 target tags ``M31_TP2WC_MOE_ENGINE``."""
    key = (target, rank, rows, int(state.mod.LAG), int(bool(state.config.get("early_trig", 0))))
    with _lock:
        if key in _engaged:
            return
        _engaged.add(key)
    cfg = state.config
    ours_ws = workspace_bytes(ours_state) if ours_state is not None else 0
    print(
        f"[{tag}] engaged: target {target} rank {rank} device {device} rows {rows} "
        f"inter {int(state.config.get('i_tp', HOST_INTER_DEFAULT))} "
        f"ctas {state.num_ctas}+{state.mod.NSH} lag {int(state.mod.LAG)} fc1d {int(state.mod.FC1D)} "
        f"b2buf {cfg.get('b2buf', '-')} stages {cfg.get('stages', '-')} b2img {cfg.get('b2img', 0)} "
        f"sf2_w6 {cfg.get('sf2_w6', 0)} roll_fc2 {cfg.get('roll_fc2', 0)} early_trig {cfg.get('early_trig', 0)} "
        f"sh_trig {cfg.get('sh_trig', 0)} dl {'tab' if cfg.get('dl_tab') else 0} | "
        f"workspace {workspace_bytes(state) / 2**20:.2f} MiB "
        f"(m3_moe {ours_ws / 2**20:.2f} MiB), weight copies 0",
        file=sys.stderr,
        flush=True,
    )


def engaged() -> Tuple[Tuple[str, int, int, int, int], ...]:
    """The (target, rank, rows, lag, early trigger) keys that printed an engagement line in this process."""
    with _lock:
        return tuple(sorted(_engaged))


__all__ = [
    "HOST_INTERS",
    "ARITH_KEYS",
    "B2LAG_FC1D",
    "B2LAG_LAG",
    "NOT_TAKEN_KEYS",
    "OURS_ONLY_KEYS",
    "SHARED_CTAS",
    "build_state",
    "check_ring",
    "engaged",
    "engine_config",
    "host_module",
    "is_engine",
    "layer",
    "log_engaged",
    "state_inter",
    "workspace_bytes",
]
