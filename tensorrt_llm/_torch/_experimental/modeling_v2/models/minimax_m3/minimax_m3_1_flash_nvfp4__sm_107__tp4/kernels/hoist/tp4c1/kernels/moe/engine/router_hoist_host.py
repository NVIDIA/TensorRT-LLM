# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Router hoist: the post publishes each token's top-4 route into tagged route lines (RouteBuf) and the engine polls
them before its grid wait. Builds the fused (_c1_build) and expert_local (_c2_build) engines on those routes,
the route-reading post adapters, and the host-side route checks."""

from __future__ import annotations

import atexit
import contextlib
import importlib
import inspect
import json
import os
import sys
import time
from types import SimpleNamespace

import torch

from ....runtime.knobs import KNOBS as _K
from ..knobs import knob
from . import router_hoist_weights as hw

REPL = int(_K.get("H9_RH_REPL", "16"))  # route-line replicas: engine CTA bx polls replica bx % REPL
POLL_NS = int(_K.get("H9_RH_POLL_NS", "64"))
DELAY_NS = int(_K.get("H9_RH_DELAY_NS", "150"))  # read at import; lbh reads the same env
EPS = 1e-6
BF16_QUARTER = 0x3E80  # bf16(0.25): seed weight of the placeholder routes
C1_PORTS = ("fused", "port_h9r1", "port_h9r2", "port_h9r3", "port_h9r4")  # r4: G-PF (r0 + pf_l2)
_st = {}
_TRACK = []  # (arm, layer_idx, t, engine state, post tdbg or None)
_VALID = []  # verify results of the learning arms
_R8 = []  # (blk, solo, h12r_ckdbg int32 [count, last oep, last rxc, 0]) of every rh o_proj


class RouteBuf:
    """Per-layer route lines [REPL][maxt][16] int64 (u64 words 0..3 used) and row counters [maxt][32] int32, both
    local to the GPU. The post (lbv3r) is the only writer of both; the engine reads the lines."""

    def __init__(self, maxt: int, dev, repl: int = REPL):
        self.maxt, self.repl = maxt, repl
        self.route = torch.zeros(repl * maxt * hw.ROUTE_LINE_U64, dtype=torch.int64, device=dev)
        self.rctr = torch.zeros(maxt * hw.RCTR_STRIDE, dtype=torch.int32, device=dev)

    def lines(self) -> torch.Tensor:
        """[repl][maxt][16] view of the route lines (the engine ops reject the flat tensor)."""
        return self.route.view(self.repl, self.maxt, hw.ROUTE_LINE_U64)

    def tag(self) -> int:
        """The last published tag (rows move in lockstep: row 0's counter)."""
        return int(self.rctr[0].item())

    def host_publish(self, ids: torch.Tensor, w: torch.Tensor):
        """Publish [t, 4] routes from the host with tag = counter + 1, exactly as the post would: keeps the post's
        counter aligned with the engine's epoch across engine-only calls (the factory's build-time compile call)."""
        t = ids.shape[0]
        tag = self.tag() + 1
        words = hw.pack_route(tag, ids.to(self.route.device), w.to(self.route.device))
        v = self.route.view(self.repl, self.maxt, hw.ROUTE_LINE_U64)
        v[:, :t, : hw.TOP_K] = words.unsqueeze(0)
        self.rctr.view(self.maxt, hw.RCTR_STRIDE)[:, 0] = tag
        return tag

    def retract(self, epochs: torch.Tensor):
        """After a call whose publish no ext engine consumed (a learning call): clear the lines so a stale tag can
        never satisfy the next poll, and set every row counter back to the engine's epoch (device ops, no sync)."""
        self.route.zero_()
        self.rctr.view(self.maxt, hw.RCTR_STRIDE)[:, 0].copy_(epochs[:1].expand(self.maxt))


def learn_table(route_dbg: torch.Tensor, t: int) -> torch.Tensor:
    """The engine's consumed routing (route_dbg [M, 8]: ids, then fp32 bits of the bf16 weights; CTA 0) as the
    skeleton table [t][4] u32 (id << 16 | bf16 bits), int32 storage."""
    ids = route_dbg[:t, :4].to(torch.int64)
    wb = (route_dbg[:t, 4:8].to(torch.int64) >> 16) & 0xFFFF
    return ((ids << 16) | wb).to(torch.int32).contiguous()


def check_route_alignment(rb: RouteBuf, epochs: torch.Tensor, ncta: int) -> None:
    """Debug assert (after a sync): every routed CTA's epoch equals the post's counter, else the next engine call
    waits for a tag that never comes (HANG_DBG stamp 57)."""
    ep = epochs[:ncta].cpu()
    if not bool((ep == rb.tag()).all()):
        raise RuntimeError(
            f"hoist: route counter {rb.tag()} != engine epochs {ep.unique().tolist()}"
        )


def route_giveup_rows(route_dbg: torch.Tensor, t: int) -> list[int]:
    """Rows of the consumed routing (route_dbg, CTA 0's view) that are the route poll's give-up fallback: experts 0..3
    with all four weights 0. Real routing never yields four zero weights (they sum to the route scale)."""
    rd = route_dbg[:t].cpu()
    ids = torch.arange(hw.TOP_K, dtype=rd.dtype)
    return [
        i
        for i in range(rd.shape[0])
        if not bool(rd[i, 4:8].any()) and torch.equal(rd[i, :4].sort().values, ids)
    ]


def check_route_consumed(st, t: int, who: str) -> None:
    """Raise if CTA 0 consumed the give-up fallback route for any row (the engine would have served weight-0
    experts)."""
    rows = route_giveup_rows(st.route_dbg, t)
    if rows:
        raise RuntimeError(
            f"hoist {who}: the route poll gave up on rows {rows} and fell back to experts 0..3 at weight "
            "0 (no post published this call's route lines)"
        )


WAIT_GIVEUP_S = 1.0  # an engine call this long has hit a bounded poll's cap (XL _wait falls through after HANG_CAP)


def check_call_time(seconds: float, who: str) -> None:
    if seconds > WAIT_GIVEUP_S:
        raise RuntimeError(
            f"hoist {who}: one engine call took {seconds:.2f} s; a bounded device wait gave up and the "
            "call continued on stale state"
        )


def _seed_routes(t, dev):
    """Placeholder routes for engine calls no post feeds (build-time compile call): experts 0..3, weight 0.25."""
    return torch.arange(hw.TOP_K, device=dev).repeat(t, 1), torch.full(
        (t, hw.TOP_K), 0.25, device=dev
    )


def _seed_table(t, dev):
    ids = torch.arange(hw.TOP_K, device=dev, dtype=torch.int32).repeat(t, 1)
    return ((ids << 16) | BF16_QUARTER).contiguous()


def _rank_world():
    import torch.distributed as dist

    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _bucket(t):
    return 8 if t <= 8 else 16 if t <= 16 else 48


# ----------------------------------------------------------------------------- engine build adapters
class _LayerProxy:
    """run()-kwarg form of the route binding: lay(hidden, out, arp) -> state.run(lay, ..., **kw)."""

    def __init__(self, lay, kw):
        self._lay, self._kw = lay, kw

    def __call__(self, hidden, out=None, arp=None):
        return self._lay.state.run(self._lay, hidden, out, arp=arp, **self._kw)

    def __getattr__(self, k):
        return getattr(self._lay, k)


@contextlib.contextmanager
def _route_kwargs(cls, on_layer=None, **kw):
    """Bind route_lines / rt_tab when the factory creates the layer, so its build-time compile call already reads them:
    op.layer(...) kwargs if the port takes them there (r0), else run(...) (x0's op16). on_layer(state) runs right
    after the state's layer() returns, i.e. after the state exists and before the factory's compile call."""
    kw = {k: v for k, v in kw.items() if v is not None}
    if not kw:
        yield
        return
    orig = cls.layer
    at_layer = all(k in inspect.signature(orig).parameters for k in kw)
    if not at_layer and not all(k in inspect.signature(cls.run).parameters for k in kw):
        raise RuntimeError(
            f"hoist: {cls.__module__}.{cls.__name__} takes {sorted(kw)} at neither layer() nor run(): "
            "wrong engine port (H4_DL_PORT / H9_C2_PORT)?"
        )

    def layer(self, *a, **k):
        lay = orig(self, *a, **k, **kw) if at_layer else _LayerProxy(orig(self, *a, **k), kw)
        if on_layer is not None:
            on_layer(self)
        return lay

    cls.layer = layer
    try:
        yield
    finally:
        cls.layer = orig


def _itp(blk):
    """op.py reads M3_G1_ITP (the routed intermediate per rank) once, at import; the factories set it only after this
    module has loaded the port, so an unset value would build the TP8 (768) constants in a TP2 run."""
    inter = getattr(getattr(blk, "mw", None), "inter", None)
    if inter:
        _K.setdefault("M3_G1_ITP", str(inter))


def _c1_port(want=C1_PORTS):
    from . import fused_die_local_host as dl

    name = os.path.basename(os.path.normpath(dl.DL_PORT))
    if name not in want:
        raise RuntimeError(
            f"h9 hoist: needs H4_DL_PORT=$C/H9-megakernel/w6/{'|'.join(want)} (got {dl.DL_PORT})"
        )
    return dl


def _c1_build(blk, reg, cfg, *, shared_ctas_le4=None, **route):
    """The shipped fused engine (fused_die_local_host.build_shipped) with extra engine cfg and route bindings."""
    dl = _c1_port()
    _itp(blk)
    op = dl._op_dl()
    with _route_kwargs(op.M3MoeState, **route):
        return dl.build_shipped(blk, reg, cfg, shared_ctas_le4=shared_ctas_le4)


def _c2_classes(mod, port):
    """The engine state classes the factory module loaded from the engine package at `port`."""
    if not port:
        raise RuntimeError("h9 hoist C2: no engine package dir")
    port = os.path.realpath(port)
    for fn in (
        "_op",
        "_op16",
        "_ops",
    ):  # load the engine package before the build so the classes can be bound
        if callable(getattr(mod, fn, None)):
            getattr(mod, fn)()
            break
    out = []
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None) or ""
        if f and os.path.realpath(f).startswith(port + os.sep):
            out += [
                getattr(m, c)
                for c in ("M3MoeState", "M3Moe16State")
                if isinstance(getattr(m, c, None), type)
            ]
    return out


def _c2_build(blk, reg, cfg, *, factory, port, **route):
    """An expert_local engine factory ("module:attr") built with cfg over expert_local_extra_cfg, and the route bindings
    on the state classes of its engine package dir `port`."""
    mname, attr = factory.split(":")
    _itp(blk)
    mod = importlib.import_module(mname)
    classes = _c2_classes(mod, port) if route else []
    if route and not classes:
        raise RuntimeError(f"h9 hoist C2: no engine state class loaded from {port}")
    extra = {**json.loads(knob("expert_local_extra_cfg") or "{}"), **cfg}
    with contextlib.ExitStack() as es:
        for c in classes:
            es.enter_context(_route_kwargs(c, **route))
        return getattr(mod, attr)(blk, extra_cfg=extra)


def _state_c1(blk):
    return blk.h4_state


def _state_c2(blk):
    return getattr(blk, "h7_state", None) or blk.h4_state


def _hold(blk):
    """The push-hold dict of the engine factory that last registered blk.ffn_out (its fn pushes with hold["arp"])."""
    for name in (
        "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4.kernels.hoist.tp4c1.kernels.moe.engine.fused_host",
        "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4.kernels.hoist.tp4c1.kernels.moe.engine.expert_local_host",
    ):
        mod = sys.modules.get(name)
        if mod is None or not hasattr(mod, "_push_reg"):
            continue
        reg = mod._push_reg().get(id(blk.ffn_out))
        if reg is not None and reg[2] is blk.ffn_out:
            return reg[0]
    return None


def _untrack_k3(st):
    """Keep plugins.h4.k3's h4-tdbg dump to the engines the graph runs (a learning engine stays out of it)."""
    mod = sys.modules.get(
        "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4.kernels.hoist.tp4c1.kernels.moe.engine.fused_host"
    )
    lst = getattr(mod, "_st", {}).get("states", []) if mod is not None else []
    for i, s in enumerate(lst):
        if s is st:
            del lst[i]
            return


# ----------------------------------------------------------------------------- engine factories (generic cores)
def _check_pairing(blk, who):
    """o_proj rh <-> post_rh on the same blk, both directions (the o_proj factory is built first, the post second)."""
    op_rh = getattr(blk, "h9_oproj_rh", None)
    post_rh = getattr(blk, "h9_post_rh", False)
    if op_rh is not None and not op_rh.solo and not post_rh:
        raise RuntimeError(
            f"h9 hoist {who} L{blk.layer_idx}: o_proj router role without make_post_rh on this layer "
            "(an unpaired rh o_proj; tier C must use make_oproj_rh_solo)"
        )
    if post_rh and (op_rh is None or op_rh.solo):
        raise RuntimeError(
            f"h9 hoist {who} L{blk.layer_idx}: make_post_rh without make_oproj_rh on this layer"
        )


def _check_dl_grid(st):
    """R11: roles by SM (C1 dl, C2 XL die roles) need one CTA per SM: grid == SM count, and the
    die split needs SM count % 4 == 0."""
    cfg = getattr(st, "config", {}) or {}
    if not (cfg.get("dl") or cfg.get("xl_smid") or cfg.get("xl_s0")):
        return
    mod = st.mod
    grid = int(st.num_ctas) + int(getattr(mod, "NXTRA", getattr(mod, "NSH", 0)))
    sms = torch.cuda.get_device_properties(st.device).multi_processor_count
    if grid != sms or sms % 4:
        raise RuntimeError(
            f"h9 hoist R11: DL roles need grid ({st.num_ctas} + {grid - int(st.num_ctas)}) == SM count "
            f"({sms}) and SM count % 4 == 0"
        )


def _plain(blk, arm, build, state_of, cfg):
    _check_pairing(blk, arm)
    fn = build(blk, f"blks_h9h{arm}", cfg)
    _check_dl_grid(state_of(blk))
    _TRACK.append((arm, blk.layer_idx, blk.t, state_of(blk), None))
    return fn


def _learner(blk, arm, fa, sa, ha, fb, sb, hb, tab, rb=None):
    """Call 0 (eager): the landed engine runs (pushing with the captured engine's arp) and its consumed routing becomes
    the table; with a RouteBuf the unconsumed publish is retracted. Call 1: the table engine, its output compared
    bitwise against call 0's (inputs repeat every step). Later calls: the table engine only."""
    st = {"n": 0, "fa": fa}
    verify = knob("hoist_verify", "1") == "1"
    strict = knob("hoist_strict", "0") == "1"
    debug = knob("hoist_debug", "0") == "1"
    rank = _rank_world()[0]

    def fn():
        n = st["n"]
        st["n"] = n + 1
        cap = torch.cuda.is_current_stream_capturing()
        if n == 0:
            if cap:
                raise RuntimeError(
                    "h9 hoist: the table engine must learn on an eager call before capture"
                )
            if ha is not None and hb is not None:
                ha["arp"] = hb["arp"]  # push exactly where the captured engine will
            st["fa"]()
            tab.copy_(learn_table(sa.route_dbg, blk.t))
            if rb is not None:
                rb.retract(sb.epochs)
            if verify:
                st["ref"] = blk.ffn_out.clone()
            return
        if n == 1:
            st["fa"] = None  # free the learning engine's state and layer
            ref = st.pop("ref", None)
            if ref is not None and not cap:
                blk.ffn_out.fill_(
                    float("nan")
                )  # a table engine that writes nothing must not pass on the old output
                fb()
                ok = bool(torch.equal(blk.ffn_out, ref))
                _VALID.append(
                    {"arm": arm, "layer": blk.layer_idx, "t": blk.t, "rank": rank, "bitexact": ok}
                )
                print(
                    f"H9_HOIST_VERIFY r{rank} {arm} L{blk.layer_idx} t{blk.t}: table engine vs landed "
                    f"{'bit-exact' if ok else 'MISMATCH'}",
                    flush=True,
                )
                if strict and not ok:
                    raise RuntimeError(
                        f"h9 hoist {arm} L{blk.layer_idx}: table engine output != landed engine output"
                    )
                if debug and rb is not None:
                    torch.cuda.synchronize()
                    check_route_alignment(rb, sb.epochs, sb.num_ctas)
                return
        fb()
        if debug and rb is not None and not cap:
            torch.cuda.synchronize()
            check_route_alignment(rb, sb.epochs, sb.num_ctas)

    return fn


def _align_for_compile(rb, st, blk):
    """Before the factory's compile call: read the new state's routed epochs (must be uniform), clear the lines, set the
    counter to that epoch and publish once, so the call finds tag = epoch + 1 however often the post published
    before."""
    n = int(getattr(st, "num_ctas", st.epochs.numel()))
    ep = st.epochs[:n].cpu()
    if ep.numel() == 0 or not bool((ep == ep[0]).all()):
        raise RuntimeError(
            f"h9 hoist L{blk.layer_idx}: engine epochs not uniform before the compile call "
            f"{ep.unique().tolist()}: cannot align the route counter"
        )
    rb.route.zero_()
    rb.rctr.view(rb.maxt, hw.RCTR_STRIDE)[:, 0] = int(ep[0])
    tag = rb.host_publish(*_seed_routes(blk.t, blk.normed2.device))
    if tag != int(ep[0]) + 1:
        raise RuntimeError(
            f"h9 hoist L{blk.layer_idx}: seeded tag {tag} != engine epoch {int(ep[0])} + 1"
        )


def _ext(blk, arm, build, state_of, extra=None):
    _check_pairing(blk, arm)
    rh = getattr(blk, "h9_hoist", None)
    if rh is None:
        raise RuntimeError(
            f"h9 hoist {arm}: this layer's boundary_post is not an lbv3r post (plugins.h9.hoist:"
            "make_post_rh): nothing would publish the route lines and the engine hangs"
        )
    rb = rh.rb
    fa = sa = ha = None
    if rh.tab is not None:
        fa = build(blk, f"blks_h9h{arm}l", dict(extra or {}))
        sa, ha = state_of(blk), _hold(blk)
        _check_dl_grid(sa)
    cfg = dict(
        extra or {},
        ext_route=1,
        xstage=0,
        pad_x=1,
        route_repl=rb.repl,
        route_poll_ns=POLL_NS,
        route_maxt=rb.maxt,
    )
    fb = build(
        blk,
        f"blks_h9h{arm}x",
        cfg,
        route_lines=rb.lines(),
        rctr=rb.rctr,  # R9: the engine's post-wait check
        on_layer=lambda st: _align_for_compile(rb, st, blk),
    )
    sb, hb = state_of(blk), _hold(blk)
    _check_dl_grid(sb)
    check_route_alignment(
        rb, sb.epochs, sb.num_ctas
    )  # the compile call consumed exactly the seeded publish
    check_route_consumed(
        sb, blk.t, arm
    )  # the seeded weights are 0.25, so a zero-weight row is a give-up
    from ...comm.allreduce_norm_quant import router_exchange as lbh

    if hasattr(
        lbh, "check_x4_pair"
    ):  # K-POST: an X4 post needs a b1_x4 engine and back (b1x4.py checks the tensors)
        lbh.check_x4_pair(blk, bool(getattr(sb, "b1_x4", 0)))
    _TRACK.append((arm, blk.layer_idx, blk.t, sb, rh.tdbg))
    if fa is None:
        if knob("hoist_debug", "0") != "1":
            return fb

        def fn_dbg():
            eager = not torch.cuda.is_current_stream_capturing()
            if eager:
                torch.cuda.synchronize()
                t0 = time.monotonic()
            fb()
            if eager:
                torch.cuda.synchronize()
                check_call_time(time.monotonic() - t0, arm)
                check_route_alignment(rb, sb.epochs, sb.num_ctas)
                check_route_consumed(sb, blk.t, arm)

        return fn_dbg
    _untrack_k3(sa)
    return _learner(blk, arm, fa, sa, ha, fb, sb, hb, rh.tab, rb)


ARRIVE_ALL = {
    "arrive_all": 1
}  # R10 (K-SAFE): routed CTAs wait for the idle arrivals; r1 asserts it off at nidle 0


# ----------------------------------------------------------------------------- post (lbv3r) and o_proj adapters
def _npart():
    return 1 + int(_K.get("H9_RH_RSPLIT", "1"))


OEP_WORDS = 192 * 32  # o_proj grid x one 128 B line per CTA


def _arm_label(blk, kind):
    """lbh debug-dump arm: H9_RH_ARM (the run's label, e.g. the preset) + post kind + this process's build index, so the
    builds of one --vs process never share a dump file."""
    key = ("armb", kind)
    if blk.layer_idx == 0 or key not in _st:
        _st[key] = _st.get(key, -1) + 1
    return f"{_K.get('H9_RH_ARM', 'rh')}-{kind}-b{_st[key]}"


def _route_state(blk, tab):
    """The layer's RouteBuf (maxt = M = the engine's route_maxt) + tier-B table; the engine factory finds it on blk."""
    rb = RouteBuf(blk.t, blk.normed2.device, REPL)
    blk.h9_hoist = SimpleNamespace(
        rb=rb, tab=tab, tdbg=None
    )  # post stamps: lbh's own H9_RH_TDBG dump
    return blk.h9_hoist


def _no_rh_oproj(blk, who):
    op_rh = getattr(blk, "h9_oproj_rh", None)
    if op_rh is not None and not op_rh.solo:
        raise RuntimeError(
            f"h9 hoist {who} L{blk.layer_idx}: this layer's o_proj is the router role (make_oproj_rh): "
            "its post must be make_post_rh"
        )


def make_post_rh(blk):
    """full trio: lbv3r polls the o_proj's tagged router partials (rx v2 slot of this blk), routes and publishes into
    this layer's RouteBuf. Pairs only with make_oproj_rh on the same blk (checked here and by the engine factory)."""
    from ...comm.allreduce_norm_quant import router_exchange as lbh

    op_rh = getattr(blk, "h9_oproj_rh", None)
    if op_rh is None or op_rh.solo:
        raise RuntimeError(
            f"h9 hoist make_post_rh L{blk.layer_idx}: the o_proj of this layer is not make_oproj_rh"
        )
    rh = _route_state(blk, None)
    blk.h9_post_rh = True
    fn = lbh.post_rh(blk, rh.rb, npart=op_rh.slot.npart, arm=_arm_label(blk, "rh"))
    assert lbh.rx_slot(blk, op_rh.slot.npart) is op_rh.slot, (
        "post_rh and o_proj rh must share one rx slot"
    )
    return fn


def _rh_weights(blk):
    """Per (checkpoint layer, rank): the o_proj router-role blob / job table / span on the GPU (built once; blk.aw /
    blk.mw are shared by every build of the same layer)."""
    rank, world = _rank_world()
    key = ("rhw", id(blk.aw), id(blk.mw), rank)
    hit = _st.get(key)
    if hit is not None and hit[0] is blk.aw:
        return hit[1]
    swz = None
    if not _st.get("swz_ok"):
        from m3block.weights import _swz

        swz = _swz
    fmt = _K.get("H9_RH_FMT", "fp32")
    hr = hw.build_from_blk(blk, rank, world, grid=192, fmt=fmt, r_split=_npart() - 1, swz_fn=swz)
    _st["swz_ok"] = True
    dev = blk.normed2.device
    jb = hr.jobs
    w = SimpleNamespace(
        blob=jb.blob.to(dev).contiguous(),
        jobs=jb.table.to(dev).contiguous(),
        span=jb.span.to(dev).contiguous(),
        maxj=jb.maxj,
        npart=jb.npart,
        fmt=fmt,
        sha=hr.sha,
    )
    _st[key] = (blk.aw, w)
    return w


def _rh_cfg(rh_w=None):
    """(rh_w, rh_r, rh_neg) from H9_RH_W (1 | 3), H9_RH_R (1 = r-jobs pre-wait | 0 = post-wait control) and H9_RH_NEG
    (1 = the stale-read red arm, needs rh_r 1); rh_w 3 needs rh_r 1 (h12r_slot asserts the same)."""
    w, r, neg = (
        int(_K.get(k, d)) for k, d in (("H9_RH_W", "1"), ("H9_RH_R", "1"), ("H9_RH_NEG", "0"))
    )
    w = w if rh_w is None else rh_w
    if w not in (1, 3):
        raise RuntimeError(
            f"h9 hoist: H9_RH_W={w}: the o_proj router role runs on 1 (warp 3) or 3 (warps 3, 0, 1) warps"
        )
    if r not in (0, 1) or neg not in (0, 1):
        raise RuntimeError(f"h9 hoist: H9_RH_R={r} / H9_RH_NEG={neg}: each 0 or 1")
    if w == 3 and not r:
        raise RuntimeError(
            "h9 hoist: H9_RH_W=3 needs H9_RH_R=1 (the 3-warp role splits the post-wait a pass only)"
        )
    if neg and not r:
        raise RuntimeError("h9 hoist: H9_RH_NEG=1 is the pre-wait r-read red arm: needs H9_RH_R=1")
    return w, r, neg


def _l_cfg(over):
    """K-OPROJ rev 7 passthrough (L-ATTN / L-HBM / L-CODE): H9_RH_PTRIG (0|1), H9_RH_XPOLL (0|1), H9_RH_XPNS (>= 0),
    H9_RH_SHP (0|1|2), H9_RH_UNROLL (>= 0); explicit factory kwargs (over) win. shp needs shp_ranges_fn (lhbm)."""
    env = {
        "ptrig": ("H9_RH_PTRIG", "0"),
        "xpoll": ("H9_RH_XPOLL", "0"),
        "xp_ns": ("H9_RH_XPNS", "0"),
        "shp": ("H9_RH_SHP", "0"),
        "rh_unroll": ("H9_RH_UNROLL", "0"),
    }
    k = {n: int(_K.get(e, d)) for n, (e, d) in env.items()}
    k.update({n: v for n, v in over.items() if v is not None})
    if (
        k["ptrig"] not in (0, 1)
        or k["xpoll"] not in (0, 1)
        or k["shp"] not in (0, 1, 2)
        or k["xp_ns"] < 0
        or k["rh_unroll"] < 0
    ):
        raise RuntimeError(
            f"h9 hoist: o_proj cfg {k}: ptrig / xpoll 0|1, shp 0|1|2, xp_ns / rh_unroll >= 0"
        )
    if k["shp"] and k.get("shp_ranges_fn") is None:
        raise RuntimeError(
            "h9 hoist: shp needs shp_ranges_fn: use plugins.h9.lhbm:make_oproj_rh_shp1 / _shp2"
        )
    return {n: v for n, v in k.items() if v or n == "shp_ranges_fn"}


def _oproj_rh(blk, solo, rh_w=None, **over):
    """rx v2: this blk's rx slot (allocated here, first, in the same factory order on every rank;
    the post reuses it), a per-layer oep call counter registered for lbh.rx_reset_all, then K-OPROJ's core with the
    slot's rxc for its R8 check (calls == consumes; traps, or counts with H9_RH_CKDBG / HANG_DBG; compiled out for tier
    C). The first eager call of the first rh o_proj of every build (R13) is an arm boundary: lbh.rx_reset_all() there
    (collective, outside capture) zeroes rx, every oep and every rxc, so build-time and unpaired calls never shift a
    pair's tags."""
    import h12r_slot

    from ...comm.allreduce_norm_quant import router_exchange as lbh
    from ...common import allreduce_push_workspace as k2c

    rh_w, rh_r, rh_neg = _rh_cfg(
        rh_w
    )  # before any allocation: a bad value must not leave a half-registered slot
    lk = _l_cfg(over)
    hang = _K.get("H9_RH_HANG_DBG", "0") == "1"
    # R8 (calls == consumes): compiled in for every matched arm (rh_ck 1; H9_RH_CKDBG 0 traps, 1 records, default 1 in
    # HANG_DBG smokes); tier C has no consuming post, so it is compiled out there (rh_ck 0)
    rh_ck = 0 if solo else 1
    ckdbg = 0 if solo else int(_K.get("H9_RH_CKDBG", "1" if hang else "0"))
    if ckdbg not in (0, 1):
        raise RuntimeError(f"h9 hoist: H9_RH_CKDBG={ckdbg}: 0 (trap) or 1 (record)")
    w = _rh_weights(blk)
    slot = lbh.rx_slot(blk, w.npart)
    assert slot.maxt == k2c._prx_spec(blk.t)[4], (slot.maxt, k2c._prx_spec(blk.t)[4])
    assert slot.npart == w.npart, (slot.npart, w.npart)
    oep = torch.zeros(OEP_WORDS, dtype=torch.int32, device=blk.normed2.device)
    lbh.register_counter(oep)
    # public: w = the router-role weights (blob / jobs / span / maxj / npart / fmt / sha), also as .blob / .jobs / .span
    blk.h9_oproj_rh = SimpleNamespace(
        solo=solo,
        oep=oep,
        slot=slot,
        w=w,
        blob=w.blob,
        jobs=w.jobs,
        span=w.span,
        rh_w=rh_w,
        rh_r=rh_r,
        rh_neg=rh_neg,
        rh_ck=rh_ck,
        rh_ckdbg=ckdbg,
        rxc=slot.rxc if rh_ck else None,
        l_cfg={n: v for n, v in lk.items() if n != "shp_ranges_fn"},
    )
    inner = h12r_slot.make_oproj_rh_core(
        blk,
        rh_blob=w.blob,
        rh_jobs=w.jobs,
        rh_span=w.span,
        rx_bases=slot.bases,
        oep=oep,
        rxc=slot.rxc if rh_ck else None,
        rh_w=rh_w,
        rh_r=rh_r,
        rh_neg=rh_neg,
        rh_ck=rh_ck,
        rh_ckdbg=ckdbg,
        **lk,
    )
    _R8.append((blk, solo, getattr(blk, "h12r_ckdbg", None)))
    st = {"n": 0}
    # R13: the arm boundary is the first tagged-buffer factory of a build (rh o_proj, K12, xpoll; the first one after
    # any of them ran), not layer_idx 0: a build whose first hoist layer is not layer 0 still resets
    boundary = _boundary_claim()

    def fn():
        _st["rh_building"] = False
        if st["n"] == 0:
            if not solo and not getattr(blk, "h9_post_rh", False):
                raise RuntimeError(
                    f"h9 hoist L{blk.layer_idx}: rh o_proj ran without make_post_rh on its layer"
                )
            if boundary:
                lbh.rx_reset_all()
        st["n"] += 1
        inner()

    return fn


def _boundary_claim():
    """R13: True for the first tagged-buffer factory of a build (none of them has run since the previous claim)."""
    first = not _st.get("rh_building")
    _st["rh_building"] = True
    return first


H9_DIR_C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")


def make_oproj_rh(blk):
    """o_proj GEMV + push, and on warp 3 the router role writing tagged fp32 partials into this blk's rx slot on every
    rank; pairs with make_post_rh on the same blk"""
    return _oproj_rh(blk, solo=False)


def make_oproj_rh_solo(blk):
    """tier C: the router role with nobody reading its rx slot (per-blk slot, so no other pair can see it); oep drifts
    ahead until the next arm boundary's rx_reset_all, which is harmless because no post waits on this slot; R8 is
    compiled out (rh_ck 0), so the arm times the router role without the check's mismatch path"""
    return _oproj_rh(blk, solo=True)


# ------------------------------------------------- stamps / hang dump (MINI_OUT + M3_TIMING=1 or M3_HANG_DBG=1)
_DURATIONS = (26, 27, 28, 30, 31)  # durations / counts, not timestamps (plugins.h4.k3._dump_tdbg)


def stamp_rows(td: torch.Tensor, n: int) -> dict:
    """plugins.h4.k3._dump_tdbg's per-stamp (median, max) over the routed CTAs of (stamp - min entry), in us."""
    td = td.cpu()
    ent = td[:n, 0][td[:n, 0] > 0]
    if ent.numel() == 0:
        return {}
    t0 = ent.min().item()
    row = {}
    for sid in range(min(64, td.shape[1])):
        if sid == 63:
            continue
        off = 0 if sid in _DURATIONS else t0
        v = sorted(td[c, sid].item() - off for c in range(n) if td[c, sid].item() > 0)
        if v:
            row[sid] = (round(v[len(v) // 2] / 1e3, 2), round(v[-1] / 1e3, 2))
    return row


def _post_rows(tdbg, maxt, t, t0_abs):
    """lbv3r tdbg ring [16 calls][maxt][4] (entry, routing start, published, CTA end): the latest call's stamps per
    row relative to the engine's min entry (negative = before the engine entered)."""
    v = tdbg.view(16, maxt, 4).cpu()
    last = int(v[:, 0, 0].argmax().item())
    return [
        [round((int(x) - t0_abs) / 1e3, 2) if int(x) > 0 else None for x in v[last, r].tolist()]
        for r in range(t)
    ]


INFO_COLS = {
    69: "r2 b2hv max re-issues",
    70: "r2 ho max poll rounds",
}  # diagnostics (TIMING / HANG_DBG), not flags
HANG_COLS = {
    57: "route give-up (routed rows)",
    62: "idle residency give-up (idle rows, r1)",
    66: "stale pre-wait epoch (routed rows; R12)",
    67: "post row counter != call tag (routed rows; R9)",
    68: "routed arrival wait give-up (routed rows; r1 arrive_all, R10)",
}
# the other G1 HANG_DBG sites (tdbg[bx, 56 + site]): C1 m3_moe 0 hx, 2 b2cnt, 3 hbuf, 4 partials, 5 dnx (any row)
HANG_COLS_C1_ANY = {
    56: "hx poll",
    58: "b2cnt poll",
    59: "hbuf records poll",
    60: "partials poll",
    61: "dnx poll",
}
# C2 m3_moe16 (TD_STRIDE 64): 56 hx, 57/58 B1 counter, 59 records + the ext_route route-word give-up, 60, 61, 62 dnx
HANG_COLS_C2 = {
    48: "stale pre-wait epoch (R12)",
    49: "post row counter != call tag (R9)",
    56: "hx poll",
    57: "B1 counter",
    58: "B1 counter",
    59: "route give-up / records poll",
    60: "group counter",
    61: "partials poll",
    62: "dnx poll",
}


def hang_flags(td: torch.Tensor, n: int, c2: bool = False) -> dict:
    """HANG_DBG flags (value 1; TIMING must be off, it reuses these columns as stamps): {col: [bx, ...]}. C1: 62 is
    read on the idle rows (bx >= n), 57 / 66 on the routed rows, the other sites on every row. C2 (m3_moe16): every
    site 56..62 and the K-SAFE cols 48 / 49 on every row (its route give-up is col 59, not 57)."""
    td = td.cpu()
    out = {}
    cols = sorted(HANG_COLS_C2) if c2 else sorted(set(HANG_COLS) | set(HANG_COLS_C1_ANY))
    for col in cols:
        if col >= td.shape[1]:
            continue
        if c2 or col in HANG_COLS_C1_ANY:
            rows = range(td.shape[0])
        else:
            rows = range(n, td.shape[0]) if col == 62 else range(min(n, td.shape[0]))
        hit = [bx for bx in rows if int(td[bx, col]) == 1]
        if hit:
            out[col] = hit
    return out


def _dump():
    out = _K.get("MINI_OUT")
    timing, hang = bool(_K.get("M3_TIMING")), bool(_K.get("M3_HANG_DBG"))
    if not out or not (timing or hang) or not (_TRACK or _VALID or _R8):
        return
    recs, builds = [], {}
    for arm, li, t, st, post_dbg in _TRACK:
        if li == 0 or arm not in builds:
            builds[arm] = builds.get(arm, -1) + 1
        rec = {"arm": arm, "build": builds[arm], "layer": li, "t": t}
        try:
            tds = int(getattr(getattr(st, "mod", None), "TD_STRIDE", 128))
            td = st.tdbg if st.tdbg.dim() > 1 else st.tdbg.view(-1, tds)
            if timing and not hang:
                rec["stamps"] = stamp_rows(td, st.num_ctas)
            if hang:
                rec["hang"] = hang_flags(td, st.num_ctas, c2=tds == 64)
            if tds > 70:  # r2 diagnostics: max over the routed CTAs, info only
                tdc = td[: st.num_ctas].cpu()
                info = {str(c): int(tdc[:, c].max()) for c in INFO_COLS if int(tdc[:, c].max()) > 0}
                if info:
                    rec["info"] = info
            if post_dbg is not None and timing:
                ent = td[: st.num_ctas, 0].cpu()
                ent = ent[ent > 0]
                if ent.numel():
                    rec["post"] = _post_rows(post_dbg, post_dbg.numel() // 64, t, ent.min().item())
        except Exception as err:  # noqa: BLE001 (a dump must not mask the run's own exit status)
            rec["error"] = repr(err)
        recs.append(rec)
    if hang:  # R8 counting builds (HANG_DBG / tier C): a matching o_proj whose calls != its post's consumes
        for blk, solo, ck in _R8:
            try:
                c = ck.cpu().tolist() if ck is not None else None
            except Exception as err:  # noqa: BLE001
                c = repr(err)
            if c and not solo and c[0]:
                recs.append(
                    {
                        "arm": "oproj-r8",
                        "build": 0,
                        "layer": blk.layer_idx,
                        "t": blk.t,
                        "hang": {"r8_mismatch [count, oep, rxc]": c[:3]},
                    }
                )
    rk = _K.get("RANK", "0")
    with open(os.path.join(out, f"h9-hoist-tdbg-r{rk}.json"), "w") as f:
        json.dump(
            {"layers": recs, "valid": _VALID, "hang_dbg": hang, "timing": timing}, f, indent=1
        )


atexit.register(_dump)
