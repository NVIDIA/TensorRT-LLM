# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Router exchange between the o_proj router CTAs and the routing post: rx workspaces and slots, the routing-post
factories, the NVFP4 hand-off buffers and the debug dumps.

Knobs: H9_V3_CL (cluster width), H9_RH_SLOTS (rx slots per bucket), H9_RH_ROLL (TP2 rolled routing path), H9_RH_X4 (the
post also writes the engine's NVFP4 activation), H9_RH_DEADLINE_NS / H9_RH_HANG_DBG (rx poll deadline; the debug mode
records and continues instead of trapping), H9_RH_ARM / H9_RH_TDBG / H9_RH_LGDBG (debug dumps).
"""

from __future__ import annotations

import atexit
import hashlib
import logging
import os
import struct

import torch
import torch.distributed as dist

from ....runtime.knobs import KNOBS as _K
from ....toolchain.paths import csrc_dir
from ...moe.engine import router_hoist_weights as hw
from . import allreduce_norm_quant_host as lb

_log = logging.getLogger(__name__)

_HERE = csrc_dir("comm", "allreduce_norm_quant")
SRC = os.path.join(_HERE, "allreduce_norm_quant_route_ext.cu")
HDRS = (
    "allreduce_norm_quant_route_ext.cu",
    "allreduce_norm_quant_route.cuh",
    "allreduce_norm_quant.cuh",
    "lamport_exchange.cuh",
)
EPS = 1e-6
NPART = hw.NPART
TDBG_W = 4  # rh::TDBG_W: entry, routing start, published, CTA end (globaltimer ns)
MAXRP = 6  # rh::MAXRP: world * npart
RCTR_STRIDE = hw.RCTR_STRIDE
_st = {}
_DBG = []
_COUNTERS = []  # tensors rx_reset_all zeroes: every post_rh rxc + registered writer counters
_SLOTS = {}  # id(blk) -> RxSlot (single owner)
_RXC = []  # (blk, rxc, arm) of every post_rh: HANG_DBG records are dumped from these


def src_hash() -> str:
    """sha1 over the extension source and every header it includes: torch's JIT key hashes only SRC."""
    h = hashlib.sha1()
    for f in HDRS:
        with open(os.path.join(_HERE, f), "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:12]


def ext():
    """Build/load once per process (rank 0 first, as lbcore.ext); the header hash is in the name and the flags."""
    if "ext" not in _st:
        from torch.utils.cpp_extension import load

        hh = src_hash()
        bd = os.path.join(_K.get("TMPDIR", "/tmp"), "h9lbh-build-" + hh)
        os.makedirs(bd, exist_ok=True)

        def _load():
            return load(
                name="h9_lbh_" + hh,
                sources=[SRC],
                build_directory=bd,
                verbose=False,
                extra_cuda_cflags=[
                    "-O3",
                    "-std=c++20",
                    "-gencode=arch=compute_107a,code=sm_107a",
                    f"-DH9_HDR_HASH=0x{hh}",
                ],
                extra_cflags=["-O3", "-std=c++20"],
            )

        if dist.is_initialized():
            if dist.get_rank() == 0:
                _st["ext"] = _load()
            dist.barrier()
            if dist.get_rank() != 0:
                _st["ext"] = _load()
        else:
            _st["ext"] = _load()
    return _st["ext"]


def prx_spec(t):
    """(push-params bytes, rank, world, maxt) of the prx bucket the o_proj pushes into (lb._prx_ws: k2.prx_ws). rank and
    world come from here (not dist), the same source as the o_proj's push spec."""
    pext, w = lb._prx_ws(t)
    pp = pext.push_params_bytes(w)
    rank, world, maxt, cluster, ndst = struct.unpack_from("5i", bytes(pp[:92].tolist()), 72)
    assert cluster == 4 and ndst == world, (cluster, ndst, world)
    return pp, rank, world, maxt


def _gloo():
    if "gloo" not in _st:
        _st["gloo"] = dist.new_group(backend="gloo")
    return _st["gloo"]


class RxWorkspace:
    """rx v2 of one prx bucket: u64 [nslots][maxt][world][npart][128] = tag << 32 | fp32, zero
    at creation and after rx_reset_all. One slot per (o_proj rh, post_rh) pair, so no other layer or arm ever writes
    an address this pair reads (per-pair tags alone would collide: every layer counts calls from 1)."""

    def __init__(self, rank: int, world: int, maxt: int, npart: int, nslots: int):
        e = ext()
        assert world * npart <= MAXRP, (world, npart)
        assert dist.get_world_size() == world, "rx workspace: prx world != process group world"
        self.rank, self.world, self.maxt, self.npart, self.nslots = rank, world, maxt, npart, nslots
        self.slot_words = maxt * world * npart * hw.E
        self.rx, h = e.router_exchange_create(rank, world, maxt, npart, nslots)
        got = [None] * world
        dist.all_gather_object(got, (rank, h.tolist()), group=_gloo())
        by = dict(got)
        assert sorted(by) == list(range(world)), f"rx workspace: prx ranks {sorted(by)}"
        self.bases0 = e.router_exchange_open(
            [torch.tensor(by[r], dtype=torch.uint8) for r in range(world)], rank, self.rx
        )
        self.bases0_host = [int(v) for v in self.bases0.cpu().tolist()]
        self.used = 0
        dist.barrier()

    def reset(self):
        """Zero the local rx words (stream-ordered; never during graph capture)."""
        ext()._router_exchange_reset(self.rx)


class RxSlot:
    """One (o_proj rh, post_rh) pair's rx slot: `bases` device int64 [world] (the o_proj's `rx_bases`), `bases_host`
    (allreduce_norm_quant_route), `maxt` (the o_proj MUST index rx with this, = the prx bucket), `index(t, src, part,
    e)` (u64 words), `rxc` device int32 [maxt][32] (the post's consumed-call counters, word 0 of row t; the o_proj's
    counter check reads rxc[0] post-wait: oep[bx] == rxc[0]). The slot owns rxc, so both factories see one tensor;
    rx_reset_all zeroes it."""

    def __init__(self, ws: RxWorkspace, k: int, layer: int):
        off = k * ws.slot_words * 8
        self.ws, self.k, self.layer = ws, k, layer
        self.maxt, self.world, self.npart = ws.maxt, ws.world, ws.npart
        self.bases_host = [b + off for b in ws.bases0_host]
        self.bases = torch.tensor(self.bases_host, dtype=torch.int64, device="cuda")
        self.rxc = torch.zeros(ws.maxt * RCTR_STRIDE, dtype=torch.int32, device="cuda")
        _COUNTERS.append(self.rxc)

    def index(self, t, src, part, e):
        return ((t * self.world + src) * self.npart + part) * hw.E + e


def rx_slot(blk, npart: int = NPART) -> RxSlot:
    """The rx slot of blk (allocated on first call by either factory, same order on every rank)."""
    s = _SLOTS.get(id(blk))
    if s is not None:
        assert s.npart == npart, (s.npart, npart)
        return s
    _, rank, world, maxt = prx_spec(blk.t)
    key = ("rx", maxt, npart)
    if key not in _st:
        _st[key] = RxWorkspace(rank, world, maxt, npart, int(_K.get("H9_RH_SLOTS", "128")))
    ws = _st[key]
    assert ws.used < ws.nslots, f"rx bucket {maxt}: out of slots (H9_RH_SLOTS={ws.nslots})"
    s = RxSlot(ws, ws.used, blk.layer_idx)
    ws.used += 1
    _SLOTS[id(blk)] = s
    blk.h9_rx_slot = s  # keeps blk's id unique for the slot's lifetime
    return s


def register_counter(t: torch.Tensor):
    """A writer-side call counter (o_proj oep) that must restart with the reader's: zeroed by rx_reset_all."""
    assert t.is_cuda
    _COUNTERS.append(t)


def rx_reset_all():
    """Collective (every rank, same point): sync, zero every rx bucket and every registered counter (post rxc,
    writer oep), sync. Call at every arm boundary and after any o_proj rh call without its post_rh. Never in capture."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("lbh.rx_reset_all: refused during graph capture")
    ext()  # every rank loads the extension before the collective below
    torch.cuda.synchronize()
    tab = sorted((s.ws.maxt, s.k, s.layer) for s in _SLOTS.values())
    got = [None] * dist.get_world_size()
    dist.all_gather_object(got, tab, group=_gloo())
    if any(g != tab for g in got):
        raise RuntimeError("lbh.rx_reset_all: rx slot tables differ across ranks")
    dist.barrier()
    for k, ws in _st.items():
        if isinstance(k, tuple) and k[0] == "rx":
            ws.reset()
    for c in _COUNTERS:
        c.zero_()
    torch.cuda.synchronize()
    dist.barrier()


def _warm(call):
    """Build-time launch with rows = 0: one idle cluster that loads the module (no spinning launch may load it later)
    and touches no rx, prx slot, flag, counter or output, so it can never hang or desync the
    counters. A live-rows call at build time would wait for an o_proj push that never comes: never do that; harnesses
    that need a build-time call use fn.build_call()."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("router_exchange: post factories must be built outside graph capture")
    call(0)
    torch.cuda.synchronize()


def _roll():
    """H9_RH_ROLL=1: rolled routing path + world-2 polls (same ops, bitwise-equal outputs). Default 0."""
    return int(_K.get("H9_RH_ROLL", "0"))


X4_ROWS_MIN = (
    16  # x4 buffers hold max(blk.t, 16) rows; row strides are fixed (3072 B values, 384 B scales)
)


def _x4_on():
    """H9_RH_X4=1: the post also writes the engine's NVFP4 B1 activation. Default 0."""
    return int(_K.get("H9_RH_X4", "0"))


class X4Bufs:
    """One blk's post-x4 buffers (single owner, blk.h9_x4): q uint8 [rows][3072] packed e2m1 (low nibble = even
    column), sf uint8 [rows][384] e4m3 block-16 scales, both row-major; sfsc fp32 [1] = the layer's fc31_input_scale
    (blk.mw.a1_gscale). `post` is set when an X4 post factory was built on this blk (the pairing guard reads it)."""

    def __init__(self, blk):
        self.rows = max(blk.t, X4_ROWS_MIN)
        self.q = torch.zeros(self.rows, hw.H // 2, dtype=torch.uint8, device="cuda")
        self.sf = torch.zeros(self.rows, hw.H // 16, dtype=torch.uint8, device="cuda")
        self.sfsc = blk.mw.a1_gscale.reshape(1).float().contiguous()
        self.post = False


def x4_bufs(blk) -> X4Bufs:
    """blk's X4 buffers (created on first call by either factory; the engine's b1_x4 must read exactly these)."""
    b = getattr(blk, "h9_x4", None)
    if b is None:
        b = blk.h9_x4 = X4Bufs(blk)
    return b


def check_x4_pair(blk, engine_b1_x4: bool, x4_q=None, x4_sf=None, sfsc=None):
    """Build-time pairing guard (call after both of blk's factories are built): engine b1_x4 <=> blk's post is an X4
    post, and the engine reads the post's own q / sf tensors with the same sfsc value. A b1_x4 engine without its X4
    post reads a previous call's bytes, silently, so this must raise."""
    b = getattr(blk, "h9_x4", None)
    post = b is not None and b.post
    if bool(engine_b1_x4) != post:
        raise RuntimeError(
            f"hoist x4 pairing (layer {blk.layer_idx}): engine b1_x4={int(bool(engine_b1_x4))} but the "
            f"post {'writes' if post else 'does not write'} x4 (H9_RH_X4 / post factory)"
        )
    if not engine_b1_x4:
        return
    if x4_q is not None and x4_q.data_ptr() != b.q.data_ptr():
        raise RuntimeError(
            "hoist x4 pairing: engine x4 values are not this blk's post buffer (lbh.x4_bufs(blk).q)"
        )
    if x4_sf is not None and x4_sf.data_ptr() != b.sf.data_ptr():
        raise RuntimeError(
            "hoist x4 pairing: engine x4 scales are not this blk's post buffer (lbh.x4_bufs(blk).sf)"
        )
    if sfsc is not None and not torch.equal(sfsc.reshape(1).float().to(b.sfsc.device), b.sfsc):
        raise RuntimeError("hoist x4 pairing: engine fc31_input_scale != the post's sfsc")


def _kw(blk):
    """Optional allreduce_norm_quant_route kwargs, passed only when on (older ext builds / fakes lack them): roll,
    x4."""
    kw = {"roll": 1} if _roll() else {}
    if _x4_on():
        b = x4_bufs(blk)
        b.post = True
        kw.update(x4_q=b.q, x4_sf=b.sf, x4_sfsc=b.sfsc)
    return kw


def _cl():
    return int(_K.get("H9_V3_CL", "8"))


def _deadline():
    hang = int(_K.get("H9_RH_HANG_DBG", "0"))
    d = int(_K.get("H9_RH_DEADLINE_NS", "50000000" if hang else "2000000000"))
    return d, hang


def _arm(arm):
    return arm or _K.get("H9_RH_ARM", "a")


def _check_rb(blk, rb):
    assert rb.maxt == blk.t, (
        f"RouteBuf maxt {rb.maxt} != blk.t {blk.t} (rows >= t would never publish)"
    )
    owner = getattr(rb, "h9_owner", None)
    assert owner is None or owner == id(blk), "RouteBuf shared by two blks: one post per RouteBuf"
    rb.h9_owner = id(blk)
    assert rb.route.data_ptr() % 128 == 0 and rb.rctr.data_ptr() % 128 == 0
    assert 1 <= rb.repl <= 32


def _dbg(blk, rb, arm, lg_on):
    """Debug buffers: tdbg ring, lg_dbg, and same-call snapshots (normed2, route words) copied right after the post."""
    d = {"arm": _arm(arm), "blk": blk, "rb": rb, "tdbg": None, "lg": None}
    if _K.get("H9_RH_TDBG", "0") == "1":
        d["tdbg"] = torch.zeros(16 * rb.maxt * TDBG_W, dtype=torch.int64, device="cuda")
    if lg_on and _K.get("H9_RH_LGDBG", "0") == "1":
        d["lg"] = torch.zeros(rb.maxt * hw.E, dtype=torch.float32, device="cuda")
        d["n2"] = torch.empty_like(blk.normed2)
        d["words"] = torch.empty(blk.t, hw.TOP_K, dtype=torch.int64, device="cuda")
    if d["tdbg"] is not None or d["lg"] is not None:
        _DBG.append(d)
    return d


def post_rh(blk, rb, npart: int = NPART, cl: int | None = None, arm: str | None = None):
    """boundary_post for the full hoist: lbv3 (residual_out, normed2) + tagged rx poll (tag == rxc[t] + 1) + MiniMax2
    top-4 + publish into RouteBuf rb (hoist.RouteBuf, maxt == blk.t, one blk per RouteBuf). Pair with the o_proj
    router role writing rx_slot(blk).bases (tag = its per-CTA counter + 1, counter registered via register_counter)."""
    _check_rb(blk, rb)
    e, t = ext(), blk.t
    pp, _, _, _ = prx_spec(t)
    slot = rx_slot(blk, npart)
    rxc = slot.rxc
    bias = blk.mw.bias.float().contiguous()
    d = _dbg(blk, rb, arm, True)
    _RXC.append((blk, rxc, _arm(arm)))
    dl, hang = _deadline()
    c = cl or _cl()
    rk = _kw(blk)
    blk.h9_rh = (slot, rxc, bias, d)  # keep alive
    tdbg, lg = d["tdbg"], d["lg"]
    words = rb.route.view(rb.repl, rb.maxt, hw.ROUTE_LINE_U64)[0, :t, : hw.TOP_K]

    def call(rows):
        e.allreduce_norm_quant_route(
            pp,
            rows,
            blk.residual_mid,
            blk.aw.post_norm,
            EPS,
            1,
            blk.normed2,
            blk.residual_out,
            None,
            None,
            c,
            slot.bases_host,
            rb.route,
            rb.rctr,
            bias,
            None,
            0,
            lg,
            tdbg,
            rb.repl,
            npart,
            rxc,
            dl,
            hang,
            **rk,
        )

    _warm(call)

    def fn():
        call(t)
        if lg is not None:  # debug only: same-call snapshots (an extra copy kernel in the chain)
            d["n2"].copy_(blk.normed2)
            d["words"].copy_(words)

    fn.build_call = lambda: call(0)
    return fn


def post_tab(
    blk,
    rb,
    tab: torch.Tensor,
    delay_ns: int | None = None,
    cl: int | None = None,
    arm: str | None = None,
):
    """Skeleton tier B boundary_post: lbv3 + publish `tab` (int32 [t][4] = id << 16 | bf16 w, hoist.learn_table)
    after spinning delay_ns (default H9_RH_DELAY_NS or 150) at the routing point. No rx, plain o_proj."""
    _check_rb(blk, rb)
    assert tab.dtype == torch.int32 and tab.is_cuda and tab.numel() >= blk.t * hw.TOP_K
    e, t = ext(), blk.t
    pp, _, _, _ = prx_spec(t)
    dns = int(_K.get("H9_RH_DELAY_NS", "150")) if delay_ns is None else int(delay_ns)
    d = _dbg(blk, rb, arm, False)
    c = cl or _cl()
    rk = _kw(blk)
    blk.h9_rh = (tab, d)
    tdbg = d["tdbg"]

    def call(rows):
        e.allreduce_norm_quant_route(
            pp,
            rows,
            blk.residual_mid,
            blk.aw.post_norm,
            EPS,
            1,
            blk.normed2,
            blk.residual_out,
            None,
            None,
            c,
            [],
            rb.route,
            rb.rctr,
            None,
            tab,
            dns,
            None,
            tdbg,
            rb.repl,
            0,
            **rk,
        )

    _warm(call)

    def fn():
        call(t)

    fn.build_call = lambda: call(0)
    return fn


def _chrono(ring, tag):
    """tdbg ring [16][t][4] (slot (tag-1)&15) -> oldest-first list of (tag, [t][4]) for the last min(16, tag) calls."""
    out = []
    for k in range(max(1, tag - 15), tag + 1):
        out.append({"tag": k, "rows": ring[(k - 1) & 15]})
    return out


def _loud(what, ex):
    _log.error("H9 LBH DUMP FAILED (%s): %r", what, ex, exc_info=ex)


def _rows(x, t, w):
    """Row-major [*, w] view of a buffer sized for any maxt >= t, cut to t rows (buffers are sized by the bucket's or
    the RouteBuf's maxt, rows by blk.t)."""
    return x.reshape(-1, w)[:t]


def collect(dbg=None, rxc_list=None):
    """{arm: {"tdbg": [...], "lg": {...}, "hang": [...]}} from the registered buffers (CPU tensors work too). Each
    section is collected under its own try so one failure cannot hide another (a hang record above all)."""
    dbg = _DBG if dbg is None else dbg
    rxc_list = _RXC if rxc_list is None else rxc_list
    per_arm = {}

    def arm_of(arm):
        return per_arm.setdefault(arm, {"tdbg": [], "lg": {}, "hang": []})

    for blk, rxc, arm in rxc_list:
        try:
            x = _rows(rxc, blk.t, RCTR_STRIDE)[:, :4].cpu()
            if bool(
                (x[:, 1] != 0).any()
            ):  # [consumed tag, deadline hits, expected tag, observed stale tag]
                arm_of(arm)["hang"].append(
                    {
                        "layer": blk.layer_idx,
                        "blk": f"{blk.layer_idx}:{id(blk):x}",
                        "rows": x.tolist(),
                    }
                )
        except Exception as ex:  # noqa: BLE001
            _loud(f"hang record layer {getattr(blk, 'layer_idx', '?')}", ex)
    for d in dbg:
        blk, rb, arm = d["blk"], d["rb"], d["arm"]
        li, t = blk.layer_idx, blk.t
        key = f"{li}:{id(blk):x}"
        if d["tdbg"] is not None:
            try:
                tag = rb.tag()
                ring = (
                    _rows(d["tdbg"], 16 * rb.maxt, TDBG_W)
                    .view(16, rb.maxt, TDBG_W)[:, :t]
                    .cpu()
                    .tolist()
                )
                arm_of(arm)["tdbg"].append(
                    {"layer": li, "blk": key, "t": t, "tag": tag, "calls": _chrono(ring, tag)}
                )
            except Exception as ex:  # noqa: BLE001
                _loud(f"tdbg layer {li}", ex)
        if d["lg"] is not None:
            try:
                tg, ids, w = hw.unpack_route(_rows(d["words"], t, hw.TOP_K).cpu())
                arm_of(arm)["lg"][key] = {
                    "layer": li,
                    "t": t,
                    "logits": _rows(d["lg"], t, hw.E).cpu(),
                    "normed2": _rows(d["n2"], t, hw.H).cpu(),
                    "bias": blk.mw.bias.float().cpu(),
                    "ids": ids,
                    "w": w,
                    "tag": tg,
                }
            except Exception as ex:  # noqa: BLE001
                _loud(f"lgdbg layer {li}", ex)
    return per_arm


def write(per_arm, out, rk):
    """One file per (kind, arm), each under its own try. Returns the paths written."""
    import json

    done = []
    for arm, a in per_arm.items():
        for kind, ext_, obj in (
            ("hang", "json", a["hang"]),
            ("tdbg", "json", a["tdbg"]),
            ("lgdbg", "pt", a["lg"]),
        ):
            if not obj:
                continue
            path = os.path.join(out, f"h9_rh_{kind}_{arm}_r{rk}.{ext_}")
            try:
                if ext_ == "json":
                    with open(path, "w") as f:
                        json.dump(obj, f)
                else:
                    torch.save(obj, path)
                done.append(path)
            except Exception as ex:  # noqa: BLE001
                _loud(path, ex)
    return done


def _dump():
    if not _DBG and not _RXC:
        return
    try:
        rk = dist.get_rank() if dist.is_initialized() else 0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception as ex:  # noqa: BLE001
        _loud("sync / rank", ex)
        rk = int(_K.get("RANK", "0"))
    write(collect(), _K.get("MINI_OUT", "."), rk)


atexit.register(_dump)
