# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The only places the TP4 C1 block package (``tp4c1``) is bent to run inside this target.

Every seam replaces process plumbing, never arithmetic or launch order:

1. ``load``: imports the ``tp4c1`` package (its modules import each other statically).
2. ``toolchain.paths.csrc_root``: a path never read (seam 5 stands in for every source the reference would hash).
3. ``dist``: the reference modules' ``torch.distributed`` is a facade over this target's TP MPI communicator
   (rank / world / barrier / all_gather_object); no torch.distributed or gloo group is created.
4. ``torch.utils.cpp_extension.load``: refused, except the two extensions the block uses, answered in-process:
   ``pt4_post_*`` (the routing-warp post: ``PostExt``, launching this target's minimax_m3_sm107tp4_hoist_post, with the
   rx slots in an MNNVL allocation) and ``pr_ar_*`` (the lamport push workspaces: ``LamportExt``, MNNVL). The reference
   other uses of their extensions are not on the block's path.
5. ``route_warp_tp4_fc1_per_token_host.src_hash``: the reference pinned H9_HDR_HASH (the C++ copies are byte-identical,
   test-checked), so no source file is read.
6. ``router_hoist_host._rank_world`` / ``fused_o_proj_tp4_host._rank``: this target's TP rank and world.
7. ``fused_die_local_host._dev_b2lag``: by default (B2LAG, default ON) the measured HYB setting
   (M3LL_DEV_MOE_B2LAG=1: b2buf 2, stages 6, lag 6), without an os.environ read. TRTLLM_M31_DISABLE_TP4WC_B2LAG=1
   makes it return None, the reference shipped 5-8 row schedule (b2buf 1, stages 7, lag = fused_phase_lag).
8. ``m3moe_dl.host._ENGINE_PATH``: the family's ``kernels/m3ll/moe_engine_kernel.py``, not a second copy.
9. ``H9_RH_DEADLINE_NS``: the post's rx deadline is the hang cap (knob), not the reference 2 s.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import struct
import threading
from types import SimpleNamespace
from typing import Dict, List

import torch

from ... import native_ops
from ...m31_switches import m31_disabled

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = f"{__package__}.tp4c1"  # the block package: tp4c1/ next to this file
# The family's engine kernel (kernels/m3ll/moe_engine_kernel.py): the reference fused/kernel.py plus the dl_tab option
# (a host-built SM -> role table; off by default, so the hoist's calls compile the reference code path).
ENGINE_KERNEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(HERE))),
    "kernels",
    "m3ll",
    "moe_engine_kernel.py",
)
HDR_HASH = (
    "40c1042cbf7d"  # peer lr4_prr4_q.HDR_HASH: sha1[:12] of the post ext source + its 3 headers
)
# The reference own hang cap (families/moe/_common.HANG_CAP_NS): 2^31 polls x 200 ns, the poll count our boundary
# all-reduces trap at. The post's rx deadline defaults to it (the reference ships 2 s, which traps on a slow peer).
HANG_CAP_NS = (1 << 31) * 200
# The measured HYB engine schedule (M3LL_DEV_MOE_B2LAG=1 for every HIS / HYB arm).
B2LAG = {"b2buf": 2, "stages": 6, "lag": 6}


# Read once at import, like every kill switch. B2LAG is default ON since the TP4 mcAllreduceNorm parity fix
# seam 7 returns the measured HYB setting. TRTLLM_M31_DISABLE_TP4WC_B2LAG=1 restores the
# reference's shipped schedule (no B2LAG).
B2LAG_OFF = m31_disabled("TP4WC_B2LAG")
EMPTY_WORD = -(2**31)  # kEmpty (allreduce_norm_quant.cuh:18 / lamport_allreduce_norm.cuh:28)
KH = 6144
CLUSTER_SZ = 8  # lamport ext kClusterSz (flags per row)
PUSH_PARAMS_FMT = (
    "8qq5i4x"  # PushParams{int* peers[8]; const int* flags; int rank, world, maxt, cluster, ndst;}
)


def _dist_modules():
    """The block modules seam 3 rebinds (seam 5 and 6 then patch the first and the last)."""
    return (
        importlib.import_module(
            ".tp4c1.kernels.comm.allreduce_norm_quant.route_warp_tp4_fc1_per_token_host",
            package=__package__,
        ),
        importlib.import_module(
            ".tp4c1.kernels.comm.allreduce_norm_quant.route_warp_tp4_host", package=__package__
        ),
        importlib.import_module(
            ".tp4c1.kernels.comm.allreduce_norm_quant.router_exchange", package=__package__
        ),
        importlib.import_module(
            ".tp4c1.kernels.comm.allreduce_norm_quant.lamport_allreduce_host", package=__package__
        ),
        importlib.import_module(
            ".tp4c1.kernels.moe.router.fused_o_proj_tp4_host", package=__package__
        ),
    )


_lock = threading.Lock()
_st: Dict[str, object] = {}


# ------------------------------------------------------------------------------------------------ 1, 2: load
def load():
    """Import the block package (idempotent) and apply seam 2 (also when the package was imported before)."""
    paths = importlib.import_module(".tp4c1.toolchain.paths", package=__package__)
    paths.csrc_root = lambda: os.path.join(HERE, "_csrc_never_read")  # seam 2
    return importlib.import_module(".tp4c1", package=__package__)


# ------------------------------------------------------------------------------------------------ 3: dist facade
class DistFacade:
    """The subset of torch.distributed the block's reference modules call, over an mpi4py TP communicator."""

    class _Group:
        WORLD = "m31-tp4-hoist-tp"

    group = _Group

    def __init__(self, comm, rank: int, world: int):
        self.comm, self.rank, self.world = comm, int(rank), int(world)

    def is_initialized(self) -> bool:
        return True

    def get_rank(self, group=None) -> int:
        return self.rank

    def get_world_size(self, group=None) -> int:
        return self.world

    def barrier(self, group=None) -> None:
        self.comm.Barrier()

    def all_gather_object(self, out: list, obj, group=None) -> None:
        got = self.comm.allgather(obj)
        if len(got) != len(out):
            raise RuntimeError(
                f"m31 tp4 hoist: all_gather_object over {len(got)} ranks into {len(out)} slots"
            )
        out[:] = got

    def new_group(self, *args, **kwargs):
        return self.group.WORLD


# ------------------------------------------------------------------------------------------------ MNNVL buffers
class Mnnvl:
    """Collective MNNVL allocations over the TP group (McAllReduceWorkspace.create's protocol): every rank allocates
    in the same order; ``alloc`` returns (handle, [world] unicast views)."""

    def __init__(self, mapping, comm):
        self.mapping, self.comm = mapping, comm
        self.keep: List[object] = []

    def alloc(self, nbytes: int, dtype: torch.dtype):
        from tensorrt_llm._torch.distributed.ops import (
            _make_mnnvl_mcast_buffer,
            _mnnvl_workspace_all_succeeded,
        )

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("m31 tp4 hoist: MNNVL allocation during graph capture")
        if not _mnnvl_workspace_all_succeeded(self.comm, True):
            raise RuntimeError("m31 tp4 hoist: MNNVL allocation skipped on another rank")
        esz = torch.empty((), dtype=dtype).element_size()
        n = -(-nbytes // esz)
        handle = _make_mnnvl_mcast_buffer(
            self.comm, n * esz, self.mapping, self.mapping.is_multi_node()
        )
        world, rank = self.mapping.tp_size, self.mapping.tp_rank
        ucs = [handle.get_uc_buffer(r, (n,), dtype, 0) for r in range(world)]
        if not _mnnvl_workspace_all_succeeded(self.comm, True):
            raise RuntimeError("m31 tp4 hoist: MNNVL allocation failed on another rank")
        self.keep += [handle, ucs]
        return handle, ucs, rank


# ------------------------------------------------------------------------------------------------ 4a: lamport ext
class LamportExt:
    """``pr_ar_*`` (lamport_allreduce_norm_ext.cu) as the block uses it: ws_create / ws_open / push_params_bytes /
    ws_destroy. Per workspace: ws_words(world, maxt) int32 words per rank filled with kEmpty (the ext's fill_empty) in
    MNNVL instead of cudaMalloc + cudaIpc, and maxt x 8 zeroed local call-parity flags (the ext's cudaMemset)."""

    def __init__(self, mn: Mnnvl):
        self.mn = mn
        self.ws: Dict[int, SimpleNamespace] = {}

    def ws_create(self, rank, world, maxt, handle_out, exact, opt_mode=1):
        if not (2 <= world <= 8 and maxt in (8, 16, 48) and 0 <= int(opt_mode) <= 3):
            raise RuntimeError(f"ws_create: world {world} maxt {maxt} opt_mode {opt_mode}")
        words = 4 * maxt * world * (KH // 2)
        _, ucs, mrank = self.mn.alloc(words * 4, torch.int32)
        if mrank != rank:
            raise RuntimeError(f"ws_create: rank {rank} != TP rank {mrank}")
        ucs[rank].fill_(EMPTY_WORD)
        flags = torch.zeros(maxt * CLUSTER_SZ, dtype=torch.int32, device=ucs[rank].device)
        torch.cuda.synchronize()
        w = SimpleNamespace(
            rank=int(rank), world=int(world), maxt=int(maxt), exact=int(exact), ucs=ucs, flags=flags
        )
        key = len(self.ws) + 1
        self.ws[key] = w
        handle_out.zero_()
        handle_out[:8].copy_(torch.tensor(list(struct.pack("<q", key)), dtype=torch.uint8))
        return key

    def ws_open(self, wsp, handles):
        w = self.ws[int(wsp)]
        if len(handles) != w.world:
            raise RuntimeError("ws_open: handle count")

    def ws_destroy(self, wsp):
        self.ws.pop(int(wsp), None)

    def push_params_bytes(self, wsp):
        w = self.ws[int(wsp)]
        peers = [u.data_ptr() for u in w.ucs] + [0] * (8 - w.world)
        b = struct.pack(
            "<" + PUSH_PARAMS_FMT,
            *peers,
            w.flags.data_ptr(),
            w.rank,
            w.world,
            w.maxt,
            4 if w.exact else CLUSTER_SZ,
            w.world,
        )
        return torch.tensor(list(b), dtype=torch.uint8)


# ------------------------------------------------------------------------------------------------ 4b: post ext
class PostExt:
    """``pt4_post_*`` (route_warp_tp4_fc1_per_token_ext.cu): the launch is the thop op; the rx workspace
    (router_exchange_create / open / _router_exchange_reset) lives in MNNVL, and this class keeps what the ext's
    rh_find checked (an rx base is a slot boundary of a workspace with the push bucket's maxt / world / rank /
    npart)."""

    MAXRP = 6  # rh::MAXRP
    E = 128

    def __init__(self, mn: Mnnvl):
        self.mn = mn
        self.rx: Dict[int, SimpleNamespace] = {}  # local base -> workspace

    def router_exchange_create(self, rank, world, maxt, npart, nslots=1):
        if not (
            1 <= world <= 8
            and 0 <= rank < world
            and 1 <= npart
            and world * npart <= self.MAXRP
            and maxt >= 1
            and nslots >= 1
        ):
            raise RuntimeError("router_exchange_create: world / rank / npart / maxt / nslots")
        slot_words = maxt * world * npart * self.E
        _, ucs, mrank = self.mn.alloc(slot_words * 8 * nslots, torch.int64)
        if mrank != rank:
            raise RuntimeError(f"router_exchange_create: rank {rank} != TP rank {mrank}")
        ucs[rank].zero_()
        torch.cuda.synchronize()
        w = SimpleNamespace(
            rank=int(rank),
            world=int(world),
            maxt=int(maxt),
            npart=int(npart),
            nslots=int(nslots),
            slot_words=slot_words,
            ucs=ucs,
        )
        self.rx[ucs[rank].data_ptr()] = w
        hdl = torch.zeros(64, dtype=torch.uint8)
        hdl[:8].copy_(
            torch.tensor(list(struct.pack("<q", ucs[rank].data_ptr())), dtype=torch.uint8)
        )
        return ucs[rank], hdl

    def router_exchange_open(self, handles, rank, rx):
        w = self.rx.get(rx.data_ptr())
        if w is None or rank != w.rank or len(handles) != w.world:
            raise RuntimeError(
                "router_exchange_open: rx is not a router_exchange_create workspace / rank / handles"
            )
        return torch.tensor([u.data_ptr() for u in w.ucs], dtype=torch.int64, device=rx.device)

    def _router_exchange_reset(self, rx):
        w = self.rx.get(rx.data_ptr())
        if w is None:
            raise RuntimeError("router_exchange_reset: not an rh workspace")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("router_exchange_reset: refused during graph capture")
        w.ucs[w.rank].zero_()

    def _rh_find(self, base: int):
        """The ext's rh_find: the workspace whose local rx holds ``base`` at a slot boundary (the C++ op no longer
        sees the registry; S0b's HYBMNNVL drop-in failed exactly here, so the check stays, in Python)."""
        for local, w in self.rx.items():
            off, nbytes = base - local, w.slot_words * 8
            if 0 <= off < nbytes * w.nslots and off % nbytes == 0:
                return w
        return None

    def allreduce_norm_quant_route(
        self,
        pp,
        rows,
        residual,
        gamma,
        eps,
        gemma,
        out_norm,
        out_res,
        out_q,
        out_sf,
        cl,
        rx_bases=(),
        route=None,
        rctr=None,
        bias=None,
        tab=None,
        delay_ns=0,
        lg_dbg=None,
        tdbg=None,
        repl=16,
        npart=2,
        rxc=None,
        deadline_ns=2000000000,
        hang_dbg=0,
        roll=0,
        x4_q=None,
        x4_sf=None,
        x4_sfsc=None,
        prr=0,
        rx2=1,
        pt=0,
        x4_pts=None,
        pwait=0,
    ):
        rx_bases = [int(b) for b in rx_bases]
        if route is not None and tab is None:
            rank, world, maxt = struct.unpack_from("<3i", bytes(pp[72:84].tolist()))
            w = self._rh_find(rx_bases[rank]) if len(rx_bases) == world else None
            if (
                w is None
                or w.maxt != maxt
                or w.world != world
                or w.npart != npart
                or w.rank != rank
            ):
                raise RuntimeError(
                    "allreduce_norm_quant_route: rx slot must match the prx bucket (maxt, world, rank) "
                    "and npart"
                )
        native_ops.minimax_m3_sm107tp4_hoist_post(
            pp,
            int(rows),
            residual,
            gamma,
            float(eps),
            int(gemma),
            out_norm,
            out_res,
            out_q,
            out_sf,
            int(cl),
            rx_bases,
            route,
            rctr,
            bias,
            tab,
            int(delay_ns),
            lg_dbg,
            tdbg,
            int(repl),
            int(npart),
            rxc,
            int(deadline_ns),
            int(hang_dbg),
            int(roll),
            x4_q,
            x4_sf,
            x4_sfsc,
            int(prr),
            int(rx2),
            int(pt),
            x4_pts,
            int(pwait),
        )


@contextlib.contextmanager
def no_jit(post: PostExt, lamport: LamportExt):
    """Seam 4: every torch JIT extension load in this scope is answered in-process or refused."""
    import torch.utils.cpp_extension as ce

    orig = ce.load

    def load(name, *args, **kwargs):
        if name.startswith("pt4_post_"):
            return post
        if name.startswith("pr_ar_"):
            return lamport
        raise RuntimeError(f"m31 tp4 hoist: JIT extension build refused ({name})")

    ce.load = load
    try:
        yield
    finally:
        ce.load = orig


# ------------------------------------------------------------------------------------------------ install
def install(mapping, comm, hang_cap_ns: int = HANG_CAP_NS):
    """Bind the process-wide reference knobs and apply seams 3, 5-9 (idempotent per process; one TP group). Returns the
    state the block builder needs: (post ext, lamport ext, dist facade, Runtime)."""
    with _lock:
        if "installed" in _st:
            return _st["installed"]
        load()
        from .tp4c1.families.front import _bind as fb
        from .tp4c1.families.moe import _common as FC
        from .tp4c1.runtime.knobs import KNOBS

        rank, world = int(mapping.tp_rank), int(mapping.tp_size)
        llk = SimpleNamespace(rank=rank)
        # The values the variants' make() binds (identical strings: KNOBS.bind refuses a different one), bound before
        # any peer host module is imported (some read them at import), plus the measured HYB arm's bindings.
        KNOBS.bind(
            {
                **FC.base_knobs(llk),
                **fb.g3_knobs(llk),
                "H4_DL_PORT": FC.engine_dir("fused"),
                "M3_G1_ITP": "768",
                "H9_RH_DEADLINE_NS": str(int(hang_cap_ns)),
            },
            "m31_tp4_hoist",
        )
        dist = DistFacade(comm, rank, world)
        dist_modules = _dist_modules()
        for module in dist_modules:  # seam 3
            module.dist = dist
        pt4 = dist_modules[0]
        pt4.src_hash = lambda: HDR_HASH  # seam 5
        hoist = importlib.import_module(
            ".tp4c1.kernels.moe.engine.router_hoist_host", package=__package__
        )
        hoist._rank_world = lambda: (rank, world)  # seam 6
        dist_modules[4]._rank = lambda: rank
        dl = importlib.import_module(
            ".tp4c1.kernels.moe.engine.fused_die_local_host", package=__package__
        )
        dl._dev_b2lag = (
            (lambda: None) if B2LAG_OFF else (lambda: dict(B2LAG))
        )  # seam 7 (B2LAG kill switch)
        if not os.path.isfile(ENGINE_KERNEL):
            raise ImportError(f"m31 tp4 hoist: the family engine kernel {ENGINE_KERNEL} is missing")
        op = dl._op_dl()
        op._ENGINE_PATH = ENGINE_KERNEL  # seam 8
        from .tp4c1.runtime.arena import Arena
        from .tp4c1.runtime.comm import LoopbackComm
        from .tp4c1.runtime.ctx import Runtime

        rt = Runtime(
            preset_id="m3.sm107a.tp4.c1.m31q7",
            arch="sm_107a",
            tp=world,
            rank=rank,
            conc=1,
            deadline_ns=int(hang_cap_ns),
            jit=None,
            local=Arena("local", 1 << 20).allocate(),
            comm=LoopbackComm(1 << 20, rank=rank, world=world),
            counters=Arena("counters", 1 << 16).allocate(),
            tp_group=dist.group.WORLD,
        )
        mn = Mnnvl(mapping, comm)
        out = SimpleNamespace(
            post=PostExt(mn),
            lamport=LamportExt(mn),
            dist=dist,
            rt=rt,
            rank=rank,
            world=world,
            hang_cap_ns=int(hang_cap_ns),
            knobs=KNOBS,
        )
        _st["installed"] = out
        return out
