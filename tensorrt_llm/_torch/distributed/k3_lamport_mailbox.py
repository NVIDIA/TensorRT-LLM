# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lamport mailbox arenas for Kimi K3 kernels that exchange data across a TP group in-kernel.

A :class:`K3LamportArena` is one multicast allocation (``McastGPUBuffer``, fabric handles across
nodes) per TP group, split into :data:`NUM_PARITIES` equal *parity regions*. Every mailbox use
(one all-reduce, or one producer-epilogue -> consumer hand-off) takes one whole region: producers
store 16-byte fragments to the region's multicast address (or to a peer's unicast address),
consumers poll their own copy and re-arm (store the sentinel back) everything they read. The
device primitives are in ``tensorrt_llm/_torch/cute_dsl_kernels/rubin/k3_lamport.py``; a
standalone all-reduce built on them is :class:`K3LamportAllReduce`.

Contract (every user of an arena must keep it; one diverging rank hangs the whole group):

1. **Allocation.** :meth:`K3LamportArena.get` is collective (MPI) and must run at model
   construction, before warmup and CUDA-graph capture; the memory is outside the graph pool. Its
   words are filled with the sentinel ``0x80000000`` (FP32 ``-0.0``) and the group is barriered
   before anyone may publish. The arena is its own workspace: never alias the MNNVL all-reduce
   buffers or their flags.
2. **Publish after the wait.** A kernel publishes into a region only after its
   ``griddepcontrol.wait`` (or without PDL), and every kernel between two mailbox uses of a group
   executes ``griddepcontrol.wait`` before it exits (or is launched without PDL). Then a rank's
   publish of use ``u + 1`` happens after its use ``u`` (polls and re-arms) completed.
3. **Every published fragment is consumed in the same use.** Producer and consumer agree on the
   row count, including CUDA-graph padding rows (padded rows are published and consumed like the
   others); a consumer re-arms exactly what it polled. A region is therefore fully armed again
   after each use, whatever layout the use had. A use must fit its region
   (:meth:`K3LamportArena.require_fit`).
4. **All-to-all uses.** In every use each rank polls at least one fragment published by every
   rank of the group (an all-reduce does; a sparse exchange would void rule 5's argument).
5. **Parities, one stream.** All uses of an arena are issued on one CUDA stream, in the same order
   on every rank, and two consecutive uses never take the same region: each forward takes its
   parities from :meth:`K3LamportArena.begin_forward` (:func:`lamport_parities`: ``0, 1, 2, 0,
   ...`` with the last one never 0), so any sequence of forwards (CUDA graphs of any size, eager
   steps) is safe. With 2 and 4, a rank reaches use ``u + 2`` only after it saw every peer's
   publish of use ``u + 1``, i.e. after every peer finished use ``u``.
6. **Payload words** are never the sentinel: BF16 ``-0`` is stored as ``+0`` (it changes no sum).

Failure: a rank whose forward raises outside graph capture after taking mailbox uses cannot know
what its peers published into its copy, so it calls :meth:`K3LamportArena.poison` and the arena
refuses every later forward (a collective :meth:`K3LamportArena.reset` would be needed).

Kill switch: ``TRTLLM_K3_LAMPORT_DISABLE=1`` makes :func:`k3_lamport_supported` false, so callers
take their stock paths. Debug: ``TRTLLM_K3_LAMPORT_SPIN_LIMIT=N`` compiles every poll of the
``k3_lamport`` primitives with a pass counter that traps after ``N`` failed passes (a hang becomes
a CUDA error); ``TRTLLM_ENABLE_PDL=0`` launches the kernels without PDL.
"""

from __future__ import annotations

import os

import torch

# The one-shot/two-shot rule of MNNVLAllReduce (same env override, read once).
from tensorrt_llm._torch.distributed.ops import _MNNVL_ONE_SHOT_THRESHOLD_BYTES
from tensorrt_llm._utils import mpi_comm, mpi_disabled
from tensorrt_llm.bindings.internal.runtime import McastGPUBuffer
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

#: Regions per arena (see the parity rule in the module docstring).
NUM_PARITIES = 3
#: An armed mailbox word (FP32 -0.0) as a signed 32-bit value (``k3_lamport.SENTINEL_I32``).
SENTINEL_I32 = -0x80000000
#: Default bytes per parity region: a one-shot of M x world <= 658 rows of 7168, or a two-shot of
#: 128 rows with 16 emulated ranks on 4 GPUs. Three regions stay under 32 MB per rank.
DEFAULT_REGION_BYTES = 9 << 20

_DISABLED = os.environ.get("TRTLLM_K3_LAMPORT_DISABLE", "0") == "1"


def _multicast_supported(index: int) -> bool:
    from cuda.bindings import driver

    err, dev = driver.cuDeviceGet(index)
    if err != driver.CUresult.CUDA_SUCCESS:
        return False
    err, value = driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, dev
    )
    return err == driver.CUresult.CUDA_SUCCESS and value != 0


def k3_lamport_supported(device: torch.device | None = None) -> bool:
    """Whether the Lamport mailbox kernels may run on this rank: not disabled by the kill switch,
    MPI available (the arena's collectives), a Blackwell or Rubin GPU with multicast support
    (``multimem`` stores, PDL) and the CuTe DSL available. Ranks must agree: the arena checks the
    multicast part collectively."""
    if _DISABLED or mpi_disabled() or not torch.cuda.is_available():
        return False
    from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

    if not IS_CUTLASS_DSL_AVAILABLE:
        return False
    index = torch.cuda.current_device()
    if device is not None and device.index is not None:
        index = device.index
    if torch.cuda.get_device_capability(index) not in ((10, 0), (10, 3), (10, 7)):
        return False
    return _multicast_supported(index)


def lamport_parities(num_uses: int) -> tuple[int, ...]:
    """Parities of the ``num_uses`` mailbox uses of one forward of a TP group, in launch order.

    The first use takes parity 0, adjacent uses differ and the last use is never 0, so the
    sequence stays safe across any order of forwards (CUDA graphs of different sizes and eager
    steps alike). A forward needs at least two uses: a single use would meet itself in the next
    forward.
    """
    if num_uses < 0:
        raise ValueError(f"num_uses {num_uses} < 0")
    if num_uses == 1:
        raise ValueError("a forward with mailbox uses needs at least two of them")
    parities = [u % NUM_PARITIES for u in range(num_uses)]
    if parities and parities[-1] == 0:
        parities[-1] = 1  # the one before it is 2 (num_uses % 3 == 1, num_uses >= 4)
    return tuple(parities)


class K3LamportSchedule:
    """Hands out :func:`lamport_parities` in order for one forward, on one stream.

    Get it from :meth:`K3LamportArena.begin_forward` (which allows one open forward per arena);
    ``finish()`` checks that every planned use was taken and closes the forward.
    """

    def __init__(self, num_uses: int, arena: K3LamportArena | None = None):
        self.parities = lamport_parities(num_uses)
        self._next = 0
        self._stream: int | None = None
        self._arena = arena

    def take(self) -> int:
        """The next use's parity. Every use must be issued on the stream of the first one."""
        if self._next >= len(self.parities):
            raise RuntimeError(f"more than the {len(self.parities)} planned mailbox uses")
        stream = torch.cuda.current_stream().cuda_stream if torch.cuda.is_available() else 0
        if self._stream is None:
            self._stream = stream
        elif stream != self._stream:
            raise RuntimeError("all mailbox uses of a forward must be issued on one stream")
        parity = self.parities[self._next]
        self._next += 1
        return parity

    def finish(self) -> None:
        if self._next != len(self.parities):
            raise RuntimeError(
                f"{self._next} of {len(self.parities)} planned mailbox uses were taken"
            )
        if self._arena is not None:
            self._arena._close(self)

    def cancel(self) -> None:
        """Close a forward none of whose uses ran (e.g. a failed CUDA-graph capture)."""
        if self._arena is not None:
            self._arena._close(self)


class K3LamportArena:
    """One multicast arena of a TP group: :data:`NUM_PARITIES` regions of ``region_bytes``.

    Args:
        mapping: The rank's mapping; the arena belongs to its TP group (split as the MNNVL
            all-reduce workspace: color pp/cp position, key tp rank).
        region_bytes: Bytes per parity region (a multiple of 16).
    """

    _arenas: dict[tuple, K3LamportArena] = {}

    @staticmethod
    def _group_key(mapping: Mapping) -> tuple:
        return (
            mapping.world_size,
            mapping.rank,
            mapping.tp_size,
            mapping.pp_rank,
            mapping.cp_size,
            mapping.cp_rank,
        )

    @classmethod
    def get(cls, mapping: Mapping, region_bytes: int = DEFAULT_REGION_BYTES) -> K3LamportArena:
        """The arena of ``mapping``'s TP group with regions of at least ``region_bytes``, one per
        group and process (collective on first use: call it at model construction on every rank
        of the group, never during capture). The first call sizes it, so users share the default
        size unless they need more."""
        key = cls._group_key(mapping)
        arena = cls._arenas.get(key)
        if arena is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("K3LamportArena.get allocates: call it before graph capture")
            arena = cls._arenas[key] = cls(mapping, region_bytes)
        elif arena.region_bytes < region_bytes:
            raise ValueError(
                f"the TP group's arena has {arena.region_bytes} B regions, not {region_bytes}"
            )
        return arena

    def __init__(self, mapping: Mapping, region_bytes: int = DEFAULT_REGION_BYTES):
        if mpi_disabled():
            raise RuntimeError("K3LamportArena needs MPI (McastGPUBuffer's communicator)")
        if region_bytes <= 0 or region_bytes % 16:
            raise ValueError(f"region_bytes {region_bytes} must be a positive multiple of 16")
        self.mapping = mapping
        self.tp_size, self.tp_rank = mapping.tp_size, mapping.tp_rank
        self.region_bytes = region_bytes
        self.total_bytes = NUM_PARITIES * region_bytes
        self.device = torch.device("cuda", mapping.local_rank)
        self._open: K3LamportSchedule | None = None
        self._poisoned: str | None = None
        # The TP group, as MNNVL's workspace splits it (color: pp/cp position, key: tp rank).
        self.comm = mpi_comm().Split(
            int(mapping.pp_rank * mapping.cp_size + mapping.cp_rank), mapping.tp_rank
        )
        # Agree before the collective allocation: a rank that cannot allocate must not leave the
        # others blocked in it.
        if not all(self.comm.allgather(_multicast_supported(mapping.local_rank))):
            raise RuntimeError("K3LamportArena: a rank of the TP group has no multicast support")
        force_fabric = os.environ.get("TRTLLM_FORCE_MNNVL_AR", "0") == "1"
        self.fabric = force_fabric or mapping.is_multi_node()
        self.handle = McastGPUBuffer(
            self.total_bytes,
            self.tp_size,
            self.tp_rank,
            mapping.local_rank,
            self.fabric,
            self.comm.py2f(),
        )
        words = self.total_bytes // 4
        self.words = self.handle.get_uc_buffer(self.tp_rank, [words], torch.int32, 0)
        self.mc_base = self.handle.get_mc_buffer([words], torch.int32, 0).data_ptr()
        self.uc_bases: list[int] = [
            self.handle.get_uc_buffer(r, [1], torch.int32, 0).data_ptr()
            for r in range(self.tp_size)
        ]
        self.peer_stride = self._agree_peer_stride()
        self.reset()
        logger.info(
            f"[K3 Lamport] arena rank {self.tp_rank}/{self.tp_size}: {NUM_PARITIES} x "
            f"{region_bytes} B, fabric={self.fabric}, peer stride={self.peer_stride}"
        )

    def _agree_peer_stride(self) -> int | None:
        """Distance between consecutive ranks' unicast mappings, if every rank's are equally
        spaced (always with fabric handles); None (no two-shot) on every rank otherwise."""
        if self.tp_size == 1:
            return 0
        strides = {b - a for a, b in zip(self.uc_bases, self.uc_bases[1:])}
        stride = strides.pop() if len(strides) == 1 else None
        if not all(s is not None for s in self.comm.allgather(stride)):
            return None
        return stride

    def reset(self) -> None:
        """Arm every word and barrier the group (collective; only at construction or in tests
        between phases, never while a mailbox use may be in flight)."""
        torch.cuda.synchronize(self.device)
        self.comm.Barrier()
        self.words.fill_(SENTINEL_I32)
        torch.cuda.synchronize(self.device)
        self.comm.Barrier()

    def fill(self, value: int) -> None:
        """Fill every word with ``value`` and barrier (tests that use raw flags, e.g. latency
        probes; follow with :meth:`reset` before any Lamport use)."""
        torch.cuda.synchronize(self.device)
        self.comm.Barrier()
        self.words.fill_(value)
        torch.cuda.synchronize(self.device)
        self.comm.Barrier()

    def begin_forward(self, num_uses: int) -> K3LamportSchedule:
        """The parity schedule of one forward with ``num_uses`` mailbox uses (every user of the
        arena in that forward takes its parities from it, in launch order)."""
        if self._poisoned is not None:
            raise RuntimeError(f"the Lamport mailbox arena is unusable: {self._poisoned}")
        if self._open is not None:
            raise RuntimeError("the previous forward's mailbox schedule was not finished")
        schedule = K3LamportSchedule(num_uses, arena=self)
        self._open = schedule
        return schedule

    def _close(self, schedule: K3LamportSchedule) -> None:
        if self._open is schedule:
            self._open = None

    def poison(self, reason: str) -> None:
        """Refuse every later forward: this rank's copy may hold publishes nobody will poll."""
        if self._poisoned is None:
            logger.error(f"[K3 Lamport] mailbox arena of TP rank {self.tp_rank} poisoned: {reason}")
            self._poisoned = reason

    def require_fit(self, nbytes: int, what: str = "mailbox use") -> None:
        """Raise unless a use of ``nbytes`` fits one region (check at plan time)."""
        if nbytes > self.region_bytes:
            raise ValueError(f"{what} needs {nbytes} B; the regions have {self.region_bytes} B")

    def local_addr(self, parity: int) -> int:
        """This rank's unicast address of region ``parity`` (where it polls)."""
        return self.uc_bases[self.tp_rank] + self._offset(parity)

    def mc_addr(self, parity: int) -> int:
        """Multicast address of region ``parity`` (a store reaches every rank's copy)."""
        return self.mc_base + self._offset(parity)

    def peer_addr(self, rank: int, parity: int) -> int:
        """Rank ``rank``'s unicast address of region ``parity`` (point-to-point pushes)."""
        return self.uc_bases[rank] + self._offset(parity)

    def unarmed_words(self) -> int:
        """Words of this rank's copy that are not the sentinel (0 when no use is in flight)."""
        return int((self.words != SENTINEL_I32).sum().item())

    def _offset(self, parity: int) -> int:
        if not 0 <= parity < NUM_PARITIES:
            raise ValueError(f"parity {parity} not in [0, {NUM_PARITIES})")
        return parity * self.region_bytes


class K3LamportAllReduce:
    """Standalone Lamport all-reduce of BF16 rows on a :class:`K3LamportArena`.

    Bitwise equal to ``MNNVLAllReduce`` (no fusion): FP32 sum in rank order from +0 (one-shot:
    chunks of 8 ranks above 8; two-shot: one chunk up to 16), BF16 round to nearest even.
    ``world`` larger than the arena's group emulates ``world`` ranks for tests: each real rank
    then passes ``[world / tp_size, M, H]`` (its virtual ranks ``r, r + tp, ...``).

    One-shot vs two-shot follows ``MNNVLAllReduce``: one-shot iff ``M * H * world * 2`` bytes are
    at most ``oneshot_max_bytes`` (default: its threshold, ``TRTLLM_MNNVL_AR_ONESHOT_MAX_BYTES``,
    1 MiB when unset) and the rows fit the region. The same choice keeps the outputs bitwise
    equal at 16 ranks, where one-shot and two-shot sum in different orders. On Rubin the
    crossover is at about 3 MiB (``TRTLLM_MNNVL_AR_ONESHOT_MAX_BYTES=3145728``).

    Args:
        arena: The TP group's arena.
        hidden: Row width in BF16 elements (a multiple of 8).
        world: Group size to run (default the arena's; a multiple of it emulates ranks).
        oneshot_max_bytes: One-shot threshold in MNNVL's metric (default MNNVL's).
        **kernel_options: ``threads``, ``use_pdl``, ``trigger``, ``publish``, ``backoff_ns``,
            ``spin_limit``, ``vote`` of ``run_k3_lamport_allreduce``.
    """

    def __init__(
        self,
        arena: K3LamportArena,
        hidden: int,
        world: int | None = None,
        oneshot_max_bytes: int | None = None,
        **kernel_options,
    ):
        from tensorrt_llm._torch.cute_dsl_kernels.rubin import k3_lamport_allreduce as kernels

        self._kernels = kernels
        self.arena = arena
        self.hidden = hidden
        self.real = arena.tp_size
        self.world = world or arena.tp_size
        if hidden <= 0 or hidden % 8:
            raise ValueError(f"hidden {hidden} must be a positive multiple of 8")
        if self.world % self.real or self.world > kernels.MAX_WORLD:
            raise ValueError(
                f"world {self.world} must be a multiple of the group size {self.real} "
                f"and at most {kernels.MAX_WORLD}"
            )
        self.v = self.world // self.real
        if oneshot_max_bytes is None:
            oneshot_max_bytes = _MNNVL_ONE_SHOT_THRESHOLD_BYTES
        self.oneshot_max_bytes = oneshot_max_bytes
        # Two-shot needs every rank's buffers at one stride and whole fragments per owner.
        self.twoshot_available = arena.peer_stride is not None and (hidden // 8) % self.world == 0
        self.kernel_options = kernel_options

    def fits(self, m: int, mode: str) -> bool:
        """Whether ``m`` rows of ``mode`` fit a region."""
        k = self._kernels
        if mode == k.MODE_ONESHOT:
            return k.oneshot_region_bytes(m, self.hidden, self.world) <= self.arena.region_bytes
        return (
            k.twoshot_region_bytes(m, self.hidden, self.world, self.real) <= self.arena.region_bytes
        )

    def supports(self, m: int, mode: str) -> bool:
        """Whether ``m`` rows can run as ``mode`` (fit, and for two-shot an equal peer stride)."""
        if mode == self._kernels.MODE_TWOSHOT and not self.twoshot_available:
            return False
        return m >= 1 and self.fits(m, mode)

    def compile(self) -> None:
        """Compile the kernels this instance can run (at warmup on every rank, then barrier the
        group before the first use, so no rank polls while a peer still compiles)."""
        k = self._kernels
        modes = [k.MODE_ONESHOT] + ([k.MODE_TWOSHOT] if self.twoshot_available else [])
        for mode in modes:
            k.compile_k3_lamport_allreduce(
                mode=mode,
                world=self.world,
                real=self.real,
                hidden=self.hidden,
                **self.kernel_options,
            )

    def mode_for(self, m: int) -> str:
        """MNNVL's choice for ``m`` rows (one-shot falls in for an unavailable two-shot when the
        two sum in the same order, i.e. up to 8 ranks)."""
        k = self._kernels
        oneshot = m * self.hidden * self.world * 2 <= self.oneshot_max_bytes
        if oneshot and self.supports(m, k.MODE_ONESHOT):
            return k.MODE_ONESHOT
        if self.supports(m, k.MODE_TWOSHOT):
            return k.MODE_TWOSHOT
        if self.world <= 8 and self.supports(m, k.MODE_ONESHOT):
            return k.MODE_ONESHOT
        raise ValueError(
            f"no kernel for {m} rows at {self.world} ranks: the region is too small or the "
            "two-shot needs fabric handles (multi-node or TRTLLM_FORCE_MNNVL_AR=1)"
        )

    def __call__(
        self,
        x: torch.Tensor,
        parity: int,
        out: torch.Tensor | None = None,
        mode: str | None = None,
    ) -> torch.Tensor:
        """All-reduce ``x`` (``[M, H]``, or ``[world / tp, M, H]`` when emulating) through region
        ``parity``; returns ``out`` (``[M, H]``)."""
        if x.dtype != torch.bfloat16 or not x.is_contiguous() or x.device != self.arena.device:
            raise ValueError("x must be a contiguous bf16 tensor on the arena's device")
        if x.shape[-1] != self.hidden:
            raise ValueError(f"x hidden {x.shape[-1]} != {self.hidden}")
        if self.v > 1:
            if x.dim() != 3 or x.shape[0] != self.v:
                raise ValueError(f"emulated world {self.world}: x must be [{self.v}, M, H]")
            m = x.shape[1]
        else:
            if x.dim() != 2:
                raise ValueError("x must be [M, H]")
            m = x.shape[0]
        if m < 1:
            raise ValueError("x has no rows")
        mode = mode or self.mode_for(m)
        if mode == self._kernels.MODE_TWOSHOT and not self.twoshot_available:
            raise ValueError(
                "two-shot needs fabric handles (multi-node or TRTLLM_FORCE_MNNVL_AR=1)"
            )
        if not self.fits(m, mode):
            raise ValueError(f"{mode} of {m} rows does not fit the arena region")
        if out is None:
            out = torch.empty((m, self.hidden), dtype=torch.bfloat16, device=x.device)
        elif (
            out.shape != (m, self.hidden)
            or out.dtype != torch.bfloat16
            or not out.is_contiguous()
            or out.device != x.device
        ):
            raise ValueError(
                f"out must be a contiguous bf16 [{m}, {self.hidden}] tensor on x's device"
            )
        if x.data_ptr() % 16 or out.data_ptr() % 16:
            raise ValueError("x and out must be 16-byte aligned")
        self._kernels.run_k3_lamport_allreduce(
            x,
            out,
            mode=mode,
            world=self.world,
            real=self.real,
            rank=self.arena.tp_rank,
            local_addr=self.arena.local_addr(parity),
            mc_addr=self.arena.mc_addr(parity),
            peer0_addr=self.arena.peer_addr(0, parity),
            peer_stride=self.arena.peer_stride or 0,
            **self.kernel_options,
        )
        return out


__all__ = [
    "DEFAULT_REGION_BYTES",
    "K3LamportAllReduce",
    "K3LamportArena",
    "K3LamportSchedule",
    "NUM_PARITIES",
    "SENTINEL_I32",
    "k3_lamport_supported",
    "lamport_parities",
]
