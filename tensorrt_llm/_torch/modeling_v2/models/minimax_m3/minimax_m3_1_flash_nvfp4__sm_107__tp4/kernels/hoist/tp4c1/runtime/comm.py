# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The comm arena: one cudaMalloc per rank, IPC handles all-gathered once on the TP group, peers opened once. Every
cross-rank buffer (G3 push slots, Lamport buckets) is a carve at the same offset on every rank."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field

from ..api.ctx import CommBuf
from .arena import Arena

_IPC_HANDLE = 64


class CudaError(RuntimeError):
    pass


def cudart():
    """The CUDA runtime library through ctypes (for IPC and raw allocations the torch API does not cover)."""
    for n in ("libcudart.so", "libcudart.so.13", "libcudart.so.12"):
        try:
            return ctypes.CDLL(n)
        except OSError:
            continue
    raise CudaError("libcudart not loadable")


def check(rc, what):
    """Raises CudaError naming `what` unless the cudart return code `rc` is cudaSuccess."""
    if rc != 0:
        raise CudaError(f"{what} failed: cudaError {rc}")


_cudart, _ck = cudart, check  # old private names; runtime.preload and this module still use them


@dataclass
class CommArena:
    nbytes: int
    rank: int
    world: int
    base: int = 0
    peer_bases: tuple = ()
    arena: Arena = field(default=None)
    _rt: object = None
    _opened: list = field(default_factory=list)

    def __post_init__(self):
        self.arena = Arena("comm", self.nbytes, device="cuda")

    def allocate(self, tp_group, all_gather_object):
        """all_gather_object(list_out, obj, group) is torch.distributed.all_gather_object (injected for tests)."""
        import torch

        self._rt = rt = _cudart()
        p = ctypes.c_void_p()
        _ck(
            rt.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(max(self.nbytes, 1))),
            "cudaMalloc(comm arena)",
        )
        _ck(rt.cudaMemset(p, 0, ctypes.c_size_t(max(self.nbytes, 1))), "cudaMemset(comm arena)")
        self.base = p.value
        h = (ctypes.c_byte * _IPC_HANDLE)()
        _ck(rt.cudaIpcGetMemHandle(ctypes.byref(h), p), "cudaIpcGetMemHandle")
        handles = [None] * self.world
        all_gather_object(handles, bytes(h), group=tp_group)
        bases = []
        for r, hb in enumerate(handles):
            if r == self.rank:
                bases.append(self.base)
                continue
            q = ctypes.c_void_p()
            hh = (ctypes.c_byte * _IPC_HANDLE).from_buffer_copy(hb)
            _ck(
                rt.cudaIpcOpenMemHandle(ctypes.byref(q), hh, ctypes.c_uint(1)),
                f"cudaIpcOpenMemHandle(rank {r})",
            )
            self._opened.append(q.value)
            bases.append(q.value)
        self.peer_bases = tuple(bases)
        self.arena.storage = _wrap_ptr(self.base, self.nbytes, torch)
        return self

    def carve(self, name: str, nbytes: int, align: int = 256) -> CommBuf:
        local = self.arena.carve(name, nbytes, align)
        off = self.arena.offset(name)
        return CommBuf(
            local=local,
            peer_ptrs=tuple(b + off for b in self.peer_bases),
            offset=off,
            nbytes=nbytes,
        )


def _wrap_ptr(ptr: int, nbytes: int, torch):
    """A uint8 tensor over raw device memory we own (lifetime = process), via __cuda_array_interface__."""

    class _Buf:
        __cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }

    return torch.as_tensor(_Buf(), device="cuda")


@dataclass
class LoopbackComm:
    """Plan-only / single-process stand-in: offsets are computed, no memory or IPC (CPU tests, tp=1 benches)."""

    nbytes: int
    rank: int = 0
    world: int = 1
    arena: Arena = field(default=None)

    def __post_init__(self):
        self.arena = Arena("comm", self.nbytes, device="cpu")

    def carve(self, name: str, nbytes: int, align: int = 256) -> CommBuf:
        local = self.arena.carve(name, nbytes, align)
        off = self.arena.offset(name)
        return CommBuf(
            local=local, peer_ptrs=tuple(off for _ in range(self.world)), offset=off, nbytes=nbytes
        )
