# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launcher of the ar2 all-reduce + residual + RMSNorm kernel variants (cute_dsl_kernels/blackwell/q35_ar2.py).

``Q35Ar2AllReduce(mapping, proto="flags" | "slot", rotate="end" | "poll")`` has the interface and outputs of
``q35_allreduce.Q35CuteAllReduce`` (one-shot only):

* ``proto="flags"`` runs on the library MNNVL workspace (interleaves with the library and CuTe kernels; bitwise
  the CuTe one-shot), with the workspace addresses as kernel arguments.
* ``proto="slot"`` runs on its own multicast workspace (3 regions of ``max_ctas`` slot blocks + per-CTA state
  words): no buffer flags. Every call of one instance must use the same CTA geometry (``ctas`` per token, fixed
  at construction).
"""

from __future__ import annotations

import functools
import os
from typing import Optional, Tuple

import torch

from .q35_allreduce import _ctas_per_token, _cute


@functools.lru_cache(maxsize=None)
def _compiled_ar2(m: int, hidden: int, eps: float, nranks: int, ctas: int, use_pdl: bool, se_fold: bool,
                  norm_out: bool, quant: bool, moe_topk: int, fc2_row: int, proto: str, rotate: str, stamps: bool,
                  region_bytes: int, lean: bool = True, xchg: str = "cluster", push: str = "mc",
                  two_shot: bool = False):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from ..cute_dsl_kernels.blackwell.q35_ar2 import Q35Ar2Kernel

    kernel = Q35Ar2Kernel(hidden=hidden, norm_eps=eps, nranks=nranks, ctas=ctas, use_pdl=use_pdl, se_fold=se_fold,
                          norm_out=norm_out, quant=quant, moe_topk=moe_topk, fc2_row=fc2_row, proto=proto,
                          rotate=rotate, stamps=stamps, region_bytes=region_bytes,
                          lean=lean, xchg=xchg, push=push, two_shot=two_shot)
    dev = torch.device("cuda")
    x = torch.empty((m, hidden), dtype=torch.bfloat16, device=dev)
    gate = torch.empty((m, ), dtype=torch.bfloat16, device=dev)
    w = torch.empty((hidden, ), dtype=torch.bfloat16, device=dev)
    scale = torch.empty((1, ), dtype=torch.float32, device=dev)
    q = torch.empty((m, hidden), dtype=torch.uint8, device=dev)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    z = cutlass.Int64(0)
    return cute.compile(kernel, _cute(x), _cute(x), _cute(x), _cute(gate), _cute(x), _cute(w), _cute(scale),
                        _cute(x), _cute(x), _cute(q), z, z, z, z, z, z, z, z, cutlass.Int32(0), stream)


class Q35Ar2AllReduce:
    """See the module docstring."""

    def __init__(self, mapping, proto: str = "slot", rotate: str = "end", hidden: int = 4096, ctas: int = 8,
                 max_rows: int = 64, lean: bool = True, xchg: str = "cluster", push: str = "mc",
                 two_shot_rows: Optional[int] = None):
        if mapping.tp_size not in (2, 4, 8, 16):
            raise ValueError("Q35Ar2AllReduce supports TP 2/4/8/16")
        self.mapping = mapping
        self.proto = proto
        self.rotate = rotate
        self.lean, self.xchg, self.push = lean, xchg, push
        # slot protocol: calls of >= two_shot_rows rows run two-shot (None: never; "auto" = the CuTe kernel's policy)
        if two_shot_rows == "auto":
            from .q35_allreduce import _two_shot_rows
            two_shot_rows = _two_shot_rows(mapping.tp_size) or None
        self.two_shot_rows = two_shot_rows if proto == "slot" else None
        self.peers = 0
        self.hidden = hidden
        self.ctas = ctas
        self.threads = hidden // 8 // ctas
        self.max_ctas = max_rows * ctas
        from ..flashinfer_utils import get_env_enable_pdl
        self._pdl = get_env_enable_pdl()
        self._dummies = {}
        dev = torch.device("cuda", torch.cuda.current_device())
        self.region_bytes = 0
        if proto == "slot":
            # per CTA block: W one-shot / scatter slots + 1 broadcast slot
            self.region_bytes = self.max_ctas * (mapping.tp_size + 1) * self.threads * 16
            self._make_slot_workspace(dev)
        else:
            self._ws = None

    def _make_slot_workspace(self, dev):
        from tensorrt_llm._utils import mpi_comm

        from .ops import _q35_fenced_mcast_buffer
        mp = self.mapping
        comm = mpi_comm().Split(int(mp.pp_rank * mp.cp_size + mp.cp_rank), mp.tp_rank)
        use_fabric = os.environ.get("TRTLLM_FORCE_MNNVL_AR", "0") == "1" or mp.is_multi_node()
        size = 3 * self.region_bytes
        # fenced construction: workaround for a known staging-buffer overrun in the multicast handle exchange;
        # a no-op once the library fix is present
        self.handle = _q35_fenced_mcast_buffer(size, mp, use_fabric, comm)
        uc = self.handle.get_uc_buffer(mp.tp_rank, (size // 4, ), torch.float32, 0)
        uc.fill_(-0.0)  # every 32-bit word = 0x80000000 = "not written"
        mc = self.handle.get_mc_buffer((size // 4, ), torch.float32, 0)
        self.state = torch.zeros((self.max_ctas * 32, ), dtype=torch.int32, device=dev)
        torch.cuda.synchronize()
        comm.Barrier()
        self._uc, self._mc, self._comm = uc, mc, comm
        self.own, self.mcast, self.ctl = uc.data_ptr(), mc.data_ptr(), self.state.data_ptr()
        # every rank's unicast mapping of the slot workspace (push="uc")
        self.peer_table = torch.tensor(
            [self.handle.get_uc_buffer(r, (1, ), torch.float32, 0).data_ptr() for r in range(mp.tp_size)],
            dtype=torch.int64, device=dev)
        self.peers = self.peer_table.data_ptr()

    def _flags_addrs(self, nbytes: int):
        from .ops import get_or_scale_allreduce_mnnvl_workspace
        ws = get_or_scale_allreduce_mnnvl_workspace(self.mapping, torch.bfloat16, buffer_size_bytes=nbytes)
        if ws is None:
            raise RuntimeError("MNNVL workspace too small")
        if ws is not self._ws:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("the MNNVL workspace changed during CUDA graph capture")
            local = ws["uc_buffer"]
            mcast = ws["handle"].get_mc_buffer(tuple(local.shape), local.dtype, 0)
            self._ws = ws
            self._views = (local, mcast)
            self._addrs = (local.data_ptr(), mcast.data_ptr(), ws["buffer_flags"].data_ptr())
        return self._addrs

    def _dummy(self, m: int, hidden: int):
        d = self._dummies.get((m, hidden))
        if d is None:
            dev = torch.device("cuda", torch.cuda.current_device())
            d = (torch.empty((m, ), dtype=torch.bfloat16, device=dev),
                 torch.empty((m, hidden), dtype=torch.uint8, device=dev),
                 torch.zeros((1, ), dtype=torch.float32, device=dev))
            self._dummies[(m, hidden)] = d
        return d

    def __call__(self, residual: torch.Tensor, norm_weight: torch.Tensor, eps: float,
                 x: Optional[torch.Tensor] = None, shared_fold: Optional[Tuple] = None,
                 scale: Optional[torch.Tensor] = None, norm_out: bool = True,
                 dbg: Optional[torch.Tensor] = None, two_shot: Optional[bool] = None) -> Tuple[torch.Tensor, ...]:
        import cuda.bindings.driver as cuda
        import cutlass

        moe = None
        if shared_fold is not None and isinstance(shared_fold[0], (tuple, list)):
            moe = shared_fold[0]
        src = x if x is not None else shared_fold[1]
        m, hidden = src.shape
        assert hidden == self.hidden and m * self.ctas <= self.max_ctas
        nranks = self.mapping.tp_size
        if self.proto == "slot":
            own, mcast, ctl = self.own, self.mcast, self.ctl
        else:
            own, mcast, ctl = self._flags_addrs(m * hidden * nranks * 2)
        topk = moe[2].shape[1] if moe is not None else 0
        fc2_row = moe[0].stride(0) if moe is not None else 0
        # flags: the library one-shot's geometry per call (bitwise the CuTe kernel); slot: fixed per workspace
        ctas = self.ctas if self.proto == "slot" else _ctas_per_token(m, hidden)
        if two_shot is None:
            two_shot = self.two_shot_rows is not None and m >= self.two_shot_rows
        if two_shot and self.proto != "slot":
            raise ValueError("ar2 two-shot needs the slot protocol")
        fn = _compiled_ar2(m, hidden, float(eps), nranks, ctas, self._pdl, shared_fold is not None, norm_out,
                           scale is not None, topk, fc2_row, self.proto, self.rotate, dbg is not None,
                           self.region_bytes, self.lean, self.xchg, self.push, bool(two_shot))
        res_out = torch.empty_like(residual)
        norm = torch.empty_like(residual) if norm_out else res_out
        quant = (torch.empty(residual.shape, dtype=torch.float8_e4m3fn, device=residual.device)
                 if scale is not None else None)
        d_gate, d_quant, d_scale = self._dummy(m, hidden)
        fc2_ptr = topw_ptr = perm_ptr = 0
        if shared_fold is not None:
            routed, shared, gate = shared_fold
            if moe is not None:
                fc2, topw, perm = moe
                fc2_ptr, topw_ptr, perm_ptr = fc2.data_ptr(), topw.contiguous().data_ptr(), perm.contiguous().data_ptr()
                routed = shared  # unread
            xin, gate = routed, gate.reshape(-1)
        else:
            xin, routed, shared, gate = x, x, x, d_gate
        fn(_cute(xin.contiguous()), _cute(routed.contiguous()), _cute(shared.contiguous()), _cute(gate.contiguous()),
           _cute(residual.contiguous()), _cute(norm_weight.reshape(-1)),
           _cute(scale.reshape(1) if scale is not None else d_scale), _cute(res_out), _cute(norm),
           _cute(quant.view(torch.uint8) if quant is not None else d_quant), cutlass.Int64(own), cutlass.Int64(mcast),
           cutlass.Int64(ctl), cutlass.Int64(fc2_ptr), cutlass.Int64(topw_ptr), cutlass.Int64(perm_ptr),
           cutlass.Int64(dbg.data_ptr() if dbg is not None else 0), cutlass.Int64(self.peers),
           cutlass.Int32(self.mapping.tp_rank),
           cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        if scale is None:
            return norm, res_out
        if norm_out:
            return norm, quant, res_out
        return quant, res_out


class Q35Ar2Dispatch:
    """The model's fused all-reduce callable under TRTLLM_Q35_AR2: one-shot calls on the ar2 kernel, the rest
    (TP16 two-shot rows, > 64 rows) on the CuTe kernel ``cute_ar`` (same interface as ``Q35CuteAllReduce``).

    ``mode``: "fe" (flags protocol, addresses as arguments), "fp" (+ buffer rotation inside the poll loop),
    "sl" (flag-free slot protocol on its own workspace); suffix "m" = st.async + mbarrier norm exchange (fpm, slm)."""

    _MODES = {"fe": dict(proto="flags", rotate="end"), "fp": dict(proto="flags", rotate="poll"),
              "fpm": dict(proto="flags", rotate="poll", xchg="mbar"),
              "sl": dict(proto="slot", two_shot_rows="auto"),
              "slm": dict(proto="slot", xchg="mbar", two_shot_rows="auto")}

    def __init__(self, mapping, cute_ar, mode: str):
        self.mapping = mapping
        self.cute = cute_ar
        self.ar2 = Q35Ar2AllReduce(mapping, **self._MODES[mode])
        # The slot modes leave the two-shot calls (TP16 >= 12 rows) on the CuTe kernel by default: in-model the slot
        # two-shot was slower than the CuTe two-shot although it wins in isolation.
        # TRTLLM_Q35_AR2_SLOT_TWOSHOT=1 opts in.
        self.slot_two_shot = os.environ.get("TRTLLM_Q35_AR2_SLOT_TWOSHOT", "0") == "1"

    def __call__(self, residual: torch.Tensor, norm_weight: torch.Tensor, eps: float,
                 x: Optional[torch.Tensor] = None, shared_fold: Optional[Tuple] = None,
                 scale: Optional[torch.Tensor] = None, norm_out: bool = True,
                 two_shot: Optional[bool] = None) -> Tuple[torch.Tensor, ...]:
        from .q35_allreduce import _two_shot_rows
        m = (x if x is not None else shared_fold[1]).shape[0]
        if two_shot is None:
            thr = _two_shot_rows(self.mapping.tp_size)
            two_shot = 0 < thr <= m
        slot = self.ar2.proto == "slot"
        if ((two_shot and not (slot and self.slot_two_shot)) or m * self.ar2.ctas > self.ar2.max_ctas
                or residual.shape[-1] != self.ar2.hidden):
            return self.cute(residual, norm_weight, eps, x=x, shared_fold=shared_fold, scale=scale,
                             norm_out=norm_out, two_shot=two_shot)
        return self.ar2(residual, norm_weight, eps, x=x, shared_fold=shared_fold, scale=scale, norm_out=norm_out,
                        two_shot=bool(two_shot) if slot else None)


_DISPATCH = {}


def get_q35_ar2(mapping, cute_ar, mode: str) -> Optional[Q35Ar2Dispatch]:
    """Lazy per (TP group, rank, mode); created outside CUDA-graph capture only (the slot protocol allocates its
    multicast workspace collectively), else None (the caller keeps the CuTe kernel)."""
    key = (tuple(mapping.tp_group), mapping.tp_rank, mode)
    inst = _DISPATCH.get(key)
    if inst is None and key not in _DISPATCH:
        if cute_ar is None or torch.cuda.is_current_stream_capturing():
            return None
        inst = Q35Ar2Dispatch(mapping, cute_ar, mode)
        _DISPATCH[key] = inst
    return inst
