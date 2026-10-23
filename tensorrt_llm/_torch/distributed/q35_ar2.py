# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Launcher of the slot-protocol all-reduce + residual + RMSNorm kernel (kernels/blackwell/q35_ar2.py).

``Q35Ar2AllReduce`` has the interface and one-shot outputs of ``Q35CuteAllReduce`` on its own multicast
workspace (3 regions of ``max_rows * ctas`` CTA slot blocks + per-CTA participation counters). Every call of
one instance must use the same CTA geometry (``ctas`` per token, fixed at construction).

``Q35Ar2Dispatch`` routes one-shot calls to the slot kernel and the rest to the CuTe kernel.
"""

from __future__ import annotations

import functools
import os
from typing import Optional, Tuple

import torch

from .q35_allreduce import _cute, _two_shot


@functools.lru_cache(maxsize=None)
def _compiled_ar2(
    m: int,
    hidden: int,
    eps: float,
    nranks: int,
    ctas: int,
    use_pdl: bool,
    se_fold: bool,
    norm_out: bool,
    quant: bool,
    moe_topk: int,
    fc2_row: int,
    region_bytes: int,
):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from ..kernels.blackwell.q35_ar2 import Q35Ar2Kernel

    kernel = Q35Ar2Kernel(
        hidden=hidden,
        norm_eps=eps,
        nranks=nranks,
        ctas=ctas,
        use_pdl=use_pdl,
        se_fold=se_fold,
        norm_out=norm_out,
        quant=quant,
        region_bytes=region_bytes,
        moe_topk=moe_topk,
        fc2_row=fc2_row,
    )
    dev = torch.device("cuda")
    x = torch.empty((m, hidden), dtype=torch.bfloat16, device=dev)
    gate = torch.empty((m,), dtype=torch.bfloat16, device=dev)
    w = torch.empty((hidden,), dtype=torch.bfloat16, device=dev)
    scale = torch.empty((1,), dtype=torch.float32, device=dev)
    q = torch.empty((m, hidden), dtype=torch.uint8, device=dev)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    z = cutlass.Int64(0)
    return cute.compile(
        kernel,
        _cute(x),
        _cute(x),
        _cute(x),
        _cute(gate),
        _cute(x),
        _cute(w),
        _cute(scale),
        _cute(x),
        _cute(x),
        _cute(q),
        z,
        z,
        z,
        z,
        z,
        z,
        cutlass.Int32(0),
        stream,
    )


class Q35Ar2AllReduce:
    """See the module docstring."""

    def __init__(self, mapping, hidden: int = 4096, ctas: int = 8, max_rows: int = 64):
        if mapping.tp_size not in (2, 4, 8, 16):
            raise ValueError("Q35Ar2AllReduce supports TP 2/4/8/16")
        self.mapping = mapping
        self.hidden = hidden
        self.ctas = ctas
        self.threads = hidden // 8 // ctas
        self.max_ctas = max_rows * ctas
        from ..flashinfer_utils import get_env_enable_pdl

        self._pdl = get_env_enable_pdl()
        self._dummies = {}
        # per CTA block: one slot of 16-byte vectors per rank
        self.region_bytes = self.max_ctas * mapping.tp_size * self.threads * 16
        self._make_slot_workspace(torch.device("cuda", torch.cuda.current_device()))

    def _make_slot_workspace(self, dev):
        """Collective over the TP group: every rank must construct the instance."""
        from .ops import (
            _get_mnnvl_workspace_comm,
            _make_mnnvl_mcast_buffer,
            _mnnvl_workspace_all_succeeded,
        )

        mp = self.mapping
        comm = _get_mnnvl_workspace_comm(mp)
        use_fabric = os.environ.get("TRTLLM_FORCE_MNNVL_AR", "0") == "1" or mp.is_multi_node()
        size = 3 * self.region_bytes
        self.handle = _make_mnnvl_mcast_buffer(comm, size, mp, use_fabric)
        uc = self.handle.get_uc_buffer(mp.tp_rank, (size // 4,), torch.float32, 0)
        uc.fill_(-0.0)  # every 32-bit word = 0x80000000 = "not written"
        mc = self.handle.get_mc_buffer((size // 4,), torch.float32, 0)
        self.state = torch.zeros((self.max_ctas * 32,), dtype=torch.int32, device=dev)
        torch.cuda.synchronize()
        # Every rank has cleared its slots before any rank can write into them.
        if not _mnnvl_workspace_all_succeeded(comm, True):
            raise RuntimeError("slot all-reduce workspace setup failed on at least one rank")
        self._uc, self._mc, self._comm = uc, mc, comm
        self.own, self.mcast, self.ctl = uc.data_ptr(), mc.data_ptr(), self.state.data_ptr()

    def _dummy(self, m: int, hidden: int):
        d = self._dummies.get((m, hidden))
        if d is None:
            dev = torch.device("cuda", torch.cuda.current_device())
            d = (
                torch.empty((m,), dtype=torch.bfloat16, device=dev),
                torch.empty((m, hidden), dtype=torch.uint8, device=dev),
                torch.zeros((1,), dtype=torch.float32, device=dev),
            )
            self._dummies[(m, hidden)] = d
        return d

    def __call__(
        self,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        x: Optional[torch.Tensor] = None,
        shared_fold: Optional[Tuple] = None,
        scale: Optional[torch.Tensor] = None,
        norm_out: bool = True,
    ) -> Tuple[torch.Tensor, ...]:
        import cuda.bindings.driver as cuda
        import cutlass

        moe = None
        if shared_fold is not None and isinstance(shared_fold[0], (tuple, list)):
            moe = shared_fold[0]
        src = x if x is not None else shared_fold[1]
        m, hidden = src.shape
        assert hidden == self.hidden and m * self.ctas <= self.max_ctas
        topk = moe[2].shape[1] if moe is not None else 0
        fc2_row = moe[0].stride(0) if moe is not None else 0
        fn = _compiled_ar2(
            m,
            hidden,
            float(eps),
            self.mapping.tp_size,
            self.ctas,
            self._pdl,
            shared_fold is not None,
            norm_out,
            scale is not None,
            topk,
            fc2_row,
            self.region_bytes,
        )
        res_out = torch.empty_like(residual)
        norm = torch.empty_like(residual) if norm_out else res_out
        quant = (
            torch.empty(residual.shape, dtype=torch.float8_e4m3fn, device=residual.device)
            if scale is not None
            else None
        )
        d_gate, d_quant, d_scale = self._dummy(m, hidden)
        fc2_ptr = topw_ptr = perm_ptr = 0
        if shared_fold is not None:
            routed, shared, gate = shared_fold
            if moe is not None:
                fc2, topw, perm = moe
                fc2_ptr, topw_ptr, perm_ptr = (
                    fc2.data_ptr(),
                    topw.contiguous().data_ptr(),
                    perm.contiguous().data_ptr(),
                )
                routed = shared  # unread
            xin, gate = routed, gate.reshape(-1)
        else:
            xin, routed, shared, gate = x, x, x, d_gate
        fn(
            _cute(xin.contiguous()),
            _cute(routed.contiguous()),
            _cute(shared.contiguous()),
            _cute(gate.contiguous()),
            _cute(residual.contiguous()),
            _cute(norm_weight.reshape(-1)),
            _cute(scale.reshape(1) if scale is not None else d_scale),
            _cute(res_out),
            _cute(norm),
            _cute(quant.view(torch.uint8) if quant is not None else d_quant),
            cutlass.Int64(self.own),
            cutlass.Int64(self.mcast),
            cutlass.Int64(self.ctl),
            cutlass.Int64(fc2_ptr),
            cutlass.Int64(topw_ptr),
            cutlass.Int64(perm_ptr),
            cutlass.Int32(self.mapping.tp_rank),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )
        if scale is None:
            return norm, res_out
        if norm_out:
            return norm, quant, res_out
        return quant, res_out


class Q35Ar2Dispatch:
    """One-shot calls on the slot kernel; two-shot calls and other hidden sizes on ``cute_ar``.

    Two-shot stays on CuTe: in-model it beats a slot two-shot, although the latter wins in isolation."""

    def __init__(self, mapping, cute_ar):
        self.mapping = mapping
        self.cute = cute_ar
        self.ar2 = Q35Ar2AllReduce(mapping)

    def __call__(
        self,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        x: Optional[torch.Tensor] = None,
        shared_fold: Optional[Tuple] = None,
        scale: Optional[torch.Tensor] = None,
        norm_out: bool = True,
        two_shot: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, ...]:
        m = (x if x is not None else shared_fold[1]).shape[0]
        if two_shot is None:
            two_shot = _two_shot(m)
        if (
            two_shot
            or m * self.ar2.ctas > self.ar2.max_ctas
            or residual.shape[-1] != self.ar2.hidden
        ):
            return self.cute(
                residual,
                norm_weight,
                eps,
                x=x,
                shared_fold=shared_fold,
                scale=scale,
                norm_out=norm_out,
                two_shot=two_shot,
            )
        return self.ar2(
            residual, norm_weight, eps, x=x, shared_fold=shared_fold, scale=scale, norm_out=norm_out
        )


_DISPATCH = {}


def get_q35_ar2(mapping, cute_ar) -> Optional[Q35Ar2Dispatch]:
    """Lazy per (TP group, rank). Created only outside CUDA-graph capture (the slot workspace is a
    collective allocation); returns None otherwise and the caller keeps the CuTe kernel."""
    key = (tuple(mapping.tp_group), mapping.tp_rank)
    inst = _DISPATCH.get(key)
    if inst is None:
        if cute_ar is None or torch.cuda.is_current_stream_capturing():
            return None
        inst = Q35Ar2Dispatch(mapping, cute_ar)
        _DISPATCH[key] = inst
    return inst
