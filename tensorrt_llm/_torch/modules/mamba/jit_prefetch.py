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
"""Variant provider for the Mamba2 SSD prefill path (Triton kernels).

Given a scheduled batch, reproduce on ``meta`` tensors exactly the Triton
launches that ``Mamba2Mixer.forward_core`` + ``Mamba2Metadata.prepare`` will
issue for its context requests, and return them as KernelCalls.

The launcher functions themselves (``_chunk_cumsum_fwd`` ... ``chunk_state_varlen``
and ``cu_seqlens_to_chunk_indices_offsets_triton``) are run unchanged under
``shadow_launches()``: their allocation, ``rearrange`` and grid code runs on
meta tensors, and every Triton ``run`` is recorded instead of executed. Only
the tensors entering the SSD are rebuilt here, with the same layout the mixer
produces (views into the in_proj output, token-major conv output).

Batch facts used, all known once the batch is scheduled:
  num context requests, their chunk lengths, whether any has cached tokens.
They decide every specialization that varies at runtime: ``HAS_INITSTATES``,
the multi-sequence chunk-index kernel, and Triton's integer specialization of
``seqlen`` / token counts (== 1, % 16 == 0, otherwise).
"""

from __future__ import annotations

from typing import Any, List, Optional

import torch

from ...jit_prefetch import KernelCall, shadow_launches


class _MixerShape:
    """The per-layer constants of one Mamba2Mixer that its kernels see."""

    def __init__(self, mixer):
        self.nheads = mixer.tp_nheads
        self.head_dim = mixer.head_dim
        self.ngroups = mixer.tp_ngroups
        self.d_state = mixer.d_state
        self.chunk_size = mixer.chunk_size
        self.d_inner = mixer.tp_d_inner
        self.conv_dim = mixer.tp_conv_dim
        self.delta_softplus = mixer.delta_softplus
        self.state_dtype = mixer._mamba_ssm_cache_dtype
        self.token_major_conv = mixer._token_major_conv
        self.has_dt_bias = mixer.dt_bias is not None
        self.has_d = mixer.D is not None
        self.io_dtype = (
            mixer.in_proj.weight.dtype
            if getattr(mixer.in_proj, "weight", None) is not None
            and mixer.in_proj.weight.dtype.is_floating_point
            and mixer.in_proj.weight.dtype.itemsize >= 2
            else torch.bfloat16
        )
        self.zxbcdt_width = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads

    def key(self):
        return (
            self.nheads,
            self.head_dim,
            self.ngroups,
            self.d_state,
            self.chunk_size,
            self.d_inner,
            self.conv_dim,
            self.delta_softplus,
            str(self.state_dtype),
            self.token_major_conv,
            self.has_dt_bias,
            self.has_d,
            str(self.io_dtype),
        )


class MambaSSDProvider:
    """Plans the SSD prefill launches of every distinct Mamba2 layer shape."""

    def __init__(self, model):
        from .mamba2_mixer import Mamba2Mixer

        shapes = {}
        for m in model.modules():
            if isinstance(m, Mamba2Mixer):
                s = _MixerShape(m)
                shapes.setdefault(s.key(), s)
        self.shapes = list(shapes.values())
        self._seen_keys = set()

    def __bool__(self):
        return bool(self.shapes)

    @staticmethod
    def _int_class(v: int) -> int:
        # Triton's integer specialization: == 1, divisible by 16, other.
        return 1 if v == 1 else (16 if v % 16 == 0 else 0)

    def batch_key(self, ctx_lens: List[int], any_cached: bool) -> tuple:
        """Coarse dedup key: re-plan only when a planned launch could differ.

        Covers every batch-dependent int that reaches a kernel signature:
        token count, sequence count, and the chunk counts derived from them
        (``nchunks`` and the chunk-index length ``N``, which adds one chunk per
        sequence boundary that is not chunk-aligned). Planning itself is exact;
        this only decides whether to plan. A miss here costs one extra plan
        (tens of ms), never a wrong key.
        """
        from .mamba2_metadata import compute_extra_chunks_cpu

        n = len(ctx_lens)
        tokens = sum(ctx_lens)
        classes = []
        for s in self.shapes:
            nchunks = -(-tokens // s.chunk_size)
            extra = compute_extra_chunks_cpu(ctx_lens, n, s.chunk_size)
            classes.append((self._int_class(nchunks), self._int_class(nchunks + extra)))
        return (
            min(n, 2),
            any_cached,
            self._int_class(tokens),
            self._int_class(n),
            self._int_class(n + 1),
            tuple(classes),
        )

    def __call__(self, batch_ctx: Any) -> List[KernelCall]:
        ctx_lens: List[int] = batch_ctx.ctx_chunk_lens
        if not ctx_lens:
            return []
        any_cached: bool = batch_ctx.any_ctx_cached
        key = self.batch_key(ctx_lens, any_cached)
        if key in self._seen_keys:
            return []
        self._seen_keys.add(key)
        calls: List[KernelCall] = []
        for s in self.shapes:
            calls.extend(self._plan_one(s, ctx_lens, any_cached))
        return calls

    def _plan_one(self, s: _MixerShape, ctx_lens: List[int], any_cached: bool) -> List[KernelCall]:
        from .mamba2_metadata import (
            compute_extra_chunks_cpu,
            cu_seqlens_to_chunk_indices_offsets_triton,
        )
        from .ssd_combined import _mamba_chunk_scan_combined_fwd

        meta = torch.device("meta")
        n = len(ctx_lens)
        T = sum(ctx_lens)
        # in_proj output: [T, zxbcdt_width] contiguous; dt is a column slice.
        zxbcdt = torch.empty(T, s.zxbcdt_width, dtype=s.io_dtype, device=meta)
        dt = zxbcdt[:, s.d_inner + s.conv_dim :].unsqueeze(0)
        bc = s.ngroups * s.d_state
        if s.token_major_conv:
            # Mamba2Mixer.forward_core: empty(T, conv_dim).t() is the conv's
            # channel-last output, and .t() again gives the token-major
            # [T, conv_dim] view that x/B/C are sliced from.
            xbc = torch.empty(T, s.conv_dim, dtype=s.io_dtype, device=meta)
            x = xbc[:, : s.d_inner].view(T, s.nheads, s.head_dim).unsqueeze(0)
            B = xbc[:, s.d_inner : s.d_inner + bc].view(T, s.ngroups, s.d_state).unsqueeze(0)
            C = xbc[:, s.d_inner + bc :].view(T, s.ngroups, s.d_state).unsqueeze(0)
        else:
            x = torch.empty(T, s.d_inner, dtype=s.io_dtype, device=meta).view(
                1, T, s.nheads, s.head_dim
            )
            B = torch.empty(T, bc, dtype=s.io_dtype, device=meta).view(1, T, s.ngroups, s.d_state)
            C = torch.empty(T, bc, dtype=s.io_dtype, device=meta).view(1, T, s.ngroups, s.d_state)
        A = torch.empty(s.nheads, dtype=torch.float32, device=meta)
        D = torch.empty(s.nheads, dtype=torch.float32, device=meta) if s.has_d else None
        dt_bias = torch.empty(s.nheads, dtype=torch.float32, device=meta) if s.has_dt_bias else None
        cu_seqlens = torch.empty(n + 1, dtype=torch.int, device=meta)
        seq_idx = torch.empty(1, T, dtype=torch.int, device=meta)
        out = torch.empty(T, s.nheads * s.head_dim, dtype=s.io_dtype, device=meta).view(
            1, T, -1, s.head_dim
        )
        initial_states: Optional[torch.Tensor] = None
        chunk_indices = chunk_offsets = None

        with shadow_launches() as rec:
            if any_cached:
                # Mamba2Metadata.prepare: only built when some request has a
                # cached prefix. The multi-seq branch launches the kernel; the
                # single-seq fast path is torch-only.
                extra = compute_extra_chunks_cpu(ctx_lens, n, s.chunk_size)
                chunk_indices, chunk_offsets = cu_seqlens_to_chunk_indices_offsets_triton(
                    cu_seqlens, s.chunk_size, total_seqlens=T, extra_chunks=extra
                )
                # forward_core: torch.where(...) over the gathered SSM states
                # yields a fresh contiguous tensor in the SSM-cache dtype.
                initial_states = torch.empty(
                    n, s.nheads, s.head_dim, s.d_state, dtype=s.state_dtype, device=meta
                )
            _mamba_chunk_scan_combined_fwd(
                x,
                dt,
                A,
                B,
                C,
                s.chunk_size,
                D=D,
                z=None,
                dt_bias=dt_bias,
                initial_states=initial_states,
                seq_idx=seq_idx,
                chunk_indices=chunk_indices,
                chunk_offsets=chunk_offsets,
                cu_seqlens=cu_seqlens,
                dt_softplus=s.delta_softplus,
                dt_limit=(0.0, float("inf")),
                out=out,
                state_dtype=s.state_dtype,
            )
        return list(rec)
