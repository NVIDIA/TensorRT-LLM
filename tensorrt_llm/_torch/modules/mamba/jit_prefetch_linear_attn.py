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
"""Triton variant provider for the GDN (Qwen3.5-3.8) and KDA (Kimi K3) layers.

Covers the Triton launches whose compiled variant depends on the scheduled
batch, outside the CUDA graphs that warmup captures:

* ``_rms_norm_gated_fwd_multirow_kernel`` (GDN and KDA output gate):
  ``LAUNCH_WITH_PDL`` flips at ``ceil(tokens * heads / 4) == num_SMs`` and
  ``M`` takes Triton's integer classes (== 1, % 16, other).
* GDN prefill / mixed-batch input prep: ``_extract_transpose_prefill_kernel``
  and ``_fused_gdn_post_conv_kernel`` (``HAS_DECODE`` and the prefill / decode
  token counts' integer classes).

Each launcher is replayed unchanged on ``meta`` tensors under
``shadow_launches()``, with the tensors entering it laid out as the mixer
lays them out (column slices of the in_proj output, a token-major conv
output). The launchers read nothing from tensor contents; the only device
access, ``torch.cuda.device(x.device.index)`` in the gated norm and its SM
count lookup, is pointed at the executor's device for the replay.
"""

import contextlib
from types import SimpleNamespace
from typing import Any, Iterator, List, Optional

import torch

from ...jit_prefetch import KernelCall, shadow_launches


def _int_class(v: int) -> int:
    return 1 if v == 1 else (16 if v % 16 == 0 else 0)


class _GdnShape:
    def __init__(self, m):
        self.k_heads = m.num_k_heads_per_tp
        self.v_heads = m.num_v_heads_per_tp
        self.k_dim = m.head_k_dim
        self.v_dim = m.head_v_dim
        self.conv_dim = m.conv_dim_per_tp
        self.dtype = (
            m.in_proj_qkvz.weight.dtype
            if m.in_proj_qkvz.weight.dtype.is_floating_point
            and m.in_proj_qkvz.weight.dtype.itemsize >= 2
            else torch.bfloat16
        )  # noqa: E501
        self.qkvz_width = self.conv_dim + self.v_heads * self.v_dim
        self.fp8_norm = getattr(m.norm, "fp8_scale", None) is not None

    def key(self):
        return ("gdn", self.k_heads, self.v_heads, self.k_dim, self.v_dim, self.conv_dim,
                str(self.dtype), self.fp8_norm)  # fmt: skip


class _KdaShape:
    def __init__(self, m):
        self.heads = m.num_heads
        self.head_dim = m.head_dim
        self.dtype = torch.bfloat16

    def key(self):
        return ("kda", self.heads, self.head_dim)


@contextlib.contextmanager
def _device_index_for_meta(device_index: int):
    """Let launchers that enter ``torch.cuda.device(t.device.index)`` run on
    meta tensors: inside, ``torch.cuda.device(None)`` targets ``device_index``."""
    orig = torch.cuda.device

    class _Dev(orig):
        def __init__(self, device):
            super().__init__(device_index if device is None else device)

    torch.cuda.device = _Dev
    try:
        yield
    finally:
        torch.cuda.device = orig


class LinearAttnProvider:
    """Plans the batch-dependent Triton launches of GDN and KDA layers.

    A batch is described by its context chunk lengths and its number of
    generation tokens (``gen_tokens``); launches inside captured CUDA graphs
    (pure generation batches at a captured size) are skipped by the caller.
    """

    def __init__(self, model, max_num_tokens: int = 0):
        from .gdn_mixer import GatedDeltaNet

        try:
            from ..kimi_kda.kimi_kda_mixer import KimiKDALinearAttention
        except ImportError:  # optional dependency (CuTe DSL / FLA)
            KimiKDALinearAttention = ()
        shapes = {}
        for m in model.modules():
            if isinstance(m, GatedDeltaNet):
                s = _GdnShape(m)
            elif KimiKDALinearAttention and isinstance(m, KimiKDALinearAttention):
                s = _KdaShape(m)
            else:
                continue
            shapes.setdefault(s.key(), s)
        self.shapes = list(shapes.values())
        self.max_num_tokens = int(max_num_tokens)
        self.device_index = torch.cuda.current_device() if torch.cuda.is_available() else 0
        self._seen: set = set()
        self._num_sms = None

    def __bool__(self):
        return bool(self.shapes)

    def _sms(self) -> int:
        if self._num_sms is None:
            from .layernorm_gated import _pdl_device_policy

            self._num_sms = _pdl_device_policy(self.device_index)[1]
        return self._num_sms

    def batch_key(self, ctx_lens: List[int], gen_tokens: int) -> tuple:
        from .layernorm_gated import _MULTIROW_ROWS

        p = sum(ctx_lens)
        t = p + gen_tokens
        pdl = []
        for s in self.shapes:
            heads = s.v_heads if isinstance(s, _GdnShape) else s.heads
            m = t * heads
            pdl.append(-(-m // _MULTIROW_ROWS) < self._sms())
        return (bool(ctx_lens), gen_tokens > 0, _int_class(p), _int_class(gen_tokens),
                _int_class(t), tuple(pdl))  # fmt: skip

    def __call__(self, batch_ctx: Any, seen: Optional[set] = None) -> List[KernelCall]:
        ctx_lens = list(batch_ctx.ctx_chunk_lens)
        gen_tokens = int(getattr(batch_ctx, "gen_tokens", 0))
        if not ctx_lens and gen_tokens == 0:
            return []
        seen = self._seen if seen is None else seen
        key = self.batch_key(ctx_lens, gen_tokens)
        if key in seen:
            return []
        seen.add(key)
        calls: List[KernelCall] = []
        for s in self.shapes:
            if isinstance(s, _GdnShape):
                calls.extend(self._plan_gdn(s, sum(ctx_lens), gen_tokens))
            else:
                calls.extend(self._plan_kda(s, sum(ctx_lens) + gen_tokens))
        return calls

    def enumerate_batches(self) -> Iterator[Any]:
        """One batch per class reachable within max_num_tokens: prefill-only
        and mixed batches across the token-count integer classes and the PDL
        threshold of every layer shape."""
        T = self.max_num_tokens
        if T <= 0:
            return
        sms = self._sms()
        from .layernorm_gated import _MULTIROW_ROWS

        cands = {1, 2, 15, 16, 17, 31, 32, 33, T - 1, T}
        for s in self.shapes:
            heads = s.v_heads if isinstance(s, _GdnShape) else s.heads
            edge = (sms * _MULTIROW_ROWS) // max(1, heads)
            cands.update({edge - 1, edge, edge + 1, edge + 16})
        cands = sorted(c for c in cands if 1 <= c <= T)
        seen = set()
        for p in cands:
            for g in (0, 1, 16, 17):
                if p + g > T:
                    continue
                k = self.batch_key([p], g)
                if k in seen:
                    continue
                seen.add(k)
                yield SimpleNamespace(ctx_chunk_lens=[p], gen_tokens=g, any_ctx_cached=False)

    # -- per-layer planners ------------------------------------------------
    def _gated_norm(self, tokens: int, heads: int, dim: int, z: torch.Tensor, fp8: bool,
                    gate: str):  # fmt: skip
        from .layernorm_gated import rms_norm_gated_token_major

        meta = torch.device("meta")
        x = torch.empty(tokens * heads, dim, dtype=z.dtype, device=meta)
        w = torch.empty(dim, dtype=z.dtype, device=meta)
        scale = torch.empty((), dtype=torch.float32, device=meta) if fp8 else None
        # Call the op's Python body (CustomOpDef._init_fn): calling the op
        # itself on meta tensors dispatches to its fake impl and launches
        # nothing.
        body = getattr(rms_norm_gated_token_major, "_init_fn", rms_norm_gated_token_major)
        body(x, z, w, 1e-6, fp8_scale=scale, gate_activation=gate)

    def _plan_gdn(self, s: _GdnShape, p: int, d: int) -> List[KernelCall]:
        from .fuse_elementwise_ops import extract_transpose_prefill_slice, fused_gdn_post_conv

        meta = torch.device("meta")
        t = p + d
        qkvz = torch.empty(t, s.qkvz_width, dtype=s.dtype, device=meta)
        ba = torch.empty(t, 2 * s.v_heads, dtype=s.dtype, device=meta)
        mixed_qkv = qkvz[:, : s.conv_dim]
        z = qkvz[:, s.conv_dim :].view(t, s.v_heads, s.v_dim)
        b, a = ba[:, : s.v_heads], ba[:, s.v_heads :]
        A_log = torch.empty(s.v_heads, dtype=torch.float32, device=meta)
        dt_bias = torch.empty(s.v_heads, dtype=torch.float32, device=meta)
        with shadow_launches() as rec, _device_index_for_meta(self.device_index):
            if p > 0:
                src = mixed_qkv[:p] if d > 0 else mixed_qkv
                extract_transpose_prefill_slice(src, src.shape[0], 0, src.shape[1])
                # causal_conv1d_fn (CUDA) writes a token-major [p, conv] buffer
                # and returns its [conv, p] transpose.
                conv_p = torch.empty(p, s.conv_dim, dtype=s.dtype, device=meta).t()
                conv_d = torch.empty(d, s.conv_dim, dtype=s.dtype, device=meta) if d > 0 else None
                fused_gdn_post_conv(conv_p, conv_d, a, b, A_log, dt_bias, s.k_heads, s.k_dim,
                                    s.v_heads, s.v_dim)  # fmt: skip
            self._gated_norm(t, s.v_heads, s.v_dim, z, s.fp8_norm, "silu")
        return list(rec)

    def _plan_kda(self, s: _KdaShape, t: int) -> List[KernelCall]:
        meta = torch.device("meta")
        g_out = torch.empty(t, s.heads * s.head_dim, dtype=s.dtype, device=meta).view(
            t, s.heads, s.head_dim
        )
        with shadow_launches() as rec, _device_index_for_meta(self.device_index):
            self._gated_norm(t, s.heads, s.head_dim, g_out, False, "sigmoid")
        return list(rec)
