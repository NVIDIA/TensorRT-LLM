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
"""``trtllm::k3_kda_qkvg``: Kimi K3's fused KDA projection (TP16 slice, W [3208, 7168]) as a three-phase stream.

Outputs (Lamport buffers, see the kernel module): ``p1`` int16 [3, 8, 1664] holds the bf16 bits of the q, k and f_a
columns; ``part`` int32 [3, 3, 2, 8, 768] the fp32 bits of the two K-half partial sums of v (region 0), og (1) and b
(2, first 8 rows), whose bf16-rounded sum is the projection. A launch writes buffer e = ``epoch[cta]`` and resets buffer
(e + 1) % 3 to the sentinel; ``epoch`` int32 [104] holds each CTA's launch count mod 3. The buffers must persist across
launches and start as all-ones (``p1``, ``part``) and zeros (``epoch``). The kernel compiles on the first call for each
configuration, which must happen outside CUDA-graph capture.
"""

from __future__ import annotations

import os
import threading
from typing import Dict

import torch

K_IN = 7168
PROJ_ROWS = 3208
P1_NUMEL = 3 * 8 * 1664
PART_NUMEL = 3 * 3 * 2 * 8 * 768
CTAS = 104
FUSED_CTAS = 128
NT = 8  # one request's verify tokens (golden + 7 drafts)

_lock = threading.Lock()
_compiled: Dict[tuple, object] = {}


def _arg(t: torch.Tensor, align: int = 16):
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(t.detach(), assumed_align=align).mark_layout_dynamic(leading_dim=t.dim() - 1)


def _index_arg(t: torch.Tensor):
    """An int32 index tensor the kernels read element by element (``slots``, ``pending``), declared at its element's
    alignment: the mixer passes slices such as ``state_indices[num_prefills:]``, which start on any 4-byte boundary."""
    return _arg(t, t.element_size())


def use_pdl() -> bool:
    return os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"


def make_buffers(device: torch.device, ctas: int = CTAS):
    """Fresh (p1, part, epoch) for :func:`k3_kda_qkvg` (``ctas`` = 104) or :func:`k3_kda_attn` (``FUSED_CTAS``)."""
    p1 = torch.full((P1_NUMEL,), -1, dtype=torch.int16, device=device)
    part = torch.full((PART_NUMEL,), -1, dtype=torch.int32, device=device)
    epoch = torch.zeros(ctas, dtype=torch.int32, device=device)
    return p1, part, epoch


@torch.library.custom_op(
    "trtllm::k3_kda_qkvg", mutates_args=("p1", "part", "epoch"), device_types="cuda"
)
def k3_kda_qkvg(
    x: torch.Tensor,
    w: torch.Tensor,
    p1: torch.Tensor,
    part: torch.Tensor,
    epoch: torch.Tensor,
) -> None:
    """x bf16 [T <= 8, 7168], w bf16 [3208, 7168] (both contiguous); writes p1 and part, advances epoch (see the
    module doc)."""
    import cuda.bindings.driver as cuda_driver

    from . import k3_kda_attn_kernel as kernel

    tokens = x.shape[0]
    if (
        x.dtype != torch.bfloat16
        or w.dtype != torch.bfloat16
        or tuple(w.shape) != (PROJ_ROWS, K_IN)
        or x.dim() != 2
        or x.shape[1] != K_IN
        or not 1 <= tokens <= 8
        or not x.is_contiguous()
        or not w.is_contiguous()
        or p1.numel() != P1_NUMEL
        or p1.dtype != torch.int16
        or part.numel() != PART_NUMEL
        or part.dtype != torch.int32
        or epoch.numel() != CTAS
        or epoch.dtype != torch.int32
    ):
        raise ValueError(
            f"k3_kda_qkvg: unsupported call x {tuple(x.shape)} {x.dtype}, w {tuple(w.shape)} {w.dtype}, "
            f"p1 {p1.numel()} {p1.dtype}, part {part.numel()} {part.dtype}, epoch {epoch.numel()} {epoch.dtype}"
        )
    args = (_arg(w), _arg(x), _arg(p1.view(-1)), _arg(part.view(-1)), _arg(epoch.view(-1)))
    stream = cuda_driver.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    pdl = use_pdl()
    key = (pdl,)
    fn = _compiled.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "trtllm::k3_kda_qkvg must run once per configuration outside CUDA-graph capture first "
                "(it compiles its kernel on the first call)."
            )
        import cutlass.cute as cute

        with _lock:
            fn = _compiled.get(key)
            if fn is None:
                fn = _compiled[key] = cute.compile(kernel.k3_kda_qkvg, *args, tokens, pdl, stream)
    fn(*args, tokens, stream)


@k3_kda_qkvg.register_fake
def _(x, w, p1, part, epoch):
    return None


@torch.library.custom_op(
    "trtllm::k3_kda_attn",
    mutates_args=("cs_q", "cs_k", "cs_v", "ssm", "state_tok", "p1", "part", "epoch"),
    device_types="cuda",
)
def k3_kda_attn(
    x: torch.Tensor,
    w: torch.Tensor,
    w_fb: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    onorm_w: torch.Tensor,
    cs_q: torch.Tensor,
    cs_k: torch.Tensor,
    cs_v: torch.Tensor,
    ssm: torch.Tensor,
    state_tok: torch.Tensor,
    slots: torch.Tensor,
    pending: torch.Tensor,
    p1: torch.Tensor,
    part: torch.Tensor,
    epoch: torch.Tensor,
    num_spec: int,
    lower_bound: float,
    scale: float,
    eps: float,
) -> torch.Tensor:
    """Stage B2: the fused projection of ``x`` (bf16 [8, 7168], one request's golden token and 7 drafts) and the KDA
    verify of ``trtllm::k3_kda_verify`` on it, in one launch; returns the gated-norm core output bf16 [8, 6, 128].
    Weights, pools and state contract as ``k3_kda_verify`` (``slots`` int32 [1]); ``p1``, ``part`` and ``epoch``
    from :func:`make_buffers` with ``FUSED_CTAS``, persistent across launches. The head CTAs read the slot's pools
    before the grid-dependency wait, so a launch must not follow another launch on the same pools directly in the
    stream: a kernel that waits (or a non-PDL one) in between, as the model's other layers are."""
    import cuda.bindings.driver as cuda_driver

    from ..k3_kda_verify.op import _flat, _slots_view
    from . import k3_kda_attn_kernel as kernel

    if (
        tuple(x.shape) != (NT, K_IN)
        or x.dtype != torch.bfloat16
        or not x.is_contiguous()
        or tuple(w.shape) != (PROJ_ROWS, K_IN)
        or w.dtype != torch.bfloat16
        or not w.is_contiguous()
        or tuple(w_fb.shape) != (768, 128)
        or w_fb.dtype != torch.bfloat16
        or not w_fb.is_contiguous()
        or num_spec != NT - 1
        or tuple(ssm.shape[1:]) != (6, 128, 128)
        or tuple(state_tok.shape[1:]) != (num_spec, 6, 128, 128)
        or ssm.stride()[1:] != (128 * 128, 128, 1)
        or not state_tok.is_contiguous()
        or cs_q.shape[-1] != 3 + num_spec
        or slots.numel() != 1
        or slots.dtype != torch.int32
        or pending.dtype != torch.int32
        or p1.numel() != P1_NUMEL
        or part.numel() != PART_NUMEL
        or epoch.numel() != FUSED_CTAS
        or ssm.dtype != torch.float32
        or state_tok.dtype != torch.float32
        or any(
            t.dtype != torch.float32
            for t in (w_q, w_k, w_v, a_log, dt_bias, onorm_w, cs_q, cs_k, cs_v)
        )
    ):
        raise ValueError(
            f"k3_kda_attn: unsupported call x {tuple(x.shape)} {x.dtype}, w {tuple(w.shape)}, "
            f"w_fb {tuple(w_fb.shape)}, ssm {tuple(ssm.shape)}, state_tok {tuple(state_tok.shape)}, "
            f"cs_q {tuple(cs_q.shape)}, "
            f"num_spec {num_spec}, slots {tuple(slots.shape)} {slots.dtype}, epoch {epoch.numel()}"
        )
    out = torch.empty(NT, 6, 128, dtype=torch.bfloat16, device=x.device)
    args = (
        _arg(w), _arg(x), _arg(w_fb), _arg(p1.view(-1)), _arg(p1.view(torch.int32)), _arg(part.view(-1)),
        _arg(epoch.view(-1)), _arg(_flat(w_q)), _arg(_flat(w_k)), _arg(_flat(w_v)), _arg(_flat(a_log)),
        _arg(_flat(dt_bias)), _arg(_flat(onorm_w)), _arg(_flat(cs_q)), _arg(_flat(cs_k)), _arg(_flat(cs_v)),
        _arg(_slots_view(ssm)), _arg(_slots_view(state_tok)), _index_arg(slots), _index_arg(pending),
        _arg(_flat(out)),
    )  # fmt: skip
    stream = cuda_driver.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    pdl = use_pdl()
    key = ("attn", float(lower_bound), float(scale), float(eps), pdl)
    fn = _compiled.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "trtllm::k3_kda_attn must run once per configuration outside CUDA-graph capture first "
                "(it compiles its kernel on the first call)."
            )
        import cutlass.cute as cute

        with _lock:
            fn = _compiled.get(key)
            if fn is None:
                fn = _compiled[key] = cute.compile(
                    kernel.k3_kda_attn, *args, ssm.stride(0), float(lower_bound), float(scale), float(eps), pdl,
                    stream,
                )  # fmt: skip
    fn(*args, ssm.stride(0), stream)
    return out


@k3_kda_attn.register_fake
def _(x, w, w_fb, w_q, w_k, w_v, a_log, dt_bias, onorm_w, cs_q, cs_k, cs_v, ssm, state_tok, slots, pending, p1, part,
      epoch, num_spec, lower_bound, scale, eps):  # fmt: skip
    return x.new_empty((NT, 6, 128), dtype=torch.bfloat16)


@torch.library.custom_op(
    "trtllm::k3_kda_decode_attn",
    mutates_args=("conv", "ssm", "p1", "part", "epoch"),
    device_types="cuda",
)
def k3_kda_decode_attn(
    x: torch.Tensor,
    w: torch.Tensor,
    w_fb: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    onorm_w: torch.Tensor,
    conv: torch.Tensor,
    ssm: torch.Tensor,
    slots: torch.Tensor,
    p1: torch.Tensor,
    part: torch.Tensor,
    epoch: torch.Tensor,
    lower_bound: float,
    scale: float,
    eps: float,
) -> torch.Tensor:
    """The fused projection of ``x`` (bf16 [R <= 8, 7168], one token of each of R requests) and the KDA plain decode
    of ``trtllm::kda_decode`` on it, in one launch; returns the gated-norm core output bf16 [R, 6, 128].

    ``conv`` bf16 [slots, 2304, 3] (q | k | v channels, the last three raw inputs; each slot dense, the slot stride a
    multiple of 8 elements) and ``ssm`` fp32 [slots, 6, 128, 128] (each slot dense, the slot stride a multiple of 4)
    are updated in place at ``slots`` int32 [R]; ``w_q / w_k / w_v`` fp32 [768, 4]; ``a_log`` fp32 [6];
    ``dt_bias`` fp32 [768]; ``onorm_w`` fp32 [128]; ``p1``, ``part`` and ``epoch`` from :func:`make_buffers` with
    ``FUSED_CTAS``, persistent across launches. The head CTAs read the slots' pools before the grid-dependency wait,
    so a launch must not follow another launch on the same pools directly in the stream: a kernel that waits (or a
    non-PDL one) in between, as the model's other layers are."""
    import cuda.bindings.driver as cuda_driver

    from ..k3_kda_verify.op import _flat, _slots_view
    from . import k3_kda_decode_kernel as kernel

    n_req = x.shape[0] if x.dim() == 2 else 0
    if (
        x.dim() != 2
        or not 1 <= n_req <= NT
        or x.shape[1] != K_IN
        or x.dtype != torch.bfloat16
        or not x.is_contiguous()
        or tuple(w.shape) != (PROJ_ROWS, K_IN)
        or w.dtype != torch.bfloat16
        or not w.is_contiguous()
        or tuple(w_fb.shape) != (768, 128)
        or w_fb.dtype != torch.bfloat16
        or not w_fb.is_contiguous()
        or conv.dtype != torch.bfloat16
        or conv.dim() != 3
        or tuple(conv.shape[1:]) != (3 * 768, 3)
        or conv.stride()[1:] != (3, 1)
        or conv.stride(0) % 8 != 0
        or conv.data_ptr() % 16 != 0
        or ssm.dtype != torch.float32
        or tuple(ssm.shape[1:]) != (6, 128, 128)
        or ssm.stride()[1:] != (128 * 128, 128, 1)
        or ssm.stride(0) % 4 != 0
        or ssm.data_ptr() % 16 != 0
        or slots.numel() != n_req
        or slots.dtype != torch.int32
        or not slots.is_contiguous()
        or p1.numel() != P1_NUMEL
        or part.numel() != PART_NUMEL
        or epoch.numel() != FUSED_CTAS
        or any(t.dtype != torch.float32 for t in (w_q, w_k, w_v, a_log, dt_bias, onorm_w))
    ):
        raise ValueError(
            f"k3_kda_decode_attn: unsupported call x {tuple(x.shape)} {x.dtype}, w {tuple(w.shape)}, "
            f"w_fb {tuple(w_fb.shape)}, conv {tuple(conv.shape)} {conv.dtype} strides {conv.stride()}, "
            f"ssm {tuple(ssm.shape)} {ssm.dtype} strides {ssm.stride()}, slots {tuple(slots.shape)} {slots.dtype}, "
            f"epoch {epoch.numel()}"
        )
    out = torch.empty(n_req, 6, 128, dtype=torch.bfloat16, device=x.device)
    args = (
        _arg(w), _arg(x), _arg(w_fb), _arg(p1.view(-1)), _arg(p1.view(torch.int32)), _arg(part.view(-1)),
        _arg(epoch.view(-1)), _arg(_flat(w_q)), _arg(_flat(w_k)), _arg(_flat(w_v)), _arg(_flat(a_log)),
        _arg(_flat(dt_bias)), _arg(_flat(onorm_w)), _arg(_slots_view(conv)), _arg(_slots_view(ssm)),
        _index_arg(slots), _arg(_flat(out)),
    )  # fmt: skip
    runtime = (n_req, ssm.stride(0), conv.stride(0))
    stream = cuda_driver.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    pdl = use_pdl()
    key = ("decode", float(lower_bound), float(scale), float(eps), pdl)
    fn = _compiled.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "trtllm::k3_kda_decode_attn must run once per configuration outside CUDA-graph capture first "
                "(it compiles its kernel on the first call)."
            )
        import cutlass.cute as cute

        with _lock:
            fn = _compiled.get(key)
            if fn is None:
                fn = _compiled[key] = cute.compile(
                    kernel.k3_kda_decode, *args, *runtime, float(lower_bound), float(scale), float(eps), pdl,
                    stream,
                )  # fmt: skip
    fn(*args, *runtime, stream)
    return out


@k3_kda_decode_attn.register_fake
def _(x, w, w_fb, w_q, w_k, w_v, a_log, dt_bias, onorm_w, conv, ssm, slots, p1, part, epoch, lower_bound, scale,
      eps):  # fmt: skip
    return x.new_empty((x.shape[0], 6, 128), dtype=torch.bfloat16)
