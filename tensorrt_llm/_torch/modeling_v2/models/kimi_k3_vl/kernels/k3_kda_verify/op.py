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
"""``trtllm::k3_kda_verify``: Kimi K3's KDA speculative verify for 1 + num_spec tokens per request, from the fused
projection rows to the gated-norm core output, with the drafts' records for the next round.

State contract (differs from ``trtllm::kda_mtp_decode``'s replay caches): after a call the pool ``ssm`` holds the
state after each request's golden token and ``state_tok[slot]`` the records of its drafts: the row innovations vn
[num_spec][H][V], then beta * k and the decay [num_spec][H][K]. The conv caches hold the raw inputs at positions
-2..num_spec around the golden token. The next call starts from the pool state with the drafts the sampler accepted
(``pending[slot]``) replayed from their records, S = fma(decay, S, vn (beta k)), the recurrence's own arithmetic.
The kernel compiles on the first call for each configuration, which must happen outside CUDA-graph capture.
"""

from __future__ import annotations

import os
import threading
from typing import Dict, Optional

import torch

from ._pool_layout import replay_pools_supported, slots_view

K = 128

_lock = threading.Lock()
_compiled: Dict[tuple, object] = {}


def _arg(t: torch.Tensor, align: int = 16):
    from cutlass.cute.runtime import from_dlpack

    # detach(): DLPack refuses tensors that require grad (weights are parameters).
    return from_dlpack(t.detach(), assumed_align=align).mark_layout_dynamic(leading_dim=t.dim() - 1)


def _index_arg(t: torch.Tensor):
    """An int32 index tensor the kernel reads element by element (``slots``, ``pending``), declared at its element's
    alignment: the mixer passes slices such as ``state_indices[num_prefills:]``, which start on any 4-byte boundary."""
    return _arg(t, t.element_size())


def _flat(t: torch.Tensor) -> torch.Tensor:
    """A 1-D view of a dense tensor's storage (a view with its last two dims transposed is fine), never a copy."""
    if t.is_contiguous():
        return t.view(-1)
    swapped = t.transpose(-1, -2)
    if swapped.is_contiguous():
        return swapped.reshape(-1)
    raise ValueError(f"expected a dense tensor, got shape {tuple(t.shape)} strides {t.stride()}")


def _words(t: torch.Tensor) -> torch.Tensor:
    return _flat(t).view(torch.int32)


def use_pdl() -> bool:
    return os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"


@torch.library.custom_op(
    "trtllm::k3_kda_verify",
    mutates_args=("cs_q", "cs_k", "cs_v", "ssm", "state_tok"),
    device_types="cuda",
)
def k3_kda_verify(
    proj: torch.Tensor,
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
    num_spec: int,
    lower_bound: float,
    scale: float,
    eps: float,
    g_ext: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    import cuda.bindings.driver as cuda_driver

    from . import k3_kda_verify_kernel as kernel

    n_req = slots.shape[0]
    num_heads = w_fb.shape[0] // K
    tokens = n_req * (1 + num_spec)
    v_dim = onorm_w.shape[0]
    if (
        proj.dtype != torch.bfloat16
        or proj.dim() != 2
        or proj.shape[0] != tokens
        or proj.shape[1] % 8 != 0
        or proj.shape[1] < 4 * num_heads * K + K + num_heads
        or not proj.is_contiguous()
        or tuple(w_fb.shape) != (num_heads * K, K)
        or w_fb.dtype != torch.bfloat16
        or not w_fb.is_contiguous()
        or v_dim != K
        or tuple(ssm.shape[1:]) != (num_heads, K, K)
        or not replay_pools_supported(cs_q, cs_k, cs_v, state_tok, pending, num_heads, num_spec)
        or ssm.stride()[1:] != (K * K, K, 1)
        or ssm.dtype != torch.float32
        or slots.dtype != torch.int32
        or pending.dtype != torch.int32
        or any(t.dtype != torch.float32 for t in (w_q, w_k, w_v, a_log, dt_bias, onorm_w))
    ):
        raise ValueError(
            f"k3_kda_verify: unsupported call (weights must be fp32; conv caches bf16/fp32 with dense channels: "
            f"{[str(t.dtype) for t in (w_q, w_k, w_v, a_log, dt_bias, onorm_w, cs_q, cs_k, cs_v)]}) "
            f"proj {tuple(proj.shape)} {proj.dtype} contiguous {proj.is_contiguous()}, "
            f"w_fb {tuple(w_fb.shape)} {w_fb.dtype} contiguous {w_fb.is_contiguous()}, onorm_w {tuple(onorm_w.shape)}, "
            f"ssm {tuple(ssm.shape)} {ssm.dtype} strides {ssm.stride()}, state_tok {tuple(state_tok.shape)} "
            f"{state_tok.dtype} contiguous {state_tok.is_contiguous()}, cs_q {tuple(cs_q.shape)}, num_spec {num_spec}, "
            f"slots {tuple(slots.shape)} {slots.dtype}, pending {pending.dtype} (tokens expected {tokens})"
        )
    fold_fb = g_ext is None
    out = torch.empty(tokens, num_heads, v_dim, dtype=torch.bfloat16, device=proj.device)
    args = (
        _arg(w_fb),
        _arg(_words(proj)),
        _arg(_words(proj if fold_fb else g_ext)),
        _arg(_flat(w_q)),
        _arg(_flat(w_k)),
        _arg(_flat(w_v)),
        _arg(_flat(a_log)),
        _arg(_flat(dt_bias)),
        _arg(_flat(onorm_w)),
        _arg(slots_view(cs_q.transpose(1, 2)), align=4 * cs_q.element_size()),
        _arg(slots_view(cs_k.transpose(1, 2)), align=4 * cs_k.element_size()),
        _arg(slots_view(cs_v.transpose(1, 2)), align=4 * cs_v.element_size()),
        _arg(slots_view(ssm)),
        _arg(slots_view(state_tok)),
        _index_arg(slots),
        _index_arg(pending.unsqueeze(-1)),
        _arg(_flat(out)),
    )
    runtime = (
        proj.shape[1] // 2,
        n_req,
        ssm.stride(0),
        cs_q.stride(0),
        cs_k.stride(0),
        cs_v.stride(0),
        state_tok.stride(0),
        pending.stride(0),
    )
    stream = cuda_driver.CUstream(torch.cuda.current_stream(proj.device).cuda_stream)
    pdl = use_pdl()
    conv_bf16 = cs_q.dtype == torch.bfloat16
    key = (
        num_heads,
        num_spec,
        float(lower_bound),
        float(scale),
        float(eps),
        fold_fb,
        conv_bf16,
        pdl,
    )
    fn = _compiled.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "trtllm::k3_kda_verify must run once per configuration outside CUDA-graph capture first "
                "(it compiles its kernel on the first call)."
            )
        import cutlass.cute as cute

        with _lock:
            fn = _compiled.get(key)
            if fn is None:
                fn = _compiled[key] = cute.compile(
                    kernel.k3_kda_verify, *args, *runtime, num_heads, num_spec, float(lower_bound), float(scale),
                    float(eps), fold_fb, conv_bf16, pdl, stream,
                )  # fmt: skip
    fn(*args, *runtime, stream)
    return out


@k3_kda_verify.register_fake
def _(proj, w_fb, w_q, w_k, w_v, a_log, dt_bias, onorm_w, cs_q, cs_k, cs_v, ssm, state_tok, slots, pending,
      num_spec, lower_bound, scale, eps, g_ext=None):  # fmt: skip
    return proj.new_empty(
        (proj.shape[0], w_fb.shape[0] // K, onorm_w.shape[0]), dtype=torch.bfloat16
    )
