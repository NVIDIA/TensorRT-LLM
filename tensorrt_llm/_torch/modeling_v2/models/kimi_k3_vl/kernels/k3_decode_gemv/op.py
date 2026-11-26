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
"""``trtllm::k3_decode_gemv``: ``x[M, K] @ weight[N, K]^T`` for M <= 8 in CuTe DSL (bf16, fp32 accumulation).

The kernel loads its whole weight slice before waiting for the producer of ``x``, so launched with
PDL behind a long predecessor it only pays for the activation load, the MMA and the store. It is
compiled on the first call for each (N, K, early trigger, PDL), which must happen outside CUDA-graph
capture.
"""

from __future__ import annotations

import os
import threading
from typing import Dict

import torch

MAX_TOKENS = 8

_lock = threading.Lock()
_compiled: Dict[tuple, object] = {}


def _arg(t: torch.Tensor):
    from cutlass.cute.runtime import from_dlpack

    # detach(): DLPack refuses tensors that require grad (weights are parameters).
    return from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=t.dim() - 1)


def _use_pdl() -> bool:
    return os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"


def supports(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether the kernel runs this call (shapes, dtypes, layout)."""
    from . import k3_decode_gemv_kernel as kernel

    return (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.dim() == 2
        and weight.dim() == 2
        and 0 < x.shape[0] <= MAX_TOKENS
        and x.shape[1] == weight.shape[1]
        and x.is_contiguous()
        and weight.is_contiguous()
        and (
            kernel.supports(weight.shape[0], weight.shape[1])
            or kernel.splitk_supports(weight.shape[0], weight.shape[1])
        )
    )


@torch.library.custom_op("trtllm::k3_decode_gemv", mutates_args=())
def k3_decode_gemv(
    x: torch.Tensor, weight: torch.Tensor, trigger_early: bool = True
) -> torch.Tensor:
    """``x @ weight.T`` for bf16 ``x`` [M <= 8, K] and ``weight`` [N, K]; returns bf16 [M, N].

    ``trigger_early`` lets the dependent grid launch once every CTA has issued its weight loads
    (for dependents that wait for this whole grid before reading the output)."""
    import cuda.bindings.driver as cuda_driver

    if not supports(x, weight):
        raise ValueError(
            f"k3_decode_gemv: unsupported call x {tuple(x.shape)} {x.dtype}, weight {tuple(weight.shape)} "
            f"{weight.dtype}"
        )
    num_tokens, k_in = x.shape
    n_out = weight.shape[0]
    y = torch.empty(num_tokens, n_out, dtype=torch.bfloat16, device=x.device)
    args = (_arg(weight), _arg(x), _arg(y.view(-1)))
    stream = cuda_driver.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    use_pdl = _use_pdl()
    key = (n_out, k_in, trigger_early, use_pdl)
    fn = _compiled.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "trtllm::k3_decode_gemv must run once per shape outside CUDA-graph capture first "
                "(it compiles its kernel on the first call)."
            )
        import cutlass.cute as cute

        from . import k3_decode_gemv_kernel as kernel

        # Short K: one CTA per weight tile, whole slice resident; long K: split-K over a cluster.
        entry = (
            kernel.k3_decode_gemv if kernel.supports(n_out, k_in) else kernel.k3_decode_gemv_splitk
        )
        with _lock:
            fn = _compiled.get(key)
            if fn is None:
                fn = _compiled[key] = cute.compile(
                    entry, *args, num_tokens, n_out, k_in, trigger_early, use_pdl, stream
                )
    # The compiled function takes the runtime arguments only.
    fn(*args, num_tokens, stream)
    return y


@k3_decode_gemv.register_fake
def _(x, weight, trigger_early=True):
    return x.new_empty((x.shape[0], weight.shape[0]), dtype=torch.bfloat16)


def supports_tail(
    latent: torch.Tensor, act: torch.Tensor, weight: torch.Tensor, width: int
) -> bool:
    """Whether the tail kernel runs this call: the latent part of the weight a whole number of k-tiles
    covering the slice, the activation too, at most MAX_K_TILES in all, a latent row the RMS loop tiles."""
    from . import k3_decode_gemv_kernel as kernel

    k_act = act.shape[1] if act.dim() == 2 else -1
    k_lat = weight.shape[1] - k_act
    return (
        latent.is_cuda
        and latent.dtype == act.dtype == weight.dtype == torch.bfloat16
        and latent.dim() == 2
        and act.dim() == 2
        and weight.dim() == 2
        and 0 < latent.shape[0] <= MAX_TOKENS
        and act.shape[0] == latent.shape[0]
        and latent.is_contiguous()
        and act.is_contiguous()
        and weight.is_contiguous()
        and k_lat % kernel.CTA_K == 0
        and k_act % kernel.CTA_K == 0
        and 0 < width <= k_lat
        and latent.shape[1] % (8 * 32) == 0
        and kernel.supports(weight.shape[0], weight.shape[1])
    )


@torch.library.custom_op("trtllm::k3_decode_gemv_tail", mutates_args=())
def k3_decode_gemv_tail(
    latent: torch.Tensor,
    act: torch.Tensor,
    weight: torch.Tensor,
    lo: int,
    width: int,
    eps: float,
    trigger_early: bool = True,
) -> torch.Tensor:
    """The row-parallel MoE tail, ``[rmsnorm(latent)[:, lo:lo+width] | act] @ weight.T``: ``latent``
    bf16 [M, H] is the whole reduced latent row, whose RMS scales the latent part; ``weight`` is
    ``[latent up columns of the slice, zero-padded to a multiple of 128 | shared down]``. The RMS is
    applied to the fp32 latent accumulator, not to bf16 inputs."""
    import cuda.bindings.driver as cuda_driver

    if not supports_tail(latent, act, weight, width):
        raise ValueError(
            f"k3_decode_gemv_tail: unsupported call latent {tuple(latent.shape)}, act {tuple(act.shape)}, "
            f"weight {tuple(weight.shape)}, width {width}"
        )
    from . import k3_decode_gemv_kernel as kernel

    num_tokens, rms_cols = latent.shape
    n_out, k_in = weight.shape
    lat_tiles = (k_in - act.shape[1]) // kernel.CTA_K
    y = torch.empty(num_tokens, n_out, dtype=torch.bfloat16, device=latent.device)
    args = (
        _arg(weight),
        _arg(latent),
        _arg(latent.view(-1).view(torch.int32)),
        _arg(act),
        _arg(y.view(-1)),
    )
    stream = cuda_driver.CUstream(torch.cuda.current_stream(latent.device).cuda_stream)
    use_pdl = _use_pdl()
    key = ("tail", n_out, k_in, lat_tiles, rms_cols, trigger_early, use_pdl)
    fn = _compiled.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "trtllm::k3_decode_gemv_tail must run once per shape outside CUDA-graph capture first "
                "(it compiles its kernel on the first call)."
            )
        import cutlass.cute as cute

        with _lock:
            fn = _compiled.get(key)
            if fn is None:
                fn = _compiled[key] = cute.compile(
                    kernel.k3_decode_gemv_tail, *args, num_tokens, lo, float(eps), n_out, k_in, lat_tiles,
                    rms_cols, trigger_early, use_pdl, stream,
                )  # fmt: skip
    fn(*args, num_tokens, lo, float(eps), stream)
    return y


@k3_decode_gemv_tail.register_fake
def _(latent, act, weight, lo, width, eps, trigger_early=True):
    return latent.new_empty((latent.shape[0], weight.shape[0]), dtype=torch.bfloat16)
