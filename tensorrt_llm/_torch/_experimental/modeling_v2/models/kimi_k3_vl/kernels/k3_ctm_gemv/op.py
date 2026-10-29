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
"""Torch ops of the CTM decode GEMV (M <= 8, bf16, fp32 accumulation, K <= 768).

``trtllm::k3_ctm_gemv``: ``x @ weight^T`` (same results as ``trtllm::k3_decode_gemv`` with ``split=1``).
``trtllm::k3_ctm_gemv_tail``: the row-parallel MoE tail (same results as ``trtllm::k3_decode_gemv_tail``).
``trtllm::k3_ctm_gemv_gated``: ``(a * sigmoid(g)) @ weight^T`` with torch's bf16 roundings, where ``g`` is a
column window of a dense bf16 matrix (the MLA output gate inside the fused q_a/kv_a/gate projection output), or
``a * s`` when that window already holds ``s = bf16(sigmoid(g))``.
``trtllm::k3_ctm_gemv_long``: ``x @ weight^T`` for long K (split-K over a 4-8 CTA cluster, weight ring and L2
prefetch before the grid-dependency wait); output columns >= ``sig_col0`` hold ``bf16(sigmoid(bf16(.)))``.
``trtllm::k3_ctm_gemv_swiglu``: ``silu_and_mul(gu) @ weight^T`` with the activation in the B prologue.
``trtllm::k3_ctm_gemv_wide``: ``x @ weight^T`` for up to 64 tokens in one MMA of 16, 32 or 64 token columns, on the
long kernel's split-K clusters (bf16 output with optional sigmoid columns, or fp32 output).

``push`` (the split-K ops) chooses how a cluster's ranks send their fp32 partials to the row owner: DSMEM
stores + a release arrive (False) or 16-byte st.async completing the owner's barrier by bytes (True). The
sums and their order are the same; each call site takes the one measured faster at its shape.

Each kernel is compiled on the first call for its shape and flags, which must happen outside CUDA-graph
capture. The whole weight slice is loaded before the grid-dependency wait.
"""

from __future__ import annotations

import os
import threading
from typing import Dict, Optional

import torch

MAX_TOKENS = 8
WIDE_MAX_TOKENS = 64

_lock = threading.Lock()
_compiled: Dict[tuple, object] = {}


def _arg(t: torch.Tensor):
    from cutlass.cute.runtime import from_dlpack

    # detach(): DLPack refuses tensors that require grad (weights are parameters).
    return from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=t.dim() - 1)


def _use_pdl() -> bool:
    return os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"


def _stream(t: torch.Tensor):
    import cuda.bindings.driver as cuda_driver

    return cuda_driver.CUstream(torch.cuda.current_stream(t.device).cuda_stream)


def _compile(key, entry, *args):
    fn = _compiled.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"{key[0]}: run once per shape outside CUDA-graph capture first (it compiles its kernel)."
            )
        import cutlass.cute as cute

        with _lock:
            fn = _compiled.get(key)
            if fn is None:
                fn = _compiled[key] = cute.compile(entry, *args)
    return fn


def supports(x: torch.Tensor, weight: torch.Tensor, split: int = 1) -> bool:
    """Whether ``k3_ctm_gemv`` runs ``x @ weight^T``."""
    from . import k3_ctm_gemv_kernel as kernel

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
        and kernel.supports(weight.shape[0], weight.shape[1], split)
    )


@torch.library.custom_op("trtllm::k3_ctm_gemv", mutates_args=())
def k3_ctm_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    trigger_early: bool = True,
    split: int = 1,
    push: bool = False,
) -> torch.Tensor:
    """``x @ weight.T`` for bf16 ``x`` [M <= 8, K <= 768] and ``weight`` [N, K]; returns bf16 [M, N].

    ``split=2`` splits the k-tiles of each 128-row tile over a 2-CTA cluster (twice the CTAs streaming the
    weight; the two fp32 partials are added before the bf16 rounding)."""
    if not supports(x, weight, split):
        raise ValueError(
            f"k3_ctm_gemv: unsupported call x {tuple(x.shape)} {x.dtype}, weight {tuple(weight.shape)} "
            f"{weight.dtype}, split {split}"
        )
    from . import k3_ctm_gemv_kernel as kernel

    num_tokens, k_in = x.shape
    n_out = weight.shape[0]
    y = torch.empty(num_tokens, n_out, dtype=torch.bfloat16, device=x.device)
    args = (_arg(weight), _arg(x), _arg(y.view(-1)))
    stream = _stream(x)
    use_pdl = _use_pdl()
    fn = _compile(
        ("k3_ctm_gemv", n_out, k_in, split, trigger_early, push, use_pdl),
        kernel.k3_ctm_gemv, *args, num_tokens, n_out, k_in, split, trigger_early, push, use_pdl, stream,
    )  # fmt: skip
    fn(*args, num_tokens, stream)
    return y


@k3_ctm_gemv.register_fake
def _(x, weight, trigger_early=True, split=1, push=False):
    return x.new_empty((x.shape[0], weight.shape[0]), dtype=torch.bfloat16)


def supports_tail(
    latent: torch.Tensor, act: torch.Tensor, weight: torch.Tensor, width: int
) -> bool:
    """Whether ``k3_ctm_gemv_tail`` runs the call (the conditions of ``k3_decode_gemv``'s tail)."""
    from . import k3_ctm_gemv_kernel as kernel

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


@torch.library.custom_op("trtllm::k3_ctm_gemv_tail", mutates_args=())
def k3_ctm_gemv_tail(
    latent: torch.Tensor,
    act: torch.Tensor,
    weight: torch.Tensor,
    lo: int,
    width: int,
    eps: float,
    trigger_early: bool = True,
) -> torch.Tensor:
    """``[rmsnorm(latent)[:, lo:lo+width] | act] @ weight.T`` with the RMS applied to the fp32 latent
    accumulator (``weight`` = [latent-up columns of the slice, zero-padded to 128 | shared down])."""
    if not supports_tail(latent, act, weight, width):
        raise ValueError(
            f"k3_ctm_gemv_tail: unsupported call latent {tuple(latent.shape)}, act {tuple(act.shape)}, "
            f"weight {tuple(weight.shape)}, width {width}"
        )
    from . import k3_ctm_gemv_kernel as kernel

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
    stream = _stream(latent)
    use_pdl = _use_pdl()
    fn = _compile(
        ("k3_ctm_gemv_tail", n_out, k_in, lat_tiles, rms_cols, trigger_early, use_pdl),
        kernel.k3_ctm_gemv_tail, *args, num_tokens, lo, float(eps), n_out, k_in, lat_tiles, rms_cols,
        trigger_early, use_pdl, stream,
    )  # fmt: skip
    fn(*args, num_tokens, lo, float(eps), stream)
    return y


@k3_ctm_gemv_tail.register_fake
def _(latent, act, weight, lo, width, eps, trigger_early=True):
    return latent.new_empty((latent.shape[0], weight.shape[0]), dtype=torch.bfloat16)


def supports_gated(
    a: torch.Tensor, gsrc: torch.Tensor, g_col0: int, weight: torch.Tensor, split: int = 2
) -> bool:
    """Whether ``k3_ctm_gemv_gated`` runs ``(a * sigmoid(gsrc[:, g_col0 : g_col0 + K])) @ weight^T``."""
    from . import k3_ctm_gemv_kernel as kernel

    k_in = weight.shape[1] if weight.dim() == 2 else -1
    return (
        a.is_cuda
        and a.dtype == gsrc.dtype == weight.dtype == torch.bfloat16
        and a.dim() == 2
        and gsrc.dim() == 2
        and weight.dim() == 2
        and 0 < a.shape[0] <= MAX_TOKENS
        and gsrc.shape[0] == a.shape[0]
        and a.shape[1] == k_in
        and a.is_contiguous()
        and gsrc.is_contiguous()
        and weight.is_contiguous()
        and g_col0 % 8 == 0
        and 0 <= g_col0
        and g_col0 + k_in <= gsrc.shape[1]
        and gsrc.shape[1] % 8 == 0
        and kernel.supports(weight.shape[0], k_in, split)
    )


@torch.library.custom_op("trtllm::k3_ctm_gemv_gated", mutates_args=())
def k3_ctm_gemv_gated(
    a: torch.Tensor,
    gsrc: torch.Tensor,
    g_col0: int,
    weight: torch.Tensor,
    trigger_early: bool = True,
    split: int = 2,
    gate_sigmoid: bool = True,
) -> torch.Tensor:
    """``(a * gsrc[:, g_col0:g_col0 + K].sigmoid()) @ weight.T``: bf16 ``a`` [M <= 8, K], ``gsrc`` [M, C] with dense
    rows, ``weight`` [N, K]; the sigmoid and the product each round to bf16 as the unfused torch ops do. With
    ``gate_sigmoid=False`` the window already holds the sigmoid: ``(a * gsrc[:, g_col0:g_col0 + K]) @ weight.T``."""
    if not supports_gated(a, gsrc, g_col0, weight, split):
        raise ValueError(
            f"k3_ctm_gemv_gated: unsupported call a {tuple(a.shape)}, gsrc {tuple(gsrc.shape)} at {g_col0}, "
            f"weight {tuple(weight.shape)}, split {split}"
        )
    from . import k3_ctm_gemv_kernel as kernel

    num_tokens, k_in = a.shape
    n_out = weight.shape[0]
    gsrc_cols = gsrc.shape[1]
    y = torch.empty(num_tokens, n_out, dtype=torch.bfloat16, device=a.device)
    args = (_arg(weight), _arg(a), _arg(gsrc), _arg(y.view(-1)))
    stream = _stream(a)
    use_pdl = _use_pdl()
    fn = _compile(
        ("k3_ctm_gemv_gated", n_out, k_in, gsrc_cols, split, gate_sigmoid, trigger_early, use_pdl),
        kernel.k3_ctm_gemv_gated, *args, num_tokens, g_col0, n_out, k_in, gsrc_cols, split, gate_sigmoid,
        trigger_early, use_pdl, stream,
    )  # fmt: skip
    fn(*args, num_tokens, g_col0, stream)
    return y


@k3_ctm_gemv_gated.register_fake
def _(a, gsrc, g_col0, weight, trigger_early=True, split=2, gate_sigmoid=True):
    return a.new_empty((a.shape[0], weight.shape[0]), dtype=torch.bfloat16)


def supports_swiglu(gu: torch.Tensor, weight: torch.Tensor, split: int = 2) -> bool:
    """Whether ``k3_ctm_gemv_swiglu`` runs ``(silu(gu[:, :K]) * gu[:, K:]) @ weight^T``."""
    from . import k3_ctm_gemv_kernel as kernel

    k_in = weight.shape[1] if weight.dim() == 2 else -1
    return (
        gu.is_cuda
        and gu.dtype == weight.dtype == torch.bfloat16
        and gu.dim() == 2
        and weight.dim() == 2
        and 0 < gu.shape[0] <= MAX_TOKENS
        and gu.shape[1] == 2 * k_in
        and gu.is_contiguous()
        and weight.is_contiguous()
        and kernel.supports(weight.shape[0], k_in, split)
    )


@torch.library.custom_op("trtllm::k3_ctm_gemv_swiglu", mutates_args=())
def k3_ctm_gemv_swiglu(
    gu: torch.Tensor,
    weight: torch.Tensor,
    trigger_early: bool = True,
    split: int = 2,
    push: bool = False,
) -> torch.Tensor:
    """``silu_and_mul(gu) @ weight.T`` for a bf16 gate_up output ``gu`` [M <= 8, 2 K] (gate columns first) and
    ``weight`` [N, K]: the activation is computed in the GEMV's B prologue with silu_and_mul's fp32 arithmetic and
    one bf16 rounding. ``split`` k-tile ranks per 128-row tile (need not divide the k-tiles)."""
    if not supports_swiglu(gu, weight, split):
        raise ValueError(
            f"k3_ctm_gemv_swiglu: unsupported call gu {tuple(gu.shape)} {gu.dtype}, weight {tuple(weight.shape)} "
            f"{weight.dtype}, split {split}"
        )
    from . import k3_ctm_gemv_kernel as kernel

    num_tokens = gu.shape[0]
    n_out, k_in = weight.shape
    y = torch.empty(num_tokens, n_out, dtype=torch.bfloat16, device=gu.device)
    args = (_arg(weight), _arg(gu), _arg(y.view(-1)))
    stream = _stream(gu)
    use_pdl = _use_pdl()
    fn = _compile(
        ("k3_ctm_gemv_swiglu", n_out, k_in, split, trigger_early, push, use_pdl),
        kernel.k3_ctm_gemv_swiglu, *args, num_tokens, n_out, k_in, split, trigger_early, push, use_pdl, stream,
    )  # fmt: skip
    fn(*args, num_tokens, stream)
    return y


@k3_ctm_gemv_swiglu.register_fake
def _(gu, weight, trigger_early=True, split=2, push=False):
    return gu.new_empty((gu.shape[0], weight.shape[0]), dtype=torch.bfloat16)


def supports_long(x: torch.Tensor, weight: torch.Tensor, split: int, ring: int) -> bool:
    """Whether ``k3_ctm_gemv_long`` runs ``x @ weight^T`` with this split and ring."""
    from . import k3_ctm_gemv_kernel as kernel

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
        and kernel.long_supports(weight.shape[0], weight.shape[1], split, ring)
    )


@torch.library.custom_op("trtllm::k3_ctm_gemv_long", mutates_args=())
def k3_ctm_gemv_long(
    x: torch.Tensor,
    weight: torch.Tensor,
    sig_col0: int = -1,
    split: int = 6,
    ring: int = 5,
    trigger_early: bool = True,
    push: bool = False,
) -> torch.Tensor:
    """``x @ weight.T`` for bf16 ``x`` [M <= 8, K] and ``weight`` [N, K] (long K); columns >= ``sig_col0`` (if >= 0)
    hold ``bf16(sigmoid(bf16(x @ weight.T)))``. Each 128-row weight tile is split over a cluster of ``split`` CTAs,
    each streaming its k-tiles through a ``ring``-stage ring filled before the grid-dependency wait."""
    return _launch_long(x, weight, sig_col0, split, ring, trigger_early, push=push)


def _launch_long(x, weight, sig_col0, split, ring, trigger_early, push=False) -> torch.Tensor:
    if not supports_long(x, weight, split, ring):
        raise ValueError(
            f"k3_ctm_gemv_long: unsupported call x {tuple(x.shape)} {x.dtype}, weight {tuple(weight.shape)} "
            f"{weight.dtype}, split {split}, ring {ring}"
        )
    from . import k3_ctm_gemv_kernel as kernel

    num_tokens, k_in = x.shape
    n_out = weight.shape[0]
    y = torch.empty(num_tokens, n_out, dtype=torch.bfloat16, device=x.device)
    args = (_arg(weight), _arg(x), _arg(y.view(-1)))
    stream = _stream(x)
    use_pdl = _use_pdl()
    sig_row0 = sig_col0 if sig_col0 >= 0 else n_out
    fn = _compile(
        ("k3_ctm_gemv_long", n_out, k_in, split, ring, trigger_early, push, use_pdl),
        kernel.k3_ctm_gemv_long, *args, num_tokens, sig_row0, n_out, k_in, split, ring, trigger_early, push, use_pdl,
        stream,
    )  # fmt: skip
    fn(*args, num_tokens, sig_row0, stream)
    return y


@k3_ctm_gemv_long.register_fake
def _(x, weight, sig_col0=-1, split=6, ring=5, trigger_early=True, push=False):
    return x.new_empty((x.shape[0], weight.shape[0]), dtype=torch.bfloat16)


def wide_config(n_out: int, k_in: int, n_tile: int, num_sms: int) -> Optional[tuple]:
    """``(split, ring, x_ring)`` of a ``k3_ctm_gemv_wide`` call, or None: the largest cluster whose CTAs fit one wave,
    then the deepest weight ring that fits shared memory beside an x ring of up to 3 stages. At the call sites that run
    ``k3_ctm_gemv_long`` at most 8 tokens (MLA [W_a; W_g] 6, dense gate_up 4, dense down 2, drafter 8) this is their
    split, so a token's sums are the same."""
    from . import k3_ctm_gemv_kernel as kernel

    tiles = (n_out + kernel.CTA_M - 1) // kernel.CTA_M
    for split in (8, 7, 6, 5, 4, 2):
        if tiles * split > num_sms:
            continue
        for ring in range(kernel.num_k_tiles(k_in) // split, 0, -1):
            if kernel.wide_supports(n_out, k_in, split, ring, n_tile, min(ring, 3)):
                return split, ring, min(ring, 3)
    return None


def _num_sms(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def supports_wide(
    x: torch.Tensor, weight: torch.Tensor, sig_col0: int = -1, out_fp32: bool = False
) -> bool:
    """Whether ``k3_ctm_gemv_wide`` runs ``x @ weight^T``."""
    from . import k3_ctm_gemv_kernel as kernel

    if not (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.dim() == 2
        and weight.dim() == 2
        and 0 < x.shape[0] <= WIDE_MAX_TOKENS
        and x.shape[1] == weight.shape[1]
        and x.is_contiguous()
        and weight.is_contiguous()
        and x.data_ptr() % 16 == 0
        and weight.data_ptr() % 16 == 0
        and sig_col0 < weight.shape[0]
        and (sig_col0 < 0 or not out_fp32)
    ):
        return False
    n_tile = kernel.wide_tile(x.shape[0])
    return wide_config(weight.shape[0], weight.shape[1], n_tile, _num_sms(x.device)) is not None


@torch.library.custom_op("trtllm::k3_ctm_gemv_wide", mutates_args=())
def k3_ctm_gemv_wide(
    x: torch.Tensor, weight: torch.Tensor, sig_col0: int = -1, out_fp32: bool = False
) -> torch.Tensor:
    """``x @ weight.T`` for bf16 ``x`` [M <= 64, K] and ``weight`` [N, K], all M tokens in one MMA of N = 16, 32 or 64
    columns (the long kernel's split-K clusters and weight ring, the partials reduced by token; split and ring from
    ``wide_config``). Returns bf16
    [M, N], whose columns >= ``sig_col0`` (if >= 0) hold ``bf16(sigmoid(bf16(x @ weight.T)))``, or fp32 [M, N] with
    ``out_fp32``."""
    if not supports_wide(x, weight, sig_col0, out_fp32):
        raise ValueError(
            f"k3_ctm_gemv_wide: unsupported call x {tuple(x.shape)} {x.dtype}, weight {tuple(weight.shape)} "
            f"{weight.dtype}, sig_col0 {sig_col0}, out_fp32 {out_fp32}"
        )
    from . import k3_ctm_gemv_kernel as kernel

    n_tile = kernel.wide_tile(x.shape[0])
    split, ring, x_ring = wide_config(weight.shape[0], weight.shape[1], n_tile, _num_sms(x.device))
    return _launch_wide(x, weight, sig_col0, out_fp32, split, ring, x_ring)


def _launch_wide(
    x, weight, sig_col0, out_fp32, split, ring, x_ring, trigger_early=True
) -> torch.Tensor:
    from . import k3_ctm_gemv_kernel as kernel

    num_tokens, k_in = x.shape
    n_out = weight.shape[0]
    n_tile = kernel.wide_tile(num_tokens)
    if not kernel.wide_supports(n_out, k_in, split, ring, n_tile, x_ring):
        raise ValueError(
            f"k3_ctm_gemv_wide: split {split}, rings {ring} / {x_ring} do not fit weight {tuple(weight.shape)} at "
            f"{n_tile} tokens"
        )
    y = torch.empty(
        num_tokens, n_out, dtype=torch.float32 if out_fp32 else torch.bfloat16, device=x.device
    )
    args = (_arg(weight), _arg(x), _arg(y.view(-1)))
    stream = _stream(x)
    use_pdl = _use_pdl()
    sig_row0 = sig_col0 if sig_col0 >= 0 else n_out
    fn = _compile(
        ("k3_ctm_gemv_wide", n_out, k_in, split, ring, x_ring, n_tile, out_fp32, trigger_early, use_pdl),
        kernel.k3_ctm_gemv_wide, *args, num_tokens, sig_row0, n_out, k_in, split, ring, x_ring, n_tile, out_fp32,
        trigger_early, use_pdl, stream,
    )  # fmt: skip
    fn(*args, num_tokens, sig_row0, stream)
    return y


@k3_ctm_gemv_wide.register_fake
def _(x, weight, sig_col0=-1, out_fp32=False):
    return x.new_empty(
        (x.shape[0], weight.shape[0]), dtype=torch.float32 if out_fp32 else torch.bfloat16
    )


def supports_situ_mul(gu: torch.Tensor) -> bool:
    """Whether ``k3_situ_mul`` runs SituAndMul on ``gu`` [M <= 8, 2 K] (K a multiple of 8, rows dense)."""
    return (
        gu.is_cuda
        and gu.dtype == torch.bfloat16
        and gu.dim() == 2
        and 0 < gu.shape[0] <= MAX_TOKENS
        and gu.shape[1] % 16 == 0
        and gu.is_contiguous()
        and gu.data_ptr() % 16 == 0
    )


@torch.library.custom_op("trtllm::k3_situ_mul", mutates_args=())
def k3_situ_mul(
    gu: torch.Tensor, beta: float = 1.0, linear_beta: Optional[float] = None
) -> torch.Tensor:
    """``SituAndMul(beta, linear_beta)(gu)`` for a bf16 gate_up output ``gu`` [M <= 8, 2 K] (gate columns first), with
    programmatic dependent launch: the next kernel launches at once and may stream its weights meanwhile."""
    if linear_beta is not None and float(linear_beta) == 0.0:
        raise ValueError(
            "k3_situ_mul: linear_beta=0.0 is not a SiTU scale; pass None to leave the up half unscaled"
        )
    if not supports_situ_mul(gu):
        raise ValueError(f"k3_situ_mul: unsupported call gu {tuple(gu.shape)} {gu.dtype}")
    from . import k3_ctm_gemv_kernel as kernel

    num_tokens, two_k = gu.shape
    k_in = two_k // 2
    out = torch.empty(num_tokens, k_in, dtype=torch.bfloat16, device=gu.device)
    args = (_arg(gu.view(-1)), _arg(out.view(-1)))
    stream = _stream(gu)
    use_pdl = _use_pdl()
    has_linear = linear_beta is not None
    fn = _compile(
        ("k3_situ_mul", k_in, has_linear, use_pdl),
        kernel.k3_situ_mul, *args, num_tokens, float(beta), float(linear_beta or 1.0), MAX_TOKENS, k_in, has_linear,
        use_pdl, stream,
    )  # fmt: skip
    fn(*args, num_tokens, float(beta), float(linear_beta or 1.0), stream)
    return out


@k3_situ_mul.register_fake
def _(gu, beta=1.0, linear_beta=None):
    return gu.new_empty((gu.shape[0], gu.shape[1] // 2))
