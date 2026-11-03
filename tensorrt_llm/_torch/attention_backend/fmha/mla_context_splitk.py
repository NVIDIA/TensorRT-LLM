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
"""Runner for the FP8 MLA context FMHA with split K operand and epilogue merge (Kimi-K3 chunked prefill).

Opt-in with ``TRTLLM_K3_FP8KV_SPLIT_K_CONTEXT_SM100=1`` (see ``MLA``). The kernel itself lives in
``cute_dsl_kernels/blackwell/attention/mla/mla_context_fp8.py``; this module owns the JIT cache, the launch
glue and the engagement conditions used by ``MLA.forward_context``.
"""

import math
from typing import Optional

import torch

from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger

_IMPORT_ERROR: Optional[BaseException] = None
try:
    import cuda.bindings.driver as _cuda_driver
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.attention.mla.mla_context_fp8 import (
        MlaContextFp8Fmha,
    )
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.helpers import fmha_helpers as _fmha_utils
except (ImportError, OSError) as _e:  # pragma: no cover - environment without the CuTe DSL
    _IMPORT_ERROR = _e

_FP8 = torch.float8_e4m3fn
_compiled: dict = {}


def unavailable_reason(attn) -> Optional[str]:
    """None when the kernel can serve ``attn`` (a TrtllmAttention MLA layer), else the reason it cannot."""
    if _IMPORT_ERROR is not None:
        return f"CuTe DSL import failed: {_IMPORT_ERROR!r}"
    sm = get_sm_version()
    if sm not in (100, 103):
        return f"requires SM100/SM103, got SM{sm}"
    if not (attn.is_mla_enable and attn.has_fp8_kv_cache):
        return "requires an FP8 KV cache MLA layer"
    if attn.num_heads <= 0 or attn.num_heads != attn.num_kv_heads:
        return (
            f"requires matching positive Q/KV head counts, got {attn.num_heads}/{attn.num_kv_heads}"
        )
    if (attn.qk_nope_head_dim, attn.qk_rope_head_dim, attn.v_head_dim) != (128, 64, 128):
        return (
            "requires (qk_nope, qk_rope, v) head dims (128, 64, 128), got "
            f"{(attn.qk_nope_head_dim, attn.qk_rope_head_dim, attn.v_head_dim)}"
        )
    try:
        scale = float(attn.kv_cache_scaling_factor.item())
    except (AttributeError, RuntimeError):
        scale = 1.0
    if abs(scale - 1.0) > 1e-6:
        return f"requires a unit kv_cache_scaling_factor (FP8 scratch is unit-scaled), got {scale}"
    return None


def _cute_tensor(t: torch.Tensor, leading_dim: int):
    if t.dtype == _FP8:
        c = from_dlpack(t.view(torch.int8), assumed_align=16).mark_layout_dynamic(
            leading_dim=leading_dim
        )
        c.element_type = cutlass.Float8E4M3FN
        return c
    return from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=leading_dim)


def _launch_args(q, kn, kp, v, o, stats, cu_q, cu_kv, merge_op, max_q, max_kv, sm_scale):
    log2e = math.log2(math.e)
    return (
        _cute_tensor(q, 2),
        _cute_tensor(kn, 2),
        _cute_tensor(kp, 1),
        _cute_tensor(v, 2),
        _cute_tensor(o, 2),
        _cute_tensor(stats, 2),
        # row views of [loops, B+1] int32 / [loops, B] int64 tables are only element-aligned
        from_dlpack(cu_q, assumed_align=4).mark_layout_dynamic(leading_dim=0),
        from_dlpack(cu_kv, assumed_align=4).mark_layout_dynamic(leading_dim=0),
        from_dlpack(merge_op, assumed_align=8).mark_layout_dynamic(leading_dim=0),
        (cu_q.numel() - 1, int(max_q), int(max_kv)),
        cutlass.Float32(sm_scale * log2e),
        cutlass.Float32(sm_scale),
        _cuda_driver.CUstream(torch.cuda.current_stream(q.device).cuda_stream),
    )


def run_partition(
    q: torch.Tensor,  # [T, H, 192] fp8
    k_nope: torch.Tensor,  # [S, H, 128] fp8, any token/head strides (view of the kv_b_proj output)
    k_pe: torch.Tensor,  # [S, 64] fp8
    v: torch.Tensor,  # [S, H, 128] fp8, any token/head strides
    output: torch.Tensor,  # [T, H, 128] bf16, merged into in place
    stats: torch.Tensor,  # [T, H, 2] fp32 running (max, sum), merged into in place
    cu_q: torch.Tensor,  # [B+1] int32
    cu_kv: torch.Tensor,  # [B+1] int32
    merge_op: torch.Tensor,  # [B] int64 (0 skip, 1 merge, 2 overwrite)
    *,
    max_q: int,
    max_kv: int,
    causal: bool,
    sm_scale: float,
) -> None:
    args = _launch_args(
        q, k_nope, k_pe, v, output, stats, cu_q, cu_kv, merge_op, max_q, max_kv, sm_scale
    )
    key = (causal, q.shape[1], torch.cuda.get_device_capability(q.device))
    fn = _compiled.get(key)
    if fn is None:
        mask = _fmha_utils.MaskEnum.WINDOW_MASK if causal else _fmha_utils.MaskEnum.RESIDUAL_MASK
        # One CTA per (q tile, head, sequence), not a persistent grid: the hardware block scheduler then balances
        # the work, which keeps the kernel fast when other kernels hold part of the SMs.
        kernel = MlaContextFp8Fmha(mask, persistent=False)
        logger.info(
            "Compiling the FP8 MLA context FMHA (split K nope/rope, epilogue merge), "
            f"causal={causal}"
        )
        fn = cute.compile(kernel, *args)
        _compiled[key] = fn
    fn(*args)


def warmup(device: torch.device, num_heads: int) -> None:
    """JIT-compile both variants (cached-prefix non-causal, new-token causal) before the first real batch."""
    n = 128
    q = torch.zeros(n, num_heads, 192, dtype=_FP8, device=device)
    kv = torch.zeros(n, 2 * num_heads * 128, dtype=_FP8, device=device)
    kn = kv[:, : num_heads * 128].view(n, num_heads, 128)
    v = kv[:, num_heads * 128 :].view(n, num_heads, 128)
    kp = torch.zeros(n, 64, dtype=_FP8, device=device)
    o = torch.zeros(n, num_heads, 128, dtype=torch.bfloat16, device=device)
    st = torch.zeros(n, num_heads, 2, dtype=torch.float32, device=device)
    cu = torch.tensor([0, n], dtype=torch.int32, device=device)
    for causal, op in ((False, 2), (True, 1)):
        run_partition(
            q,
            kn,
            kp,
            v,
            o,
            st,
            cu,
            cu,
            torch.full((1,), op, dtype=torch.int64, device=device),
            max_q=n,
            max_kv=n,
            causal=causal,
            sm_scale=1.0,
        )
    torch.cuda.synchronize(device)


_warmed: set = set()


def warmup_once(device: torch.device, num_heads: int) -> None:
    key = (str(device), num_heads)
    if key not in _warmed:
        _warmed.add(key)
        warmup(device, num_heads)
