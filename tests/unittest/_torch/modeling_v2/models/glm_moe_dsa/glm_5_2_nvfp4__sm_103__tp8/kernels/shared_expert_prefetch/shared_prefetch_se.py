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
"""Host side of shared_expert_prefetch's two-kernel BF16 shared expert
(``shared_prefetch_se_kernel.py``), an alternative.

It is not part of package shared_expert_prefetch: in the two-host chain it ends later than the
native shared expert beside the routed kernel,
so the served path keeps the native ``GatedMLP``. The chain bench's ``se7``
arms and ``test_shared_prefetch_shared.py`` run it from here.

``shared_output(mlp, x)`` returns the shared expert's BF16 ``[R, 6144]``
output for a served ``GatedMLP`` (the BF16 shared expert of a GLM-5.2 MoE
layer, one TP8 rank shard), or ``None`` when the call is not eligible; the
decision depends only on host-side state (module configuration, shapes,
dtypes, device) and is taken before any device work, so ``None`` leaves the
caller on the native ``GatedMLP`` path with nothing launched.

The two kernels run on the current stream (the MoE's auxiliary stream in the
served graph) with programmatic dependent launch. Numerical variant
``se-reorder`` (see ``kernel_shared``). Side selectors (tests and
measurements): ``TRTLLM_STAIRCASE_GLM_LL_SHARED_EXPERT_PREFETCH_TWO_KERNEL_UP_CTAS`` (up-kernel CTAs, 32 or 16),
``TRTLLM_STAIRCASE_GLM_LL_SHARED_EXPERT_PREFETCH_TWO_KERNEL_DOWN_CTAS`` (down-kernel CTAs, 48 or 24),
``TRTLLM_STAIRCASE_GLM_LL_SHARED_EXPERT_PREFETCH_TWO_KERNEL_L2_PREFETCH`` (L2 prefetch bits, default 3),
``TRTLLM_STAIRCASE_GLM_LL_SHARED_EXPERT_PREFETCH_TWO_KERNEL_SILU_VARIANT`` (SiLU instruction variant, tests only),
``TRTLLM_STAIRCASE_GLM_LL_SHARED_EXPERT_PREFETCH_TWO_KERNEL_STAGE_WEIGHTS`` (1: weights staged in shared memory by bulk
copies before the dependency wait, bit-identical outputs; default 0),
``TRTLLM_STAIRCASE_GLM_LL_SHARED_EXPERT_PREFETCH_TWO_KERNEL_TRIGGER`` (``end`` or ``early``: where the up kernel
releases the down kernel; default ``end``).
"""

import os
import threading
from typing import Dict, Optional, Tuple

import torch

FEATURE = "shared_expert_prefetch"
HIDDEN = 6144
INTER = 256
MAX_ROWS = 8

_LOCK = threading.Lock()
_COMPILED: Dict[tuple, tuple] = {}


def _knobs() -> tuple:
    import shared_prefetch_se_ptx as ptx

    up = int(
        os.environ.get("TRTLLM_STAIRCASE_GLM_LL_SHARED_EXPERT_PREFETCH_TWO_KERNEL_UP_CTAS") or 32
    )
    down = int(
        os.environ.get("TRTLLM_STAIRCASE_GLM_LL_SHARED_EXPERT_PREFETCH_TWO_KERNEL_DOWN_CTAS") or 48
    )
    pf = int(
        os.environ.get("TRTLLM_STAIRCASE_GLM_LL_SHARED_EXPERT_PREFETCH_TWO_KERNEL_L2_PREFETCH") or 3
    )
    silu = int(
        os.environ.get("TRTLLM_STAIRCASE_GLM_LL_SHARED_EXPERT_PREFETCH_TWO_KERNEL_SILU_VARIANT")
        or ptx.SILU_DEFAULT
    )
    stage = (
        os.environ.get("TRTLLM_STAIRCASE_GLM_LL_SHARED_EXPERT_PREFETCH_TWO_KERNEL_STAGE_WEIGHTS")
        or "0"
    ) == "1"
    trig = (
        os.environ.get("TRTLLM_STAIRCASE_GLM_LL_SHARED_EXPERT_PREFETCH_TWO_KERNEL_TRIGGER") or "end"
    )
    return up, down, pf, silu, stage, trig


def _compiled(key: tuple):
    with _LOCK:
        fns = _COMPILED.get(key)
        if fns is not None:
            return fns
        import cutlass
        import cutlass.cute as cute
        from cuda.bindings import driver as cuda
        from shared_prefetch_se_kernel import (
            PF_W2_OWN,
            SharedExpertTwoKernelDown,
            SharedExpertTwoKernelUp,
        )

        up, down, pf, silu, stage, trig = key
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        z = cutlass.Int64(0)
        up_fn = cute.compile(
            SharedExpertTwoKernelUp(up, prefetch=pf & 3, silu=silu, stage=stage, trigger=trig).run,
            z,
            z,
            z,
            z,
            cutlass.Int32(1),
            stream,
        )
        down_fn = cute.compile(
            SharedExpertTwoKernelDown(down, prefetch=pf & PF_W2_OWN, stage=stage).run,
            z,
            z,
            z,
            cutlass.Int32(1),
            stream,
        )
        _COMPILED[key] = (up_fn, down_fn)
        return up_fn, down_fn


def eligibility(mlp, x) -> Tuple[bool, str]:
    """Host-only check that the shared_expert_prefetch kernels compute ``mlp(x)`` (rank-identical inputs)."""
    import torch.nn.functional as F

    if not isinstance(x, torch.Tensor):
        return False, "input is not a plain tensor"
    if x.dtype != torch.bfloat16 or x.dim() != 2 or x.shape[1] != HIDDEN:
        return False, f"input {x.dtype} {tuple(x.shape)}"
    rows = x.shape[0]
    if not 1 <= rows <= MAX_ROWS:
        return False, f"rows {rows} outside 1..{MAX_ROWS}"
    if not x.is_cuda or not x.is_contiguous() or x.data_ptr() % 16:
        return False, "input is not a contiguous 16-byte aligned device tensor"
    if torch.cuda.get_device_capability(x.device)[0] != 10:
        return False, "not an SM100-family device"
    if type(mlp).__name__ != "GatedMLP" or getattr(mlp, "split_gate_up", False):
        return False, "not a fused gate/up GatedMLP"
    if mlp.activation is not F.silu:
        return False, "activation is not SiLU"
    if (
        mlp.swiglu_limit is not None
        or mlp.swiglu_alpha not in (None, 1.0)
        or mlp.swiglu_beta not in (None, 0.0)
    ):
        return False, "SwiGLU limit/alpha/beta"
    gu, dn = mlp.gate_up_proj, mlp.down_proj
    for name, lin, shape in (("gate_up", gu, (2 * INTER, HIDDEN)), ("down", dn, (HIDDEN, INTER))):
        w = getattr(lin, "weight", None)
        if (
            w is None
            or w.dtype != torch.bfloat16
            or tuple(w.shape) != shape
            or not w.is_contiguous()
        ):
            return False, f"{name} weight is not a contiguous BF16 {shape}"
        if w.device != x.device or w.data_ptr() % 16:
            return False, f"{name} weight is not a 16-byte aligned tensor on the input's device"
        if getattr(lin, "bias", None) is not None:
            return False, f"{name} has a bias"
        if getattr(lin, "has_any_quant", False):
            return False, f"{name} is quantized"
    if getattr(dn, "reduce_output", False) and getattr(dn, "tp_size", 1) > 1:
        return False, "down projection reduces across ranks"
    for probe in (
        "_can_fuse_gate_up_swiglu_fp4out",
        "_can_fuse_gate_up_swiglu",
        "_can_fuse_swiglu_fp8_quant",
    ):
        fn = getattr(mlp, probe, None)
        if fn is not None and fn():
            return False, f"native path is {probe[5:]}"
    return True, ""


class SharedExpertTwoKernel:
    """The shared_expert_prefetch shared expert of one ``GatedMLP`` (one per module: its scratch is private)."""

    def __init__(self, mlp):
        self.w1 = mlp.gate_up_proj.weight
        self.w2 = mlp.down_proj.weight
        self.a = torch.zeros(MAX_ROWS, INTER, dtype=torch.bfloat16, device=self.w1.device)

    @staticmethod
    def ready_for_capture() -> bool:
        return _knobs() in _COMPILED

    def __call__(self, x: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        import cutlass
        from cuda.bindings import driver as cuda

        up_fn, down_fn = _compiled(_knobs())
        rows = x.shape[0]
        if out is None:
            out = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=x.device)
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        n = cutlass.Int32(rows)
        up_fn(
            cutlass.Int64(x.data_ptr()),
            cutlass.Int64(self.w1.data_ptr()),
            cutlass.Int64(self.w2.data_ptr()),
            cutlass.Int64(self.a.data_ptr()),
            n,
            stream,
        )
        down_fn(
            cutlass.Int64(self.a.data_ptr()),
            cutlass.Int64(self.w2.data_ptr()),
            cutlass.Int64(out.data_ptr()),
            n,
            stream,
        )
        return out


def shared_eligible(mlp, x) -> bool:
    """Whether ``shared_output(mlp, x)`` engages (host-only, same decision)."""
    ok, _ = eligibility(mlp, x)
    if ok and torch.cuda.is_current_stream_capturing():
        runner = getattr(mlp, "_shared_expert_two_kernel", None)
        ok = runner is not None and runner.ready_for_capture()
    return ok


def shared_output(mlp, x) -> Optional[torch.Tensor]:
    """The shared_expert_prefetch shared expert's output for ``mlp(x)``, or ``None`` (native path; nothing launched)."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch

    ok, reason = eligibility(mlp, x)
    runner = getattr(mlp, "_shared_expert_two_kernel", None) if ok else None
    if (
        ok
        and runner is not None
        and (runner.w1 is not mlp.gate_up_proj.weight or runner.w2 is not mlp.down_proj.weight)
    ):
        runner = None
    if (
        ok
        and torch.cuda.is_current_stream_capturing()
        and (runner is None or not runner.ready_for_capture())
    ):
        ok, reason = False, "not warmed up before CUDA graph capture"
    dispatch.log_capability(
        FEATURE, f"shared_expert rows={x.shape[0] if ok else '-'} {reason or 'engaged'}", ok, reason
    )
    if not ok:
        return None
    if runner is None:
        runner = SharedExpertTwoKernel(mlp)
        mlp._shared_expert_two_kernel = runner
    return runner(x)
