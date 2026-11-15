# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import sys

import torch
import torch.nn as nn

from tensorrt_llm._utils import nvtx_range

from ..attention.backends import AttentionMetadata
from .linear import Linear

# Opt-in (default off): replace the BF16 -> FP32 logits cast with a
# vectorized Triton kernel that writes the same bits. PyTorch's dynamic-cast
# path for this cast (``Tensor.float()`` on a BF16 CUDA tensor) is not
# vectorized; the Triton kernel processes the tensor in fixed-size blocks.
# Widening BF16 to FP32 is exact, so the kernel is a bit placement: fp32_bits = bf16_bits << 16 (c10::detail::f32_from_bits).
# Read once per process; the worker's environment is fixed before import.
_EXACT_LOGITS_CAST = os.environ.get("TLLM_EXACT_LOGITS_CAST", "") == "1"
# Rows the first eager call compares against Tensor.float(); a few rows keep
# the one-time check off the memory profile.
_LOGITS_CAST_SELFCHECK_ROWS = 4
_LOGITS_CAST_BLOCK = 4096
_logits_cast_state = {
    "kernel": None,
    "ready": False,
    "disabled": False,
    "printed": set()
}


def exact_logits_cast_enabled() -> bool:
    return _EXACT_LOGITS_CAST


def logits_cast_print_once(tag: str, message: str) -> None:
    """Print one TLLM_EXACT_LOGITS_CAST line per process and tag, on stderr.

    Worker-rank Python INFO logging can be off in disaggregated workers, so
    the engagement lines go to stderr.
    """
    printed = _logits_cast_state["printed"]
    if tag in printed:
        return
    printed.add(tag)
    print(f"TLLM_EXACT_LOGITS_CAST {message} pid={os.getpid()}",
          file=sys.stderr,
          flush=True)


def _build_logits_cast_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def _bf16_bits_to_f32_bits(src_ptr, dst_ptr, n, BLOCK: tl.constexpr):
        # src: BF16 bits viewed as int16; dst: FP32 bits viewed as int32.
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        bits = tl.load(src_ptr + offs, mask=mask, other=0)
        wide = bits.to(tl.uint16, bitcast=True).to(tl.uint32) << 16
        tl.store(dst_ptr + offs, wide.to(tl.int32, bitcast=True), mask=mask)

    return _bf16_bits_to_f32_bits


def _launch_logits_cast(src: torch.Tensor, dst: torch.Tensor) -> None:
    n = src.numel()
    grid = ((n + _LOGITS_CAST_BLOCK - 1) // _LOGITS_CAST_BLOCK, )
    _logits_cast_state["kernel"][grid](src.view(torch.int16),
                                     dst.view(torch.int32),
                                     n,
                                     BLOCK=_LOGITS_CAST_BLOCK,
                                     num_warps=4)


def _logits_cast_eligible(logits: torch.Tensor) -> bool:
    # One Triton specialization only (16-byte aligned pointers, an int32
    # element count divisible by 16), so no call after the eager check can
    # trigger a JIT compile, e.g. under CUDA graph capture.
    n = logits.numel()
    return (logits.dtype == torch.bfloat16 and logits.is_cuda
            and logits.is_contiguous() and 0 < n < 2**31 and n % 16 == 0
            and logits.data_ptr() % 16 == 0)


def exact_logits_to_float(logits: torch.Tensor) -> torch.Tensor:
    """``logits.float()`` with the same bits, through a vectorized kernel.

    Falls back to ``logits.float()`` while torch.compile traces, for anything
    ``_logits_cast_eligible`` rejects, under CUDA graph capture until the kernel
    was built and checked eagerly, and for the rest of the process if the
    kernel fails to build or its one-time check against ``Tensor.float()``
    disagrees.
    """
    state = _logits_cast_state
    if (torch.compiler.is_compiling() or state["disabled"]
            or not _logits_cast_eligible(logits)):
        return logits.float()
    if not state["ready"]:
        if torch.cuda.is_current_stream_capturing():
            return logits.float()
        try:
            if state["kernel"] is None:
                state["kernel"] = _build_logits_cast_kernel()
            rows = logits.view(-1, logits.shape[-1])[:_LOGITS_CAST_SELFCHECK_ROWS]
            got = torch.empty(rows.shape,
                              dtype=torch.float32,
                              device=rows.device)
            _launch_logits_cast(rows, got)
            ref = rows.float()
            nan = torch.isnan(ref)
            # Equal bits wherever the value is a number; NaN stays NaN (no
            # consumer distinguishes NaN payloads).
            same = bool(
                torch.equal(nan, torch.isnan(got))
                and torch.equal(torch.where(nan, 0, got.view(torch.int32)),
                                torch.where(nan, 0, ref.view(torch.int32))))
        except Exception as error:  # build or launch failure: keep the stock cast
            state["disabled"] = True
            logits_cast_print_once("fallback",
                      f"engaged=False reason={type(error).__name__}: {error}")
            return logits.float()
        if not same:
            state["disabled"] = True
            logits_cast_print_once("fallback", "engaged=False reason=selfcheck_mismatch")
            return logits.float()
        state["ready"] = True
        logits_cast_print_once(
            "target", "engaged=True cast=triton_bf16_bits_to_f32 "
            f"selfcheck_rows={rows.shape[0]} bitwise_equal=True "
            f"first_shape={tuple(logits.shape)}")
    out = torch.empty(logits.shape, dtype=torch.float32, device=logits.device)
    _launch_logits_cast(logits, out)
    return out


class LogitsProcessor(nn.Module):

    def __init__(self):
        super().__init__()

    @nvtx_range("LogitsProcessor")
    def forward(self,
                hidden_states: torch.Tensor,
                lm_head: Linear,
                attn_metadata: AttentionMetadata,
                return_context_logits: bool = False,
                *,
                keep_lm_head_dtype: bool = False) -> torch.Tensor:

        if not return_context_logits:
            if attn_metadata is not None:
                last_tokens = torch.cumsum(
                    attn_metadata.seq_lens_cuda,
                    dim=0,
                    dtype=torch.long,
                ) - 1
                hidden_states = hidden_states[last_tokens]
            else:
                hidden_states = hidden_states[-1]

        logits = lm_head(hidden_states)
        if keep_lm_head_dtype:
            # TLLM_EXACT_LOGITS_CAST: the caller's only consumer is an argmax, which
            # picks the same index from the lm_head dtype as from its FP32 cast.
            return logits
        if _EXACT_LOGITS_CAST:
            return exact_logits_to_float(logits)
        logits = logits.float()
        return logits
