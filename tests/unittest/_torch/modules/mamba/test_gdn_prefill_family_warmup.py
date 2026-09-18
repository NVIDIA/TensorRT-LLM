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
"""Triton specialization-family tripwire for the GDN prefill elementwise kernels.

The GDN-prefill family warmup in ``model_engine._run_mamba_hybrid_warmup``
(gated by ``TLLM_GDN_PREFILL_FAMILY_WARMUP``) relies on Triton's int-arg
specialization policy: kernels re-JIT once per family (``==1`` / ``%16==0`` /
other) of the packed prefill token count (and the strides equal to it), plus
the ``HAS_DECODE`` constexpr fork of ``_fused_gdn_post_conv_kernel`` — NOT per
exact token count. This test warms one representative per family combo and
asserts that other token counts in the same families are then compile-free.
If a Triton upgrade changes the specialization policy, this fails and the
warmup shape list in ``_run_mamba_hybrid_warmup`` must be revisited.

NB: the tripwire is strongest when this file runs in a fresh process (Triton's
in-process kernel cache is shared with any earlier test that compiled these
kernels).
"""

import time

import pytest
import torch

from tensorrt_llm._torch.modules.mamba.fuse_elementwise_ops import (
    extract_transpose_prefill_slice,
    fused_gdn_post_conv,
)

skip_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for triton kernels",
)

# Small GDN-like geometry (family keying only depends on the token-count args,
# not on head geometry, which is constant across calls anyway).
NUM_K_HEADS = 2
HEAD_K_DIM = 64
NUM_V_HEADS = 4
HEAD_V_DIM = 64
QKV_DIM = 2 * NUM_K_HEADS * HEAD_K_DIM + NUM_V_HEADS * HEAD_V_DIM

# One representative per triton int-specialization family of the prefill token
# count: %16==0, %16!=0, ==1.
WARMUP_TOKEN_COUNTS = (256, 252, 1)
# Battery: different token counts mapping onto the same three families.
BATTERY_TOKEN_COUNTS = (4096, 65536, 16500, 16501, 4095, 1)
# Decode-token counts: the key is the CARTESIAN PRODUCT of the prefill and
# decode families, so every decode family (none / ==1 / %16!=0 / %16==0)
# needs a representative per prefill family. Sampling only one decode class
# is exactly the production bug: the first serving batches of the unwarmed
# cross-terms re-JIT'd mid-serving (~230 ms, rank-lockstep).
WARMUP_DECODE_COUNTS = (0, 1, 8, 16)
# Battery includes the token counts of the four mid-serving JITs observed in
# production (1, 54, 48) mapping onto already-warmed families.
BATTERY_DECODE_COUNTS = (0, 1, 2, 54, 48, 16)

# A previously-compiled launch is <5 ms; a triton compile is >80 ms.
MAX_COMPILE_FREE_CALL_MS = 50.0


def _timed_ms(fn, *args) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e3


def _run_extract(num_prefill_tokens: int) -> float:
    src = torch.randn(num_prefill_tokens, QKV_DIM + 32, dtype=torch.bfloat16, device="cuda")
    return _timed_ms(extract_transpose_prefill_slice, src, num_prefill_tokens, 0, QKV_DIM)


def _run_post_conv(num_prefill_tokens: int, num_decode_tokens: int) -> float:
    prefill = torch.randn(QKV_DIM, num_prefill_tokens, dtype=torch.bfloat16, device="cuda")
    decode = (
        torch.randn(num_decode_tokens, QKV_DIM, dtype=torch.bfloat16, device="cuda")
        if num_decode_tokens
        else None
    )
    num_tokens = num_prefill_tokens + num_decode_tokens
    a = torch.randn(num_tokens, NUM_V_HEADS, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(num_tokens, NUM_V_HEADS, dtype=torch.bfloat16, device="cuda")
    a_log = torch.randn(NUM_V_HEADS, dtype=torch.float32, device="cuda")
    dt_bias = torch.randn(NUM_V_HEADS, dtype=torch.float32, device="cuda")
    return _timed_ms(
        fused_gdn_post_conv,
        prefill,
        decode,
        a,
        b,
        a_log,
        dt_bias,
        NUM_K_HEADS,
        HEAD_K_DIM,
        NUM_V_HEADS,
        HEAD_V_DIM,
    )


@skip_no_cuda
def test_gdn_prefill_family_warmup_makes_other_token_counts_compile_free():
    # Simulated warmup: one call per (token-count family x HAS_DECODE) combo,
    # analogous to the shapes _run_mamba_hybrid_warmup issues.
    for num_prefill_tokens in WARMUP_TOKEN_COUNTS:
        _run_extract(num_prefill_tokens)
        for num_decode_tokens in WARMUP_DECODE_COUNTS:
            _run_post_conv(num_prefill_tokens, num_decode_tokens)

    # Battery: every other token count must hit an already-compiled family.
    for num_prefill_tokens in BATTERY_TOKEN_COUNTS:
        elapsed_ms = _run_extract(num_prefill_tokens)
        assert elapsed_ms < MAX_COMPILE_FREE_CALL_MS, (
            f"extract_transpose_prefill_slice(L={num_prefill_tokens}) took "
            f"{elapsed_ms:.1f} ms after family warmup (likely a re-JIT: the "
            f"triton specialization policy has changed)"
        )
        for num_decode_tokens in BATTERY_DECODE_COUNTS:
            elapsed_ms = _run_post_conv(num_prefill_tokens, num_decode_tokens)
            assert elapsed_ms < MAX_COMPILE_FREE_CALL_MS, (
                f"fused_gdn_post_conv(L={num_prefill_tokens}, "
                f"decode={num_decode_tokens}) took {elapsed_ms:.1f} ms after "
                f"family warmup (likely a re-JIT: the triton specialization "
                f"policy has changed)"
            )
