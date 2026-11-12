# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fake (meta) implementations of this target's native ops, built from its csrc/.

The cuda-core W4A16 GEMV (plain and with the fused conv1d update), the cuda-core
W4A4 NVFP4 GEMM for decode rows, the Marlin W4A16 NVFP4 GEMM, the SM120 gate_up
GEMM with the SwiGLU + NVFP4 epilogue, the attention op, and the test-only entry of the
SM120 e4m3 head-256 context FMHA (csrc/fmha/). The loader imports this module right
after the target is found in the family library, so a fake for an op the library
does not define fails the load.
"""

import torch

from tensorrt_llm._torch._experimental.modeling_v2._native import namespace_of

_NS = namespace_of(__name__)

_SF_VEC = 16


@torch.library.register_fake(f"{_NS}::nvfp4_w4a16_gemv")
def _nvfp4_w4a16_gemv(mat_a, weight, weight_sf, global_scale):
    # weight [N, K/2] packed E2M1 -> output [M, N] BF16.
    return mat_a.new_empty((mat_a.shape[0], weight.shape[0]))


@torch.library.register_fake(f"{_NS}::nvfp4_w4a16_gemv_conv")
def _nvfp4_w4a16_gemv_conv(
    mat_a,
    weight,
    weight_sf,
    global_scale,
    conv_state,
    conv_weight,
    conv_state_indices,
    conv_dim,
    pad_slot_id,
):
    # Mutates conv_state in place; output [M, N] BF16.
    return mat_a.new_empty((mat_a.shape[0], weight.shape[0]))


@torch.library.register_fake(f"{_NS}::cuda_core_nvfp4_gemm")
def _cuda_core_nvfp4_gemm(mat_a, mat_b, scale_a, scale_b, alpha):
    # [M, K/2] x [N, K/2] packed E2M1 -> [M, N] BF16.
    return mat_a.new_empty((mat_a.shape[0], mat_b.shape[0]), dtype=torch.bfloat16)


@torch.library.register_fake(f"{_NS}::marlin_nvfp4_w4a16_gemm")
def _marlin_nvfp4_w4a16_gemm(
    mat_a, mat_b, scale_b, weight_global_scale, size_n, size_k, workspace_mode, launch_hint
):
    # [M, K] BF16 against a Marlin-packed weight -> [M, size_n] BF16.
    return mat_a.new_empty((mat_a.shape[0], size_n))


@torch.library.register_fake(f"{_NS}::nvfp4_gate_up_swiglu_nvfp4")
def _nvfp4_gate_up_swiglu_nvfp4(mat1, mat2, mat1Scale, mat2Scale, alpha, outScale, tile=0):
    # (fp4 [m, n/4] uint8, swizzled SF [ceil128(m) * ceil4(n/32)] uint8).
    m, n = mat1.shape[0], mat2.shape[0]
    d = n // 2
    m_pad = (m + 127) // 128 * 128
    num_k_tiles = (d // _SF_VEC + 3) // 4
    return [
        mat1.new_empty((m, d // 2), dtype=torch.uint8),
        mat1.new_empty((m_pad * num_k_tiles * 4,), dtype=torch.uint8),
    ]


@torch.library.register_fake(f"{_NS}::attention_sm120")
def _attention_sm120(qkv, output, *args):
    # Writes output in place.
    return None


@torch.library.register_fake(f"{_NS}::context_fmha_sm120_test")
def _context_fmha_sm120_test(
    q,
    kv_pool,
    block_offsets,
    cu_q_seqlens,
    cu_kv_seqlens,
    max_q_len,
    bmm1_scale,
    bmm2_scale,
    split,
    skip_leading,
):
    # FP8 q [T, 24, 256] -> BF16 [T, 24, 256].
    return q.new_empty(q.shape, dtype=torch.bfloat16)


@torch.library.register_fake(f"{_NS}::attention_sm120_workspace_bytes")
def _attention_sm120_workspace_bytes(max_tokens, max_seqs):
    # A host-side sizing query, called once when the KV cache is bound; never traced.
    raise NotImplementedError("attention_sm120_workspace_bytes is a host-side query")


@torch.library.register_fake(f"{_NS}::attention_sm120_semaphore_words")
def _attention_sm120_semaphore_words(max_seqs):
    # A host-side sizing query, called once when the KV cache is bound; never traced.
    raise NotImplementedError("attention_sm120_semaphore_words is a host-side query")
