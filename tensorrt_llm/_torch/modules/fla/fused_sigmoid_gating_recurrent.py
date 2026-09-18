# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import os
from typing import Optional

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modules.fla.utils import custom_device_ctx
from tensorrt_llm._utils import is_flashinfer_gdn_supported_arch
from tensorrt_llm.logger import logger

try:
    # A missing build raises ImportError; a CuTe/CUTLASS mismatch raises
    # RuntimeError (mirror FlashInfer's own guard) -> Triton fallback.
    # gated_delta_rule: T=1 decode entry (dispatches to the wide_vec fast path
    # when B*HV is large). gated_delta_rule_mtp: T>=1 with batch-scoped
    # intermediate_states_buffer and disable_state_update support, used by the
    # speculative-decoding target-verify path.
    from flashinfer.gdn_kernels.gdn_decode_bf16_state import \
        gated_delta_rule as _fi_gdn_decode_bf16_state_t1
    from flashinfer.gdn_kernels.gdn_decode_bf16_state import \
        gated_delta_rule_mtp as _fi_gdn_decode_bf16_state_mtp
    _FLASHINFER_GDN_BF16_STATE_AVAILABLE = True
except (ImportError, RuntimeError):
    _FLASHINFER_GDN_BF16_STATE_AVAILABLE = False

# The two launchers behind gated_delta_rule (internal FlashInfer symbols) for
# flashinfer_gdn_decode_t1 below; either missing only disables that shortcut.
try:
    from flashinfer.gdn_kernels.gdn_decode_bf16_state import \
        _select_wide_vec_tile_v as _fi_select_wide_vec_tile_v
    from flashinfer.gdn_kernels.gdn_decode_bf16_state import \
        gated_delta_rule_t1_wide_vec as _fi_gdn_decode_t1_wide_vec
except (ImportError, RuntimeError):
    _fi_select_wide_vec_tile_v = None
    _fi_gdn_decode_t1_wide_vec = None

# FlashInfer's compile cache of the MTP kernel and its config / key helpers, for
# the compiled-kernel shortcut of flashinfer_gdn_tail_recurrent (internal
# symbols; any of them missing keeps the tail on the public entry).
try:
    from flashinfer.gdn_kernels import gdn_decode_bf16_state as _fi_gdn_decode_mod
    _fi_mtp_compiled = _fi_gdn_decode_mod._compiled_kernels_mtp
    _fi_wide_vec_compiled = _fi_gdn_decode_mod._compiled_kernels_wide_vec
    _fi_get_bf16_mtp_config = _fi_gdn_decode_mod._get_bf16_mtp_config
    _fi_dtype_key = _fi_gdn_decode_mod._dtype_key
    _fi_use_packed_fma = _fi_gdn_decode_mod._USE_PACKED_FMA
    import cuda.bindings.driver as _cuda_driver
except (ImportError, RuntimeError, AttributeError):
    _fi_mtp_compiled = _fi_wide_vec_compiled = None
    _fi_get_bf16_mtp_config = _fi_dtype_key = _fi_use_packed_fma = _cuda_driver = None

_cu_streams: dict = {}


def _current_cu_stream(device_index: int):
    """The CUstream of the current torch stream, from the raw handle (the
    torch.cuda.current_stream() Stream object costs ~2 us per call; the
    handle lookup ~0.3 us) with the CUstream objects cached per handle."""
    handle = torch._C._cuda_getCurrentRawStream(device_index)
    stream = _cu_streams.get(handle)
    if stream is None:
        if len(_cu_streams) >= 64:
            _cu_streams.clear()
        stream = _cu_streams[handle] = _cuda_driver.CUstream(handle)
    return stream

# TLLM_GDN_FI_DIRECT_LAUNCH=0 keeps every FlashInfer GDN call on the validating
# public entries (A/B; shared with the prefill adapter's knob).
_FI_GDN_DECODE_DIRECT = os.environ.get("TLLM_GDN_FI_DIRECT_LAUNCH", "1") == "1"

# Max per-sequence token count served by the FlashInfer MTP verify kernel; the
# parity test (test_flashinfer_gdn_verify.py) covers T=1..8 against the Triton
# reference. Longer drafts fall back to the Triton recurrent kernel.
_FI_GDN_MAX_MTP_T = 8


@triton.heuristics({
    "USE_INITIAL_STATE": lambda args: args["h0_source"] is not None,
    "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
})
@triton.jit(do_not_specialize=["T"])
def fused_sigmoid_gating_delta_rule_update_kernel(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    q,
    k,
    v,
    b,
    o,
    h0_source,
    h0_indices,
    cu_seqlens,
    scale,
    T,
    total_nh,
    stride_q,
    stride_k,
    stride_v,
    stride_a,
    stride_b,
    s_h0_0,
    h0_dim0,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Fused kernel that combines sigmoid gating computation with recurrent delta rule update.
    """
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]
    grid_stride_nh = tl.num_programs(2)

    while i_nh < total_nh:
        i_n, i_hv = i_nh // HV, i_nh % HV
        i_h = i_hv // (HV // H)

        if IS_VARLEN:
            bos, eos = (
                tl.load(cu_seqlens + i_n).to(tl.int64),
                tl.load(cu_seqlens + i_n + 1).to(tl.int64),
            )
            all = T
            seq_T = eos - bos
        else:
            bos, eos = i_n * T, i_n * T + T
            all = B * T
            seq_T = T

        # Decode q/k/v/a/b often arrive as views sliced out of larger packed tensors.
        # Use the caller-provided token strides so the kernel can consume those views
        # directly instead of relying on a packed contiguous layout.
        p_q = q + bos * stride_q + i_h * K + o_k
        p_k = k + bos * stride_k + i_h * K + o_k
        p_v = v + bos * stride_v + i_hv * V + o_v
        p_b = b + bos * stride_b + i_hv
        # o is allocated in this wrapper and kept contiguous, so the output
        # pointer arithmetic can use the packed [NK, B, T, HV, V] layout.
        p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

        # Gating computation pointers
        p_A_log = A_log + i_hv
        p_a = a + bos * stride_a + i_hv
        p_dt_bias = dt_bias + i_hv

        b_h = tl.zeros([BK, BV], dtype=tl.float32)
        if USE_INITIAL_STATE:
            idx = tl.load(h0_indices + i_n).to(tl.int64)
            if idx >= 0:
                tl.device_assert(idx < h0_dim0,
                                 "idx out of bounds in h0_source load")
                # Pool layout [slots, HV, V, K] with K innermost (stride 1).
                # b_h is logically [BK, BV]; element [k, v] lives at
                # offset v*K + k within a (V, K) tile.
                p_h0 = (h0_source + idx * s_h0_0 + i_hv * V * K + o_k[:, None] +
                        o_v[None, :] * K)
                b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

        for _ in range(0, seq_T):
            # Load inputs
            b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
            b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
            b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
            b_b = tl.load(p_b).to(tl.float32)

            # Compute sigmoid gating
            # Load gating parameters
            b_A_log = tl.load(p_A_log).to(tl.float32)
            b_a = tl.load(p_a).to(tl.float32)
            b_dt_bias = tl.load(p_dt_bias).to(tl.float32)

            # Compute g = -exp(A_log) * softplus(a + dt_bias)
            x = b_a + b_dt_bias
            beta_x = softplus_beta * x
            # Apply softplus with numerical stability
            softplus_x = tl.where(
                beta_x <= softplus_threshold,
                (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
                x,
            )
            b_g = -tl.exp(b_A_log) * softplus_x

            # Compute beta = sigmoid(b)
            b_beta = 1.0 / (1.0 + tl.exp(-b_b))

            # Apply L2 normalization if enabled
            if USE_QK_L2NORM_IN_KERNEL:
                b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q)) + 1e-6)
                b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k)) + 1e-6)

            b_q = b_q * scale

            # Apply gating to hidden state: h *= exp(g)
            b_h *= tl.exp(b_g)

            # Delta rule: v -= sum(h * k, dim=0)
            b_v -= tl.sum(b_h * b_k[:, None], 0)

            # Apply beta gating: v *= beta
            b_v *= b_beta

            # Update hidden state: h += k[:, None] * v[None, :]
            b_h += b_k[:, None] * b_v[None, :]

            # Compute output: o = sum(h * q, dim=0)
            b_o = tl.sum(b_h * b_q[:, None], 0)
            tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

            # Update pointers for next timestep
            p_q += stride_q
            p_k += stride_k
            p_o += HV * V
            p_v += stride_v
            p_b += stride_b
            p_a += stride_a

        # Store final state back to h0_source with bounds checking
        if USE_INITIAL_STATE:
            idx = tl.load(h0_indices + i_n).to(tl.int64)
            if idx >= 0:
                tl.device_assert(idx < h0_dim0,
                                 "idx out of bounds in h0_source store")
                # Pool layout [slots, HV, V, K] with K innermost (stride 1).
                p_h0 = (h0_source + idx * s_h0_0 + i_hv * V * K + o_k[:, None] +
                        o_v[None, :] * K)
                tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)

        i_nh += grid_stride_nh


@triton.jit
def gdn_decode_pdl_update_kernel(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    xr,  # post-conv mixed_qkv [tokens, >= 2*KEY_DIM + HV*V] (producer output)
    b,
    o,
    h0_source,
    h0_indices,
    scale,
    total_nh,
    stride_xr,
    stride_a,
    stride_b,
    s_h0_0,
    h0_dim0,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    KEY_DIM: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    """T=1 decode specialization of the sigmoid-gating delta-rule update that
    overlaps with the preceding causal-conv1d update kernel via PDL.

    The pre-wait section performs every load that does not depend on the conv
    output -- dominated by the SSM state-pool tile (the per-layer bandwidth
    floor) -- then ``gdc_wait`` blocks until the conv producer grid completes
    before q/k/v are read from its output. The producer must be launched with
    ``launch_dependent_kernels=True`` (it fires ``gdc_launch_dependents`` at
    kernel start) and this kernel with ``launch_pdl=True``.

    Safety: everything read pre-wait (state pool, a/b projections, gating
    params, slot indices) is produced by ops that complete before the conv
    kernel *starts*, so reading it concurrently with the conv is race-free.
    """
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]
    grid_stride_nh = tl.num_programs(2)

    while i_nh < total_nh:
        i_n, i_hv = i_nh // HV, i_nh % HV
        i_h = i_hv // (HV // H)
        # Exactly one token per sequence: token index == sequence index.
        bos = i_n.to(tl.int64)
        idx = tl.load(h0_indices + i_n).to(tl.int64)

        # ---- pre-wait: loads independent of the conv output ----
        b_h = tl.zeros([BK, BV], dtype=tl.float32)
        if idx >= 0:
            tl.device_assert(idx < h0_dim0,
                             "idx out of bounds in h0_source load")
            # Pool layout [slots, HV, V, K] with K innermost (stride 1).
            p_h0 = (h0_source + idx * s_h0_0 + i_hv * V * K + o_k[:, None] +
                    o_v[None, :] * K)
            b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)
        b_b = tl.load(b + bos * stride_b + i_hv).to(tl.float32)
        b_A_log = tl.load(A_log + i_hv).to(tl.float32)
        b_a = tl.load(a + bos * stride_a + i_hv).to(tl.float32)
        b_dt_bias = tl.load(dt_bias + i_hv).to(tl.float32)

        # g = -exp(A_log) * softplus(a + dt_bias); beta = sigmoid(b)
        x = b_a + b_dt_bias
        beta_x = softplus_beta * x
        softplus_x = tl.where(
            beta_x <= softplus_threshold,
            (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
            x,
        )
        b_g = -tl.exp(b_A_log) * softplus_x
        b_beta = 1.0 / (1.0 + tl.exp(-b_b))
        b_h *= tl.exp(b_g)

        # ---- wait for the conv producer, then consume its output ----
        if USE_PDL:
            tl.extra.cuda.gdc_wait()
        p_x = xr + bos * stride_xr
        b_q = tl.load(p_x + i_h * K + o_k, mask=mask_k,
                      other=0.0).to(tl.float32)
        b_k = tl.load(p_x + KEY_DIM + i_h * K + o_k, mask=mask_k,
                      other=0.0).to(tl.float32)
        b_v = tl.load(p_x + 2 * KEY_DIM + i_hv * V + o_v,
                      mask=mask_v,
                      other=0.0).to(tl.float32)

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q)) + 1e-6)
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k)) + 1e-6)
        b_q = b_q * scale

        b_v -= tl.sum(b_h * b_k[:, None], 0)
        b_v *= b_beta
        b_h += b_k[:, None] * b_v[None, :]
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(o + (bos * HV + i_hv) * V + o_v,
                 b_o.to(o.dtype.element_ty),
                 mask=mask_v)

        if idx >= 0:
            p_h0 = (h0_source + idx * s_h0_0 + i_hv * V * K + o_k[:, None] +
                    o_v[None, :] * K)
            tl.store(p_h0, b_h.to(h0_source.dtype.element_ty), mask=mask_h)

        i_nh += grid_stride_nh


def can_use_gdn_decode_pdl_pair(
    initial_state_source: Optional[torch.Tensor],
    num_tokens: int,
    num_seqs: int,
    head_k_dim: int,
    head_v_dim: int,
    activation: Optional[str],
) -> bool:
    """Whether the standard decode step can run as the PDL-overlapped pair
    (Triton causal-conv1d producer + ``gdn_decode_pdl_update`` consumer).

    Requires PDL hardware (SM >= 90), one token per sequence, a state pool,
    and silu/swish conv activation. The FlashInfer decode dispatch is checked
    by the caller first and takes priority where available.
    """
    if os.environ.get("TRTLLM_GDN_DISABLE_PDL_DECODE_PAIR", "0") == "1":
        return False
    from tensorrt_llm._utils import get_sm_version
    if get_sm_version() < 90:
        return False
    if initial_state_source is None:
        return False
    if num_tokens != num_seqs or num_seqs == 0:
        return False
    if activation not in ("silu", "swish"):
        return False
    if triton.next_power_of_2(head_k_dim) != head_k_dim:
        return False
    return True


def gdn_decode_pdl_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    mixed_qkv: torch.Tensor,  # post-conv [tokens, >= 2*key_dim + HV*V]
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = False,
    output: Optional[torch.Tensor] = None,
    use_pdl: bool = True,
):
    """Launch the PDL decode update. ``mixed_qkv`` is the causal-conv1d output
    laid out [Q | K | V] per token; q/k/v are addressed in-kernel so no view
    plumbing is needed. Numerics are identical to the fused Triton fallback
    kernel (measured <= 1-2 bf16 ulp on the stored state, bit-equal conv pool).
    """
    H, HV, K, V = num_k_heads, num_v_heads, head_k_dim, head_v_dim
    N = a.shape[-2]

    if scale is None:
        scale = K**-0.5

    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported"
    assert mixed_qkv.stride(-1) == 1

    s_h0_0 = initial_state_source.stride(0)
    slot_num = initial_state_source.shape[0]
    # Pool layout [slots, HV, V, K] with K innermost (stride 1).
    assert initial_state_source.stride(-1) == 1
    assert initial_state_source.stride(2) == K
    assert initial_state_source.stride(1) == V * K

    o = (output.view(N, HV, V) if output is not None else mixed_qkv.new_empty(
        N, HV, V))

    grid = (NK, NV, min(N * HV, 65535))
    with custom_device_ctx(mixed_qkv.device.index):
        gdn_decode_pdl_update_kernel[grid](
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            xr=mixed_qkv,
            b=b,
            o=o,
            h0_source=initial_state_source,
            h0_indices=initial_state_indices,
            scale=scale,
            total_nh=N * HV,
            stride_xr=mixed_qkv.stride(0),
            stride_a=a.stride(-2),
            stride_b=b.stride(-2),
            s_h0_0=s_h0_0,
            h0_dim0=slot_num,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BK=BK,
            BV=BV,
            KEY_DIM=H * K,
            USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
            USE_PDL=use_pdl,
            num_warps=4,
            num_stages=3,
            launch_pdl=use_pdl,
        )
    return o.view(1, N, HV, V)


def _can_use_flashinfer_gdn_decode(
    initial_state_source: Optional[torch.Tensor],
    K: int,
    V: int,
    T: int,
    N: int,
) -> bool:
    """Check whether FlashInfer GDN bf16-state decode kernel can be used."""
    # Env-var escape hatch for A/B comparison against the Triton fallback.
    if os.environ.get("TRTLLM_FLA_DISABLE_FLASHINFER_GDN", "0") == "1":
        return False
    if not _FLASHINFER_GDN_BF16_STATE_AVAILABLE:
        return False
    # FlashInfer's GDN decode kernel is built for Hopper (SM90) and datacenter
    # Blackwell (SM100/SM103) only; on consumer Blackwell (SM120) and other archs
    # it aborts at launch -> fall back to the Triton fused-recurrent kernel.
    if not is_flashinfer_gdn_supported_arch():
        return False
    if initial_state_source is None:
        return False
    if initial_state_source.dtype != torch.bfloat16:
        return False
    if K != 128 or V != 128:
        return False
    if N == 0:
        return False
    # Standard decode only: T is the flattened token total, so T == N forces
    # exactly 1 token/sequence (making the [N, 1, ...] reshape valid). Varlen or
    # multi-token batches (T != N) can't be reshaped from T alone -> Triton.
    if T != N:
        return False

    return True


def _aligned_int32(indices: torch.Tensor) -> torch.Tensor:
    """``indices`` as a contiguous int32 tensor whose base pointer the FlashInfer
    kernel accepts (32-byte aligned). A tail slice of a larger index buffer (the
    decode requests of a mixed iteration) is copied only when misaligned."""
    out = indices.int()
    if out.data_ptr() % 32 != 0 or not out.is_contiguous():
        out = out.clone(memory_format=torch.contiguous_format)
    return out


def _flashinfer_gdn_decode(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: float,
    use_qk_l2norm_in_kernel: bool,
    cu_seqlens: torch.Tensor,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """GDN standard decode via the FlashInfer CuTe-DSL bf16-state kernel.

    Guarded to ``T_per_seq == 1`` (uses ``gated_delta_rule``); the state pool +
    indices are passed directly, no caller-side gather/scatter.
    """
    N = len(cu_seqlens) - 1
    T_total = q.shape[1]
    T_per_seq = T_total // N
    HV = v.shape[2]
    V = v.shape[3]

    # The FlashInfer CuTe-DSL kernel requires every input tensor's data pointer
    # to be 32-byte aligned (enforced in build_memref_desc). ``a`` and ``b`` are
    # per-head-scalar slices of the fused ``in_proj_ba`` output: ``b`` starts at
    # offset 0 (aligned) but ``a`` starts ``num_v_heads_per_tp`` bf16 elements in,
    # so when ``num_v_heads_per_tp`` is not a multiple of 16 (e.g. Qwen3.6-35B-A3B
    # TEP4: 32 v-heads / 4 = 8 -> 16-byte offset) the slice base is not 32-byte
    # aligned and the kernel aborts. ``.contiguous()`` is NOT enough: at decode
    # the token dim is 1, so the strided/offset slice already reports as
    # contiguous (size-1 dims are ignored by is_contiguous) and ``.contiguous()``
    # is a no-op that keeps the misaligned pointer. Clone into fresh (allocator-
    # aligned) storage instead, and only when misaligned so the common aligned
    # case (e.g. Qwen3.5-397B TEP4: 64 / 4 = 16 -> 32-byte offset) stays zero-copy.
    # q/k/v are sliced on 128-element head boundaries (>=256 B), always aligned.
    if a.data_ptr() % 32 != 0:
        a = a.clone(memory_format=torch.contiguous_format)
    if b.data_ptr() % 32 != 0:
        b = b.clone(memory_format=torch.contiguous_format)

    # Reshape from packed varlen [1, N*T, ...] to batched [N, T, ...].
    q_bat = q.view(N, T_per_seq, q.shape[2], q.shape[3])
    k_bat = k.view(N, T_per_seq, k.shape[2], k.shape[3])
    v_bat = v.view(N, T_per_seq, v.shape[2], v.shape[3])
    a_bat = a.view(N, T_per_seq, -1)
    b_bat = b.view(N, T_per_seq, -1)

    output = (output.view(N, T_per_seq, HV, V)
              if output is not None else q.new_empty(N, T_per_seq, HV, V))

    assert T_per_seq == 1, (
        f"_flashinfer_gdn_decode expects standard decode (T_per_seq == 1), got "
        f"{T_per_seq}; _can_use_flashinfer_gdn_decode should keep T == N")
    # TP8 leaves eight BF16 ``a`` values per Qwen3.5 GDN shard. Odd shards can
    # start 16 bytes into fused projection storage, but CuTe requires 32B.
    if a_bat.data_ptr() % 32:
        a_bat = a_bat.clone(memory_format=torch.contiguous_format)
    _fi_gdn_decode_bf16_state_t1(
        A_log=A_log,
        a=a_bat,
        dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        q=q_bat,
        k=k_bat,
        v=v_bat,
        b=b_bat,
        initial_state_source=initial_state_source,
        initial_state_indices=_aligned_int32(initial_state_indices),
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        scale=scale,
        output=output,
    )

    # Reshape output from [N, T, HV, V] back to [1, N*T, HV, V].
    return output.reshape(1, T_total, HV, -1)


@functools.lru_cache(maxsize=8)
def flashinfer_gdn_bf16_state_available(pool_dtype: torch.dtype, K: int,
                                        V: int) -> bool:
    """Whether FlashInfer's bf16-state GDN launchers (the T=1 decode kernels and
    the recurrent MTP kernel behind the fold tails) serve a state pool of this
    dtype and head sizes on this device: the batch-independent conditions of
    ``_can_use_flashinfer_gdn_decode`` plus the availability of the launchers,
    evaluated once instead of per layer. Independent of the direct-launch knob,
    which only picks how the kernels are launched."""
    return (_FLASHINFER_GDN_BF16_STATE_AVAILABLE
            and _fi_gdn_decode_t1_wide_vec is not None
            and _fi_select_wide_vec_tile_v is not None
            and os.environ.get("TRTLLM_FLA_DISABLE_FLASHINFER_GDN", "0") != "1"
            and is_flashinfer_gdn_supported_arch()
            and pool_dtype == torch.bfloat16 and K == 128 and V == 128)


@functools.lru_cache(maxsize=8)
def flashinfer_gdn_decode_direct_available(pool_dtype: torch.dtype, K: int,
                                           V: int) -> bool:
    """Whether ``flashinfer_gdn_decode_t1`` serves one-token decode batches on a
    state pool of this dtype and head sizes: the launchers are available and the
    direct launches are on (TLLM_GDN_FI_DIRECT_LAUNCH)."""
    return _FI_GDN_DECODE_DIRECT and flashinfer_gdn_bf16_state_available(
        pool_dtype, K, V)


def flashinfer_gdn_decode_t1(A_log: torch.Tensor, dt_bias: torch.Tensor,
                             q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                             a: torch.Tensor, b: torch.Tensor,
                             pool: torch.Tensor, indices: torch.Tensor,
                             output: torch.Tensor, scale: float) -> torch.Tensor:
    """One-token-per-sequence GDN decode straight into FlashInfer's bf16-state
    launchers, for callers that already hold the kernel's layouts.

    ``q``/``k`` ``[N, 1, H, K]``, ``v`` ``[N, 1, HV, V]``, ``a``/``b`` ``[N, 1, HV]``
    with 32-byte aligned base pointers, ``indices`` contiguous int32 (aligned),
    ``output`` ``[N, 1, HV, V]`` bf16; the pool rows ``indices`` are updated in
    place. Same dispatch as FlashInfer's ``gated_delta_rule`` (the wide-vector
    kernel from ``N * HV >= 512``, the MTP T=1 kernel below) without its per-call
    validation and without the two adapter layers of
    ``fused_sigmoid_gating_delta_rule_update`` above it: ~20 us of host time per
    GDN layer of a mixed iteration, whose host thread is the critical path.
    Callers check ``flashinfer_gdn_decode_direct_available`` and the alignment."""
    tile_v = _fi_select_wide_vec_tile_v(q.shape[0], v.shape[2])
    if tile_v is not None and tile_v >= 64:
        entry = _fi_wide_vec_compiled_entry(q.shape[2], v.shape[2], pool, A_log, dt_bias, indices, scale,
                                            tile_v)
        if entry is not None:
            # FlashInfer's compiled wide-vector kernel straight from its cache
            # (filled by the public entry on the first call for this tile):
            # the entry's per-call Python (casts, asserts, key, defaults) goes
            # from ~12 us to ~7 us. Same kernel, same arguments.
            compiled, inter = entry
            compiled(pool, inter, A_log, a, dt_bias, q, k, v, b, output, indices, indices,
                     _current_cu_stream(q.device.index))
            return output
        return _fi_gdn_decode_t1_wide_vec(
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=pool,
            initial_state_indices=indices,
            output_state_indices=None,
            intermediate_states_buffer=None,
            disable_state_update=False,
            use_qk_l2norm_in_kernel=True,
            scale=scale,
            output=output,
            tile_v=tile_v,
        )
    return _fi_gdn_decode_bf16_state_mtp(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=pool,
        initial_state_indices=indices,
        output_state_indices=None,
        output=output,
        use_qk_l2norm_in_kernel=True,
        scale=scale,
    )


def flashinfer_gdn_tail_recurrent(A_log: torch.Tensor, dt_bias: torch.Tensor,
                                  q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                  a: torch.Tensor, b: torch.Tensor,
                                  pool: torch.Tensor, s1: torch.Tensor,
                                  s2: torch.Tensor, output: torch.Tensor,
                                  scale: float) -> torch.Tensor:
    """A folded save-last tail of ``n`` tokens through FlashInfer's recurrent
    (MTP) kernel: the state is read from pool slot ``s1`` and written to slot
    ``s2`` (split pool), q/k are L2-normalised and the gates computed in the
    kernel from the raw conv output rows and the raw ``a`` / ``b`` columns.

    ``q``/``k`` ``[1, n, H, K]``, ``v`` ``[1, n, HV, V]``, ``a``/``b`` ``[1, n, HV]``
    (32-byte aligned base pointers), ``s1``/``s2`` int32 ``[1]``, ``output``
    ``[1, n, HV, V]`` bf16. The chunked kernel spends ~35 us on such a tail
    regardless of its length (state load / store and setup); the recurrent
    kernel 12-14 us for n <= 32, with the terminal state within one bf16 ulp of
    the two-chunk schedule (fold_tail_mtp_bench.py). The kernel is compiled per
    ``n`` (~1 s each); warm up the lengths that occur."""
    n = q.shape[1]
    entry = _fi_tail_recurrent_compiled(n, q.shape[2], v.shape[2], pool, A_log, dt_bias, s1, scale)
    if entry is not None:
        # The compiled kernel from FlashInfer's own cache (filled by the public
        # entry on the first call for this tail length): the entry's ~14 us of
        # per-call Python (dtype casts, asserts, key building, defaults) become
        # ~7 us. Same kernel, same arguments as the public entry passes.
        compiled, defaults, inter = entry
        compiled(pool, inter, A_log, a, dt_bias, q, k, v, b, output, s1, s2,
                 defaults["accepted_steps"], defaults["ssm_state_indices"],
                 _current_cu_stream(q.device.index))
        return output
    return _fi_gdn_decode_bf16_state_mtp(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=pool,
        initial_state_indices=s1,
        output_state_indices=s2,
        output=output,
        use_qk_l2norm_in_kernel=True,
        scale=scale,
    )


_FI_TAIL_DIRECT = _FI_GDN_DECODE_DIRECT
_fi_tail_entries: dict = {}
_fi_wide_vec_entries: dict = {}


def _fi_wide_vec_compiled_entry(H: int, HV: int, pool: torch.Tensor, A_log: torch.Tensor,
                                dt_bias: torch.Tensor, indices: torch.Tensor, scale: float,
                                tile_v: int):
    """``(compiled, dummy_intermediate)`` of FlashInfer's wide-vector T=1 decode
    kernel for this pool geometry / tile, or None when the shortcut is off,
    FlashInfer's internals are unavailable or the kernel is not compiled yet."""
    if not _FI_TAIL_DIRECT or _fi_wide_vec_compiled is None:
        return None
    key = (H, HV, pool.shape[0], tuple(pool.stride()), pool.dtype, A_log.dtype, dt_bias.dtype,
           indices.dtype, scale, tile_v)
    entry = _fi_wide_vec_entries.get(key)
    if entry is not None:
        return entry
    K = V = pool.shape[-1]
    contiguous = pool.is_contiguous()
    cache_key = (
        "v3_mtp_bf16_tiled_dynB", 1, H, HV, K, V,
        -1 if contiguous else pool.shape[0],
        (-1,) if contiguous else tuple(int(s) for s in pool.stride()),
        tile_v,
        False,  # effective_disable_final
        False,  # cache_intermediate_states
        True,   # use_qk_l2norm_in_kernel
        scale, 1.0, 20.0,
        _fi_use_packed_fma,
        True,   # same_pool: read and write the same slots
        _fi_dtype_key(A_log, dt_bias, indices),
    )
    cache = _fi_wide_vec_compiled.get(cache_key)
    if cache is None:
        return None
    entry = (cache["compiled"], pool[:1, :1, :1])
    if len(_fi_wide_vec_entries) >= 256:
        _fi_wide_vec_entries.clear()
    _fi_wide_vec_entries[key] = entry
    return entry


def _fi_tail_recurrent_compiled(n: int, H: int, HV: int, pool: torch.Tensor,
                                A_log: torch.Tensor, dt_bias: torch.Tensor,
                                indices: torch.Tensor, scale: float):
    """``(compiled, defaults_for_B=1, dummy_intermediate)`` of FlashInfer's MTP
    kernel for a one-sequence tail of ``n`` tokens with split-pool state I/O, or
    None when the shortcut is off, FlashInfer's internals are unavailable or the
    kernel has not been compiled yet (the public entry compiles it)."""
    if not _FI_TAIL_DIRECT or _fi_mtp_compiled is None:
        return None
    key = (n, H, HV, pool.shape[0], tuple(pool.stride()), pool.dtype, A_log.dtype, dt_bias.dtype,
           indices.dtype, scale)
    entry = _fi_tail_entries.get(key)
    if entry is not None:
        return entry
    K = V = pool.shape[-1]
    tile_v, ilp_rows = _fi_get_bf16_mtp_config(1, n, HV, V)
    contiguous = pool.is_contiguous()
    cache_key = (
        "mtp_bf16_dynB", n, H, HV, K, V,
        -1 if contiguous else pool.shape[0],
        (-1,) if contiguous else tuple(int(s) for s in pool.stride()),
        tile_v, ilp_rows,
        False,  # disable_state_update
        False,  # cache_intermediate_states
        True,   # use_qk_l2norm_in_kernel
        scale, 1.0, 20.0,  # scale, softplus_beta, softplus_threshold
        _fi_use_packed_fma,
        False,  # same_pool: the tail reads S1 and writes S2
        False,  # disable_output
        False,  # per_request_accepted_steps
        False,  # per_token_pool_scatter
        False,  # per_token_pool_scatter_flat
        _fi_dtype_key(A_log, dt_bias, indices),
    )
    cache = _fi_mtp_compiled.get(cache_key)
    if cache is None or 1 not in cache.get("defaults_by_B", {}):
        return None
    entry = (cache["compiled"], cache["defaults_by_B"][1], pool[:1, :1, :1])
    if len(_fi_tail_entries) >= 256:
        _fi_tail_entries.clear()
    _fi_tail_entries[key] = entry
    return entry


def _can_use_flashinfer_gdn_verify(
    initial_state_source: Optional[torch.Tensor],
    head_k_dim: int,
    head_v_dim: int,
    draft_token_num: int,
) -> bool:
    """Whether the FlashInfer MTP kernel should serve the speculative verify step.

    Default ON when eligible; set ``TRTLLM_FLA_DISABLE_FLASHINFER_GDN_VERIFY=1``
    to force the Triton recurrent verify kernel (``TRTLLM_FLA_DISABLE_FLASHINFER_GDN=1``
    disables all FlashInfer GDN decode paths, including this one). The same
    constraints as the decode path apply (bf16 state pool, K==V==128, supported
    arch, FI MTP API available) plus a per-sequence draft length in
    [1, _FI_GDN_MAX_MTP_T]; longer drafts fall back to Triton.
    """
    if os.environ.get("TRTLLM_FLA_DISABLE_FLASHINFER_GDN", "0") == "1":
        return False
    if os.environ.get("TRTLLM_FLA_DISABLE_FLASHINFER_GDN_VERIFY", "0") == "1":
        return False
    if not _FLASHINFER_GDN_BF16_STATE_AVAILABLE:
        return False
    if not is_flashinfer_gdn_supported_arch():
        return False
    if initial_state_source is None or initial_state_source.dtype != torch.bfloat16:
        return False
    if head_k_dim != 128 or head_v_dim != 128:
        return False
    if not (1 <= draft_token_num <= _FI_GDN_MAX_MTP_T):
        return False
    return True


def _flashinfer_gdn_verify(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    intermediate_states_buffer: torch.Tensor,
    scale: float,
    use_qk_l2norm_in_kernel: bool,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """GDN MTP *verify* via the FlashInfer bf16-state kernel.

    Inputs are batched ``[N, draft_token_num, H, D]``. The kernel gathers the
    initial state from the pool via ``initial_state_indices`` (no host-side
    gather copy), writes the SSM state after each draft token into the
    batch-scoped ``intermediate_states_buffer`` (``[N, draft_token_num, HV, V,
    K]``, matching the Triton verify kernel) and leaves the live state pool
    untouched (``disable_state_update``) so the cache manager selects the
    accepted-position state afterwards. Returns the attention output
    ``[N, draft_token_num, HV, V]``.
    """
    logger.info_once(
        "Using FlashInfer CuTe-DSL kernel for GDN MTP verify "
        "(bf16 state, K=V=128)",
        key="flashinfer_gdn_verify")
    N, T = q.shape[0], q.shape[1]
    HV, V = v.shape[2], v.shape[3]
    output = (output.view(N, T, HV, V) if output is not None else q.new_empty(
        N, T, HV, V))
    # The FI CuTe-DSL kernel asserts 32-byte data alignment on every tensor
    # argument. ``a`` starts ``num_v_heads_per_tp`` bf16 elements into the fused
    # ``in_proj_ba`` output, so it is misaligned when that count is not a
    # multiple of 16 (e.g. Qwen3.5 TEP16: 128 / 16 = 8) -- see the note in
    # _flashinfer_gdn_decode for why this clones instead of .contiguous().
    if a.data_ptr() % 32 != 0:
        a = a.clone(memory_format=torch.contiguous_format)
    if b.data_ptr() % 32 != 0:
        b = b.clone(memory_format=torch.contiguous_format)
    # The int32 index tensor may likewise be a slice of a larger buffer
    # (e.g. state_indices_d = cache_indices[num_prefills:]) whose 4*offset
    # storage offset breaks alignment; .int() is a no-op for int32, so realign
    # with an explicit copy when needed.
    initial_state_indices = initial_state_indices.int()
    if initial_state_indices.data_ptr() % 32 != 0:
        initial_state_indices = initial_state_indices.clone()
    _fi_gdn_decode_bf16_state_mtp(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=initial_state_source,
        initial_state_indices=initial_state_indices,
        intermediate_states_buffer=intermediate_states_buffer,
        disable_state_update=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        scale=scale,
        output=output,
    )
    return output


def fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
):
    """
    Fused triton implementation of sigmoid gating delta rule update.
    This function uses a single fused kernel that combines both sigmoid gating computation
    and the recurrent delta rule update for better performance.

    When FlashInfer's CuTe-DSL GDN decode kernel is available and the state
    dtype is bfloat16, dispatches to the faster FlashInfer path automatically.
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    if scale is None:
        scale = k.shape[-1]**-0.5
    else:
        assert scale > 0, "scale must be positive"

    # Dispatch to FlashInfer CuTe-DSL kernel when available and conditions met.
    if (cu_seqlens is not None and _can_use_flashinfer_gdn_decode(
            initial_state_source, K, V, T, N)):
        logger.info_once(
            "Using FlashInfer CuTe-DSL kernel for GDN decode "
            "(bf16 state, K=V=128)",
            key="flashinfer_gdn_decode")
        return _flashinfer_gdn_decode(
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=initial_state_source,
            initial_state_indices=initial_state_indices,
            scale=scale,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
            output=output,
        )

    # Fallback: Triton kernel path.
    # Accept native view layouts from forward_decode rather than forcing packed
    # copies through input_guard.
    stride_q = q.stride(1)
    stride_k = k.stride(1)
    stride_v = v.stride(1)
    stride_a = a.stride(-2)
    stride_b = b.stride(-2)
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 1

    o = output.unsqueeze(0) if output is not None else q.new_empty(NK, *v.shape)
    # (NK, NV, N * HV) is found faster than (N * HV, NV, NK)
    # As max of grid.z is 65535, we cap grid.z and let each Triton program
    # grid-stride across the remaining N * HV tiles.
    grid = (NK, NV, min(N * HV, 65535))

    if initial_state_source is not None:
        s_h0_0, s_h0_1, s_h0_2, s_h0_3 = initial_state_source.stride()
        slot_num = initial_state_source.shape[0]
        # Pool layout is [slots, HV, V, K] with K innermost (stride 1).
        assert s_h0_3 == 1, f"s_h0_3: {s_h0_3} is not 1"
        assert s_h0_2 == K, f"s_h0_2: {s_h0_2} is not {K}"
        assert s_h0_1 == V * K, f"s_h0_1: {s_h0_1} is not {V * K}"
    else:
        s_h0_0 = 0
        slot_num = 0

    # input_guard used to set the active CUDA device and make inputs contiguous.
    # We keep only the device-context part here so Triton launches on q's device
    # without re-packing the decode views.
    with custom_device_ctx(q.device.index):
        fused_sigmoid_gating_delta_rule_update_kernel[grid](
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            q=q,
            k=k,
            v=v,
            b=b,
            o=o,
            h0_source=initial_state_source,
            h0_indices=initial_state_indices,
            cu_seqlens=cu_seqlens,
            scale=scale,
            T=T,
            total_nh=N * HV,
            stride_q=stride_q,
            stride_k=stride_k,
            stride_v=stride_v,
            stride_a=stride_a,
            stride_b=stride_b,
            s_h0_0=s_h0_0,
            h0_dim0=slot_num,
            B=B,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BK=BK,
            BV=BV,
            USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    o = o.squeeze(0)
    return o
