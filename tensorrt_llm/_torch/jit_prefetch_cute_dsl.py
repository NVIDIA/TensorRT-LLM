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
"""CuTe DSL support for ``jit_prefetch.py``: compile in a helper, load in the executor.

``cute.compile`` never consults CuTe DSL's on-disk cache, so a helper cannot
warm a cache the executor reads. Instead the helper compiles the variant from
fake tensors with the real call's layout and an explicit ``--gpu-arch``,
exports it (``export_to_c``) to ``<dir>/<key>.o``, and the executor, at the
point it would call ``cute.compile``, loads that file with
``cute.runtime.load_module``: a ~1 ms load replaces a compile of 0.2 s to
several seconds.

Covered kernels, each a builder in ``_BUILDERS`` / ``_PRECOMPILED``:

* the fused K1+K2+K3 kernel of Kimi K3's KDA prefill
  (``_launch_fused_k123_inv``). Its compiled form depends only on constexpr
  specializers (heads, varlen, bias, safe gate, beta sigmoid, chunk alignment
  ``varlen_pure``, cu/ci dtypes); shapes are runtime arguments, and which
  variant a batch needs is known on the host before the forward pass.
* the persistent K4 kernel of the same prefill (``_launch_k4_persistent``).
  The op keys its compile cache on the full layout of the recurrent-state
  pool it updates. With a BF16 state cache the KDA mixer stages the batch's
  rows into a dense FP32 copy, so that pool has one row per context request,
  and a batch with a new request count compiles K4 again (~1.4 s). The pool
  goes in with dynamic extents and strides, so all those compiles produce
  the same function: one export per (heads, head dims, stride width) serves
  every request count.
* FlashInfer's CuTe DSL RMSNorm and fused add + RMSNorm (TVM-FFI kernels).
  The compiled form depends on dtype, hidden size, weight bias, PDL, SM
  version and whether the rows are contiguous (``flashinfer.norm``'s own
  cache key); every combination an engine's norm layers can reach is
  enumerated once from the model.
* the radix top-k runners of the sparse-attention indexers (TVM-FFI),
  keyed on the score width bucket, the decode row band and the
  engine-fixed top_k / next_n / dtype (``TopKProvider``).
* CSA2's candidate publication (``Csa2PublishProvider``), keyed on the
  layer's block geometry and the power-of-two block capacity.
* the FP8 blockwise GEMM's autotune tactics (``Fp8GemmProvider``), for the
  block-FP8 Linear layers that opt out of DeepGEMM and for the grouped FP8
  O-LoRA projection of DeepSeek-V4 / V4.1 attention, whose batched GEMM is
  the same TVM-FFI kernel.
* FlashInfer's low-M BF16 direct GEMM (``LowMDirectGemmProvider``), which
  ``modules.low_m_gemm``'s AutoTuner race benchmarks: specialized on M, N, K
  and the tactic, so one variant per (M bucket, Linear shape, tactic).

A spec is canonical JSON naming the kernel and its compile-time arguments;
its SHA-1 is the file name, so helper and executor agree without sharing
state. The helper runs with no visible GPU: the device properties kernel
constructors and the CuTe DSL compiler read (SM version, SM count, shared
memory, the largest cluster) are taken from the executor's device and
answered by a shim (``device_props`` / ``install_device_shim``).
"""

# No ``from __future__ import annotations``: CuTe DSL reads the scalar
# annotations of the ``@cute.jit`` wrapper below at trace time and resolves
# string annotations against module globals, where ``cutlass`` is not bound.
import hashlib
import inspect
import json
import os
import types
from typing import Any, Callable, Dict, List, Optional, Tuple

KIND = "cute_dsl"
_K123 = "kda_fused_k123"
_K4 = "kda_k4_persistent"
_FI_NORM = "flashinfer_norm"
_BT = 64
_K_DIM = 128


def spec_key(spec: str) -> str:
    return hashlib.sha1(spec.encode()).hexdigest()[:20]


def k123_spec(
    H: int,
    has_bias: bool,
    safe_gate: bool,
    varlen_pure: bool,
    cu_dtype: str,
    ci_dtype: str,
    use_beta_sigmoid: bool,
    beta_dtype: str,
) -> str:
    """Spec for one varlen K123 variant (the prefill path; varlen fixes B=1)."""
    return json.dumps(
        {
            "kernel": _K123,
            "H": int(H),
            "has_bias": bool(has_bias),
            "safe_gate": bool(safe_gate),
            "varlen_pure": bool(varlen_pure),
            "cu": cu_dtype,
            "ci": ci_dtype,
            "beta_sigmoid": bool(use_beta_sigmoid),
            "beta": beta_dtype,
        },
        sort_keys=True,
    )


def k123_spec_from_cache_key(cache_key: tuple) -> Optional[str]:
    """Spec for a ``_fused_k123_cache`` key, or None if not a varlen key."""
    H, is_varlen, _dev, has_bias, safe_gate, varlen_pure, cu_ci, _t, beta_sig, beta_dt = cache_key
    if not is_varlen or cu_ci is None:
        return None
    return k123_spec(
        H,
        has_bias,
        safe_gate,
        varlen_pure,
        str(cu_ci[0]).replace("torch.", ""),
        str(cu_ci[1]).replace("torch.", ""),
        beta_sig,
        str(beta_dt).replace("torch.", ""),
    )


def k4_spec(num_sm: int, H: int, K: int, V: int, pool_32bit_stride: bool) -> str:
    """Spec for one persistent K4 variant (``_launch_k4_persistent``).

    The op passes the state pool with every extent and every stride but the
    innermost (1) dynamic, and the state indices as a vector of dynamic
    extent, so the compiled function does not depend on the pool's row
    count: the op's ``_k4p_cache`` key separates pools of different row
    counts, but every such key maps to one spec. What does change the
    compiled function is whether the pool's dynamic strides are 32-bit.
    """
    return json.dumps(
        {
            "kernel": _K4,
            "num_sm": int(num_sm),
            "H": int(H),
            "K": int(K),
            "V": int(V),
            "pool_32bit": bool(pool_32bit_stride),
        },
        sort_keys=True,
    )


def k4_spec_from_cache_key(cache_key: tuple) -> Optional[str]:
    """Spec for a ``_k4p_cache`` key, or None if its layout is not one the
    builder reproduces: an FP32 [slots, H, V, K] pool whose innermost stride
    is 1, and int32 state indices with 32-bit strides."""
    _dev, num_sm, H, K, V, pool_key, indices_key = cache_key
    pool_dtype, pool_shape, pool_stride, pool_32 = pool_key
    indices_dtype, indices_stride, indices_32 = indices_key
    if (
        str(pool_dtype) != "torch.float32"
        or len(pool_shape) != 4
        or pool_stride[-1] != 1
        or str(indices_dtype) != "torch.int32"
        or tuple(indices_stride) != (1,)
        or not indices_32
    ):
        return None
    return k4_spec(num_sm, H, K, V, pool_32)


def fi_norm_spec(
    op: str,
    dtype_str: str,
    H: int,
    weight_bias: float,
    enable_pdl: bool,
    sm_version: int,
    contiguous: bool,
) -> str:
    """Spec for one FlashInfer CuTe DSL norm variant.

    ``op`` is ``"rmsnorm"`` or ``"fused_add_rmsnorm"``; the other fields are
    the arguments of FlashInfer's ``_get_compiled_<op>_kernel`` getter, which
    is FlashInfer's own cache key for the compiled kernel.
    """
    return json.dumps(
        {
            "kernel": _FI_NORM,
            "op": op,
            "dtype": dtype_str,
            "H": int(H),
            "weight_bias": float(weight_bias),
            "pdl": bool(enable_pdl),
            "sm": int(sm_version),
            "contiguous": bool(contiguous),
        },
        sort_keys=True,
    )


def _cutlass_int(name: str):
    import cutlass

    return {"int64": cutlass.Int64, "int32": cutlass.Int32}[name]


def _build_k123(s: Dict[str, Any]):
    """Return (host_fn, fake_args, compile_options) for one varlen K123 variant.

    The fake tensors reproduce the layouts ``_launch_fused_k123_inv`` passes:
    q/k/g [1, T, H, K] bf16, beta [1, T, H], scratch buffers from
    ``_get_buffers`` (T + one chunk of slack), cu [N+1], ci [NT, 2]. ``from_dlpack``
    wrappers are static-layout, so T / N / NT here are placeholders: the
    kernel reads every extent from its runtime scalar arguments.
    """
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from .attention.kernels.blackwell.kimi_k3_kda.fused_k123 import make_host_function

    H, K, BT = s["H"], _K_DIM, _BT
    T, NT, N = 337, 7, 3  # placeholders; see docstring
    Ta = T + BT
    inner = make_host_function(
        1,
        NT,
        H,
        is_varlen=True,
        T_padded=T,
        has_bias=s["has_bias"],
        use_safe_gate=s["safe_gate"],
        use_beta_sigmoid=s["beta_sigmoid"],
        varlen_pure=s["varlen_pure"],
    )

    @cute.jit
    def host_fn(
        mQ, mK, mG, mA_log, mBeta, mBetaActivated, scale: cutlass.Float32,
        mKscaled, mKg, mQscaled, mGkLast, mAqk, mAkk, mCuSeqlens, mChunkIndices,
        mDtBias, lower_bound_val: cutlass.Float32, rt_nt: cutlass.Int32,
        rt_b: cutlass.Int32, rt_t_total: cutlass.Int32, valid_tokens: cutlass.Int32,
        stream: cuda.CUstream,
    ):  # fmt: skip
        # export_to_c types scalars from annotations; the kernel's host_fn
        # leaves scale / lower_bound unannotated.
        inner(mQ, mK, mG, mA_log, mBeta, mBetaActivated, scale, mKscaled, mKg, mQscaled,
              mGkLast, mAqk, mAkk, mCuSeqlens, mChunkIndices, mDtBias, lower_bound_val,
              rt_nt, rt_b, rt_t_total, valid_tokens, stream)  # fmt: skip

    def t(dtype, *shape):
        stride, acc = [], 1
        for d in reversed(shape):
            stride.append(acc)
            acc *= d
        return cute.runtime.make_fake_tensor(
            dtype, tuple(shape), tuple(reversed(stride)), assumed_align=16
        )

    bf, f32 = cutlass.BFloat16, cutlass.Float32
    beta_t = f32 if s["beta"] == "float32" else bf
    cu_t, ci_t = _cutlass_int(s["cu"]), _cutlass_int(s["ci"])
    args = [
        t(bf, 1, T, H, K),  # q
        t(bf, 1, T, H, K),  # k
        t(bf, 1, T, H, K),  # g
        t(f32, H),  # A_log
        t(beta_t, 1, T, H),  # beta
        t(bf, 1, Ta, H),  # beta_activated
        cutlass.Float32(1.0),  # scale
        t(bf, 1, Ta, H, K),  # k_scaled
        t(bf, 1, Ta, H, K),  # kg
        t(bf, 1, Ta, H, K),  # q_scaled
        t(f32, 1, NT, H, K),  # gk_last_exp
        t(bf, 1, Ta, H, BT),  # A_qk
        t(bf, 1, Ta, H, BT),  # A_kk
        t(cu_t, N + 1),  # cu_seqlens
        t(ci_t, NT, 2),  # chunk_indices
        t(f32, H, K) if s["has_bias"] else t(f32, 1, 1),  # dt_bias
        cutlass.Float32(0.0),  # lower_bound
        cutlass.Int32(NT),
        cutlass.Int32(1),
        cutlass.Int32(T),
        cutlass.Int32(T),
        cute.runtime.make_fake_stream(),
    ]
    return host_fn, args, ""


def _build_k4(s: Dict[str, Any]):
    """Return (host_fn, fake_args, compile_options) for one persistent K4 variant.

    The fake tensors reproduce the CuTe types ``_launch_k4_persistent`` passes:
    the [T, H, dim] activations and scratch with every extent and every stride
    but the innermost dynamic (``mark_layout_dynamic``, 64-bit strides); the
    [slots, H, V, K] state pool likewise, with 32-bit strides when its rows
    fit in int32 offsets (``_dynamic_dlpack_arg``); the state indices,
    cu_seqlens and chunk offsets as vectors of dynamic extent; the tensormap
    workspace static (one slot per SM of the persistent grid).
    """
    import cutlass
    import cutlass.cute as cute

    from .attention.kernels.blackwell.kimi_k3_kda.k4_persistent import (
        BYTES_PER_TENSORMAP,
        NUM_TENSORMAPS,
        make_host_fn,
    )

    num_sm = s["num_sm"]

    def act(dtype):
        return cute.runtime.make_fake_tensor(
            dtype,
            (cute.sym_int(), cute.sym_int(), cute.sym_int()),
            (cute.sym_int64(), cute.sym_int64(), 1),
            assumed_align=16,
        )

    width = 32 if s["pool_32bit"] else 64
    pool = cute.runtime.make_fake_tensor(
        cutlass.Float32,
        tuple(cute.sym_int() for _ in range(4)),
        (cute.sym_int(width), cute.sym_int(width), cute.sym_int(width), 1),
        assumed_align=16,
    )

    def vec(align):
        return cute.runtime.make_fake_tensor(
            cutlass.Int32, (cute.sym_int(),), (1,), assumed_align=align
        )

    bf, f32 = cutlass.BFloat16, cutlass.Float32
    tm = cute.runtime.make_fake_tensor(
        cutlass.Uint8, (num_sm * NUM_TENSORMAPS * BYTES_PER_TENSORMAP,), (1,), assumed_align=16
    )
    args = [
        act(bf),  # A_kk
        act(bf),  # k_scaled
        act(bf),  # v
        act(bf),  # q_scaled
        act(bf),  # A_qk
        act(bf),  # kg
        act(bf),  # o
        act(f32),  # gk_last_exp
        pool,
        vec(16),  # state_indices
        vec(4),  # cu_seqlens
        vec(4),  # chunk offsets
        tm,
        cutlass.Int32(1),  # num_seqs
        cute.runtime.make_fake_stream(),
    ]
    return make_host_fn(num_sm=num_sm), args, ""


def _build_fi_norm(s: Dict[str, Any]):
    """Return (kernel, fake_args, compile_options) for one FlashInfer norm.

    The kernel object and fake tensors are built exactly as FlashInfer's
    ``_get_compiled_rmsnorm_kernel`` / ``_get_compiled_fused_add_rmsnorm_kernel``
    build them (row count symbolic, compact or symbolically strided rows), so
    the export has the same TVM-FFI signature as the kernel FlashInfer would
    compile.
    """
    import math

    import cutlass.cute as cute
    from cutlass import Float32, Int64
    from flashinfer.norm.utils import get_cutlass_dtype

    dtype = get_cutlass_dtype(s["dtype"])
    H, contiguous = s["H"], s["contiguous"]
    if s["op"] == "rmsnorm":
        from flashinfer.norm.kernels.rmsnorm import RMSNormKernel as Kernel
    elif s["op"] == "fused_add_rmsnorm":
        from flashinfer.norm.kernels.fused_add_rmsnorm import FusedAddRMSNormKernel as Kernel
    else:
        raise ValueError(f"unknown FlashInfer norm op {s['op']!r}")
    kernel_obj = Kernel(dtype, H, s["weight_bias"], sm_version=s["sm"])
    sym_m = cute.sym_int(64)
    if contiguous:
        align = math.gcd(128, H * (dtype.width // 8))

        def row_tensor():
            return cute.runtime.make_fake_compact_tensor(
                dtype, (sym_m, H), stride_order=(1, 0), assumed_align=align
            )
    else:

        def row_tensor():
            stride = cute.sym_int64(divisibility=kernel_obj.vec_size)
            return cute.runtime.make_fake_tensor(dtype, (sym_m, H), (stride, 1), assumed_align=16)

    a, b = row_tensor(), row_tensor()
    w = cute.runtime.make_fake_compact_tensor(dtype, (H,), assumed_align=16)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    if s["op"] == "rmsnorm":
        args = [a, w, b]  # (input, weight, out)
    else:
        args = [a, b, w]  # (input, residual, weight)
    args += [Int64(1), Float32(1e-6), s["pdl"], stream]
    return kernel_obj, args, "--enable-tvm-ffi"


# Radix top-k runners in custom_ops.cute_dsl_custom_ops whose ``_compile``
# builds the kernel from fake tensors and its cache key alone (TVM-FFI).
_TOPK = "cute_dsl_topk"
TOPK_RUNNERS = (
    "CuteDSLTopKDecodeSingleCTARunner",
    "CuteDSLTopKDecodeRadixFilterSPMultiCTARunner",
    "CuteDSLTopKPrefillSingleCTARunner",
    "CuteDSLTopKDecodeSinglePassMultiCTARunner",
    "CuteDSLTopKDecodeSinglePassMultiCTAClusterRunner",
)


def topk_spec(runner: str, key: tuple) -> str:
    """Spec for one radix top-k runner's ``_compile(*key)``. The key holds
    cutlass dtypes, ints, bools and strings."""
    return json.dumps({"kernel": _TOPK, "runner": runner, "key": _encode(key)}, sort_keys=True)


def _encode(v: Any) -> Any:
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    if isinstance(v, (tuple, list)):
        return {"tuple": [_encode(x) for x in v]}
    name = getattr(v, "__name__", None)
    import cutlass

    if name is not None and getattr(cutlass, name, None) is v:
        return {"cutlass": name}
    raise TypeError(f"cannot encode top-k key element {v!r}")


def _decode(v: Any) -> Any:
    if isinstance(v, dict):
        if "tuple" in v:
            return tuple(_decode(x) for x in v["tuple"])
        import cutlass

        return getattr(cutlass, v["cutlass"])
    return v


def _build_topk(s: Dict[str, Any]):
    """Return the compiled top-k kernel (``_compile`` builds it from fake
    tensors; the helper exports it)."""
    from .custom_ops import cute_dsl_custom_ops as ops

    runner = getattr(ops, s["runner"])
    key = _decode(s["key"])
    runner._compile(*key)
    cached = runner.kernel_cache[key]
    return cached[0] if isinstance(cached, tuple) else cached


# CSA2's candidate publication (attention/backends/sparse/csa2/kernel.py),
# keyed like its ``_launch``: (block_size, padded blocks, sparse block) plus
# each pointer argument's (element type, assumed alignment).
_CSA2_PUBLISH = "csa2_candidate_publish"


def csa2_publish_spec(block_size: int, padded: int, sparse_block: int, ptrs) -> str:
    """Spec for one CandidatePublishKernel compile; ``ptrs`` is the five
    pointer arguments' ``(element type name, assumed alignment)``."""
    return json.dumps(
        {
            "kernel": _CSA2_PUBLISH,
            "block_size": int(block_size),
            "padded": int(padded),
            "sparse_block": int(sparse_block),
            "ptrs": [[str(e), int(a)] for e, a in ptrs],
        },
        sort_keys=True,
    )


# Element types a CSA2 glue-kernel pointer can have (csa2.kernel._cute_dtype).
_CSA2_ELEMENTS = ("Float32", "Int64", "Int32", "Uint32", "Uint8", "BFloat16", "Float8E4M3FN")


def _cutlass_element_name(text: str) -> str:
    """The ``cutlass`` attribute name of an element type given as
    ``str(type)`` (what ``csa2.kernel._ptr`` records in a launch key)."""
    import cutlass

    for name in _CSA2_ELEMENTS:
        if text == name or text == str(getattr(cutlass, name)):
            return name
    raise ValueError(f"unknown CSA2 pointer element type {text!r}")


def csa2_spec_from_launch_key(key: tuple) -> Optional[str]:
    """Spec for a ``csa2.kernel._launch`` key, or None if it is not a
    candidate publication (the only CSA2 glue kernel prefetched)."""
    (name, *fields), signature = key
    if name != "candidate_publish":
        return None
    block_size, padded, sparse_block = fields[:3]
    ptrs = [(_cutlass_element_name(e), a) for e, a in (s for s in signature if s is not None)]
    return csa2_publish_spec(block_size, padded, sparse_block, ptrs)


def _build_csa2_publish(s: Dict[str, Any]):
    """Return (kernel, fake args, options) for one candidate publication,
    with the argument kinds ``csa2.kernel.publish_candidates_`` passes."""
    import cutlass
    import cutlass.cute as cute

    from .attention.backends.sparse.csa2 import kernel as csa2

    ns = csa2._glue_kernels()
    fn = ns.CandidatePublishKernel(s["block_size"], s["padded"], s["sparse_block"])
    ptrs = []
    for element, align in s["ptrs"]:
        dtype = getattr(cutlass, element)
        # Any address with the recorded alignment: the compile reads only the
        # type and the assumed alignment.
        ptrs.append(ns.make_ptr(dtype, 1 << 20, cute.AddressSpace.gmem, assumed_align=align))
    i32, i64 = cutlass.Int32, cutlass.Int64
    args = [*ptrs, i32(1), i32(1), i32(1), i32(1), i32(1), i64(1), i64(1), i64(1)]
    args.append(cute.runtime.make_fake_stream())
    return fn, args, ""


# The FP8 blockwise GEMM (custom_ops' CuteDSLFp8BlackwellRunner, the block-FP8
# Linear path of models that opt out of DeepGEMM), keyed like the runner's
# kernel cache: the tactic, plus the cluster occupancy the runner passes as a
# compile-time constant.
_FP8_GEMM = "cute_dsl_fp8_blockwise_gemm"


def fp8_gemm_spec(
    use_2cta_instrs: bool, mma_tiler_mn, cluster_shape_mn, max_active_clusters: int
) -> str:
    """Spec for one TVM-FFI ``CuteDSLFp8BlackwellRunner`` kernel."""
    return json.dumps(
        {
            "kernel": _FP8_GEMM,
            "use_2cta": bool(use_2cta_instrs),
            "mma_tiler": [int(x) for x in mma_tiler_mn],
            "cluster": [int(x) for x in cluster_shape_mn],
            "max_active_clusters": int(max_active_clusters),
        },
        sort_keys=True,
    )


def _build_fp8_gemm(s: Dict[str, Any]):
    """Return (wrapper, fake args, options) for one FP8 blockwise GEMM tactic.

    The kernel takes every extent as a runtime scalar and its operands as
    pointers (16-byte aligned, as the runner makes them); only the output is a
    tensor, the runner's ``(m, n, 1)`` view with strides ``(n, 1, m * n)``
    marked dynamic except the unit stride, which a fake tensor of 32-bit
    extents and 64-bit strides reproduces. So any address and extent compile
    the same kernel the runner would compile for any shape.
    """
    import cutlass
    import cutlass.cute as cute

    from .kernels.blackwell.blockwise_gemm.blockwise_gemm import Sm100BlockwiseGemmKernel
    from .kernels.blackwell.utils import make_ptr

    gemm = Sm100BlockwiseGemmKernel(
        cutlass.Float32,
        use_2cta_instrs=s["use_2cta"],
        mma_tiler_mn=tuple(s["mma_tiler"]),
        cluster_shape_mn=tuple(s["cluster"]),
    )

    def ptr(dtype):
        return make_ptr(dtype, 1 << 20, cute.AddressSpace.gmem, assumed_align=16)

    c = cute.runtime.make_fake_tensor(
        cutlass.BFloat16,
        (cute.sym_int(), cute.sym_int(), cute.sym_int()),
        stride=(cute.sym_int64(), 1, cute.sym_int64()),
    )
    i32 = cutlass.Int32
    args = [i32(1), i32(1), i32(1), i32(1), i32(1), i32(1), i32(1)]  # m n k sf_m sf_n sf_k batch
    args += [ptr(cutlass.Float8E4M3FN), ptr(cutlass.Float8E4M3FN)]
    args += [ptr(cutlass.Float32), ptr(cutlass.Float32), c]
    kwargs = {
        "max_active_clusters": s["max_active_clusters"],
        "stream": cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    }
    return gemm.wrapper, (args, kwargs), "--opt-level 2 --enable-tvm-ffi"


# FlashInfer's low-M BF16 direct (SIMT) GEMM, which ``modules.low_m_gemm``'s
# AutoTuner race benchmarks once per tactic and runtime shape
# (``TRTLLM_LOW_M_GEMM_BACKEND``). Its compiled form is specialized on M, N
# and K, so every (M, tactic) of every eligible Linear is its own kernel:
# keyed like ``dense_bf16_gemm_direct._get_compiled_direct_kernel``.
_LOWM_DIRECT = "flashinfer_lowm_direct_gemm"


def lowm_direct_spec(m: int, n: int, k: int, tactic, use_pdl: bool) -> str:
    """Spec for one ``_get_compiled_direct_kernel(bf16, m, n, k, tactic, pdl)``."""
    return json.dumps(
        {
            "kernel": _LOWM_DIRECT,
            "m": int(m),
            "n": int(n),
            "k": int(k),
            "tactic": [int(x) for x in tactic],
            "pdl": bool(use_pdl),
        },
        sort_keys=True,
    )


def _build_lowm_direct(s: Dict[str, Any]):
    """Compile one low-M direct GEMM the way ``_get_compiled_direct_kernel``
    does (``cute.experimental.compile``, static compact BF16 operands, 32-byte
    aligned), from fake tensors, for the helper's target arch."""
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute import experimental as cute_ext
    from flashinfer.gemm.kernels import dense_bf16_gemm_direct as d

    m, n, k = s["m"], s["n"], s["k"]
    kernel = d.DirectDenseGemmKernel(
        element_type=cutlass.BFloat16,
        num_rows=m,
        k_extent=k,
        tactic=d.DirectTactic(*s["tactic"]),
        use_pdl=s["pdl"],
    )

    def t(shape):
        return cute.runtime.make_fake_compact_tensor(
            cutlass.BFloat16, shape, stride_order=(1, 0), assumed_align=32
        )

    options = d._COMPILE_OPTIONS
    arch = os.environ.get("CUTE_DSL_ARCH")
    if arch:
        options = f"{options} --gpu-arch {arch}"
    stream = cute.runtime.make_fake_stream()
    return cute_ext.compile(kernel, t((m, k)), t((n, k)), t((m, n)), stream, options=options)


_BUILDERS = {
    _K123: _build_k123,
    _K4: _build_k4,
    _FI_NORM: _build_fi_norm,
    _CSA2_PUBLISH: _build_csa2_publish,
    _FP8_GEMM: _build_fp8_gemm,
}
# Builders that return an already compiled kernel instead of (fn, args, options).
_PRECOMPILED = {_TOPK: _build_topk, _LOWM_DIRECT: _build_lowm_direct}
# Kernels compiled with TVM-FFI are loaded as TVM-FFI functions.
_TVM_FFI = {_FI_NORM, _TOPK, _FP8_GEMM}


def build(spec: str, gpu_arch: str, out_dir: str) -> bool:
    """Compile ``spec`` and export it to ``<out_dir>/<key>.o``.

    Returns True if it compiled, False if the file already existed. The file
    appears atomically (written under a temporary name, then renamed). The
    ranks of a node share ``out_dir``; a per-variant lock makes one helper
    compile while the others wait and then find the file.
    """
    import fcntl

    key = spec_key(spec)
    final = os.path.join(out_dir, f"{key}.o")
    if os.path.exists(final):
        return False
    with open(os.path.join(out_dir, f".{key}.lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if os.path.exists(final):
            return False
        _compile_export(spec, gpu_arch, out_dir, key, final)
    return True


def _compile_export(spec: str, gpu_arch: str, out_dir: str, key: str, final: str) -> None:
    import cutlass.cute as cute

    s = json.loads(spec)
    if s["kernel"] in _PRECOMPILED:
        # The kernel's own compile path (its options include the TVM-FFI
        # flag); the target arch comes from CUTE_DSL_ARCH in the helper.
        compiled = _PRECOMPILED[s["kernel"]](s)
    else:
        fn, args, options = _BUILDERS[s["kernel"]](s)
        args, kwargs = args if isinstance(args, tuple) else (args, {})
        options = f"{options} --gpu-arch {gpu_arch}".strip()
        compiled = cute.compile(fn, *args, **kwargs, options=options)
    tmp_name = f".{key}.{os.getpid()}"
    if s["kernel"] in _TVM_FFI:
        compiled.export_to_c(
            os.path.join(out_dir, f"{tmp_name}.o"),
            function_name=f"p{key}",
            export_only_tvm_ffi_symbols=True,
        )
    else:
        compiled.export_to_c(out_dir, tmp_name, f"p{key}")
    os.replace(os.path.join(out_dir, f"{tmp_name}.o"), final)
    try:
        os.remove(os.path.join(out_dir, f"{tmp_name}.h"))
    except FileNotFoundError:
        pass


def load(spec: str, out_dir: str):
    """The compiled function for ``spec`` if a helper exported it, else None."""
    path = os.path.join(out_dir, f"{spec_key(spec)}.o")
    if not os.path.exists(path):
        return None
    import cutlass.cute as cute

    tvm_ffi = json.loads(spec)["kernel"] in _TVM_FFI
    return cute.runtime.load_module(path, enable_tvm_ffi=tvm_ffi)[f"p{spec_key(spec)}"]


def gpu_arch() -> str:
    import torch

    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}a"


def device_props() -> Dict[str, Any]:
    """The properties of the executor's device that kernel constructors read
    on the host (passed to the helpers, which have no visible GPU)."""
    import torch

    p = torch.cuda.get_device_properties(torch.cuda.current_device())
    props = {
        "major": p.major,
        "minor": p.minor,
        "multi_processor_count": p.multi_processor_count,
        "shared_memory_per_block_optin": p.shared_memory_per_block_optin,
        "shared_memory_per_multiprocessor": p.shared_memory_per_multiprocessor,
        "name": p.name,
    }
    try:  # the radix top-k cluster kernels clamp their cluster size to this
        from .kernels.blackwell.top_k import single_pass_multi_cta_radix_topk_cluster as c

        props["max_cluster_size"] = int(c._query_max_cluster_size())
    except Exception:  # noqa: BLE001 - CuTe DSL missing, or the query failed
        pass
    return props


def _shim_cute_dsl_device_attributes(props: Dict[str, Any]) -> None:
    """Answer CuTe DSL's own driver queries made while compiling (the shared
    memory per SM, for kernels with a minimum blocks-per-SM hint) from
    ``props``; any other attribute raises as it would without a GPU."""
    try:
        from cutlass.base_dsl.runtime import cuda as cuda_helpers
    except ImportError:
        return
    attrs = cuda_helpers.cuda.CUdevice_attribute
    known = {
        attrs.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: props["major"],
        attrs.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: props["minor"],
        attrs.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: props["multi_processor_count"],
    }
    if "shared_memory_per_multiprocessor" in props:
        known[attrs.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR] = props[
            "shared_memory_per_multiprocessor"
        ]
    orig = cuda_helpers.get_device_attribute

    def get_device_attribute(attribute, device_id: int = 0):
        if attribute in known:
            return known[attribute]
        return orig(attribute, device_id)

    cuda_helpers.get_device_attribute = get_device_attribute


def install_device_shim(props: Dict[str, Any]) -> None:
    """In a helper process: answer the device queries of kernel constructors
    (``torch.cuda.get_device_properties`` and friends, the radix top-k max
    cluster size, the top-k runners' SM count) with ``props``."""
    import torch

    ns = types.SimpleNamespace(**props)
    torch.cuda.get_device_properties = lambda device=None: ns
    torch.cuda.current_device = lambda: 0
    torch.cuda.get_device_capability = lambda device=None: (props["major"], props["minor"])
    _shim_cute_dsl_device_attributes(props)
    if "max_cluster_size" in props:
        try:
            from .custom_ops import cute_dsl_custom_ops as ops
            from .kernels.blackwell.top_k import (
                single_pass_multi_cta_radix_topk_cluster as cluster_mod,
            )
        except ImportError:
            return
        size = int(props["max_cluster_size"])
        cluster_mod._query_max_cluster_size = lambda: size
        if hasattr(ops, "_query_max_cluster_size"):
            ops._query_max_cluster_size = lambda: size


class KdaPrefillProvider:
    """K123 and K4 variants a Kimi K3 KDA prefill batch needs.

    The heads / bias / gate / beta settings come from the model's KDA layers
    (all layers share them); a batch selects ``varlen_pure`` from its context
    chunk lengths. Every variant is also queued once in the background, so a
    batch shape seen for the first time finds it compiled.
    """

    def __init__(self, model, num_sm: Optional[int] = None):
        from .modules.kimi_kda.kimi_kda_mixer import KimiKDALinearAttention

        cfgs = set()
        heads = set()
        for mod in model.modules():
            if not isinstance(mod, KimiKDALinearAttention):
                continue
            H = getattr(mod, "num_heads", None)
            if H is None:
                continue
            has_bias = getattr(mod, "dt_bias", None) is not None
            # The mixer passes safe_gate = (gate_lower_bound is not None).
            safe_gate = getattr(mod, "gate_lower_bound", None) is not None
            cfgs.add((int(H), bool(has_bias), bool(safe_gate)))
            heads.add((int(H), int(mod.head_k_dim), int(mod.head_dim)))
        self.cfgs = sorted(cfgs)
        self.k4_specs: List[str] = []
        if heads and num_sm is None:
            try:
                from .custom_ops import cute_dsl_kimi_k3_custom_ops as ops
            except ImportError:
                heads = set()  # the KDA prefill op is unavailable: nothing to prefetch
            else:
                # The persistent-grid ceiling the op compiles K4 for.
                num_sm = inspect.signature(ops._launch_k4_persistent).parameters["num_sm"].default
        # The pool's dynamic strides are 32-bit whenever its rows fit in int32
        # offsets, as a staged pool of max_batch_size rows does.
        self.k4_specs = [k4_spec(num_sm, H, K, V, True) for H, K, V in sorted(heads)]

    def __bool__(self):
        return bool(self.cfgs)

    def specs(self, varlen_pure: bool) -> List[str]:
        # The production mixer: in-kernel beta sigmoid, fp32 beta (b_proj
        # output .float()), int64 cu_seqlens / chunk_indices.
        return [
            k123_spec(H, b, sg, varlen_pure, "int64", "int64", True, "float32")
            for H, b, sg in self.cfgs
        ]

    def enumerate_specs(self) -> List[str]:
        return self.specs(True) + self.specs(False) + self.k4_specs


class Fp8GemmProvider:
    """Every TVM-FFI FP8 blockwise GEMM tactic the model's layers reach.

    A block-FP8 ``Linear`` that opts out of DeepGEMM (``disable_deep_gemm`` or
    ``use_cute_dsl_blockscaling_mm``) runs ``trtllm::cute_dsl_fp8_gemm_blackwell``
    on SM100/SM103, and the autotuner's first pass benchmarks every tactic the
    runner lists (30 on these GPUs: the 1-CTA 64/128-row and 2-CTA 128/256-row
    tiles over every cluster shape up to 16 CTAs), compiling each at its first
    benchmark on the GPU's thread. The runner's kernel is shape-generic, so the
    set follows from the device alone; every tactic valid for some layer's
    ``(n, k)`` is enumerated, with the cluster occupancy of this device, before
    the first forward.

    The grouped FP8 O-LoRA projection of DeepSeek-V4 / V4.1 attention (an FP8
    ``o_a_proj`` of shape ``(groups, o_lora_rank, group_dim)``) runs the same
    kernel as a batched GEMM (``trtllm::cute_dsl_fp8_bmm_blackwell``, one batch
    per group; the batch count is a runtime scalar too), so those layers add
    their ``(n, k) = (o_lora_rank, group_dim)``. CSA2 takes that path for
    128-block checkpoint scales (DeepSeek-V4.1-Flash's), which are known only
    once the weights load, so every FP8 ``o_a_proj`` is planned.
    """

    def __init__(self, model):
        import torch

        from .._utils import get_sm_version
        from .modules.linear import Linear

        self.shapes = []
        self.specs: List[str] = []
        if get_sm_version() not in (100, 103):
            return
        shapes = set()
        for mod in model.modules():
            o_a_proj = getattr(mod, "o_a_proj", None)
            if (
                isinstance(o_a_proj, torch.Tensor)
                and o_a_proj.dim() == 3
                and o_a_proj.dtype == torch.float8_e4m3fn
            ):
                shapes.add((int(o_a_proj.shape[1]), int(o_a_proj.shape[2])))
                continue
            if not isinstance(mod, Linear):
                continue
            if not (
                getattr(mod, "use_cute_dsl_blockscaling_mm", False)
                or getattr(mod, "disable_deep_gemm", False)
            ):
                continue
            # The quant mode, not the quant method: right after construction
            # the layers have no weights (nor a quant method) yet.
            quant_config = getattr(mod, "quant_config", None)
            mode = getattr(quant_config, "layer_quant_mode", None)
            if mode is None or not mode.has_fp8_block_scales():
                continue
            n, k = getattr(mod, "out_features", None), getattr(mod, "in_features", None)
            if not n or not k:
                continue
            shapes.add((int(n), int(k)))
        self.shapes = sorted(shapes)
        if not self.shapes:
            return
        self.specs = self._enumerate()

    def __bool__(self):
        return bool(self.specs)

    def _enumerate(self) -> List[str]:
        import cutlass

        from .custom_ops.cute_dsl_custom_ops import CuteDSLFp8BlackwellRunner

        kernel = CuteDSLFp8BlackwellRunner.kernel_class
        tactics = set()
        # get_valid_tactics' candidate grid, checked the way it checks it
        # (M only enters the 16-byte alignment test, which any M passes).
        for use_2cta in (False, True):
            for tiler in ((64, 128), (128, 128), (256, 128)):
                for cluster in ((1, 1), (1, 2), (1, 4), (2, 1), (2, 2), (2, 4),
                                (4, 1), (4, 2), (4, 4)):  # fmt: skip
                    for n, k in self.shapes:
                        if kernel.can_implement(
                            cutlass.Float8E4M3FN, cutlass.Float32, cutlass.BFloat16,
                            use_2cta, tiler, cluster, 128, n, k, 1, "k", "k", "n",
                        ):  # fmt: skip
                            tactics.add((use_2cta, tiler, cluster))
                            break
        # The cluster occupancy, computed as the runner computes it.
        hw = cutlass.utils.HardwareInfo()
        occupancy: Dict[int, int] = {}
        specs = []
        for use_2cta, tiler, cluster in sorted(tactics):
            size = cluster[0] * cluster[1]
            if size not in occupancy:
                occupancy[size] = hw.get_max_active_clusters(size)
            specs.append(fp8_gemm_spec(use_2cta, tiler, cluster, occupancy[size]))
        return specs

    def enumerate_specs(self) -> List[str]:
        return list(self.specs)


def install_fp8_gemm_loader(loader: Callable[[str], Optional[Callable]]) -> bool:
    """Route the TVM-FFI compiles of ``CuteDSLFp8BlackwellRunner`` and
    ``CuteDSLFp8BlackwellBmmRunner`` through ``loader(spec)``; returns False if
    CuTe DSL is unavailable."""
    try:
        from .custom_ops import cute_dsl_custom_ops as ops
    except ImportError:
        return False

    def _loader(use_2cta_instrs, mma_tiler_mn, cluster_shape_mn, max_active_clusters):
        return loader(
            fp8_gemm_spec(use_2cta_instrs, mma_tiler_mn, cluster_shape_mn, max_active_clusters)
        )

    ops.fp8_gemm_loader = _loader
    return True


class FlashInferNormProvider:
    """Every FlashInfer CuTe DSL RMSNorm variant the model's norm layers reach.

    ``RMSNorm`` (``modules/rms_norm.py``) calls FlashInfer's rmsnorm, or its
    fused add + RMSNorm when given a residual, for bf16/fp16 activations.
    FlashInfer keys the compiled kernel on (dtype, hidden size, weight bias,
    PDL, SM version, contiguous rows); a norm's hidden size and dtype are
    fixed, and PDL and the SM version are fixed per process, so each layer
    shape reaches at most four variants (plain or fused add, contiguous or
    strided rows). All of them are enumerated up front; a batch adds nothing.
    """

    def __init__(self, model):
        import torch

        from .flashinfer_utils import IS_FLASHINFER_AVAILABLE, get_env_enable_pdl
        from .modules.rms_norm import RMSNorm

        self.specs: List[str] = []
        if not IS_FLASHINFER_AVAILABLE or not _flashinfer_norm_uses_cute_dsl():
            return
        from flashinfer.norm.utils import get_sm_version
        from flashinfer.utils import device_support_pdl

        device = torch.device("cuda", torch.cuda.current_device())
        pdl = bool(get_env_enable_pdl()) and device_support_pdl(device)
        sm = get_sm_version(device)
        shapes = set()
        for mod in model.modules():
            if not isinstance(mod, RMSNorm) or getattr(mod, "use_cuda_tile", False):
                continue
            if (
                getattr(mod, "is_nvfp4", False)
                and getattr(mod, "nvfp4_scale", None) is not None
                and not getattr(mod, "use_gemma", False)
            ):
                continue  # the fused RMSNorm + NVFP4 quantize kernel, not FlashInfer
            weight = getattr(mod, "weight", None)
            if weight is None or weight.dtype not in (torch.bfloat16, torch.float16):
                continue
            dtype_str = {torch.bfloat16: "bfloat16", torch.float16: "float16"}[weight.dtype]
            bias = 1.0 if getattr(mod, "use_gemma", False) else 0.0
            shapes.add((dtype_str, int(weight.shape[-1]), bias))
        self.shapes = sorted(shapes)
        for dtype_str, H, bias in self.shapes:
            for op in ("rmsnorm", "fused_add_rmsnorm"):
                for contiguous in (True, False):
                    self.specs.append(fi_norm_spec(op, dtype_str, H, bias, pdl, sm, contiguous))

    def __bool__(self):
        return bool(self.specs)

    def enumerate_specs(self) -> List[str]:
        return list(self.specs)


class LowMDirectGemmProvider:
    """Every FlashInfer low-M direct GEMM variant ``modules.low_m_gemm``'s
    AutoTuner race benchmarks, while the low-M path is active
    (``TRTLLM_LOW_M_GEMM_BACKEND``).

    The race runs on every bias-free BF16 ``Linear`` the dispatcher attaches
    to, at every M the autotuner profiles (``_M_TUNING_BUCKETS`` and the M of
    each warmup batch, i.e. 1..``max_num_tokens`` within the low-M range), and
    benchmarks each direct tactic valid for the shape; each (M, N, K, tactic)
    is a separate ~1 s compile on the executor thread. They follow from the
    Linear shapes and the engine limits, so the helpers build them while the
    weights load.
    """

    def __init__(self, model, max_num_tokens: int):
        self.shapes: List[Tuple[int, int]] = []
        self.specs: List[str] = []
        try:
            from flashinfer.gemm.kernels import dense_bf16_gemm_direct as d

            from .._utils import get_sm_version, is_sm_100f
            from .modules import low_m_gemm as lm
        except ImportError:
            return
        if not lm.LOW_M_GEMM_ACTIVE or not is_sm_100f(get_sm_version()):
            return
        import dataclasses

        import torch

        from .flashinfer_utils import get_env_enable_pdl
        from .modules.linear import Linear, UnquantizedLinearMethod

        # The modules the dispatcher attaches to (``LowMGemmDispatcher.attach``)
        # whose forward reaches it: unquantized BF16 Linears off the custom
        # cuBLAS path (``UnquantizedLinearMethod.apply``), and opt-in modules.
        shapes = set()
        for mod in model.modules():
            if isinstance(mod, Linear):
                if not isinstance(getattr(mod, "quant_method", None), UnquantizedLinearMethod):
                    continue
                if getattr(mod, "use_custom_cublas_mm", False):
                    continue
            elif not getattr(mod, lm.LOW_M_GEMM_OPT_IN_ATTR, False):
                continue
            if getattr(mod, "bias", None) is not None:
                continue  # the direct kernel takes no bias, so it never races
            w = getattr(mod, "weight", None)
            if w is None or w.dim() != 2 or w.dtype != torch.bfloat16:
                continue
            n, k = int(w.shape[0]), int(w.shape[1])
            if k % 128:
                continue
            shapes.add((n, k))
        self.shapes = sorted(shapes)
        if not self.shapes:
            return
        # The race profiles every tuning bucket (the autotuner's bucket list
        # is not capped by the batch) plus the M of the shape that starts it:
        # warmup's batches, up to max_num_tokens rows within the low-M range.
        top = min(int(max_num_tokens), lm._MAX_M)
        ms = sorted(set(lm._M_TUNING_BUCKETS) | set(range(1, top + 1)))
        pdl = bool(get_env_enable_pdl())
        seen = set()
        for m in ms:
            for n, k in self.shapes:
                try:
                    tactics = [d.default_tactic(m, n, k), *d.autotune_tactics(m, n, k)]
                except ValueError:
                    continue
                for t in tactics:
                    spec = lowm_direct_spec(m, n, k, dataclasses.astuple(t), pdl)
                    if spec not in seen:
                        seen.add(spec)
                        self.specs.append(spec)

    def __bool__(self):
        return bool(self.specs)

    def enumerate_specs(self) -> List[str]:
        return list(self.specs)


def install_lowm_direct_loader(loader: Callable[[str], Optional[Callable]]) -> bool:
    """Route FlashInfer's ``_get_compiled_direct_kernel`` through
    ``loader(spec)``: a helper-exported kernel, or None to compile in place
    as before. The wrapper keeps the getter's ``functools.cache``, so a
    variant is resolved once per process. Returns False if FlashInfer's
    direct GEMM is unavailable."""
    import functools

    try:
        import torch
        from flashinfer.gemm.kernels import dense_bf16_gemm_direct as d
    except ImportError:
        return False
    orig = d._get_compiled_direct_kernel
    if getattr(orig, "_jitp_wrapped", False):
        return True
    import dataclasses

    @functools.cache
    def getter(dtype, m, n, k, tactic, use_pdl):
        if dtype == torch.bfloat16:
            fn = loader(lowm_direct_spec(m, n, k, dataclasses.astuple(tactic), use_pdl))
            if fn is not None:
                return fn
        return orig(dtype, m, n, k, tactic, use_pdl)

    getter._jitp_wrapped = True
    d._get_compiled_direct_kernel = getter
    return True


def _modules_with_attention_backends(model) -> List[Any]:
    """``model.modules()`` plus the modules owned by its attention backends.

    An attention backend is a plain object, not an ``nn.Module``, and some
    sparse backends build their indexer (and its ``TopK``) in the first
    forward, so after warmup these are reachable only through the backend's
    attributes (``attn``, ``mqa``, ``mha``, ``backend`` of the attention
    modules, then the backend's ``indexer``).
    """
    import torch

    if model is None:
        return []
    out, seen = [], set()

    def add(mod):
        for m in mod.modules():
            if id(m) not in seen:
                seen.add(id(m))
                out.append(m)

    add(model)
    for mod in list(out):
        for name in ("attn", "mqa", "mha", "backend", "attn_backend"):
            backend = mod.__dict__.get(name)
            # Only plain Python backends: a module is already walked, and an
            # extension or slotted object has no attributes to look through.
            attrs = getattr(backend, "__dict__", None)
            if attrs is None or isinstance(backend, torch.nn.Module):
                continue
            for value in list(attrs.values()):
                if isinstance(value, torch.nn.Module):
                    add(value)
    return out


# The loader the wrapped top-k ``_compile``s consult (None: compile in place).
_topk_loader: Optional[Callable[[str], Optional[Callable]]] = None


def _wrap_topk_runners() -> List[str]:
    """Wrap each radix top-k runner's ``_compile`` once: under ``_KeyCapture``
    it records its key and stops the launch; otherwise it consults
    ``_topk_loader`` before compiling. Returns the names of the runners whose
    ``_compile`` is wrapped."""
    from .custom_ops import cute_dsl_custom_ops as ops

    wrapped = []
    for name in TOPK_RUNNERS:
        runner = getattr(ops, name, None)
        if runner is None:
            continue
        # A runner that inherits _compile gets its own wrapper: the spec names
        # the runner whose kernel_cache (and kernel class) the key belongs to.
        orig = inspect.getattr_static(runner, "_compile", None)
        if not isinstance(orig, classmethod):
            continue
        if getattr(orig.__func__, "_jitp_wrapped", False) and "_compile" in runner.__dict__:
            wrapped.append(name)
            continue
        inner = getattr(orig.__func__, "_jitp_inner", orig.__func__)

        def make(inner=inner, name=name):
            def _compile(cls, *key):
                capture = _KeyCapture.active()
                if capture is not None:
                    capture.append((name, key))
                    raise _KeyCaptured()
                if key in cls.kernel_cache:
                    return
                loader = _topk_loader
                fn = None
                if loader is not None:
                    try:
                        fn = loader(topk_spec(name, key))
                    except TypeError:
                        fn = None
                if fn is not None:
                    cls.kernel_cache[key] = fn
                    return
                inner(cls, *key)

            _compile._jitp_wrapped = True
            _compile._jitp_inner = inner
            return classmethod(_compile)

        setattr(runner, "_compile", make())
        wrapped.append(name)
    return wrapped


def install_topk_loader(loader: Callable[[str], Optional[Callable]]) -> List[str]:
    """Route the radix top-k runners' ``_compile`` through ``loader``.

    ``loader(spec)`` returns a helper-exported kernel, which is stored in the
    runner's ``kernel_cache`` under the key ``_compile`` would have used, or
    None to compile as before. Returns the names of the wrapped runners.
    """
    global _topk_loader
    _topk_loader = loader
    return _wrap_topk_runners()


class _KeyCaptured(Exception):
    """Raised by a wrapped top-k ``_compile`` under ``_KeyCapture``."""


class _KeyCapture:
    """Within the context, the wrapped top-k ``_compile``s record their key
    and return without compiling (the forward that follows fails on the
    missing cache entry, which the planner expects)."""

    _local = __import__("threading").local()

    def __enter__(self) -> List[Tuple[str, tuple]]:
        self._local.keys = []
        return self._local.keys

    def __exit__(self, *exc):
        self._local.keys = None
        return False

    @classmethod
    def active(cls) -> Optional[List[Tuple[str, tuple]]]:
        return getattr(cls._local, "keys", None)


class TopKProvider:
    """Every radix top-k variant the model's ``TopK`` modules can reach.

    A ``TopK`` with a CuTe DSL radix implementation compiles one kernel per
    key of the runner the dispatch picks: the score width bucket
    (``next_pow2`` of the score columns), and in decode the row count's band
    (single CTA, radix select, or a multi-CTA cluster size), plus the
    engine-fixed top_k / next_n / dtype. The provider replays the real
    dispatch (``torch.ops.trtllm.cute_dsl_indexer_topk_decode`` /
    ``..._prefill_blackwell``) on ``meta`` scores for every width bucket up to
    the longest score row and every row band up to the token budget, with the
    runners' compiles capturing their key instead of compiling.
    """

    def __init__(self, model, max_num_tokens: int, max_seq_len: int):
        import torch

        from .modules.top_k import TopK, TopKImplementation

        self.cfgs = set()
        radix = TopKImplementation.CUTE_DSL_RADIX
        for mod in _modules_with_attention_backends(model):
            if not isinstance(mod, TopK):
                continue
            for phase, impl in (
                ("prefill", mod.prefill_implementation),
                ("decode", mod.decode_implementation),
            ):
                if impl == radix:
                    self.cfgs.add((phase, int(mod.top_k)))
        self.max_rows = max(1, int(max_num_tokens))
        self.max_cols = max(1, int(max_seq_len))
        self.dtype = torch.float32

    def __bool__(self):
        return bool(self.cfgs)

    def _widths(self) -> List[int]:
        out, w = [], 16
        while w < 2 * self.max_cols:
            out.append(w)
            w *= 2
        return out

    def _rows(self, next_n: int) -> List[int]:
        # One row count per band of the decode dispatch (cluster sizes change
        # at 4 / 8 / 32 / 64 / SM count rows) and around the SM count.
        cands = {1, 2, 4, 5, 8, 9, 16, 32, 33, 64, 65, 128, 148, 152, 153, 160, 256, 512, 4096}
        rows = sorted(r for r in cands if r <= self.max_rows)
        return [max(next_n, r - r % next_n) for r in rows]

    def enumerate_specs(self, next_ns=(1,)) -> List[str]:
        import torch

        from .custom_ops import cute_dsl_custom_ops as ops
        from .modules.top_k import _CUTE_DSL_PREFILL_COPY_BITS

        # The ops' Python bodies (CustomOpDef._init_fn): calling the ops on
        # meta tensors would dispatch to their fake implementations.
        _wrap_topk_runners()  # so their _compile records the key and stops
        decode = getattr(ops.cute_dsl_indexer_topk_decode, "_init_fn", None)
        prefill = getattr(ops.cute_dsl_indexer_topk_prefill_blackwell, "_init_fn", None)
        specs = []
        meta = torch.device("meta")
        for phase, top_k in sorted(self.cfgs):
            body = prefill if phase == "prefill" else decode
            if body is None:
                continue
            # Scores narrower than top_k (short prefixes; CSA2 also selects
            # over compressed rows) still run the kernel, keyed on their width.
            for cols in self._widths():
                if phase == "prefill":
                    batches = [(1, 1)]
                else:
                    batches = [(r, n) for n in next_ns for r in self._rows(n)]
                for rows, next_n in batches:
                    scores = torch.empty(rows, cols, dtype=self.dtype, device=meta)
                    out = torch.empty(rows, top_k, dtype=torch.int32, device=meta)
                    with _KeyCapture() as keys:
                        try:
                            if phase == "prefill":
                                bounds = torch.empty(rows, dtype=torch.int32, device=meta)
                                body(scores, bounds, bounds, out, top_k,
                                     _CUTE_DSL_PREFILL_COPY_BITS)  # fmt: skip
                            else:
                                lens = torch.empty(rows // next_n, dtype=torch.int32, device=meta)
                                body(scores, lens, out, top_k, next_n)
                        except _KeyCaptured:
                            pass
                    for runner, key in keys:
                        specs.append(topk_spec(runner, key))
        return sorted(set(specs))


class Csa2PublishProvider:
    """Every CSA2 candidate-publication variant the model's layers can reach.

    ``publish_candidates_`` runs in each candidate-source indexer (a CSA2
    layer whose ``candidate_source`` is itself) for every scores chunk. Its
    compile key is the layer's candidate block size and sparse block, the
    power-of-two block capacity ``padded`` (``next_pow2(ceil(width /
    block_size))`` for the chunk's score width, any width from 1 in eager
    prefill), and each pointer's element type and the alignment ``_ptr``
    gives it, which is fixed: 4 bytes for the fp32 scores and the int32
    outputs (their element size), 8 for the int64 visible lengths (whose
    rows start at a multiple of 8 bytes). Every capacity up to the kernel's
    limit of 4096 blocks, or the widest ``max_seq_len`` positions reach, is
    enumerated up front.
    """

    def __init__(self, model, max_seq_len: int):
        self.geometries = set()
        for mod in _modules_with_attention_backends(model):
            # A CSA2 indexer (layer_idx) or attention module (its layer).
            layout = getattr(mod, "layout", None)
            layer_of = getattr(layout, "layer", None)
            block_size = getattr(layout, "candidate_block_size", None)
            if layer_of is None or block_size is None:
                continue
            idx = getattr(mod, "layer_idx", None)
            if idx is None:
                idx = getattr(getattr(mod, "layer", None), "layer_idx", None)
            if idx is None:
                continue
            try:
                layer = layer_of(idx)
            except (KeyError, IndexError, ValueError):
                continue
            if getattr(layer, "candidate_source", None) != idx:
                continue
            # The indexer (built in the first forward) holds sparse_block;
            # it follows from the block size the same way.
            sparse_block = getattr(mod, "sparse_block", None)
            if sparse_block is None:
                sparse_block = next((b for b in (16, 8) if block_size % b == 0), 0)
            self.geometries.add((int(block_size), int(sparse_block)))
        self.max_seq_len = max(1, int(max_seq_len))

    def __bool__(self):
        return bool(self.geometries)

    def enumerate_specs(self) -> List[str]:
        # (scores, visible, output, sparse ids, sparse counts); without a
        # sparse block the last two are placeholders: the output again.
        ptrs = [("Float32", 4), ("Int64", 8), ("Int32", 4), ("Int32", 4), ("Int32", 4)]
        specs = []
        for block_size, sparse_block in sorted(self.geometries):
            blocks = min(4096, -(-self.max_seq_len // block_size))
            padded = 1
            while padded < 2 * blocks and padded <= 4096:
                specs.append(csa2_publish_spec(block_size, padded, sparse_block, ptrs))
                padded *= 2
        return specs


def install_csa2_loader(loader: Callable[[str], Optional[Callable]]) -> bool:
    """Route CSA2's glue-kernel compiles through ``loader(spec)``; returns
    False if CSA2 is unavailable."""
    try:
        from .attention.backends.sparse.csa2 import kernel as csa2
    except ImportError:
        return False

    def _loader(key):
        try:
            spec = csa2_spec_from_launch_key(key)
        except (TypeError, ValueError):
            return None
        return None if spec is None else loader(spec)

    csa2.csa2_loader = _loader
    return True


def _flashinfer_norm_uses_cute_dsl() -> bool:
    """Whether FlashInfer's norm dispatch takes its CuTe DSL kernels here."""
    try:
        import torch
        from flashinfer import norm as fi_norm
    except ImportError:
        return False
    use_cuda = getattr(fi_norm, "_use_cuda_norm", None)
    if use_cuda is None:
        return False
    return not use_cuda(torch.device("cuda", torch.cuda.current_device()))


def install_flashinfer_norm_loader(
    loader: Callable[[str], Optional[Callable]],
) -> List[str]:
    """Route FlashInfer's compiled-norm getters through ``loader``.

    ``loader(spec)`` returns a helper-exported kernel, or None. On None the
    original getter runs, called the way FlashInfer calls it, so a variant
    warmup already compiled comes from FlashInfer's own cache and any other
    compiles in place as before. Each wrapped getter caches its result
    (``functools.cache``), so a variant is resolved once per process. Returns
    the names of the getters that were wrapped.
    """
    import functools

    wrapped = []
    targets: List[Tuple[Any, str, str]] = []
    try:
        from flashinfer.norm.kernels import fused_add_rmsnorm as fa_mod
        from flashinfer.norm.kernels import rmsnorm as rms_mod
    except ImportError:
        return wrapped
    targets.append((rms_mod, "_get_compiled_rmsnorm_kernel", "rmsnorm"))
    targets.append((fa_mod, "_get_compiled_fused_add_rmsnorm_kernel", "fused_add_rmsnorm"))
    for module, name, op in targets:
        orig = getattr(module, name, None)
        if orig is None or getattr(orig, "_jitp_wrapped", False):
            continue

        def make(orig=orig, op=op):
            @functools.cache
            def getter(dtype_str, H, weight_bias, enable_pdl, sm_version, contiguous=True):
                spec = fi_norm_spec(
                    op, dtype_str, H, weight_bias, enable_pdl, sm_version, contiguous
                )
                fn = loader(spec)
                if fn is not None:
                    return fn
                # FlashInfer passes ``contiguous`` by keyword; the same call
                # form finds the entries its cache already holds.
                return orig(
                    dtype_str, H, weight_bias, enable_pdl, sm_version, contiguous=contiguous
                )

            getter._jitp_wrapped = True
            getter._jitp_original = orig
            return getter

        setattr(module, name, make())
        wrapped.append(name)
    return wrapped
