# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 TP4 absorbed-MLA K-B / V-B batched BF16 GEMMs at PREFILL row counts.

Owner adapters for the two per-head BMMs of the absorbed-MLA context step,
built on the decode line's batched persistent-GEMM machinery
(``self_attn_k_b_bmm_h16_m64_n512_k192_bf16.py``: swapped operands, TMA store,
serving strides kept, one compiled object per schedule group with dynamic
layouts) -- no device code is copied, only a prefill ABI and a large-N
schedule are added.  Serving views (T = packed context tokens of the step)::

    k_b:  q_nope^T [16,T,192] (stride 256, 4096, 1)  x  k_b_proj_trans [16,512,192]
          -> fused_q[..., :512]^T [16,T,512] (stride 576, 9216, 1)
    v_b:  attn_out_latent^T [16,T,512] (stride 512, 8192, 1)  x  v_b_proj [16,256,512]
          -> attn_output^T [16,T,256] (stride 256, 4096, 1)

Runtime-M: ``m_rows_allowed`` = every ``1 <= T <= MAX_ROWS``; the compiled
object serves any row count (the GEMM's N in the swapped orientation; the TMA
store clips the N edge, so partial token tiles are safe).  Kernel symbols carry
``glm52_self_attn_prefill_{k_b,v_b}_bmm``.  Contract: FP32 accumulation, BF16
output.

Launch path (host cost).  The default launcher compiles the same persistent
GEMM as a TVM-FFI function (``--enable-tvm-ffi``, environment stream) through
``_bmm_serving_views``, a ``@cute.jit`` host wrapper that takes the serving
views ``(weight [16,N,K], input [16,T,K], output [16,T,N])`` as they are and
permutes the layouts to CuTe's ``(m,k,l) / (n,k,l) / (m,n,l)`` -- exactly the
layouts the decode machinery's ``bmm(weight, input^T, output^T)`` builds, so
the device kernel is unchanged; only the per-call Python (DLPack tensor
construction, two transposes, stream wrapping) is gone.  The legacy path
(decode ``make_build``; several times the host cost of ``torch.bmm`` per call
at prefill row counts) stays available as ``build(op, launch_mode="legacy")``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch

from . import self_attn_k_b_bmm_h16_m64_n512_k192_bf16 as _kbbmm

BatchedGemmAbi = _kbbmm.BatchedGemmAbi
GemmSchedule = _kbbmm.GemmSchedule

import cuda.bindings.driver as cuda  # noqa: E402
import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from cutlass.cute.runtime import from_dlpack, make_fake_stream  # noqa: E402

LaunchMode = Literal["tvm_ffi", "legacy"]
DEFAULT_LAUNCH_MODE: LaunchMode = "tvm_ffi"
_ALIGNMENT_BYTES = _kbbmm._ALIGNMENT_BYTES

NUM_HEADS = 16
MAX_ROWS = 8_192
# Swapped orientation: GEMM M = output features (512 / 256), N = token rows, K = 192 / 512.
# Schedule groups (measured against cuBLAS bmm):
#   T <= 2048: 2CTA (256,128) c(2,1) raster n
#   T >  2048: 2CTA (256,256) c(2,1) raster n
# TMA store (N-edge clipping) in both.
SCHEDULE_SMALL_T = GemmSchedule(True, (256, 128), (2, 1), raster_along="n")
SCHEDULE_LARGE_T = GemmSchedule(True, (256, 256), (2, 1), raster_along="n")
SCHEDULE_SPLIT_ROWS = 2_048
ROWS_SMALL = tuple(range(1, SCHEDULE_SPLIT_ROWS + 1))
ROWS_LARGE = tuple(range(SCHEDULE_SPLIT_ROWS + 1, MAX_ROWS + 1))


@dataclass(frozen=True)
class PrefillBmmSpec:
    op: str
    input_features: int  # GEMM K
    output_features: int  # GEMM M
    input_row_width: int  # storage width of one token row the input view is cut from
    output_row_width: int
    kernel_name_prefix: str

    @property
    def input_stride(self) -> tuple[int, int, int]:
        return (self.input_features_head_stride, self.input_row_width, 1)

    @property
    def input_features_head_stride(self) -> int:
        return self.input_row_width // NUM_HEADS

    @property
    def output_stride(self) -> tuple[int, int, int]:
        return (self.output_row_width // NUM_HEADS, self.output_row_width, 1)


SPECS: dict[str, PrefillBmmSpec] = {
    # q [T,16,256] -> q_nope = [..., :192] -> transpose(0,1): stride (256, 4096, 1)
    "k_b": PrefillBmmSpec("k_b", 192, 512, 4_096, 9_216, "glm52_self_attn_prefill_k_b_bmm"),
    # attn_out_latent [T,16,512] contiguous -> transpose(0,1): stride (512, 8192, 1);
    # output attn_output [T,16,256] -> transpose(0,1): stride (256, 4096, 1)
    "v_b": PrefillBmmSpec("v_b", 512, 256, 8_192, 4_096, "glm52_self_attn_prefill_v_b_bmm"),
}


def make_abi(op: str, rows: int) -> BatchedGemmAbi:
    spec = SPECS[op]
    return BatchedGemmAbi(
        module_name=f"self_attn.prefill.{op}_bmm",
        num_heads=NUM_HEADS,
        active_m=int(rows),
        output_features=spec.output_features,
        input_features=spec.input_features,
        kernel_name_prefix=spec.kernel_name_prefix,
        input_stride=spec.input_stride,
        output_stride=spec.output_stride,
    )


def _build_legacy(op: str) -> Callable:
    """Decode-machinery launcher (per-call cute tensors + explicit stream)."""
    small = _kbbmm.make_build(
        make_abi(op, SCHEDULE_SPLIT_ROWS), SCHEDULE_SMALL_T, m_rows_allowed=ROWS_SMALL
    )()
    large = _kbbmm.make_build(make_abi(op, MAX_ROWS), SCHEDULE_LARGE_T, m_rows_allowed=ROWS_LARGE)()

    def launcher(input_tensor: torch.Tensor, weight: torch.Tensor, output: torch.Tensor):
        rows = int(input_tensor.shape[1])
        if not 1 <= rows <= MAX_ROWS:
            raise ValueError(f"{op}_bmm: 1 <= T <= {MAX_ROWS} required, got {rows}")
        return (small if rows <= SCHEDULE_SPLIT_ROWS else large)(input_tensor, weight, output)

    return launcher


# ----------------------------------------------------------------------------- TVM-FFI launch path
@cute.jit
def _bmm_serving_views(
    gemm_op: cutlass.Constexpr,
    weight: cute.Tensor,  # (l, m, k)   k_b_proj_trans / v_b_proj, contiguous
    input_view: cute.Tensor,  # (l, n, k)   token-major serving view, stride-1 k
    output_view: cute.Tensor,  # (l, n, m)   token-major serving view, stride-1 m
    max_active_clusters: cutlass.Constexpr,
    stream: cuda.CUstream,
    epilogue_op: cutlass.Constexpr = lambda x: x,
):
    """``dense_gemm_persistent.bmm`` on the untransposed serving views.

    ``bmm(weight, input^T, output^T)`` selects modes ``[1,2,0] / [2,1,0] / [1,2,0]``
    of the transposed views; selecting ``[1,2,0] / [1,2,0] / [2,1,0]`` of the
    untransposed views yields the identical ``(m,k,l) / (n,k,l) / (m,n,l)``
    layouts (same iterator, same shape and stride tuples, same static stride-1
    mode), so ``gemm_op`` compiles to the same kernel.
    """
    a = cute.make_tensor(weight.iterator, cute.select(weight.layout, mode=[1, 2, 0]))
    b = cute.make_tensor(input_view.iterator, cute.select(input_view.layout, mode=[1, 2, 0]))
    c = cute.make_tensor(output_view.iterator, cute.select(output_view.layout, mode=[2, 1, 0]))
    gemm_op(a, b, c, max_active_clusters, stream, epilogue_op)


def _compile_tvm_ffi(
    abi: BatchedGemmAbi,
    schedule: GemmSchedule,
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    rows_allowed: tuple[int, ...],
    *,
    can_implement_rows: int,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], object]:
    """Compile one schedule group as a TVM-FFI function over the serving views.

    Same kernel class, ``can_implement`` gate, name prefix and dynamic layouts
    as ``self_attn_k_b_bmm_h16_m64_n512_k192_bf16._compile`` (pdl_role None,
    BF16 output); differs only in the host entry (``_bmm_serving_views``) and
    the export mode.
    """
    _kbbmm.reject_unpredicated_store(abi.module_name, schedule, rows_allowed, emit_fp8_q_base=False)
    _kbbmm._validate_contract(abi, input_tensor, weight, output)
    weight = weight.detach()
    a = from_dlpack(
        weight, assumed_align=_ALIGNMENT_BYTES, enable_tvm_ffi=True
    ).mark_layout_dynamic(leading_dim=2)
    b = from_dlpack(
        input_tensor, assumed_align=_ALIGNMENT_BYTES, enable_tvm_ffi=True
    ).mark_layout_dynamic(leading_dim=2)
    c = from_dlpack(
        output, assumed_align=_ALIGNMENT_BYTES, enable_tvm_ffi=True
    ).mark_layout_dynamic(leading_dim=2)
    mnkl = (
        abi.output_features,
        int(can_implement_rows),
        abi.input_features,
        abi.num_heads,
    )
    cluster_size = schedule.cluster_shape_mn[0] * schedule.cluster_shape_mn[1]
    max_active_clusters = _kbbmm._DENSE_GEMM.utils.HardwareInfo().get_max_active_clusters(
        cluster_size
    )
    gemm = _kbbmm._GlmPersistentDenseGemmKernel(
        cutlass.Float32,
        schedule.use_2cta_instrs,
        schedule.mma_tiler_mn,
        schedule.cluster_shape_mn,
        schedule.use_tma_store,
        schedule.swizzle_size,
        schedule.raster_along,
    )
    if not gemm.can_implement(mnkl, a.element_type, b.element_type, c.element_type, "k", "k", "m"):
        raise ValueError(f"{abi.module_name}: persistent schedule cannot implement {mnkl}")
    gemm.kernel.set_name_prefix(abi.kernel_name_prefix, remove_cutlass_symbol=True)
    return cute.compile(
        _bmm_serving_views,
        gemm,
        a,
        b,
        c,
        max_active_clusters,
        make_fake_stream(use_tvm_ffi_env_stream=True),
        lambda x: x,
        options="--enable-tvm-ffi",
    )


class TvmFfiBmmLauncher:
    """``launcher(input, weight, output)`` on the serving views; one TVM-FFI instance per schedule group.

    Instances compile lazily on the first call of their group (the real tensors
    supply the strides, as the decode ``make_build`` does), then every call is
    attribute checks + one TVM-FFI call on torch's current stream.
    """

    def __init__(self, op: str) -> None:
        self.op = op
        self.spec = SPECS[op]
        self._compiled: dict[int, Callable] = {}
        self._weight_operand: dict[int, torch.Tensor] = {}
        # device of the first call; the launchers are compiled for it
        self._device: torch.device | None = None
        self._input_shape_tail = (self.spec.input_features,)
        self._output_shape_tail = (self.spec.output_features,)
        self._weight_shape = (
            NUM_HEADS,
            self.spec.output_features,
            self.spec.input_features,
        )

    def _compile_group(
        self,
        key: int,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        output: torch.Tensor,
    ) -> Callable:
        if key == SCHEDULE_SPLIT_ROWS:
            schedule, rows_allowed = SCHEDULE_SMALL_T, ROWS_SMALL
        else:
            schedule, rows_allowed = SCHEDULE_LARGE_T, ROWS_LARGE
        cutlass.cuda.initialize_cuda_context(_kbbmm._device_index(input_tensor))
        # can_implement is checked at the group's largest row count; the contract
        # (shapes, dtypes, strides, alignment) at the first call's real tensors
        compiled = _compile_tvm_ffi(
            make_abi(self.op, int(input_tensor.shape[1])),
            schedule,
            input_tensor,
            weight,
            output,
            rows_allowed,
            can_implement_rows=key,
        )
        self._compiled[key] = compiled
        return compiled

    def _weight_view(self, weight: torch.Tensor) -> torch.Tensor:
        key = weight.data_ptr()
        view = self._weight_operand.get(key)
        if view is None:
            _kbbmm._validate_tensor(weight, "weight", self._weight_shape, require_contiguous=True)
            view = weight.detach()
            self._weight_operand[key] = view
        return view

    def __call__(
        self, input_tensor: torch.Tensor, weight: torch.Tensor, output: torch.Tensor
    ) -> object:
        shape = input_tensor.shape
        if len(shape) != 3 or shape[0] != NUM_HEADS or shape[2:] != self._input_shape_tail:
            raise ValueError(
                f"{self.op}_bmm: input must be [{NUM_HEADS},T,{self.spec.input_features}], got {tuple(shape)}"
            )
        rows = shape[1]
        if not 1 <= rows <= MAX_ROWS:
            raise ValueError(f"{self.op}_bmm: 1 <= T <= {MAX_ROWS} required, got {rows}")
        if output.shape != (NUM_HEADS, rows, self.spec.output_features):
            raise ValueError(
                f"{self.op}_bmm: output must be "
                f"[{NUM_HEADS},{rows},{self.spec.output_features}], got {tuple(output.shape)}"
            )
        if (
            input_tensor.stride() != self.spec.input_stride
            or output.stride() != self.spec.output_stride
        ):
            raise ValueError(
                f"{self.op}_bmm: serving strides {self.spec.input_stride} / {self.spec.output_stride} required, "
                f"got {tuple(input_tensor.stride())} / {tuple(output.stride())}"
            )
        if input_tensor.dtype != torch.bfloat16 or output.dtype != torch.bfloat16:
            raise ValueError(f"{self.op}_bmm: input/output must be BF16")
        # One launcher per process, compiled for the device of the first call: reject CPU or
        # cross-device operands as a contract error instead of a driver error inside the call.
        dev = input_tensor.device
        if dev.type != "cuda" or weight.device != dev or output.device != dev:
            raise ValueError(
                f"{self.op}_bmm: input/weight/output must be on one CUDA device, got "
                f"{input_tensor.device}/{weight.device}/{output.device}"
            )
        if self._device is None:
            self._device = dev
        elif dev != self._device:
            raise ValueError(
                f"{self.op}_bmm: launcher compiled for {self._device}, called with {dev}"
            )
        key = SCHEDULE_SPLIT_ROWS if rows <= SCHEDULE_SPLIT_ROWS else MAX_ROWS
        compiled = self._compiled.get(key)
        if compiled is None:
            compiled = self._compile_group(key, input_tensor, weight, output)
        weight_view = self._weight_view(weight)
        if input_tensor.requires_grad:
            input_tensor = input_tensor.detach()
        if output.requires_grad:
            output = output.detach()
        return compiled(weight_view, input_tensor, output)


def build(op: str, launch_mode: LaunchMode = DEFAULT_LAUNCH_MODE):
    """Registry entry: ``build(op) -> launcher(input, weight, output)`` serving every T <= MAX_ROWS."""
    if op not in SPECS:
        raise ValueError(f"unknown prefill BMM {op!r}; known: {tuple(SPECS)}")
    if launch_mode == "tvm_ffi":
        return TvmFfiBmmLauncher(op)
    if launch_mode == "legacy":
        return _build_legacy(op)
    raise ValueError(f"{op}_bmm: launch_mode must be tvm_ffi or legacy, got {launch_mode!r}")
