# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 TP4 attention projections at PREFILL (context-step) row counts.

Owner adapters for the three dense BF16 projections of the absorbed-MLA
context step, on the shared Blackwell persistent dense GEMM
(``dense_gemm_persistent.py`` through the GLM
helper ``_swapped_dense_gemm_persistent.py``; no device code is copied)::

    kv_a_proj_with_mqa   [T,6144] x [2624,6144]^T -> [T,2624]   (q_a 2048 | kv 512 | k_pe 64)
    q_b_proj             [T,2048] x [4096,2048]^T -> [T,4096]
    o_proj               [T,4096] x [6144,4096]^T -> [T,6144]

``T`` is the number of packed context tokens of the step (TensorRT-LLM packs up
to ``max_num_tokens`` = 8192; the ISL-1024 B32 wave gives ~7.5k).  The decode
adapters of the same ops are M<=64 cells; these are the prefill counterparts.

Runtime-M: the persistent GEMM is compiled with dynamic layouts, so ONE compiled
instance per op serves every ``1 <= T <= MAX_ROWS`` (the same kernel family is
bit-identical at every M on the decode line's per-M cells).  The schedule is
chosen per M range at call time from a large-M schedule sweep:
``(256,176) c(2,1) direct`` for T <= 2048
and ``(256,256) c(2,1) direct`` above; each op therefore holds two compiled
instances.  Kernel symbols carry the owner prefix
``glm52_self_attn_prefill_<op>`` so a context-step census can count them.

Contract: FP32 accumulation, BF16 output (same as cuBLAS nvjet); not bit-exact
vs cuBLAS by construction.

Launch path (host cost).  The GEMM is compiled with ``--enable-tvm-ffi`` and
an environment stream, so ``__call__`` hands the torch tensors straight to the
compiled function: no per-call DLPack -> ``cute.Tensor`` construction and no
Python stream wrapping (the TVM-FFI bridge reads torch's current stream through
the DLPack exchange API, CUDA-graph capture included).  The legacy path
(``_make_runtime_arguments`` + ``compiled_bmm(a, b, c, stream)``) costs several
times the host time of ``torch.matmul`` per call at prefill row counts and made
the routed context step host-bound; the TVM-FFI path is the fix.  The device
kernel is the same object either way (same ``bmm`` jit entry, same dynamic
layouts).  ``launch_mode="legacy"`` keeps the old path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch

from . import _swapped_dense_gemm_persistent as _swappeddensegemmpersistent

GemmSchedule = _swappeddensegemmpersistent.GemmSchedule
compile_persistent_gemm = _swappeddensegemmpersistent.compile_persistent_gemm
_make_runtime_arguments = _swappeddensegemmpersistent._make_runtime_arguments
_problem_shape_and_majors = _swappeddensegemmpersistent._problem_shape_and_majors
_DENSE_GEMM = _swappeddensegemmpersistent._DENSE_GEMM
_GlmPersistentDenseGemmKernel = _swappeddensegemmpersistent._GlmPersistentDenseGemmKernel

import cuda.bindings.driver as cuda  # noqa: E402
import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from cutlass import testing  # noqa: E402
from cutlass.cute.runtime import from_dlpack, make_fake_stream  # noqa: E402

LaunchMode = Literal["tvm_ffi", "legacy"]
DEFAULT_LAUNCH_MODE: LaunchMode = "tvm_ffi"
# DLPack alignment assumption of every operand (torch CUDA allocations are 512 B aligned).
_ALIGNMENT_BYTES = 16


@dataclass(frozen=True)
class PrefillProjectionAbi:
    """One projection contract: ``[T,in] x [out,in]^T -> [T,out]``.

    Operands are BF16; the output is ``out_dtype`` (BF16 for the attention
    projections, FP32 for the indexer's fused wk + weights projection whose
    consumers read FP32 ``weights`` and cast ``indexer_k`` themselves).
    """

    name: str
    out_features: int
    in_features: int
    out_dtype: torch.dtype = torch.bfloat16
    # Row-range schedule table ``((max_rows, schedule), ...)`` ascending; ``None`` = the attention
    # projections' default (``SCHEDULE_SMALL`` to ``SCHEDULE_SPLIT_ROWS``, ``SCHEDULE_LARGE`` above).
    schedules: tuple[tuple[int, "GemmSchedule"], ...] | None = None

    @property
    def kernel_name_prefix(self) -> str:
        return f"glm52_self_attn_prefill_{self.name}"

    def schedule_table(self) -> tuple[tuple[int, "GemmSchedule"], ...]:
        if self.schedules is not None:
            return self.schedules
        return ((SCHEDULE_SPLIT_ROWS, SCHEDULE_SMALL), (MAX_ROWS, SCHEDULE_LARGE))


MAX_ROWS = 8_192
# Large-M sweep winners.  (256,256) c(2,1) direct measured best or within noise for all
# three ops at T=7680, so it is the single large-T schedule.
SCHEDULE_SMALL = GemmSchedule(True, (256, 176), (2, 1), orientation="direct")
SCHEDULE_LARGE = GemmSchedule(True, (256, 256), (2, 1), orientation="direct")
SCHEDULE_SPLIT_ROWS = 2_048


# Indexer-chain schedules (measured for T from 256 to 7500): N=160 is best served by a
# 128x80 single-CTA tile at every T; N=4096 wants smaller tiles below 2k rows.
INDEXER_WQ_B_SCHEDULES = (
    (512, GemmSchedule(False, (128, 80), (1, 1), orientation="direct")),
    (1_024, GemmSchedule(True, (256, 128), (2, 1), orientation="direct")),
    (MAX_ROWS, GemmSchedule(True, (256, 160), (2, 1), orientation="direct")),
)
INDEXER_WK_WEIGHTS_SCHEDULES = (
    (MAX_ROWS, GemmSchedule(False, (128, 80), (1, 1), orientation="direct")),
)


ABIS: dict[str, PrefillProjectionAbi] = {
    "kv_a_proj_with_mqa": PrefillProjectionAbi("kv_a_proj_with_mqa", 2_624, 6_144),
    "q_b_proj": PrefillProjectionAbi("q_b_proj", 4_096, 2_048),
    "o_proj": PrefillProjectionAbi("o_proj", 6_144, 4_096),
    # Indexer Q side on context / mixed steps : same TP4 shape as q_b_proj,
    # its own owner symbol so a context-step census can tell the two apart.
    "indexer_wq_b": PrefillProjectionAbi(
        "indexer_wq_b", 4_096, 2_048, schedules=INDEXER_WQ_B_SCHEDULES
    ),
    # Indexer pre-projection: wk [128,6144] + weights_proj [32,6144]
    # fused, FP32 output like the TRT-LLM indexer's ``fused_out`` (k = cols 0-127 -> BF16 by the caller,
    # weights = cols 128-159 FP32).  BF16 operands: the fused FP32 weight must be exactly
    # BF16-representable (``bf16_operand_weight``); products are exact in FP32, only the
    # accumulation grouping differs from cuBLAS TF32 (as in the decode module).
    "indexer_wk_weights": PrefillProjectionAbi(
        "indexer_wk_weights",
        160,
        6_144,
        out_dtype=torch.float32,
        schedules=INDEXER_WK_WEIGHTS_SCHEDULES,
    ),
    # The three remaining cuBLAS GEMMs of the GLM-5.2 tep4 context step after the
    # attention and MoE ports: the dense layers' MLP (gate|up fused, TP4 slice of the
    # 12288 intermediate -> 2 x 3072; down 3072 -> 6144) and the MTP module's eh
    # projection over the [enorm(e) | hnorm(h)] concat (12288 -> 6144). Same dense
    # BF16 persistent GEMM, default row-range schedules; each is one launch per layer
    # per step, so the point is the launch count and the lazy cuBLAS module loads
    # (cuBLAS loads a new kernel the first time it sees a new M), not GPU time.
    "dense_gate_up": PrefillProjectionAbi("dense_gate_up", 6_144, 6_144),
    "dense_down": PrefillProjectionAbi("dense_down", 6_144, 3_072),
    "mtp_eh_proj": PrefillProjectionAbi("mtp_eh_proj", 6_144, 12_288),
    # The MTP module's MoE at context row counts, which the trunk's prefill frontends
    # do not cover: its shared expert (gate|up 2 x 512 TP4 slice; down 512 -> 6144) and
    # its router logits (256 experts, FP32 out like the decode cell). The same shapes
    # as the trunk's shared expert and router decode cells.
    "shared_gate_up": PrefillProjectionAbi("shared_gate_up", 1_024, 6_144),
    "shared_down": PrefillProjectionAbi("shared_down", 6_144, 512),
    "router": PrefillProjectionAbi("router", 256, 6_144, out_dtype=torch.float32),
}


def _validate_weight(abi: PrefillProjectionAbi, weight: torch.Tensor) -> None:
    if (
        weight.shape != (abi.out_features, abi.in_features)
        or weight.dtype != torch.bfloat16
        or not weight.is_cuda
        or not weight.is_contiguous()
    ):
        raise ValueError(
            f"{abi.name}: weight must be contiguous CUDA BF16 "
            f"[{abi.out_features},{abi.in_features}], got shape={tuple(weight.shape)} "
            f"dtype={weight.dtype} contiguous={weight.is_contiguous()}"
        )


# ----------------------------------------------------------------------------- TVM-FFI launch path
def _bmm_operand_views(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    orientation: Literal["direct", "swapped"],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The three ``(l, m, k) / (l, k, n) / (l, m, n)`` torch views ``bmm`` expects.

    Same views as ``_make_runtime_arguments`` builds its cute tensors from; here
    they stay torch tensors (the TVM-FFI bridge converts them in C).
    """
    if orientation == "direct":
        return input_tensor.unsqueeze(0), weight.t().unsqueeze(0), output.unsqueeze(0)
    if orientation == "swapped":
        return (
            weight.unsqueeze(0),
            input_tensor.unsqueeze(0).transpose(1, 2),
            output.unsqueeze(0).transpose(1, 2),
        )
    raise ValueError(f"unsupported GEMM orientation: {orientation}")


# Stride-1 mode of each ``bmm`` operand view per orientation (``mark_layout_dynamic`` leading dim).
_LEADING_DIMS = {"direct": (2, 1, 2), "swapped": (2, 1, 1)}


def _tvm_ffi_compile_arguments(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    orientation: Literal["direct", "swapped"],
) -> tuple[cute.Tensor, cute.Tensor, cute.Tensor]:
    """Compile-time cute tensors: same dynamic layouts as the legacy path, TVM-FFI enabled."""
    views = _bmm_operand_views(input_tensor, weight.detach(), output, orientation)
    a_view, b_view, c_view = (
        from_dlpack(view, assumed_align=_ALIGNMENT_BYTES, enable_tvm_ffi=True).mark_layout_dynamic(
            leading_dim=leading_dim
        )
        for view, leading_dim in zip(views, _LEADING_DIMS[orientation], strict=True)
    )
    return a_view, b_view, c_view


def compile_persistent_gemm_tvm_ffi(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    schedule: GemmSchedule,
    *,
    kernel_name_prefix: str,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], object]:
    """Compile the persistent GEMM ``bmm`` entry as a TVM-FFI function.

    Mirrors ``_swapped_dense_gemm_persistent.compile_persistent_gemm`` (same
    kernel class, schedule, ``can_implement`` gate and dynamic layouts) but the
    returned callable takes the three ``bmm`` torch views directly and launches
    on torch's current stream (TVM-FFI environment stream).  Runtime-M: the
    tensors passed here fix only dtypes, ranks and the stride-1 modes.
    """
    active_m, input_features = input_tensor.shape
    output_features, weight_k = weight.shape
    if weight_k != input_features or output.shape != (active_m, output_features):
        raise ValueError(
            "incompatible projection shapes: "
            f"input={input_tensor.shape}, weight={weight.shape}, output={output.shape}"
        )
    mnkl, majors = _problem_shape_and_majors(input_tensor, weight, schedule.orientation)
    a, b, c = _tvm_ffi_compile_arguments(input_tensor, weight, output, schedule.orientation)
    cluster_size = schedule.cluster_shape_mn[0] * schedule.cluster_shape_mn[1]
    max_active_clusters = _DENSE_GEMM.utils.HardwareInfo().get_max_active_clusters(cluster_size)
    gemm = _GlmPersistentDenseGemmKernel(
        cutlass.Float32,
        schedule.use_2cta_instrs,
        schedule.mma_tiler_mn,
        schedule.cluster_shape_mn,
        schedule.use_tma_store,
        schedule.swizzle_size,
        schedule.raster_along,
    )
    if not gemm.can_implement(mnkl, a.element_type, b.element_type, c.element_type, *majors):
        raise testing.CantImplementError(
            f"persistent GLM projection cannot implement mnkl={mnkl}, majors={majors}, schedule={schedule}"
        )
    gemm.kernel.set_name_prefix(kernel_name_prefix, remove_cutlass_symbol=True)
    return cute.compile(
        _DENSE_GEMM.bmm,
        gemm,
        a,
        b,
        c,
        max_active_clusters,
        make_fake_stream(use_tvm_ffi_env_stream=True),
        lambda x: x,
        options="--enable-tvm-ffi",
    )


# Compiled runtime-M GEMM instances shared by every adapter of the same (ABI, schedule, device, launch
# mode): the compiled ``bmm`` fixes dtypes, ranks, stride-1 modes and the schedule only -- never the weight
# values -- so the 79 attention layers and the 22 indexer layers must not each compile their own copy.
# Without this cache every ``build(op, weight)`` compiles per layer.
_COMPILED_INSTANCES: dict[tuple[object, ...], tuple[object, str]] = {}
# number of real cute.compile calls made by this module
COMPILE_COUNT = 0


def compiled_instance_key(
    abi: "PrefillProjectionAbi",
    split_rows: int,
    schedule: "GemmSchedule",
    launch_mode: str,
    device: torch.device,
) -> tuple[object, ...]:
    return (
        abi.name,
        abi.out_features,
        abi.in_features,
        abi.out_dtype,
        int(split_rows),
        schedule,
        launch_mode,
        device.type,
        device.index,
    )


class PrefillProjection:
    """One owner projection with two runtime-M compiled instances.

    ``__call__(input, weight, output)`` is the registry launcher ABI; ``forward``
    allocates the output.  The weight passed at call time must be the tensor the
    adapter was built from (same shape/dtype; pointer may differ -- the weight
    operand view is re-derived and cached per data pointer).  Compiled instances
    are shared process-wide per ``compiled_instance_key`` (see ``_COMPILED_INSTANCES``).
    """

    def __init__(
        self,
        abi: PrefillProjectionAbi,
        weight: torch.Tensor,
        launch_mode: LaunchMode = DEFAULT_LAUNCH_MODE,
    ) -> None:
        _validate_weight(abi, weight)
        if launch_mode not in ("tvm_ffi", "legacy"):
            raise ValueError(
                f"{abi.name}: launch_mode must be tvm_ffi or legacy, got {launch_mode!r}"
            )
        self.abi = abi
        self.launch_mode = launch_mode
        self.weight_shape = tuple(weight.shape)
        self.out_features = abi.out_features
        self.in_features = abi.in_features
        self._compiled: dict[int, object] = {}
        self._orientation: dict[int, str] = {}
        # (data_ptr) -> detached ``bmm`` weight view; one entry per weight storage seen
        self._weight_operand: dict[int, torch.Tensor] = {}
        # the two launchers are compiled on the weight's device; calls on another device are refused
        self._device = weight.device
        self._table = abi.schedule_table()
        global COMPILE_COUNT
        for split_rows, schedule in self._table:
            key = compiled_instance_key(abi, split_rows, schedule, launch_mode, weight.device)
            shared = _COMPILED_INSTANCES.get(key)
            if shared is not None:
                self._compiled[split_rows], self._orientation[split_rows] = shared
                continue
            COMPILE_COUNT += 1
            compile_input = weight.new_empty((split_rows, abi.in_features))
            compile_output = weight.new_empty((split_rows, abi.out_features), dtype=abi.out_dtype)
            if launch_mode == "tvm_ffi":
                compiled = compile_persistent_gemm_tvm_ffi(
                    compile_input,
                    weight,
                    compile_output,
                    schedule,
                    kernel_name_prefix=abi.kernel_name_prefix,
                )
            else:
                # keep only the dynamic-layout compiled bmm; the helper's exact-shape
                # guard is replaced by the runtime-M guard below
                compiled = compile_persistent_gemm(
                    compile_input,
                    weight,
                    compile_output,
                    schedule,
                    kernel_name_prefix=abi.kernel_name_prefix,
                ).compiled_bmm
            self._compiled[split_rows] = compiled
            self._orientation[split_rows] = schedule.orientation
            _COMPILED_INSTANCES[key] = (compiled, schedule.orientation)

    def _weight_view(self, weight: torch.Tensor, orientation: str) -> torch.Tensor:
        """Cached detached weight operand view (TRT-LLM weights carry requires_grad)."""
        key = weight.data_ptr()
        view = self._weight_operand.get(key)
        if view is None:
            if tuple(weight.shape) != self.weight_shape or weight.dtype != torch.bfloat16:
                raise ValueError(f"{self.abi.name}: weight does not match the built ABI")
            if not weight.is_contiguous():
                raise ValueError(f"{self.abi.name}: weight must be contiguous")
            detached = weight.detach()
            view = detached.t().unsqueeze(0) if orientation == "direct" else detached.unsqueeze(0)
            self._weight_operand[key] = view
        return view

    def __call__(
        self, input_tensor: torch.Tensor, weight: torch.Tensor, output: torch.Tensor
    ) -> None:
        # Hot path (5 launches per layer x 79 layers in serving): plain attribute
        # checks only, no torch ops besides the two unsqueeze views.
        shape = input_tensor.shape
        if len(shape) != 2 or shape[1] != self.in_features:
            raise ValueError(
                f"{self.abi.name}: input must be [T,{self.in_features}], got {tuple(shape)}"
            )
        rows = shape[0]
        if not 1 <= rows <= MAX_ROWS:
            raise ValueError(f"{self.abi.name}: 1 <= T <= {MAX_ROWS} required, got {rows}")
        if (
            input_tensor.dtype != torch.bfloat16
            or output.dtype != self.abi.out_dtype
            or not input_tensor.is_contiguous()
            or not output.is_contiguous()
        ):
            raise ValueError(
                f"{self.abi.name}: input must be contiguous CUDA BF16 and output "
                f"contiguous CUDA {self.abi.out_dtype}"
            )
        if output.shape != (rows, self.out_features):
            raise ValueError(f"{self.abi.name}: output must be contiguous [T,{self.out_features}]")
        # The launchers are compiled for the build-time CUDA device: a CPU or cross-device operand must
        # fail as a contract error here, not as a driver error inside the compiled call.
        dev = input_tensor.device
        if dev.type != "cuda" or weight.device != dev or output.device != dev:
            raise ValueError(
                f"{self.abi.name}: input/weight/output must be on one CUDA device, got "
                f"{input_tensor.device}/{weight.device}/{output.device}"
            )
        if dev != self._device:
            raise ValueError(
                f"{self.abi.name}: launchers compiled for {self._device}, called on {dev}"
            )
        key = next(bound for bound, _ in self._table if rows <= bound)
        compiled = self._compiled[key]
        orientation = self._orientation[key]
        if self.launch_mode == "legacy":
            if tuple(weight.shape) != self.weight_shape or weight.dtype != torch.bfloat16:
                raise ValueError(f"{self.abi.name}: weight does not match the built ABI")
            a, b, c = _make_runtime_arguments(input_tensor, weight, output, orientation)
            stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            compiled(a, b, c, stream)
            return
        weight_view = self._weight_view(weight, orientation)
        if input_tensor.requires_grad:
            input_tensor = input_tensor.detach()
        if output.requires_grad:
            output = output.detach()
        if orientation == "direct":
            compiled(input_tensor.unsqueeze(0), weight_view, output.unsqueeze(0))
        else:
            compiled(
                weight_view,
                input_tensor.unsqueeze(0).transpose(1, 2),
                output.unsqueeze(0).transpose(1, 2),
            )

    def forward(self, input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        output = input_tensor.new_empty(
            (input_tensor.shape[0], self.abi.out_features), dtype=self.abi.out_dtype
        )
        self(input_tensor, weight, output)
        return output


def build(
    op: str, weight: torch.Tensor, launch_mode: LaunchMode = DEFAULT_LAUNCH_MODE
) -> PrefillProjection:
    """Registry entry: ``build(op, weight) -> launcher(input, weight, output)``."""
    if op not in ABIS:
        raise ValueError(f"unknown prefill projection {op!r}; known: {tuple(ABIS)}")
    return PrefillProjection(ABIS[op], weight, launch_mode)
