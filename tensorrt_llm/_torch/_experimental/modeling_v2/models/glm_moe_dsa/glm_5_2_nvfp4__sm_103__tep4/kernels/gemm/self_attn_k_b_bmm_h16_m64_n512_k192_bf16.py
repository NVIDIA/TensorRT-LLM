# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 TP4 absorbed-MLA K-B batched BF16 GEMM.

The registry ABI is ``build(active_m=M) -> launcher(input, weight, output)`` for
the strided views ``[16,M,192] x [16,512,192] -> [16,M,512]``; ``M`` is one of
:data:`ACTIVE_M_CELLS` and selects the schedule group (``(128, 8)`` for M <= 8,
``(128, 32)`` for M >= 16; :data:`SCHEDULE_GROUPS`).  One compiled object
per group serves its whole row range (dynamic layouts).  The input is the
NoPE slice of token-major Q and the output is the latent slice of fused Q, so
neither view is required to be contiguous.

``build(emit_fp8_q_base=True)`` (default off): the output is instead the
``uint8`` ``quant_q_buffer[..., :512]`` view (same element strides
``(576, 9216, 1)``) and the kernel writes the FP8 q-base bytes itself, so the
MLA RoPE generation kernel does not copy them (``emit_q_base=False``) and can
run concurrently with this GEMM.
Same GEMM schedule, PDL role and mainloop; the option changes only the C dtype
and the epilogue conversion (``epilogue_tma_store_fp8_word_packed`` in
``dense_gemm_persistent.py``).
Numerics byte-identical to the BF16 kernel followed by the RoPE generation
kernel's conversion, because both apply FP32->BF16 (RN) then BF16->E4M3 (RN,
SATFINITE). Kernel names of this variant carry ``FP8_Q_BASE_KERNEL_SUFFIX``.

The batched persistent-GEMM adapter (``BatchedGemmAbi``, ``CompiledBatchedGemm``,
``_compile``, ``make_build``) comes first in this file, the K-B owner contract
(``make_abi``, the schedule groups, ``build``) after it. TensorRT-LLM presents K-B as a strided batch
matmul ``[H, M, K] x [H, N, K] -> [H, M, N]`` whose input and output are
``transpose(0, 1)`` views of token-major storage; the adapter swaps the GEMM
operands and keeps those strides (no copy on the decode path).
"""

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack, make_fake_stream
from cutlass.utils import create_cute_tensor_for_fp8

from ..common.cuda_stream import DIET as _HOST_DIET
from . import _swapped_dense_gemm_pdl_chain as _swappeddensegemmpdlchain
from . import _swapped_dense_gemm_persistent as _swappeddensegemmpersistent

GemmSchedule = _swappeddensegemmpersistent.GemmSchedule
_DENSE_GEMM = _swappeddensegemmpersistent._DENSE_GEMM
_GlmPersistentDenseGemmKernel = _swappeddensegemmpersistent._GlmPersistentDenseGemmKernel

PersistentPdlRole = _swappeddensegemmpdlchain.PersistentPdlRole
WAIT_SIGNAL_PRELOAD_WEIGHT_ROLE = _swappeddensegemmpdlchain.WAIT_SIGNAL_PRELOAD_WEIGHT_ROLE
# decode K-B / RoPE overlap: same role plus a dependent signal right after the activation wait, so a
# late-waiting the RoPE generation kernel (``emit_q_base=False``) runs concurrently with this GEMM.
WAIT_EARLY_SIGNAL_PRELOAD_WEIGHT_ROLE = (
    _swappeddensegemmpdlchain.WAIT_EARLY_SIGNAL_PRELOAD_WEIGHT_ROLE
)
make_pdl_chain_kernel = _swappeddensegemmpdlchain.make_pdl_chain_kernel
Mxf8SideOutput = _swappeddensegemmpdlchain.Mxf8SideOutput
# Kernel-name fragment of a build with the MXF8 side output (decode act-quant fusion, V-B -> o_proj).
MXF8_SIDE_KERNEL_SUFFIX = "_x8"

_ALIGNMENT_BYTES = 16
_BF16_VECTOR_SIZE = _ALIGNMENT_BYTES // 2
# Kernel-name fragment of the FP8 q-base variant; serving validators identify
# the FP8 q-base K-B kernels by it.
FP8_Q_BASE_KERNEL_SUFFIX = "_word_packed_fp8"


@dataclass(frozen=True)
class BatchedGemmAbi:
    """One fixed GLM absorbed-MLA batched GEMM contract."""

    module_name: str
    num_heads: int
    active_m: int
    output_features: int
    input_features: int
    kernel_name_prefix: str
    input_stride: tuple[int, int, int]
    output_stride: tuple[int, int, int]

    def __post_init__(self) -> None:
        if not self.kernel_name_prefix.startswith("glm52_self_attn_"):
            raise ValueError(f"{self.module_name}: kernel prefix must identify GLM self-attention")


@dataclass(frozen=True)
class CompiledBatchedGemm:
    """Launch a compiled kernel only for the strides it captured and the rows it serves.

    The persistent GEMM is compiled with dynamic layouts, so one compiled object
    serves every row count of its schedule group (``m_rows_allowed``); the row
    count of the tensors handed in is checked against that set, every other
    shape and stride must match the captured ABI exactly (output is
    bit-identical at every M).
    """

    input_shape: torch.Size
    weight_shape: torch.Size
    output_shape: torch.Size
    input_stride: tuple[int, ...]
    weight_stride: tuple[int, ...]
    output_stride: tuple[int, ...]
    compiled_bmm: Callable
    fp8_output: bool = False
    m_rows_allowed: tuple[int, ...] = ()

    def __call__(
        self,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        output: torch.Tensor,
    ) -> object:
        """Launch on the current PyTorch stream after guarding the layout ABI."""
        rows = int(input_tensor.shape[1])
        allowed = self.m_rows_allowed or (int(self.input_shape[1]),)
        if rows not in allowed or int(output.shape[1]) != rows:
            raise ValueError(
                "compiled batched GEMM row-count mismatch: "
                f"rows={rows} (output rows {int(output.shape[1])}), this object "
                f"serves {allowed}"
            )
        expected = (
            (self.input_shape[0], rows, self.input_shape[2]),
            tuple(self.weight_shape),
            (self.output_shape[0], rows, self.output_shape[2]),
            self.input_stride,
            self.weight_stride,
            self.output_stride,
        )
        runtime = (
            tuple(input_tensor.shape),
            tuple(weight.shape),
            tuple(output.shape),
            tuple(input_tensor.stride()),
            tuple(weight.stride()),
            tuple(output.stride()),
        )
        if runtime != expected:
            raise ValueError(
                f"compiled batched GEMM layout mismatch: expected {expected}, got {runtime}"
            )
        a, b, c = _make_runtime_arguments(input_tensor, weight, output, fp8_output=self.fp8_output)
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        return self.compiled_bmm(a, b, c, stream)


def _make_runtime_arguments(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    *,
    fp8_output: bool = False,
) -> tuple[cute.Tensor, cute.Tensor, cute.Tensor]:
    """Swap operands while retaining token-major view strides."""
    a = create_cute_tensor_for_fp8(weight.detach(), cutlass.BFloat16, leading_dim=2)
    b = create_cute_tensor_for_fp8(input_tensor.transpose(1, 2), cutlass.BFloat16, leading_dim=1)
    if fp8_output:
        # Byte-backed E4M3 destination (quant_q_buffer view) without a
        # conversion launch: re-type the uint8 DLPack view in place.
        c = from_dlpack(output.transpose(1, 2), assumed_align=_ALIGNMENT_BYTES)
        c.element_type = cutlass.Float8E4M3FN
        c = c.mark_layout_dynamic(leading_dim=1)
    else:
        c = create_cute_tensor_for_fp8(output.transpose(1, 2), cutlass.BFloat16, leading_dim=1)
    return a, b, c


def _validate_tensor(
    tensor: torch.Tensor,
    name: str,
    shape: tuple[int, int, int],
    *,
    require_contiguous: bool,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    if tensor.shape != shape or tensor.dtype != dtype or not tensor.is_cuda:
        raise ValueError(
            f"{name} must be CUDA {dtype} {shape}, got shape={tuple(tensor.shape)} "
            f"dtype={tensor.dtype} device={tensor.device}"
        )
    if tensor.stride(2) != 1:
        raise ValueError(f"{name} innermost dimension must be contiguous")
    if any(tensor.stride(dimension) % _BF16_VECTOR_SIZE for dimension in (0, 1)):
        raise ValueError(
            f"{name} head and row strides must be divisible by {_BF16_VECTOR_SIZE} BF16 elements"
        )
    if require_contiguous and not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if tensor.data_ptr() % _ALIGNMENT_BYTES:
        raise ValueError(f"{name} must be {_ALIGNMENT_BYTES}-byte aligned")


def _validate_contract(
    abi: BatchedGemmAbi,
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    *,
    fp8_output: bool = False,
) -> None:
    _validate_tensor(
        input_tensor,
        "input_tensor",
        (abi.num_heads, abi.active_m, abi.input_features),
        require_contiguous=False,
    )
    _validate_tensor(
        weight,
        "weight",
        (abi.num_heads, abi.output_features, abi.input_features),
        require_contiguous=True,
    )
    _validate_tensor(
        output,
        "output",
        (abi.num_heads, abi.active_m, abi.output_features),
        require_contiguous=False,
        # FP8 q-base: this GEMM writes E4M3 bytes straight into quant_q_buffer.
        dtype=torch.uint8 if fp8_output else torch.bfloat16,
    )
    if input_tensor.device != weight.device or output.device != weight.device:
        raise ValueError("input_tensor, weight, and output must share one CUDA device")
    if tuple(input_tensor.stride()) != abi.input_stride:
        raise ValueError(
            f"{abi.module_name}: expected input stride {abi.input_stride}, "
            f"got {tuple(input_tensor.stride())}"
        )
    if tuple(output.stride()) != abi.output_stride:
        raise ValueError(
            f"{abi.module_name}: expected output stride {abi.output_stride}, "
            f"got {tuple(output.stride())}"
        )


def reject_unpredicated_store(
    module_name: str,
    schedule,
    rows_allowed: tuple[int, ...],
    *,
    emit_fp8_q_base: bool,
) -> None:
    """Fail closed on a schedule whose non-TMA store could write past N or skip the FP8 path.

    In the swapped orientation the GEMM's N is the token-row count.  The
    library's register store is unpredicated on the N edge, so a schedule with
    ``use_tma_store=False`` may only serve row counts that are whole multiples
    of ``tile_n``; other row counts write past the end.  The FP8 q-base
    epilogue is the TMA-store word-packed path, so it requires the TMA store
    as well.
    """
    if getattr(schedule, "use_tma_store", True):
        return
    tile_n = int(schedule.mma_tiler_mn[1])
    if emit_fp8_q_base:
        raise ValueError(f"{module_name}: use_tma_store=False bypasses the FP8 q-base epilogue")
    partial = tuple(rows for rows in rows_allowed if int(rows) % tile_n)
    if partial:
        raise ValueError(
            f"{module_name}: use_tma_store=False stores the N edge unpredicated; "
            f"rows {partial} are not multiples of tile_n={tile_n}"
        )


def _make_gemm(
    schedule: GemmSchedule,
    pdl_role: PersistentPdlRole | None,
    mxf8_side: Mxf8SideOutput | None,
):
    """The persistent GEMM kernel object of one build: the library kernel, or its PDL-chain
    variant (with the MXF8 side output when given)."""
    if pdl_role is None:
        return _GlmPersistentDenseGemmKernel(
            cutlass.Float32,
            schedule.use_2cta_instrs,
            schedule.mma_tiler_mn,
            schedule.cluster_shape_mn,
            schedule.use_tma_store,
            schedule.swizzle_size,
            schedule.raster_along,
        )
    return make_pdl_chain_kernel(schedule, pdl_role, mxf8_side)


def _kernel_name_prefix(
    abi: BatchedGemmAbi,
    pdl_role: PersistentPdlRole | None,
    *,
    emit_fp8_q_base: bool,
    mxf8_side: Mxf8SideOutput | None,
) -> str:
    """The kernel symbol prefix of one build (serving validators read its suffixes)."""
    kernel_name_prefix = abi.kernel_name_prefix
    if emit_fp8_q_base:
        kernel_name_prefix = f"{kernel_name_prefix}{FP8_Q_BASE_KERNEL_SUFFIX}"
    if mxf8_side is not None:
        kernel_name_prefix = f"{kernel_name_prefix}{MXF8_SIDE_KERNEL_SUFFIX}"
    if pdl_role is not None:
        kernel_name_prefix = f"{kernel_name_prefix}_pdl_chain"
        if pdl_role.signal_after_wait:
            kernel_name_prefix = f"{kernel_name_prefix}_es"
    return kernel_name_prefix


@dataclass(frozen=True)
class EagerBatchedGemm:
    """A BF16 build's kernel as a TVM-FFI function, for eager calls (CTX-HOST-DIET).

    `CompiledBatchedGemm` builds three DLPack cute tensors and packs the JIT executor's
    arguments on every call: about 100 us of Python per call for a mixed step's K-B / V-B
    at 16 generation rows (cProfile, B300 serving run). This is `_compile`'s kernel for the same
    arguments -- same kernel object (`_make_gemm`), `can_implement` gate, symbol prefix,
    `bmm` entry and dynamic layouts (`from_dlpack(assumed_align=16)` with the same stride-1
    modes) -- exported with `--enable-tvm-ffi`, so the same device code: a call checks the
    row count, every shape and stride against the build's, the device and 16-byte alignment,
    then hands the three `bmm` torch views to one TVM-FFI call on torch's current stream.
    (`_validate_contract`'s remaining terms -- BF16, one device -- are checked too.)
    """

    heads: int
    k: int
    n: int
    rows_allowed: frozenset[int]
    input_stride: tuple[int, ...]
    weight_stride: tuple[int, ...]
    output_stride: tuple[int, ...]
    device: torch.device
    fn: Callable

    def __call__(self, input_tensor: torch.Tensor, weight: torch.Tensor, output: torch.Tensor):
        rows = input_tensor.shape[1]
        if (
            rows not in self.rows_allowed
            or input_tensor.shape != (self.heads, rows, self.k)
            or weight.shape != (self.heads, self.n, self.k)
            or output.shape != (self.heads, rows, self.n)
            or input_tensor.stride() != self.input_stride
            or weight.stride() != self.weight_stride
            or output.stride() != self.output_stride
            or input_tensor.dtype != torch.bfloat16
            or weight.dtype != torch.bfloat16
            or output.dtype != torch.bfloat16
            or input_tensor.device != self.device
            or weight.device != self.device
            or output.device != self.device
        ):
            raise ValueError(
                f"eager batched GEMM: rows {sorted(self.rows_allowed)}, strides "
                f"{self.input_stride} / {self.weight_stride} / {self.output_stride} on "
                f"{self.device}; got input {tuple(input_tensor.shape)} {input_tensor.stride()}, "
                f"weight {tuple(weight.shape)} {weight.stride()}, output "
                f"{tuple(output.shape)} {output.stride()} on {input_tensor.device}"
            )
        if (input_tensor.data_ptr() | weight.data_ptr() | output.data_ptr()) & 15:
            raise ValueError("eager batched GEMM: operands must be 16-byte aligned")
        if weight.requires_grad:
            weight = weight.detach()
        return self.fn(weight, input_tensor.transpose(1, 2), output.transpose(1, 2))


def _compile_eager(
    abi: BatchedGemmAbi,
    schedule: GemmSchedule,
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    pdl_role: PersistentPdlRole | None,
    rows_allowed: tuple[int, ...],
) -> EagerBatchedGemm:
    """`EagerBatchedGemm` of the BF16 build `_compile` makes for these arguments."""
    a, b, c = (
        from_dlpack(view, assumed_align=_ALIGNMENT_BYTES, enable_tvm_ffi=True).mark_layout_dynamic(
            leading_dim=leading_dim
        )
        for view, leading_dim in (
            (weight.detach(), 2),
            (input_tensor.transpose(1, 2), 1),
            (output.transpose(1, 2), 1),
        )
    )
    mnkl = (abi.output_features, abi.active_m, abi.input_features, abi.num_heads)
    cluster_size = schedule.cluster_shape_mn[0] * schedule.cluster_shape_mn[1]
    max_active_clusters = _DENSE_GEMM.utils.HardwareInfo().get_max_active_clusters(cluster_size)
    gemm = _make_gemm(schedule, pdl_role, None)
    if not gemm.can_implement(mnkl, a.element_type, b.element_type, c.element_type, "k", "k", "m"):
        raise ValueError(f"{abi.module_name}: persistent schedule cannot implement {mnkl}")
    gemm.kernel.set_name_prefix(
        _kernel_name_prefix(abi, pdl_role, emit_fp8_q_base=False, mxf8_side=None),
        remove_cutlass_symbol=True,
    )
    fn = cute.compile(
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
    return EagerBatchedGemm(
        abi.num_heads,
        abi.input_features,
        abi.output_features,
        frozenset(int(m) for m in rows_allowed),
        tuple(input_tensor.stride()),
        tuple(weight.stride()),
        tuple(output.stride()),
        input_tensor.device,
        fn,
    )


def _compile(
    abi: BatchedGemmAbi,
    schedule: GemmSchedule,
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    pdl_role: PersistentPdlRole | None,
    *,
    emit_fp8_q_base: bool = False,
    m_rows_allowed: tuple[int, ...] = (),
    mxf8_side: Mxf8SideOutput | None = None,
) -> CompiledBatchedGemm:
    if schedule.orientation != "swapped":
        raise ValueError(f"{abi.module_name}: batched MLA adapter requires swapped orientation")
    reject_unpredicated_store(
        abi.module_name,
        schedule,
        tuple(m_rows_allowed) or (int(abi.active_m),),
        emit_fp8_q_base=emit_fp8_q_base,
    )
    if emit_fp8_q_base and pdl_role is None:
        raise ValueError(f"{abi.module_name}: emit_fp8_q_base requires a PDL-chain role")
    if mxf8_side is not None and (
        pdl_role is None or emit_fp8_q_base or not getattr(schedule, "use_tma_store", True)
    ):
        raise ValueError(
            f"{abi.module_name}: the MXF8 side output needs a PDL-chain role, a BF16 C and "
            "the TMA-store epilogue"
        )
    a, b, c = _make_runtime_arguments(input_tensor, weight, output, fp8_output=emit_fp8_q_base)
    mnkl = (
        abi.output_features,
        abi.active_m,
        abi.input_features,
        abi.num_heads,
    )
    cluster_size = schedule.cluster_shape_mn[0] * schedule.cluster_shape_mn[1]
    dense_gemm = _DENSE_GEMM
    max_active_clusters = dense_gemm.utils.HardwareInfo().get_max_active_clusters(cluster_size)
    gemm = _make_gemm(schedule, pdl_role, mxf8_side)
    if not gemm.can_implement(
        mnkl,
        a.element_type,
        b.element_type,
        c.element_type,
        "k",
        "k",
        "m",
    ):
        raise ValueError(f"{abi.module_name}: persistent schedule cannot implement {mnkl}")
    gemm.kernel.set_name_prefix(
        _kernel_name_prefix(abi, pdl_role, emit_fp8_q_base=emit_fp8_q_base, mxf8_side=mxf8_side),
        remove_cutlass_symbol=True,
    )
    compiled_bmm = cute.compile(
        dense_gemm.bmm,
        gemm,
        a,
        b,
        c,
        max_active_clusters,
        make_fake_stream(),
        lambda x: x,
    )
    return CompiledBatchedGemm(
        input_tensor.shape,
        weight.shape,
        output.shape,
        tuple(input_tensor.stride()),
        tuple(weight.stride()),
        tuple(output.stride()),
        compiled_bmm,
        fp8_output=emit_fp8_q_base,
        m_rows_allowed=tuple(m_rows_allowed) or (int(abi.active_m),),
    )


def _device_index(tensor: torch.Tensor) -> int:
    index = tensor.device.index
    return torch.cuda.current_device() if index is None else index


def make_build(
    abi: BatchedGemmAbi,
    schedule: GemmSchedule,
    *,
    pdl_role: PersistentPdlRole | None = None,
    emit_fp8_q_base: bool = False,
    m_rows_allowed: tuple[int, ...] = (),
    mxf8_side: Mxf8SideOutput | None = None,
) -> Callable[[], CompiledBatchedGemm]:
    """Create the no-argument factory consumed by the TRT module registry.

    :param emit_fp8_q_base: K-B only: the output is the ``uint8``
        ``quant_q_buffer[..., :512]`` view and the epilogue writes exact
        FP32->BF16->FP8 bytes (see the module docstring). Default off.
    :param m_rows_allowed: row counts one compiled object serves (runtime-M);
        empty = exactly ``abi.active_m``.
    :param mxf8_side: V-B only (decode act-quant fusion): the epilogue also writes the BF16 output as
        MXF8 into the fixed buffers it names (``Mxf8SideOutput``). Default off.
    """
    compiled_by_layout: dict[tuple[object, ...], CompiledBatchedGemm] = {}
    # CTX-HOST-DIET: the BF16 builds' eager calls take `EagerBatchedGemm` (built beside the
    # first compile, which the target makes at load); captured calls keep `compiled`.
    eager_by_layout: dict[tuple[object, ...], EagerBatchedGemm] = {}
    want_eager = _HOST_DIET and not emit_fp8_q_base and mxf8_side is None
    rows_allowed = tuple(m_rows_allowed) or (int(abi.active_m),)
    reject_unpredicated_store(
        abi.module_name, schedule, rows_allowed, emit_fp8_q_base=emit_fp8_q_base
    )

    def build() -> CompiledBatchedGemm:
        compiled: CompiledBatchedGemm | None = None
        eager: EagerBatchedGemm | None = None

        def run_bmm(
            input_tensor: torch.Tensor,
            weight: torch.Tensor,
            output: torch.Tensor,
        ) -> object:
            nonlocal compiled, eager
            if eager is not None and not torch.cuda.is_current_stream_capturing():
                return eager(input_tensor, weight, output)
            rows = int(input_tensor.shape[1]) if input_tensor.ndim == 3 else -1
            if rows not in rows_allowed:
                raise ValueError(
                    f"{abi.module_name}: rows={rows} not served by this build "
                    f"(schedule group rows {rows_allowed})"
                )
            _validate_contract(
                dataclasses.replace(abi, active_m=rows),
                input_tensor,
                weight,
                output,
                fp8_output=emit_fp8_q_base,
            )
            if compiled is None:
                key = (
                    _device_index(input_tensor),
                    tuple(input_tensor.stride()),
                    tuple(weight.stride()),
                    tuple(output.stride()),
                    schedule,
                    pdl_role,
                    emit_fp8_q_base,
                    mxf8_side,
                )
                compiled = compiled_by_layout.get(key)
                if compiled is None:
                    cutlass.cuda.initialize_cuda_context(key[0])
                    compiled = _compile(
                        dataclasses.replace(abi, active_m=rows),
                        schedule,
                        input_tensor,
                        weight.detach(),
                        output,
                        pdl_role,
                        emit_fp8_q_base=emit_fp8_q_base,
                        m_rows_allowed=rows_allowed,
                        mxf8_side=mxf8_side,
                    )
                    compiled_by_layout[key] = compiled
                if want_eager and not torch.cuda.is_current_stream_capturing():
                    eager = eager_by_layout.get(key)
                    if eager is None:
                        eager = eager_by_layout[key] = _compile_eager(
                            dataclasses.replace(abi, active_m=rows),
                            schedule,
                            input_tensor,
                            weight,
                            output,
                            pdl_role,
                            rows_allowed,
                        )
            return compiled(input_tensor, weight, output)

        return run_bmm

    return build


# ---------------------------------------------------------------------------
# K-B owner contract (uses the adapter above).
# ---------------------------------------------------------------------------

# Compiled cells: the padded decode row counts of the serving config
# (CUDA-graph buckets 1/2/4/8/16/24/32 requests x MTP 2 rows), plus 128 / 192
# (32 requests x MTP3 / MTP5 widths 4 / 6).  ``active_m`` is
# the GEMM's N in the swapped orientation; the schedule, PDL role, epilogue and
# every other option are the same object at every cell.
ACTIVE_M_CELLS: tuple[int, ...] = (2, 4, 8, 16, 32, 48, 64, 128, 192)
_FORMAL_ACTIVE_M = 64


def make_abi(active_m: int) -> BatchedGemmAbi:
    """The K-B contract at one compiled cell (strides are M-independent)."""
    if int(active_m) not in ACTIVE_M_CELLS:
        raise ValueError(
            f"self_attn.k_b_bmm: no compiled cell for active_m={active_m}; "
            f"cells are {ACTIVE_M_CELLS}"
        )
    return BatchedGemmAbi(
        module_name="self_attn.k_b_bmm",
        num_heads=16,
        active_m=int(active_m),
        output_features=512,
        input_features=192,
        kernel_name_prefix="glm52_self_attn_k_b_dense_persistent",
        input_stride=(256, 4_096, 1),
        output_stride=(576, 9_216, 1),
    )


# Per-M schedule: (128, 8) for M <= 8 and (128, 32) for M >= 16. Tile N (the
# token-row dimension in the swapped orientation) sets the speed. Every schedule
# matches torch.bmm within tolerance. 2-CTA (256-row) tiles measured slower at
# M = 48 / 64. A (128, 16) cell without the TMA store is not used: the FP8 q-base
# epilogue is the TMA-store path, see ``reject_unpredicated_store``.  The
# persistent GEMM is compiled with dynamic layouts, so one compiled object per
# schedule serves its whole row range (``SCHEDULE_GROUPS``).
SCHEDULE_SMALL_M = GemmSchedule(False, (128, 8), (1, 1), raster_along="n")
SCHEDULE_LARGE_M = GemmSchedule(False, (128, 32), (1, 1), raster_along="n")
# MTP3 / MTP5 (128 / 192 rows): (128, 64). B300, cold weights: 2.80 / 3.13 us, against
# 4.24 us on the prefill K-B those rows took; (128, 32) 2.92 / 3.25, (64, 64) 2.84 / 3.09,
# (128, 128) 3.15 / 3.16. The target also builds it with the fp8 q-base epilogue and the
# early signal (`build_decode_fp8_q_base`): the rope kernel of widths 4 / 6 runs beside it.
SCHEDULE_MTP_M = GemmSchedule(False, (128, 64), (1, 1), raster_along="n")
# (upper row bound inclusive, group name) in ascending order.
SCHEDULE_GROUPS: tuple[tuple[int, str], ...] = (
    (8, "m_le_8"),
    (64, "m_ge_16"),
    (192, "m_ge_128"),
)
_SCHEDULE_BY_GROUP = {
    "m_le_8": SCHEDULE_SMALL_M,
    "m_ge_16": SCHEDULE_LARGE_M,
    "m_ge_128": SCHEDULE_MTP_M,
}


def schedule_group_for_m(active_m: int) -> str:
    """Name of the schedule group serving ``active_m`` (one of ACTIVE_M_CELLS)."""
    if int(active_m) not in ACTIVE_M_CELLS:
        raise ValueError(
            f"self_attn.k_b_bmm: no compiled cell for active_m={active_m}; "
            f"cells are {ACTIVE_M_CELLS}"
        )
    for upper, name in SCHEDULE_GROUPS:
        if int(active_m) <= upper:
            return name
    raise AssertionError("SCHEDULE_GROUPS does not cover ACTIVE_M_CELLS")


def rows_of_group(group: str) -> tuple[int, ...]:
    """The cells one compiled object of ``group`` serves."""
    return tuple(m for m in ACTIVE_M_CELLS if schedule_group_for_m(m) == group)


#: The MTP group's compiled object (built at 192 rows) also serves every other MTP3 / MTP5
#: decode row count above 64 at runtime M (a batch of fewer than 32 sequences: 65..191
#: rows). The layouts are dynamic and the epilogue is the TMA store, which clips the last
#: tile, so each row is bitwise what 128 / 192 compute for it.
MTP_GROUP_RUNTIME_ROWS: tuple[int, ...] = tuple(range(65, 193))

#: The M >= 16 group's compiled object (built at 64 rows) also serves every decode row count
#: in 9..64 at runtime M: the MTP3 / MTP5 decode counts without a cell (MTP5 at 2 / 3
#: requests: 12 / 18 rows; MTP3 at 3: 12). Same reasoning as the MTP group: dynamic layouts,
#: rows past M load zeros, and the TMA store clips the last tile. Its tile N is 32, so 16 and
#: 48 rows were already partial tiles; 9..15 is one partial tile as 16 is. Each row is
#: bitwise what the group's cells compute for it.
DECODE_GROUP_RUNTIME_ROWS: tuple[int, ...] = tuple(range(9, 65))


def rows_allowed(group: str) -> tuple[int, ...]:
    """The row counts one compiled object of ``group`` accepts at launch."""
    if group == "m_ge_128":
        return MTP_GROUP_RUNTIME_ROWS
    if group == "m_ge_16":
        return DECODE_GROUP_RUNTIME_ROWS
    return rows_of_group(group)


@lru_cache(maxsize=None)
def _build_factory(emit_fp8_q_base: bool, group: str, early_signal: bool = False):
    rows = rows_of_group(group)
    return make_build(
        make_abi(rows[-1]),
        _SCHEDULE_BY_GROUP[group],
        pdl_role=(
            WAIT_EARLY_SIGNAL_PRELOAD_WEIGHT_ROLE
            if early_signal
            else WAIT_SIGNAL_PRELOAD_WEIGHT_ROLE
        ),
        emit_fp8_q_base=emit_fp8_q_base,
        m_rows_allowed=rows_allowed(group),
    )


def build(
    *,
    emit_fp8_q_base: bool = False,
    active_m: int = _FORMAL_ACTIVE_M,
    schedule_group: str | None = None,
    early_signal: bool = False,
):
    """Registry factory.

    ``early_signal`` (decode K-B / RoPE overlap, needs ``emit_fp8_q_base``): the dependent grid is
    signalled right after the activation wait as well, so the RoPE generation kernel compiled with
    ``late_wait`` (which does not read this GEMM's output) runs concurrently with it;
    kernel name gains ``_es``. Same GEMM and output bytes.

    ``emit_fp8_q_base`` selects the FP8 q-base epilogue; ``active_m`` selects the
    cell (one of :data:`ACTIVE_M_CELLS`) and thereby the schedule group.  The
    runtime passes ``schedule_group`` as well;
    it must agree with the cell -- a disagreement is a routing fault and raises.
    """
    group = schedule_group_for_m(int(active_m))
    if schedule_group is not None and schedule_group != group:
        raise ValueError(
            f"self_attn.k_b_bmm: schedule_group={schedule_group!r} does not serve "
            f"active_m={active_m} (expected {group!r})"
        )
    if early_signal and not emit_fp8_q_base:
        raise ValueError(
            "self_attn.k_b_bmm: early_signal needs emit_fp8_q_base (the RoPE generation kernel then skips the q-base)"
        )
    return _build_factory(bool(emit_fp8_q_base), group, bool(early_signal))()
