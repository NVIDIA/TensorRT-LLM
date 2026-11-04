# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resource-bounded tensor-core router for GLM decode shapes T=1..64.

The SIMT router reuses one input vector across two experts, which is ideal for
T<=16.  Extending that schedule by adding token tiles reloads the entire 3 MiB
router weight once per tile and can launch 512 CTAs at T=64.  This variant uses
split-K4 to expose 64 tensor-core CTAs while leaving more than half of B300's
SMs available to the shared-expert stream.  The fused TopK consumer adds the
four FP32 partials in registers, so the selected path has no standalone router
reduction launch.  Split-K2 and split-K8 remain as diagnostic options (``_SPLIT_K``).
"""

import math
from collections.abc import Callable

import cutlass
import cutlass.cute as cute
import torch
from cutlass.experimental import primitives as prims

from tensorrt_llm._torch.kernels.blackwell import dense_gemm_persistent as _DENSE_GEMM
from tensorrt_llm._torch.kernels.blackwell import utils as _TRTLLM_UTILS


class PersistentDenseGemmKernel(_DENSE_GEMM.PersistentDenseGemmKernel):
    """TRT-LLM's persistent dense GEMM under the router's constructor names.

    The router passes three options whose defaults select the behaviour this
    kernel already has (static persistent scheduler, heuristic stage counts);
    anything else is rejected rather than ignored.
    """

    def __init__(
        self,
        *args,
        use_direct_tile_scheduler: bool = False,
        forced_num_ab_stage: int | None = None,
        forced_num_c_stage: int | None = None,
        **kwargs,
    ):
        if use_direct_tile_scheduler or forced_num_ab_stage or forced_num_c_stage:
            raise NotImplementedError(
                "direct tile scheduler and forced stage counts are not supported"
            )
        super().__init__(*args, **kwargs)

    # Same signature: explicit (M|N, batch) strides for both A and B.
    wrapper_strided_ab = _DENSE_GEMM.PersistentDenseGemmKernel.wrapper_strided


make_ptr = _TRTLLM_UTILS.make_ptr

_MIN_TOKENS = 1
_MAX_TOKENS = 64
_FORCED_AB_STAGES = 0
_FORCED_C_STAGES = 0
_USE_TMA_STORE = True
_USE_DIRECT_SCHEDULER = False
_EXPERTS = 256
_HIDDEN = 6144
_SPLIT_K = 4
if _SPLIT_K not in (2, 4, 8):
    raise ValueError("_SPLIT_K must be 2, 4, or 8")
_MMA_TILER_MN = (64, 32)
_CLUSTER_SHAPE_MN = (1, 2)
_SPLITK_CLUSTER_SHAPE_MN = (
    2,
    2,
)

RouterTcFn = Callable[..., None]
_GEMM_CACHE: dict[tuple[int, int], RouterTcFn] = {}


def _device_index(tensor: torch.Tensor) -> int:
    if tensor.device.index is not None:
        return tensor.device.index
    return torch.cuda.current_device()


def router_tc_uses_pdl() -> bool:
    """Return the PDL mode compiled into the reused persistent GEMM."""
    return bool(_DENSE_GEMM.TRTLLM_ENABLE_PDL)


def _require_pdl_contract(use_pdl: bool) -> None:
    """Keep the public launch contract aligned with the reused TRT-LLM GEMM.

    ``PersistentDenseGemmKernel`` specializes its PDL wait and producer signal
    from ``TRTLLM_ENABLE_PDL`` when its module is imported.  The GLM wrapper
    cannot override that constexpr per call, so accepting a conflicting
    ``use_pdl`` value would silently advertise a dependency edge the launched
    kernel does not implement (or execute one the caller did not request).
    """
    compiled_use_pdl = router_tc_uses_pdl()
    if use_pdl != compiled_use_pdl:
        raise ValueError(
            "use_pdl must match TRTLLM_ENABLE_PDL for the reused persistent "
            f"router GEMM; requested use_pdl={use_pdl}, but the imported "
            f"kernel has TRTLLM_ENABLE_PDL={compiled_use_pdl}. Set the "
            "environment before importing moe_frontend_router_tc."
        )


@cute.kernel
def _splitk2_reduce_kernel(
    workspace: cute.Tensor,
    output: cute.Tensor,
    total_elements: cutlass.Constexpr,
    n: cutlass.Constexpr,
    use_pdl: cutlass.Constexpr,
) -> None:
    workspace_ptr = workspace.iterator
    output_ptr = output.iterator
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdimx, _, _ = cute.arch.block_dim()
    idx = bidx * bdimx + tidx

    if idx < total_elements:
        row = idx // n
        col = idx - row * n
        first_offset = row * cutlass.Int32(_SPLIT_K * n) + col
        total = cutlass.Float32(0.0)
        for split in cutlass.range_constexpr(_SPLIT_K):
            total = total + (workspace_ptr + first_offset + cutlass.Int32(split * n)).load()
        (output_ptr + idx).store(total)

    if cutlass.const_expr(use_pdl):
        prims.barrier_cta_sync(0)
        if tidx == cutlass.Int32(0):
            prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)


_splitk2_reduce_kernel.set_name_prefix("glm52_moe_router_tc_reduce", remove_cutlass_symbol=True)


def _work_clusters(
    m: int,
    n: int,
    cluster_shape_mn: tuple[int, int] = _CLUSTER_SHAPE_MN,
    mma_tiler_mn: tuple[int, int] = _MMA_TILER_MN,
) -> int:
    m_tiles = math.ceil(m / mma_tiler_mn[0])
    n_tiles = math.ceil(n / mma_tiler_mn[1])
    return math.ceil(m_tiles / cluster_shape_mn[0]) * math.ceil(n_tiles / cluster_shape_mn[1])


def _compile_gemm(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    workspace_view: torch.Tensor,
) -> RouterTcFn:
    m, k_total = input_tensor.shape
    n = weight.shape[0]
    device = _device_index(input_tensor)
    key = (device, m)
    if key in _GEMM_CACHE:
        return _GEMM_CACHE[key]

    k_split = k_total // _SPLIT_K
    if not PersistentDenseGemmKernel.can_implement(
        cutlass.BFloat16,
        cutlass.Float32,
        cutlass.Float32,
        False,
        _MMA_TILER_MN,
        _SPLITK_CLUSTER_SHAPE_MN,
        m,
        n,
        k_split,
        _SPLIT_K,
        "k",
        "k",
        "n",
    ):
        raise ValueError(f"persistent router GEMM cannot implement T={m}")

    a_ptr = make_ptr(
        cutlass.BFloat16,
        input_tensor.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    b_ptr = make_ptr(
        cutlass.BFloat16,
        weight.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    c_tensor = cute.runtime.from_dlpack(workspace_view).mark_layout_dynamic(leading_dim=1)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    gemm = PersistentDenseGemmKernel(
        cutlass.Float32,
        use_2cta_instrs=False,
        mma_tiler_mn=_MMA_TILER_MN,
        cluster_shape_mn=_SPLITK_CLUSTER_SHAPE_MN,
        use_tma_store=_USE_TMA_STORE,
        use_direct_tile_scheduler=_USE_DIRECT_SCHEDULER,
        forced_num_ab_stage=_FORCED_AB_STAGES or None,
        forced_num_c_stage=_FORCED_C_STAGES or None,
    )
    gemm.kernel.set_name_prefix("glm52_moe_router_tc_splitk", remove_cutlass_symbol=True)
    hardware_info = cutlass.utils.HardwareInfo()
    hardware_limit = hardware_info.get_max_active_clusters(
        _SPLITK_CLUSTER_SHAPE_MN[0] * _SPLITK_CLUSTER_SHAPE_MN[1]
    )
    max_active_clusters = min(
        hardware_limit,
        _work_clusters(m, n, _SPLITK_CLUSTER_SHAPE_MN),
    )

    _GEMM_CACHE[key] = cute.compile(
        gemm.wrapper_strided_ab,
        m,
        n,
        k_split,
        _SPLIT_K,
        a_ptr,
        b_ptr,
        c_tensor,
        k_total,
        k_split,
        k_total,
        k_split,
        max_active_clusters,
        stream,
        options="--opt-level 2 --enable-tvm-ffi",
    )
    return _GEMM_CACHE[key]


def _validate_router_tc_partials(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
) -> int:
    """Validate the split-logits ABI and return the selected CUDA device."""
    m, k = input_tensor.shape
    if not _MIN_TOKENS <= m <= _MAX_TOKENS:
        raise ValueError(f"tensor-core router supports T in [1,64], got {m}")
    if weight.shape != (_EXPERTS, _HIDDEN) or k != _HIDDEN:
        raise ValueError("tensor-core router requires [T,6144] x [256,6144]")
    if output.shape != (m, _SPLIT_K, _EXPERTS):
        raise ValueError(f"split router output must be [{m},{_SPLIT_K},{_EXPERTS}]")
    if input_tensor.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("router input and weight must be BF16")
    if output.dtype != torch.float32:
        raise TypeError("split router output must be FP32")
    if not input_tensor.is_contiguous() or not weight.is_contiguous():
        raise ValueError("router input and weight must be contiguous")
    if not output.is_contiguous():
        raise ValueError("split router output must be contiguous")

    device = _device_index(input_tensor)
    if _device_index(weight) != device or _device_index(output) != device:
        raise ValueError("router tensors must share one CUDA device")
    return device


def compile_router_tc_partials(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    *,
    use_pdl: bool,
) -> RouterTcFn:
    """Compile or retrieve the split-K router used by fused TopK/routing."""
    _validate_router_tc_partials(input_tensor, weight, output)
    _require_pdl_contract(use_pdl)
    output_view = output.permute(0, 2, 1)
    return _compile_gemm(input_tensor, weight, output_view)


def run_router_tc_partials(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    *,
    use_pdl: bool,
) -> None:
    """Launch split-K router GEMM and leave FP32 partials for TopK.

    The fused consumer adds the partial logits in registers immediately
    before sigmoid/TopK.  Keeping the split dimension in this interface avoids
    a standalone reduction launch and avoids writing then rereading a reduced
    ``[T,256]`` logits tensor.  Splitting K increases the number of bounded
    router CTAs, which is useful for the latency-limited 3 MiB router-weight
    read while retaining a small grid relative to the 148-SM B300.
    """
    gemm = compile_router_tc_partials(
        input_tensor,
        weight,
        output,
        use_pdl=use_pdl,
    )
    m = int(input_tensor.shape[0])
    output_view = output.permute(0, 2, 1)
    k_split = _HIDDEN // _SPLIT_K
    gemm(
        m,
        _EXPERTS,
        k_split,
        _SPLIT_K,
        input_tensor.data_ptr(),
        weight.data_ptr(),
        output_view,
        _HIDDEN,
        k_split,
        _HIDDEN,
        k_split,
    )
