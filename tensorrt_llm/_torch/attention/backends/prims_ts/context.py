# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Task-scheduled contiguous and paged context attention.

The public surface intentionally exposes attention semantics and optional
compile-time metadata contracts, not scheduler choices. Reusable contiguous
and paged plans select resource topology from capacities, while logical Q/KV
lengths and request metadata remain runtime inputs. Both wrappers consume
the current request extents at launch. Paged plans
conservatively treat all request lengths as variable unless the caller opts
into an exact-uniform or zero-causal-offset contract.
Single-instance dense paged plans use a static persistent launch when their
logical CTA grid exceeds one resident wave; smaller grids launch one CTA per
tile. Causal and paired persistent paged plans use CLC. The private policy is
query-paired unless a positive left window requires head-paired GQA.
Causal windows are bottom-right aligned: for row ``q``, the inclusive right
position is ``q + (S_kv - S_q)`` and ``window_left`` is measured from that
position.

PrimTS context entry points are intentionally excluded from ``fi_trace`` for
now; unlike the decode APIs, their ``@flashinfer_experimental_api`` decorators do not
register trace templates.
"""

from dataclasses import dataclass, field, replace
import functools
import itertools
import math
from typing import TYPE_CHECKING, Callable, Literal, Optional

import torch

from ._compile_cache import build_and_load_attention_kernel
from ._hardware import get_max_active_clusters

from .vc_attention import (
    VCAttentionConfig,
    VCAttentionParams,
    validate_vc_params,
    vc_repair_kv_len,
    vc_scale_shapes,
)

from flashinfer.api_logging import flashinfer_api as flashinfer_experimental_api
from .decode import (
    _validate_native_hnd_tma_strides,
    prepare_lse,
    validate_lse_mode,
    validate_attention_sinks,
    _validate_scale,
    _validate_layout as _validate_kv_layout,
)

from .sage.config import (
    SageAttentionConfig,
    SageAttentionParams,
    validate_sage_params,
    validate_sage_recipe,
    _validate_int8_qk_device,
)
from .sage.layout import (
    _SAGE_ATOM_TOKENS,
    _SageKSection,
    _SageQSection,
    _SageScaleLayout,
    _SageVSection,
)
from .sage.plan import SagePreparedScales, build_sage_prepared_scales


if TYPE_CHECKING:
    from .kernels.fmha_context.fmha_resources import FmhaConfig


_COMPILE_OPTIONS = "--enable-tvm-ffi --opt-level 3"
_SUPPORTED_HEAD_DIMS = (128, 192, 256)
_DEFAULT_PAGED_KV_PAGE_SIZE = 32
_SUPPORTED_PAGED_KV_PAGE_SIZES = (4, 8, 16, 32, 64, 128)
_SUPPORTED_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float8_e4m3fn,
)
_SUPPORTED_COMPUTE_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
_INT32_MAX = 2**31 - 1
_CUDA_GRID_YZ_MAX = 65_535
_CONTEXT_KV_TILE_N = 128
_CONTEXT_TILE_SIZE_Q = 128
# Query-paired D128 represents two 128-row Q tiles in one work tile.  Kernel
# coordinates include the padded tail of that 256-row span and, for masking,
# its exclusive right boundary.  Reserve the maximum 255-row tail padding in
# every packed logical data extent so all such coordinates remain signed
# Int32 even when the final request ends at the plan's packed extent.
_CONTEXT_MAX_Q_ROWS_PER_WORK_TILE = 256
_CONTEXT_PADDED_EXTENT_MAX = _INT32_MAX - (_CONTEXT_MAX_Q_ROWS_PER_WORK_TILE - 1)


def _variable_window_tile_size_q(head_dim: int) -> int:
    """Return the number of Q rows sharing one variable-window CTA bound."""

    return (
        _CONTEXT_MAX_Q_ROWS_PER_WORK_TILE
        if head_dim == _CONTEXT_TILE_SIZE_Q
        else _CONTEXT_TILE_SIZE_Q
    )


@dataclass(frozen=True)
class _ContextGeometry:
    """Validated request geometry used by the contiguous one-shot adapter."""

    device: torch.device
    device_index: int
    packed_query: bool
    batch_size: int
    total_q: int
    total_k: int
    max_q_len: int
    max_seq_len_k: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    qk_dtype: torch.dtype
    pv_dtype: torch.dtype
    out_dtype: torch.dtype
    mask_type: str
    window_left: int
    head_paired: bool
    uniform_packed_lengths: bool
    has_q_offset: bool
    causal_single_kv_tile: bool
    packed_dense_k_mask: bool
    q_shape: tuple[int, ...]
    kv_shape: tuple[int, ...]
    head_dim_vo: int | None = None


@dataclass(frozen=True)
class _ContextPlanGeometry:
    """Static compile geometry for one reusable contiguous context plan."""

    device: torch.device
    device_index: int
    packed_query: bool
    batch_size: int
    max_q_len: int
    max_seq_len_k: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    qk_dtype: torch.dtype
    pv_dtype: torch.dtype
    out_dtype: torch.dtype
    mask_type: str
    window_left: int
    head_paired: bool
    uniform_packed_lengths: bool
    has_q_offset: bool
    causal_single_kv_tile: bool
    packed_dense_k_mask: bool
    head_dim_vo: int | None = None
    # VC-Attention recipe, or ``None`` for a plain plan.
    vc: Optional[VCAttentionConfig] = None
    # The plan's Sage recipe, part of the compile identity; ``None`` without
    # Sage attention.
    sage: SageAttentionConfig | None = None


@dataclass(frozen=True)
class _ContextPlanState:
    """Atomically published state for one compiled contiguous context plan."""

    geometry: _ContextPlanGeometry
    scale_softmax_log2: float | torch.Tensor
    output_scale: float | torch.Tensor
    empty_i32: torch.Tensor
    variable_window_padded_starts: Optional[torch.Tensor]
    variable_window_cta_starts: torch.Tensor
    compiled: Callable[..., None]
    policy: tuple[tuple[str, object], ...]
    vc_scale_shapes: Optional[dict[str, tuple[int, ...]]] = None
    # Output scale of VC runs: the per-channel V scale times the planned or passed scale.
    vc_output_scale: Optional[torch.Tensor] = None
    lse_mode: str = "none"
    use_attention_sinks: bool = False
    sage: Optional[SagePreparedScales] = None
    cuda_scales: bool = False


@dataclass(frozen=True)
class _PagedContextPlanGeometry:
    """Static compile geometry for one reusable paged plan."""

    device: torch.device
    device_index: int
    batch_size: int
    max_q_len: int
    max_kv_len: int
    page_size: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    qk_dtype: torch.dtype
    pv_dtype: torch.dtype
    out_dtype: torch.dtype
    mask_type: str
    window_left: int
    head_paired: bool
    uniform_packed_lengths: bool
    has_q_offset: bool
    paged_v_tail_is_zero: bool
    packed_dense_k_mask: bool


@dataclass(frozen=True)
class _PagedContextPlanState:
    """Atomically published state for one compiled paged-context plan."""

    geometry: _PagedContextPlanGeometry
    scale_softmax_log2: float
    output_scale: float
    compiled: Callable[..., None]
    policy: tuple[tuple[str, object], ...]
    lse_mode: str = "none"
    use_attention_sinks: bool = False


_ContextScheduler = Literal[
    "nonpersistent",
    "static_persistent",
    "clc_dynamic_persistent",
]


@dataclass(frozen=True)
class _ContextSchedulerProbe:
    """Static kernel traits and device capacity used by scheduler policy."""

    single_qkv_instance: bool
    logical_work_tiles: int
    max_active_clusters: int


@dataclass(frozen=True)
class _ContextCompileSpec:
    """Batch-independent identity for one contiguous context compile."""

    device_index: int
    max_q_len: int | None = field(compare=False, hash=False)
    max_seq_len_k: int | None = field(compare=False, hash=False)
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    qk_dtype_key: str
    pv_dtype_key: str
    out_dtype_key: str
    mask_type: str
    window_left: int
    packed_query: bool
    head_paired: bool
    uniform_packed_lengths: bool
    has_q_offset: bool
    causal_single_kv_tile: bool
    packed_dense_k_mask: bool
    scheduler: _ContextScheduler
    head_dim_vo: int | None = None
    two_cta_umma: bool = False
    vc: Optional[VCAttentionConfig] = None
    lse_mode: str = "none"
    use_attention_sinks: bool = False
    runtime_lengths: bool = False
    shape_signature: tuple[int, ...] = ()
    # A Sage plan's prepared scale layout; the kernel reads the scales.
    sage_scale_layout: _SageScaleLayout | None = None
    cuda_scales: bool = False


@dataclass(frozen=True)
class _PagedContextCompileSpec:
    """Batch-independent identity for one paged context compile."""

    device_index: int
    max_q_len: int | None = field(compare=False, hash=False)
    max_kv_len: int | None = field(compare=False, hash=False)
    page_size: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    qk_dtype_key: str
    pv_dtype_key: str
    out_dtype_key: str
    mask_type: str
    window_left: int
    head_paired: bool
    uniform_packed_lengths: bool
    has_q_offset: bool
    paged_v_tail_is_zero: bool
    packed_dense_k_mask: bool
    scheduler: _ContextScheduler
    lse_mode: str = "none"
    use_attention_sinks: bool = False
    runtime_lengths: bool = False
    shape_signature: tuple[int, ...] = ()


def _make_context_kernel(
    *,
    device_index: int,
    input_qk_dtype,
    input_pv_dtype,
    out_dtype,
    head_dim: int,
    head_dim_vo: int | None = None,
    mask_type: str,
    window_left: int,
    head_paired: bool,
    num_qo_heads: int,
    num_kv_heads: int,
    has_q_offset: bool,
    causal_single_kv_tile: bool,
    scheduler: _ContextScheduler,
    uses_ldtm_stat: bool,
    fp8_psmem_early_token: bool = False,
    kv_ring_stages: int = 0,
    umma_fp8_k64: bool = False,
    two_cta_umma: bool = False,
    page_size: int | None = None,
    max_kv_len: int | None = None,
    vc_attention: str | None = None,
    sage_scale_layout: _SageScaleLayout | None = None,
):
    """Build one context kernel from its batch-independent static topology."""

    import cutlass

    from .kernels.fmha_context.fmha_kernel import FmhaTs

    use_paged_kv = page_size is not None
    # VC QK8 also fixes the K-scale extent through max_kv_len. Paged runtime
    # context only needs page geometry; its live KV bound is a launch operand.
    is_persistent = scheduler != "nonpersistent"
    is_clc_dynamic = scheduler == "clc_dynamic_persistent"
    paged_kwargs = (
        {"use_paged_kv": True, "num_tokens_per_page": page_size} if use_paged_kv else {}
    )
    if max_kv_len is not None:
        paged_kwargs["max_kv_len"] = max_kv_len
    fmha = FmhaTs(
        qk_acc_dtype=cutlass.Float32,
        pv_acc_dtype=cutlass.Float32,
        in_qk_dtype=input_qk_dtype,
        in_pv_dtype=input_pv_dtype,
        out_dtype=out_dtype,
        d=head_dim,
        d_v=head_dim_vo,
        is_persistent=is_persistent,
        fp8_psmem_early_token=fp8_psmem_early_token,
        kv_ring_stages=kv_ring_stages,
        umma_fp8_k64=umma_fp8_k64,
        two_cta_umma=two_cta_umma,
        is_causal=mask_type == "causal",
        has_variable_window=mask_type == "variable_window",
        balance_causal_workload=(
            scheduler == "static_persistent"
            and _uses_heavy_first_static_causal_raster(
                mask_type=mask_type,
                window_left=window_left,
                has_q_offset=has_q_offset,
            )
        ),
        is_clc_dynamic=is_clc_dynamic,
        head_paired=head_paired,
        window_size_left=window_left if window_left > 0 else 0,
        h_r=num_qo_heads // num_kv_heads,
        enable_skip_correction=True,
        uses_ldtm_stat=uses_ldtm_stat,
        causal_single_kv_tile=(causal_single_kv_tile and not use_paged_kv),
        vc_attention=vc_attention,
        use_blackwell_tma_oob_workaround=(
            torch.cuda.get_device_capability(device_index) in ((10, 0), (10, 3))
        ),
        sage_scale_layout=sage_scale_layout,
        **paged_kwargs,
    )
    return fmha


def _validate_tensor(tensor: torch.Tensor, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor")


def _compact_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    strides = []
    for extent in reversed(shape):
        strides.append(stride)
        stride *= int(extent)
    return tuple(reversed(strides))


def _validate_compact(tensor: torch.Tensor, name: str, layout: str) -> None:
    expected = _compact_strides(tuple(tensor.shape))
    if tensor.stride() != expected:
        raise ValueError(
            f"{name} must have compact {layout} strides {expected}, "
            f"but has {tensor.stride()}"
        )


def _validate_alignment(tensor: torch.Tensor, name: str, alignment: int) -> None:
    if tensor.data_ptr() % alignment != 0:
        raise ValueError(f"{name} data pointer must be {alignment}-byte aligned")


def _dtype_key(dtype: torch.dtype) -> str:
    keys = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float8_e4m3fn: "float8_e4m3fn",
        torch.int8: "int8",
    }
    try:
        return keys[dtype]
    except KeyError as error:
        raise NotImplementedError(
            "attention-ts context supports torch.float16, torch.bfloat16, "
            f"torch.float8_e4m3fn, and torch.int8 Q/K; got {dtype}"
        ) from error


def _v_dtype_follows_qk(q_dtype: torch.dtype, v_dtype: torch.dtype) -> bool:
    """Whether V may take ``v_dtype`` next to ``q_dtype`` Q/K.

    V matches Q/K except on the QK-BF16/PV-FP8 path and under a Sage recipe's
    Int8 Q/K, which keep E4M3 V.
    """
    return v_dtype == q_dtype or (
        q_dtype in (torch.bfloat16, torch.int8) and v_dtype == torch.float8_e4m3fn
    )


def _validate_qkv_dtype(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    """Validate run-time Q/K/V tensor dtypes.

    Q and K must match; V may differ only for the QK-BF16/PV-FP8 path and
    for Int8 Q/K with E4M3 V.
    """
    _dtype_key(q.dtype)
    if q.dtype != k.dtype:
        raise NotImplementedError(
            "attention-ts context requires Q and K to use the same dtype; "
            f"got Q {q.dtype} and K {k.dtype}"
        )
    if not _v_dtype_follows_qk(q.dtype, v.dtype):
        raise NotImplementedError(
            "attention-ts context requires Q, K, and V to use the same dtype "
            "except for QK-BF16/PV-FP8 and Int8 Q/K with E4M3 V; "
            f"got Q {q.dtype}, K {k.dtype}, and V {v.dtype}"
        )


def _validate_output_dtype(out_dtype: torch.dtype) -> None:
    if not isinstance(out_dtype, torch.dtype):
        raise TypeError("out_dtype must be a torch.dtype")
    _dtype_key(out_dtype)
    if out_dtype == torch.int8:
        raise NotImplementedError("attention-ts context does not write an int8 output")


def _validate_context_dtype_pair(
    q_dtype: torch.dtype,
    k_dtype: torch.dtype,
    v_dtype: torch.dtype,
    out_dtype: torch.dtype,
) -> None:
    """Validate the explicit dtypes of a contiguous-context specialization."""

    for dtype, name in (
        (q_dtype, "q_dtype"),
        (k_dtype, "k_dtype"),
        (v_dtype, "v_dtype"),
    ):
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"{name} must be a torch.dtype")
        _dtype_key(dtype)
    _validate_output_dtype(out_dtype)
    if k_dtype != q_dtype:
        raise NotImplementedError(
            "attention-ts context requires Q and K to use the same dtype; "
            f"got q_dtype={q_dtype} and k_dtype={k_dtype}"
        )
    if not _v_dtype_follows_qk(q_dtype, v_dtype):
        raise NotImplementedError(
            "attention-ts context requires Q, K, and V to use the same dtype "
            "except for QK-BF16/PV-FP8 and Int8 Q/K with E4M3 V; "
            f"got q_dtype={q_dtype} and v_dtype={v_dtype}"
        )


def _validate_paged_dtype_pair(
    q_dtype: torch.dtype,
    k_dtype: torch.dtype,
    v_dtype: torch.dtype,
    out_dtype: torch.dtype,
) -> None:
    """Validate the explicit dtypes of a paged-context specialization."""

    for dtype, name in (
        (q_dtype, "q_dtype"),
        (k_dtype, "k_dtype"),
        (v_dtype, "v_dtype"),
        (out_dtype, "out_dtype"),
    ):
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"{name} must be a torch.dtype")
        _dtype_key(dtype)
        if dtype == torch.int8:
            raise NotImplementedError(
                "attention-ts paged context does not support int8 tensors"
            )
    if k_dtype != q_dtype:
        raise NotImplementedError(
            "attention-ts paged context requires Q and K to use the same "
            f"dtype; got q_dtype={q_dtype} and k_dtype={k_dtype}"
        )
    if not _v_dtype_follows_qk(q_dtype, v_dtype):
        raise NotImplementedError(
            "attention-ts paged context requires Q, K, and V to use the same "
            "dtype except for QK-BF16/PV-FP8; "
            f"got q_dtype={q_dtype} and v_dtype={v_dtype}"
        )


def _device_index(device: torch.device) -> int:
    if device.index is not None:
        return int(device.index)
    return int(torch.cuda.current_device())


def _validate_device(device: torch.device) -> int:
    device_index = _device_index(device)
    with torch.cuda.device(device_index):
        capability = torch.cuda.get_device_capability(device_index)
    if capability not in _SUPPORTED_COMPUTE_CAPABILITIES:
        raise NotImplementedError(
            "attention-ts context requires an SM100a/B200, SM103a/B300 or "
            "SM107a/Rubin GPU; "
            f"device cuda:{device_index} has compute capability {capability}"
        )
    # Rubin runs through the sm_100f family target; a CuTe DSL older than 4.8
    # cannot emit for it unless CUTE_DSL_ARCH=sm_100f is set before import.
    if capability == (10, 7):
        from flashinfer.cute_dsl.utils import require_cute_dsl_arch

        require_cute_dsl_arch(device_index)
    return device_index


def _default_exp2_fma_pairs(device_index: int, cfg) -> int:
    """FMA-pipe exp2 pairs per 16-pair softmax chunk on SM100.

    Measured on B200. The paired D128 dense path is MUFU bound: with 16-bit V
    it takes 4, and with 8-bit V and P in SMEM, fixed-length plans take 3.
    Packed and paged 8-bit plans spend the FMA pipe on their masks and take 0.
    Single-QKV dense takes 2. Causal is issue bound and takes 0. Callers can
    set cfg.exp2_fma_pairs to any value on any path."""
    if torch.cuda.get_device_capability(device_index) != (10, 0) or cfg.is_causal:
        return 0
    if cfg.v_dtype.width == 8:
        return 0 if cfg.has_varlen or cfg.single_qkv_instance else 3
    if cfg.v_dtype.width != 16:
        return 0
    if cfg.pv_half_overlap:
        return 4
    return 2 if cfg.single_qkv_instance else 0


def _default_exp2_packed_f16x2(device_index: int, cfg) -> bool:
    """Softmax exp2 as one packed f16x2 MUFU op per score pair, on for the SM107
    paths with 8-bit V. SM100 and SM103 split the packed op into two MUFU ops and
    keep the scalar form."""
    return (
        torch.cuda.get_device_capability(device_index) == (10, 7)
        and cfg.v_dtype.width == 8
    )


def _default_kv_ring_stages(device_index: int, qk_bits: int, pv_bits: int) -> int:
    """K/V ring depth of the SM107 dense paths with 8-bit V, 0 elsewhere for the
    budget-derived depth. Deeper rings stall the softmax P stores behind the TMA
    fills, so the fp8 K ring stops at 6 stages, and the bf16 K ring takes the 4
    stages the larger SM107 shared memory fits."""
    if torch.cuda.get_device_capability(device_index) == (10, 7) and pv_bits == 8:
        return 6 if qk_bits == 8 else 4
    return 0


def _default_umma_fp8_k64(device_index: int, qk_bits: int) -> bool:
    """K=64 fp8 UMMA instructions for the PV of the SM107 bf16-QK paths. The
    fp8-QK paths are shared-memory-port bound on operand fetch and keep K=32."""
    return torch.cuda.get_device_capability(device_index) == (10, 7) and qk_bits == 16


def _default_fp8_psmem_early_token(device_index: int) -> bool:
    """Release the S0/S1 token early on the fp8 SMEM-P path, on for SM103."""
    return torch.cuda.get_device_capability(device_index) == (10, 3)


def _default_two_cta_umma(device_index: int, qk_bits: int) -> bool:
    """Two-CTA UMMA for the dense paired D128 context kernel. On for SM103, where
    the kernel is tensor-pipe bound and halving the UMMA issues and the K/V
    traffic per SM pays. On SM107 only for bf16 QK, which is shared-memory-port
    bound on operand fetch, since fp8 QK is slower with it."""
    capability = torch.cuda.get_device_capability(device_index)
    if capability == (10, 3):
        return True
    return capability == (10, 7) and qk_bits == 16


def _default_uses_ldtm_stat(device_index: int, qk_bits: int, pv_bits: int) -> bool:
    """Enable LDTM.STAT where the shared tcgen05 gate allows it. SM107 keeps it
    off for the dense QK-BF16 PV-FP8 path, which is faster with the register
    row max."""
    from .kernels.tcgen05_compat import ldtm_stat_supported

    capability = torch.cuda.get_device_capability(device_index)
    if capability == (10, 7) and qk_bits == 16 and pv_bits == 8:
        return False
    return ldtm_stat_supported(capability)


def _resolve_cuda_device(
    device: int | str | torch.device,
) -> tuple[torch.device, int]:
    """Resolve one explicit CUDA plan device and validate its architecture."""

    if isinstance(device, bool):
        raise TypeError("device must identify a CUDA device")
    try:
        resolved = (
            torch.device("cuda", device)
            if isinstance(device, int)
            else torch.device(device)
        )
    except (RuntimeError, TypeError, ValueError) as error:
        raise TypeError("device must identify a CUDA device") from error
    if resolved.type != "cuda":
        raise ValueError(f"device must be a CUDA device, got {resolved}")
    device_index = _validate_device(resolved)
    return torch.device("cuda", device_index), device_index


def _validate_mask(mask_type: str) -> None:
    if not isinstance(mask_type, str):
        raise TypeError("mask_type must be a string")
    if mask_type not in ("dense", "causal", "variable_window"):
        raise ValueError(
            "mask_type must be exactly 'dense', 'causal', or "
            f"'variable_window', got {mask_type!r}"
        )


def _validate_window_left(window_left: int, mask_type: str) -> int:
    if isinstance(window_left, bool) or not isinstance(window_left, int):
        raise TypeError("window_left must be an integer")
    if window_left == 0:
        raise ValueError("window_left=0 is unsupported; use -1 to disable the window")
    if window_left < -1:
        raise ValueError("window_left must be -1 (disabled) or positive")
    if window_left > _INT32_MAX - 1:
        raise ValueError(f"window_left must be no larger than {_INT32_MAX - 1}")
    if window_left > 0 and mask_type != "causal":
        raise ValueError("a positive window_left requires mask_type='causal'")
    return window_left


def _validate_variable_window_bounds(
    starts: Optional[torch.Tensor],
    ends: Optional[torch.Tensor],
    *,
    geometry: _ContextGeometry | _ContextPlanGeometry,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    """Validate inclusive per-query K bounds for contiguous attention."""
    if starts is None or ends is None:
        raise ValueError(
            "variable_window_token_starts and variable_window_token_ends are "
            "required for mask_type='variable_window'"
        )
    if geometry.packed_query:
        raise NotImplementedError(
            "variable_window currently requires fixed [B, S, H, D] tensors"
        )
    expected_shape = (geometry.batch_size, geometry.max_q_len)
    for tensor, name in (
        (starts, "variable_window_token_starts"),
        (ends, "variable_window_token_ends"),
    ):
        _validate_tensor(tensor, name)
        if tensor.device != geometry.device:
            raise ValueError(
                f"{name} must be on {geometry.device}, got {tensor.device}"
            )
        if tensor.dtype != torch.int32:
            raise ValueError(f"{name} must have dtype torch.int32, got {tensor.dtype}")
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}"
            )
        _validate_compact(tensor, name, "[B, Sq]")
        _validate_alignment(tensor, name, 4)
    start_values = tuple(int(value) for value in starts.flatten().tolist())
    end_values = tuple(int(value) for value in ends.flatten().tolist())
    for offset, (start, end) in enumerate(zip(start_values, end_values, strict=True)):
        if start < 0 or start > end or end >= geometry.max_seq_len_k:
            raise ValueError(
                "variable-window bounds must satisfy "
                "0 <= start <= end < max_kv_len; flattened row "
                f"{offset} has start={start}, end={end}, and "
                f"max_kv_len={geometry.max_seq_len_k}"
            )
        if offset % geometry.max_q_len and end < end_values[offset - 1]:
            batch_idx, query_idx = divmod(offset, geometry.max_q_len)
            raise ValueError(
                "variable_window_token_ends must be nondecreasing within each "
                f"batch item; batch {batch_idx}, query {query_idx} has "
                f"end={end} after end={end_values[offset - 1]}"
            )
    return starts.flatten(), ends.flatten(), start_values


def _refresh_variable_window_cta_starts(
    starts: torch.Tensor, *, state: _ContextPlanState
) -> torch.Tensor:
    """Refresh plan-owned CTA minima from the current variable-window bounds."""

    geometry = state.geometry
    tile_size_q = _variable_window_tile_size_q(geometry.head_dim)
    num_seq_tiles = (geometry.max_q_len + tile_size_q - 1) // tile_size_q
    padded_rows = num_seq_tiles * tile_size_q
    starts_2d = starts.view(geometry.batch_size, geometry.max_q_len)
    padded = state.variable_window_padded_starts
    if padded is not None:
        padded.fill_(_INT32_MAX)
        padded[:, : geometry.max_q_len].copy_(starts_2d)
        starts_2d = padded
    elif padded_rows != geometry.max_q_len:
        raise RuntimeError("variable-window padded-start scratch is missing")
    torch.amin(
        starts_2d.view(geometry.batch_size, num_seq_tiles, tile_size_q),
        dim=-1,
        out=state.variable_window_cta_starts.view(geometry.batch_size, num_seq_tiles),
    )
    return state.variable_window_cta_starts


def _validate_variable_window_cta_starts(
    cta_starts: torch.Tensor,
    *,
    token_start_values: tuple[int, ...],
    geometry: _ContextPlanGeometry,
) -> torch.Tensor:
    """Validate caller-precomputed per-CTA minimum K-token starts."""

    _validate_tensor(cta_starts, "variable_window_cta_starts")
    if cta_starts.device != geometry.device:
        raise ValueError(
            "variable_window_cta_starts must be on "
            f"{geometry.device}, got {cta_starts.device}"
        )
    if cta_starts.dtype != torch.int32:
        raise ValueError(
            "variable_window_cta_starts must have dtype torch.int32, "
            f"got {cta_starts.dtype}"
        )
    tile_size_q = _variable_window_tile_size_q(geometry.head_dim)
    num_seq_tiles = (geometry.max_q_len + tile_size_q - 1) // tile_size_q
    expected_shape = (geometry.batch_size, num_seq_tiles)
    if tuple(cta_starts.shape) != expected_shape:
        raise ValueError(
            "variable_window_cta_starts must have shape "
            f"{expected_shape}, got {tuple(cta_starts.shape)}"
        )
    _validate_compact(cta_starts, "variable_window_cta_starts", "[B, Q tiles]")
    _validate_alignment(cta_starts, "variable_window_cta_starts", 4)

    actual_values = tuple(int(value) for value in cta_starts.flatten().tolist())
    for batch_idx in range(geometry.batch_size):
        row_offset = batch_idx * geometry.max_q_len
        for tile_idx in range(num_seq_tiles):
            tile_begin = row_offset + tile_idx * tile_size_q
            tile_end = min(
                tile_begin + tile_size_q,
                row_offset + geometry.max_q_len,
            )
            expected = min(token_start_values[tile_begin:tile_end])
            actual = actual_values[batch_idx * num_seq_tiles + tile_idx]
            if actual != expected:
                raise ValueError(
                    "variable_window_cta_starts must contain the exact minimum "
                    f"token start for each Q tile; entry [{batch_idx}, "
                    f"{tile_idx}] is {actual}, expected {expected}"
                )
    return cta_starts.flatten()


def _validate_extent(value: int, name: str) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    if value > _INT32_MAX:
        raise NotImplementedError(f"{name} must fit in a signed int32")
    return value


def _validate_static_extent(value: object, name: str) -> int:
    """Validate one explicit positive plan bound."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return _validate_extent(value, name)


def _validate_padded_data_extent(value: int, name: str) -> int:
    """Validate an Int32 data extent while reserving work-tile tail padding."""
    _validate_extent(value, name)
    if value > _CONTEXT_PADDED_EXTENT_MAX:
        raise NotImplementedError(
            f"{name} must be <= {_CONTEXT_PADDED_EXTENT_MAX} so padded "
            "context work-tile coordinates fit in a signed int32"
        )
    return value


def _validate_query_work_tile_span(config: "FmhaConfig") -> None:
    """Keep the public Int32 padding guard coupled to generated topology."""
    q_rows = int(config.q_tile_m) * int(config.work_tile_q_seq_tiles)
    if q_rows > _CONTEXT_MAX_Q_ROWS_PER_WORK_TILE:
        raise RuntimeError(
            "context Int32 extent safety assumes at most "
            f"{_CONTEXT_MAX_Q_ROWS_PER_WORK_TILE} Q rows per work tile, got {q_rows}"
        )


def _validate_indptr_tensor(
    indptr: torch.Tensor,
    name: str,
    *,
    device: torch.device,
) -> None:
    _validate_tensor(indptr, name)
    if indptr.device != device:
        raise ValueError(f"{name} must be on {device}, got {indptr.device}")
    if indptr.dtype != torch.int32:
        raise TypeError(f"{name} must have dtype torch.int32")
    if indptr.ndim != 1:
        raise ValueError(f"{name} must be rank 1, got rank {indptr.ndim}")
    if indptr.numel() < 2:
        raise ValueError(f"{name} must contain at least start and end offsets")
    _validate_compact(indptr, name, "[B+1]")
    _validate_alignment(indptr, name, 4)


def _read_indptr(
    indptr: torch.Tensor,
    name: str,
    *,
    expected_total: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Read cumulative offsets and validate nonnegative row lengths."""

    values = tuple(int(value) for value in indptr.tolist())
    if values[0] != 0:
        raise ValueError(f"{name} must start at 0")
    if values[-1] != expected_total:
        raise ValueError(
            f"the final {name} offset must equal the packed tensor extent; "
            f"expected {expected_total}, got {values[-1]}"
        )
    lengths = tuple(curr - prev for prev, curr in itertools.pairwise(values))
    if any(length < 0 for length in lengths):
        raise ValueError(f"{name} offsets must be nondecreasing")
    return values, lengths


def _read_int32_values(
    tensor: torch.Tensor,
    name: str,
    *,
    expected_count: int,
) -> tuple[int, ...]:
    """Read one metadata vector after validating its extent."""

    if tensor.numel() != expected_count:
        raise ValueError(
            f"{name} must contain {expected_count} elements, got {tensor.numel()}"
        )
    return tuple(int(value) for value in tensor.tolist())


def _validate_paged_metadata_tensor(
    tensor: torch.Tensor,
    name: str,
    *,
    device: torch.device,
) -> None:
    _validate_tensor(tensor, name)
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if tensor.dtype != torch.int32:
        raise TypeError(f"{name} must have dtype torch.int32")
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be rank 1, got rank {tensor.ndim}")
    _validate_compact(tensor, name, "one-dimensional")
    _validate_alignment(tensor, name, 4)


def _validate_block_tables_tensor(
    block_tables: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    required_page_columns: int,
) -> None:
    """Validate a fixed page table while preserving its runtime row stride."""

    _validate_tensor(block_tables, "block_tables")
    if block_tables.device != device:
        raise ValueError(f"block_tables must be on {device}, got {block_tables.device}")
    if block_tables.dtype != torch.int32:
        raise TypeError("block_tables must have dtype torch.int32")
    if block_tables.ndim != 2:
        raise ValueError(f"block_tables must be rank 2, got rank {block_tables.ndim}")
    if int(block_tables.shape[0]) != batch_size:
        raise ValueError(
            "block_tables must have one row per request; expected "
            f"{batch_size}, got {block_tables.shape[0]}"
        )
    page_columns = int(block_tables.shape[1])
    if page_columns < required_page_columns:
        raise ValueError(
            "block_tables must have at least ceil(max_kv_len / page_size) "
            f"columns ({required_page_columns}), got {page_columns}"
        )
    row_stride = int(block_tables.stride(0))
    if block_tables.stride(1) != 1 or row_stride < page_columns:
        raise ValueError(
            "block_tables must have row-strided [B, C] storage with "
            f"stride(1) == 1 and stride(0) >= C; got shape "
            f"{tuple(block_tables.shape)} and strides {block_tables.stride()}"
        )
    _validate_extent(row_stride, "block_tables row stride")
    row_span = (batch_size - 1) * row_stride + page_columns
    _validate_extent(row_span, "block_tables addressable row span")
    _validate_alignment(block_tables, "block_tables", 4)


def _validate_page_size(page_size: int) -> int:
    if isinstance(page_size, bool) or not isinstance(page_size, int):
        raise TypeError("page_size must be an integer")
    if page_size not in _SUPPORTED_PAGED_KV_PAGE_SIZES:
        raise NotImplementedError(
            "attention-ts paged context supports page_size in "
            f"{_SUPPORTED_PAGED_KV_PAGE_SIZES}; got {page_size}"
        )
    return page_size


def _validate_base_tensors(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    allow_mla: bool = False,
) -> None:
    for tensor, name in ((q, "q"), (k, "k"), (v, "v")):
        _validate_tensor(tensor, name)
        _validate_alignment(tensor, name, 16)
    if k.device != q.device or v.device != q.device:
        raise ValueError(
            "Q, K, and V must be on one CUDA device; "
            f"got {q.device}, {k.device}, and {v.device}"
        )
    _validate_qkv_dtype(q, k, v)
    mla_shape = (
        allow_mla
        and k.ndim >= 1
        and tuple(v.shape) == (*k.shape[:-1], 128)
        and k.shape[-1] == 192
    )
    if tuple(v.shape) != tuple(k.shape) and not mla_shape:
        raise ValueError(
            f"v must have the same shape as k; got {tuple(v.shape)} and {tuple(k.shape)}"
        )


def _validate_head_geometry(num_qo_heads: int, num_kv_heads: int) -> int:
    if num_qo_heads <= 0 or num_kv_heads <= 0:
        raise ValueError("Q and KV head counts must be positive")
    if num_qo_heads % num_kv_heads != 0:
        raise ValueError(
            "the Q head count must be divisible by the KV head count; "
            f"got {num_qo_heads} and {num_kv_heads}"
        )
    return num_qo_heads // num_kv_heads


def _validate_head_dim(
    q_head_dim: int, kv_head_dim: int, v_head_dim: int | None = None
) -> int:
    if q_head_dim != kv_head_dim:
        raise ValueError(
            "Q and K head dimensions must match; "
            f"got Q {q_head_dim} and K {kv_head_dim}"
        )
    if (q_head_dim, v_head_dim) == (192, 128):
        return q_head_dim
    if v_head_dim is not None and v_head_dim != q_head_dim:
        raise NotImplementedError(
            "separate QK/V dimensions require contiguous QK/V=(192, 128); "
            f"got ({q_head_dim}, {v_head_dim})"
        )
    if q_head_dim not in _SUPPORTED_HEAD_DIMS:
        raise NotImplementedError(
            "attention-ts context supports head_dim in "
            f"{_SUPPORTED_HEAD_DIMS}, or contiguous QK/V=(192, 128); got {q_head_dim}"
        )
    return q_head_dim


def _derive_has_q_offset(
    q_lengths: tuple[int, ...],
    k_lengths: tuple[int, ...],
    mask_type: str,
) -> bool:
    """Return whether any request has a bottom-right causal Q offset."""

    if mask_type != "causal":
        return False
    return any(
        k_length != q_length
        for q_length, k_length in zip(q_lengths, k_lengths, strict=True)
    )


def _uses_heavy_first_static_causal_raster(
    *,
    mask_type: str,
    window_left: int,
    has_q_offset: bool,
) -> bool:
    """Return whether a static paged plan has triangular causal tile costs.

    Zero-offset, unwindowed causal attention grows from short to long K-tile
    domains across Q tiles, so reversing the sequence raster retires the
    heaviest work first. A bottom-right offset makes the short-Q context rows
    similarly heavy, while a finite left window bounds their work; both keep
    the ordinary sequence-local raster.
    """

    return mask_type == "causal" and window_left < 0 and not has_q_offset


def _paged_context_uses_clc_scheduler(
    *,
    is_persistent: bool,
    single_qkv_instance: bool,
    is_causal: bool,
    uniform_packed_lengths: bool,
) -> bool:
    """Return whether a persistent paged plan needs the dynamic CLC queue."""

    return is_persistent and (
        not single_qkv_instance or (is_causal and not uniform_packed_lengths)
    )


def _contiguous_context_uses_clc_scheduler(
    *,
    single_qkv_instance: bool,
    head_paired: bool,
    packed_query: bool,
    uniform_packed_lengths: bool,
    is_causal: bool,
    has_q_offset: bool,
) -> bool:
    """Return whether contiguous persistence needs dynamic work discovery.

    Paired and zero-offset triangular task graphs use CLC to distribute their
    heavier work tiles. A bottom-right-offset single-instance fixed or
    uniform-packed plan has a near-uniform immutable domain, so its persistent
    CTAs can advance through the static queue without a CLC request/response
    on every tile. General packed plans retain CLC because live cumulative
    offsets can change the active query-tile domain.
    """

    return (
        head_paired
        or not single_qkv_instance
        or (packed_query and not uniform_packed_lengths)
        or (is_causal and not has_q_offset)
    )


def _contiguous_context_uses_persistent_scheduler(
    *,
    single_qkv_instance: bool,
    head_paired: bool,
    packed_query: bool,
    uniform_packed_lengths: bool,
    logical_work_tiles: int,
    max_active_clusters: int,
    batch_size: int,
    num_qo_heads: int,
    is_causal: bool,
    has_q_offset: bool,
) -> bool:
    """Return whether contiguous work needs a persistent launch.

    Paired, live-ragged, and zero-offset triangular task graphs keep dynamic
    persistence. Near-uniform bottom-right-offset single-instance domains
    launch directly when they fit in one resident wave and use static
    persistence only when more work remains. Oversized logical Y/Z dimensions
    also require the flattened persistent grid.
    """

    return (
        head_paired
        or not single_qkv_instance
        or (packed_query and not uniform_packed_lengths)
        or (is_causal and not has_q_offset)
        or logical_work_tiles > max_active_clusters
        or batch_size > _CUDA_GRID_YZ_MAX
        or num_qo_heads > _CUDA_GRID_YZ_MAX
    )


def _needs_packed_dense_k_mask(
    *,
    packed_query: bool,
    mask_type: str,
    k_lengths: tuple[int, ...],
) -> bool:
    """Return whether packed-contiguous or paged dense K needs right masking.

    Uniform K lengths that are exact K-tile multiples give every request the
    same full-tile domain.  That specialization needs no runtime ``seqlen_k``
    mask; mixed lengths or a partial final tile retain the general mask.
    """

    return (
        packed_query
        and mask_type == "dense"
        and (
            any(length != k_lengths[0] for length in k_lengths[1:])
            or k_lengths[0] % _CONTEXT_KV_TILE_N != 0
        )
    )


def _paged_context_uses_persistent_scheduler(
    *,
    mask_type: str,
    head_paired: bool,
    logical_work_tiles: int,
    max_active_clusters: int,
    batch_size: int,
    num_qo_heads: int,
) -> bool:
    """Return whether paged context should use the static persistent grid.

    A one-wave dense query-paired grid has no inter-wave launch imbalance to
    amortize, so it launches one CTA per logical tile.  Larger grids use the
    persistent scheduler to balance work across resident CTAs.  CUDA limits
    grid Y and Z to 65,535; retaining persistent scheduling for oversized
    public geometries keeps every otherwise-valid int32 plan launchable.
    """

    return (
        mask_type == "causal"
        or head_paired
        or logical_work_tiles > max_active_clusters
        or batch_size > _CUDA_GRID_YZ_MAX
        or num_qo_heads > _CUDA_GRID_YZ_MAX
    )


def _resolve_geometry(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    qo_indptr: Optional[torch.Tensor],
    kv_indptr: Optional[torch.Tensor],
    mask_type: str,
    window_left: int,
    out_dtype: torch.dtype,
) -> _ContextGeometry:
    """Validate one-shot inputs and derive their static wrapper bounds."""

    _validate_base_tensors(q, k, v, allow_mla=True)
    _validate_output_dtype(out_dtype)
    _validate_mask(mask_type)
    window_left = _validate_window_left(window_left, mask_type)
    device_index = _validate_device(q.device)
    device = torch.device("cuda", device_index)
    if (qo_indptr is None) != (kv_indptr is None):
        raise ValueError("qo_indptr and kv_indptr must be provided together")
    packed_query = qo_indptr is not None

    if packed_query:
        if q.ndim != 3 or k.ndim != 3:
            raise ValueError(
                "packed Q/K/V must use [total_tokens, H, D] storage; "
                f"got q rank {q.ndim} and k rank {k.ndim}"
            )
        total_q, num_qo_heads, q_head_dim = map(int, q.shape)
        total_k, num_kv_heads, kv_head_dim = map(int, k.shape)
        _validate_padded_data_extent(total_q, "total_q")
        _validate_padded_data_extent(total_k, "total_k")
        _validate_compact(q, "q", "[total_q, Hq, D]")
        _validate_compact(k, "k", "[total_k, Hkv, D]")
        _validate_compact(v, "v", "[total_k, Hkv, D]")
        assert qo_indptr is not None and kv_indptr is not None
        _validate_indptr_tensor(qo_indptr, "qo_indptr", device=device)
        _validate_indptr_tensor(kv_indptr, "kv_indptr", device=device)
        if qo_indptr.numel() != kv_indptr.numel():
            raise ValueError(
                "qo_indptr and kv_indptr must describe the same batch; "
                f"got {qo_indptr.numel() - 1} and {kv_indptr.numel() - 1} rows"
            )
        _, q_lengths = _read_indptr(qo_indptr, "qo_indptr", expected_total=total_q)
        _, k_lengths = _read_indptr(kv_indptr, "kv_indptr", expected_total=total_k)
        batch_size = len(q_lengths)
        _validate_extent(batch_size, "batch_size")
        max_q_len = max(q_lengths)
        max_seq_len_k = max(k_lengths)
        q_shape = tuple(q.shape)
        kv_shape = tuple(k.shape)
    else:
        if q.ndim != 4 or k.ndim != 4:
            raise ValueError(
                "fixed Q/K/V must use [B, S, H, D] storage; "
                f"got q rank {q.ndim} and k rank {k.ndim}"
            )
        batch_size, max_q_len, num_qo_heads, q_head_dim = map(int, q.shape)
        k_batch, max_seq_len_k, num_kv_heads, kv_head_dim = map(int, k.shape)
        if batch_size != k_batch:
            raise ValueError(
                f"q and k batch dimensions must match; got {batch_size} and {k_batch}"
            )
        _validate_extent(batch_size, "batch_size")
        _validate_extent(max_q_len, "seq_len_q")
        _validate_extent(max_seq_len_k, "seq_len_k")
        total_q = batch_size * max_q_len
        total_k = batch_size * max_seq_len_k
        _validate_padded_data_extent(total_q, "B * seq_len_q")
        _validate_padded_data_extent(total_k, "B * seq_len_k")
        _validate_compact(q, "q", "[B, Sq, Hq, D]")
        _validate_compact(k, "k", "[B, Sk, Hkv, D]")
        _validate_compact(v, "v", "[B, Sk, Hkv, D]")
        q_lengths = (max_q_len,) * batch_size
        k_lengths = (max_seq_len_k,) * batch_size
        q_shape = tuple(q.shape)
        kv_shape = tuple(k.shape)

    _validate_head_dim(q_head_dim, kv_head_dim, int(v.shape[-1]))
    head_ratio = _validate_head_geometry(num_qo_heads, num_kv_heads)
    if mask_type == "causal":
        for batch_idx, (q_length, k_length) in enumerate(
            zip(q_lengths, k_lengths, strict=True)
        ):
            if k_length > 0 and q_length > k_length:
                raise ValueError(
                    "bottom-right causal context requires Sq <= Sk for each "
                    f"request; got batch {batch_idx}: Sq={q_length}, Sk={k_length}"
                )

    if (q_head_dim, int(v.shape[-1])) == (192, 128) and window_left > 0:
        raise NotImplementedError("192/128 context does not support a left window")
    head_paired = window_left > 0
    if head_paired and (head_ratio <= 1 or head_ratio % 2 != 0):
        raise NotImplementedError(
            "a positive left window requires grouped-query attention with an "
            f"even Hq/Hkv ratio greater than one; got {head_ratio}"
        )
    has_q_offset = _derive_has_q_offset(q_lengths, k_lengths, mask_type)
    uniform_packed_lengths = (
        packed_query
        and all(length == q_lengths[0] for length in q_lengths[1:])
        and all(length == k_lengths[0] for length in k_lengths[1:])
    )
    causal_single_kv_tile = (
        mask_type == "causal"
        and not packed_query
        and not head_paired
        and max_q_len == max_seq_len_k
        and max_seq_len_k <= 128
    )
    packed_dense_k_mask = _needs_packed_dense_k_mask(
        packed_query=packed_query,
        mask_type=mask_type,
        k_lengths=k_lengths,
    )
    return _ContextGeometry(
        device=device,
        device_index=device_index,
        packed_query=packed_query,
        batch_size=batch_size,
        total_q=total_q,
        total_k=total_k,
        max_q_len=max_q_len,
        max_seq_len_k=max_seq_len_k,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=q_head_dim,
        head_dim_vo=int(v.shape[-1]),
        qk_dtype=q.dtype,
        pv_dtype=v.dtype,
        out_dtype=out_dtype,
        mask_type=mask_type,
        window_left=window_left,
        head_paired=head_paired,
        uniform_packed_lengths=uniform_packed_lengths,
        has_q_offset=has_q_offset,
        causal_single_kv_tile=causal_single_kv_tile,
        packed_dense_k_mask=packed_dense_k_mask,
        q_shape=q_shape,
        kv_shape=kv_shape,
    )


def _resolve_context_plan_geometry(
    *,
    device: int | str | torch.device,
    batch_size: int,
    max_q_len: int,
    max_kv_len: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    head_dim_vo: int | None = None,
    qk_dtype: torch.dtype,
    pv_dtype: torch.dtype,
    packed_query: bool,
    mask_type: str,
    window_left: int,
    out_dtype: torch.dtype,
    sage: SageAttentionConfig | None = None,
) -> _ContextPlanGeometry:
    """Validate explicit static bounds for a reusable contiguous plan."""

    _validate_context_dtype_pair(qk_dtype, qk_dtype, pv_dtype, out_dtype)
    _validate_mask(mask_type)
    window_left = _validate_window_left(window_left, mask_type)
    if not isinstance(packed_query, bool):
        raise TypeError("packed_query must be a bool")
    if packed_query and mask_type == "variable_window":
        raise NotImplementedError(
            "mask_type='variable_window' requires fixed [B, S, H, D] tensors"
        )

    batch_size = _validate_static_extent(batch_size, "batch_size")
    max_q_len = _validate_static_extent(max_q_len, "max_q_len")
    max_kv_len = _validate_static_extent(max_kv_len, "max_kv_len")
    num_qo_heads = _validate_static_extent(num_qo_heads, "num_qo_heads")
    num_kv_heads = _validate_static_extent(num_kv_heads, "num_kv_heads")
    head_dim = _validate_static_extent(head_dim, "head_dim")
    head_dim_vo = (
        head_dim
        if head_dim_vo is None
        else _validate_static_extent(head_dim_vo, "head_dim_vo")
    )
    device, device_index = _resolve_cuda_device(device)

    _validate_padded_data_extent(batch_size * max_q_len, "batch_size * max_q_len")
    _validate_padded_data_extent(batch_size * max_kv_len, "batch_size * max_kv_len")
    head_ratio = _validate_head_geometry(num_qo_heads, num_kv_heads)
    _validate_head_dim(head_dim, head_dim, head_dim_vo)
    if (head_dim, head_dim_vo) == (192, 128) and window_left > 0:
        raise NotImplementedError(
            "QK/V=(192, 128) does not support a positive left window"
        )
    if not packed_query and mask_type == "causal" and max_q_len > max_kv_len:
        raise ValueError(
            "bottom-right causal context requires max_q_len <= max_kv_len; "
            f"got {max_q_len} and {max_kv_len}"
        )
    head_paired = window_left > 0
    if head_paired and (head_ratio <= 1 or head_ratio % 2 != 0):
        raise NotImplementedError(
            "a positive left window requires grouped-query attention with an "
            f"even Hq/Hkv ratio greater than one; got {head_ratio}"
        )

    # Packed request lengths are run inputs, so select conservative compile
    # facts that remain valid for every request within the plan capacities.
    has_q_offset = mask_type == "causal" and (packed_query or max_q_len != max_kv_len)
    geometry = _ContextPlanGeometry(
        device=device,
        device_index=device_index,
        packed_query=packed_query,
        batch_size=batch_size,
        max_q_len=max_q_len,
        max_seq_len_k=max_kv_len,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        head_dim_vo=head_dim_vo,
        qk_dtype=qk_dtype,
        pv_dtype=pv_dtype,
        out_dtype=out_dtype,
        mask_type=mask_type,
        window_left=window_left,
        head_paired=head_paired,
        uniform_packed_lengths=False,
        has_q_offset=has_q_offset,
        causal_single_kv_tile=(
            mask_type == "causal"
            and not packed_query
            and not head_paired
            and max_q_len == max_kv_len
            and max_kv_len <= _CONTEXT_KV_TILE_N
        ),
        packed_dense_k_mask=packed_query and mask_type == "dense",
        sage=sage,
    )
    if sage is not None:
        _validate_context_sage_config(geometry, sage)
    elif qk_dtype == torch.int8:
        raise NotImplementedError(
            "int8 Q/K require a Sage recipe (sage=SageAttentionConfig(...)) that "
            "dequantizes them"
        )
    # The recipe is rejected before the device, as in block-sparse plans.
    _validate_int8_qk_device(qk_dtype, device_index)
    return geometry


def _resolve_paged_geometry(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    qo_indptr: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens_kv: torch.Tensor,
    page_size: int,
    mask_type: str,
    window_left: int,
    out_dtype: torch.dtype,
) -> _PagedContextPlanGeometry:
    """Validate one fixed-table paged launch and derive its plan bounds."""

    _validate_mask(mask_type)
    if mask_type == "variable_window":
        raise NotImplementedError(
            "mask_type='variable_window' is not supported for paged context"
        )
    _validate_base_tensors(q, k_cache, v_cache)
    window_left = _validate_window_left(window_left, mask_type)
    page_size = _validate_page_size(page_size)
    device_index = _validate_device(q.device)
    device = torch.device("cuda", device_index)

    if q.ndim != 3:
        raise ValueError(
            "paged context Q must use packed [total_q, Hq, D] storage; "
            f"got rank {q.ndim}"
        )
    if k_cache.ndim != 4:
        raise ValueError(
            "paged K/V caches must use [num_pages, Hkv, page_size, D] "
            f"storage; got rank {k_cache.ndim}"
        )
    total_q, num_qo_heads, q_head_dim = map(int, q.shape)
    num_physical_pages, num_kv_heads, cache_page_size, kv_head_dim = map(
        int, k_cache.shape
    )
    _validate_padded_data_extent(total_q, "total_q")
    _validate_extent(num_physical_pages, "num_physical_pages")
    if cache_page_size != page_size:
        raise ValueError(
            f"K/V cache page extent must equal page_size={page_size}; "
            f"got {cache_page_size}"
        )
    _validate_head_dim(q_head_dim, kv_head_dim)
    _validate_compact(q, "q", "[total_q, Hq, D]")
    _validate_native_hnd_tma_strides(k_cache, "k_cache")
    _validate_native_hnd_tma_strides(v_cache, "v_cache")

    _validate_indptr_tensor(qo_indptr, "qo_indptr", device=device)
    batch_size = int(qo_indptr.numel()) - 1
    _validate_extent(batch_size, "batch_size")
    _, q_lengths = _read_indptr(qo_indptr, "qo_indptr", expected_total=total_q)
    _validate_paged_metadata_tensor(seq_lens_kv, "seq_lens_kv", device=device)
    k_lengths = _read_int32_values(
        seq_lens_kv,
        "seq_lens_kv",
        expected_count=batch_size,
    )
    for batch_idx, k_length in enumerate(k_lengths):
        if k_length < 0:
            raise ValueError(
                "seq_lens_kv entries must be nonnegative; "
                f"batch {batch_idx} has {k_length}"
            )
    max_q_len = max(q_lengths)
    max_kv_len = max(k_lengths)
    uniform_packed_lengths = all(length == max_q_len for length in q_lengths) and all(
        length == max_kv_len for length in k_lengths
    )
    has_q_offset = _derive_has_q_offset(q_lengths, k_lengths, mask_type)

    geometry = _resolve_paged_plan_geometry(
        device=device,
        batch_size=batch_size,
        max_q_len=max_q_len,
        max_kv_len=max_kv_len,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=q_head_dim,
        qk_dtype=q.dtype,
        pv_dtype=v_cache.dtype,
        page_size=page_size,
        mask_type=mask_type,
        window_left=window_left,
        out_dtype=out_dtype,
        uniform_packed_lengths=uniform_packed_lengths,
        has_q_offset=has_q_offset,
    )
    _validate_paged_runtime_inputs(q, k_cache, v_cache, geometry)
    _validate_paged_runtime_metadata(
        qo_indptr,
        block_tables,
        seq_lens_kv,
        geometry,
        total_q=total_q,
        num_physical_pages=num_physical_pages,
    )
    return geometry


def _make_context_scheduler_probe(
    geometry: _ContextPlanGeometry | _PagedContextPlanGeometry,
    *,
    causal_single_kv_tile: bool,
    page_size: int | None = None,
    max_kv_len: int | None = None,
) -> _ContextSchedulerProbe:
    """Build the common static probe and logical work count for policy."""

    import cutlass

    dtype_map = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float8_e4m3fn: cutlass.Float8E4M3FN,
        torch.int8: cutlass.Int8,
    }
    probe = _make_context_kernel(
        device_index=geometry.device_index,
        input_qk_dtype=dtype_map[geometry.qk_dtype],
        input_pv_dtype=dtype_map[geometry.pv_dtype],
        out_dtype=dtype_map[geometry.out_dtype],
        head_dim=geometry.head_dim,
        head_dim_vo=getattr(geometry, "head_dim_vo", None),
        mask_type=geometry.mask_type,
        window_left=geometry.window_left,
        head_paired=geometry.head_paired,
        num_qo_heads=geometry.num_qo_heads,
        num_kv_heads=geometry.num_kv_heads,
        has_q_offset=geometry.has_q_offset,
        causal_single_kv_tile=causal_single_kv_tile,
        scheduler="static_persistent",
        uses_ldtm_stat=_default_uses_ldtm_stat(
            geometry.device_index,
            geometry.qk_dtype.itemsize * 8,
            geometry.pv_dtype.itemsize * 8,
        ),
        page_size=page_size,
        max_kv_len=max_kv_len,
    )
    with torch.cuda.device(geometry.device_index):
        max_active_clusters = get_max_active_clusters(1)
    num_seq_tiles = (
        geometry.max_q_len + probe.cfg.cta_tiler[0] - 1
    ) // probe.cfg.cta_tiler[0]
    num_head_tiles = geometry.num_qo_heads // probe.cfg.work_tile_q_heads
    logical_work_tiles = geometry.batch_size * num_seq_tiles * num_head_tiles
    return _ContextSchedulerProbe(
        single_qkv_instance=bool(probe.cfg.single_qkv_instance),
        logical_work_tiles=int(logical_work_tiles),
        max_active_clusters=max_active_clusters,
    )


def _resolve_paged_plan_geometry(
    *,
    device: int | str | torch.device,
    batch_size: int,
    max_q_len: int,
    max_kv_len: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    qk_dtype: torch.dtype,
    pv_dtype: torch.dtype,
    page_size: int,
    mask_type: str,
    window_left: int,
    out_dtype: torch.dtype,
    uniform_packed_lengths: bool = False,
    has_q_offset: bool = True,
    paged_v_tail_is_zero: bool = False,
) -> _PagedContextPlanGeometry:
    """Validate explicit static bounds for a reusable paged specialization."""

    _validate_paged_dtype_pair(qk_dtype, qk_dtype, pv_dtype, out_dtype)
    _validate_mask(mask_type)
    if not isinstance(uniform_packed_lengths, bool):
        raise TypeError("uniform_packed_lengths must be a bool")
    if not isinstance(has_q_offset, bool):
        raise TypeError("has_q_offset must be a bool")
    if not isinstance(paged_v_tail_is_zero, bool):
        raise TypeError("paged_v_tail_is_zero must be a bool")
    if mask_type == "variable_window":
        raise NotImplementedError(
            "mask_type='variable_window' is not supported for paged context"
        )
    window_left = _validate_window_left(window_left, mask_type)
    page_size = _validate_page_size(page_size)
    batch_size = _validate_static_extent(batch_size, "batch_size")
    max_q_len = _validate_static_extent(max_q_len, "max_q_len")
    max_kv_len = _validate_static_extent(max_kv_len, "max_kv_len")
    num_qo_heads = _validate_static_extent(num_qo_heads, "num_qo_heads")
    num_kv_heads = _validate_static_extent(num_kv_heads, "num_kv_heads")
    head_dim = _validate_static_extent(head_dim, "head_dim")
    device, device_index = _resolve_cuda_device(device)

    _validate_padded_data_extent(batch_size * max_q_len, "batch_size * max_q_len")
    head_ratio = _validate_head_geometry(num_qo_heads, num_kv_heads)
    _validate_head_dim(head_dim, head_dim)
    head_paired = window_left > 0
    if head_paired and (head_ratio <= 1 or head_ratio % 2 != 0):
        raise NotImplementedError(
            "a positive left window requires grouped-query attention with an "
            f"even Hq/Hkv ratio greater than one; got {head_ratio}"
        )

    # A causal Q offset is not part of dense attention semantics. Canonicalize
    # it out of the compile identity even when the conservative public default
    # was used, so dense callers keep one cache entry and the leaner kernel.
    has_q_offset = mask_type == "causal" and has_q_offset

    return _PagedContextPlanGeometry(
        device=device,
        device_index=device_index,
        batch_size=batch_size,
        max_q_len=max_q_len,
        max_kv_len=max_kv_len,
        page_size=page_size,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        qk_dtype=qk_dtype,
        pv_dtype=pv_dtype,
        out_dtype=out_dtype,
        mask_type=mask_type,
        window_left=window_left,
        head_paired=head_paired,
        uniform_packed_lengths=uniform_packed_lengths,
        has_q_offset=has_q_offset,
        paged_v_tail_is_zero=paged_v_tail_is_zero,
        packed_dense_k_mask=(
            mask_type == "dense"
            and (not uniform_packed_lengths or max_kv_len % _CONTEXT_KV_TILE_N != 0)
        ),
    )


def _two_cta_umma_geometry_eligible(geometry: _ContextPlanGeometry) -> bool:
    """Dense contiguous MHA or GQA at D=128 with bf16 QK runs the two-CTA UMMA
    form, which pairs adjacent Q tiles of one head through the grid.
    The two-CTA launch is non-persistent with heads on grid Y and batch on
    grid Z. CUDA limits grid Y and Z to 65,535. Oversized geometries keep the
    persistent flattened grid so the valid int32 plans stay launchable."""
    # VC-Attention-QK16 always runs two-CTA. With P in SMEM, the bf16 K ring,
    # tile-mean ring and row-sum operands leave room for one K/V stage per CTA
    # at one CTA and three in the halved form. The VC tail needs two.
    # VC-Attention-QK8 keeps the E4M3 K ring and follows the device default.
    vc_qk16 = geometry.vc is not None and geometry.qk_dtype != torch.float8_e4m3fn
    two_cta = _default_two_cta_umma(
        geometry.device_index, geometry.qk_dtype.itemsize * 8
    )
    if two_cta and torch.cuda.get_device_capability(geometry.device_index) == (10, 7):
        # Pairs are coarser work units. Keep them only where they fill their
        # waves at least as well as single CTAs fill theirs.
        sms = torch.cuda.get_device_properties(
            geometry.device_index
        ).multi_processor_count
        tiles = math.ceil(geometry.max_q_len / (2 * _CONTEXT_TILE_SIZE_Q))
        heads = geometry.batch_size * geometry.num_qo_heads
        single_waves = heads * tiles / sms
        pair_waves = heads * math.ceil(tiles / 2) / (sms // 2)
        two_cta = pair_waves / math.ceil(pair_waves) >= single_waves / math.ceil(
            single_waves
        )
    return (
        (two_cta or vc_qk16)
        and geometry.head_dim == 128
        and geometry.head_dim_vo in (None, 128)
        and geometry.mask_type == "dense"
        and not geometry.packed_query
        and not geometry.head_paired
        and geometry.qk_dtype in (torch.float16, torch.bfloat16, torch.float8_e4m3fn)
        and not (
            geometry.batch_size > _CUDA_GRID_YZ_MAX
            or geometry.num_qo_heads > _CUDA_GRID_YZ_MAX
        )
    )


def _resolve_context_scheduler(geometry: _ContextPlanGeometry) -> _ContextScheduler:
    """Select a scheduler from plan-time work while keeping batch out of JIT."""

    probe = _make_context_scheduler_probe(
        geometry,
        causal_single_kv_tile=geometry.causal_single_kv_tile,
    )
    if _two_cta_umma_geometry_eligible(geometry):
        # Two-CTA pairs Q tiles through the grid, so no persistent scheduler.
        return "nonpersistent"
    is_persistent = _contiguous_context_uses_persistent_scheduler(
        single_qkv_instance=probe.single_qkv_instance,
        head_paired=geometry.head_paired,
        packed_query=geometry.packed_query,
        uniform_packed_lengths=geometry.uniform_packed_lengths,
        logical_work_tiles=probe.logical_work_tiles,
        max_active_clusters=probe.max_active_clusters,
        batch_size=geometry.batch_size,
        num_qo_heads=geometry.num_qo_heads,
        is_causal=geometry.mask_type == "causal",
        has_q_offset=geometry.has_q_offset,
    )
    if not is_persistent:
        return "nonpersistent"
    # Fixed causal requests with one work item per head all traverse the same
    # KV domain. Static persistence avoids CLC traffic without a load-balance
    # penalty; live packed offsets and multiple Q work items still use CLC.
    if (
        geometry.mask_type == "causal"
        and not geometry.packed_query
        and not geometry.head_paired
        and geometry.window_left <= 0
        and probe.logical_work_tiles == geometry.batch_size * geometry.num_qo_heads
    ):
        return "static_persistent"
    if _contiguous_context_uses_clc_scheduler(
        single_qkv_instance=probe.single_qkv_instance,
        head_paired=geometry.head_paired,
        packed_query=geometry.packed_query,
        uniform_packed_lengths=geometry.uniform_packed_lengths,
        is_causal=geometry.mask_type == "causal",
        has_q_offset=geometry.has_q_offset,
    ):
        return "clc_dynamic_persistent"
    return "static_persistent"


def _resolve_paged_context_scheduler(
    geometry: _PagedContextPlanGeometry,
) -> _ContextScheduler:
    """Select paged-context scheduling from plan work, not JIT identity."""

    probe = _make_context_scheduler_probe(
        geometry,
        causal_single_kv_tile=False,
        page_size=geometry.page_size,
        max_kv_len=geometry.max_kv_len,
    )
    is_persistent = _paged_context_uses_persistent_scheduler(
        mask_type=geometry.mask_type,
        head_paired=geometry.head_paired,
        logical_work_tiles=probe.logical_work_tiles,
        max_active_clusters=probe.max_active_clusters,
        batch_size=geometry.batch_size,
        num_qo_heads=geometry.num_qo_heads,
    )
    if not is_persistent:
        return "nonpersistent"
    if _paged_context_uses_clc_scheduler(
        is_persistent=True,
        single_qkv_instance=probe.single_qkv_instance,
        is_causal=geometry.mask_type == "causal",
        uniform_packed_lengths=geometry.uniform_packed_lengths,
    ):
        return "clc_dynamic_persistent"
    return "static_persistent"


def _fixed_kv_rows(max_seq_len_k: int, vc: Optional[VCAttentionConfig]) -> int:
    """K/V rows of a fixed plan. V repair adds its tiles to the sequence."""
    if vc is not None and vc.repair_tiles:
        return vc_repair_kv_len(max_seq_len_k, vc.repair_tiles)
    return max_seq_len_k


def _validate_context_sage_config(
    geometry: _ContextPlanGeometry, sage: SageAttentionConfig
) -> None:
    """Reject a Sage recipe the contiguous context kernel does not run.

    Beyond the recipe every wrapper checks (``validate_sage_recipe``), Sage
    attention covers the dense, fixed-length, query-paired D128 path.
    """

    validate_sage_recipe(
        sage,
        q_dtype=geometry.qk_dtype,
        k_dtype=geometry.qk_dtype,
        v_dtype=geometry.pv_dtype,
        output_dtype=geometry.out_dtype,
    )
    if geometry.mask_type != "dense" or geometry.packed_query or geometry.head_paired:
        raise NotImplementedError(
            "Sage attention requires a dense, fixed-length, query-paired context plan"
        )
    if geometry.head_dim != 128 or geometry.head_dim_vo not in (None, 128):
        raise NotImplementedError("Sage attention requires head_dim 128")


def _context_sage_layout(geometry: _ContextPlanGeometry) -> _SageScaleLayout:
    """Return the prepared scale layout of a contiguous context Sage plan.

    The Q rows hold one Q tile row per token of one Q head (``_SageQSection``
    without interleaved heads): the query-paired D128 tile.
    """

    sage = geometry.sage
    return _SageScaleLayout(
        k_tokens=_SageKSection(
            length=geometry.max_seq_len_k,
            block_size=sage.k_block_size,
            source_block_size=sage.k_block_size,
            tile_atoms=_CONTEXT_KV_TILE_N // _SAGE_ATOM_TOKENS,
        ),
        q_rows=_SageQSection(
            seq_len_q=geometry.max_q_len,
            q_block_size=sage.q_block_size,
            heads_q_per_kv=geometry.num_qo_heads // geometry.num_kv_heads,
            rows_per_tile=_CONTEXT_TILE_SIZE_Q,
            heads_interleaved=False,
        ),
        v_rows=_SageVSection(headdim=geometry.head_dim, has_mean=sage.v_mean),
    )


def _build_context_sage_state(geometry: _ContextPlanGeometry) -> SagePreparedScales:
    """Allocate a plan's prepared scale buffer and compile its prepare launch."""

    return build_sage_prepared_scales(
        _context_sage_layout(geometry),
        geometry.sage,
        batch_size=geometry.batch_size,
        seq_len_q=geometry.max_q_len,
        seq_len_kv=geometry.max_seq_len_k,
        num_qo_heads=geometry.num_qo_heads,
        num_kv_heads=geometry.num_kv_heads,
        head_dim=geometry.head_dim,
        device=geometry.device,
        device_index=geometry.device_index,
    )


@functools.cache
def _context_compile_spec(
    geometry: _ContextPlanGeometry,
    *,
    runtime_lengths: bool | None = None,
    lse_mode: str = "none",
    use_attention_sinks: bool = False,
    cuda_scales: bool = False,
) -> _ContextCompileSpec:
    """Keep resource/precision choices static and ordinary lengths runtime."""
    fixed_seqlen = geometry.sage is not None or geometry.vc is not None
    runtime_lengths = not fixed_seqlen
    shape_signature = (
        () if runtime_lengths else (geometry.max_q_len, geometry.max_seq_len_k)
    )
    return _ContextCompileSpec(
        device_index=geometry.device_index,
        max_q_len=geometry.max_q_len if fixed_seqlen else None,
        max_seq_len_k=geometry.max_seq_len_k if fixed_seqlen else None,
        num_qo_heads=geometry.num_qo_heads,
        num_kv_heads=geometry.num_kv_heads,
        head_dim=geometry.head_dim,
        head_dim_vo=geometry.head_dim_vo,
        qk_dtype_key=_dtype_key(geometry.qk_dtype),
        pv_dtype_key=_dtype_key(geometry.pv_dtype),
        out_dtype_key=_dtype_key(geometry.out_dtype),
        mask_type=geometry.mask_type,
        window_left=geometry.window_left,
        packed_query=geometry.packed_query,
        head_paired=geometry.head_paired,
        uniform_packed_lengths=geometry.uniform_packed_lengths and fixed_seqlen,
        has_q_offset=geometry.has_q_offset,
        causal_single_kv_tile=geometry.causal_single_kv_tile,
        packed_dense_k_mask=(
            geometry.packed_dense_k_mask
            if fixed_seqlen
            else geometry.mask_type != "causal"
        ),
        shape_signature=shape_signature,
        runtime_lengths=runtime_lengths,
        lse_mode=lse_mode,
        use_attention_sinks=use_attention_sinks,
        cuda_scales=cuda_scales,
        scheduler=_resolve_context_scheduler(geometry),
        two_cta_umma=_two_cta_umma_geometry_eligible(geometry),
        vc=geometry.vc,
        sage_scale_layout=(
            _context_sage_layout(geometry) if geometry.sage is not None else None
        ),
    )


@functools.cache
def _paged_context_compile_spec(
    geometry: _PagedContextPlanGeometry,
    *,
    runtime_lengths: bool = False,
    lse_mode: str = "none",
    use_attention_sinks: bool = False,
) -> _PagedContextCompileSpec:
    """Build the paged compile key, including separate QK and PV dtype keys."""
    return _PagedContextCompileSpec(
        device_index=geometry.device_index,
        max_q_len=None,
        max_kv_len=None,
        page_size=geometry.page_size,
        num_qo_heads=geometry.num_qo_heads,
        num_kv_heads=geometry.num_kv_heads,
        head_dim=geometry.head_dim,
        qk_dtype_key=_dtype_key(geometry.qk_dtype),
        pv_dtype_key=_dtype_key(geometry.pv_dtype),
        out_dtype_key=_dtype_key(geometry.out_dtype),
        mask_type=geometry.mask_type,
        window_left=geometry.window_left,
        head_paired=geometry.head_paired,
        uniform_packed_lengths=False,
        has_q_offset=geometry.has_q_offset,
        paged_v_tail_is_zero=geometry.paged_v_tail_is_zero,
        packed_dense_k_mask=(geometry.mask_type != "causal"),
        shape_signature=(),
        runtime_lengths=True,
        lse_mode=lse_mode,
        use_attention_sinks=use_attention_sinks,
        scheduler=_resolve_paged_context_scheduler(geometry),
    )


@functools.cache
def _get_compiled_context(
    compile_spec: _ContextCompileSpec,
):
    """Compile one batch-dynamic context specialization and cache by topology."""

    device_index = compile_spec.device_index
    # Only Sage and VC plans keep exact lengths in the key; they compile static shapes.
    fixed_seqlen = compile_spec.max_q_len is not None
    max_q_len = compile_spec.max_q_len or 0
    max_seq_len_k = compile_spec.max_seq_len_k or 0
    num_qo_heads = compile_spec.num_qo_heads
    num_kv_heads = compile_spec.num_kv_heads
    head_dim = compile_spec.head_dim
    head_dim_vo = compile_spec.head_dim_vo or head_dim
    qk_dtype_key = compile_spec.qk_dtype_key
    pv_dtype_key = compile_spec.pv_dtype_key
    out_dtype_key = compile_spec.out_dtype_key
    mask_type = compile_spec.mask_type
    window_left = compile_spec.window_left
    packed_query = compile_spec.packed_query
    head_paired = compile_spec.head_paired
    uniform_packed_lengths = compile_spec.uniform_packed_lengths
    has_q_offset = compile_spec.has_q_offset
    causal_single_kv_tile = compile_spec.causal_single_kv_tile
    packed_dense_k_mask = compile_spec.packed_dense_k_mask
    scheduler = compile_spec.scheduler
    two_cta_umma = compile_spec.two_cta_umma
    vc_attention = compile_spec.vc is not None
    sage_scale_layout = compile_spec.sage_scale_layout
    vc_qk8 = vc_attention and qk_dtype_key == "float8_e4m3fn"

    import cutlass
    import cutlass.cute as cute
    from cuda.bindings import driver as cuda_drv

    dtype_map = {
        "float16": cutlass.Float16,
        "bfloat16": cutlass.BFloat16,
        "float8_e4m3fn": cutlass.Float8E4M3FN,
        "int8": cutlass.Int8,
    }
    input_qk_dtype = dtype_map[qk_dtype_key]
    input_pv_dtype = dtype_map[pv_dtype_key]
    out_dtype = dtype_map[out_dtype_key]
    has_variable_window = mask_type == "variable_window"
    with torch.cuda.device(device_index):
        max_active_clusters = get_max_active_clusters(1)
    fmha = _make_context_kernel(
        device_index=device_index,
        input_qk_dtype=input_qk_dtype,
        input_pv_dtype=input_pv_dtype,
        out_dtype=out_dtype,
        head_dim=head_dim,
        head_dim_vo=head_dim_vo,
        mask_type=mask_type,
        window_left=window_left,
        head_paired=head_paired,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        has_q_offset=has_q_offset,
        causal_single_kv_tile=causal_single_kv_tile,
        scheduler=scheduler,
        two_cta_umma=two_cta_umma,
        uses_ldtm_stat=_default_uses_ldtm_stat(
            device_index, input_qk_dtype.width, input_pv_dtype.width
        ),
        fp8_psmem_early_token=_default_fp8_psmem_early_token(device_index),
        kv_ring_stages=_default_kv_ring_stages(
            device_index, input_qk_dtype.width, input_pv_dtype.width
        ),
        umma_fp8_k64=_default_umma_fp8_k64(device_index, input_qk_dtype.width),
        vc_attention=("means" if compile_spec.vc.demean else "outliers")
        if vc_attention
        else None,
        sage_scale_layout=sage_scale_layout,
        # The QK8 K-scale table is sized from the K/V rows of the fixed plan.
        max_kv_len=_fixed_kv_rows(max_seq_len_k, compile_spec.vc) if vc_qk8 else None,
    )
    if vc_attention and torch.cuda.get_device_capability(device_index) == (10, 7):
        raise ValueError(
            "vc_config is not supported on SM107, where the dense fp8 kernel runs "
            "the softmax exp2 natively"
        )
    fmha.cfg.lse_mode = compile_spec.lse_mode
    fmha.cfg.use_attention_sinks = compile_spec.use_attention_sinks
    fmha.cfg.has_varlen = packed_query
    fmha.cfg.has_uniform_varlen = uniform_packed_lengths
    if vc_qk8:
        # The QK8 scale tables are addressed with the fixed plan lengths, the K/V
        # rows including the repair tiles.
        fmha.cfg.uniform_seq_len_q = max_q_len
        fmha.cfg.uniform_seq_len_k = _fixed_kv_rows(max_seq_len_k, compile_spec.vc)
    if uniform_packed_lengths:
        fmha.cfg.uniform_seq_len_q = max_q_len
        fmha.cfg.uniform_seq_len_k = max_seq_len_k
    fmha.cfg.has_q_offset = has_q_offset
    fmha.cfg.exp2_fma_pairs = _default_exp2_fma_pairs(device_index, fmha.cfg)
    fmha.cfg.exp2_packed_f16x2 = _default_exp2_packed_f16x2(device_index, fmha.cfg)
    if fmha.cfg.kv_tile_n != _CONTEXT_KV_TILE_N:
        raise RuntimeError(
            "context packed-K specialization assumes kv_tile_n="
            f"{_CONTEXT_KV_TILE_N}, got {fmha.cfg.kv_tile_n}"
        )
    _validate_query_work_tile_span(fmha.cfg)
    fmha.cfg.packed_dense_k_mask = packed_dense_k_mask
    if mask_type != "causal" and not packed_query:
        fmha.cfg.fixed_dense_k_tail = (
            max_seq_len_k % fmha.cfg.kv_tile_n if fixed_seqlen else fmha.cfg.kv_tile_n
        )

    # Compile both bounded and general masks once; choose from runtime K.
    fmha.runtime_dense_tail = (
        not fixed_seqlen and not packed_query and mask_type == "dense"
    )

    @cute.jit
    def tensor_adapter(
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        out: cute.Tensor,
        lse: cute.Tensor | None,
        attention_sinks: cute.Tensor | None,
        scale_softmax_log2: cutlass.Float32 | cute.Tensor,
        output_scale: cutlass.Float32 | cute.Tensor,
        qo_indptr: cute.Tensor,
        kv_indptr: cute.Tensor,
        variable_window_token_starts: cute.Tensor,
        variable_window_token_ends: cute.Tensor,
        variable_window_cta_starts: cute.Tensor,
        vc_mu: cute.Tensor | None,
        vc_q_scale: cute.Tensor | None,
        vc_k_scale: cute.Tensor | None,
        sage_scales: cute.Tensor | None,
        stream: cuda_drv.CUstream,
        static_max_active_clusters: cutlass.Constexpr[int],
        static_packed_query: cutlass.Constexpr[bool],
        max_q_len: cutlass.Int32,
        runtime_max_seq_len_k: cutlass.Int32,
        static_vc_repair_tiles: cutlass.Constexpr[int],
    ) -> None:
        """Adapt torch TVM-FFI tensors to the FmhaTs host entry point."""

        if cutlass.const_expr(static_packed_query):
            fmha(
                q,
                k,
                v,
                out,
                scale_softmax_log2,
                output_scale,
                static_max_active_clusters,
                stream,
                qo_indptr,
                kv_indptr,
                max_q_len,
                runtime_max_seq_len_k,
                variable_window_token_starts=variable_window_token_starts,
                variable_window_token_ends=variable_window_token_ends,
                variable_window_cta_starts=variable_window_cta_starts,
                vc_mu=vc_mu,
                vc_q_scale=vc_q_scale,
                vc_k_scale=vc_k_scale,
                vc_repair_tiles=static_vc_repair_tiles,
                lse_output=lse,
                attention_sinks=attention_sinks,
            )
        else:
            fmha(
                q,
                k,
                v,
                out,
                scale_softmax_log2,
                output_scale,
                static_max_active_clusters,
                stream,
                variable_window_token_starts=variable_window_token_starts,
                variable_window_token_ends=variable_window_token_ends,
                variable_window_cta_starts=variable_window_cta_starts,
                vc_mu=vc_mu,
                vc_q_scale=vc_q_scale,
                vc_k_scale=vc_k_scale,
                vc_repair_tiles=static_vc_repair_tiles,
                lse_output=lse,
                attention_sinks=attention_sinks,
                sage_scales=sage_scales,
            )

    # Describe row-major symbolic extents; assumed_align is in bytes.
    def fake_compact(dtype, shape, assumed_align):
        return cute.runtime.make_fake_compact_tensor(
            dtype,
            shape,
            stride_order=tuple(reversed(range(len(shape)))),
            assumed_align=assumed_align,
        )

    q_shape: tuple[object, ...]
    kv_shape: tuple[object, ...]
    out_shape: tuple[object, ...]
    if packed_query:
        runtime_total_q = cute.sym_int()
        runtime_total_k = cute.sym_int()
        runtime_num_q_offsets = cute.sym_int()
        runtime_num_k_offsets = cute.sym_int()
        q_shape = (runtime_total_q, num_qo_heads, head_dim)
        kv_shape = (runtime_total_k, num_kv_heads, head_dim)
        out_shape = (runtime_total_q, num_qo_heads, head_dim_vo)
        qo_indptr_shape = (runtime_num_q_offsets,)
        kv_indptr_shape = (runtime_num_k_offsets,)
    else:
        batch_size = cute.sym_int()
        physical_sq = cute.sym_int() if compile_spec.runtime_lengths else max_q_len
        physical_sk = (
            cute.sym_int()
            if compile_spec.runtime_lengths
            else _fixed_kv_rows(max_seq_len_k, compile_spec.vc)
        )
        q_shape = (batch_size, physical_sq, num_qo_heads, head_dim)
        kv_shape = (batch_size, physical_sk, num_kv_heads, head_dim)
        out_shape = (*q_shape[:-1], head_dim_vo)
        qo_indptr_shape = (1,)
        kv_indptr_shape = (1,)
    q_fake = fake_compact(input_qk_dtype, q_shape, 16)
    k_fake = fake_compact(input_qk_dtype, kv_shape, 16)
    v_fake = fake_compact(input_pv_dtype, (*kv_shape[:-1], head_dim_vo), 16)
    out_fake = fake_compact(out_dtype, out_shape, 16)
    lse_fake = (
        fake_compact(cutlass.Float32, out_fake.shape[:-1], 4)
        if compile_spec.lse_mode != "none"
        else None
    )
    scale_fake = cutlass.Float32(1.0)
    output_scale_fake = cutlass.Float32(1.0)
    if compile_spec.cuda_scales:
        scale_fake = fake_compact(cutlass.Float32, (1,), 4)
        output_scale_fake = fake_compact(cutlass.Float32, (1,), 4)
    if vc_attention:
        output_scale_fake = fake_compact(
            cutlass.Float32, (batch_size, num_qo_heads, head_dim_vo), 64
        )
    qo_indptr_fake = fake_compact(cutlass.Int32, qo_indptr_shape, 4)
    kv_indptr_fake = fake_compact(cutlass.Int32, kv_indptr_shape, 4)
    if has_variable_window:
        if packed_query:
            raise RuntimeError("variable-window context requires fixed tensors")
        variable_window_shape = (batch_size * q_shape[1],)
        variable_window_tile_size_q = _variable_window_tile_size_q(head_dim)
        variable_window_cta_shape = (
            (cute.sym_int(),)
            if not fixed_seqlen
            else (batch_size * cute.ceil_div(max_q_len, variable_window_tile_size_q),)
        )
    else:
        variable_window_shape = (1,)
        variable_window_cta_shape = (1,)
    variable_window_starts_fake = fake_compact(cutlass.Int32, variable_window_shape, 4)
    variable_window_ends_fake = fake_compact(cutlass.Int32, variable_window_shape, 4)
    variable_window_cta_starts_fake = fake_compact(
        cutlass.Int32, variable_window_cta_shape, 4
    )
    vc_mu_fake = None
    vc_q_scale_fake = None
    vc_k_scale_fake = None
    if vc_attention:
        if packed_query:
            raise RuntimeError("VC-Attention context requires fixed tensors")
        # The plan fixes every VC operand shape, so the kernel addresses the
        # tile means and the Q and K scale tables with compile-time extents.
        vc_shapes = vc_scale_shapes(
            compile_spec.vc,
            batch_size=batch_size,
            seq_len_kv=max_seq_len_k,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim_vo or head_dim,
            seq_len_q=max_q_len,
            num_qo_heads=num_qo_heads,
            qk_fp8=vc_qk8,
        )
        if compile_spec.vc.demean:
            vc_mu_fake = fake_compact(cutlass.BFloat16, vc_shapes["tile_means"], 16)
        if vc_qk8:
            # The kernel reads the Q and K scale tables as one row of blocks per head.
            vc_q_scale_fake = fake_compact(
                cutlass.Float32,
                (num_qo_heads, batch_size * vc_shapes["q_scale"][2]),
                16,
            )
            vc_k_scale_fake = fake_compact(
                cutlass.Float32,
                (num_kv_heads, batch_size * vc_shapes["k_scale"][2]),
                16,
            )
    sinks_fake = (
        fake_compact(cutlass.Float32, (num_qo_heads,), 4)
        if compile_spec.use_attention_sinks
        else None
    )
    # The prepared Sage scales span ``batch * Hkv`` sequences, so their length
    # stays a runtime extent like the batch size; a plan without Sage has no
    # scales.
    sage_scales_fake = (
        fake_compact(cutlass.Float32, (cute.sym_int(),), 16)
        if sage_scale_layout is not None
        else None
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    # Task objects carry loop-local state through generated control flow, so
    # select the public staged frontend for this compilation.
    with torch.cuda.device(device_index):
        compiled = build_and_load_attention_kernel(
            "context",
            "main",
            compile_spec,
            _COMPILE_OPTIONS,
            lambda: cute.compile[cute.FrontendNext](
                tensor_adapter,
                q_fake,
                k_fake,
                v_fake,
                out_fake,
                lse_fake,
                sinks_fake,
                scale_fake,
                output_scale_fake,
                qo_indptr_fake,
                kv_indptr_fake,
                variable_window_starts_fake,
                variable_window_ends_fake,
                variable_window_cta_starts_fake,
                vc_mu_fake,
                vc_q_scale_fake,
                vc_k_scale_fake,
                sage_scales_fake,
                stream_fake,
                max_active_clusters,
                packed_query,
                cutlass.Int32(0),
                cutlass.Int32(0),
                compile_spec.vc.repair_tiles if vc_attention else 0,
                options=_COMPILE_OPTIONS,
            ),
            extra=(max_active_clusters,),
        )
    policy = (
        ("scheduler", scheduler),
        ("pairing", "head" if head_paired else "query"),
        ("uniform_packed_lengths", uniform_packed_lengths),
        ("causal_single_kv_tile", causal_single_kv_tile),
        ("packed_dense_k_mask", packed_dense_k_mask),
        ("vc_attention", vc_attention),
    )
    return compiled, policy


@functools.cache
def _get_compiled_paged_context(
    compile_spec: _PagedContextCompileSpec,
):
    """Compile one batch-dynamic packed-Q, paged-K/V specialization."""

    device_index = compile_spec.device_index
    max_q_len = compile_spec.max_q_len or 0
    max_kv_len = compile_spec.max_kv_len or 0
    page_size = compile_spec.page_size
    num_qo_heads = compile_spec.num_qo_heads
    num_kv_heads = compile_spec.num_kv_heads
    head_dim = compile_spec.head_dim
    qk_dtype_key = compile_spec.qk_dtype_key
    pv_dtype_key = compile_spec.pv_dtype_key
    out_dtype_key = compile_spec.out_dtype_key
    mask_type = compile_spec.mask_type
    window_left = compile_spec.window_left
    head_paired = compile_spec.head_paired
    uniform_packed_lengths = compile_spec.uniform_packed_lengths
    has_q_offset = compile_spec.has_q_offset
    paged_v_tail_is_zero = compile_spec.paged_v_tail_is_zero
    packed_dense_k_mask = compile_spec.packed_dense_k_mask
    scheduler = compile_spec.scheduler

    import cutlass
    import cutlass.cute as cute
    from cuda.bindings import driver as cuda_drv

    dtype_map = {
        "float16": cutlass.Float16,
        "bfloat16": cutlass.BFloat16,
        "float8_e4m3fn": cutlass.Float8E4M3FN,
    }
    input_qk_dtype = dtype_map[qk_dtype_key]
    input_pv_dtype = dtype_map[pv_dtype_key]
    out_dtype = dtype_map[out_dtype_key]
    with torch.cuda.device(device_index):
        max_active_clusters = get_max_active_clusters(1)
    fmha = _make_context_kernel(
        device_index=device_index,
        input_qk_dtype=input_qk_dtype,
        input_pv_dtype=input_pv_dtype,
        out_dtype=out_dtype,
        head_dim=head_dim,
        mask_type=mask_type,
        window_left=window_left,
        head_paired=head_paired,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        has_q_offset=has_q_offset,
        causal_single_kv_tile=False,
        scheduler=scheduler,
        uses_ldtm_stat=_default_uses_ldtm_stat(
            device_index, input_qk_dtype.width, input_pv_dtype.width
        ),
        fp8_psmem_early_token=_default_fp8_psmem_early_token(device_index),
        kv_ring_stages=_default_kv_ring_stages(
            device_index, input_qk_dtype.width, input_pv_dtype.width
        ),
        umma_fp8_k64=_default_umma_fp8_k64(device_index, input_qk_dtype.width),
        page_size=page_size,
        max_kv_len=compile_spec.max_kv_len,
    )
    fmha.cfg.lse_mode = compile_spec.lse_mode
    fmha.cfg.use_attention_sinks = compile_spec.use_attention_sinks
    fmha.cfg.has_varlen = True
    fmha.cfg.has_uniform_varlen = uniform_packed_lengths
    if uniform_packed_lengths:
        fmha.cfg.uniform_seq_len_q = max_q_len
        fmha.cfg.uniform_seq_len_k = max_kv_len
    fmha.cfg.has_q_offset = has_q_offset
    fmha.cfg.exp2_fma_pairs = _default_exp2_fma_pairs(device_index, fmha.cfg)
    fmha.cfg.exp2_packed_f16x2 = _default_exp2_packed_f16x2(device_index, fmha.cfg)
    fmha.cfg.paged_v_tail_is_zero = paged_v_tail_is_zero
    if fmha.cfg.kv_tile_n != _CONTEXT_KV_TILE_N:
        raise RuntimeError(
            "context packed-K specialization assumes kv_tile_n="
            f"{_CONTEXT_KV_TILE_N}, got {fmha.cfg.kv_tile_n}"
        )
    _validate_query_work_tile_span(fmha.cfg)
    fmha.cfg.packed_dense_k_mask = packed_dense_k_mask

    # Bind packed Q, native pages and live token bounds to the context launcher.
    @cute.jit
    def tensor_adapter(
        q: cute.Tensor,
        k_cache: cute.Tensor,
        v_cache: cute.Tensor,
        out: cute.Tensor,
        lse: cute.Tensor | None,
        attention_sinks: cute.Tensor | None,
        scale_softmax_log2: cutlass.Float32,
        output_scale: cutlass.Float32,
        qo_indptr: cute.Tensor,
        block_tables: cute.Tensor,
        seq_lens_kv: cute.Tensor,
        stream: cuda_drv.CUstream,
        static_max_active_clusters: cutlass.Constexpr[int],
        max_q_len: cutlass.Int32,
        runtime_max_kv_len: cutlass.Int32,
    ) -> None:
        fmha(
            q,
            k_cache,
            v_cache,
            out,
            scale_softmax_log2,
            output_scale,
            static_max_active_clusters,
            stream,
            cum_seqlen_q=qo_indptr,
            max_seqlen_q=max_q_len,
            max_seqlen_k=runtime_max_kv_len,
            block_tables=block_tables,
            seq_lens_kv=seq_lens_kv,
            lse_output=lse,
            attention_sinks=attention_sinks,
        )

    # Describe compact metadata or Q/O storage with byte-aligned pointers.
    def fake_compact(dtype, shape, assumed_align):
        return cute.runtime.make_fake_compact_tensor(
            dtype,
            shape,
            stride_order=tuple(reversed(range(len(shape)))),
            assumed_align=assumed_align,
        )

    runtime_total_q = cute.sym_int()
    runtime_num_pages = cute.sym_int()
    batch_size = cute.sym_int()
    runtime_num_q_offsets = cute.sym_int()
    q_fake = fake_compact(input_qk_dtype, (runtime_total_q, num_qo_heads, head_dim), 16)
    kv_shape = (
        runtime_num_pages,
        num_kv_heads,
        page_size,
        head_dim,
    )

    # Keep page/head/token strides dynamic while each head dimension is contiguous.
    def fake_paged_cache(dtype):
        return cute.runtime.make_fake_tensor(
            dtype,
            kv_shape,
            stride=(cute.sym_int64(), cute.sym_int64(), cute.sym_int64(), 1),
            assumed_align=16,
        )

    k_fake = fake_paged_cache(input_qk_dtype)
    v_fake = fake_paged_cache(input_pv_dtype)
    out_fake = fake_compact(out_dtype, (runtime_total_q, num_qo_heads, head_dim), 16)
    lse_fake = (
        fake_compact(cutlass.Float32, out_fake.shape[:-1], 4)
        if compile_spec.lse_mode != "none"
        else None
    )
    scale_fake = cutlass.Float32(1.0)
    output_scale_fake = cutlass.Float32(1.0)
    qo_indptr_fake = fake_compact(cutlass.Int32, (runtime_num_q_offsets,), 4)
    runtime_page_columns = cute.sym_int()
    runtime_block_table_row_stride = cute.sym_int()
    block_tables_fake = cute.runtime.make_fake_tensor(
        cutlass.Int32,
        (batch_size, runtime_page_columns),
        stride=(runtime_block_table_row_stride, 1),
        assumed_align=4,
    )
    seq_lens_fake = fake_compact(cutlass.Int32, (batch_size,), 4)
    sinks_fake = (
        fake_compact(cutlass.Float32, (num_qo_heads,), 4)
        if compile_spec.use_attention_sinks
        else None
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    with torch.cuda.device(device_index):
        compiled = build_and_load_attention_kernel(
            "context",
            "paged",
            compile_spec,
            _COMPILE_OPTIONS,
            lambda: cute.compile[cute.FrontendNext](
                tensor_adapter,
                q_fake,
                k_fake,
                v_fake,
                out_fake,
                lse_fake,
                sinks_fake,
                scale_fake,
                output_scale_fake,
                qo_indptr_fake,
                block_tables_fake,
                seq_lens_fake,
                stream_fake,
                max_active_clusters,
                cutlass.Int32(0),
                cutlass.Int32(0),
                options=_COMPILE_OPTIONS,
            ),
            extra=(max_active_clusters,),
        )
    policy = (
        ("scheduler", scheduler),
        ("pairing", "head" if head_paired else "query"),
        ("kv_layout", "paged_hnd"),
        ("page_size", page_size),
        ("uniform_packed_lengths", uniform_packed_lengths),
        ("has_q_offset", has_q_offset),
        ("paged_v_tail_is_zero", paged_v_tail_is_zero),
        ("causal_single_kv_tile", False),
        ("packed_dense_k_mask", packed_dense_k_mask),
    )
    return compiled, policy


def _validate_runtime_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    geometry: _ContextPlanGeometry,
    *,
    qo_indptr: Optional[torch.Tensor],
    kv_indptr: Optional[torch.Tensor],
) -> tuple[int, int]:
    """Validate contiguous data and request metadata against one plan."""

    _validate_base_tensors(q, k, v, allow_mla=True)
    if q.device != geometry.device:
        raise ValueError(f"q must be on {geometry.device}, got {q.device}")
    if q.dtype != geometry.qk_dtype:
        raise ValueError(f"q must have dtype {geometry.qk_dtype}, got {q.dtype}")
    if k.dtype != geometry.qk_dtype:
        raise ValueError(f"k must have dtype {geometry.qk_dtype}, got {k.dtype}")
    if v.dtype != geometry.pv_dtype:
        raise ValueError(f"v must have dtype {geometry.pv_dtype}, got {v.dtype}")
    v_shape = (*k.shape[:-1], geometry.head_dim_vo or geometry.head_dim)
    if tuple(v.shape) != v_shape:
        raise ValueError(f"v must have shape {v_shape}, got {tuple(v.shape)}")

    if geometry.packed_query:
        if q.ndim != 3 or tuple(q.shape[1:]) != (
            geometry.num_qo_heads,
            geometry.head_dim,
        ):
            raise ValueError(
                "q must have packed shape [total_q, Hq, D] with "
                f"Hq/D=({geometry.num_qo_heads}, {geometry.head_dim}), got "
                f"{tuple(q.shape)}"
            )
        if k.ndim != 3 or tuple(k.shape[1:]) != (
            geometry.num_kv_heads,
            geometry.head_dim,
        ):
            raise ValueError(
                "k and v must have packed shape [total_k, Hkv, D] with "
                f"Hkv/D=({geometry.num_kv_heads}, {geometry.head_dim}), got "
                f"{tuple(k.shape)}"
            )
        total_q = int(q.shape[0])
        total_k = int(k.shape[0])
        _validate_padded_data_extent(total_q, "total_q")
        _validate_padded_data_extent(total_k, "total_k")
        if total_q > (geometry.batch_size * geometry.max_q_len):
            raise ValueError(
                "q must contain no more than "
                "batch_size * max_q_len rows; got "
                f"{total_q} for batch_size={geometry.batch_size} and "
                f"max_q_len={geometry.max_q_len}"
            )
        if total_k > (geometry.batch_size * geometry.max_seq_len_k):
            raise ValueError(
                "k and v must contain no more than "
                "batch_size * max_kv_len rows; got "
                f"{total_k} for batch_size={geometry.batch_size} and "
                f"max_kv_len={geometry.max_seq_len_k}"
            )
        if qo_indptr is None or kv_indptr is None:
            raise ValueError("packed plans require qo_indptr and kv_indptr in run()")
        request_lengths: dict[str, tuple[int, ...]] = {}
        for indptr, name, expected_total in (
            (qo_indptr, "qo_indptr", total_q),
            (kv_indptr, "kv_indptr", total_k),
        ):
            _validate_indptr_tensor(indptr, name, device=geometry.device)
            if indptr.numel() != geometry.batch_size + 1:
                raise ValueError(
                    f"{name} must contain batch_size + 1 elements "
                    f"({geometry.batch_size + 1}), got {indptr.numel()}"
                )
            _, lengths = _read_indptr(
                indptr,
                name,
                expected_total=expected_total,
            )
            request_lengths[name] = lengths
            capacity = (
                geometry.max_q_len if name == "qo_indptr" else geometry.max_seq_len_k
            )
            for batch_idx, length in enumerate(lengths):
                if length > capacity:
                    raise ValueError(
                        f"{name} deltas must not exceed the planned capacity "
                        f"{capacity}; batch {batch_idx} has {length}"
                    )
        q_lengths = request_lengths["qo_indptr"]
        k_lengths = request_lengths["kv_indptr"]
        if geometry.mask_type == "causal":
            for batch_idx, (q_length, k_length) in enumerate(
                zip(q_lengths, k_lengths, strict=True)
            ):
                if k_length > 0 and q_length > k_length:
                    raise ValueError(
                        "bottom-right causal context requires Sq <= Sk for each "
                        f"request; got batch {batch_idx}: Sq={q_length}, "
                        f"Sk={k_length}"
                    )
        _validate_compact(q, "q", "[total_q, Hq, D]")
        _validate_compact(k, "k", "[total_k, Hkv, D]")
        _validate_compact(v, "v", "[total_k, Hkv, D]")
        return max(q_lengths), max(k_lengths)
    else:
        if qo_indptr is not None or kv_indptr is not None:
            raise ValueError(
                "qo_indptr and kv_indptr require packed_query=True in plan()"
            )
        q_shape = (
            geometry.batch_size,
            geometry.max_q_len,
            geometry.num_qo_heads,
            geometry.head_dim,
        )
        kv_shape = (
            geometry.batch_size,
            _fixed_kv_rows(geometry.max_seq_len_k, geometry.vc),
            geometry.num_kv_heads,
            geometry.head_dim,
        )
        if tuple(q.shape) != q_shape:
            raise ValueError(f"q must have shape {q_shape}, got {tuple(q.shape)}")
        if tuple(k.shape) != kv_shape:
            raise ValueError(
                f"k and v must have shape {kv_shape}, got {tuple(k.shape)}"
            )
        _validate_compact(q, "q", "[B, Sq, Hq, D]")
        _validate_compact(k, "k", "[B, Sk, Hkv, D]")
        _validate_compact(v, "v", "[B, Sk, Hkv, D]")
        return int(q.shape[1]), geometry.max_seq_len_k


def _validate_paged_runtime_inputs(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    geometry: _PagedContextPlanGeometry,
) -> None:
    """Validate per-run data tensors without fixing packed/page extents."""

    _validate_base_tensors(q, k_cache, v_cache)
    if q.device != geometry.device:
        raise ValueError(f"q must be on {geometry.device}, got {q.device}")
    if q.dtype != geometry.qk_dtype:
        raise ValueError(f"q must have dtype {geometry.qk_dtype}, got {q.dtype}")
    if k_cache.dtype != geometry.qk_dtype:
        raise ValueError(
            f"k_cache must have dtype {geometry.qk_dtype}, got {k_cache.dtype}"
        )
    if v_cache.dtype != geometry.pv_dtype:
        raise ValueError(
            f"v_cache must have dtype {geometry.pv_dtype}, got {v_cache.dtype}"
        )
    if q.ndim != 3 or tuple(q.shape[1:]) != (
        geometry.num_qo_heads,
        geometry.head_dim,
    ):
        raise ValueError(
            "q must have shape [total_q, Hq, D] with "
            f"Hq/D=({geometry.num_qo_heads}, {geometry.head_dim}), got {tuple(q.shape)}"
        )
    expected_kv_tail = (
        geometry.num_kv_heads,
        geometry.page_size,
        geometry.head_dim,
    )
    if k_cache.ndim != 4 or tuple(k_cache.shape[1:]) != expected_kv_tail:
        raise ValueError(
            "k_cache must have shape [num_pages, Hkv, page_size, D] with "
            f"Hkv/page/D={expected_kv_tail}, got {tuple(k_cache.shape)}"
        )
    total_q = int(q.shape[0])
    _validate_padded_data_extent(total_q, "total_q")
    if total_q > (geometry.batch_size * geometry.max_q_len):
        raise ValueError(
            "q must contain no more than batch_size * max_q_len "
            f"rows; got {total_q} rows for batch_size={geometry.batch_size} and "
            f"max_q_len={geometry.max_q_len}"
        )
    _validate_extent(int(k_cache.shape[0]), "num_physical_pages")
    _validate_compact(q, "q", "[total_q, Hq, D]")
    _validate_native_hnd_tma_strides(k_cache, "k_cache")
    _validate_native_hnd_tma_strides(v_cache, "v_cache")


def _validate_paged_runtime_metadata(
    qo_indptr: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens_kv: torch.Tensor,
    geometry: _PagedContextPlanGeometry,
    *,
    total_q: int,
    num_physical_pages: int,
) -> tuple[int, int]:
    """Validate per-run fixed-table storage and active metadata values."""

    _validate_indptr_tensor(qo_indptr, "qo_indptr", device=geometry.device)
    if qo_indptr.numel() != geometry.batch_size + 1:
        raise ValueError(
            "qo_indptr must contain batch_size + 1 elements "
            f"({geometry.batch_size + 1}), got {qo_indptr.numel()}"
        )

    required_page_columns = (
        geometry.max_kv_len + geometry.page_size - 1
    ) // geometry.page_size
    _validate_block_tables_tensor(
        block_tables,
        device=geometry.device,
        batch_size=geometry.batch_size,
        required_page_columns=required_page_columns,
    )

    _validate_paged_metadata_tensor(seq_lens_kv, "seq_lens_kv", device=geometry.device)
    if seq_lens_kv.numel() != geometry.batch_size:
        raise ValueError(
            "seq_lens_kv must contain one value per request; expected "
            f"{geometry.batch_size}, got {seq_lens_kv.numel()}"
        )

    _, q_lengths = _read_indptr(qo_indptr, "qo_indptr", expected_total=total_q)
    seq_lens_kv = _read_int32_values(
        seq_lens_kv,
        "seq_lens_kv",
        expected_count=geometry.batch_size,
    )
    block_table_rows = block_tables.tolist()
    for batch_idx, (q_length, kv_length, block_table_row) in enumerate(
        zip(q_lengths, seq_lens_kv, block_table_rows, strict=True)
    ):
        if q_length > geometry.max_q_len:
            raise ValueError(
                "qo_indptr deltas must not exceed max_q_len; "
                f"batch {batch_idx} has {q_length}, maximum is "
                f"{geometry.max_q_len}"
            )
        if geometry.uniform_packed_lengths and q_length != geometry.max_q_len:
            raise ValueError(
                "uniform_packed_lengths=True requires every qo_indptr delta "
                f"to equal max_q_len={geometry.max_q_len}; batch "
                f"{batch_idx} has {q_length}"
            )
        if kv_length < 0 or kv_length > geometry.max_kv_len:
            raise ValueError(
                "seq_lens_kv entries must be nonnegative and not exceed "
                f"max_kv_len={geometry.max_kv_len}; batch {batch_idx} has "
                f"{kv_length}"
            )
        if geometry.uniform_packed_lengths and kv_length != geometry.max_kv_len:
            raise ValueError(
                "uniform_packed_lengths=True requires every seq_lens_kv entry "
                f"to equal max_kv_len={geometry.max_kv_len}; batch "
                f"{batch_idx} has {kv_length}"
            )
        required_pages = (kv_length + geometry.page_size - 1) // geometry.page_size
        if geometry.mask_type == "causal" and kv_length > 0 and q_length > kv_length:
            raise ValueError(
                "bottom-right causal context requires Sq <= Sk for each "
                f"request; got batch {batch_idx}: Sq={q_length}, Sk={kv_length}"
            )
        if (
            geometry.mask_type == "causal"
            and not geometry.has_q_offset
            and q_length != kv_length
        ):
            raise ValueError(
                "has_q_offset=False requires Sq == Sk for every causal "
                f"request; got batch {batch_idx}: Sq={q_length}, Sk={kv_length}"
            )
        for row_offset, page_idx in enumerate(block_table_row[:required_pages]):
            if page_idx < 0 or page_idx >= num_physical_pages:
                raise ValueError(
                    "active block_tables entries must index the physical "
                    f"page pool; batch {batch_idx}, row entry {row_offset} is "
                    f"{page_idx}, pool has {num_physical_pages} pages"
                )
    return max(q_lengths), max(seq_lens_kv)


def _prepare_out(
    out: Optional[torch.Tensor],
    *,
    q: torch.Tensor,
    out_dtype: torch.dtype,
    head_dim_vo: int | None = None,
) -> torch.Tensor:
    out_shape = (*q.shape[:-1], head_dim_vo or q.shape[-1])
    if out is None:
        return torch.empty(out_shape, dtype=out_dtype, device=q.device)
    _validate_tensor(out, "out")
    if tuple(out.shape) != out_shape:
        raise ValueError(f"out must have shape {out_shape}, got {tuple(out.shape)}")
    if out.dtype != out_dtype:
        raise ValueError(f"out must have dtype {out_dtype}, got {out.dtype}")
    if out.device != q.device:
        raise ValueError(f"out must be on {q.device}, got {out.device}")
    layout = "[total_q, Hq, D]" if q.ndim == 3 else "[B, Sq, Hq, D]"
    _validate_compact(out, "out", layout)
    _validate_alignment(out, "out", 16)
    return out


def _variable_window_buffers(geometry, empty_i32):
    if geometry.mask_type != "variable_window":
        return None, empty_i32
    tile_size_q = _variable_window_tile_size_q(geometry.head_dim)
    num_seq_tiles = (geometry.max_q_len + tile_size_q - 1) // tile_size_q
    padded_rows = num_seq_tiles * tile_size_q
    cta_starts = torch.empty(
        geometry.batch_size * num_seq_tiles, dtype=torch.int32, device=geometry.device
    )
    padded_starts = (
        torch.empty(
            (geometry.batch_size, padded_rows),
            dtype=torch.int32,
            device=geometry.device,
        )
        if padded_rows != geometry.max_q_len
        else None
    )
    return padded_starts, cta_starts


def _cuda_scale(value: torch.Tensor, name: str, device: torch.device) -> torch.Tensor:
    """Validate a live device scalar without reading it back to the host."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a CUDA FP32 tensor")
    if value.device != device or value.dtype != torch.float32 or value.numel() != 1:
        raise ValueError(f"{name} must contain one FP32 value on {device}")
    return value.reshape(1)


class BatchPrefillTSWrapper:
    """Reusable contiguous context attention with eager planning.

    run() accepts fixed BSHD or packed THD Q/K/V. Plan owns model defaults and
    variable-window fallback metadata, while every run supplies request tensors.
    Supports equal D128/D192/D256 or contiguous non-absorbed MLA QK192/V128 geometry.
    Dense fixed D128 Sage plans bind prepared per-block/channel scale tensors.
    Replanning invalidates graphs; complete prior launches and replays first.
    VC plans also own a refreshed per-channel output-scale buffer; their runs
    must not overlap."""

    @flashinfer_experimental_api
    def __init__(self) -> None:
        """Initialize an unplanned task-scheduled context-attention wrapper."""
        self._plan_state: Optional[_ContextPlanState] = None

    @flashinfer_experimental_api
    def plan(
        self,
        *,
        device: int | str | torch.device,
        batch_size: int,
        max_q_len: int,
        max_kv_len: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        head_dim_vo: int | None = None,
        q_dtype: torch.dtype,
        k_dtype: torch.dtype,
        v_dtype: Optional[torch.dtype] = None,
        out_dtype: Optional[torch.dtype] = None,
        packed_query: bool = False,
        mask_type: Literal["dense", "causal", "variable_window"] = "dense",
        window_left: int = -1,
        softmax_scale: Optional[float] = None,
        output_scale: float = 1.0,
        vc_config: Optional[VCAttentionConfig] = None,
        lse_mode: Literal["none", "base2"] = "none",
        use_attention_sinks: bool = False,
        sage: Optional[SageAttentionConfig] = None,
        cuda_scales: bool = False,
    ) -> None:
        """Select/cache one execution policy, obtain compiled kernels, and bind resources.

        ``vc_config`` selects a VC-Attention recipe and the planned Q/K dtype
        picks its Q/K side. bf16 Q/K runs VC-Attention-QK16 and E4M3
        Q/K runs VC-Attention-QK8, whose Hadamard-rotated, per-block scaled Q
        and K need ``vc.q_scale`` and ``vc.k_scale`` on every :meth:`run`. V is
        E4M3 in both, stored as per-128-token-tile residuals whose bf16 tile
        means the kernel restores. Every :meth:`run` then supplies the permuted
        K, the E4M3 V and ``vc=VCAttentionParams(...)``.

        batch_size is exact. Fixed contiguous inputs use exact Q/KV tensor extents;
        packed/paged request lengths may vary within max_q_len/max_kv_len.
        Complete planning outside CUDA graph capture. Replanning replaces owned
        resources and invalidates previous graphs; finish their launches first.
        lse_mode='base2' enables final LSE stores, and use_attention_sinks enables
        a per-run sink binding. Context scales are positive host scalar defaults.
        Sage plans use BF16 output by default and require per-run Sage scales.
        Sage V-mean restoration cannot be combined with attention sinks.
        cuda_scales binds one-element CUDA FP32 scales for graph-replayed
        per-tensor quantization; it cannot be combined with Sage or VC plans.
        Equivalent plans reuse policy and compiled-kernel caches.
        VC runs sharing a plan must not overlap.
        """

        lse_mode = validate_lse_mode(lse_mode)
        if not isinstance(use_attention_sinks, bool):
            raise TypeError("use_attention_sinks must be a bool")
        if out_dtype is not None:
            resolved_out_dtype = out_dtype
        elif sage is not None:
            # As in block-sparse Sage plans; an 8-bit output is not a recipe.
            resolved_out_dtype = torch.bfloat16
        else:
            resolved_out_dtype = q_dtype
        resolved_v_dtype = k_dtype if v_dtype is None else v_dtype
        if k_dtype != q_dtype:
            raise NotImplementedError(
                "attention-ts context requires Q and K to use the same dtype; "
                f"got q_dtype={q_dtype} and k_dtype={k_dtype}"
            )
        geometry = _resolve_context_plan_geometry(
            device=device,
            batch_size=batch_size,
            max_q_len=max_q_len,
            max_kv_len=max_kv_len,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            head_dim_vo=head_dim_vo,
            qk_dtype=k_dtype,
            pv_dtype=resolved_v_dtype,
            packed_query=packed_query,
            mask_type=mask_type,
            window_left=window_left,
            out_dtype=resolved_out_dtype,
            sage=sage,
        )
        if sage is not None and sage.v_mean and use_attention_sinks:
            raise NotImplementedError(
                "Sage V-mean restoration does not support attention sinks"
            )
        if vc_config is not None and not isinstance(vc_config, VCAttentionConfig):
            raise TypeError("vc_config must be a VCAttentionConfig instance")
        if vc_config is not None and (packed_query or mask_type != "dense"):
            raise ValueError(
                "vc_config requires a dense contiguous fixed-length plan; VC-Attention "
                "has no packed, causal or variable-window kernel"
            )
        if vc_config is not None and (lse_mode != "none" or use_attention_sinks):
            # VC uses quantized P sums; ordinary context LSE/sink math does not apply.
            raise NotImplementedError(
                "VC-Attention does not support LSE or attention sinks"
            )
        if not isinstance(cuda_scales, bool):
            raise TypeError("cuda_scales must be a bool")
        if cuda_scales and (vc_config is not None or sage is not None):
            raise ValueError("cuda_scales requires a plain context plan")
        geometry = replace(geometry, vc=vc_config)
        if softmax_scale is None:
            softmax_scale = geometry.head_dim**-0.5
        scale_softmax_log2 = _validate_scale(
            _validate_scale(softmax_scale, "softmax_scale") * math.log2(math.e),
            "softmax_scale * log2(e)",
        )
        output_scale = _validate_scale(output_scale, "output_scale")
        if cuda_scales:
            scale_softmax_log2 = torch.tensor(
                [scale_softmax_log2], dtype=torch.float32, device=geometry.device
            )
            output_scale = torch.tensor(
                [output_scale], dtype=torch.float32, device=geometry.device
            )
        vc_output_scale_buffer = (
            torch.empty(
                (
                    geometry.batch_size,
                    geometry.num_qo_heads,
                    geometry.head_dim_vo or geometry.head_dim,
                ),
                dtype=torch.float32,
                device=geometry.device,
            )
            if vc_config is not None
            else None
        )
        empty_i32 = torch.empty(1, dtype=torch.int32, device=geometry.device)
        variable_window_padded_starts, variable_window_cta_starts = (
            _variable_window_buffers(geometry, empty_i32)
        )
        # Keep all runtime tensor allocation ahead of CUTLASS JIT. Besides
        # making plan publication atomic, this lets compute-sanitizer patch the
        # generated attention kernel without interleaving later PyTorch setup
        # launches with the DSL compiler/runtime callbacks.
        vc_shapes = None
        if vc_config is not None:
            vc_shapes = vc_scale_shapes(
                vc_config,
                batch_size=geometry.batch_size,
                seq_len_kv=geometry.max_seq_len_k,
                num_kv_heads=geometry.num_kv_heads,
                head_dim=geometry.head_dim_vo or geometry.head_dim,
                seq_len_q=geometry.max_q_len,
                num_qo_heads=geometry.num_qo_heads,
                qk_fp8=geometry.qk_dtype == torch.float8_e4m3fn,
            )
        sage_state = (
            _build_context_sage_state(geometry) if geometry.sage is not None else None
        )
        compiled, policy = _get_compiled_context(
            _context_compile_spec(
                geometry,
                runtime_lengths=vc_config is None,
                lse_mode=lse_mode,
                use_attention_sinks=use_attention_sinks,
                cuda_scales=cuda_scales,
            )
        )
        self._plan_state = _ContextPlanState(
            geometry=geometry,
            scale_softmax_log2=scale_softmax_log2,
            output_scale=output_scale,
            empty_i32=empty_i32,
            variable_window_padded_starts=variable_window_padded_starts,
            variable_window_cta_starts=variable_window_cta_starts,
            compiled=compiled,
            policy=policy,
            vc_scale_shapes=vc_shapes,
            vc_output_scale=vc_output_scale_buffer,
            cuda_scales=cuda_scales,
            lse_mode=lse_mode,
            use_attention_sinks=use_attention_sinks,
            sage=sage_state,
        )

    @flashinfer_experimental_api
    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        qo_indptr: Optional[torch.Tensor] = None,
        kv_indptr: Optional[torch.Tensor] = None,
        *,
        variable_window_token_starts: Optional[torch.Tensor] = None,
        variable_window_token_ends: Optional[torch.Tensor] = None,
        variable_window_cta_starts: Optional[torch.Tensor] = None,
        max_q_len: int | None = None,
        max_kv_len: int | None = None,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        attention_sinks: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        output_scale: float | torch.Tensor | None = None,
        scale_softmax_log2: torch.Tensor | None = None,
        vc: Optional[VCAttentionParams] = None,
        sage: Optional[SageAttentionParams] = None,
        validate: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Launch the prepared contiguous attention with live request tensors.

        Uses the policy and compiled kernels prepared by plan(). Batch and query
        shape must match that plan; live K/V lengths stay within its capacity.
        Supply output/LSE buffers and validate=False during graph capture, and
        retain the wrapper and captured buffers for every replay.

        An enabled lse_mode="base2" plan writes compact FP32 log2(sum(exp(scores)))
        with the query shape minus its final dimension. return_lse controls tuple
        return only; an omitted enabled buffer is allocated. Output/LSE/workspace must
        be disjoint from inputs and live buffers. One wrapper serves one ordered lane.

        max_q_len/max_kv_len are launch bounds within plan capacities.
        Validation infers omitted values from existing metadata; trusted packed
        calls use plan bounds. Exact runtime lengths do not specialize kernels.

        Packed inputs require qo_indptr and kv_indptr. Variable-window fixed inputs
        require inclusive start/end bounds; ends must be nondecreasing within each
        request, while starts may be unordered. Optional CTA minima avoid refreshing
        wrapper-owned fallback scratch. Scale overrides are finite positive host
        scalars: softmax_scale is the effective QK scale, and
        output_scale scales V/output. They are runtime kernel operands, not JIT keys.
        With cuda_scales, supply scale_softmax_log2 and output_scale as
        one-element CUDA FP32 tensors instead. Their values remain on device
        and must be positive and finite, including after graph updates.

        Enabled attention_sinks supplies compact CUDA FP32 [Hq] natural-log sink
        logits after QK scaling. Values must be finite or -inf; validation checks only
        metadata. Each sink adds denominator mass with zero V contribution.
        Sage plans require matching SageAttentionParams; their prepared scales
        are refreshed on the current stream before attention.
        VC plans require vc=VCAttentionParams(...) with permuted K and residual V;
        its per-channel V scale multiplies output_scale in the plan-owned buffer."""

        if not isinstance(validate, bool):
            raise TypeError("validate must be a bool")

        state = self._plan_state
        if state is None:
            raise RuntimeError("plan() must be called before run()")
        geometry = state.geometry
        if geometry.mask_type != "variable_window" and (
            variable_window_token_starts is not None
            or variable_window_token_ends is not None
            or variable_window_cta_starts is not None
        ):
            raise ValueError(
                "variable-window metadata requires mask_type='variable_window'"
            )
        if not validate and not geometry.packed_query:
            max_q_len = int(q.shape[1])
            max_kv_len = (
                geometry.max_seq_len_k if geometry.vc is not None else int(k.shape[1])
            )
        if geometry.packed_query:
            if qo_indptr is None or kv_indptr is None:
                raise ValueError(
                    "packed plans require qo_indptr and kv_indptr in run()"
                )
            runtime_qo_indptr = qo_indptr
            runtime_kv_indptr = kv_indptr
        else:
            if qo_indptr is not None or kv_indptr is not None:
                raise ValueError(
                    "qo_indptr and kv_indptr require packed_query=True in plan()"
                )
            runtime_qo_indptr = state.empty_i32
            runtime_kv_indptr = state.empty_i32

        if (vc is None) == (geometry.vc is not None):
            raise ValueError(
                "vc=VCAttentionParams(...) is required by a VC plan and rejected "
                "by a plan without vc_config"
            )
        vc_output_scale: Optional[torch.Tensor] = None
        vc_mu: Optional[torch.Tensor] = None
        vc_q_scale: Optional[torch.Tensor] = None
        vc_k_scale: Optional[torch.Tensor] = None
        if vc is not None:
            if validate:
                validate_vc_params(vc, state.vc_scale_shapes, device=geometry.device)
            if geometry.qk_dtype == torch.float8_e4m3fn:
                if vc.q_scale is None or vc.k_scale is None:
                    raise ValueError(
                        "VC-Attention-QK8 plans require vc.q_scale and vc.k_scale"
                    )
                vc_q_scale = vc.q_scale.view(geometry.num_qo_heads, -1)
                vc_k_scale = vc.k_scale.view(geometry.num_kv_heads, -1)
            elif vc.q_scale is not None or vc.k_scale is not None:
                raise ValueError(
                    "vc.q_scale and vc.k_scale belong to VC-Attention-QK8 plans with "
                    "E4M3 Q/K"
                )
            vc_mu = vc.tile_means
            # VC plans have equal Q and K/V head counts, so the per-channel V
            # scale is already in output order.
            vc_output_scale = vc.v_scale
        if validate:
            actual_q, actual_kv = _validate_runtime_inputs(
                q,
                k,
                v,
                geometry,
                qo_indptr=qo_indptr,
                kv_indptr=kv_indptr,
            )
            from .decode import resolve_max_sequence_length

            max_q_len = resolve_max_sequence_length(
                max_q_len, actual_q, "max_q_len", geometry.max_q_len
            )
            max_kv_len = resolve_max_sequence_length(
                max_kv_len, actual_kv, "max_kv_len", geometry.max_seq_len_k
            )
        else:
            if max_q_len is None:
                max_q_len = geometry.max_q_len
            if max_kv_len is None:
                max_kv_len = geometry.max_seq_len_k
        if (sage is None) != (state.sage is None):
            raise ValueError(
                "sage=SageAttentionParams(...) is required by a Sage plan and "
                "rejected by a plan without Sage attention"
            )
        if sage is not None and validate:
            validate_sage_params(sage, state.sage.scale_shapes, device=geometry.device)
        if geometry.mask_type == "variable_window":
            if (
                variable_window_token_starts is None
                or variable_window_token_ends is None
            ):
                raise ValueError(
                    "variable-window plans require start and end bounds in run()"
                )
            if validate:
                (
                    runtime_window_starts,
                    runtime_window_ends,
                    runtime_window_start_values,
                ) = _validate_variable_window_bounds(
                    variable_window_token_starts,
                    variable_window_token_ends,
                    geometry=geometry,
                )
            else:
                runtime_window_starts = variable_window_token_starts.flatten()
                runtime_window_ends = variable_window_token_ends.flatten()
            if variable_window_cta_starts is None:
                runtime_window_cta_starts = _refresh_variable_window_cta_starts(
                    runtime_window_starts,
                    state=state,
                )
            elif validate:
                runtime_window_cta_starts = _validate_variable_window_cta_starts(
                    variable_window_cta_starts,
                    token_start_values=runtime_window_start_values,
                    geometry=geometry,
                )
            else:
                runtime_window_cta_starts = variable_window_cta_starts.flatten()
        else:
            runtime_window_starts = state.empty_i32
            runtime_window_ends = state.empty_i32
            runtime_window_cta_starts = state.empty_i32
        if state.cuda_scales:
            if softmax_scale is not None:
                raise ValueError("cuda_scales uses the device scale_softmax_log2 operand")
            scale_softmax_log2 = _cuda_scale(
                state.scale_softmax_log2 if scale_softmax_log2 is None else scale_softmax_log2,
                "scale_softmax_log2",
                geometry.device,
            )
            output_scale = _cuda_scale(
                state.output_scale if output_scale is None else output_scale,
                "output_scale",
                geometry.device,
            )
        else:
            if scale_softmax_log2 is not None:
                raise ValueError("device scale_softmax_log2 requires cuda_scales=True")
            scale_softmax_log2 = (
                state.scale_softmax_log2
                if softmax_scale is None
                else _validate_scale(
                    _validate_scale(softmax_scale, "softmax_scale") * math.log2(math.e),
                    "softmax_scale * log2(e)",
                )
            )
            output_scale = (
                state.output_scale
                if output_scale is None
                else _validate_scale(output_scale, "output_scale")
            )
        if vc_output_scale is not None:
            output_scale = torch.mul(
                vc_output_scale, output_scale, out=state.vc_output_scale
            )

        if out is None:
            out = torch.empty(
                (*q.shape[:-1], geometry.head_dim_vo or geometry.head_dim),
                dtype=geometry.out_dtype,
                device=q.device,
            )
        elif validate:
            out = _prepare_out(
                out,
                q=q,
                out_dtype=geometry.out_dtype,
                head_dim_vo=geometry.head_dim_vo,
            )
        lse = prepare_lse(
            lse,
            q=q,
            mode=state.lse_mode,
            return_lse=return_lse,
            validate=validate,
        )
        attention_sinks = validate_attention_sinks(
            attention_sinks,
            q=q,
            enabled=state.use_attention_sinks,
            validate=validate,
        )
        if sage is not None:
            state.sage.prepare(
                sage.q_scale, sage.k_scale, sage.v_scale, sage.v_mean, state.sage.scales
            )
            runtime_sage_scales = state.sage.scales
        else:
            runtime_sage_scales = None
        state.compiled(
            q,
            k,
            v,
            out,
            lse,
            attention_sinks,
            scale_softmax_log2,
            output_scale,
            runtime_qo_indptr,
            runtime_kv_indptr,
            runtime_window_starts,
            runtime_window_ends,
            runtime_window_cta_starts,
            vc_mu,
            vc_q_scale,
            vc_k_scale,
            runtime_sage_scales,
            max_q_len,
            max_kv_len,
        )
        return (out, lse) if return_lse else out


class BatchPrefillPagedTSWrapper:
    """Reusable packed-Q context attention over HND or NHD paged K/V.

    plan() retains model semantics and optional exact-uniform/zero-offset/zero-V-tail
    promises. run() supplies Q offsets, a row-strided page table, KV lengths, and
    host execution bounds. Cache normalization changes metadata without copying.
    Replanning invalidates graphs; complete prior launches and replays first."""

    @flashinfer_experimental_api
    def __init__(self, kv_layout: Literal["HND", "NHD"] = "HND") -> None:
        """Initialize an unplanned paged-context wrapper with a K/V layout."""
        _validate_kv_layout(kv_layout)
        self._kv_layout = kv_layout
        self._plan_state: Optional[_PagedContextPlanState] = None

    @flashinfer_experimental_api
    def plan(
        self,
        *,
        device: int | str | torch.device,
        batch_size: int,
        max_q_len: int,
        max_kv_len: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        q_dtype: torch.dtype,
        k_dtype: torch.dtype,
        v_dtype: Optional[torch.dtype] = None,
        out_dtype: Optional[torch.dtype] = None,
        page_size: int = _DEFAULT_PAGED_KV_PAGE_SIZE,
        mask_type: Literal["dense", "causal"] = "dense",
        window_left: int = -1,
        softmax_scale: Optional[float] = None,
        output_scale: float = 1.0,
        lse_mode: Literal["none", "base2"] = "none",
        use_attention_sinks: bool = False,
        uniform_packed_lengths: bool = False,
        has_q_offset: bool = True,
        paged_v_tail_is_zero: bool = False,
    ) -> None:
        """Select/cache one execution policy, obtain compiled kernels, and bind resources.

        Page-table padding exposes ceil(max_kv_len / page_size) columns.
        Uniform-length and zero-V-tail promises remain caller-owned.
        batch_size is exact. Fixed contiguous inputs use exact Q/KV tensor extents;
        packed/paged request lengths may vary within max_q_len/max_kv_len.
        Complete planning outside CUDA graph capture. Replanning replaces owned
        resources and invalidates previous graphs; finish their launches first.
        lse_mode='base2' enables final LSE stores, and use_attention_sinks enables
        a per-run sink binding. Context scales are positive host scalar defaults.
        Equivalent plans reuse policy and compiled-kernel caches."""

        lse_mode = validate_lse_mode(lse_mode)
        if not isinstance(use_attention_sinks, bool):
            raise TypeError("use_attention_sinks must be a bool")
        resolved_out_dtype = q_dtype if out_dtype is None else out_dtype
        resolved_v_dtype = k_dtype if v_dtype is None else v_dtype
        if k_dtype != q_dtype:
            raise NotImplementedError(
                "attention-ts paged context requires Q and K to use the same "
                f"dtype; got q_dtype={q_dtype} and k_dtype={k_dtype}"
            )
        geometry = _resolve_paged_plan_geometry(
            device=device,
            batch_size=batch_size,
            max_q_len=max_q_len,
            max_kv_len=max_kv_len,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            qk_dtype=k_dtype,
            pv_dtype=resolved_v_dtype,
            page_size=page_size,
            mask_type=mask_type,
            window_left=window_left,
            out_dtype=resolved_out_dtype,
            uniform_packed_lengths=uniform_packed_lengths,
            has_q_offset=has_q_offset,
            paged_v_tail_is_zero=paged_v_tail_is_zero,
        )
        if softmax_scale is None:
            softmax_scale = geometry.head_dim**-0.5
        scale_softmax_log2 = _validate_scale(
            _validate_scale(softmax_scale, "softmax_scale") * math.log2(math.e),
            "softmax_scale * log2(e)",
        )
        output_scale = _validate_scale(output_scale, "output_scale")
        compiled, policy = _get_compiled_paged_context(
            _paged_context_compile_spec(
                geometry,
                runtime_lengths=True,
                lse_mode=lse_mode,
                use_attention_sinks=use_attention_sinks,
            )
        )
        policy = tuple(
            (key, f"paged_{self._kv_layout.lower()}" if key == "kv_layout" else value)
            for key, value in policy
        )
        # Publish one immutable state object only after every fallible step.
        self._plan_state = _PagedContextPlanState(
            geometry=geometry,
            scale_softmax_log2=scale_softmax_log2,
            output_scale=output_scale,
            compiled=compiled,
            policy=policy,
            lse_mode=lse_mode,
            use_attention_sinks=use_attention_sinks,
        )

    @flashinfer_experimental_api
    def run(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        qo_indptr: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens_kv: torch.Tensor,
        *,
        max_q_len: int | None = None,
        max_kv_len: int | None = None,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        attention_sinks: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        output_scale: Optional[float] = None,
        validate: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Launch packed-Q context with live offsets, page tables and KV lengths.

        Uses the policy and compiled kernels prepared by plan(). Batch and query
        shape must match that plan; live K/V lengths stay within its capacity.
        Supply output/LSE buffers and validate=False during graph capture, and
        retain the wrapper and captured buffers for every replay.

        An enabled lse_mode="base2" plan writes compact FP32 log2(sum(exp(scores)))
        with the query shape minus its final dimension. return_lse controls tuple
        return only; an omitted enabled buffer is allocated. Output/LSE/workspace must
        be disjoint from inputs and live buffers. One wrapper serves one ordered lane.

        * ``qo_indptr`` starts at zero, is nondecreasing, ends at the packed
          Q extent, and has deltas no greater than ``max_q_len``;
        * ``C`` is at least ``ceil(max_kv_len / page_size)``;
        * ``seq_lens_kv`` entries are nonnegative and do not exceed
          ``max_kv_len``;
        * active causal requests additionally satisfy ``Sq[b] <= Sk[b]``; and
        * every active page-table entry is a valid physical page ID in both
          separate, isomorphic K and V pools. Padding entries beyond the
          logical K/V length are not dereferenced.

        Metadata uses CUDA int32 on Q's device. qo_indptr starts at zero, is
        nondecreasing, and ends at the packed Q extent. Page-table columns are contiguous;
        rows may have non-overlapping padding. Active page IDs must index both caches.
        Active lengths must respect the supplied bounds and selected model promises, with
        Sq[b] <= Sk[b] for causal attention.

        When ``plan`` selected ``uniform_packed_lengths=True``, every Q delta
        must equal ``max_q_len`` and every K/V length must equal
        ``max_kv_len``. For a causal plan with ``has_q_offset=False``, every
        request must additionally satisfy ``Sq[b] == Sk[b]``.

        Scale overrides are finite positive host scalars in the same units as the
        contiguous wrapper. Enabled attention_sinks is a compact CUDA FP32 [Hq]
        vector of finite or -inf natural-log sink logits after QK scaling, with zero
        V contribution. Sink values and cache zero-tail promises are caller-owned."""

        if not isinstance(validate, bool):
            raise TypeError("validate must be a bool")

        state = self._plan_state
        if state is None:
            raise RuntimeError("plan() must be called before run()")
        geometry = state.geometry
        if self._kv_layout == "NHD":
            k_cache = k_cache.transpose(1, 2)
            v_cache = v_cache.transpose(1, 2)
        if validate:
            _validate_paged_runtime_inputs(q, k_cache, v_cache, geometry)
            actual_q, actual_kv = _validate_paged_runtime_metadata(
                qo_indptr,
                block_tables,
                seq_lens_kv,
                geometry,
                total_q=int(q.shape[0]),
                num_physical_pages=int(k_cache.shape[0]),
            )
            from .decode import resolve_max_sequence_length

            max_q_len = resolve_max_sequence_length(
                max_q_len,
                actual_q,
                "max_q_len",
                geometry.max_q_len,
            )
            max_kv_len = resolve_max_sequence_length(
                max_kv_len, actual_kv, "max_kv_len", geometry.max_kv_len
            )
        else:
            if max_q_len is None:
                max_q_len = geometry.max_q_len
            if max_kv_len is None:
                max_kv_len = geometry.max_kv_len
        scale_softmax_log2 = (
            state.scale_softmax_log2
            if softmax_scale is None
            else _validate_scale(
                _validate_scale(softmax_scale, "softmax_scale") * math.log2(math.e),
                "softmax_scale * log2(e)",
            )
        )
        output_scale = (
            state.output_scale
            if output_scale is None
            else _validate_scale(output_scale, "output_scale")
        )

        if out is None:
            out = torch.empty(tuple(q.shape), dtype=geometry.out_dtype, device=q.device)
        elif validate:
            out = _prepare_out(out, q=q, out_dtype=geometry.out_dtype)
        lse = prepare_lse(
            lse,
            q=q,
            mode=state.lse_mode,
            return_lse=return_lse,
            validate=validate,
        )
        attention_sinks = validate_attention_sinks(
            attention_sinks,
            q=q,
            enabled=state.use_attention_sinks,
            validate=validate,
        )
        state.compiled(
            q,
            k_cache,
            v_cache,
            out,
            lse,
            attention_sinks,
            scale_softmax_log2,
            output_scale,
            qo_indptr,
            block_tables,
            seq_lens_kv,
            max_q_len,
            max_kv_len,
        )
        return (out, lse) if return_lse else out


@flashinfer_experimental_api
def batch_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    qo_indptr: Optional[torch.Tensor] = None,
    kv_indptr: Optional[torch.Tensor] = None,
    mask_type: Literal["dense", "causal", "variable_window"] = "dense",
    window_left: int = -1,
    variable_window_token_starts: Optional[torch.Tensor] = None,
    variable_window_token_ends: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    output_scale: float = 1.0,
    out_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
    vc: Optional[VCAttentionParams] = None,
    vc_config: Optional[VCAttentionConfig] = None,
    lse: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    attention_sinks: Optional[torch.Tensor] = None,
    lse_mode: Literal["base2"] = "base2",
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run one-shot fixed or packed-ragged task-scheduled context attention.

    Fixed tensors use ``[B, S, H, D]`` storage. Providing both cumulative
    int32 offset tensors selects packed ``[total_tokens, H, D]`` storage.
    Equal Q/K/V head dimensions may be 128, 192, or 256. The dtype contract
    is shared by all three dimensions. Separate Q/K=192 and
    V=128 supports non-absorbed MLA; O uses V's head dimension.
    Causal masking is bottom-right aligned.  ``window_left=-1`` disables the
    left window; a positive value selects the private head-paired GQA policy
    and retains at most ``window_left + 1`` keys at each causal row, including
    when ``S_q < S_kv``.

    Parameters
    ----------
    q, k, v : torch.Tensor
        Fixed or packed query, key, and value tensors. Q and K share one
        dtype; V matches it or is ``torch.float8_e4m3fn`` with
        ``torch.bfloat16`` Q/K.
    qo_indptr, kv_indptr : torch.Tensor, optional
        Cumulative query and K/V offsets for packed-ragged input.
    mask_type : {"dense", "causal", "variable_window"}
        Attention mask mode. ``variable_window`` is supported only for
        fixed-shape inputs.
    window_left : int
        Left sliding-window extent, or ``-1`` to disable the window.
    variable_window_token_starts, variable_window_token_ends : torch.Tensor, optional
        Inclusive per-query K bounds required for ``variable_window``. Both
        must be CUDA int32 tensors shaped ``[B, Sq]`` and satisfy
        ``0 <= starts[b, q] <= ends[b, q] < Sk``.
        Ends must be nondecreasing along Q within each batch item; starts
        need not be ordered.
    softmax_scale : float, optional
        Softmax scale; defaults to the inverse square root of head size.
    output_scale : float
        Scale applied to the attention output.
    out_dtype : torch.dtype, optional
        Output dtype; defaults to ``out.dtype`` when ``out`` is provided,
        otherwise the query dtype.
    out : torch.Tensor, optional
        Caller-owned output tensor.

    Returns
    -------
    torch.Tensor
        The fixed or packed attention output.

    ``return_lse=True`` returns ``(output, lse)``. ``lse_mode`` selects
    ``"base2"`` (the only supported base); it is used only when returning LSE
    or writing a caller-provided ``lse`` buffer.

    ``lse`` is an optional independent, compact float32 output with shape
    ``query.shape[:-1]`` (``q.shape[:-1]`` for FMHA). An enabled LSE plan
    allocates it when omitted, including when ``return_lse=False``. A supplied
    buffer is written regardless of ``return_lse``; the flag only controls
    the Python return type. LSE includes any sink mass and excludes output
    scaling. Keep LSE disjoint from output, inputs, and workspace. Supply
    both output buffers and disable validation for allocation-free capture.

    ``attention_sinks`` is a compact CUDA float32 vector ``[Hq]`` shared
    across batch/query positions. Values are natural-log sink logits after
    QK scaling, with zero V contribution. Finite values and ``-inf`` are
    accepted. Validation rejects NaN/positive infinity and may synchronize.
    With validation disabled, the caller must preserve these contracts.
    Sink values may change between completed CUDA graph replays.
    """

    resolved_out_dtype = (
        out.dtype if out_dtype is None and isinstance(out, torch.Tensor) else out_dtype
    )
    if resolved_out_dtype is None:
        if not isinstance(q, torch.Tensor):
            raise TypeError("q must be a torch.Tensor")
        resolved_out_dtype = q.dtype
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "batch_prefill cannot derive host plan bounds during CUDA graph "
            "capture; plan BatchPrefillTSWrapper before capture"
        )
    if vc_config is not None and vc is None:
        raise ValueError("vc_config requires the vc operands of this call")
    if (vc_config is not None and vc_config.repair_tiles) or (
        vc_config is None and vc is not None and vc.tile_means is None
    ):
        # The K/V rows of V repair operands include the repair tiles, so the
        # plan bounds cannot be derived from the operands alone. Without a
        # config, operands with no tile means may be repair operands.
        raise NotImplementedError(
            "batch_prefill does not run V repair operands; plan "
            "BatchPrefillTSWrapper with VCAttentionConfig(repair_tiles=...), or "
            "pass vc_config=VCAttentionConfig(demean=False) for plain operands"
        )
    if vc is not None and vc_config is None:
        vc_config = VCAttentionConfig(demean=vc.tile_means is not None)
    geometry = _resolve_geometry(
        q,
        k,
        v,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        mask_type=mask_type,
        window_left=window_left,
        out_dtype=resolved_out_dtype,
    )
    wrapper = BatchPrefillTSWrapper()
    wrapper.plan(
        device=geometry.device,
        batch_size=geometry.batch_size,
        max_q_len=geometry.max_q_len,
        max_kv_len=geometry.max_seq_len_k,
        num_qo_heads=geometry.num_qo_heads,
        num_kv_heads=geometry.num_kv_heads,
        head_dim=geometry.head_dim,
        head_dim_vo=geometry.head_dim_vo,
        q_dtype=geometry.qk_dtype,
        k_dtype=geometry.qk_dtype,
        v_dtype=geometry.pv_dtype,
        out_dtype=geometry.out_dtype,
        packed_query=geometry.packed_query,
        mask_type=mask_type,
        window_left=window_left,
        softmax_scale=softmax_scale,
        output_scale=output_scale,
        vc_config=vc_config,
        use_attention_sinks=attention_sinks is not None,
        lse_mode=validate_lse_mode(lse_mode)
        if return_lse or lse is not None
        else "none",
    )
    return wrapper.run(
        q,
        k,
        v,
        qo_indptr,
        kv_indptr,
        variable_window_token_starts=variable_window_token_starts,
        variable_window_token_ends=variable_window_token_ends,
        max_q_len=geometry.max_q_len,
        max_kv_len=geometry.max_seq_len_k,
        out=out,
        vc=vc,
        lse=lse,
        return_lse=return_lse,
        attention_sinks=attention_sinks,
    )


@flashinfer_experimental_api
def batch_prefill_with_paged_kv_cache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    qo_indptr: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens_kv: torch.Tensor,
    *,
    page_size: int = _DEFAULT_PAGED_KV_PAGE_SIZE,
    kv_layout: Literal["HND", "NHD"] = "HND",
    mask_type: Literal["dense", "causal"] = "dense",
    window_left: int = -1,
    softmax_scale: Optional[float] = None,
    output_scale: float = 1.0,
    out_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    attention_sinks: Optional[torch.Tensor] = None,
    lse_mode: Literal["base2"] = "base2",
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run one-shot packed-Q context attention over HND or NHD page pools.

    Q/output use ``[total_q, Hq, D]`` storage and K/V each use
    ``[num_pages, Hkv, page_size, D]`` for HND or
    ``[num_pages, page_size, Hkv, D]`` for NHD, with page size 4/8/16/32/64/128.
    ``qo_indptr`` describes Q rows, ``block_tables`` supplies one fixed
    row-strided page table, and ``seq_lens_kv`` supplies logical K/V lengths.
    Physical page indices need not be identity ordered. ``D`` may be 128 or
    256. Q and K must share one supported dtype; V matches it or is
    ``torch.float8_e4m3fn`` alongside ``torch.bfloat16`` Q/K.

    This convenience API reads request metadata on the host to derive exact
    plan bounds and is not CUDA-graph-capturable. Capture-sensitive callers
    should plan :class:`BatchPrefillPagedTSWrapper` before capture and bind the
    same metadata tensors directly in ``run``.

    Parameters
    ----------
    q : torch.Tensor
        Packed query tensor.
    k_cache : torch.Tensor
        HND key page pool.
    v_cache : torch.Tensor
        HND value page pool isomorphic to ``k_cache``.
    qo_indptr : torch.Tensor
        Cumulative packed-query offsets.
    block_tables : torch.Tensor
        Physical page IDs with row-strided shape ``[B, C]``.
    seq_lens_kv : torch.Tensor
        Logical K/V lengths with shape ``[B]``.
    page_size : int
        Number of K/V tokens stored in each page.
    kv_layout : {"HND", "NHD"}
        Layout of the separate K and V page pools.
    mask_type : {"dense", "causal"}
        Attention mask mode.
    window_left : int
        Left sliding-window extent, or ``-1`` to disable the window.
    softmax_scale : float, optional
        Softmax scale; defaults to the inverse square root of head size.
    output_scale : float
        Scale applied to the attention output.
    out_dtype : torch.dtype, optional
        Output dtype; defaults to ``out.dtype`` when ``out`` is provided,
        otherwise the query dtype.
    out : torch.Tensor, optional
        Caller-owned output tensor.

    Returns
    -------
    torch.Tensor
        The packed attention output.

    ``return_lse=True`` returns ``(output, lse)``. ``lse_mode`` selects
    ``"base2"`` (the only supported base); it is used only when returning LSE
    or writing a caller-provided ``lse`` buffer.

    ``lse`` is an optional independent, compact float32 output with shape
    ``query.shape[:-1]`` (``q.shape[:-1]`` for FMHA). An enabled LSE plan
    allocates it when omitted, including when ``return_lse=False``. A supplied
    buffer is written regardless of ``return_lse``; the flag only controls
    the Python return type. LSE includes any sink mass and excludes output
    scaling. Keep LSE disjoint from output, inputs, and workspace. Supply
    both output buffers and disable validation for allocation-free capture.

    ``attention_sinks`` is a compact CUDA float32 vector ``[Hq]`` shared
    across batch/query positions. Values are natural-log sink logits after
    QK scaling, with zero V contribution. Finite values and ``-inf`` are
    accepted. Validation rejects NaN/positive infinity and may synchronize.
    With validation disabled, the caller must preserve these contracts.
    Sink values may change between completed CUDA graph replays.
    """

    resolved_out_dtype = (
        out.dtype if out_dtype is None and isinstance(out, torch.Tensor) else out_dtype
    )
    if resolved_out_dtype is None:
        if not isinstance(q, torch.Tensor):
            raise TypeError("q must be a torch.Tensor")
        resolved_out_dtype = q.dtype
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "batch_prefill_with_paged_kv_cache cannot derive host plan bounds "
            "during CUDA graph capture; plan BatchPrefillPagedTSWrapper before "
            "capture"
        )
    _validate_kv_layout(kv_layout)
    logical_k = k_cache.transpose(1, 2) if kv_layout == "NHD" else k_cache
    logical_v = v_cache.transpose(1, 2) if kv_layout == "NHD" else v_cache
    geometry = _resolve_paged_geometry(
        q,
        logical_k,
        logical_v,
        qo_indptr=qo_indptr,
        block_tables=block_tables,
        seq_lens_kv=seq_lens_kv,
        page_size=page_size,
        mask_type=mask_type,
        window_left=window_left,
        out_dtype=resolved_out_dtype,
    )
    wrapper = BatchPrefillPagedTSWrapper(kv_layout=kv_layout)
    wrapper.plan(
        device=geometry.device,
        batch_size=geometry.batch_size,
        max_q_len=geometry.max_q_len,
        max_kv_len=geometry.max_kv_len,
        num_qo_heads=geometry.num_qo_heads,
        num_kv_heads=geometry.num_kv_heads,
        head_dim=geometry.head_dim,
        q_dtype=geometry.qk_dtype,
        k_dtype=geometry.qk_dtype,
        v_dtype=geometry.pv_dtype,
        out_dtype=geometry.out_dtype,
        page_size=page_size,
        mask_type=mask_type,
        window_left=window_left,
        softmax_scale=softmax_scale,
        output_scale=output_scale,
        use_attention_sinks=attention_sinks is not None,
        lse_mode=validate_lse_mode(lse_mode)
        if return_lse or lse is not None
        else "none",
        uniform_packed_lengths=False,
        has_q_offset=geometry.has_q_offset,
        paged_v_tail_is_zero=geometry.paged_v_tail_is_zero,
    )
    return wrapper.run(
        q,
        k_cache,
        v_cache,
        qo_indptr,
        block_tables,
        seq_lens_kv,
        max_q_len=geometry.max_q_len,
        max_kv_len=geometry.max_kv_len,
        out=out,
        lse=lse,
        return_lse=return_lse,
        attention_sinks=attention_sinks,
    )


__all__ = [
    "BatchPrefillPagedTSWrapper",
    "BatchPrefillTSWrapper",
    "batch_prefill",
    "batch_prefill_with_paged_kv_cache",
]
