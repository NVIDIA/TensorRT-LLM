# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Thin plan/run adapter for PrimTS attention over persistent CSA2 packed KV."""

import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import cache
from hashlib import sha256
from pathlib import Path

import cutlass
import cutlass.cute as cute
import torch

from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

from ...prims_ts._compile_cache import build_and_load_attention_kernel
from ...prims_ts.decode import (
    _append_workspace_section,
    _resolve_cuda_device,
    _validate_workspace_buffer,
    _workspace_section_view,
    _WorkspaceSection,
)
from ...prims_ts.kernels.mla_decode.helpers.nvfp4_kv import is_sm107_target
from ...prims_ts.kernels.mla_decode.sparse_policy import SparseMlaProfile, select_sparse_mla_profile
from ...prims_ts.kernels.mla_decode.throughput_latency_1cta.config import (
    MlaProfile,
    compute_workspace_size,
)
from ...prims_ts.kernels.mla_decode.throughput_latency_1cta.kernel import (
    ThroughputLatencyMlaDecodeTs,
)
from ...prims_ts.mla_decode import (
    _fake_output_quant_args,
    _get_compiled_mla_decode,
    _make_mla_decode_compile_spec,
    _MLADecodeCompileSpec,
    _MLADecodeLaunchSpec,
)
from ._prims_ts_output import MXFP8OutputQuant, mxfp8_output_args
from ._prims_ts_producer import PackedKvProducer, main_row_bytes
from ._prims_ts_quantization import UnitScaleQuery
from .params import PrimsTSMainFormat, SwaCacheFormat
from .prims_ts import _validate_softmax_scale
from .quantization import row_bytes

__all__ = [
    "PackedSparseMLA",
    "decodes_nvfp4_main",
    "fuses_mxfp8_output",
    "get_packed_sparse_mla_workspace_size",
    "precompile_packed_sparse_mla",
]

_COMPILE_OPTIONS = "--enable-tvm-ffi --opt-level 2"
# Split-KV producer plans merge their splits in the attention kernel's last CTA per query
# tile instead of a separate reducer launch (the same arithmetic, so the same output).
_FUSED_SPLIT_FINISH = False
# Split-KV plans launch each query tile and V partition's splits as one cluster whose first CTA
# merges them, instead of a reducer launch (the same arithmetic, so the same output).
_CLUSTER_SPLIT_FINISH = True
# Largest V partition count per query tile and split.
_MAX_HEAD_DIM_CTAS = 4
# The first CTA gathers every split's partials of its rows and V columns in shared memory
# beside the kernel's own buffers (one Q stage, eight K/V slices, two P tiles and the page
# offsets): this many bytes fit next to them.
_CLUSTER_SPLIT_PUSH_LIMIT = 44 * 1024
_HEAD_DIM = 512
_QUERY_TILE = 64
_KV_TILE = 128


@dataclass(frozen=True)
class _CompileSpec:
    device_index: int
    kernel_signature: tuple
    heads: int
    max_topk: int
    max_extra_topk: int
    has_sinks: bool
    split_kv: int
    has_workspace: bool
    swa_cache_format: SwaCacheFormat
    main_cache_format: PrimsTSMainFormat
    # O-A groups of an MXFP8 output kernel; None stores BF16.
    output_groups: int | None
    kernel: ThroughputLatencyMlaDecodeTs = field(compare=False, hash=False, repr=False)
    # INT64 SWA slots and logical MAIN positions mapped in the kernel (PackedKvProducer).
    raw_selections: bool = False
    # The kernel rotates and rounds the BF16 query itself (PackedKvProducer.query_rope).
    query_rope: bool = False
    # A runahead kernel's MAIN selections and rows were final before its preceding launch;
    # False waits for that launch before reading them (PackedKvProducer.main_after_wait).
    main_ready: bool = True


@dataclass(frozen=True)
class _GatherLaunch:
    """A native kernel without the packed producer: TMA gather4 over unit-scale E4M3 rows
    of both sources, or, for NVFP4 MAIN records, the native NVFP4 loader schedule."""

    profile: SparseMlaProfile
    spec: _MLADecodeCompileSpec
    core_bytes: int


@dataclass(frozen=True)
class _Launch:
    device: torch.device
    profile: SparseMlaProfile
    kernel: ThroughputLatencyMlaDecodeTs
    sections: dict[str, _WorkspaceSection]
    workspace_bytes: int
    # Split-KV scratch of the packed producer kernel; "core" also fits the native kernels'.
    core_bytes: int
    gather: _GatherLaunch | None = None
    nvfp4: _GatherLaunch | None = None


@dataclass(frozen=True)
class _GatherPlan:
    compiled: Callable[..., object]
    split_kv: int
    core: torch.Tensor | None


@dataclass(frozen=True)
class _Plan:
    device: torch.device
    rows: int
    heads: int
    max_topk: int
    max_extra_topk: int
    has_sinks: bool
    split_kv: int
    swa_cache_format: SwaCacheFormat
    main_cache_format: PrimsTSMainFormat
    valid_prefix: bool
    output_groups: int | None
    spec: _CompileSpec
    raw_selections: bool
    query_rope: bool
    # The kernel runs ahead of its preceding launch (``plan``'s ``inputs_final``).
    inputs_final: bool
    # ... with its MAIN reads after the wait for that launch (``plan``'s ``main_ready``).
    main_ready: bool
    # The packed producer kernel; a gather or NVFP4 plan compiles it only for a
    # pool whose rows are not dense records of their format.
    compiled: Callable[..., object] | None
    gather: _GatherPlan | None
    nvfp4: _GatherPlan | None
    query: UnitScaleQuery
    buffers: dict[str, torch.Tensor]
    core: torch.Tensor | None
    lse_kernel: torch.Tensor
    dummy_main: torch.Tensor


def _validate_geometry(rows: int, heads: int, max_topk: int, max_extra_topk: int) -> int:
    for name, value in (("rows", rows), ("heads", heads), ("max_topk", max_topk)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if (
        not isinstance(max_extra_topk, int)
        or isinstance(max_extra_topk, bool)
        or max_extra_topk < 0
    ):
        raise ValueError("max_extra_topk must be a nonnegative integer")
    if heads > 128:
        raise ValueError("packed sparse MLA supports at most 128 query heads")
    capacity = max(256, ((max_topk + 127) // 128 + (max_extra_topk + 127) // 128) * 128)
    if rows * heads >= 2**31 or capacity >= 2**31 - 32768:
        raise ValueError("packed sparse MLA geometry exceeds the int32 kernel domain")
    return capacity


def _cluster_push_bytes(splits: int, heads: int, head_dim_ctas: int) -> int:
    """Shared memory the cluster merge gathers: per row and split, the BF16 partial O of the
    V columns and the FP32 LSE."""
    return splits * min(heads, _QUERY_TILE) * (_HEAD_DIM // head_dim_ctas * 2 + 4)


@cache
def _select_packed_profile(
    rows: int, heads: int, capacity: int, sm_count: int, inputs_final: bool = False
) -> SparseMlaProfile:
    """Choose only the producer-supported family, including padded narrow heads.

    ``inputs_final``: the query, selections, page table and MAIN rows were final before the
    launch preceding this one (see ``SparseMlaProfile.inputs_final``).
    """
    work_tiles = rows * ((heads + _QUERY_TILE - 1) // _QUERY_TILE)
    kv_tiles = (capacity + _KV_TILE - 1) // _KV_TILE
    partitions = min(kv_tiles, max(1, sm_count // work_tiles), 128)
    splits = (kv_tiles + (kv_tiles + partitions - 1) // partitions - 1) // (
        (kv_tiles + partitions - 1) // partitions
    )
    # Split V over more CTAs while the grid still fits one wave. Each V owner
    # reads its own D128 slices of the retained K tile, so only the PV,
    # correction and output work divides; a second wave would cost more.
    head_dim_ctas = next(
        (v for v in (4, 2) if v <= _MAX_HEAD_DIM_CTAS and work_tiles * splits * v <= sm_count), 1
    )
    if inputs_final:
        # A plan that runs ahead of the preceding launch overlaps it only on free SMs (one CTA
        # fills an SM), so its grid keeps to half of them where the cluster merge still fits.
        head_dim_ctas = next(
            (
                v
                for v in (4, 2, 1)
                if v <= _MAX_HEAD_DIM_CTAS
                and work_tiles * splits * v <= sm_count // 2
                and _cluster_push_bytes(splits, heads, v) <= _CLUSTER_SPLIT_PUSH_LIMIT
            ),
            head_dim_ctas,
        )
    # Beyond one resident wave (one CTA per SM), every tile has one split and one V
    # owner, and the static persistent scheduler overlaps each tile's prologue and
    # epilogue with its CTA's next tile. It also lifts the nonpersistent grid.z
    # limit of 65,535 rows. Persistent queues do not enumerate KV splits.
    persistent = splits == 1 and head_dim_ctas == 1 and work_tiles > sm_count
    push_bytes = _cluster_push_bytes(splits, heads, head_dim_ctas)
    return SparseMlaProfile(
        fused_split_finish=_FUSED_SPLIT_FINISH and splits > 1,
        cluster_split_finish=_CLUSTER_SPLIT_FINISH
        and splits > 1
        and push_bytes <= _CLUSTER_SPLIT_PUSH_LIMIT,
        inputs_final=inputs_final,
        family="keep",
        tile_size_q=_QUERY_TILE,
        split_kv=splits,
        scheduler="static" if persistent else "nonpersistent",
        reduction="gmem_separate",
        head_dim_ctas=head_dim_ctas,
        # The producer's issuer warps copy and decode rows themselves rather than
        # via TMA, so each tile's latency scales with the quads one issuer handles.
        gather_issue_warps=8,
        # Keep the native E4M3 KV8/Q2 defaults: an explicit retained-stage
        # override also changes native Q staging.
        reuse_kv=True,
        # As the native Keep-AB policy does: issuers share one cached copy of a
        # tile's sparse coordinates, and correction warps rescale paired rows.
        uniform_offset_cache=True,
        paired_correction=True,
    )


def _quantizes_output(profile: SparseMlaProfile) -> bool:
    """Whether the main kernel finalizes whole head rows, which it can then store as MXFP8.

    Split-KV schedules finalize in their reduction kernel instead.
    """
    return profile.split_kv == 1 and profile.head_dim_ctas == 1


def _launch_dependent(kernel: ThroughputLatencyMlaDecodeTs) -> None:
    """Make ``kernel`` a programmatic dependent launch (unless TRTLLM_ENABLE_PDL=0).

    It may start while the preceding kernel of its stream still runs. A persistent
    grid then prepares its barriers and TMEM before waiting for that kernel, and
    every grid reads and writes global memory only after the wait.
    """
    if get_env_enable_pdl():
        kernel.use_pdl = True
        kernel.pdl_wait_after_prologue = True


@cache
def _fuses_mxfp8_output(device_index: int, rows: int, heads: int, capacity: int) -> bool:
    # Only the 64-head WS gather schedule stages the quantized output for TMA stores.
    # Elsewhere the quantizing epilogue runs per lane and costs about what the
    # separate inverse-RoPE pass it replaces does.
    sm_count = torch.cuda.get_device_properties(device_index).multi_processor_count
    gather = _select_gather_profile(rows, heads, capacity, sm_count)
    return gather is not None and gather.ws_tma_store and _quantizes_output(gather)


def fuses_mxfp8_output(
    device: torch.device, rows: int, heads: int, max_topk: int, max_extra_topk: int
) -> bool:
    """Whether a plan of this geometry can write the grouped MXFP8 O-A input itself."""
    capacity = _validate_geometry(rows, heads, max_topk, max_extra_topk)
    device, _ = _resolve_cuda_device(device)
    return _fuses_mxfp8_output(device.index, rows, heads, capacity)


def gathers_plain_rows(
    swa_cache_format: SwaCacheFormat, main_cache_format: PrimsTSMainFormat, max_extra_topk: int
) -> bool:
    """Whether every selected row is stored as its own unit-scale E4M3 operand."""
    return swa_cache_format == "swa_fp8" and (
        max_extra_topk == 0 or main_cache_format == "main_fp8"
    )


# The WS gather schedule's page-offset prefetch depth, the quantized tail's stripes
# held across the O release, and the D128 K/V stages it retains for plans with MAIN
# rows once Q is staged in the K/V ring (see _select_gather_profile).
_WS_PAGE_PIPELINE_STAGES = 2
_WS_TAIL_STAGED_STRIPES = 5
_WS_MULTI_TILE_KV_STAGES = 10


@cache
def _select_gather_profile(
    rows: int, heads: int, capacity: int, sm_count: int
) -> SparseMlaProfile | None:
    """The native TMA gather4 schedule for plain E4M3 rows, or None to keep the producer.

    Gather4 copies four rows per TMA instruction where the producer's issuer
    warps copy 16 bytes per lane; it runs the producer's own schedule.

    A 64-head plan of one KV split and one V owner takes it at every grid size,
    with the warp-specialized M64 MMAs, Q copied into TMEM, and TMA tensor
    stores of the output. Outputs are bitwise equal to the producer's (GB300:
    SWA + 512 MAIN, 16384 rows 1031 -> 873 us and 8064 rows 512 -> 452 us;
    SWA-only, 16384 rows 435 -> 328 us and 8064 rows 238 -> 182 us; B200,
    CUDA-graph replays: SWA + 512 MAIN, 75-148 rows 22-23 -> 11.5-12 us;
    SWA-only, 140 rows 12.2 -> 7.0 us and 296 rows 20.0 -> 9.5 us).

    Other plans keep the plain gather schedule, which wins once each persistent
    CTA owns more than four KV tiles (GB300, H64: SWA + 512 MAIN, 153 rows
    62 -> 53 us and 8064 rows 915 -> 492 us; SWA-only, 608 rows 52 -> 49 us)
    and loses on fewer (SWA-only, 304 rows: 39 -> 43 us) or within one
    resident wave (140 rows: 33 -> 44 us); retaining ten KV stages or CLC
    persistence is slower. Sixteen heads keep it, as the WS layout does not
    beat it, and split plans keep the producer.

    The WS schedule prefetches two pages of offsets, and its quantizing tail holds
    five of a lane's eight output stripes as BF16 words across the O release, so
    the next tile's first PV does not wait for the whole epilogue (GB300, SWA + 512
    MAIN, fused MXFP8 output, 16384 rows with real-like scores: 806 -> 768 us; six
    stripes or more cost registers on rescale-heavy inputs). With MAIN rows (a
    capacity above the 256-row minimum of SWA-only plans) it also stages Q in two
    K/V ring slices instead of its own 32 KB stage and retains ten D128 K/V stages
    instead of eight, so the next tile's K is resident before the MMA warp needs
    it (16384 rows 760 -> 689 us real-like, 788 -> 772 us random; 8064 rows
    381 -> 349 us). SWA-only plans keep their own Q stage, which is faster for
    them (16384 rows: 388 against 398 us).
    """
    profile = _select_packed_profile(rows, heads, capacity, sm_count)
    if heads == _QUERY_TILE and _quantizes_output(profile):
        multi_tile = (
            dict(ws_q_in_kv=True, reuse_kv_stages=_WS_MULTI_TILE_KV_STAGES)
            if capacity > 2 * _KV_TILE
            else {}
        )
        return replace(
            profile,
            ws_mma=True,
            ws_store_cols=64,
            ws_tma_store=True,
            page_pipeline_stages=_WS_PAGE_PIPELINE_STAGES,
            ws_tail_staged_stripes=_WS_TAIL_STAGED_STRIPES,
            **multi_tile,
        )
    work_tiles = rows * ((heads + _QUERY_TILE - 1) // _QUERY_TILE)
    kv_tiles = (capacity + _KV_TILE - 1) // _KV_TILE
    if profile.scheduler == "nonpersistent" or work_tiles * kv_tiles <= 4 * sm_count:
        return None
    return profile


def _gather_launch(
    device: torch.device,
    rows: int,
    heads: int,
    capacity: int,
    sm_count: int,
    max_topk: int,
    max_extra_topk: int,
    output_quant: MXFP8OutputQuant | None = None,
) -> _GatherLaunch | None:
    """Build the native Keep-AB kernel whose TMA producer gathers four rows per copy.

    The kernel reads both pools as dense E4M3 [rows, 512] storage. It trusts every
    entry before a live length to be a valid row of its pool, as gather4 fills rows
    beyond a pool with zeros instead of masking them. With ``output_quant`` it stores
    the grouped MXFP8 O-A input instead of BF16.
    """
    profile = _select_gather_profile(rows, heads, capacity, sm_count)
    if profile is None:
        return None
    kernel = ThroughputLatencyMlaDecodeTs(
        batch_size=rows,
        num_heads=_QUERY_TILE,
        seq_len_q=(heads + _QUERY_TILE - 1) // _QUERY_TILE,
        seq_len_k=capacity,
        rope_dim=0,
        page_size=1,
        max_active_clusters=sm_count,
        qkv_dtype="e4m3",
        out_dtype="bf16",
        profile=MlaProfile(
            name="csa2_gather_e4m3_keep_v1",
            kernel_variant="keeps_mma_ab",
            tile_size_q=_QUERY_TILE,
            num_ctas_per_head_dim=profile.head_dim_ctas,
            num_ctas_per_seq_kv=profile.split_kv,
            use_multi_ctas_kv=int(profile.split_kv > 1),
            use_cluster_reduction=0,
            use_persistent_scheduler=int(profile.scheduler != "nonpersistent"),
            use_clc_dynamic_persistent_scheduler=int(profile.scheduler == "clc"),
        ),
        reduction_mode=profile.reduction,
        logical_num_heads=heads,
        logical_seq_len_q=1,
        tile_size_q=_QUERY_TILE,
        mask_type="dense",
        sparse_profile=profile,
    )
    # Sized before the sparse launch flags are set, as the native sparse plan does.
    core_bytes = compute_workspace_size(
        cfg=kernel._make_config(),
        partial_o_dtype=cutlass.BFloat16,
        lse_dtype=cutlass.Float32,
    )
    kernel.finalize_output = True
    kernel.direct_sparse = True
    kernel.direct_sparse_capacities = (max_topk, max_extra_topk)
    kernel.assume_valid_prefix = True
    kernel.output_quant = output_quant
    _launch_dependent(kernel)
    spec = _make_mla_decode_compile_spec(
        _MLADecodeLaunchSpec(kernel, (), core_bytes, profile.split_kv),
        device_index=device.index,
        num_qo_heads=heads,
        kv_lora_rank=_HEAD_DIM,
        qk_rope_head_dim=0,
        page_size=1,
        q_dtype_key="float8_e4m3fn",
        out_dtype_key="bfloat16" if output_quant is None else "float8_e4m3fn",
        max_q_len=1,
        packed_query=False,
    )
    return _GatherLaunch(profile, spec, core_bytes)


@cache
def _select_nvfp4_profile(
    rows: int, heads: int, capacity: int, sm_count: int, compute_capability: tuple[int, int] | None
) -> SparseMlaProfile | None:
    """The native schedule that decodes NVFP4 MAIN records in the kernel, or None.

    With unit-scale E4M3 SWA rows (``swa_fp8``) and NVFP4 MAIN records, the native
    sparse plan of an FP8 primary pool with an NVFP4 extra pool reads both formats as
    stored: its loader decodes each selected MAIN record into the E4M3 operand stage,
    so no selection bank is decoded or read back. Outputs are bitwise equal to the
    selection bank's wherever that also runs one KV split. GB300, 64 heads, SWA 128 +
    512 MAIN, CUDA-graph replays, us (bank = compaction, MAIN decode and gather
    attention; native fused decode in parentheses), independent selections: 150 rows 31.9 -> 23.2 (24.5), 252 rows
    52.1 -> 35.4 (39.0), 504 rows 90.7 -> 63.0 (81.7), 1252 rows 197.9 -> 126.4
    (163.0); query rows of a request sharing 80% of their selections: 150 rows
    28.8 -> 22.5, 1252 rows 130.8 -> 122.7.

    Only the native one-CTA Keep-AB family is taken, for 64 heads, whose single-split
    plans also fuse the grouped MXFP8 output. Below half a wave of rows the native
    policy divides each row across CTAs (V over two or four CTAs, or five KV splits,
    with four issuers), which is slower here than one CTA per row: those grids take
    the schedule the policy picks for half a wave (GB300, 64 rows: 44.4 us divided,
    native fused decode 24.7 us; 76 rows on the one-CTA schedule 22.4 us).
    """
    if heads != _QUERY_TILE:
        return None
    profile = select_sparse_mla_profile(
        rows=max(rows, (sm_count + 1) // 2),
        heads=heads,
        query_length=1,
        capacity=capacity,
        dtype="fp8",
        sm_count=sm_count,
        rope_dim=0,
        kv_cache_format="fp8",
        compute_capability=compute_capability,
    )
    return profile if profile.family == "keep" else None


def _compute_capability(device: torch.device) -> tuple[int, int] | None:
    """The device's capability as the native sparse policy keys it."""
    capability = torch.cuda.get_device_capability(device)
    # SM107 schedules assume SM107 kernel resources, which only an SM107 target compiles.
    return None if capability == (10, 7) and not is_sm107_target() else capability


def decodes_nvfp4_main(
    device: torch.device | str | int,
    rows: int,
    heads: int,
    max_topk: int,
    max_extra_topk: int,
    quantized_output: bool = False,
) -> bool:
    """Whether a valid-prefix plan over ``swa_fp8`` SWA and NVFP4 MAIN of this geometry
    decodes the MAIN records in the attention kernel (see :func:`_select_nvfp4_profile`).

    ``quantized_output`` asks it of a plan that stores the grouped MXFP8 output.
    """
    if not max_extra_topk:
        return False
    capacity = _validate_geometry(rows, heads, max_topk, max_extra_topk)
    device, _ = _resolve_cuda_device(device)
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    profile = _select_nvfp4_profile(rows, heads, capacity, sm_count, _compute_capability(device))
    return profile is not None and (not quantized_output or _quantizes_output(profile))


def _nvfp4_launch(
    device: torch.device,
    rows: int,
    heads: int,
    capacity: int,
    sm_count: int,
    max_topk: int,
    max_extra_topk: int,
    output_quant: MXFP8OutputQuant | None = None,
) -> _GatherLaunch | None:
    """Build the native kernel over E4M3 SWA rows and NVFP4 MAIN records.

    Both pools are read as dense records: UINT8[rows, 512] SWA and UINT8[rows, 288]
    MAIN. Like the gather kernel, it trusts every entry before a live length to be a
    valid row of its pool. With ``output_quant`` it stores the grouped MXFP8 O-A
    input instead of BF16; a plan whose profile cannot store it gets no kernel.
    """
    profile = _select_nvfp4_profile(rows, heads, capacity, sm_count, _compute_capability(device))
    if profile is None or (output_quant is not None and not _quantizes_output(profile)):
        return None
    kernel = ThroughputLatencyMlaDecodeTs(
        batch_size=rows,
        num_heads=profile.tile_size_q,
        seq_len_q=(heads + profile.tile_size_q - 1) // profile.tile_size_q,
        seq_len_k=capacity,
        rope_dim=0,
        page_size=1,
        max_active_clusters=sm_count,
        qkv_dtype="e4m3",
        out_dtype="bf16",
        profile=MlaProfile(
            name="csa2_nvfp4_main_keep_v1",
            kernel_variant="keeps_mma_ab",
            tile_size_q=profile.tile_size_q,
            num_ctas_per_head_dim=profile.head_dim_ctas,
            num_ctas_per_seq_kv=profile.split_kv,
            use_multi_ctas_kv=int(profile.split_kv > 1),
            use_cluster_reduction=int(profile.reduction == "cluster"),
            use_persistent_scheduler=int(profile.scheduler != "nonpersistent"),
            use_clc_dynamic_persistent_scheduler=int(profile.scheduler == "clc"),
        ),
        reduction_mode=profile.reduction,
        logical_num_heads=heads,
        logical_seq_len_q=1,
        tile_size_q=profile.tile_size_q,
        mask_type="dense",
        sparse_profile=profile,
        kv_cache_format="fp8",
        extra_kv_cache_format="nvfp4",
    )
    core_bytes = compute_workspace_size(
        cfg=kernel._make_config(),
        partial_o_dtype=cutlass.BFloat16,
        lse_dtype=cutlass.Float32,
    )
    kernel.finalize_output = True
    kernel.direct_sparse = True
    kernel.direct_sparse_capacities = (max_topk, max_extra_topk)
    kernel.assume_valid_prefix = True
    kernel.output_quant = output_quant
    _launch_dependent(kernel)
    spec = _make_mla_decode_compile_spec(
        _MLADecodeLaunchSpec(kernel, (), core_bytes, profile.split_kv),
        device_index=device.index,
        num_qo_heads=heads,
        kv_lora_rank=_HEAD_DIM,
        qk_rope_head_dim=0,
        page_size=1,
        q_dtype_key="float8_e4m3fn",
        out_dtype_key="bfloat16" if output_quant is None else "float8_e4m3fn",
        max_q_len=1,
        packed_query=False,
    )
    return _GatherLaunch(profile, spec, core_bytes)


def _dense_rows(pool: torch.Tensor, width: int = _HEAD_DIM) -> bool:
    """Whether a UINT8[N,width] pool is the dense record storage the native kernels address."""
    return pool.shape[0] > 0 and pool.shape[1] == width and pool.stride(0) == width


def _gather_pool(pool: torch.Tensor) -> torch.Tensor:
    """Dense E4M3 rows as the native one-token-page [1, 512, rows] cache view."""
    return pool.view(torch.float8_e4m3fn).unsqueeze(1).permute(1, 2, 0)


def _record_pool(pool: torch.Tensor) -> torch.Tensor:
    """Dense UINT8 records as the native one-token-page [1, width, rows] cache view."""
    return pool.unsqueeze(1).permute(1, 2, 0)


def _resolve_launch(
    device: torch.device | str | int,
    rows: int,
    heads: int,
    max_topk: int,
    max_extra_topk: int,
    swa_cache_format: SwaCacheFormat = "swa",
    aligned_strides: bool = True,
    main_cache_format: PrimsTSMainFormat = "main",
    valid_prefix: bool = False,
    output_groups: int | None = None,
    raw_selections: bool = False,
    query_rope: bool = False,
    inputs_final: bool = False,
    main_ready: bool = True,
) -> _Launch:
    capacity = _validate_geometry(rows, heads, max_topk, max_extra_topk)
    if raw_selections and valid_prefix:
        raise ValueError("raw selections keep their holes and cannot promise a valid prefix")
    if query_rope and not raw_selections:
        raise ValueError("an in-kernel query RoPE needs a raw-selection plan")
    device, _ = _resolve_cuda_device(device)
    if torch.cuda.get_device_capability(device) not in ((10, 0), (10, 3), (10, 7)):
        raise NotImplementedError("CSA2 packed PrimTS requires SM100, SM103 or SM107")
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    profile = _select_packed_profile(rows, heads, capacity, sm_count, inputs_final)
    output_quant = None
    if output_groups is not None:
        if not _quantizes_output(profile):
            raise ValueError("CSA2 PrimTS MXFP8 output needs a single-split, single-V-CTA plan")
        output_quant = MXFP8OutputQuant(heads, output_groups)
    kernel_profile = MlaProfile(
        name="csa2_packed_e4m3_keep_v3",
        kernel_variant="keeps_mma_ab",
        tile_size_q=_QUERY_TILE,
        num_ctas_per_head_dim=profile.head_dim_ctas,
        num_ctas_per_seq_kv=profile.split_kv,
        use_multi_ctas_kv=int(profile.split_kv > 1),
        use_cluster_reduction=0,
        use_persistent_scheduler=int(profile.scheduler != "nonpersistent"),
    )
    kernel = ThroughputLatencyMlaDecodeTs(
        batch_size=rows,
        num_heads=_QUERY_TILE,
        seq_len_q=(heads + _QUERY_TILE - 1) // _QUERY_TILE,
        seq_len_k=capacity,
        latent_dim=_HEAD_DIM,
        rope_dim=0,
        page_size=1,
        max_active_clusters=sm_count,
        qkv_dtype="e4m3",
        out_dtype="bf16",
        profile=kernel_profile,
        reduction_mode=profile.reduction,
        logical_num_heads=heads,
        logical_seq_len_q=1,
        tile_size_q=_QUERY_TILE,
        mask_type="dense",
        sparse_profile=profile,
        finalize_output=True,
        kv_producer=PackedKvProducer(
            has_extra=max_extra_topk > 0,
            swa_cache_format=swa_cache_format,
            aligned_strides=aligned_strides,
            main_cache_format=main_cache_format,
            raw_selections=raw_selections,
            query_rope=query_rope,
            main_after_wait=inputs_final and not main_ready,
        ),
        # Compile the planned SWA + MAIN capacity; live lengths still arrive
        # with the metadata on every launch.
        fixed_seqlen=True,
    )
    kernel.direct_sparse = True
    kernel.direct_sparse_capacities = (max_topk, max_extra_topk)
    kernel.output_quant = output_quant
    core_bytes = compute_workspace_size(
        cfg=kernel._make_config(),
        partial_o_dtype=cutlass.BFloat16,
        lse_dtype=cutlass.Float32,
    )
    # Size the arena for every kernel whatever the formats, so one arena serves
    # every format, prefix promise and output of the geometry.
    gather = _gather_launch(
        device, rows, heads, capacity, sm_count, max_topk, max_extra_topk, output_quant
    )
    nvfp4 = (
        _nvfp4_launch(
            device, rows, heads, capacity, sm_count, max_topk, max_extra_topk, output_quant
        )
        if max_extra_topk
        else None
    )
    native_core = max(
        (launch.core_bytes for launch in (gather, nvfp4) if launch is not None), default=0
    )
    sections = {}
    byte_end = 0
    for name, shape, dtype in (
        ("core", (max(core_bytes, native_core),), torch.int8),
        ("lse", (rows, 1, heads), torch.float32),
        ("q8", (rows, 1, heads, _HEAD_DIM), torch.uint8),
        # One record wide enough for either MAIN format, so both share an arena.
        ("dummy_main", (1, main_row_bytes("main_fp8")), torch.uint8),
        # The gather ABI always binds an extra list and sinks: SWA-only launches
        # bind empty extra lists, and plans without sinks bind -inf logits.
        ("dummy_extra", (2, rows), torch.int32),
        ("no_sinks", (heads,), torch.float32),
        # Raw-selection launches: both lists' constant live widths, and an INT32 stand-in
        # for the page-offset operand the INT64 SWA slots cannot alias.
        ("raw_lengths", (2, rows), torch.int32),
        ("raw_offsets", (rows, max_topk), torch.int32),
    ):
        sections[name], byte_end = _append_workspace_section(byte_end, shape, dtype)
    if not (
        valid_prefix and gathers_plain_rows(swa_cache_format, main_cache_format, max_extra_topk)
    ):
        gather = None
    if not (valid_prefix and swa_cache_format == "swa_fp8" and main_cache_format == "main"):
        nvfp4 = None
    return _Launch(device, profile, kernel, sections, byte_end, core_bytes, gather, nvfp4)


def get_packed_sparse_mla_workspace_size(
    device: torch.device | str | int,
    rows: int,
    heads: int,
    *,
    max_topk: int = 128,
    max_extra_topk: int = 0,
    swa_cache_format: SwaCacheFormat = "swa",
    main_cache_format: PrimsTSMainFormat = "main",
) -> int:
    """Return bytes for the exact packed profile, without allocating or compiling.

    Includes native split-KV BF16 partial outputs/FP32 statistics for the packed
    producer kernel and, where the geometry has them, the TMA gather and NVFP4 MAIN
    kernels; FP32
    LSE and unit-scale E4M3 Q; one dummy record, as wide as either MAIN format,
    for launches without a MAIN pool; and the gather kernel's empty extra lists
    and -inf sinks. The size depends on neither the formats nor a prefix promise, and covers
    plans that run ahead of their preceding launch.
    """
    return max(
        _resolve_launch(
            device,
            rows,
            heads,
            max_topk,
            max_extra_topk,
            swa_cache_format,
            main_cache_format=main_cache_format,
            inputs_final=inputs_final,
        ).workspace_bytes
        for inputs_final in (False, True)
    )


def _compile_spec(
    launch: _Launch,
    heads: int,
    max_topk: int,
    max_extra_topk: int,
    has_sinks: bool,
    swa_cache_format: SwaCacheFormat,
    main_cache_format: PrimsTSMainFormat,
    raw_selections: bool = False,
    query_rope: bool = False,
    main_ready: bool = True,
) -> _CompileSpec:
    return _CompileSpec(
        device_index=launch.device.index,
        kernel_signature=launch.kernel.compile_signature(),
        heads=heads,
        max_topk=max_topk,
        max_extra_topk=max_extra_topk,
        has_sinks=has_sinks,
        split_kv=launch.profile.split_kv,
        has_workspace=launch.core_bytes > 0,
        swa_cache_format=swa_cache_format,
        main_cache_format=main_cache_format,
        output_groups=(
            None if launch.kernel.output_quant is None else launch.kernel.output_quant.num_groups
        ),
        kernel=launch.kernel,
        raw_selections=raw_selections,
        query_rope=query_rope,
        main_ready=main_ready,
    )


def precompile_packed_sparse_mla(
    device: torch.device | str | int,
    max_rows: int,
    heads: int,
    *,
    max_topk: int = 128,
    max_extra_topk: int = 0,
    has_sinks: bool = True,
    swa_cache_format: SwaCacheFormat = "swa",
    main_cache_format: PrimsTSMainFormat = "main",
    valid_prefix: bool = False,
    output_groups: int | None = None,
    raw_selections: bool = False,
    query_rope: bool = False,
    inputs_final: bool = False,
    main_ready: bool = True,
) -> int:
    """Compile every kernel that plans of 1 to ``max_rows`` rows select for these pools.

    The row count picks the split-KV partition count and its reduction
    topology, both compile-time kernel parameters, and, under a valid-prefix
    promise, the TMA gather kernel for plain E4M3 rows or the NVFP4 MAIN kernel
    for ``swa_fp8`` SWA with NVFP4 MAIN records instead of the packed producer
    kernel, so a plan for a row count first seen while serving would otherwise
    compile on the request path. Only dense records for the native kernels and
    16-byte-aligned producer rows are compiled. With ``output_groups``, compiles
    instead the MXFP8 output kernels of the row counts whose plans can store it
    (see :func:`fuses_mxfp8_output`). Returns the number of distinct kernels.
    """
    if not isinstance(max_rows, int) or isinstance(max_rows, bool) or max_rows <= 0:
        raise ValueError("max_rows must be a positive integer")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("precompile_packed_sparse_mla() must run before CUDA Graph capture")
    capacity = _validate_geometry(max_rows, heads, max_topk, max_extra_topk)
    device, _ = _resolve_cuda_device(device)
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    capability = _compute_capability(device)
    gathers = valid_prefix and gathers_plain_rows(
        swa_cache_format, main_cache_format, max_extra_topk
    )
    nvfp4s = (
        valid_prefix
        and swa_cache_format == "swa_fp8"
        and main_cache_format == "main"
        and max_extra_topk > 0
    )
    # Specs compare without their kernel object, so equal kernels collapse.
    specs = {}
    native_specs = {}
    previous = None
    for rows in range(1, max_rows + 1):
        packed = _select_packed_profile(rows, heads, capacity, sm_count, inputs_final)
        gather = _select_gather_profile(rows, heads, capacity, sm_count) if gathers else None
        if output_groups is not None and not _quantizes_output(gather or packed):
            continue
        nvfp4 = (
            _select_nvfp4_profile(rows, heads, capacity, sm_count, capability) if nvfp4s else None
        )
        if nvfp4 is not None and output_groups is not None and not _quantizes_output(nvfp4):
            nvfp4 = None
        kind = "nvfp4" if nvfp4 is not None else "gather" if gather is not None else "packed"
        profile = nvfp4 or gather or packed
        # Beyond its profile, a kernel depends on the row count only through
        # the split-KV reduction topology. A single split has no reduction.
        if (kind, profile) == previous and profile.split_kv == 1:
            continue
        previous = (kind, profile)
        launch = _resolve_launch(
            device,
            rows,
            heads,
            max_topk,
            max_extra_topk,
            swa_cache_format,
            main_cache_format=main_cache_format,
            valid_prefix=valid_prefix,
            output_groups=output_groups,
            raw_selections=raw_selections,
            query_rope=query_rope,
            inputs_final=inputs_final,
            main_ready=main_ready,
        )
        native = launch.nvfp4 or launch.gather
        if native is not None:
            native_specs.setdefault(native.spec)
            continue
        specs.setdefault(
            _compile_spec(
                launch,
                heads,
                max_topk,
                max_extra_topk,
                has_sinks,
                swa_cache_format,
                main_cache_format,
                raw_selections,
                query_rope,
                main_ready,
            )
        )
    for spec in specs:
        _compile_packed(spec)
    for spec in native_specs:
        _get_compiled_mla_decode(spec)
    return len(specs) + len(native_specs)


@cache
def _compile_packed(spec: _CompileSpec) -> Callable[..., object]:
    """Compile symbolic raw-pool and batch layouts without allocating fake backing."""
    batch = cute.sym_int()
    q_stride = (_HEAD_DIM, 1, spec.heads * _HEAD_DIM, spec.heads * _HEAD_DIM)
    q_shape = (spec.heads, _HEAD_DIM, 1, batch)
    query = cute.runtime.make_fake_tensor(
        cutlass.Float8E4M3FN, q_shape, stride=q_stride, assumed_align=16
    )
    # An MXFP8 output kernel stores through its output quantizer; the E4M3 output
    # only gates the stores, as in the native quantized plan.
    output = cute.runtime.make_fake_tensor(
        cutlass.BFloat16 if spec.output_groups is None else cutlass.Float8E4M3FN,
        q_shape,
        stride=q_stride,
        assumed_align=16,
    )
    lse = cute.runtime.make_fake_tensor(
        cutlass.Float32,
        (spec.heads, 1, batch),
        stride=(1, spec.heads, spec.heads),
        assumed_align=16,
    )

    def raw_pool(width: int) -> cute.Tensor:
        return cute.runtime.make_fake_tensor(
            cutlass.Uint8,
            (cute.sym_int(), width),
            stride=(cute.sym_int64(divisibility=1), 1),
            assumed_align=16,
        )

    def indices(width: int, dtype: type = cutlass.Int32) -> cute.Tensor:
        return cute.runtime.make_fake_tensor(
            dtype,
            (batch, width),
            stride=(cute.sym_int64(divisibility=1), 1),
            assumed_align=dtype.width // 8,
        )

    def lengths() -> cute.Tensor:
        return cute.runtime.make_fake_compact_tensor(
            cutlass.Int32, (batch,), stride_order=(0,), assumed_align=4
        )

    swa = raw_pool(row_bytes(_HEAD_DIM, spec.swa_cache_format))
    main = raw_pool(main_row_bytes(spec.main_cache_format))
    si = indices(spec.max_topk, cutlass.Int64 if spec.raw_selections else cutlass.Int32)
    sl = lengths()
    ci = indices(spec.max_extra_topk) if spec.max_extra_topk else None
    cl = lengths() if spec.max_extra_topk else None
    sinks = (
        cute.runtime.make_fake_compact_tensor(
            cutlass.Float32, (spec.heads,), stride_order=(0,), assumed_align=4
        )
        if spec.has_sinks
        else None
    )
    # Native six-slot sparse ABI: the producer-owned binder reads both lists from
    # the first slot; no FP8-primary tensor scale is bound. Raw selections add the
    # MAIN owner's INT32 page table, INT64 requests and visible lengths, the position
    # limit and the page size.
    lists = (si, ci)
    if spec.raw_selections:
        mapping = (None, None, None)
        if spec.max_extra_topk:
            mapping = (
                cute.runtime.make_fake_tensor(
                    cutlass.Int32,
                    (cute.sym_int(), cute.sym_int()),
                    stride=(cute.sym_int64(divisibility=1), 1),
                    assumed_align=4,
                ),
                cute.runtime.make_fake_compact_tensor(
                    cutlass.Int64, (batch,), stride_order=(0,), assumed_align=8
                ),
                cute.runtime.make_fake_compact_tensor(
                    cutlass.Int64, (batch,), stride_order=(0,), assumed_align=8
                ),
            )
        lists = (si, ci, *mapping, cutlass.Int64(0), cutlass.Int32(1))
        if spec.query_rope:
            # The BF16 query before its RoPE, each row's INT32 position and the FP32
            # [P, 2, 32] cos/sin table, flat.
            lists = (
                *lists,
                cute.runtime.make_fake_compact_tensor(
                    cutlass.BFloat16,
                    (batch, spec.heads, _HEAD_DIM),
                    stride_order=(2, 1, 0),
                    assumed_align=16,
                ),
                cute.runtime.make_fake_compact_tensor(
                    cutlass.Int32, (batch,), stride_order=(0,), assumed_align=4
                ),
                cute.runtime.make_fake_compact_tensor(
                    cutlass.Float32, (cute.sym_int(),), stride_order=(0,), assumed_align=16
                ),
            )
    sparse_inputs = (lists, None, sl, cl, sinks, None)
    page_offsets = cute.runtime.make_fake_tensor(
        cutlass.Int32,
        (spec.max_topk, batch),
        stride=(1, cute.sym_int64(divisibility=1)),
        assumed_align=4,
    )
    workspace = None
    if spec.has_workspace:
        workspace = cute.runtime.make_fake_compact_tensor(
            cutlass.Int8, (cute.sym_int(),), stride_order=(0,), assumed_align=32
        )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    def compile_kernel() -> object:
        return cute.compile[cute.FrontendNext](
            spec.kernel,
            query,
            query,
            swa,
            main,
            page_offsets,
            output,
            lse,
            workspace,
            cutlass.Int32(spec.split_kv),
            sl,
            None,
            sparse_inputs,
            cutlass.Float32(1.0),
            cutlass.Float32(1.0),
            stream,
            output_quant_args=_fake_output_quant_args(spec.kernel.output_quant),
            # One query per virtual request; sparse plans keep Q lengths static.
            max_q_len=cutlass.Int32(1),
            options=_COMPILE_OPTIONS,
        )

    # The producer's inlined helpers and packed ABI must invalidate the native
    # persistent cache too, without incorporating any live tensors.
    source = Path(__file__)
    source_keys = tuple(
        sha256(path.read_bytes()).hexdigest()
        for path in (
            source,
            source.with_name("_prims_ts_output.py"),
            source.with_name("_prims_ts_producer.py"),
            source.with_name("_prims_ts_quantization.py"),
        )
    )
    with torch.cuda.device(spec.device_index):
        return build_and_load_attention_kernel(
            "mla_decode", "csa2_packed", spec, _COMPILE_OPTIONS, compile_kernel, extra=source_keys
        )


class PackedSparseMLA:
    """One serial PrimTS plan/workspace over live packed cache arguments.

    Args:
        workspace_buffer: Caller-owned contiguous CUDA INT8/UINT8 byte arena.
            Size it using :func:`get_packed_sparse_mla_workspace_size`.

    A plan must be warmed outside CUDA Graph capture. Runs require preallocated
    output; softmax and output scales are host floats, which a captured CUDA
    Graph replays as captured. No cache tensor or pointer is retained by the
    plan; callers keep graph-bound inputs alive. Different
    concurrent streams require separate workspace arenas. Planning compiles the
    plan's kernel: for a ``valid_prefix`` plan, the TMA gather kernel for dense
    512-byte plain E4M3 rows whose schedule favors it, or the NVFP4 MAIN kernel
    for dense ``swa_fp8`` SWA rows and NVFP4 MAIN records where
    :func:`decodes_nvfp4_main` admits it; otherwise the packed producer's
    asynchronous-copy kernel for 16-byte-aligned pool row strides. The first
    eager run with other strides compiles the producer kernel they need.
    """

    def __init__(self, workspace_buffer: torch.Tensor) -> None:
        self.workspace_buffer = workspace_buffer
        self.workspace_size_bytes = 0
        self._state: _Plan | None = None
        # Producer kernels compiled lazily, keyed by 16-byte row alignment.
        self._packed: dict[bool, Callable[..., object]] = {}

    @property
    def workspace_tensors(self) -> tuple[torch.Tensor, ...]:
        """Actual plan-held views; all share the caller's byte arena storage."""
        if self._state is None:
            return ()
        return tuple(self._state.buffers.values())

    def plan(
        self,
        device: torch.device | str | int,
        rows: int,
        heads: int,
        *,
        max_topk: int = 128,
        max_extra_topk: int = 0,
        has_sinks: bool = True,
        swa_cache_format: SwaCacheFormat = "swa",
        main_cache_format: PrimsTSMainFormat = "main",
        valid_prefix: bool = False,
        output_groups: int | None = None,
        raw_selections: bool = False,
        query_rope: bool = False,
        inputs_final: bool = False,
        main_ready: bool = True,
    ) -> None:
        """Warm the format-specific uint8 ABI and reduction outside graph capture.

        Bind native reduction/LSE, unit-scale E4M3 Q, a single packed dummy
        record and the gather kernel's empty extra lists and -inf sinks from the
        supplied workspace. No source-sized tensors or decoded KV are allocated.

        ``valid_prefix=True`` promises that every entry before each live length
        is a valid row of its pool. When both sources then hold plain unit-scale
        E4M3 rows (``swa_fp8`` SWA and, if planned, ``main_fp8`` MAIN), launches
        whose schedule favors it read the rows with native TMA gather4 copies
        rather than the packed producer's issuer-warp copies. With ``swa_fp8``
        SWA and NVFP4 MAIN records (``main``), launches that
        :func:`decodes_nvfp4_main` admits run the native NVFP4 loader schedule,
        which decodes the selected MAIN records in the kernel.

        ``output_groups`` plans kernels that store, instead of the BF16 output, the
        grouped MXFP8 O-A input of that many groups (see ``run``'s ``output_quant``).
        Only geometries for which :func:`fuses_mxfp8_output` holds accept it.

        ``raw_selections=True`` plans for the selections as a layer publishes them:
        ``run`` then takes INT64 SWA slots and, with MAIN, INT32 logical positions with
        their ``main_mapping``, and needs no live lengths (see ``run``).

        ``query_rope=True`` (raw selections only) plans a kernel that takes the BF16 query
        before its GPT-J RoPE and rotates and rounds it itself (see ``run``'s ``query_rope``).

        ``inputs_final=True`` promises that the query, both selection lists, the page table
        and the MAIN rows were final before the launch preceding this one on the stream; a
        cluster-merged kernel then runs ahead of that launch and waits for it only before
        its SWA rows and outputs. With ``main_ready=False`` only the query, the SWA slots and
        the page table carry that promise: the kernel also waits before it reads the MAIN
        selections, the MAIN owners' visible lengths and the MAIN rows.
        """
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("PackedSparseMLA.plan() must run before CUDA Graph capture")
        if not isinstance(has_sinks, bool) or not isinstance(valid_prefix, bool):
            raise TypeError("has_sinks and valid_prefix must be bool")
        launch = _resolve_launch(
            device,
            rows,
            heads,
            max_topk,
            max_extra_topk,
            swa_cache_format,
            main_cache_format=main_cache_format,
            valid_prefix=valid_prefix,
            output_groups=output_groups,
            raw_selections=raw_selections,
            query_rope=query_rope,
            inputs_final=inputs_final,
            main_ready=main_ready,
        )
        _validate_workspace_buffer(
            self.workspace_buffer, device=launch.device, required_bytes=launch.workspace_bytes
        )
        buffers = {
            name: _workspace_section_view(self.workspace_buffer, section)
            for name, section in launch.sections.items()
        }
        buffers["q8"] = buffers["q8"].view(torch.float8_e4m3fn)
        buffers["dummy_main"].zero_()
        buffers["dummy_extra"].zero_()
        buffers["no_sinks"].fill_(-math.inf)
        # The split finisher's arrival counters follow the split partials and start at zero;
        # every launch leaves them zero again.
        buffers["core"].zero_()
        buffers["raw_lengths"][0].fill_(max_topk)
        buffers["raw_lengths"][1].fill_(max_extra_topk)
        buffers["raw_offsets"].zero_()
        dummy_main = buffers["dummy_main"][:0, : main_row_bytes(main_cache_format)]
        spec = _compile_spec(
            launch,
            heads,
            max_topk,
            max_extra_topk,
            has_sinks,
            swa_cache_format,
            main_cache_format,
            raw_selections,
            query_rope,
            main_ready,
        )

        def native_plan(native: _GatherLaunch | None) -> _GatherPlan | None:
            if native is None:
                return None
            return _GatherPlan(
                compiled=_get_compiled_mla_decode(native.spec),
                split_kv=native.profile.split_kv,
                core=buffers["core"] if native.core_bytes > 0 else None,
            )

        gather = native_plan(launch.gather)
        nvfp4 = native_plan(launch.nvfp4)
        compiled = _compile_packed(spec) if gather is None and nvfp4 is None else None
        self._state = _Plan(
            device=launch.device,
            rows=rows,
            heads=heads,
            max_topk=max_topk,
            max_extra_topk=max_extra_topk,
            has_sinks=has_sinks,
            split_kv=launch.profile.split_kv,
            swa_cache_format=swa_cache_format,
            main_cache_format=main_cache_format,
            valid_prefix=valid_prefix,
            output_groups=output_groups,
            spec=spec,
            raw_selections=raw_selections,
            query_rope=query_rope,
            inputs_final=inputs_final,
            main_ready=main_ready,
            compiled=compiled,
            gather=gather,
            nvfp4=nvfp4,
            query=UnitScaleQuery.plan(launch.device, rows, heads, buffers["q8"]),
            buffers=buffers,
            core=buffers["core"] if spec.has_workspace else None,
            lse_kernel=buffers["lse"].permute(2, 1, 0),
            dummy_main=dummy_main,
        )
        self._packed = {}
        self.workspace_size_bytes = launch.workspace_bytes

    def _packed_kernel(self, aligned: bool) -> Callable[..., object]:
        """The packed producer kernel for these pool strides.

        A producer plan compiles its asynchronous-copy kernel when planned. The
        synchronous-load variant, and any producer kernel of a gather plan,
        compile on the first eager run with pools that need them.
        """
        state = self._state
        if aligned and state.compiled is not None:
            return state.compiled
        kernel = self._packed.get(aligned)
        if kernel is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Run PackedSparseMLA once with these pool strides before CUDA Graph capture"
                )
            launch = _resolve_launch(
                state.device,
                state.rows,
                state.heads,
                state.max_topk,
                state.max_extra_topk,
                state.swa_cache_format,
                aligned_strides=aligned,
                main_cache_format=state.main_cache_format,
                output_groups=state.output_groups,
                raw_selections=state.raw_selections,
                query_rope=state.query_rope,
                inputs_final=state.inputs_final,
                main_ready=state.main_ready,
            )
            kernel = _compile_packed(
                replace(
                    state.spec,
                    kernel_signature=launch.kernel.compile_signature(),
                    kernel=launch.kernel,
                )
            )
            self._packed[aligned] = kernel
        return kernel

    @staticmethod
    def _pool(tensor: torch.Tensor, width: int, device: torch.device, name: str) -> None:
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device != device
            or tensor.dtype != torch.uint8
            or tensor.ndim != 2
            or tensor.shape[1] != width
            or tensor.shape[0] >= 0x7FFFFFFF
            or tensor.stride(1) != 1
            or tensor.stride(0) < width
            or tensor.data_ptr() % 16
        ):
            raise ValueError(
                f"{name} must be aligned CUDA UINT8[N,{width}] with a contiguous inner byte axis"
            )

    @staticmethod
    def _source_metadata(
        indices: torch.Tensor,
        lengths: torch.Tensor,
        rows: int,
        capacity: int,
        tokens: int,
        device: torch.device,
        name: str,
        validate: bool,
        valid_prefix: bool = False,
    ) -> None:
        if (
            not isinstance(indices, torch.Tensor)
            or indices.device != device
            or indices.dtype != torch.int32
            or indices.shape != (rows, capacity)
            or indices.stride(1) != 1
            or indices.stride(0) < capacity
            or indices.data_ptr() % 4
        ):
            raise ValueError(
                f"{name} indices must be CUDA INT32[rows,capacity] with unit inner stride"
            )
        if (
            not isinstance(lengths, torch.Tensor)
            or lengths.device != device
            or lengths.dtype != torch.int32
            or lengths.shape != (rows,)
            or not lengths.is_contiguous()
            or lengths.data_ptr() % 4
        ):
            raise ValueError(f"{name} lengths must be contiguous CUDA INT32[rows]")
        if validate:
            if ((lengths < 0) | (lengths > capacity)).any().item():
                raise ValueError(f"{name} lengths must be within the planned source capacity")
            active = torch.arange(capacity, device=device)[None, :] < lengths[:, None]
            if (active & ((indices < -1) | (indices >= tokens))).any().item():
                raise ValueError(f"{name} active indices must be -1 or valid physical pool rows")
            if valid_prefix and (active & (indices < 0)).any().item():
                raise ValueError(f"{name} active indices must be valid rows under valid_prefix")

    def _sources(
        self,
        state: _Plan,
        swa_cache: torch.Tensor,
        swa_indices: torch.Tensor,
        swa_lengths: torch.Tensor,
        extra_kv_cache: torch.Tensor | None,
        extra_sparse_indices: torch.Tensor | None,
        extra_sparse_lens: torch.Tensor | None,
        validate: bool,
    ) -> torch.Tensor:
        """Check both source lists and return the MAIN pool the kernels read."""
        self._pool(
            swa_cache, row_bytes(_HEAD_DIM, state.swa_cache_format), state.device, "swa_cache"
        )
        self._source_metadata(
            swa_indices,
            swa_lengths,
            state.rows,
            state.max_topk,
            swa_cache.shape[0],
            state.device,
            "SWA",
            validate,
            state.valid_prefix,
        )
        if not state.max_extra_topk:
            if any(
                v is not None for v in (extra_kv_cache, extra_sparse_indices, extra_sparse_lens)
            ):
                raise ValueError("MAIN was not enabled by the plan")
            return state.dummy_main
        main = state.dummy_main if extra_kv_cache is None else extra_kv_cache
        self._pool(main, main_row_bytes(state.main_cache_format), state.device, "extra_kv_cache")
        self._source_metadata(
            extra_sparse_indices,
            extra_sparse_lens,
            state.rows,
            state.max_extra_topk,
            main.shape[0],
            state.device,
            "MAIN",
            validate,
            state.valid_prefix,
        )
        return main

    def _raw_sources(
        self,
        state: _Plan,
        swa_cache: torch.Tensor,
        swa_indices: torch.Tensor,
        extra_kv_cache: torch.Tensor | None,
        extra_sparse_indices: torch.Tensor | None,
        main_mapping: tuple | None,
    ) -> torch.Tensor:
        """Check a raw-selection launch's lists and mapping; return the MAIN pool it reads."""
        self._pool(
            swa_cache, row_bytes(_HEAD_DIM, state.swa_cache_format), state.device, "swa_cache"
        )

        def check(indices, dtype, width, name):
            if (
                not isinstance(indices, torch.Tensor)
                or indices.device != state.device
                or indices.dtype != dtype
                or indices.shape != (state.rows, width)
                or indices.stride(1) != 1
                or indices.stride(0) < width
                or indices.data_ptr() % indices.element_size()
            ):
                raise ValueError(
                    f"raw {name} selections must be CUDA {dtype}[rows,{width}] "
                    "with unit inner stride"
                )

        check(swa_indices, torch.int64, state.max_topk, "SWA")
        if not state.max_extra_topk:
            if any(v is not None for v in (extra_kv_cache, extra_sparse_indices, main_mapping)):
                raise ValueError("MAIN was not enabled by the plan")
            return state.dummy_main
        if extra_kv_cache is None or main_mapping is None:
            raise ValueError("raw MAIN selections need their pool and main_mapping")
        self._pool(
            extra_kv_cache,
            main_row_bytes(state.main_cache_format),
            state.device,
            "extra_kv_cache",
        )
        check(extra_sparse_indices, torch.int32, state.max_extra_topk, "MAIN")
        pages, page_size, max_positions, requests, visible = main_mapping
        if (
            not isinstance(pages, torch.Tensor)
            or pages.device != state.device
            or pages.dtype != torch.int32
            or pages.ndim != 2
            or pages.numel() == 0
            or pages.stride(1) != 1
            or not isinstance(page_size, int)
            or page_size <= 0
            or not isinstance(max_positions, int)
            or max_positions < 0
            # Physical rows are formed in INT32 before their bound check.
            or extra_kv_cache.shape[0] > 0x7FFFFFFF - page_size
        ):
            raise ValueError(
                "raw MAIN selections need a nonempty CUDA INT32 page table and valid bounds"
            )
        for values in (requests, visible):
            if (
                not isinstance(values, torch.Tensor)
                or values.device != state.device
                or values.dtype != torch.int64
                or values.shape != (state.rows,)
                or not values.is_contiguous()
            ):
                raise ValueError("raw MAIN requests and visibility must be CUDA INT64[rows]")
        return extra_kv_cache

    def run(
        self,
        q: torch.Tensor,
        swa_cache: torch.Tensor,
        swa_indices: torch.Tensor,
        swa_lengths: torch.Tensor | None,
        *,
        extra_kv_cache: torch.Tensor | None = None,
        extra_sparse_indices: torch.Tensor | None = None,
        extra_sparse_lens: torch.Tensor | None = None,
        softmax_scale: float,
        output_scale: float = 1.0,
        sinks: torch.Tensor | None,
        out: torch.Tensor | None,
        validate: bool = False,
        query_ready: torch.cuda.Event | None = None,
        output_quant: tuple[torch.Tensor, ...] | None = None,
        main_mapping: tuple | None = None,
        query_rope: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Launch joint sparse attention without compilation or device allocation.

        With ``query_ready``, the current stream waits for that event before Q
        is read.

        Q is contiguous BF16[rows,1,H,512], or E4M3 already rounded RN/satfinite
        at unit scale, which is used as is; out is contiguous BF16 of that shape.
        A plan with ``output_groups`` takes no ``out`` and requires instead
        ``output_quant = (data, scales, positions, cos_sin)``: E4M3 ``[groups, rows,
        D]`` and UINT8 ``groups * pad128(rows) * D / 32`` buffers receiving the grouped
        MXFP8 O-A input, made exactly as ``inverse_rope_mxfp8`` makes it of the BF16
        output, and the INT32 ``[rows]`` positions and FP32 ``[P, 64]`` cos/sin of its
        inverse RoPE. The MXFP8 data is returned.
        The planned SWA format is
        UINT8[N,528] E4M3/group-32 UE8M0 (``swa``, the default) or UINT8[N,512]
        unit-scale E4M3 without a footer (``swa_fp8``). Optional MAIN is the
        planned format: UINT8[N,288] packed E2M1/group-16 E4M3 (``main``) or
        UINT8[N,512] unit-scale E4M3 rows decoded from such records
        (``main_fp8``). Outer byte strides may be padded.
        Each source has INT32[rows,capacity] physical indices and contiguous
        INT32[rows] live lengths. Row zero is valid; -1 and out-of-pool indices
        are masked before payload/scale reads. Lengths must be in [0,capacity].
        A ``valid_prefix`` plan requires instead that every entry before a live
        length is a valid row of its pool; its TMA gather or NVFP4 MAIN kernel reads
        pools whose rows are dense records of their format, and other pools use the
        packed producer.
        Both lists share one softmax and optional FP32[H] sinks are applied once.

        All operands use unit tensor scales: BF16 Q is rounded RN/satfinite to
        E4M3 in arena scratch, and the loader decodes each KV row with its group
        scales into RN/satfinite E4M3 operands. Values beyond the E4M3 range
        therefore saturate at +-448. ``softmax_scale`` and ``output_scale`` are
        the host FP32 factors applied to Q.K and to the output; a captured CUDA
        Graph replays the values it was captured with.
        ``validate=True`` also checks input values on the host and is not graph-safe.

        A ``raw_selections`` plan takes the lists as a layer publishes them and no
        lengths: INT64 ``[rows, max_topk]`` SWA slots, and INT32 ``[rows, max_extra_topk]``
        logical MAIN positions with ``main_mapping = (pages, page_size, max_positions,
        requests, visible)``, the owner's INT32 page table and each row's INT64 request and
        visible length. Every entry of both lists is live; an entry
        ``prepare_prims_ts_metadata`` would drop is masked in place.

        A ``query_rope`` plan takes ``q`` as the BF16 projection before its GPT-J RoPE and
        ``query_rope = (positions, cos_sin)``: each row's INT32 position and the contiguous
        FP32 ``[P, 2, 32]`` cos/sin table. The kernel rotates the last 64 channels of every
        head and rounds the query to unit-scale E4M3, the bytes ``gptj_rope_e4m3_small_m``
        writes.
        """
        state = self._state
        if state is None:
            raise RuntimeError("PackedSparseMLA.plan() must complete before run() or graph capture")
        if validate and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("validate=True is not supported during CUDA Graph capture")
        shape = (state.rows, 1, state.heads, _HEAD_DIM)
        quantized = state.output_groups is not None
        if quantized != (output_quant is not None) or (quantized and out is not None):
            raise ValueError("output_quant, not out, is required exactly by MXFP8 output plans")
        checked = [(q, "q", (torch.bfloat16, torch.float8_e4m3fn))]
        if not quantized:
            checked.append((out, "out", (torch.bfloat16,)))
        for tensor, name, dtypes in checked:
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.shape != shape
                or tensor.device != state.device
                or tensor.dtype not in dtypes
                or not tensor.is_contiguous()
                or tensor.data_ptr() % 16
            ):
                raise ValueError(
                    f"{name} must be aligned contiguous CUDA {'/'.join(map(str, dtypes))}"
                    "[rows,1,H,512]"
                )
        if state.raw_selections:
            if swa_lengths is not None or extra_sparse_lens is not None:
                raise ValueError("raw selections take no live lengths")
            main = self._raw_sources(
                state, swa_cache, swa_indices, extra_kv_cache, extra_sparse_indices, main_mapping
            )
            swa_lengths = state.buffers["raw_lengths"][0]
            if state.max_extra_topk:
                extra_sparse_lens = state.buffers["raw_lengths"][1]
        else:
            if main_mapping is not None:
                raise ValueError("main_mapping requires a raw_selections plan")
            main = self._sources(
                state,
                swa_cache,
                swa_indices,
                swa_lengths,
                extra_kv_cache,
                extra_sparse_indices,
                extra_sparse_lens,
                validate,
            )
        _validate_softmax_scale(softmax_scale)
        if not math.isfinite(output_scale) or output_scale <= 0:
            raise ValueError("output_scale must be positive and finite")
        if state.has_sinks != (sinks is not None):
            raise ValueError("sinks presence must match the plan")
        if sinks is not None:
            if (
                sinks.device != state.device
                or sinks.dtype != torch.float32
                or sinks.shape != (state.heads,)
                or not sinks.is_contiguous()
                or sinks.data_ptr() % 4
            ):
                raise ValueError("sinks must be contiguous CUDA FP32[H]")
            if validate and not torch.isfinite(sinks).all().item():
                raise ValueError("sinks must be finite")
        kernel_shape = (state.heads, _HEAD_DIM, 1, state.rows)
        kernel_stride = (_HEAD_DIM, 1, state.heads * _HEAD_DIM, state.heads * _HEAD_DIM)
        if state.query_rope != (query_rope is not None):
            raise ValueError("query_rope is required exactly by query-RoPE plans")
        if query_ready is not None:
            query_ready.wait()
        q8 = q
        if query_rope is not None:
            positions, cos_sin = query_rope
            if q.dtype != torch.bfloat16:
                raise ValueError("a query-RoPE plan takes the BF16 query before its RoPE")
            if (
                positions.dtype != torch.int32
                or positions.shape != (state.rows,)
                or not positions.is_contiguous()
                or positions.device != state.device
                or cos_sin.dtype != torch.float32
                or cos_sin.dim() < 1
                or cos_sin.numel() % 64
                or not cos_sin.is_contiguous()
                or cos_sin.device != state.device
                or cos_sin.data_ptr() % 16
            ):
                raise ValueError(
                    "query_rope needs contiguous INT32[rows] positions and FP32[P, 2, 32] cos/sin"
                )
            # The kernel stages Q itself; its unused descriptors still need a mapped tensor.
            q8 = state.buffers["q8"]
        elif q.dtype == torch.bfloat16:
            state.query.run(q, state.buffers["q8"])
            q8 = state.buffers["q8"]
        query = q8.as_strided(kernel_shape, kernel_stride)
        quant_args = None
        if quantized:
            data = output_quant[0]
            if data.ndim != 3 or data.shape[:2] != (state.output_groups, state.rows):
                raise ValueError("output_quant data must be E4M3 [groups, rows, D] of the plan")
            quant_args = mxfp8_output_args(*output_quant, state.heads)
            # The output quantizer addresses its own buffers; this view only gates its stores.
            out = data
            output = quant_args[0].as_strided(kernel_shape, kernel_stride)
        else:
            output = out.as_strided(kernel_shape, kernel_stride)
        lists = (swa_indices, extra_sparse_indices)
        page_offsets = swa_indices.transpose(0, 1)
        if state.raw_selections:
            pages, page_size, max_positions, requests, visible = main_mapping or (
                None,
                1,
                0,
                None,
                None,
            )
            lists = (*lists, pages, requests, visible, max_positions, page_size)
            if query_rope is not None:
                lists = (
                    *lists,
                    q.view(state.rows, state.heads, _HEAD_DIM),
                    query_rope[0],
                    query_rope[1].view(-1),
                )
            page_offsets = state.buffers["raw_offsets"].transpose(0, 1)
        sparse_inputs = (lists, None, swa_lengths, extra_sparse_lens, sinks, None)
        sinks_or_none = state.buffers["no_sinks"] if sinks is None else sinks
        nvfp4 = state.nvfp4
        # An empty or omitted MAIN pool has no live rows under the prefix promise;
        # the native kernels still need a descriptor over one real record.
        main_bytes = main_row_bytes(state.main_cache_format)
        native_main = (
            main if main.shape[0] else state.buffers["dummy_main"].view(-1)[:main_bytes].view(1, -1)
        )
        if nvfp4 is not None and _dense_rows(swa_cache) and _dense_rows(native_main, main_bytes):
            # The native sparse ABI, as the gather kernel takes it, over UINT8 record views.
            nvfp4.compiled(
                query,
                query,
                _record_pool(swa_cache),
                _record_pool(native_main),
                swa_indices.transpose(0, 1),
                output,
                state.lse_kernel,
                nvfp4.core,
                nvfp4.split_kv,
                swa_lengths,
                None,
                (
                    swa_indices,
                    extra_sparse_indices,
                    swa_lengths,
                    extra_sparse_lens,
                    sinks_or_none,
                    1.0,
                ),
                softmax_scale,
                output_scale,
                quant_args,
                1,
            )
            return out
        gather = state.gather
        if gather is not None and _dense_rows(swa_cache) and _dense_rows(native_main):
            extra_indices, extra_lengths = extra_sparse_indices, extra_sparse_lens
            if not state.max_extra_topk:
                extra_indices = state.buffers["dummy_extra"][0].view(state.rows, 1)
                extra_lengths = state.buffers["dummy_extra"][1]
            # The native sparse ABI, positionally: both page views, the six
            # sparse slots ending with the unit KV scale, the output quantizer's
            # arguments and one query per virtual request.
            gather.compiled(
                query,
                query,
                _gather_pool(swa_cache),
                _gather_pool(native_main),
                swa_indices.transpose(0, 1),
                output,
                state.lse_kernel,
                gather.core,
                gather.split_kv,
                swa_lengths,
                None,
                (
                    swa_indices,
                    extra_indices,
                    swa_lengths,
                    extra_lengths,
                    sinks_or_none,
                    1.0,
                ),
                softmax_scale,
                output_scale,
                quant_args,
                1,
            )
            return out
        compiled = self._packed_kernel(swa_cache.stride(0) % 16 == 0 and main.stride(0) % 16 == 0)
        # Persisted FFI callables take every parameter positionally: the output
        # quantizer's arguments and one query per virtual request.
        compiled(
            query,
            query,
            swa_cache,
            main,
            page_offsets,
            output,
            state.lse_kernel,
            state.core,
            state.split_kv,
            swa_lengths,
            None,
            sparse_inputs,
            softmax_scale,
            output_scale,
            quant_args,
            1,
        )
        return out
