# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ruff: noqa: B905, F821, F841

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Device program for the PrimsTS FP16/BF16/FP8/NVFP4 GEMM.

PrimsTS is CUTLASS primitives plus task scheduling. This is the module to
vendor into FlashInfer as ``flashinfer/prims_ts/kernel.py``. It owns the task-scheduled
2-CTA GEMM, its fused epilogues, and the compile-time flags those
``@cute.jit`` functions close over.

``api.py`` is the host launch API. ``harness.py`` checks and times the kernel.
Serving must not import this module once and share it: CuTe captures the flags
below from this module's globals, so each epilogue mode loads its own copy.

Resource and task flow:

                   +-------------------+
                   |   GmemAbResource  |
                   |  tile coordinates |
                   +----+----------+---+
                        |          |
        LoadATask:      |          |      LoadBTask:
        PDL wait,       |          |      compute B coords,
        compute A coords|          |      TMA B into SMEM
        TMA A into SMEM |          |
                        v          v
                   +----+----+ +---+-----+
                   |  SmemA  | |  SmemB  |
                   | TmaUmma | | TmaUmma |
                   +--+------+ +---+-----+
                      |            |
                      | MmaTask waits A/B full stages,
                      | builds descriptors, issues MMA
                      v
                +-----+-----+
                |   TmemC   |
                | UmmaAsync |
                +-----+-----+
                      |
                      | StoreTask preloads optional rowcol scales,
                      | waits accumulator, Tmem to Regs load,
                      | scales / bias, optional gated SiLU, TMA store
                      v
                 +----+----+
                 |  GmemD  |
                 | output  |
                 +---------+

WorkQueue: every task advances the same persistent tile stream.
PdlWait runs before LoadATask; LoadATask signals PdlLaunch after the loop.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, Type

import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass import Numeric
from cutlass._mlir import ir as _ir
from cutlass._mlir.dialects import nvvm as _nvvm_dialect
from cutlass.experimental import primitives as prims
from cutlass.experimental.task_scheduling.enums import (
    SignalingThreads,
    WorkAttr,
)
from cutlass.experimental.task_scheduling.memory import (
    SmemAllocation,
    SmemAllocator,
    TmemAllocation,
    TmemAllocator,
)
from cutlass.experimental.task_scheduling.resources import (
    MemoryResource,
    PdlLaunchBarrier,
    PdlWaitBarrier,
    PipelineConfig,
    StageInfo,
    TaskLocalVariable,
    TileSchedulerConfig,
    WorkQueue,
    consumer_work,
    producer_work,
)
from cutlass.experimental.task_scheduling.schedule_builder import (
    domain_loop,
    schedule,
    work_tile_loop,
)
from cutlass.experimental.task_scheduling.task import Task
from cutlass.experimental.task_scheduling.task_manager import TaskManager


def _prims_ts_debug_checks_enabled() -> bool:
    """Whether this compile should run TaskManager validation.

    Production JIT leaves ``skip_validation`` on and the exhaustive
    deadlock/race search off. Unit tests set
    ``FLASHINFER_PRIMS_TS_DEBUG_CHECKS`` so the same construction checks
    the schedule before the kernel is cached.
    """
    value = os.environ.get("FLASHINFER_PRIMS_TS_DEBUG_CHECKS", "0").lower()
    return value not in {"0", "false", "no", "off"}


_DTYPE_MAP = {
    "fp16": cutlass.Float16,
    "bf16": cutlass.BFloat16,
    "fp8": cutlass.Float8E4M3FN,
    "fp4": cutlass.Float4E2M1FN,
}
_SUPPORTED_CLUSTER_DIMS = (1, 2, 4)
_FP8_E4M3_MAX = 448.0
# E2M1 magnitude, times the E4M3 block-scale max. NVFP4 global scales use
# this range; FP8 per-tensor scales use 448 alone.
_FP4_E2M1_MAX = 6.0
_NVFP4_GLOBAL_MAX = _FP8_E4M3_MAX * _FP4_E2M1_MAX
# One E4M3 block scale per 16 FP4 K elements. E8M0 is not supported.
_SF_VEC_SIZE = 16
sf_dtype = cutlass.Float8E4M3FN
# Compile-time: FP8 per-token (M) and per-channel (N) epilogue scales.
use_per_token_channel_scale = False
# Compile-time: fused SwiGLU in the epilogue. B/bias/weight_scale are
# interleaved along N so each 32-column TMEM subtile is (gate, act) pairs.
# After scaling, out = gate * silu(act) and the store is N/2 columns.
use_gated_activation = False
# Keep 128 B along K so TMA/SMEM stay on the 128 B swizzle path.
# Data tile K must be a multiple of this TMA box. The SM103 split pipeline
# independently sizes scale tiles to a multiple of MMA instruction K.
_K_TILE_BYTES = 128
# Ordinary MMA instructions consume 32 B of K; SM103 NVFP4 can consume 48 B.
_MMA_K_BYTES = 32
nvfp4_mma_k = 64
# Optional override for pipeline tile K in elements. None → one 128 B K-box.
_tile_k_override = None
# Optional override for pipeline tile N in elements. None → 256.
# MMA instruction N is always equal to tile N. The epilogue drains TMEM
# in 32-column chunks, so tile N is a power of 2 in [32, 256].
_tile_n_override = None
_DEFAULT_TILE_N = 256
_EPILOGUE_TILE_N = 32
_TILE_N_MIN = 32
_TILE_N_MAX = 256


def _convert_f32x2_to_e2m1x2(a, b):
    """Pack two FP32 values into one E2M1 byte (``b`` is the low nibble)."""
    dst_type = _ir.TypeAttr.get(cutlass.Float4E2M1FN.mlir_type)
    return cutlass.Int8(
        _nvvm_dialect.convert_f32x2_to_f4x2(
            cutlass.Float32(a).ir_value(),
            cutlass.Float32(b).ir_value(),
            dst_type,
        )
    )


def _parse_dtype_name(dtype_name: str, *, option_name: str) -> Type[Numeric]:
    normalized = dtype_name.strip().lower()
    if normalized not in _DTYPE_MAP:
        choices = ", ".join(_DTYPE_MAP.keys())
        raise ValueError(
            f"Unsupported {option_name} '{dtype_name}'. Expected one of: {choices}"
        )
    return _DTYPE_MAP[normalized]


def _dtype_name(dtype: Type[Numeric]) -> str:
    for name, mapped in _DTYPE_MAP.items():
        if dtype is mapped:
            return name
    return str(dtype)


def _is_fp8(dtype: Type[Numeric]) -> bool:
    return dtype is cutlass.Float8E4M3FN


def _is_fp4(dtype: Type[Numeric]) -> bool:
    return dtype is cutlass.Float4E2M1FN


def _span_bytes(num_elems: int, dtype: Type[Numeric]) -> int:
    """Storage bytes for ``num_elems`` values. FP4 packs two values per byte."""
    return num_elems * dtype.width // 8


def _elems_in_bytes(num_bytes: int, dtype: Type[Numeric]) -> int:
    return num_bytes * 8 // dtype.width


def _validate_tile_n(tile_n: int) -> None:
    if tile_n < _TILE_N_MIN or tile_n > _TILE_N_MAX:
        raise ValueError(
            f"tile_n must be in [{_TILE_N_MIN}, {_TILE_N_MAX}], got {tile_n}"
        )
    if tile_n & (tile_n - 1) != 0:
        raise ValueError(f"tile_n must be a power of 2, got {tile_n}")
    if tile_n % _EPILOGUE_TILE_N != 0:
        raise ValueError(
            f"tile_n={tile_n} must be a multiple of the epilogue tile "
            f"({_EPILOGUE_TILE_N})"
        )
    if tile_n % num_mma_ctas != 0:
        raise ValueError(
            f"tile_n={tile_n} must be divisible by num_mma_ctas={num_mma_ctas}"
        )


def _refresh_input_dependent_config() -> None:
    """Derive data, scale, and scheduling tiles from the instruction shape."""
    k_inst = (
        nvfp4_mma_k
        if _is_fp4(input_dtype)
        else _elems_in_bytes(_MMA_K_BYTES, input_dtype)
    )
    k_box = _elems_in_bytes(_K_TILE_BYTES, input_dtype)
    k_tile = k_box if _tile_k_override is None else int(_tile_k_override)
    tile_n = _DEFAULT_TILE_N if _tile_n_override is None else int(_tile_n_override)
    nvfp4 = _is_fp4(input_dtype)
    if nvfp4 and nvfp4_mma_k == 96 and k_tile not in (256, 768):
        raise ValueError("NVFP4 MMA-K=96 supports tile_k=256 or tile_k=768")
    if k_tile <= 0:
        raise ValueError(f"tile_k must be positive, got {k_tile}")
    split_k = nvfp4 and nvfp4_mma_k == 96 and k_tile == 256
    globals()["use_nvfp4_split_k"] = split_k
    globals()["sf_tile_k"] = 384 if split_k else k_tile
    globals()["schedule_tile_k"] = 768 if split_k else k_tile
    if k_tile % k_inst != 0 and not split_k:
        raise ValueError(
            f"tile_k={k_tile} must be a multiple of MMA-K={k_inst} for "
            f"input dtype {_dtype_name(input_dtype)}"
        )
    if k_tile % k_box != 0:
        raise ValueError(
            f"tile_k={k_tile} must be a multiple of the 128B TMA K-box "
            f"({k_box} elements for input dtype {_dtype_name(input_dtype)})"
        )
    if nvfp4 and tile_n % 128 != 0:
        raise ValueError(f"NVFP4 tile_n must be a multiple of 128, got {tile_n}")
    if nvfp4 and use_per_token_channel_scale:
        raise ValueError("NVFP4 does not support --per-token-channel-scale")
    if nvfp4 and use_block_major_k:
        raise ValueError("NVFP4 does not support --block-major-k")
    _validate_tile_n(tile_n)
    globals()["mma_inst_shape_mnk"] = (256, tile_n, k_inst)
    globals()["mma_tiler_mnk"] = (256, tile_n, k_tile)
    globals()["mma_tiler_mnk_per_cta"] = (
        128,
        tile_n // num_mma_ctas,
        k_tile,
    )
    globals()["tma_k_box_elems"] = k_box
    globals()["num_tma_k_boxes"] = k_tile // k_box
    globals()["use_nvfp4_block_scale"] = nvfp4
    globals()["mma_kind"] = (
        prims.Tcgen05MMAKind.F8F6F4
        if _is_fp8(input_dtype) or nvfp4
        else prims.Tcgen05MMAKind.F16
    )
    if use_nvfp4_tmem_overlap and not nvfp4:
        raise ValueError("--tmem-overlap requires --input-dtype fp4")
    tmem_overlap = use_nvfp4_tmem_overlap and nvfp4
    if tmem_overlap:
        if tile_n != 256 or k_tile != 256 or num_epilogue_warps != 8:
            raise ValueError(
                "NVFP4 TMEM overlap requires tile_n=256, tile_k=256, "
                "and --epilogue-warps 8"
            )
        if use_tma_store:
            raise ValueError(
                "NVFP4 TMEM overlap is incompatible with TMA output stores"
            )
        # Fused QKNorm/RoPE keeps only subtile 0 in registers across the early
        # release; SF384 (MMA-K=96) scales would also overwrite subtile 1.
        if use_fused_qknorm_rope and nvfp4_mma_k != 64:
            raise ValueError(
                "NVFP4 TMEM overlap with fused QKNorm/RoPE requires MMA-K=64"
            )
    # The overlap path uses two physical accumulator windows in one 512-column
    # allocation. It deliberately keeps one logical UMMA pipeline stage so an
    # early consumer release can hand the retired scale-factor regions to MMA.
    # The regular NVFP4 path leaves TMEM columns for a single accumulator plus
    # its block scales. FP16/BF16/FP8 retain two ordinary pipeline stages.
    acc_stage_count = 2 if tmem_overlap or not nvfp4 else 1
    globals()["acc_stages"] = acc_stage_count
    sf_k_per_tile = sf_tile_k // _SF_VEC_SIZE
    sfa_cols = (mma_tiler_mnk_per_cta[0] * sf_k_per_tile) // 128
    sfb_cols = ((tile_n + 127) // 128) * sf_k_per_tile
    globals()["sfa_stage_bytes"] = mma_tiler_mnk_per_cta[0] * sf_k_per_tile
    globals()["sfb_stage_bytes"] = ((tile_n + 127) // 128) * 128 * sf_k_per_tile
    globals()["tmem_sfa_cols_per_stage"] = sfa_cols
    globals()["tmem_sfb_cols_per_stage"] = sfb_cols
    tmem_cols = tile_n * acc_stage_count
    if nvfp4 and not tmem_overlap:
        tmem_cols += sfa_cols + sfb_cols
    if tmem_cols > 512:
        raise ValueError(
            f"TMEM column count {tmem_cols} exceeds 512 "
            f"(tile_n={tile_n}, acc_stages={acc_stage_count})"
        )
    # tcgen05.alloc can reserve only 32, 64, 128, 256, or 512 columns.
    # NVFP4 additionally keeps its A/B block scales in TMEM, making the
    # 256-column tile require 304 columns.  Reserve the next legal physical
    # allocation; TmemAllocator still places the logical resources in their
    # compact 304-column layout.
    globals()["num_tmem_alloc_cols"] = max(32, 1 << (tmem_cols - 1).bit_length())
    # Per-token/per-channel FP8 dequant replaces the single per-tensor scale.
    # NVFP4 keeps that scalar for the product of the two global scales.
    globals()["needs_epilogue_scale"] = not use_per_token_channel_scale and (
        _is_fp8(input_dtype) or _is_fp8(output_dtype) or nvfp4
    )
    # BlockMajorK K-block is the 128 B MMA/TMA box, independent of tile K.
    globals()["block_major_k_elems"] = k_box
    globals()["super_tile_m"] = num_pair_rows * 256
    globals()["super_tile_n"] = num_pair_cols * tile_n


def _set_io_dtypes(input_dtype_name: str, output_dtype_name: str) -> None:
    parsed_output = _parse_dtype_name(output_dtype_name, option_name="output dtype")
    globals()["input_dtype"] = _parse_dtype_name(
        input_dtype_name, option_name="input dtype"
    )
    globals()["output_dtype"] = parsed_output
    _refresh_input_dependent_config()


def _tensor_max_fp32(tensor) -> float:
    return float(tensor.detach().float().max().item())


def _nvfp4_global_scale(tensor) -> float:
    """Dequant global scale: absmax / (448 * 6).

    ``flashinfer.fp4_quantize`` takes the reciprocal of this value. The
    epilogue multiplies by the product of the A and B scales.
    """
    amax = float(tensor.detach().float().abs().clamp(min=1e-8).max().item())
    if amax < 1e-8:
        amax = 1e-8
    return amax / _NVFP4_GLOBAL_MAX


def _compute_epilogue_scale(a, b, ref_c) -> float:
    """Host-side scale applied in the epilogue: dequant_ab / quant_c."""
    if _is_fp4(input_dtype):
        return _nvfp4_global_scale(a) * _nvfp4_global_scale(b)
    dequant_ab = 1.0
    if _is_fp8(input_dtype):
        dequant_ab = max(
            _tensor_max_fp32(a) / _FP8_E4M3_MAX,
            _tensor_max_fp32(b) / _FP8_E4M3_MAX,
        )
    quant_c = 1.0
    if _is_fp8(output_dtype):
        ref_max = _tensor_max_fp32(ref_c)
        if ref_max == 0.0:
            quant_c = 1.0
        else:
            quant_c = _FP8_E4M3_MAX / ref_max
    return dequant_ab / quant_c


def _validate_cluster_shape(
    shape: tuple[int, int, int],
    *,
    option_name: str,
) -> None:
    if len(shape) != 3:
        raise ValueError(f"{option_name} must contain exactly 3 values")
    cm, cn, ck = shape
    if cm not in _SUPPORTED_CLUSTER_DIMS or cn not in _SUPPORTED_CLUSTER_DIMS:
        raise ValueError(
            f"{option_name} M and N dimensions must be one of "
            f"{_SUPPORTED_CLUSTER_DIMS}, got {shape}"
        )
    if ck != 1:
        raise ValueError(f"{option_name} K dimension must be 1, got {shape}")
    if cm % num_mma_ctas != 0:
        raise ValueError(f"{option_name} cluster_m must be divisible by {num_mma_ctas}")


def _validate_fallback_cluster_shape(
    fallback_shape: tuple[int, int, int],
    preferred_shape: tuple[int, int, int],
) -> None:
    _validate_cluster_shape(
        fallback_shape,
        option_name="--fallback-cluster",
    )
    for fb_dim, preferred_dim in zip(fallback_shape, preferred_shape):
        if preferred_dim % fb_dim != 0:
            raise ValueError(
                "--fallback-cluster must divide --cluster in every dimension; "
                f"got fallback={fallback_shape}, cluster={preferred_shape}"
            )


input_dtype = _DTYPE_MAP["fp16"]
output_dtype = _DTYPE_MAP["fp16"]
acc_dtype = cutlass.Float32
cluster_shape_mnk = (2, 1, 1)
mma_kind = prims.Tcgen05MMAKind.F16
needs_epilogue_scale = False
use_nvfp4_block_scale = False
use_nvfp4_split_k = False
sf_tile_k = 64
schedule_tile_k = 64
use_tma_store = False
use_fused_qknorm_rope = False
separate_qkv_output = False
# Experimental NVFP4 tile-N=256 layout that ping-pongs two accumulator windows
# through all 512 TMEM columns and reuses each retired first epilogue subtile for
# the opposite window's scale factors.
use_nvfp4_tmem_overlap = False
_QKV_HEAD_DIM = 128
_QKNORM_EPS = 1.0e-6
# Compile-time epilogue task width. The 8-warp mode uses two 4-warp groups:
# both groups cover the CTA's 128 rows, while group 0 drains the first half of
# tile N and group 1 drains the second half.
num_epilogue_warps = 4
# Named barrier used only by the configured epilogue threads after filling
# the per-channel weight_scale SMEM tile. Must not be a full CTA
# __syncthreads (MMA/TMA warps are not in this task).
_epilogue_scale_barrier_id = 8
_epilogue_store_barrier_id = 9
_epilogue_qknorm_barrier_id = 10
_epilogue_store_stages = 2

# Cluster decomposition for CTA_2 MMA pairs.
num_mma_ctas = 2
cluster_m = cluster_shape_mnk[0]
cluster_n = cluster_shape_mnk[1]
cluster_size = cluster_m * cluster_n
num_pairs = cluster_size // num_mma_ctas
num_pair_rows = cluster_m // num_mma_ctas
num_pair_cols = cluster_n

# Clustered CTA_2 shapes: instruction, data pipeline tile, and per-CTA tile.
# Data tile K defaults to one 128 B TMA box. _refresh_input_dependent_config
# also derives the independently sized scale tile for SM103 K=96.
mma_inst_shape_mnk = (256, 256, 16)
mma_tiler_mnk = (256, 256, 64)
mma_tiler_mnk_per_cta = (128, mma_tiler_mnk[1] // num_mma_ctas, 64)
tma_k_box_elems = _elems_in_bytes(_K_TILE_BYTES, input_dtype)
num_tma_k_boxes = 1
_refresh_input_dependent_config()
super_tile_m = num_pair_rows * mma_tiler_mnk[0]
super_tile_n = num_pair_cols * mma_tiler_mnk[1]

# Precomputed multicast mask templates (compile-time constants).
# A: bits for all pair-columns at stride num_mma_ctas
# B: bits for all pair-rows at stride num_pair_cols * num_mma_ctas
_a_mcast_template = sum(1 << (num_mma_ctas * c) for c in range(num_pair_cols))
_b_mcast_template = sum(
    1 << (num_pair_cols * num_mma_ctas * r) for r in range(num_pair_rows)
)

threads_in_epilogue = num_epilogue_warps * 32

# Pipeline stage configuration
_DEFAULT_AB_STAGES = 6
_FUSED_QKNORM_AB_STAGES = 4
# NVFP4 keeps A, B, SFA, and SFB in independent multistage SMEM pipelines.
# Scale factors are copied into a single TMEM window immediately before the
# corresponding block-scaled MMA consumes them.
_NVFP4_AB_STAGES = 5
ab_stages = _DEFAULT_AB_STAGES
acc_stages = 2
num_scheduler_stages = 2
debug_print = False

# Scheduling mode option: set to True for CLC dynamic persistent,
# False for static persistent.
# Can also be overridden via --clc-dynamic-scheduler CLI flag.
use_clc_dynamic_scheduler = False


# Fallback cluster: when set, the kernel supports a smaller fallback cluster.
# The hardware launches preferred-sized clusters when possible, falling back
# to this shape on GPCs that cannot accommodate the preferred size.
# Must divide cluster_shape_mnk in each dimension (e.g. (2,1,1) for (4,2,1)).
fallback_cluster_shape_mnk = None

# Optional B (weight) storage: K-major (N, K) vs BlockMajorK (K/block, N, block).
use_block_major_k = False
block_major_k_elems = _elems_in_bytes(_K_TILE_BYTES, input_dtype)


########################################################
# Resource definitions
########################################################


@dataclass
class GmemAbResource(MemoryResource):
    """Global-memory A/B input resource (read-only source).

    This resource has no producer side — it simply exposes the global A/B
    tensors so that downstream consumers can compute TMA load coordinates.

    ``compute_coords`` computes the TMA load coordinates for the current work
    tile and publishes them to the downstream SMEM resource.
    """

    coord_k: cutlass.Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    coord_m: cutlass.Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    coord_n: cutlass.Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    cta_rank_in_cluster: Any = field(init=False, default=None)
    bx: Any = field(init=False, default=None)
    by: Any = field(init=False, default=None)
    bz: Any = field(init=False, default=None)

    def __init__(
        self,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.coord_k = TaskLocalVariable(
            dtype=cutlass.Int32,
            default=cutlass.Int32(0),
            docs="K coordinate for the current TMA load tile.",
        )
        self.coord_m = TaskLocalVariable(
            dtype=cutlass.Int32,
            default=cutlass.Int32(0),
            docs="M coordinate for the current TMA load tile.",
        )
        self.coord_n = TaskLocalVariable(
            dtype=cutlass.Int32,
            default=cutlass.Int32(0),
            docs="N coordinate for the current TMA load tile.",
        )

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_tile_coords(self, stage_info: StageInfo) -> None:
        self.bx, self.by, self.bz = stage_info.work_tile.tile_idx
        self.cta_rank_in_cluster = (self.by % cluster_shape_mnk[1]) * (
            cluster_shape_mnk[0]
        ) + (self.bx % cluster_shape_mnk[0])

    @consumer_work(returns=(coord_k, coord_m, coord_n))
    @cute.jit
    def compute_coords(
        self, stage_info: StageInfo, *, k_offset: cutlass.Constexpr[int] = 0
    ) -> tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]:
        mma_tile_coord_mnl = (
            self.bx // cluster_shape_mnk[0],
            self.by // cluster_shape_mnk[1],
            self.bz,
        )

        pair_id = self.cta_rank_in_cluster // num_mma_ctas
        rank_in_pair = self.cta_rank_in_cluster % num_mma_ctas
        pair_row = pair_id // num_pair_cols
        pair_col = pair_id % num_pair_cols

        coord_k = stage_info.loop_offset * schedule_tile_k + k_offset
        coord_m = (
            mma_tile_coord_mnl[0] * super_tile_m
            + pair_row * mma_tiler_mnk[0]
            + rank_in_pair * mma_tiler_mnk_per_cta[0]
        )
        coord_n = (
            mma_tile_coord_mnl[1] * super_tile_n
            + pair_col * mma_tiler_mnk[1]
            + rank_in_pair * mma_tiler_mnk_per_cta[1]
        )
        return coord_k, coord_m, coord_n


@dataclass
class SmemAbResource(MemoryResource):
    """Shared-memory A or B buffer filled by an asynchronous TMA load.

    The producer side issues a TMA bulk-copy instruction to move one operand
    from global memory into staged shared-memory buffers, using the
    coordinates provided by GmemAbResource's consumer_work().

    The consumer side builds SMEM descriptors that the Tensor Cores (MMA)
    read during the MMA phase.

    Producer auxiliary work initializes SMEM state before TMA loads. Consumer
    auxiliary work initializes the same state before descriptor construction.
    """

    tma_desc_a: cutlass.Pointer = field(init=False, default=None)
    tma_desc_b: cutlass.Pointer = field(init=False, default=None)
    operand: cutlass.Constexpr[str] = field(init=False, default=None)
    shared_smem: Any = field(init=False, default=None)
    copy_elems: Any = field(init=False, default=None)
    stage_bytes_val: cutlass.Constexpr[int] = field(init=False, default=None)
    cta_rank_in_cluster: Any = field(init=False, default=None)
    rank_in_pair: Any = field(init=False, default=None)
    tma_mcast_mask: Any = field(init=False, default=None)
    is_leader: Any = field(init=False, default=None)
    desc_a_base: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    desc_b_base: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )

    # Actual-cluster multicast parameters (may differ from module globals
    # when running on a fallback cluster).
    act_num_pair_cols: Any = field(init=False, default=None)
    act_a_mcast_template: Any = field(init=False, default=None)
    act_b_mcast_template: Any = field(init=False, default=None)

    desc_previous: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    desc_current: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    _last_desc: Any = field(init=False, default=None)

    # SMEM allocation declarations (offsets set by SmemAllocator)
    _alloc: cutlass.Constexpr = field(init=False, default=None)

    def __init__(
        self,
        tma_desc_a: cutlass.Pointer,
        tma_desc_b: cutlass.Pointer,
        operand: str,
        act_num_pair_cols: int = None,
        act_a_mcast_template: int = None,
        act_b_mcast_template: int = None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.tma_desc_a = tma_desc_a
        self.tma_desc_b = tma_desc_b
        self.desc_previous = TaskLocalVariable(
            dtype=cutlass.Int32, default=cutlass.Int32(0)
        )
        self.desc_current = TaskLocalVariable(
            dtype=cutlass.Int32, default=cutlass.Int32(0)
        )
        self.desc_a_base = TaskLocalVariable(
            dtype=cutlass.Int64,
            default=cutlass.Int64(0),
            docs="SMEM descriptor base for operand A.",
        )
        self.desc_b_base = TaskLocalVariable(
            dtype=cutlass.Int64,
            default=cutlass.Int64(0),
            docs="SMEM descriptor base for operand B.",
        )
        if operand not in {"a", "b"}:
            raise ValueError(
                f"SmemAbResource operand must be 'a' or 'b', got {operand}"
            )
        self.operand = operand
        if operand == "a":
            copy_elems = mma_tiler_mnk_per_cta[0] * mma_tiler_mnk[2]
        else:
            copy_elems = mma_tiler_mnk_per_cta[1] * mma_tiler_mnk[2]
        self.copy_elems = copy_elems
        self.stage_bytes_val = _span_bytes(copy_elems, input_dtype)
        self.act_num_pair_cols = (
            act_num_pair_cols if act_num_pair_cols is not None else num_pair_cols
        )
        self.act_a_mcast_template = (
            act_a_mcast_template
            if act_a_mcast_template is not None
            else _a_mcast_template
        )
        self.act_b_mcast_template = (
            act_b_mcast_template
            if act_b_mcast_template is not None
            else _b_mcast_template
        )
        smem_bytes = self.stage_bytes_val * ab_stages
        self._alloc = SmemAllocation(f"smem_{operand}", smem_bytes, alignment=128)

    @cute.jit
    def _init_smem_state(self, stage_info: StageInfo) -> None:
        # Derive SMEM pointers from the unified allocator base.
        self._last_desc = cutlass.Int32(0)
        smem_base = stage_info.context.smem_base
        self.shared_smem = cutlass.Array(
            smem_base.data_ptr() + self._alloc.offset,
            dtype=cutlass.Uint8,
            shape=(self.stage_bytes_val * ab_stages,),
            addrspace=3,
        )

        # WORKAROUND: materialize multicast parameters as
        # staged values before using them in dynamic control flow. Keep the
        # object fields constexpr so the preferred/fallback branch that owns
        # this resource cannot overwrite the other branch's constants.
        act_num_pair_cols = cutlass.Int32(self.act_num_pair_cols)
        act_a_mcast_template = cutlass.Int32(self.act_a_mcast_template)
        act_b_mcast_template = cutlass.Int32(self.act_b_mcast_template)

        # Use actual-cluster rank for multicast masks and leader determination,
        # since TMA multicast only reaches CTAs within the actual cluster.
        self.cta_rank_in_cluster = cute.arch.block_idx_in_cluster()

        pair_id = self.cta_rank_in_cluster // num_mma_ctas
        self.rank_in_pair = self.cta_rank_in_cluster % num_mma_ctas
        pair_row = pair_id // act_num_pair_cols
        pair_col = pair_id % act_num_pair_cols

        if cutlass.const_expr(self.operand == "a"):
            base = pair_row * act_num_pair_cols * num_mma_ctas + self.rank_in_pair
            self.tma_mcast_mask = act_a_mcast_template << base
            self.is_leader = pair_col == 0
        else:
            base = pair_col * num_mma_ctas + self.rank_in_pair
            self.tma_mcast_mask = act_b_mcast_template << base
            self.is_leader = pair_row == 0

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_load_state(self, stage_info: StageInfo) -> None:
        self._init_smem_state(stage_info)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_descriptors(self, stage_info: StageInfo) -> None:
        self._init_smem_state(stage_info)

    def get_smem_requirements(self):
        return [self._alloc]

    def create_pipeline(self, pipeline_config: PipelineConfig) -> object:
        """Override to set a cluster-wide consumer_mask on the empty barrier.

        PipelineTmaUmma.create() sets consumer_mask = producer_mask (the
        TMA multicast pattern), which only covers CTAs sharing A or B data.
        For the empty barrier, tcgen05_commit must reach ALL CTAs in the
        cluster so that every pair leader's arrival is counted.  Without
        this, diagonal pairs (e.g. CTA 0 and CTA 6 in a 4x2 cluster)
        would miss each other's commits and deadlock.
        """
        pipe = super().create_pipeline(pipeline_config)
        layout = pipeline_config.cta_layout_vmnk
        if layout is not None:
            cluster_total = 1
            for d in layout:
                cluster_total *= d
            if cluster_total > 2:
                object.__setattr__(pipe, "consumer_mask", (1 << cluster_total) - 1)
        return pipe

    @consumer_work(returns=desc_a_base)
    @cute.jit
    def build_desc_a(
        self,
        stage_info: StageInfo,
    ) -> cutlass.Int64:
        desc_a_base = cutlass.Int64(0)
        if self.rank_in_pair == 0:
            sA_curr = self.shared_smem.subview(
                self.stage_bytes_val * stage_info.stage_idx
            )
            if cutlass.const_expr(mma_tiler_mnk[2] > tma_k_box_elems):
                leading_byte_offset = _span_bytes(self.copy_elems, input_dtype)
                stride_byte_offset = _span_bytes(8 * tma_k_box_elems, input_dtype)
            else:
                leading_byte_offset = 16
                stride_byte_offset = _span_bytes(8 * mma_tiler_mnk[2], input_dtype)
            desc_a_base = prims.Tcgen05SmemDesc.build(
                sA_curr,
                leading_byte_offset=leading_byte_offset,
                stride_byte_offset=stride_byte_offset,
                layout=2,
            )
        return desc_a_base

    @consumer_work(returns=desc_b_base)
    @cute.jit
    def build_desc_b(
        self,
        stage_info: StageInfo,
    ) -> cutlass.Int64:
        desc_b_base = cutlass.Int64(0)
        if self.rank_in_pair == 0:
            sB_curr = self.shared_smem.subview(
                self.stage_bytes_val * stage_info.stage_idx
            )
            if cutlass.const_expr(mma_tiler_mnk[2] > tma_k_box_elems):
                leading_byte_offset = _span_bytes(self.copy_elems, input_dtype)
                stride_byte_offset = _span_bytes(8 * tma_k_box_elems, input_dtype)
            else:
                leading_byte_offset = 16
                stride_byte_offset = _span_bytes(8 * mma_tiler_mnk[2], input_dtype)
            desc_b_base = prims.Tcgen05SmemDesc.build(
                sB_curr,
                leading_byte_offset=leading_byte_offset,
                stride_byte_offset=stride_byte_offset,
                layout=2,
            )
        return desc_b_base

    @producer_work
    @cute.jit
    def tma_load_a(
        self,
        stage_info: StageInfo,
        *,
        coord_k: cutlass.Int32,
        coord_m: cutlass.Int32,
    ) -> None:
        if prims.elect_sync():
            if self.is_leader:
                if cutlass.const_expr(num_tma_k_boxes == 1):
                    sA_curr = self.shared_smem.subview(
                        self.stage_bytes_val * stage_info.stage_idx
                    )
                    prims.cp_async_bulk_tensor_shared_cluster_global(
                        sA_curr,
                        self.tma_desc_a,
                        (coord_k, coord_m),
                        stage_info.barrier,
                        [],
                        multicast_mask=self.tma_mcast_mask,
                        group=prims.CTAGroup.CTA_2,
                    )
                else:
                    box_bytes = self.stage_bytes_val // num_tma_k_boxes
                    stage_off = self.stage_bytes_val * stage_info.stage_idx
                    for bi in cutlass.range_constexpr(num_tma_k_boxes):
                        sA_box = self.shared_smem.subview(stage_off + bi * box_bytes)
                        prims.cp_async_bulk_tensor_shared_cluster_global(
                            sA_box,
                            self.tma_desc_a,
                            (
                                coord_k + cutlass.Int32(bi * tma_k_box_elems),
                                coord_m,
                            ),
                            stage_info.barrier,
                            [],
                            multicast_mask=self.tma_mcast_mask,
                            group=prims.CTAGroup.CTA_2,
                        )

    @consumer_work(returns=(desc_previous, desc_current))
    @cute.jit
    def build_desc_3x(self, stage_info: StageInfo):
        """Retain compact SMEM byte addresses across data-tile boundaries.

        The descriptor's stride/layout bits are constant. Carrying only the
        address avoids 64-bit arithmetic and descriptor masks in the MMA
        loop; the elected issuer constructs each descriptor directly.
        """
        previous = self._last_desc
        current = self._stage_address_3x(stage_info)
        self._last_desc = current
        return previous, current

    @consumer_work
    @cute.jit
    def remember_stage_3x(self, stage_info: StageInfo) -> None:
        """Remember AB0 before waiting for AB1 in the first scale group."""
        self._last_desc = self._stage_address_3x(stage_info)

    @cute.jit
    def _stage_address_3x(self, stage_info: StageInfo) -> cutlass.Int32:
        return cutlass.Int32(
            self.shared_smem.data_ptr(
                self.stage_bytes_val * stage_info.stage_idx
            ).toint()
        )

    @producer_work
    @cute.jit
    def tma_load_b(
        self,
        stage_info: StageInfo,
        *,
        coord_k: cutlass.Int32,
        coord_n: cutlass.Int32,
    ) -> None:
        if prims.elect_sync():
            sB_curr = self.shared_smem.subview(
                self.stage_bytes_val * stage_info.stage_idx
            )
            if self.is_leader:
                if cutlass.const_expr(use_block_major_k):
                    block_k = cutlass.Int32(block_major_k_elems)
                    prims.cp_async_bulk_tensor_shared_cluster_global(
                        sB_curr,
                        self.tma_desc_b,
                        (
                            coord_k % block_k,
                            coord_n,
                            coord_k // block_k,
                        ),
                        stage_info.barrier,
                        [],
                        multicast_mask=self.tma_mcast_mask,
                        group=prims.CTAGroup.CTA_2,
                    )
                elif cutlass.const_expr(num_tma_k_boxes == 1):
                    prims.cp_async_bulk_tensor_shared_cluster_global(
                        sB_curr,
                        self.tma_desc_b,
                        (coord_k, coord_n),
                        stage_info.barrier,
                        [],
                        multicast_mask=self.tma_mcast_mask,
                        group=prims.CTAGroup.CTA_2,
                    )
                else:
                    box_bytes = self.stage_bytes_val // num_tma_k_boxes
                    stage_off = self.stage_bytes_val * stage_info.stage_idx
                    for bi in cutlass.range_constexpr(num_tma_k_boxes):
                        sB_box = self.shared_smem.subview(stage_off + bi * box_bytes)
                        prims.cp_async_bulk_tensor_shared_cluster_global(
                            sB_box,
                            self.tma_desc_b,
                            (
                                coord_k + cutlass.Int32(bi * tma_k_box_elems),
                                coord_n,
                            ),
                            stage_info.barrier,
                            [],
                            multicast_mask=self.tma_mcast_mask,
                            group=prims.CTAGroup.CTA_2,
                        )


@dataclass
class SmemSfResource(MemoryResource):
    """Shared-memory block scales for NVFP4, filled by the A or B load task.

    SFA uses A's multicast mask. SFB uses B's mask, plus the MMA-pair bits,
    so both CTAs in a pair end up with the full N tile of scales. The MMA
    task builds an unswizzled S2T descriptor from this buffer.
    """

    tma_desc: cutlass.Pointer = field(init=False, default=None)
    is_a: cutlass.Constexpr[bool] = field(init=False, default=None)
    stage_bytes_val: cutlass.Constexpr[int] = field(init=False, default=None)
    smem_buf: Any = field(init=False, default=None)
    cta_rank_in_cluster: Any = field(init=False, default=None)
    rank_in_pair: Any = field(init=False, default=None)
    tma_mcast_mask: Any = field(init=False, default=None)
    is_leader: Any = field(init=False, default=None)
    desc_s2t_base: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    act_num_pair_cols: Any = field(init=False, default=None)
    act_a_mcast_template: Any = field(init=False, default=None)
    act_b_mcast_template: Any = field(init=False, default=None)
    _alloc: cutlass.Constexpr = field(init=False, default=None)

    def __init__(
        self,
        tma_desc: cutlass.Pointer,
        is_a: bool,
        stage_bytes_val: int,
        act_num_pair_cols: int = None,
        act_a_mcast_template: int = None,
        act_b_mcast_template: int = None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.tma_desc = tma_desc
        self.is_a = is_a
        self.stage_bytes_val = stage_bytes_val
        self.act_num_pair_cols = (
            act_num_pair_cols if act_num_pair_cols is not None else num_pair_cols
        )
        self.act_a_mcast_template = (
            act_a_mcast_template
            if act_a_mcast_template is not None
            else _a_mcast_template
        )
        self.act_b_mcast_template = (
            act_b_mcast_template
            if act_b_mcast_template is not None
            else _b_mcast_template
        )
        self.desc_s2t_base = TaskLocalVariable(
            dtype=cutlass.Int64,
            default=cutlass.Int64(0),
            docs="Unswizzled SMEM descriptor for the S2T scale copy.",
        )
        label = "smem_sfa" if is_a else "smem_sfb"
        self._alloc = SmemAllocation(label, stage_bytes_val * ab_stages, alignment=128)

    def get_smem_requirements(self):
        return [self._alloc]

    def create_pipeline(self, pipeline_config: PipelineConfig) -> object:
        """Same empty-barrier consumer mask as the A/B TMA pipelines."""
        pipe = super().create_pipeline(pipeline_config)
        layout = pipeline_config.cta_layout_vmnk
        if layout is not None:
            cluster_total = 1
            for d in layout:
                cluster_total *= d
            if cluster_total > 2:
                object.__setattr__(pipe, "consumer_mask", (1 << cluster_total) - 1)
        return pipe

    @cute.jit
    def _init_smem_state(self, stage_info: StageInfo) -> None:
        smem_base = stage_info.context.smem_base
        self.smem_buf = cutlass.Array(
            smem_base.data_ptr() + self._alloc.offset,
            dtype=cutlass.Uint8,
            shape=(self.stage_bytes_val * ab_stages,),
            addrspace=3,
        )
        act_num_pair_cols = cutlass.Int32(self.act_num_pair_cols)
        act_a_mcast_template = cutlass.Int32(self.act_a_mcast_template)
        act_b_mcast_template = cutlass.Int32(self.act_b_mcast_template)
        self.cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
        pair_id = self.cta_rank_in_cluster // num_mma_ctas
        self.rank_in_pair = self.cta_rank_in_cluster % num_mma_ctas
        pair_row = pair_id // act_num_pair_cols
        pair_col = pair_id % act_num_pair_cols
        if cutlass.const_expr(self.is_a):
            base = pair_row * act_num_pair_cols * num_mma_ctas + self.rank_in_pair
            self.tma_mcast_mask = act_a_mcast_template << base
            self.is_leader = pair_col == 0
        else:
            # Each SFB box is loaded by one CTA but is needed by both CTAs
            # in every pair sharing B. Include the peer in every row, not
            # just the issuing pair: otherwise remote pairs receive half
            # the expected TMA bytes and never complete their full barrier.
            base = pair_col * num_mma_ctas
            b_mask = act_b_mcast_template << base
            self.tma_mcast_mask = b_mask | (b_mask << 1)
            self.is_leader = pair_row == 0

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_load_state(self, stage_info: StageInfo) -> None:
        self._init_smem_state(stage_info)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_descriptors(self, stage_info: StageInfo) -> None:
        self._init_smem_state(stage_info)

    @producer_work
    @cute.jit
    def tma_load(
        self,
        stage_info: StageInfo,
        *,
        coord_k: cutlass.Int32,
        coord_mn: cutlass.Int32,
    ) -> None:
        sfk = coord_k // (_SF_VEC_SIZE * 4)
        if cutlass.const_expr(self.is_a):
            sfa_boxes = mma_tiler_mnk_per_cta[0] // 128
            box_bytes = self.stage_bytes_val // sfa_boxes
            batch = coord_mn // 128
            for box in cutlass.range_constexpr(sfa_boxes):
                if self.is_leader and prims.elect_sync():
                    smem_stage = self.smem_buf.subview(
                        self.stage_bytes_val * stage_info.stage_idx + box * box_bytes
                    )
                    prims.cp_async_bulk_tensor_shared_cluster_global(
                        smem_stage,
                        self.tma_desc,
                        (0, sfk, batch * sfa_boxes + box, 0),
                        stage_info.barrier,
                        [],
                        multicast_mask=self.tma_mcast_mask,
                        group=prims.CTAGroup.CTA_2,
                    )
        else:
            sfb_boxes = (mma_tiler_mnk[1] + 127) // 128
            box_bytes = self.stage_bytes_val // sfb_boxes
            tile_n_origin = coord_mn - self.rank_in_pair * mma_tiler_mnk_per_cta[1]
            batch0 = tile_n_origin // 128
            for box in cutlass.range_constexpr(sfb_boxes):
                if (
                    self.is_leader
                    and box % num_mma_ctas == self.rank_in_pair
                    and prims.elect_sync()
                ):
                    smem_stage = self.smem_buf.subview(
                        self.stage_bytes_val * stage_info.stage_idx + box * box_bytes
                    )
                    prims.cp_async_bulk_tensor_shared_cluster_global(
                        smem_stage,
                        self.tma_desc,
                        (0, sfk, batch0 + box, 0),
                        stage_info.barrier,
                        [],
                        multicast_mask=self.tma_mcast_mask,
                        group=prims.CTAGroup.CTA_2,
                    )

    @consumer_work(returns=desc_s2t_base)
    @cute.jit
    def build_s2t_descriptor(self, stage_info: StageInfo) -> cutlass.Int64:
        desc_s2t_base = cutlass.Int64(0)
        if self.cta_rank_in_cluster % num_mma_ctas == 0:
            smem_stage = self.smem_buf.subview(
                stage_info.stage_idx * self.stage_bytes_val
            )
            desc_s2t_base = prims.Tcgen05SmemDesc.build(
                smem_stage,
                leading_byte_offset=16,
                stride_byte_offset=128,
                layout=0,
            )
        return desc_s2t_base


@dataclass
class TmemCResource(MemoryResource):
    """Tensor-memory (TMEM) accumulator written by MMA and read by the epilogue.

    The producer side executes tcgen05 MMA instructions that accumulate
    results into TMEM.  The consumer side loads TMEM sub-tiles into
    register memory (RMEM) so the epilogue warps can convert and store
    the final output.

    Producer auxiliary work initializes TMEM state for MMA. Consumer auxiliary
    work initializes the same state for epilogue loads. Per-work-tile producer
    auxiliary work resets the MMA accumulate flag.
    """

    # this must be a constant, but I think putting it here is making it into a variable
    t2r_inst_shape: cutlass.Constexpr[int] = field(init=False, default=None)

    t2r_inst_repx: cutlass.Constexpr[int] = field(init=False, default=None)
    scale_d: Any = field(init=False, default=None)
    idesc: Any = field(init=False, default=None)
    tmem_raw_addr: Any = field(init=False, default=None)
    cta_rank_in_cluster: Any = field(init=False, default=None)
    sfa_tmem_addr_base: Any = field(init=False, default=None)
    sfb_tmem_addr_base: Any = field(init=False, default=None)
    _mma_local_idx: Any = field(init=False, default=None)
    _epi_local_idx: Any = field(init=False, default=None)
    _alloc_acc: cutlass.Constexpr = field(init=False, default=None)
    _alloc_sfa: cutlass.Constexpr = field(init=False, default=None)
    _alloc_sfb: cutlass.Constexpr = field(init=False, default=None)
    t2r_rmem: cutlass.Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    t2r_rmem_prefix: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )

    def __init__(
        self,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        # Pick the right T2R instruction configuration
        self.t2r_inst_shape = "32x32b"
        # This requires tile N to be a multiple of the 32-column epilogue tile.
        self.t2r_inst_repx = _EPILOGUE_TILE_N
        self._alloc_acc = TmemAllocation("tmem_acc", mma_tiler_mnk[1] * acc_stages)
        self._alloc_sfa = TmemAllocation("tmem_sfa", tmem_sfa_cols_per_stage)
        self._alloc_sfb = TmemAllocation("tmem_sfb", tmem_sfb_cols_per_stage)
        self.t2r_rmem = TaskLocalVariable(
            dtype=cutlass.Float32,
            default=cutlass.full([self.t2r_inst_repx], 0.0, cutlass.Float32),
            docs="Register-memory subtile loaded from TMEM for the epilogue.",
        )
        self.t2r_rmem_prefix = TaskLocalVariable(
            dtype=cutlass.Float32,
            default=cutlass.full([self.t2r_inst_repx], 0.0, cutlass.Float32),
            docs="First SF384 overlap subtile retained across the early release.",
        )

    @cute.jit
    def _init_tmem_state(self, stage_info: StageInfo) -> None:
        context = stage_info.context
        # PTX: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor
        if cutlass.const_expr(use_nvfp4_block_scale):
            # FP4 format encoding is 1 in both operand fields (E5M2 in the
            # generic MX builder). Bit 31 selects SM103's dense K=96 mode;
            # scale_format=0 selects E4M3 block scales.
            self.idesc = prims.Tcgen05MxInstrDesc.build(
                a_dtype=cutlass.Float8E5M2,
                b_dtype=cutlass.Float8E5M2,
                scale_format=0,
                n_dim=mma_inst_shape_mnk[1],
                m_dim=mma_inst_shape_mnk[0],
                k_dim=1 if nvfp4_mma_k == 96 else 0,
            )
        else:
            self.idesc = prims.Tcgen05InstrDesc.build(
                a_dtype=input_dtype,
                b_dtype=input_dtype,
                c_dtype=cutlass.Float32,
                n_dim=mma_inst_shape_mnk[1],
                m_dim=mma_inst_shape_mnk[0],
            )
        self.tmem_raw_addr = context.tmem_ptr_i32.load()
        self.cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
        if cutlass.const_expr(use_nvfp4_tmem_overlap):
            self._mma_local_idx = cutlass.Int32(0)
            self._epi_local_idx = cutlass.Int32(0)
        if cutlass.const_expr(use_nvfp4_block_scale):
            base_col_id = self.tmem_raw_addr & 0xFFFF
            base_row_id = self.tmem_raw_addr >> 16
            sfa_col_id = base_col_id + mma_tiler_mnk[1] * acc_stages
            sfb_col_id = sfa_col_id + tmem_sfa_cols_per_stage
            self.sfa_tmem_addr_base = (base_row_id << 16) | sfa_col_id
            self.sfb_tmem_addr_base = (base_row_id << 16) | sfb_col_id
        # Initialize scale_d before dynamic while loops so the MLIR
        # structure stays consistent when the work-tile reset reassigns it.
        self.scale_d = cutlass.Boolean(False)

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_accumulator_state(self, stage_info: StageInfo) -> None:
        self._init_tmem_state(stage_info)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_store_state(self, stage_info: StageInfo) -> None:
        self._init_tmem_state(stage_info)

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_work_tile_state(self, stage_info: StageInfo) -> None:
        del stage_info
        self.scale_d = False

    def get_tmem_requirements(self):
        if use_nvfp4_block_scale:
            if use_nvfp4_tmem_overlap:
                return [self._alloc_acc]
            return [self._alloc_acc, self._alloc_sfa, self._alloc_sfb]
        return [self._alloc_acc]

    @consumer_work(returns=t2r_rmem)
    @cute.jit
    def load_subtile(
        self, stage_info: StageInfo, *, subtile_idx: cutlass.Constexpr[int]
    ) -> cutlass.Float32:
        return self._load_subtile_impl(stage_info, subtile_idx)

    @consumer_work(
        returns=t2r_rmem,
        work_attrs=WorkAttr.AUXILIARY,
    )
    @cute.jit
    def load_overlap_subtile(
        self, stage_info: StageInfo, *, subtile_idx: cutlass.Constexpr[int]
    ) -> cutlass.Float32:
        result = self._load_subtile_impl(stage_info, subtile_idx)
        if cutlass.const_expr(
            subtile_idx
            == mma_tiler_mnk[1] // (num_epilogue_warps // 4) // _EPILOGUE_TILE_N - 1
        ):
            self._epi_local_idx = self._epi_local_idx ^ cutlass.Int32(1)
        return result

    @consumer_work(returns=t2r_rmem_prefix)
    @cute.jit
    def load_prefix_subtile(self, stage_info: StageInfo) -> cutlass.Float32:
        """Subtile 0 into the prefix slot, kept across the QKNorm early release."""
        return self._load_subtile_impl(stage_info, 0)

    @consumer_work(returns=(t2r_rmem_prefix, t2r_rmem))
    @cute.jit
    def load_overlap_prefix(self, stage_info: StageInfo):
        """Retire both SF384 scale regions before doing any global stores."""
        first = self._load_subtile_impl(stage_info, 0)
        second = self._load_subtile_impl(stage_info, 1)
        return first, second

    @cute.jit
    def _load_subtile_impl(self, stage_info: StageInfo, subtile_idx):
        warp_idx = cute.arch.warp_idx()
        if cutlass.const_expr(num_epilogue_warps == 8):
            epilogue_group = warp_idx // 4
            row_warp_idx = warp_idx % 4
            group_col_offset = epilogue_group * (mma_tiler_mnk[1] // 2)
        else:
            row_warp_idx = warp_idx
            group_col_offset = 0
        # Compute the tensor memory address
        physical_stage_idx = stage_info.stage_idx
        if cutlass.const_expr(use_nvfp4_tmem_overlap):
            physical_stage_idx = self._epi_local_idx
        base_col_id = (self.tmem_raw_addr & 0xFFFF) + (
            physical_stage_idx * mma_tiler_mnk[1]
        )
        base_row_id = self.tmem_raw_addr >> 16
        # Each warp accesses different TMEM rows
        # (warp 0 -> row 0-31, warp 1 -> row 32-63, etc.)
        row_id_with_warp_offset = base_row_id + row_warp_idx * 32
        current_tmem_raw_addr = (row_id_with_warp_offset << 16) | base_col_id

        # TMEM -> RMEM
        # step1: compute tensor memory address for subtile
        curr_tmem_raw_subtile_addr = (
            current_tmem_raw_addr + group_col_offset + subtile_idx * self.t2r_inst_repx
        )

        # step2: get TMEM pointer and load data as Float32
        tmem_ptr = prims.make_tmem_ptr(curr_tmem_raw_subtile_addr, cutlass.Float32)

        t2r_rmem = prims.tcgen05_ld(
            self.t2r_inst_shape, tmem_ptr, num=self.t2r_inst_repx
        )
        prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
        cute.arch.fence_view_async_tmem_load()
        return t2r_rmem

    @producer_work
    @cute.jit
    def mma(
        self,
        stage_info: StageInfo,
        *,
        desc_a_base: cutlass.Int64,
        desc_b_base: cutlass.Int64,
    ) -> None:
        self._mma_impl(stage_info, desc_a_base, desc_b_base, None, None)

    @producer_work
    @cute.jit
    def mma_block_scaled(
        self,
        stage_info: StageInfo,
        *,
        desc_a_base: cutlass.Int64,
        desc_b_base: cutlass.Int64,
        desc_sfa_base: cutlass.Int64,
        desc_sfb_base: cutlass.Int64,
    ) -> None:
        self._mma_impl(
            stage_info,
            desc_a_base,
            desc_b_base,
            desc_sfa_base,
            desc_sfb_base,
        )

    @cute.jit
    def _mma_impl(
        self,
        stage_info: StageInfo,
        desc_a_base,
        desc_b_base,
        desc_sfa_base,
        desc_sfb_base,
        desc_a_previous=None,
        desc_b_previous=None,
        sf_group: cutlass.Constexpr[int] = 0,
    ) -> None:
        if cutlass.const_expr(use_nvfp4_block_scale and nvfp4_mma_k == 96):
            # Keep descriptor arithmetic, scale copies and the MMA batch in
            # one elected lane. Re-electing around each instruction makes
            # ptxas shuttle the 64-bit descriptors between GPRs and URs.
            if self.cta_rank_in_cluster % num_mma_ctas == 0:
                if prims.elect_sync():
                    self._mma_impl_body(
                        stage_info,
                        desc_a_base,
                        desc_b_base,
                        desc_sfa_base,
                        desc_sfb_base,
                        desc_a_previous,
                        desc_b_previous,
                        sf_group,
                    )
            # The flag is warp state, not elected-lane state. Every lane
            # must carry the same value into the next elected batch.
            self.scale_d = True
        else:
            self._mma_impl_body(
                stage_info,
                desc_a_base,
                desc_b_base,
                desc_sfa_base,
                desc_sfb_base,
            )

    @cute.jit
    def _mma_impl_body(
        self,
        stage_info: StageInfo,
        desc_a_base,
        desc_b_base,
        desc_sfa_base,
        desc_sfb_base,
        desc_a_previous=None,
        desc_b_previous=None,
        sf_group: cutlass.Constexpr[int] = 0,
    ) -> None:
        sfa_tmem_addr_base = self.sfa_tmem_addr_base
        sfb_tmem_addr_base = self.sfb_tmem_addr_base
        if cutlass.const_expr(use_nvfp4_tmem_overlap):
            base_col_id = self.tmem_raw_addr & 0xFFFF
            base_row_id = self.tmem_raw_addr >> 16
            # Put scales in the opposite accumulator window, at offsets
            # 0 and 128. The epilogue retires enough subtiles before its
            # early release to cover both regions: one for SF256 (16/32
            # columns), two for SF384 (24/48 columns).
            phase = self._mma_local_idx
            sfa_col_id = (
                base_col_id
                + cutlass.Int32(mma_tiler_mnk[1])
                - phase * cutlass.Int32(mma_tiler_mnk[1])
            )
            sfb_col_id = (
                base_col_id
                + cutlass.Int32(mma_tiler_mnk[1] + mma_tiler_mnk[1] // 2)
                - phase * cutlass.Int32(mma_tiler_mnk[1])
            )
            sfa_tmem_addr_base = (base_row_id << 16) | sfa_col_id
            sfb_tmem_addr_base = (base_row_id << 16) | sfb_col_id
        if cutlass.const_expr(use_nvfp4_block_scale):
            self._copy_block_scales(
                desc_sfa_base,
                desc_sfb_base,
                sfa_tmem_addr_base,
                sfb_tmem_addr_base,
            )
        if self.cta_rank_in_cluster % num_mma_ctas == 0:
            tmem_ptr = prims.make_tmem_ptr(self.tmem_raw_addr, acc_dtype)
            physical_stage_idx = stage_info.stage_idx
            if cutlass.const_expr(use_nvfp4_tmem_overlap):
                physical_stage_idx = self._mma_local_idx
            tmem_ptr_for_mma = (
                tmem_ptr.data_ptr() + physical_stage_idx * mma_tiler_mnk[1]
            )
            tmem_ptr_curr = cutlass.Array(
                tmem_ptr_for_mma,
                dtype=cutlass.Int32,
                addrspace=6,
            )

            # Execute one scale tile. In the split-K path this is four K=96
            # MMAs, which can span two independently buffered AB stages.
            num_k_blocks = sf_tile_k // mma_inst_shape_mnk[2]
            inc_minor = _span_bytes(mma_inst_shape_mnk[2], input_dtype) >> 4
            k_blocks_per_box = tma_k_box_elems // mma_inst_shape_mnk[2]
            inc_a_major = (
                _span_bytes(mma_tiler_mnk_per_cta[0] * tma_k_box_elems, input_dtype)
                >> 4
            )
            inc_b_major = (
                _span_bytes(mma_tiler_mnk_per_cta[1] * tma_k_box_elems, input_dtype)
                >> 4
            )
            for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                if cutlass.const_expr(use_nvfp4_split_k):
                    k_element = sf_group * sf_tile_k + k_block_idx * 96
                    k_major = k_element // tma_k_box_elems
                    byte_offset = (k_element % tma_k_box_elems) // 2
                    if cutlass.const_expr(k_major == sf_group + 1):
                        addr_a = desc_a_base + byte_offset
                        addr_b = desc_b_base + byte_offset
                    else:
                        addr_a = desc_a_previous + byte_offset
                        addr_b = desc_b_previous + byte_offset
                    crosses_box = k_element % tma_k_box_elems + 96 > tma_k_box_elems
                    desc_a = prims.Tcgen05SmemDesc.build(
                        addr_a,
                        leading_byte_offset=desc_a_base if crosses_box else 16,
                        stride_byte_offset=1024,
                        layout=2,
                        leading_dim_mode=1 if crosses_box else 0,
                    )
                    desc_b = prims.Tcgen05SmemDesc.build(
                        addr_b,
                        leading_byte_offset=desc_b_base if crosses_box else 16,
                        stride_byte_offset=1024,
                        layout=2,
                        leading_dim_mode=1 if crosses_box else 0,
                    )
                elif cutlass.const_expr(use_nvfp4_block_scale and nvfp4_mma_k == 96):
                    # The large-tile path keeps all three boxes contiguous.
                    k_element = k_block_idx * 96
                    k_major = k_element // tma_k_box_elems
                    k_minor = (k_element % tma_k_box_elems) // 32
                    desc_a = desc_a_base + k_major * inc_a_major + k_minor
                    desc_b = desc_b_base + k_major * inc_b_major + k_minor
                    if cutlass.const_expr(
                        k_element % tma_k_box_elems + 96 > tma_k_box_elems
                    ):
                        # SM103's absolute LBO points at the next TMA box when
                        # a 48-byte instruction straddles the 128-byte boundary.
                        next_a = (desc_a_base + (k_major + 1) * inc_a_major) & 0x3FFF
                        next_b = (desc_b_base + (k_major + 1) * inc_b_major) & 0x3FFF
                        desc_a = (desc_a & ~(0x3FFF << 16)) | (next_a << 16) | (1 << 52)
                        desc_b = (desc_b & ~(0x3FFF << 16)) | (next_b << 16) | (1 << 52)
                elif cutlass.const_expr(mma_tiler_mnk[2] > tma_k_box_elems):
                    k_minor = k_block_idx % k_blocks_per_box
                    k_major = k_block_idx // k_blocks_per_box
                    desc_a = desc_a_base + k_major * inc_a_major + k_minor * inc_minor
                    desc_b = desc_b_base + k_major * inc_b_major + k_minor * inc_minor
                else:
                    increment = inc_minor * k_block_idx
                    desc_a = desc_a_base + increment
                    desc_b = desc_b_base + increment

                if cutlass.const_expr(nvfp4_mma_k == 96) or prims.elect_sync():
                    # submit MMA from one elected thread from leader CTA only
                    if cutlass.const_expr(use_nvfp4_block_scale):
                        sfa_kblock_cols = tmem_sfa_cols_per_stage // num_k_blocks
                        sfb_kblock_cols = tmem_sfb_cols_per_stage // num_k_blocks
                        sfa_offset = k_block_idx * sfa_kblock_cols
                        sfb_offset = k_block_idx * sfb_kblock_cols
                        idesc = self.idesc
                        if cutlass.const_expr(nvfp4_mma_k == 96):
                            # Six scales per instruction: start at byte 0 of
                            # one 4-column group, then byte 2 of the next.
                            sf_index = k_block_idx * 6
                            sfa_offset = (sf_index // 4) * 4
                            sfb_offset = sfa_offset * ((mma_tiler_mnk[1] + 127) // 128)
                            idesc = self.idesc.set_sf_ids(sf_index % 4, sf_index % 4)
                        sfa_ptr = prims.make_tmem_ptr(
                            sfa_tmem_addr_base + sfa_offset,
                            cutlass.Int32,
                        )
                        sfb_ptr = prims.make_tmem_ptr(
                            sfb_tmem_addr_base + sfb_offset,
                            cutlass.Int32,
                        )
                        prims.tcgen05_mma_block_scale(
                            prims.MMABlockScaleKind.MXF4NVF4,
                            prims.CTAGroup.CTA_2,
                            tmem_ptr_curr,
                            desc_a,
                            desc_b,
                            idesc,
                            enable_input_d=self.scale_d,
                            scale_a=sfa_ptr,
                            scale_b=sfb_ptr,
                            scale_vec_size=prims.Tcgen05MMABlockScale.BLOCK16,
                        )
                    else:
                        prims.tcgen05_mma(
                            mma_kind,
                            prims.CTAGroup.CTA_2,
                            tmem_ptr_curr,
                            desc_a,
                            desc_b,
                            self.idesc,
                            self.scale_d,
                        )
                # switch to accumulate after first iteration
                self.scale_d = True

    @producer_work
    @cute.jit
    def mma_block_scaled_3x(
        self,
        stage_info: StageInfo,
        *,
        desc_a_previous: cutlass.Int32,
        desc_a_base: cutlass.Int32,
        desc_b_previous: cutlass.Int32,
        desc_b_base: cutlass.Int32,
        desc_sfa_base: cutlass.Int64,
        desc_sfb_base: cutlass.Int64,
        sf_group: cutlass.Constexpr[int],
    ) -> None:
        self._mma_impl(
            stage_info,
            desc_a_base,
            desc_b_base,
            desc_sfa_base,
            desc_sfb_base,
            desc_a_previous,
            desc_b_previous,
            sf_group,
        )

    @cute.jit
    def _copy_block_scales(
        self,
        desc_sfa_base,
        desc_sfb_base,
        sfa_tmem_addr_base,
        sfb_tmem_addr_base,
    ) -> None:
        """SMEM to TMEM copy of the block scales, fused into the MMA task."""
        if self.cta_rank_in_cluster % num_mma_ctas == 0:
            s2t_shape, s2t_multicast = prims.S2TCopyMode.S2T_32x128b_WARPX4
            for s2t_idx in cutlass.range_constexpr(sfa_stage_bytes // 512):
                sfa_ptr = prims.make_tmem_ptr(
                    sfa_tmem_addr_base + s2t_idx * 4,
                    cutlass.Int32,
                )
                if cutlass.const_expr(nvfp4_mma_k == 96) or prims.elect_sync():
                    prims.tcgen05_cp(
                        s2t_shape,
                        sfa_ptr,
                        desc_sfa_base + 32 * s2t_idx,
                        group=prims.CTAGroup.CTA_2,
                        multicast=s2t_multicast,
                    )
            sfb_blocks = (mma_tiler_mnk[1] + 127) // 128
            sfb_block_stride = (sfb_stage_bytes // sfb_blocks) // 16
            for s2t_idx in cutlass.range_constexpr(sfb_stage_bytes // 512):
                sfb_ptr = prims.make_tmem_ptr(
                    sfb_tmem_addr_base + s2t_idx * 4,
                    cutlass.Int32,
                )
                increment = 32 * (s2t_idx // sfb_blocks) + sfb_block_stride * (
                    s2t_idx % sfb_blocks
                )
                if cutlass.const_expr(nvfp4_mma_k == 96) or prims.elect_sync():
                    prims.tcgen05_cp(
                        s2t_shape,
                        sfb_ptr,
                        desc_sfb_base + increment,
                        group=prims.CTAGroup.CTA_2,
                        multicast=s2t_multicast,
                    )
            # cp -> MMA and MMA -> cp are pipelined in the issuing warp.
            # wait::st only tracks tcgen05.st, which this path does not issue.
            if cutlass.const_expr(nvfp4_mma_k != 96):
                prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def advance_mma_overlap_window(self, stage_info: StageInfo) -> None:
        del stage_info
        if cutlass.const_expr(use_nvfp4_tmem_overlap):
            self._mma_local_idx = self._mma_local_idx ^ cutlass.Int32(1)


@dataclass
class GmemDResource(MemoryResource):
    """Global-memory D output resource (write-only sink).

    This resource has no consumer side — it is the final destination of
    the dataflow pipeline.  The producer side takes RMEM data from
    TmemCResource, optionally applies a column bias of shape [N],
    dequantizes, converts to the output dtype (FP16/BF16/FP8), and
    issues vectorized global stores.

    Dequant is one of:

      * per-tensor: multiply by a host FP32 scalar (FP8 I/O default)
      * per-token / per-channel: ``acc[m, n] * x_scale[m] * weight_scale[n]``
        with FP32 vectors of shape [M] and [N]. Each epilogue thread owns
        one output row, so ``x_scale[m]`` is loaded to a register once per
        tile. ``weight_scale`` for the tile-N-wide N tile is cooperatively
        staged to SMEM, then read while the tile is stored.

    Optional gated activation (after scale/bias): neighbouring GEMM columns
    are (gate, act) pairs from an interleaved weight layout. Each 32-wide
    TMEM subtile yields 16 outputs, ``gate * silu(act)``, stored to D of
    shape ``(M, N/2)``.

    Methods
    -------
    producer_work()
        Applies optional bias, scales, optional gated SiLU, converts FP32
        accumulators, and writes the result to global memory D.

    The producer-side t2r_rmem slot is auto-allocated from the upstream
    TmemCResource TaskLocalVariable flow.
    """

    t2r_inst_repx: cutlass.Constexpr[int] = field(init=False, default=None)
    num_output_subtiles: cutlass.Constexpr[int] = field(init=False, default=None)
    bias: Any = field(init=False, default=None)
    scale: Any = field(init=False, default=None)
    x_scale: Any = field(init=False, default=None)
    weight_scale: Any = field(init=False, default=None)
    sf_c: Any = field(init=False, default=None)
    scale_c: Any = field(init=False, default=None)
    scale_gate: Any = field(init=False, default=None)
    qkv_scale: Any = field(init=False, default=None)
    x_scale_reg: Any = field(init=False, default=None)
    weight_scale_smem: Any = field(init=False, default=None)
    mC_mn: Any = field(init=False, default=None)
    gC: Any = field(init=False, default=None)
    gC_bytes: Any = field(init=False, default=None)
    gSfC_bytes: Any = field(init=False, default=None)
    tma_c_desc: Any = field(init=False, default=None)
    output_smem: Any = field(init=False, default=None)
    q_norm_weight: Any = field(init=False, default=None)
    k_norm_weight: Any = field(init=False, default=None)
    cos_sin_cache: Any = field(init=False, default=None)
    positions: Any = field(init=False, default=None)
    position_reg: Any = field(init=False, default=None)
    qk_norm_weight_smem: Any = field(init=False, default=None)
    norm_sum: Any = field(init=False, default=None)
    norm_factor: Any = field(init=False, default=None)
    vsize: cutlass.Constexpr[int] = field(init=False, default=None)
    _weight_scale_alloc: cutlass.Constexpr = field(init=False, default=None)
    _output_alloc: cutlass.Constexpr = field(init=False, default=None)
    _qk_norm_weight_alloc: cutlass.Constexpr = field(init=False, default=None)

    def __init__(
        self,
        mC_mn: cute.Tensor,
        tma_c_desc: cutlass.Pointer,
        bias: Optional[cute.Tensor] = None,
        scale: cutlass.Float32 = 1.0,
        x_scale: Optional[cute.Tensor] = None,
        weight_scale: Optional[cute.Tensor] = None,
        sf_c: Optional[cute.Tensor] = None,
        scale_c: Optional[cute.Tensor] = None,
        scale_gate: Optional[cute.Tensor] = None,
        qkv_scale: Optional[cute.Tensor] = None,
        q_norm_weight: Optional[cute.Tensor] = None,
        k_norm_weight: Optional[cute.Tensor] = None,
        cos_sin_cache: Optional[cute.Tensor] = None,
        positions: Optional[cute.Tensor] = None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.t2r_inst_repx = _EPILOGUE_TILE_N
        self.num_output_subtiles = mma_tiler_mnk[1] // self.t2r_inst_repx
        self.bias = bias
        self.scale = scale
        self.x_scale = x_scale
        self.weight_scale = weight_scale
        self.sf_c = sf_c
        self.scale_c = scale_c
        self.scale_gate = scale_gate
        self.qkv_scale = qkv_scale
        self.q_norm_weight = q_norm_weight
        self.k_norm_weight = k_norm_weight
        self.cos_sin_cache = cos_sin_cache
        self.positions = positions
        self.position_reg = cutlass.Int64(0)
        self.x_scale_reg = cutlass.Float32(1.0)
        self.norm_sum = cutlass.Float32(0.0)
        self.norm_factor = cutlass.Float32(1.0)
        # Global C view supplies the output element type/vector width. Output
        # data itself is staged through SMEM and written by TMA.
        self.mC_mn = mC_mn
        self.gC = cutlass.make_array_view(mC_mn)
        if _is_fp4(output_dtype):
            self.gC_bytes = cutlass.Array(self.gC.data_ptr(), dtype=cutlass.Int8)
            self.gSfC_bytes = cutlass.Array(
                cutlass.make_array_view(sf_c).data_ptr(), dtype=cutlass.Int8
            )
        self.tma_c_desc = tma_c_desc
        # One T2R call yields 32 FP32 values. FP4 consumes all 32 and, in the
        # gated path, turns them into one 16-value quantization block.
        self.vsize = min(_EPILOGUE_TILE_N, 256 // self.gC.dtype.width)
        self._weight_scale_alloc = SmemAllocation(
            "weight_scale_smem",
            mma_tiler_mnk[1] * (cutlass.Float32.width // 8),
            alignment=16,
        )
        self._qk_norm_weight_alloc = SmemAllocation(
            "qk_norm_weight_smem",
            2 * _QKV_HEAD_DIM * (cutlass.BFloat16.width // 8),
            alignment=16,
        )
        output_subtile_n = 16 if use_gated_activation else _EPILOGUE_TILE_N
        self._output_alloc = SmemAllocation(
            "output_smem",
            mma_tiler_mnk_per_cta[0]
            * output_subtile_n
            * _epilogue_store_stages
            * (num_epilogue_warps // 4)
            * max(1, output_dtype.width // 8),
            alignment=128,
        )

    def get_smem_requirements(self):
        requirements = []
        if use_per_token_channel_scale:
            requirements.append(self._weight_scale_alloc)
        if use_tma_store:
            requirements.append(self._output_alloc)
        if use_fused_qknorm_rope:
            requirements.append(self._qk_norm_weight_alloc)
        return requirements

    @cute.jit
    def _cta_output_mn(
        self, stage_info: StageInfo
    ) -> tuple[cutlass.Int32, cutlass.Int32]:
        bx, by, bz = stage_info.work_tile.tile_idx
        cta_rank_for_coords = (by % cluster_shape_mnk[1]) * cluster_shape_mnk[0] + (
            bx % cluster_shape_mnk[0]
        )
        mma_tile_coord_mnl = (
            bx // cluster_shape_mnk[0],
            by // cluster_shape_mnk[1],
            bz,
        )

        pair_id = cta_rank_for_coords // num_mma_ctas
        rank_in_pair = cta_rank_for_coords % num_mma_ctas
        pair_row = pair_id // num_pair_cols
        pair_col = pair_id % num_pair_cols

        coordc_m = (
            mma_tile_coord_mnl[0] * super_tile_m
            + pair_row * mma_tiler_mnk[0]
            + rank_in_pair * mma_tiler_mnk_per_cta[0]
        )
        coordc_n = mma_tile_coord_mnl[1] * super_tile_n + pair_col * mma_tiler_mnk[1]
        return coordc_m, coordc_n

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def preload_rowcol_scales(self, stage_info: StageInfo) -> None:
        """Load x_scale[row] to a register and the N-tile of weight_scale to SMEM.

        Scheduled before the accumulator wait: both scales depend only on the
        work tile's M/N origin, so these global loads overlap the TMEM wait.
        """
        coordc_m, coordc_n = self._cta_output_mn(stage_info)
        tx, _, _ = cute.arch.thread_idx()
        row = coordc_m + tx % mma_tiler_mnk_per_cta[0]
        num_rows = self.mC_mn.shape[0]

        smem_base = stage_info.context.smem_base
        self.weight_scale_smem = cutlass.Array(
            smem_base.data_ptr() + self._weight_scale_alloc.offset,
            dtype=cutlass.Float32,
            shape=(mma_tiler_mnk[1],),
            addrspace=3,
        )

        x_scale_ptr = self.x_scale.iterator.raw_ptr()
        if row < num_rows:
            self.x_scale_reg = (x_scale_ptr + cutlass.Int64(row)).load()
        else:
            self.x_scale_reg = cutlass.Float32(1.0)

        # The SMEM tile is reused by every work tile, so the previous tile's
        # readers must be done before this tile overwrites it.
        cute.arch.barrier(
            barrier_id=_epilogue_scale_barrier_id,
            number_of_threads=threads_in_epilogue,
        )

        # Epilogue threads cooperatively move the tile-N FP32 scale vector.
        weight_scale_ptr = self.weight_scale.iterator.raw_ptr()
        smem_ptr = self.weight_scale_smem.data_ptr()
        vec_n = 4
        n_vecs = mma_tiler_mnk[1] // vec_n
        if tx < n_vecs:
            chunk = coordc_n + tx * vec_n
            ws_vec = (weight_scale_ptr + cutlass.Int64(chunk)).load(
                count=vec_n, alignment=16
            )
            (smem_ptr + cutlass.Int64(tx * vec_n)).store(ws_vec, alignment=16)

        cute.arch.barrier(
            barrier_id=_epilogue_scale_barrier_id,
            number_of_threads=threads_in_epilogue,
        )

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_qknorm_weights(self, stage_info: StageInfo) -> None:
        """Stage the launch-invariant Q/K affine weights once per CTA."""
        smem_base = stage_info.context.smem_base
        self.qk_norm_weight_smem = cutlass.Array(
            smem_base.data_ptr() + self._qk_norm_weight_alloc.offset,
            dtype=cutlass.BFloat16,
            shape=(2 * _QKV_HEAD_DIM,),
            addrspace=3,
        )
        tx, _, _ = cute.arch.thread_idx()
        if tx < _QKV_HEAD_DIM:
            q_weight = (
                self.q_norm_weight.iterator.raw_ptr() + cutlass.Int64(tx)
            ).load()
            k_weight = (
                self.k_norm_weight.iterator.raw_ptr() + cutlass.Int64(tx)
            ).load()
            (self.qk_norm_weight_smem.data_ptr() + cutlass.Int64(tx)).store(q_weight)
            (
                self.qk_norm_weight_smem.data_ptr() + cutlass.Int64(_QKV_HEAD_DIM + tx)
            ).store(k_weight)
        cute.arch.barrier(
            barrier_id=_epilogue_qknorm_barrier_id,
            number_of_threads=threads_in_epilogue,
        )

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def preload_qknorm_rope(self, stage_info: StageInfo) -> None:
        """Hoist this thread's RoPE position out of the Q/K vector loop."""
        coordc_m, _ = self._cta_output_mn(stage_info)
        tx, _, _ = cute.arch.thread_idx()
        row_in_tile = tx % mma_tiler_mnk_per_cta[0]
        row = coordc_m + row_in_tile
        num_rows = self.mC_mn.shape[0]
        safe_row = row
        if row >= num_rows:
            safe_row = cutlass.Int32(0)
        position = (self.positions.iterator.raw_ptr() + cutlass.Int64(safe_row)).load()
        # cos_sin has one row per token. An index outside [0, M) would form
        # an address past that table; row 0 is a defined result.
        if position < cutlass.Int64(0):
            position = cutlass.Int64(0)
        if position >= cutlass.Int64(num_rows):
            position = cutlass.Int64(0)
        self.position_reg = position

    @cute.jit
    def _silu(self, x: cutlass.Float32) -> cutlass.Float32:
        one = cutlass.Float32(1.0)
        return x * (one / (one + cute.math.exp(-x, fastmath=True)))

    @cute.jit
    def _apply_linear_epilogue_vec(
        self,
        vec_f32,
        chunk_col,
        local_col,
        bias_ptr,
        count: cutlass.Constexpr[int],
    ):
        if cutlass.const_expr(use_per_token_channel_scale):
            ws_vec = (
                self.weight_scale_smem.data_ptr() + cutlass.Int64(local_col)
            ).load(count=count, alignment=16)
            if cutlass.const_expr(self.bias is not None):
                bias_vec = (
                    (bias_ptr + cutlass.Int64(chunk_col))
                    .load(count=count, alignment=16)
                    .to(cutlass.Float32)
                )

            # Generic Vector arithmetic is scalarized before NVVM lowering.
            # Use Blackwell's explicit packed f32x2 intrinsics so each thread
            # scales two columns per issued instruction.
            scaled = []
            for i in cutlass.range_constexpr(count // 2):
                elem = 2 * i
                row_scaled = cute.arch.mul_packed_f32x2(
                    (vec_f32[elem], vec_f32[elem + 1]),
                    (self.x_scale_reg, self.x_scale_reg),
                )
                weight_pair = (ws_vec[elem], ws_vec[elem + 1])
                if cutlass.const_expr(self.bias is not None):
                    scaled_pair = cute.arch.fma_packed_f32x2(
                        row_scaled,
                        weight_pair,
                        (bias_vec[elem], bias_vec[elem + 1]),
                    )
                else:
                    scaled_pair = cute.arch.mul_packed_f32x2(row_scaled, weight_pair)
                scaled.extend(scaled_pair)
            vec_f32 = cutlass.Vector.from_elements(tuple(scaled), cutlass.Float32)
        elif cutlass.const_expr(self.bias is not None):
            bias_vec = (
                (bias_ptr + cutlass.Int64(chunk_col))
                .load(count=count, alignment=16)
                .to(cutlass.Float32)
            )
            vec_f32 = vec_f32 + bias_vec
        if cutlass.const_expr(self.qkv_scale is not None):
            # Packed QKV is laid out as contiguous, equal-sized [Q | K | V]
            # channel ranges. Each entry is the complete factor for that
            # projection, so it replaces the single per-tensor ``scale``.
            qkv_hidden = self.mC_mn.shape[1] // 3
            qkv_idx = chunk_col // qkv_hidden
            qkv_scale_value = (
                self.qkv_scale.iterator.raw_ptr() + cutlass.Int64(qkv_idx)
            ).load()
            vec_f32 = vec_f32 * qkv_scale_value
        if cutlass.const_expr(
            needs_epilogue_scale
            and not use_per_token_channel_scale
            and not _is_fp4(output_dtype)
            and self.scale_gate is None
            and self.qkv_scale is None
        ):
            vec_f32 = vec_f32 * self.scale
        return vec_f32

    @cute.jit
    def _apply_linear_epilogue_scalar(
        self, value, col_idx, local_col, bias_ptr
    ) -> cutlass.Float32:
        if cutlass.const_expr(use_per_token_channel_scale):
            value = (
                value
                * self.x_scale_reg
                * (self.weight_scale_smem.data_ptr() + cutlass.Int64(local_col)).load()
            )
        if cutlass.const_expr(self.bias is not None):
            value = value + (bias_ptr + cutlass.Int64(col_idx)).load().to(
                cutlass.Float32
            )
        if cutlass.const_expr(self.qkv_scale is not None):
            # Scalar tail path mirrors the vector path above.
            qkv_hidden = self.mC_mn.shape[1] // 3
            qkv_idx = col_idx // qkv_hidden
            value = (
                value
                * (self.qkv_scale.iterator.raw_ptr() + cutlass.Int64(qkv_idx)).load()
            )
        if cutlass.const_expr(
            needs_epilogue_scale
            and not use_per_token_channel_scale
            and not _is_fp4(output_dtype)
            and self.scale_gate is None
            and self.qkv_scale is None
        ):
            value = value * self.scale
        return value

    @cute.jit
    def _gated_pair_silu(self, vec_f32, count: cutlass.Constexpr[int]):
        """Neighbouring columns are (gate, act); return gate * silu(act)."""
        outs = []
        one_pair = (cutlass.Float32(1.0), cutlass.Float32(1.0))
        # Process two (gate, act) pairs together. MUFU.EX2 and MUFU.RCP are
        # scalar, but the surrounding add and multiplies map to FADD2/FMUL2.
        for i in cutlass.range_constexpr(count // 4):
            elem = 4 * i
            gate_pair = (vec_f32[elem], vec_f32[elem + 2])
            act_pair = (vec_f32[elem + 1], vec_f32[elem + 3])
            exp_pair = (
                cute.math.exp(-act_pair[0], fastmath=True),
                cute.math.exp(-act_pair[1], fastmath=True),
            )
            denom_pair = cute.arch.add_packed_f32x2(exp_pair, one_pair)
            sigmoid_pair = (
                cutlass.Float32(1.0) / denom_pair[0],
                cutlass.Float32(1.0) / denom_pair[1],
            )
            silu_pair = cute.arch.mul_packed_f32x2(act_pair, sigmoid_pair)
            out_pair = cute.arch.mul_packed_f32x2(gate_pair, silu_pair)
            outs.extend(out_pair)
        return cutlass.Vector.from_elements(tuple(outs), cutlass.Float32)

    @cute.jit
    def _gated_pair_silu_quant(
        self,
        vec_f32,
        scale_gate: cutlass.Float32,
        count: cutlass.Constexpr[int],
    ):
        """Gated SiLU before output encoding.

        ``scale_gate`` is the input dequantization product. The second global
        factor, ``scale_c``, is deliberately applied by the quantizer below:

          encoded = ScaleC * (ScaleGate * gate) * act * sigmoid(ScaleGate * act)

        Thus ``ScaleC=ScaleGate`` for dequantized output and
        ``ScaleC=ScaleGate*SEncC`` for NVFP4/FP8 output.
        """
        outs = []
        one_pair = (cutlass.Float32(1.0), cutlass.Float32(1.0))
        scale_pair = (scale_gate, scale_gate)
        for i in cutlass.range_constexpr(count // 4):
            elem = 4 * i
            gate_pair = (vec_f32[elem], vec_f32[elem + 2])
            act_pair = (vec_f32[elem + 1], vec_f32[elem + 3])
            gate_scaled = cute.arch.mul_packed_f32x2(gate_pair, scale_pair)
            act_scaled = cute.arch.mul_packed_f32x2(act_pair, scale_pair)
            exp_pair = (
                cute.math.exp(-act_scaled[0], fastmath=True),
                cute.math.exp(-act_scaled[1], fastmath=True),
            )
            denom_pair = cute.arch.add_packed_f32x2(exp_pair, one_pair)
            sigmoid_pair = (
                cutlass.Float32(1.0) / denom_pair[0],
                cutlass.Float32(1.0) / denom_pair[1],
            )
            act_silu_without_outer_scale = cute.arch.mul_packed_f32x2(
                act_pair, sigmoid_pair
            )
            outs.extend(
                cute.arch.mul_packed_f32x2(gate_scaled, act_silu_without_outer_scale)
            )
        return cutlass.Vector.from_elements(tuple(outs), cutlass.Float32)

    @cute.jit
    def _sf_c_index_128x4(self, row, sf_col, padded_sf_cols):
        """Index one scale byte in TensorRT-LLM's swizzled 128x4 layout."""
        col_in_group = sf_col % cutlass.Int32(4)
        col_group = sf_col // cutlass.Int32(4)
        row_in_32 = row % cutlass.Int32(32)
        row_32_in_128 = (row % cutlass.Int32(128)) // cutlass.Int32(32)
        row_group = row // cutlass.Int32(128)
        return (
            col_in_group
            + col_group * cutlass.Int32(512)
            + row_in_32 * cutlass.Int32(16)
            + row_32_in_128 * cutlass.Int32(4)
            + row_group * cutlass.Int32(128) * padded_sf_cols
        )

    @cute.jit
    def _store_nvfp4_block(
        self,
        values,
        row,
        output_col,
        num_rows,
        num_cols,
        scale_c: cutlass.Float32,
    ) -> None:
        """Quantize one thread-owned 16-value block to NVFP4."""
        block_absmax = cutlass.Float32(0.0)
        for i in cutlass.range_constexpr(_SF_VEC_SIZE):
            block_absmax = cute.math.max(block_absmax, cute.math.abs(values[i]))

        sf = block_absmax * scale_c * cutlass.Float32(1.0 / _FP4_E2M1_MAX)
        # Encode with the stored E4M3 scale, not the FP32 one, so the payload
        # decodes against the scale a reader sees (as in TensorRT-LLM's
        # cvt_warp_fp16_to_fp4). E4M3 rounding moves the scale up to 6.25%,
        # and a block whose scale saturates at 448 would otherwise decode
        # every non-max element at 448 / sf of its value.
        sf_e4m3 = sf.to(cutlass.Float8E4M3FN)
        sf_for_rcp = cute.math.max(
            sf_e4m3.to(cutlass.Float32), cutlass.Float32(1.0e-12)
        )
        output_scale = scale_c * cute.math.rcp(sf_for_rcp, approx=True, ftz=True)

        packed = []
        for i in cutlass.range_constexpr(_SF_VEC_SIZE // 2):
            lo = values[2 * i] * output_scale
            hi = values[2 * i + 1] * output_scale
            # NVVM maps the second conversion operand to the low nibble.
            packed.append(_convert_f32x2_to_e2m1x2(hi, lo))

        if row < num_rows:
            byte_col = output_col // cutlass.Int32(2)
            byte_stride = num_cols // cutlass.Int32(2)
            byte_idx = cutlass.Int64(row) * cutlass.Int64(byte_stride) + cutlass.Int64(
                byte_col
            )
            packed_vec = cutlass.Vector.from_elements(tuple(packed), cutlass.Int8)
            (self.gC_bytes.data_ptr() + byte_idx).store(packed_vec, alignment=8)

            sf_cols = num_cols // cutlass.Int32(_SF_VEC_SIZE)
            padded_sf_cols = (
                (sf_cols + cutlass.Int32(3)) // cutlass.Int32(4)
            ) * cutlass.Int32(4)
            sf_col = output_col // cutlass.Int32(_SF_VEC_SIZE)
            sf_idx = self._sf_c_index_128x4(row, sf_col, padded_sf_cols)
            sf_packed = sf_e4m3.bitcast(cutlass.Int8)
            self.gSfC_bytes.subview(sf_idx).store(sf_packed)

    @producer_work
    @cute.jit
    def reset_qknorm_accumulator(self, stage_info: StageInfo) -> None:
        del stage_info
        self.norm_sum = cutlass.Float32(0.0)

    @producer_work
    @cute.jit
    def accumulate_qknorm(
        self,
        stage_info: StageInfo,
        *,
        t2r_rmem: cutlass.Float32,
        subtile_idx: cutlass.Constexpr[int],
    ) -> None:
        """First Q/K pass: reproduce the BF16 GEMM boundary and sum squares."""
        self._accumulate_qknorm_impl(stage_info, t2r_rmem, subtile_idx)

    @producer_work
    @cute.jit
    def accumulate_qknorm_prefix(
        self, stage_info: StageInfo, *, t2r_rmem_prefix: cutlass.Float32
    ) -> None:
        self._accumulate_qknorm_impl(stage_info, t2r_rmem_prefix, 0)

    @cute.jit
    def _accumulate_qknorm_impl(
        self,
        stage_info: StageInfo,
        t2r_rmem,
        subtile_idx: cutlass.Constexpr[int],
    ) -> None:
        coordc_m, coordc_n = self._cta_output_mn(stage_info)
        tx, _, _ = cute.arch.thread_idx()
        epilogue_group = tx // 128
        global_subtile_idx = subtile_idx + epilogue_group * (
            mma_tiler_mnk[1] // 2 // self.t2r_inst_repx
        )
        chunk_col = coordc_n + global_subtile_idx * self.t2r_inst_repx
        qkv_hidden = self.mC_mn.shape[1] // 3
        if chunk_col < 2 * qkv_hidden:
            bias_ptr = self.bias.iterator.raw_ptr() if self.bias is not None else None
            for j in cutlass.range_constexpr(self.t2r_inst_repx // self.vsize):
                vec_col = chunk_col + j * self.vsize
                local_col = global_subtile_idx * self.t2r_inst_repx + j * self.vsize
                vec_f32 = t2r_rmem[j * self.vsize : j * self.vsize + self.vsize]
                scaled = self._apply_linear_epilogue_vec(
                    vec_f32, vec_col, local_col, bias_ptr, self.vsize
                )
                # The unfused path writes BF16 from fp8_scaled_mm before the
                # standalone QKNorm+RoPE kernel reads it back.
                rounded = scaled.to(output_dtype).to(cutlass.Float32)
                for i in cutlass.range_constexpr(self.vsize):
                    self.norm_sum += rounded[i] * rounded[i]

    @producer_work
    @cute.jit
    def finish_qknorm_accumulator(self, stage_info: StageInfo) -> None:
        del stage_info
        self.norm_factor = cute.rsqrt(
            self.norm_sum / cutlass.Float32(_QKV_HEAD_DIM)
            + cutlass.Float32(_QKNORM_EPS)
        )

    @cute.jit
    def _apply_qknorm_rope_vec(
        self,
        vec_f32,
        chunk_col,
        count: cutlass.Constexpr[int],
    ):
        """Non-NeoX Q/K RMSNorm + interleaved-pair RoPE."""
        qkv_hidden = self.mC_mn.shape[1] // 3
        result = vec_f32
        if chunk_col < 2 * qkv_hidden:
            rounded = vec_f32.to(output_dtype).to(cutlass.Float32)
            weight_offset = cutlass.Int64(0)
            if chunk_col >= qkv_hidden:
                weight_offset = cutlass.Int64(_QKV_HEAD_DIM)
            head_col = chunk_col % _QKV_HEAD_DIM
            weight = (
                (
                    self.qk_norm_weight_smem.data_ptr()
                    + weight_offset
                    + cutlass.Int64(head_col)
                )
                .load(count=count, alignment=16)
                .to(cutlass.Float32)
            )
            normalized = rounded * self.norm_factor * weight

            cache_ptr = self.cos_sin_cache.iterator.raw_ptr() + (
                cutlass.Int64(self.position_reg) * cutlass.Int64(_QKV_HEAD_DIM)
            )
            cache_idx = head_col // 2
            # Keep the FP32 cache in global/L2: staging the full 128x128 CTA
            # tile consumes 64 KiB, reduces the A/B pipeline depth, and is
            # slower on B200. CTA-scoped evict-last retains each vector slice
            # for the second epilogue warp group without adding barriers. The
            # fused configuration uses four A/B stages to leave more L1 space.
            cos_vec = (cache_ptr + cutlass.Int64(cache_idx)).nvvm_load_ext(
                count=count // 2,
                evict="last",
                scope="cta",
            )
            sin_vec = (
                cache_ptr + cutlass.Int64(_QKV_HEAD_DIM // 2 + cache_idx)
            ).nvvm_load_ext(
                count=count // 2,
                evict="last",
                scope="cta",
            )
            rotated = []
            for i in cutlass.range_constexpr(count // 2):
                elem = 2 * i
                x = normalized[elem]
                y = normalized[elem + 1]
                rotated.extend(
                    (
                        x * cos_vec[i] - y * sin_vec[i],
                        y * cos_vec[i] + x * sin_vec[i],
                    )
                )
            result = cutlass.Vector.from_elements(tuple(rotated), cutlass.Float32)
        return result

    # Pure sink — producer-side t2r_rmem slot is auto-allocated by
    # Task.init_variables from upstream TmemCResource.

    @cute.jit
    def _store_impl(
        self,
        stage_info: StageInfo,
        *,
        t2r_rmem: cutlass.Float32,
        subtile_idx: cutlass.Constexpr[int],
    ) -> None:
        coordc_m, coordc_n = self._cta_output_mn(stage_info)
        tx, _, _ = cute.arch.thread_idx()
        if cutlass.const_expr(num_epilogue_warps == 8):
            epilogue_group = tx // 128
            row_in_tile = tx % 128
            row = coordc_m + row_in_tile
            global_subtile_idx = subtile_idx + epilogue_group * (
                mma_tiler_mnk[1] // 2 // self.t2r_inst_repx
            )
        else:
            epilogue_group = 0
            row_in_tile = tx
            row = coordc_m + tx
            global_subtile_idx = subtile_idx
        col = coordc_n + global_subtile_idx * self.t2r_inst_repx
        num_rows = self.mC_mn.shape[0]
        num_cols = self.mC_mn.shape[1]
        gC_ptr = self.mC_mn.iterator.raw_ptr()
        row_offset = cutlass.Int64(row) * cutlass.Int64(num_cols)
        if cutlass.const_expr(separate_qkv_output):
            group_n = num_cols // 3
            group = col // group_n
            row_offset = (
                cutlass.Int64(group) * cutlass.Int64(num_rows * group_n)
                + cutlass.Int64(row) * cutlass.Int64(group_n)
                - cutlass.Int64(group * group_n)
            )
        bias_ptr = self.bias.iterator.raw_ptr() if self.bias is not None else None
        scale_gate_value = cutlass.Float32(self.scale)
        scale_c_value = cutlass.Float32(self.scale)
        if cutlass.const_expr(self.scale_gate is not None):
            _, _, batch_idx = stage_info.work_tile.tile_idx
            scale_gate_value = (
                self.scale_gate.iterator.raw_ptr() + cutlass.Int64(batch_idx)
            ).load()
            scale_c_value = (
                self.scale_c.iterator.raw_ptr() + cutlass.Int64(batch_idx)
            ).load()
        if cutlass.const_expr(use_gated_activation):
            gemm_n = num_cols * 2
            out_vsize = self.vsize // 2
            output_subtile_n = 16
            output_col = col // 2
        else:
            gemm_n = num_cols
            out_vsize = self.vsize
            output_subtile_n = self.t2r_inst_repx
            output_col = col

        stage_idx = subtile_idx % _epilogue_store_stages
        stage_elems = mma_tiler_mnk_per_cta[0] * output_subtile_n
        stage_offset = (
            epilogue_group * _epilogue_store_stages + stage_idx
        ) * stage_elems
        if cutlass.const_expr(use_tma_store):
            smem_base = stage_info.context.smem_base
            self.output_smem = cutlass.Array(
                smem_base.data_ptr() + self._output_alloc.offset,
                dtype=output_dtype,
                shape=(
                    mma_tiler_mnk_per_cta[0]
                    * output_subtile_n
                    * _epilogue_store_stages
                    * (num_epilogue_warps // 4),
                ),
                addrspace=3,
            )

            # Keep two TMA-store groups in flight. Before a stage is reused,
            # wait until only the newer group remains, then release all
            # epilogue threads to overwrite the selected ping-pong stage.
            if cutlass.const_expr(subtile_idx >= _epilogue_store_stages):
                if row_in_tile == 0:
                    cute.arch.cp_async_bulk_wait_group(1, read=True)
                cute.arch.barrier(
                    barrier_id=_epilogue_store_barrier_id,
                    number_of_threads=threads_in_epilogue,
                )

        smem_row_offset = cutlass.Int64(stage_offset + row_in_tile * output_subtile_n)
        for j in cutlass.range_constexpr(self.t2r_inst_repx // self.vsize):
            chunk_col = col + j * self.vsize
            local_col = global_subtile_idx * self.t2r_inst_repx + j * self.vsize
            vec_f32 = t2r_rmem[j * self.vsize : j * self.vsize + self.vsize]

            if chunk_col + self.vsize <= gemm_n:
                # Fresh names: the gated vector is half as wide, and a
                # value carried out of this branch must keep one shape.
                scaled_vec = self._apply_linear_epilogue_vec(
                    vec_f32, chunk_col, local_col, bias_ptr, self.vsize
                )
                if cutlass.const_expr(use_fused_qknorm_rope):
                    scaled_vec = self._apply_qknorm_rope_vec(
                        scaled_vec, chunk_col, self.vsize
                    )
                if cutlass.const_expr(use_gated_activation):
                    if cutlass.const_expr(self.scale_gate is not None):
                        out_vec = self._gated_pair_silu_quant(
                            scaled_vec, scale_gate_value, self.vsize
                        )
                    else:
                        out_vec = self._gated_pair_silu(scaled_vec, self.vsize)
                else:
                    out_vec = scaled_vec
                if cutlass.const_expr(
                    self.scale_gate is not None and not _is_fp4(output_dtype)
                ):
                    out_vec = out_vec * cutlass.vector.full_like(out_vec, scale_c_value)
                output_local_col = j * out_vsize
                if cutlass.const_expr(_is_fp4(output_dtype)):
                    self._store_nvfp4_block(
                        out_vec,
                        row,
                        output_col + output_local_col,
                        num_rows,
                        num_cols,
                        scale_c_value,
                    )
                elif cutlass.const_expr(use_tma_store):
                    (
                        self.output_smem.data_ptr()
                        + smem_row_offset
                        + cutlass.Int64(output_local_col)
                    ).store(out_vec.to(output_dtype), alignment=16)
                else:
                    if row < num_rows:
                        (
                            gC_ptr
                            + row_offset
                            + cutlass.Int64(output_col + output_local_col)
                        ).store(out_vec.to(output_dtype), alignment=16)
            else:
                # Scalar predication for a partial N tile. Elements outside
                # the tensor are ignored by TMA's destination bounds check.
                if cutlass.const_expr(use_gated_activation):
                    # FP4 output is validated to whole 16-value scale blocks,
                    # so it always takes the vector path above.
                    if cutlass.const_expr(not _is_fp4(output_dtype)):
                        for i in cutlass.range_constexpr(out_vsize):
                            gate_col = chunk_col + 2 * i
                            act_col = gate_col + 1
                            if act_col < gemm_n:
                                gate = self._apply_linear_epilogue_scalar(
                                    vec_f32[2 * i],
                                    gate_col,
                                    local_col + 2 * i,
                                    bias_ptr,
                                )
                                act = self._apply_linear_epilogue_scalar(
                                    vec_f32[2 * i + 1],
                                    act_col,
                                    local_col + 2 * i + 1,
                                    bias_ptr,
                                )
                                output_local_col = j * out_vsize + i
                                if cutlass.const_expr(use_tma_store):
                                    (
                                        self.output_smem.data_ptr()
                                        + smem_row_offset
                                        + cutlass.Int64(output_local_col)
                                    ).store((gate * self._silu(act)).to(output_dtype))
                                else:
                                    if row < num_rows:
                                        (
                                            gC_ptr
                                            + row_offset
                                            + cutlass.Int64(
                                                output_col + output_local_col
                                            )
                                        ).store(
                                            (gate * self._silu(act)).to(output_dtype)
                                        )
                else:
                    for i in cutlass.range_constexpr(self.vsize):
                        col_idx = chunk_col + i
                        if col_idx < num_cols:
                            value = self._apply_linear_epilogue_scalar(
                                vec_f32[i], col_idx, local_col + i, bias_ptr
                            )
                            output_local_col = j * self.vsize + i
                            if cutlass.const_expr(use_tma_store):
                                (
                                    self.output_smem.data_ptr()
                                    + smem_row_offset
                                    + cutlass.Int64(output_local_col)
                                ).store(value.to(output_dtype))
                            else:
                                if row < num_rows:
                                    (
                                        gC_ptr
                                        + row_offset
                                        + cutlass.Int64(output_col + output_local_col)
                                    ).store(value.to(output_dtype))

        if cutlass.const_expr(use_tma_store):
            cute.arch.barrier(
                barrier_id=_epilogue_store_barrier_id,
                number_of_threads=threads_in_epilogue,
            )
            prims.fence_proxy_async_release_sync_restrict()
            if row_in_tile == 0:
                prims.cp_async_bulk_tensor_global_shared_cta(
                    self.tma_c_desc,
                    self.output_smem.data_ptr() + cutlass.Int64(stage_offset),
                    [output_col, coordc_m],
                )
                cute.arch.cp_async_bulk_commit_group()
                if cutlass.const_expr(
                    subtile_idx
                    == self.num_output_subtiles // (num_epilogue_warps // 4) - 1
                ):
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
            if cutlass.const_expr(
                subtile_idx == self.num_output_subtiles // (num_epilogue_warps // 4) - 1
            ):
                # Do not let another persistent work tile reuse either stage
                # until the issuing thread has drained the final bulk group.
                cute.arch.barrier(
                    barrier_id=_epilogue_store_barrier_id,
                    number_of_threads=threads_in_epilogue,
                )

    # ──────────────────────────────────────────────────────────────────────
    @producer_work
    @cute.jit
    def store(
        self,
        stage_info: StageInfo,
        *,
        t2r_rmem: cutlass.Float32,
        subtile_idx: cutlass.Constexpr[int],
    ) -> None:
        self._store_impl(
            stage_info,
            t2r_rmem=t2r_rmem,
            subtile_idx=subtile_idx,
        )

    @producer_work
    @cute.jit
    def store_overlap_prefix(
        self, stage_info: StageInfo, *, t2r_rmem_prefix: cutlass.Float32
    ) -> None:
        # A distinct input slot keeps this prefetched value independent
        # from the ordinary subtile route in the TaskManager data flow.
        self._store_impl(stage_info, t2r_rmem=t2r_rmem_prefix, subtile_idx=0)


# Resource construction helpers
# ──────────────────────────────────────────────────────────────────────


@cute.jit
def create_gmem_ab_resource() -> GmemAbResource:
    """
    Create the global memory A/B input resource.
    """
    return GmemAbResource(name="GmemAb")


def create_smem_ab_resource(
    tma_a_desc: cutlass.Pointer,
    tma_b_desc: cutlass.Pointer,
    operand: str,
    cluster_shape_vmnk: cute.Layout,
    act_num_pairs: int = None,
    act_num_pair_cols: int = None,
    act_a_mcast_template: int = None,
    act_b_mcast_template: int = None,
) -> SmemAbResource:
    """
    Create a shared memory A or B resource with TMA+UMMA async pipeline.
    """
    if act_num_pairs is None:
        act_num_pairs = num_pairs
    sA_copy_bytes = _span_bytes(
        mma_tiler_mnk_per_cta[0] * mma_tiler_mnk[2], input_dtype
    )
    sB_copy_bytes = _span_bytes(
        mma_tiler_mnk_per_cta[1] * mma_tiler_mnk[2], input_dtype
    )
    if operand == "a":
        tma_copy_bytes_per_cta = sA_copy_bytes
    elif operand == "b":
        tma_copy_bytes_per_cta = sB_copy_bytes
    else:
        raise ValueError(f"operand must be 'a' or 'b', got {operand}")

    # TmaUmma full barriers are armed per UMMA CTA group. Even when the
    # cluster has several independent pairs, each pair leader only receives
    # the V-group's TMA transaction bytes for its own full barrier.
    num_tma_copy_bytes = tma_copy_bytes_per_cta * num_mma_ctas

    num_umma_consumers = act_num_pairs
    smem_ab_pipeline_config = PipelineConfig.create_tma_umma_pipeline_cfg(
        num_stages=ab_stages,
        num_bytes=num_tma_copy_bytes,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_umma_consumers
        ),
        cta_layout_vmnk=cluster_shape_vmnk,
        consumer_signaling_threads=SignalingThreads.CtaLeader,
        num_bytes_per_warp_per_cta=tma_copy_bytes_per_cta,
        advance_on_wait=use_nvfp4_split_k,
    )
    return SmemAbResource(
        tma_desc_a=tma_a_desc,
        tma_desc_b=tma_b_desc,
        operand=operand,
        act_num_pair_cols=act_num_pair_cols,
        act_a_mcast_template=act_a_mcast_template,
        act_b_mcast_template=act_b_mcast_template,
        pipeline_config=smem_ab_pipeline_config,
        name=f"Smem{operand.upper()}",
    )


@cute.jit
def create_tmem_c_resource(
    num_epilogue_warps: int, cluster_shape_vmnk: cute.Layout
) -> TmemCResource:
    """
    Create the TMEM accumulator resource with UMMA async pipeline.
    """
    # Accumulator is pair-scoped (2-CTA): only the 2 CTAs in the MMA group
    # participate, not the full cluster.
    tmem_c_pipeline_consumer_group = pipeline.CooperativeGroup(
        pipeline.Agent.Thread, size=num_epilogue_warps * 32 * num_mma_ctas
    )
    tmem_c_pipeline_config = PipelineConfig.create_umma_async_pipeline_cfg(
        num_stages=1 if use_nvfp4_tmem_overlap else acc_stages,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=tmem_c_pipeline_consumer_group,
        cta_layout_vmnk=cluster_shape_vmnk,
        producer_signaling_threads=SignalingThreads.CtaLeader,
    )
    return TmemCResource(
        pipeline_config=tmem_c_pipeline_config,
        name="TmemC",
    )


@cute.jit
def create_gmem_d_resource(
    mC_mn: cute.Tensor,
    tma_c_desc: cutlass.Pointer,
    bias: Optional[cute.Tensor] = None,
    scale: cutlass.Float32 = 1.0,
    x_scale: Optional[cute.Tensor] = None,
    weight_scale: Optional[cute.Tensor] = None,
    sf_c: Optional[cute.Tensor] = None,
    scale_c: Optional[cute.Tensor] = None,
    scale_gate: Optional[cute.Tensor] = None,
    qkv_scale: Optional[cute.Tensor] = None,
    q_norm_weight: Optional[cute.Tensor] = None,
    k_norm_weight: Optional[cute.Tensor] = None,
    cos_sin_cache: Optional[cute.Tensor] = None,
    positions: Optional[cute.Tensor] = None,
) -> GmemDResource:
    """
    Create the global memory D (output) resource.
    """
    return GmemDResource(
        mC_mn=mC_mn,
        tma_c_desc=tma_c_desc,
        bias=bias,
        scale=scale,
        x_scale=x_scale,
        weight_scale=weight_scale,
        sf_c=sf_c,
        scale_c=scale_c,
        scale_gate=scale_gate,
        qkv_scale=qkv_scale,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        cos_sin_cache=cos_sin_cache,
        positions=positions,
        name="GmemD",
    )


@cute.jit
def create_work_queue(
    tile_sched_params: object,
    cluster_shape_vmnk: cute.Layout,
    num_load_warps: int,
    num_epilogue_warps: int,
    num_mma_warps: int,
    num_padding_warps: int,
    num_scheduler_warps: int = 0,
    clc_response_ptr: Optional[cute.Pointer] = None,
) -> WorkQueue:
    """
    Create the work queue for static or CLC-dynamic scheduling.
    """
    if cutlass.const_expr(use_clc_dynamic_scheduler):
        cluster_size = (
            cluster_shape_vmnk[0]
            * cluster_shape_vmnk[1]
            * cluster_shape_vmnk[2]
            * cluster_shape_vmnk[3]
        )
        # All consumer tasks (load, mma, store, scheduler) run on every CTA
        # in the cluster and call consumer_release, so every warp across all
        # CTAs must be counted in the arrive count.
        num_clc_consumer_threads = (
            32
            * cluster_size
            * (
                num_load_warps
                + num_epilogue_warps
                + num_mma_warps
                + num_padding_warps
                + num_scheduler_warps
            )
        )
        num_clc_response_bytes = 16
        scheduler_pipeline_config = PipelineConfig.create_clc_fetch_async_pipeline_cfg(
            num_stages=num_scheduler_stages,
            num_bytes=num_clc_response_bytes,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, num_clc_consumer_threads
            ),
            cta_layout_vmnk=cluster_shape_vmnk,
            # Only CTA 0 runs the producer side (CLC is cluster-wide)
            producer_signaling_threads=SignalingThreads.CtaLeader,
            consumer_signaling_threads=SignalingThreads.All,
        )
        tile_scheduler_config = (
            TileSchedulerConfig.create_clc_dynamic_persistent_tile_scheduler_params(
                tile_scheduler_params=tile_sched_params,
                response_ptr=clc_response_ptr,
            )
        )
        return WorkQueue(
            tile_scheduler_config=tile_scheduler_config,
            pipeline_config=scheduler_pipeline_config,
            name="WorkQueue",
        )
    else:
        # Static persistent tile scheduler - no pipeline, no scheduler warp
        tile_scheduler_config = (
            TileSchedulerConfig.create_static_persistent_tile_scheduler_params(
                tile_scheduler_params=tile_sched_params,
            )
        )
        return WorkQueue(
            tile_scheduler_config=tile_scheduler_config,
            name="WorkQueue",
        )


########################################################
# Task schedule construction helpers
########################################################


@cute.jit
def create_load_a_task(
    gmem_ab_resource: GmemAbResource,
    smem_a_resource: SmemAbResource,
    pdl_wait: PdlWaitBarrier,
    pdl_launch: PdlLaunchBarrier,
    work_queue: WorkQueue,
    num_k_tiles: int,
    smem_sf_resource: SmemSfResource = None,
) -> Task:
    """
    Create the A TMA load task. NVFP4 also loads SFA on this warp.
    """

    if cutlass.const_expr(use_nvfp4_block_scale):

        @schedule
        def load_a_schedule(
            gmem_ab: GmemAbResource,
            smem_a: SmemAbResource,
            smem_sf: SmemSfResource,
            pdl_wait_resource: PdlWaitBarrier,
            pdl_launch_resource: PdlLaunchBarrier,
            wq: WorkQueue,
        ) -> None:
            pdl_wait_resource.wait_griddep()
            smem_a.init_load_state()
            smem_sf.init_load_state()
            with work_tile_loop(wq):
                gmem_ab.init_tile_coords()
                with domain_loop(0, num_k_tiles, 1):
                    if cutlass.const_expr(not use_nvfp4_split_k):
                        coord_k, coord_m, coord_n = gmem_ab.compute_coords()
                        smem_a.try_acquire()
                        smem_sf.try_acquire()
                        smem_a.acquire()
                        smem_sf.acquire()
                        smem_a.tma_load_a(coord_k=coord_k, coord_m=coord_m)
                        smem_sf.tma_load(coord_k=coord_k, coord_mn=coord_m)
                        smem_a.commit()
                        smem_sf.commit()
                    for group in cutlass.range_constexpr(3 if use_nvfp4_split_k else 0):
                        coord_k, coord_m, coord_n = gmem_ab.compute_coords(
                            k_offset=group * 256
                        )
                        smem_a.try_acquire()
                        smem_a.acquire()
                        smem_a.tma_load_a(coord_k=coord_k, coord_m=coord_m)
                        smem_a.commit()
                        if cutlass.const_expr(group != 1):
                            coord_k, coord_m, coord_n = gmem_ab.compute_coords(
                                k_offset=(group // 2) * sf_tile_k
                            )
                            smem_sf.try_acquire()
                            smem_sf.acquire()
                            smem_sf.tma_load(coord_k=coord_k, coord_mn=coord_m)
                            smem_sf.commit()
                wq.try_wait()
                wq.wait()
                wq.get_and_advance_work_tile()
                wq.release()
            pdl_launch_resource.launch_griddep()

        result = load_a_schedule(
            gmem_ab_resource,
            smem_a_resource,
            smem_sf_resource,
            pdl_wait,
            pdl_launch,
            work_queue,
        )
        dst_resources = [smem_a_resource, smem_sf_resource, pdl_launch]
    else:

        @schedule
        def load_a_schedule(
            gmem_ab: GmemAbResource,
            smem_a: SmemAbResource,
            pdl_wait_resource: PdlWaitBarrier,
            pdl_launch_resource: PdlLaunchBarrier,
            wq: WorkQueue,
        ) -> None:
            # PDL wait gates the A-load stream before any persistent work is issued.
            pdl_wait_resource.wait_griddep()
            smem_a.init_load_state()
            with work_tile_loop(wq):
                gmem_ab.init_tile_coords()
                with domain_loop(0, num_k_tiles, 1):
                    coord_k, coord_m, coord_n = gmem_ab.compute_coords()
                    # Producer side of SmemA: reserve empty stage, TMA-fill it,
                    # then commit full.
                    smem_a.try_acquire()
                    smem_a.acquire()
                    smem_a.tma_load_a(coord_k=coord_k, coord_m=coord_m)
                    smem_a.commit()
                # TAIL: advance to next work tile
                wq.try_wait()
                wq.wait()
                wq.get_and_advance_work_tile()
                wq.release()
            # PDL launch is emitted only after all persistent A-load work completes.
            pdl_launch_resource.launch_griddep()

        result = load_a_schedule(
            gmem_ab_resource, smem_a_resource, pdl_wait, pdl_launch, work_queue
        )
        dst_resources = [smem_a_resource, pdl_launch]
    return Task(
        src_resources=[gmem_ab_resource, pdl_wait, work_queue],
        dst_resources=dst_resources,
        warp_idx=num_epilogue_warps,
        num_warps=1,
        schedule=result,
        num_registers=40,
        name="LoadATask",
        debug_print=debug_print,
    )


@cute.jit
def create_load_b_task(
    gmem_ab_resource: GmemAbResource,
    smem_b_resource: SmemAbResource,
    work_queue: WorkQueue,
    num_k_tiles: int,
    smem_sf_resource: SmemSfResource = None,
) -> Task:
    """
    Create the B TMA load task. NVFP4 also loads SFB on this warp.
    """

    if cutlass.const_expr(use_nvfp4_block_scale):

        @schedule
        def load_b_schedule(
            gmem_ab: GmemAbResource,
            smem_b: SmemAbResource,
            smem_sf: SmemSfResource,
            wq: WorkQueue,
        ) -> None:
            smem_b.init_load_state()
            smem_sf.init_load_state()
            with work_tile_loop(wq):
                gmem_ab.init_tile_coords()
                with domain_loop(0, num_k_tiles, 1):
                    if cutlass.const_expr(not use_nvfp4_split_k):
                        coord_k, coord_m, coord_n = gmem_ab.compute_coords()
                        smem_b.try_acquire()
                        smem_sf.try_acquire()
                        smem_b.acquire()
                        smem_sf.acquire()
                        smem_b.tma_load_b(coord_k=coord_k, coord_n=coord_n)
                        smem_sf.tma_load(coord_k=coord_k, coord_mn=coord_n)
                        smem_b.commit()
                        smem_sf.commit()
                    for group in cutlass.range_constexpr(3 if use_nvfp4_split_k else 0):
                        coord_k, coord_m, coord_n = gmem_ab.compute_coords(
                            k_offset=group * 256
                        )
                        smem_b.try_acquire()
                        smem_b.acquire()
                        smem_b.tma_load_b(coord_k=coord_k, coord_n=coord_n)
                        smem_b.commit()
                        if cutlass.const_expr(group != 1):
                            coord_k, coord_m, coord_n = gmem_ab.compute_coords(
                                k_offset=(group // 2) * sf_tile_k
                            )
                            smem_sf.try_acquire()
                            smem_sf.acquire()
                            smem_sf.tma_load(coord_k=coord_k, coord_mn=coord_n)
                            smem_sf.commit()
                wq.try_wait()
                wq.wait()
                wq.get_and_advance_work_tile()
                wq.release()

        result = load_b_schedule(
            gmem_ab_resource, smem_b_resource, smem_sf_resource, work_queue
        )
        dst_resources = [smem_b_resource, smem_sf_resource]
    else:

        @schedule
        def load_b_schedule(
            gmem_ab: GmemAbResource,
            smem_b: SmemAbResource,
            wq: WorkQueue,
        ) -> None:
            # B-load schedule has the same coordinate flow as A but no PDL dependency.
            smem_b.init_load_state()
            with work_tile_loop(wq):
                gmem_ab.init_tile_coords()
                with domain_loop(0, num_k_tiles, 1):
                    coord_k, coord_m, coord_n = gmem_ab.compute_coords()
                    # Producer side of SmemB: reserve empty stage, TMA-fill it,
                    # then commit full.
                    smem_b.try_acquire()
                    smem_b.acquire()
                    smem_b.tma_load_b(coord_k=coord_k, coord_n=coord_n)
                    smem_b.commit()
                # TAIL: advance to next work tile
                wq.try_wait()
                wq.wait()
                wq.get_and_advance_work_tile()
                wq.release()

        result = load_b_schedule(gmem_ab_resource, smem_b_resource, work_queue)
        dst_resources = [smem_b_resource]
    return Task(
        src_resources=[gmem_ab_resource, work_queue],
        dst_resources=dst_resources,
        warp_idx=num_epilogue_warps + 1,
        num_warps=1,
        schedule=result,
        num_registers=40,
        name="LoadBTask",
        debug_print=debug_print,
    )


@cute.jit
def create_padding_task(
    work_queue: WorkQueue,
    num_padding_warps: int,
    total_num_warps_so_far: int,
    num_k_tiles: int,
) -> Task:
    """
    Create the padding task that does nothing.
    """

    @schedule
    def padding_schedule(wq: WorkQueue) -> None:
        # Padding warps only participate in WorkQueue synchronization.
        with work_tile_loop(wq):
            with domain_loop(0, num_k_tiles, 1):
                pass  # padding warps do no loop work
            # TAIL: advance to next work tile
            wq.try_wait()
            wq.wait()
            wq.get_and_advance_work_tile()
            wq.release()

    result = padding_schedule(work_queue)
    return Task(
        src_resources=[work_queue],
        dst_resources=[],
        warp_idx=total_num_warps_so_far,
        num_warps=num_padding_warps,
        schedule=result,
        num_registers=40,
        name="PaddingTask",
        debug_print=debug_print,
    )


@cute.jit
def create_mma_task(
    smem_a_resource: SmemAbResource,
    smem_b_resource: SmemAbResource,
    tmem_c_resource: TmemCResource,
    work_queue: WorkQueue,
    num_k_tiles: int,
    num_mma_warps: int,
    num_load_warps: int = 2,
    smem_sfa_resource: SmemSfResource = None,
    smem_sfb_resource: SmemSfResource = None,
) -> Task:
    """
    Create the MMA compute task. NVFP4 copies block scales into TMEM here.
    """

    if cutlass.const_expr(use_nvfp4_block_scale):

        @schedule
        def mma_schedule(
            smem_a: SmemAbResource,
            smem_b: SmemAbResource,
            smem_sfa: SmemSfResource,
            smem_sfb: SmemSfResource,
            tmem_c: TmemCResource,
            wq: WorkQueue,
        ) -> None:
            smem_a.init_descriptors()
            smem_b.init_descriptors()
            smem_sfa.init_descriptors()
            smem_sfb.init_descriptors()
            tmem_c.init_accumulator_state()
            with work_tile_loop(wq):
                tmem_c.init_work_tile_state()
                tmem_c.try_acquire()
                tmem_c.acquire()
                with domain_loop(0, num_k_tiles, 1):
                    if cutlass.const_expr(not use_nvfp4_split_k):
                        smem_a.try_wait()
                        smem_b.try_wait()
                        smem_sfa.try_wait()
                        smem_sfb.try_wait()
                        smem_a.wait()
                        smem_b.wait()
                        smem_sfa.wait()
                        smem_sfb.wait()
                        desc_a_base = smem_a.build_desc_a()
                        desc_b_base = smem_b.build_desc_b()
                        desc_sfa_base = smem_sfa.build_s2t_descriptor()
                        desc_sfb_base = smem_sfb.build_s2t_descriptor()
                        tmem_c.mma_block_scaled(
                            desc_a_base=desc_a_base,
                            desc_b_base=desc_b_base,
                            desc_sfa_base=desc_sfa_base,
                            desc_sfb_base=desc_sfb_base,
                        )
                        smem_a.release()
                        smem_b.release()
                        smem_sfa.release()
                        smem_sfb.release()
                    # Four MMAs per scale group amortize scale-pipeline
                    # bookkeeping. The first group spans AB0/AB1; the
                    # second spans AB1/AB2. AB1 stays live across groups.
                    if cutlass.const_expr(use_nvfp4_split_k):
                        smem_a.try_wait()
                        smem_b.try_wait()
                        smem_a.wait()
                        smem_b.wait()
                        smem_a.remember_stage_3x()
                        smem_b.remember_stage_3x()
                    for group in cutlass.range_constexpr(2 if use_nvfp4_split_k else 0):
                        smem_a.try_wait()
                        smem_b.try_wait()
                        smem_a.wait()
                        smem_b.wait()
                        desc_a_previous, desc_a_base = smem_a.build_desc_3x()
                        desc_b_previous, desc_b_base = smem_b.build_desc_3x()
                        smem_sfa.try_wait()
                        smem_sfb.try_wait()
                        smem_sfa.wait()
                        smem_sfb.wait()
                        desc_sfa_base = smem_sfa.build_s2t_descriptor()
                        desc_sfb_base = smem_sfb.build_s2t_descriptor()
                        tmem_c.mma_block_scaled_3x(
                            desc_a_previous=desc_a_previous,
                            desc_a_base=desc_a_base,
                            desc_b_previous=desc_b_previous,
                            desc_b_base=desc_b_base,
                            desc_sfa_base=desc_sfa_base,
                            desc_sfb_base=desc_sfb_base,
                            sf_group=group,
                        )
                        smem_sfa.release()
                        smem_sfb.release()
                        smem_a.release()
                        smem_b.release()
                        if cutlass.const_expr(group > 0):
                            smem_a.release()
                            smem_b.release()
                tmem_c.commit()
                if cutlass.const_expr(use_nvfp4_tmem_overlap):
                    tmem_c.advance_mma_overlap_window()
                wq.try_wait()
                wq.wait()
                wq.get_and_advance_work_tile()
                wq.release()

        result = mma_schedule(
            smem_a_resource,
            smem_b_resource,
            smem_sfa_resource,
            smem_sfb_resource,
            tmem_c_resource,
            work_queue,
        )
        src_resources = [
            smem_a_resource,
            smem_b_resource,
            smem_sfa_resource,
            smem_sfb_resource,
            work_queue,
        ]
    else:

        @schedule
        def mma_schedule(
            smem_a: SmemAbResource,
            smem_b: SmemAbResource,
            tmem_c: TmemCResource,
            wq: WorkQueue,
        ) -> None:
            smem_a.init_descriptors()
            smem_b.init_descriptors()
            tmem_c.init_accumulator_state()
            with work_tile_loop(wq):
                tmem_c.init_work_tile_state()
                # HEAD: acquire TMEM accumulator stage
                tmem_c.try_acquire()
                tmem_c.acquire()
                with domain_loop(0, num_k_tiles, 1):
                    # Consumer side of SmemA/B: wait for TMA-full stages before MMA.
                    smem_a.try_wait()
                    smem_b.try_wait()
                    smem_a.wait()
                    smem_b.wait()
                    desc_a_base = smem_a.build_desc_a()
                    desc_b_base = smem_b.build_desc_b()
                    tmem_c.mma(desc_a_base=desc_a_base, desc_b_base=desc_b_base)
                    # Release both SMEM stages after descriptors have been consumed.
                    smem_a.release()
                    smem_b.release()
                # TAIL: commit TMEM, advance to next work tile
                tmem_c.commit()
                wq.try_wait()
                wq.wait()
                wq.get_and_advance_work_tile()
                wq.release()

        result = mma_schedule(
            smem_a_resource, smem_b_resource, tmem_c_resource, work_queue
        )
        src_resources = [smem_a_resource, smem_b_resource, work_queue]
    return Task(
        src_resources=src_resources,
        dst_resources=[tmem_c_resource],
        warp_idx=num_epilogue_warps + num_load_warps,
        num_warps=num_mma_warps,
        schedule=result,
        num_registers=40,
        name="MmaTask",
        debug_print=debug_print,
    )


@cute.jit
def create_store_task(
    tmem_c_resource: TmemCResource,
    gmem_d_resource: GmemDResource,
    work_queue: WorkQueue,
    num_k_tiles: int,
    num_epilogue_warps: int,
    pdl_wait: Optional[PdlWaitBarrier] = None,
) -> Task:
    """
    Create the epilogue store task.

    ``pdl_wait`` is the same barrier LoadATask waits on. It is passed into
    the schedule only for the per-token/per-channel epilogue, which reads
    ``x_scale``/``weight_scale`` from global memory before the first
    accumulator wait and cannot inherit the grid dependency through
    TMA -> MMA -> TMEM. ``@schedule`` rejects ``None``, so the unused
    barrier is omitted from the call rather than passed as a null.
    """
    subtile_cnt = mma_tiler_mnk[1] // (num_epilogue_warps // 4) // _EPILOGUE_TILE_N
    overlap_prefix_subtiles = (
        max(tmem_sfa_cols_per_stage, tmem_sfb_cols_per_stage) + _EPILOGUE_TILE_N - 1
    ) // _EPILOGUE_TILE_N

    @schedule
    def store_schedule(
        tmem_c: TmemCResource,
        gmem_d: GmemDResource,
        wq: WorkQueue,
        *optional_resources,
    ) -> None:
        # t2r_rmem flows from TmemCResource.load_subtile() into GmemDResource.store().
        if cutlass.const_expr(use_per_token_channel_scale):
            pdl_wait_resource = optional_resources[0]
            # Gate the scale loads on the producing kernel under PDL.
            pdl_wait_resource.wait_griddep()
        tmem_c.init_store_state()
        if cutlass.const_expr(use_fused_qknorm_rope):
            gmem_d.init_qknorm_weights()
        with work_tile_loop(wq):
            with domain_loop(0, num_k_tiles, 1):
                pass  # no loop-body work for the store task
            # TAIL: drain TMEM subtiles and store to global memory
            if cutlass.const_expr(use_per_token_channel_scale):
                gmem_d.preload_rowcol_scales()
            if cutlass.const_expr(use_fused_qknorm_rope):
                gmem_d.preload_qknorm_rope()
            tmem_c.try_wait()
            tmem_c.wait()
            if cutlass.const_expr(use_fused_qknorm_rope and use_nvfp4_tmem_overlap):
                # Sum squares over the whole head, keeping subtile 0 in
                # registers. Next-tile scales overwrite only subtile 0's
                # columns, so release before the store pass and store subtile 0
                # from registers.
                gmem_d.reset_qknorm_accumulator()
                t2r_prefix = tmem_c.load_prefix_subtile()
                gmem_d.accumulate_qknorm_prefix(t2r_rmem_prefix=t2r_prefix)
                for subtile_idx in cutlass.range_constexpr(1, subtile_cnt):
                    t2r_rmem = tmem_c.load_subtile(subtile_idx=subtile_idx)
                    gmem_d.accumulate_qknorm(t2r_rmem=t2r_rmem, subtile_idx=subtile_idx)
                gmem_d.finish_qknorm_accumulator()
                tmem_c.release()
                gmem_d.store_overlap_prefix(t2r_rmem_prefix=t2r_prefix)
                for subtile_idx in cutlass.range_constexpr(1, subtile_cnt):
                    t2r_rmem = tmem_c.load_overlap_subtile(subtile_idx=subtile_idx)
                    gmem_d.store(t2r_rmem=t2r_rmem, subtile_idx=subtile_idx)
            elif cutlass.const_expr(use_fused_qknorm_rope):
                gmem_d.reset_qknorm_accumulator()
                for subtile_idx in cutlass.range_constexpr(subtile_cnt):
                    t2r_rmem = tmem_c.load_subtile(subtile_idx=subtile_idx)
                    gmem_d.accumulate_qknorm(t2r_rmem=t2r_rmem, subtile_idx=subtile_idx)
                gmem_d.finish_qknorm_accumulator()
            if cutlass.const_expr(
                not (use_fused_qknorm_rope and use_nvfp4_tmem_overlap)
            ):
                if cutlass.const_expr(use_nvfp4_tmem_overlap):
                    # Retire the entire next-phase scale footprint before
                    # allowing scale copies to overwrite this window. SF384
                    # needs two subtiles per half, rather than SF256's one.
                    if cutlass.const_expr(overlap_prefix_subtiles == 2):
                        t2r_prefix, t2r_rmem = tmem_c.load_overlap_prefix()
                        tmem_c.release()
                        gmem_d.store_overlap_prefix(t2r_rmem_prefix=t2r_prefix)
                        gmem_d.store(t2r_rmem=t2r_rmem, subtile_idx=1)
                    else:
                        t2r_rmem = tmem_c.load_subtile(subtile_idx=0)
                        tmem_c.release()
                        gmem_d.store(t2r_rmem=t2r_rmem, subtile_idx=0)
                    for subtile_idx in cutlass.range_constexpr(
                        overlap_prefix_subtiles, subtile_cnt
                    ):
                        t2r_rmem = tmem_c.load_overlap_subtile(subtile_idx=subtile_idx)
                        gmem_d.store(t2r_rmem=t2r_rmem, subtile_idx=subtile_idx)
                else:
                    for subtile_idx in cutlass.range_constexpr(subtile_cnt):
                        t2r_rmem = tmem_c.load_subtile(subtile_idx=subtile_idx)
                        gmem_d.store(t2r_rmem=t2r_rmem, subtile_idx=subtile_idx)
                    # Release TMEM only after all output subtiles have been stored.
                    tmem_c.release()
            wq.try_wait()
            wq.wait()
            wq.get_and_advance_work_tile()
            wq.release()

    if cutlass.const_expr(use_per_token_channel_scale):
        # CuTe DSL 4.9 rejects mutating a Python list while tracing this
        # staged branch. Keep the source schedule and resources identical
        # while constructing the two constexpr variants directly.
        result = store_schedule(tmem_c_resource, gmem_d_resource, work_queue, pdl_wait)
        store_src_resources = [tmem_c_resource, work_queue, pdl_wait]
    else:
        result = store_schedule(tmem_c_resource, gmem_d_resource, work_queue)
        store_src_resources = [tmem_c_resource, work_queue]
    return Task(
        src_resources=store_src_resources,
        dst_resources=[gmem_d_resource],
        warp_idx=0,
        num_warps=num_epilogue_warps,
        schedule=result,
        num_registers=160,
        name="StoreTask",
        debug_print=debug_print,
    )


@cute.jit
def create_work_schedule_task(
    work_queue: WorkQueue,
    num_scheduler_warps: int,
    scheduler_warp_idx: int,
) -> Task:
    """
    Create the CLC dynamic persistent scheduler task. Dynamic mode only.
    """

    @schedule
    def scheduler_schedule(wq: WorkQueue) -> None:
        # Dedicated CLC task fetches new work tiles; data tasks only consume them.
        with work_tile_loop(wq) as work_tile:
            with domain_loop(0, 0, 1):
                pass  # no K-loop work for the scheduler
            # TAIL: fetch and distribute next work tile
            wq.try_acquire()
            wq.acquire()
            wq.fetch_work_tile()
            wq.commit()
            wq.try_wait()
            wq.wait()
            wq.get_and_advance_work_tile()
            wq.release()

    result = scheduler_schedule(work_queue)
    return Task(
        src_resources=[work_queue],
        dst_resources=[work_queue],
        warp_idx=scheduler_warp_idx,
        num_warps=num_scheduler_warps,
        schedule=result,
        num_registers=40,
        name="WorkScheduleTask",
        debug_print=debug_print,
    )


########################################################
# Resource and task construction
########################################################


def _create_smem_sf_resource(
    tma_desc: cutlass.Pointer,
    is_a: bool,
    stage_bytes_val: int,
    cluster_shape_vmnk: cute.Layout,
    act_num_pairs: int,
    act_num_pair_cols: int,
    act_a_mcast_template: int,
    act_b_mcast_template: int,
) -> SmemSfResource:
    """TMA+UMMA pipeline for one NVFP4 scale operand."""
    num_tma_copy_bytes = stage_bytes_val * num_mma_ctas
    pipeline_config = PipelineConfig.create_tma_umma_pipeline_cfg(
        num_stages=ab_stages,
        num_bytes=num_tma_copy_bytes,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, act_num_pairs),
        cta_layout_vmnk=cluster_shape_vmnk,
        consumer_signaling_threads=SignalingThreads.CtaLeader,
        num_bytes_per_warp_per_cta=stage_bytes_val,
    )
    return SmemSfResource(
        tma_desc=tma_desc,
        is_a=is_a,
        stage_bytes_val=stage_bytes_val,
        act_num_pair_cols=act_num_pair_cols,
        act_a_mcast_template=act_a_mcast_template,
        act_b_mcast_template=act_b_mcast_template,
        pipeline_config=pipeline_config,
        name="SmemSfA" if is_a else "SmemSfB",
    )


def _create_gemm_pipeline(
    smem_allocator: SmemAllocator,
    smem_a_alloc: SmemAllocation,
    smem_b_alloc: SmemAllocation,
    smem_sfa_alloc: SmemAllocation,
    smem_sfb_alloc: SmemAllocation,
    tma_sfa_desc: object,
    tma_sfb_desc: object,
    weight_scale_alloc: SmemAllocation,
    output_alloc: SmemAllocation,
    qk_norm_weight_alloc: SmemAllocation,
    tma_a_desc: object,
    tma_b_desc: object,
    tma_c_desc: object,
    mC_mn: object,
    bias: object,
    scale: object,
    x_scale: object,
    weight_scale: object,
    sf_c: object,
    scale_c: object,
    scale_gate: object,
    qkv_scale: object,
    q_norm_weight: object,
    k_norm_weight: object,
    cos_sin_cache: object,
    positions: object,
    tile_sched_params: object,
    num_k_tiles: int,
    act_num_pairs: int,
    act_num_pair_cols: int,
    act_a_mcast_template: int,
    act_b_mcast_template: int,
    act_cluster_shape_vmnk: object,
    clc_response_ptr: object = None,
) -> Tuple[TaskManager, Task, Task]:
    """Create resources, tasks, and a TaskManager (compile-time).

    Must be called outside any dynamic branch so that pipeline
    configurations keep their compile-time sizes.

    The caller owns the shared-memory allocator and allocation descriptors.
    Preferred and fallback resources therefore use identical offsets instead
    of emitting one full shared-memory block per cluster variant.
    """
    num_mma_warps = 1
    num_load_warps = 2
    num_scheduler_warps = 1 if use_clc_dynamic_scheduler else 0

    ########################################################
    # Resource construction
    ########################################################

    # Resource skeleton: GMEM coords -> SMEM A/B -> TMEM accumulator -> GMEM D.
    gmem_ab_resource = create_gmem_ab_resource()

    smem_a_resource = create_smem_ab_resource(
        tma_a_desc,
        tma_b_desc,
        "a",
        act_cluster_shape_vmnk,
        act_num_pairs=act_num_pairs,
        act_num_pair_cols=act_num_pair_cols,
        act_a_mcast_template=act_a_mcast_template,
        act_b_mcast_template=act_b_mcast_template,
    )
    smem_a_resource._alloc = smem_a_alloc
    smem_b_resource = create_smem_ab_resource(
        tma_a_desc,
        tma_b_desc,
        "b",
        act_cluster_shape_vmnk,
        act_num_pairs=act_num_pairs,
        act_num_pair_cols=act_num_pair_cols,
        act_a_mcast_template=act_a_mcast_template,
        act_b_mcast_template=act_b_mcast_template,
    )
    smem_b_resource._alloc = smem_b_alloc
    smem_sfa_resource = None
    smem_sfb_resource = None
    if cutlass.const_expr(use_nvfp4_block_scale):
        smem_sfa_resource = _create_smem_sf_resource(
            tma_sfa_desc,
            True,
            sfa_stage_bytes,
            act_cluster_shape_vmnk,
            act_num_pairs,
            act_num_pair_cols,
            act_a_mcast_template,
            act_b_mcast_template,
        )
        smem_sfa_resource._alloc = smem_sfa_alloc
        smem_sfb_resource = _create_smem_sf_resource(
            tma_sfb_desc,
            False,
            sfb_stage_bytes,
            act_cluster_shape_vmnk,
            act_num_pairs,
            act_num_pair_cols,
            act_a_mcast_template,
            act_b_mcast_template,
        )
        smem_sfb_resource._alloc = smem_sfb_alloc
    pdl_wait = PdlWaitBarrier(name="PdlWait")
    pdl_launch = PdlLaunchBarrier(name="PdlLaunch")

    tmem_c_resource = create_tmem_c_resource(
        num_epilogue_warps,
        act_cluster_shape_vmnk,
    )
    gmem_d_resource = create_gmem_d_resource(
        mC_mn,
        tma_c_desc,
        bias,
        scale,
        x_scale,
        weight_scale,
        sf_c,
        scale_c,
        scale_gate,
        qkv_scale,
        q_norm_weight,
        k_norm_weight,
        cos_sin_cache,
        positions,
    )
    gmem_d_resource._weight_scale_alloc = weight_scale_alloc
    gmem_d_resource._output_alloc = output_alloc
    gmem_d_resource._qk_norm_weight_alloc = qk_norm_weight_alloc

    # TMEM allocator tracks accumulator ownership for TaskManager validation.
    tmem_allocator = TmemAllocator()
    tmem_allocator.add_resource(tmem_c_resource)
    tmem_allocator.compute_layout()

    total_num_warps_so_far = num_mma_warps + num_load_warps + num_epilogue_warps
    if cutlass.const_expr(use_clc_dynamic_scheduler):
        total_num_warps_so_far += num_scheduler_warps
    num_padding_warps = (total_num_warps_so_far + 3) // 4 * 4 - total_num_warps_so_far

    if cutlass.const_expr(use_clc_dynamic_scheduler):
        work_queue = create_work_queue(
            tile_sched_params,
            act_cluster_shape_vmnk,
            num_load_warps,
            num_epilogue_warps,
            num_mma_warps,
            num_padding_warps,
            num_scheduler_warps=num_scheduler_warps,
            clc_response_ptr=clc_response_ptr,
        )
    else:
        work_queue = create_work_queue(
            tile_sched_params,
            act_cluster_shape_vmnk,
            num_load_warps,
            num_epilogue_warps,
            num_mma_warps,
            num_padding_warps,
        )

    ########################################################
    # Task schedule construction
    ########################################################

    # Bind the captured schedules to concrete warp ranges.
    load_a_task = create_load_a_task(
        gmem_ab_resource,
        smem_a_resource,
        pdl_wait,
        pdl_launch,
        work_queue,
        num_k_tiles,
        smem_sfa_resource,
    )
    load_b_task = create_load_b_task(
        gmem_ab_resource,
        smem_b_resource,
        work_queue,
        num_k_tiles,
        smem_sfb_resource,
    )
    mma_task = create_mma_task(
        smem_a_resource,
        smem_b_resource,
        tmem_c_resource,
        work_queue,
        num_k_tiles,
        num_mma_warps,
        num_load_warps,
        smem_sfa_resource,
        smem_sfb_resource,
    )
    store_task = create_store_task(
        tmem_c_resource,
        gmem_d_resource,
        work_queue,
        num_k_tiles,
        num_epilogue_warps,
        pdl_wait if use_per_token_channel_scale else None,
    )
    task_list = [load_a_task, load_b_task, mma_task, store_task]

    if cutlass.const_expr(num_padding_warps > 0):
        padding_task = create_padding_task(
            work_queue,
            num_padding_warps,
            total_num_warps_so_far,
            num_k_tiles,
        )
        task_list.append(padding_task)

    if cutlass.const_expr(use_clc_dynamic_scheduler):
        scheduler_warp_idx = total_num_warps_so_far - num_scheduler_warps
        work_schedule_task = create_work_schedule_task(
            work_queue, num_scheduler_warps, scheduler_warp_idx
        )
        task_list.append(work_schedule_task)

    # Dependency graph records value/pipeline flow between resources.
    resource_dependency_graph = {
        pdl_launch: [],
        smem_a_resource: [gmem_ab_resource, pdl_wait, work_queue],
        smem_b_resource: [gmem_ab_resource, work_queue],
        tmem_c_resource: (
            [
                smem_a_resource,
                smem_b_resource,
                smem_sfa_resource,
                smem_sfb_resource,
                work_queue,
            ]
            if use_nvfp4_block_scale
            else [smem_a_resource, smem_b_resource, work_queue]
        ),
        gmem_d_resource: (
            [tmem_c_resource, work_queue, pdl_wait]
            if use_per_token_channel_scale
            else [tmem_c_resource, work_queue]
        ),
    }
    if cutlass.const_expr(use_nvfp4_block_scale):
        resource_dependency_graph[smem_sfa_resource] = [
            gmem_ab_resource,
            pdl_wait,
            work_queue,
        ]
        resource_dependency_graph[smem_sfb_resource] = [gmem_ab_resource, work_queue]
    if cutlass.const_expr(use_clc_dynamic_scheduler):
        resource_dependency_graph[work_queue] = [work_queue]

    ########################################################
    # TaskManager construction
    ########################################################

    # TaskManager validates the skeleton and wires resource contexts/allocators.
    # The exhaustive search is host-side work on the JIT path, so production
    # skips it. Unit tests opt in through FLASHINFER_PRIMS_TS_DEBUG_CHECKS.
    debug_checks = _prims_ts_debug_checks_enabled()
    task_manager = TaskManager(
        tasks=task_list,
        resource_dependency_graph=resource_dependency_graph,
        smem_allocator=smem_allocator,
        tmem_allocator=tmem_allocator,
        skip_validation=not debug_checks,
        exhaustive_deadlock_race_check=debug_checks,
        verbose=False,
    )
    return task_manager, mma_task, store_task


@cute.jit
def _run_gemm_execution(
    task_manager: TaskManager,
    mma_task: Task,
    store_task: Task,
    warp_idx: object,
    tmem_ptr_alloc: SmemAllocation,
    dealloc_mbar_alloc: SmemAllocation,
) -> None:
    """Setup barriers, sync cluster, allocate TMEM, run, deallocate TMEM.

    Safe to call inside a dynamic branch — no pipeline / resource creation.
    Infrastructure pointers (``tmem_ptr_i32``, ``tmem_dealloc_mbar_ptr``)
    are derived from the unified SMEM block after ``allocate()``.
    """
    num_mma_warps = 1
    tmem_allocator_warp_id = 0

    task_manager.setup_resources_and_tasks()

    # Derive infrastructure pointers from the unified SMEM block.
    # tmem_ptr_i32 for ResourceContext is auto-populated by TaskManager
    # in setup_resources_and_tasks() via SmemAllocator.tmem_ptr_alloc.
    allocator = task_manager.smem_allocator
    tmem_ptr_i32 = allocator.get(tmem_ptr_alloc)
    tmem_dealloc_mbar_ptr = allocator.get(dealloc_mbar_alloc)

    if warp_idx == tmem_allocator_warp_id:
        if prims.elect_sync():
            prims.mbarrier_init(tmem_dealloc_mbar_ptr, cute.arch.WARP_SIZE)

    prims.fence_mbarrier_init()
    prims.barrier_cluster_arrive_relaxed()
    prims.barrier_cluster_wait()

    num_tmem_cols = num_tmem_alloc_cols
    tmem_bar_id = 2
    tmem_bar_threads = (num_epilogue_warps + num_mma_warps) * 32

    if warp_idx == tmem_allocator_warp_id:
        prims.tcgen05_alloc(tmem_ptr_i32, num_tmem_cols, group="cta_2")
        prims.tcgen05_relinquish_alloc_permit(group="cta_2")

    if store_task.is_selected() or mma_task.is_selected():
        prims.barrier_cta_sync(tmem_bar_id, thread_count=tmem_bar_threads)

    tmem_raw_addr = cutlass.Int32(0)
    if store_task.is_selected() or mma_task.is_selected():
        tmem_raw_addr = tmem_ptr_i32.load()

    tmem_ptr = prims.make_tmem_ptr(tmem_raw_addr, acc_dtype)

    task_manager.run()

    dealloc_bar_id = 3
    dealloc_bar_threads = num_epilogue_warps * 32
    if store_task.is_selected():
        prims.barrier_cta_sync(dealloc_bar_id, thread_count=dealloc_bar_threads)

    if warp_idx == tmem_allocator_warp_id:
        cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
        peer_cta_rank = cta_rank_in_cluster ^ 1

        peer_mbar = prims.mapa(tmem_dealloc_mbar_ptr, peer_cta_rank)
        prims.mbarrier_arrive(peer_mbar, count=1, scope=prims.MemScope.CTA)

        while not prims.mbarrier_try_wait_parity(
            tmem_dealloc_mbar_ptr, 0, time_limit=10000000
        ):
            pass

        prims.tcgen05_dealloc(tmem_ptr, num_tmem_cols, group="cta_2")


########################################################
# Kernel
########################################################


@cute.kernel
def kernel(
    tma_a_desc: cutlass.GridConstant[cuda.TensorMap],
    tma_b_desc: cutlass.GridConstant[cuda.TensorMap],
    tma_c_desc: cutlass.GridConstant[cuda.TensorMap],
    tma_sfa_desc: cutlass.GridConstant[cuda.TensorMap],
    tma_sfb_desc: cutlass.GridConstant[cuda.TensorMap],
    mC_mn: cute.Tensor,
    mnk: Tuple[int, int, int],
    tile_sched_params: object,
    fallback_tile_sched_params: object = None,
    bias: Optional[cute.Tensor] = None,
    scale: cutlass.Float32 = 1.0,
    x_scale: Optional[cute.Tensor] = None,
    weight_scale: Optional[cute.Tensor] = None,
    sf_c: Optional[cute.Tensor] = None,
    scale_c: Optional[cute.Tensor] = None,
    scale_gate: Optional[cute.Tensor] = None,
    qkv_scale: Optional[cute.Tensor] = None,
    q_norm_weight: Optional[cute.Tensor] = None,
    k_norm_weight: Optional[cute.Tensor] = None,
    cos_sin_cache: Optional[cute.Tensor] = None,
    positions: Optional[cute.Tensor] = None,
) -> None:
    """Warp-specialised persistent GEMM kernel with optional fallback cluster support.

    Execution flow
    ==============

    1. Configuration & prefetch
       Extract problem size, compute K-tile count, prefetch TMA descriptors.

    2. Shared-memory layout and pipeline creation (compile-time)
       Lay out one set of A/B buffers and infrastructure slots, then pass
       those allocation descriptors to every cluster variant.
       Build resources (GmemAb, SmemAb, TmemC, GmemD, WorkQueue),
       tasks (Load, Mma, Store, Padding, WorkSchedule), and TaskManager.
       Data SMEM buffers, the TMEM pointer slot, and the TMEM-dealloc
       mbarrier are unified via ``SmemAllocator``; the CLC response
       buffer (when enabled) is allocated separately because the work
       queue consumes it before ``allocate()`` runs.
       When a fallback cluster is enabled, TWO pipelines are created before
       any dynamic branch — one for the preferred cluster shape and one
       for the fallback — so that pipeline.CooperativeGroup sizes remain
       compile-time constants.

    3. Fallback-cluster detection (when fallback_cluster_shape_mnk is set)
       Query runtime cluster dimensions via block_in_cluster_dim() and
       branch to select which pre-built pipeline to execute.

    4. Execution via _run_gemm_execution (runtime)

       a. setup_resources_and_tasks()
          Unified SMEM allocation (data + infra) and pipeline barrier
          init.

       b. Derive infrastructure pointers
          ``tmem_ptr_i32`` and ``tmem_dealloc_mbar_ptr`` from the
          unified SMEM block via ``SmemAllocator.get()``.

       c. Barrier init & cluster sync
          Init TMEM-dealloc mbarrier, fence, cluster_arrive / cluster_wait.

       d. TMEM allocation (2-CTA)
          tcgen05_alloc, relinquish alloc permit, per-CTA barrier,
          read TMEM pointer.

       e. task_manager.run()
          For each task: select warps, set register budget, create
          function/work/loop variables, execute head/loop/tail schedules
          inside the persistent work loop.

       f. TMEM deallocation
          Peer-CTA mbarrier handshake, tcgen05_dealloc.
    """
    m, n, k = mnk

    num_k_tiles = (k + schedule_tile_k - 1) // schedule_tile_k

    warp_idx = cute.arch.warp_idx()

    if warp_idx == num_epilogue_warps:
        prims.prefetch_tensormap(tma_a_desc.get_ptr())
        prims.prefetch_tensormap(tma_b_desc.get_ptr())
        if cutlass.const_expr(use_nvfp4_block_scale):
            prims.prefetch_tensormap(tma_sfa_desc.get_ptr())
            prims.prefetch_tensormap(tma_sfb_desc.get_ptr())
        if cutlass.const_expr(use_tma_store):
            prims.prefetch_tensormap(tma_c_desc.get_ptr())

    # clc_response_ptr remains a separate allocation because the work queue
    # needs a concrete pointer while building TileSchedulerConfig, before a
    # descriptor in the shared allocator could be resolved with get().
    clc_response_ptr = None
    if cutlass.const_expr(use_clc_dynamic_scheduler):
        # Allocate per-stage CLC response buffers (16 bytes per stage).
        clc_response_ptr = cute.arch.alloc_smem(cutlass.Int128, num_scheduler_stages)

    pref_vmnk = (
        num_mma_ctas,
        cluster_shape_mnk[0] // num_mma_ctas,
        cluster_shape_mnk[1],
        cluster_shape_mnk[2],
    )

    # Lay out shared memory once for every cluster variant. Each pipeline
    # owns distinct resource objects (and compile-time pipeline constants),
    # but those resources reuse these allocation descriptors and offsets.
    smem_allocator = SmemAllocator()
    smem_a_alloc = smem_allocator.add(
        SmemAllocation(
            "smem_a",
            _span_bytes(
                mma_tiler_mnk_per_cta[0] * mma_tiler_mnk[2] * ab_stages, input_dtype
            ),
            alignment=128,
        )
    )
    smem_b_alloc = smem_allocator.add(
        SmemAllocation(
            "smem_b",
            _span_bytes(
                mma_tiler_mnk_per_cta[1] * mma_tiler_mnk[2] * ab_stages, input_dtype
            ),
            alignment=128,
        )
    )
    smem_sfa_alloc = SmemAllocation(
        "smem_sfa",
        sfa_stage_bytes * ab_stages,
        alignment=128,
    )
    smem_sfb_alloc = SmemAllocation(
        "smem_sfb",
        sfb_stage_bytes * ab_stages,
        alignment=128,
    )
    if cutlass.const_expr(use_nvfp4_block_scale):
        smem_allocator.add(smem_sfa_alloc)
        smem_allocator.add(smem_sfb_alloc)
    weight_scale_alloc = SmemAllocation(
        "weight_scale_smem",
        mma_tiler_mnk[1] * (cutlass.Float32.width // 8),
        alignment=16,
    )
    if cutlass.const_expr(use_per_token_channel_scale):
        smem_allocator.add(weight_scale_alloc)
    output_subtile_n = 16 if use_gated_activation else _EPILOGUE_TILE_N
    output_alloc = SmemAllocation(
        "output_smem",
        mma_tiler_mnk_per_cta[0]
        * output_subtile_n
        * _epilogue_store_stages
        * (num_epilogue_warps // 4)
        * max(1, output_dtype.width // 8),
        alignment=128,
    )
    if cutlass.const_expr(use_tma_store):
        smem_allocator.add(output_alloc)
    qk_norm_weight_alloc = SmemAllocation(
        "qk_norm_weight_smem",
        2 * _QKV_HEAD_DIM * (cutlass.BFloat16.width // 8),
        alignment=16,
    )
    if cutlass.const_expr(use_fused_qknorm_rope):
        smem_allocator.add(qk_norm_weight_alloc)
    # TMEM allocation returns the raw address through this SMEM mailbox.
    tmem_ptr_alloc = smem_allocator.add_tmem_ptr(
        SmemAllocation("tmem_ptr_i32", dtype=cutlass.Int32, alignment=4)
    )
    # TMEM deallocation uses a shared-memory barrier in the same block.
    dealloc_mbar_alloc = smem_allocator.add(
        SmemAllocation("tmem_dealloc_mbar", dtype=cutlass.Int64, alignment=8)
    )
    smem_allocator.compute_layout()
    # Materialize the single block before the preferred/fallback runtime
    # branch so its base pointer dominates both execution paths.
    smem_allocator.allocate()

    if cutlass.const_expr(fallback_cluster_shape_mnk is not None):
        # Compute fallback constants at Python level (compile-time)
        fb = fallback_cluster_shape_mnk
        fb_num_pairs = (fb[0] * fb[1]) // num_mma_ctas
        fb_num_pair_rows = fb[0] // num_mma_ctas
        fb_num_pair_cols = fb[1]
        fb_a_mcast = sum(1 << (num_mma_ctas * c) for c in range(fb_num_pair_cols))
        fb_b_mcast = sum(
            1 << (fb_num_pair_cols * num_mma_ctas * r) for r in range(fb_num_pair_rows)
        )
        fb_vmnk = (num_mma_ctas, fb[0] // num_mma_ctas, fb[1], fb[2])

        # Create BOTH pipelines outside any dynamic branch so that
        # pipeline.CooperativeGroup sizes remain compile-time constants.
        pref_tm, pref_mma, pref_store = _create_gemm_pipeline(
            smem_allocator,
            smem_a_alloc,
            smem_b_alloc,
            smem_sfa_alloc,
            smem_sfb_alloc,
            tma_sfa_desc.get_ptr(),
            tma_sfb_desc.get_ptr(),
            weight_scale_alloc,
            output_alloc,
            qk_norm_weight_alloc,
            tma_a_desc.get_ptr(),
            tma_b_desc.get_ptr(),
            tma_c_desc.get_ptr(),
            mC_mn,
            bias,
            scale,
            x_scale,
            weight_scale,
            sf_c,
            scale_c,
            scale_gate,
            qkv_scale,
            q_norm_weight,
            k_norm_weight,
            cos_sin_cache,
            positions,
            tile_sched_params,
            num_k_tiles,
            act_num_pairs=num_pairs,
            act_num_pair_cols=num_pair_cols,
            act_a_mcast_template=_a_mcast_template,
            act_b_mcast_template=_b_mcast_template,
            act_cluster_shape_vmnk=pref_vmnk,
            clc_response_ptr=clc_response_ptr,
        )
        fb_tm, fb_mma, fb_store = _create_gemm_pipeline(
            smem_allocator,
            smem_a_alloc,
            smem_b_alloc,
            smem_sfa_alloc,
            smem_sfb_alloc,
            tma_sfa_desc.get_ptr(),
            tma_sfb_desc.get_ptr(),
            weight_scale_alloc,
            output_alloc,
            qk_norm_weight_alloc,
            tma_a_desc.get_ptr(),
            tma_b_desc.get_ptr(),
            tma_c_desc.get_ptr(),
            mC_mn,
            bias,
            scale,
            x_scale,
            weight_scale,
            sf_c,
            scale_c,
            scale_gate,
            qkv_scale,
            q_norm_weight,
            k_norm_weight,
            cos_sin_cache,
            positions,
            (
                fallback_tile_sched_params
                if fallback_tile_sched_params is not None
                else tile_sched_params
            ),
            num_k_tiles,
            act_num_pairs=fb_num_pairs,
            act_num_pair_cols=fb_num_pair_cols,
            act_a_mcast_template=fb_a_mcast,
            act_b_mcast_template=fb_b_mcast,
            act_cluster_shape_vmnk=fb_vmnk,
            clc_response_ptr=clc_response_ptr,
        )

        # Runtime: detect actual cluster size and branch for execution only
        cbdim_x, cbdim_y, _ = cute.arch.block_in_cluster_dim()
        is_preferred = (cbdim_x == cluster_shape_mnk[0]) & (
            cbdim_y == cluster_shape_mnk[1]
        )

        if is_preferred:
            _run_gemm_execution(
                pref_tm,
                pref_mma,
                pref_store,
                warp_idx,
                tmem_ptr_alloc,
                dealloc_mbar_alloc,
            )
        else:
            _run_gemm_execution(
                fb_tm,
                fb_mma,
                fb_store,
                warp_idx,
                tmem_ptr_alloc,
                dealloc_mbar_alloc,
            )
    else:
        tm, mma, store = _create_gemm_pipeline(
            smem_allocator,
            smem_a_alloc,
            smem_b_alloc,
            smem_sfa_alloc,
            smem_sfb_alloc,
            tma_sfa_desc.get_ptr(),
            tma_sfb_desc.get_ptr(),
            weight_scale_alloc,
            output_alloc,
            qk_norm_weight_alloc,
            tma_a_desc.get_ptr(),
            tma_b_desc.get_ptr(),
            tma_c_desc.get_ptr(),
            mC_mn,
            bias,
            scale,
            x_scale,
            weight_scale,
            sf_c,
            scale_c,
            scale_gate,
            qkv_scale,
            q_norm_weight,
            k_norm_weight,
            cos_sin_cache,
            positions,
            tile_sched_params,
            num_k_tiles,
            act_num_pairs=num_pairs,
            act_num_pair_cols=num_pair_cols,
            act_a_mcast_template=_a_mcast_template,
            act_b_mcast_template=_b_mcast_template,
            act_cluster_shape_vmnk=pref_vmnk,
            clc_response_ptr=clc_response_ptr,
        )
        _run_gemm_execution(
            tm,
            mma,
            store,
            warp_idx,
            tmem_ptr_alloc,
            dealloc_mbar_alloc,
        )


def compute_grid(
    problem_m,
    problem_n,
    mma_tiler_mnk: Tuple[int, int, int],
    cluster_shape_mnk: Tuple[int, int, int],
    max_active_clusters: cutlass.Constexpr,
) -> Tuple[object, Tuple[int, int, int]]:
    # Tile counts follow the GEMM problem N, not the D tensor. Gated SiLU
    # writes N/2 columns, but MMA tiles are still tile-N-wide in GEMM-N.
    tile_m = mma_tiler_mnk[0]
    tile_n = mma_tiler_mnk[1]
    # The scheduler operates in per-CTA tiles. Round up so the final cluster
    # can cover a partial tile. In particular, TMA zero-fills out-of-bounds A
    # rows and the epilogue predicates M-tail stores against the tensor shape.
    num_ctas_mn = (
        (problem_m + tile_m - 1) // tile_m,
        (problem_n + tile_n - 1) // tile_n,
    )
    cluster_shape_mnl = (*cluster_shape_mnk[:2], 1)

    if use_clc_dynamic_scheduler:
        # CLC dynamic persistent: swizzle_size=1, raster_along_m=True
        tile_sched_params = utils.ClcDynamicPersistentTileSchedulerParams(
            (*num_ctas_mn, 1), cluster_shape_mnl, 1, True
        )
        grid = utils.ClcDynamicPersistentTileScheduler.get_grid_shape(tile_sched_params)
    else:
        tile_sched_params = utils.PersistentTileSchedulerParams(
            (*num_ctas_mn, 1), cluster_shape_mnk
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )
    return tile_sched_params, grid


########################################################
# Host runners and CLI
########################################################


def _make_fp4_ab_tensormap(tensor, k: int, rows: int, box_rows: int):
    """K-major fp4 map. ``rows`` is M for A and N for B."""
    layout = cute.make_layout(
        (cute.assume(k, 32), rows),
        stride=(1, cute.assume(k, 32)),
    )
    viewed = cute.make_tensor(
        cute.recast_ptr(tensor.iterator, dtype=cutlass.Float4E2M1FN),
        layout,
    )
    return cuda.create_tensor_map_tiled_from_view(
        tensor=viewed,
        box_dims=(tma_k_box_elems, box_rows),
        stride_order=(0, 1),
        swizzle=cuda.TensorMapSwizzle.s128b,
        dtype=cutlass.Float4E2M1FNx2,
        tma_format=cuda.TensorMapDataFormat.B4X16,
    )


def _make_nvfp4_scale_tensormap(tensor, rows: int, k: int):
    """Unswizzled Uint16 view of the CUTLASS 128x4 scale layout."""
    rest_k = k // _SF_VEC_SIZE // 4
    rest_mn = (rows + 127) // 128
    layout = cute.make_layout(
        (256, rest_k, rest_mn, 1),
        stride=(1, 256, 256 * rest_k, 256 * rest_k * rest_mn),
    )
    viewed = cute.make_tensor(
        cute.recast_ptr(tensor.iterator, dtype=cutlass.Float16),
        layout,
    )
    return cuda.create_tensor_map_tiled_from_view(
        tensor=viewed,
        box_dims=(256, sf_tile_k // _SF_VEC_SIZE // 4, 1, 1),
        stride_order=(0, 1, 2, 3),
        swizzle=cuda.TensorMapSwizzle.none,
        dtype=cutlass.Uint16,
    )


@cute.jit
def host_function(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    mnk: Tuple[int, int, int],
    max_active_clusters: cutlass.Constexpr,
    stream,
    bias: Optional[cute.Tensor] = None,
    scale: cutlass.Float32 = 1.0,
    x_scale: Optional[cute.Tensor] = None,
    weight_scale: Optional[cute.Tensor] = None,
    q_norm_weight: Optional[cute.Tensor] = None,
    k_norm_weight: Optional[cute.Tensor] = None,
    cos_sin_cache: Optional[cute.Tensor] = None,
    positions: Optional[cute.Tensor] = None,
    sfa: Optional[cute.Tensor] = None,
    sfb: Optional[cute.Tensor] = None,
    sf_c: Optional[cute.Tensor] = None,
    scale_c: Optional[cute.Tensor] = None,
    scale_gate: Optional[cute.Tensor] = None,
    qkv_scale: Optional[cute.Tensor] = None,
) -> None:
    # Construct TMA Descriptors
    # A/B use cutlass to create tensormap descriptors (like fp16_gemm_0_prim.py)
    # box_dims in tensor's original mode order: A is (M, K), B is (N, K)
    # unless BlockMajorK remaps B to (block_k, N, K/block_k).
    # NVFP4 rebuilds K-major logical maps: one element is 4 bits, so the
    # packed tensor shape is not the TMA element count.
    m, n, k = mnk
    if cutlass.const_expr(use_nvfp4_block_scale):
        tma_a_desc = _make_fp4_ab_tensormap(a, k, m, mma_tiler_mnk_per_cta[0])
        tma_b_desc = _make_fp4_ab_tensormap(b, k, n, mma_tiler_mnk_per_cta[1])
    else:
        tma_a_desc = cuda.create_tensor_map_tiled_from_view(
            a,
            box_dims=(mma_tiler_mnk_per_cta[0], tma_k_box_elems),
            stride_order=(1, 0),
            swizzle=cuda.TensorMapSwizzle.s128b,
        )
    if cutlass.const_expr(not use_nvfp4_block_scale and use_block_major_k):
        b_block_k = block_major_k_elems
        b_tile_block_k = min(b_block_k, mma_tiler_mnk[2])
        b_layout = cute.make_layout(
            (b_block_k, n, cute.assume(k // b_block_k, 1)),
            stride=(
                1,
                b_block_k,
                cute.assume(n * b_block_k, 32),
            ),
        )
        b_tensor = cute.make_tensor(b.iterator, b_layout)
        tma_b_desc = cuda.create_tensor_map_tiled_from_view(
            b_tensor,
            box_dims=(
                b_tile_block_k,
                mma_tiler_mnk_per_cta[1],
                mma_tiler_mnk[2] // b_tile_block_k,
            ),
            stride_order=(0, 1, 2),
            swizzle=cuda.TensorMapSwizzle.s128b,
        )
    elif cutlass.const_expr(not use_nvfp4_block_scale):
        tma_b_desc = cuda.create_tensor_map_tiled_from_view(
            b,
            box_dims=(mma_tiler_mnk_per_cta[1], tma_k_box_elems),
            stride_order=(1, 0),
            swizzle=cuda.TensorMapSwizzle.s128b,
        )
    output_subtile_n = 16 if use_gated_activation else _EPILOGUE_TILE_N
    c_kernel = c
    if cutlass.const_expr(_is_fp4(output_dtype)):
        # Torch exposes packed FP4 as a byte buffer. Give the kernel a logical
        # MxN E2M1 view while retaining the original storage owner.
        output_n = n // 2 if use_gated_activation else n
        c_layout = cute.make_layout(
            (m, output_n),
            stride=(cute.assume(output_n, 32), 1),
        )
        c_kernel = cute.make_tensor(
            cute.recast_ptr(c.iterator, dtype=cutlass.Float4E2M1FN), c_layout
        )
        # FP4 output uses direct packed stores; keep the uniform kernel ABI by
        # supplying an otherwise-unused valid tensor-map descriptor.
        tma_c_desc = tma_a_desc
    else:
        tma_c_desc = cuda.create_tensor_map_tiled_from_view(
            c,
            box_dims=(mma_tiler_mnk_per_cta[0], output_subtile_n),
            stride_order=(1, 0),
            swizzle=cuda.TensorMapSwizzle.none,
        )
    if cutlass.const_expr(use_nvfp4_block_scale):
        tma_sfa_desc = _make_nvfp4_scale_tensormap(sfa, m, k)
        tma_sfb_desc = _make_nvfp4_scale_tensormap(sfb, n, k)
    else:
        # Unused on this compile. Keep the kernel signature uniform.
        tma_sfa_desc = tma_c_desc
        tma_sfb_desc = tma_c_desc

    cta_tile_shape_mnk = (
        mma_tiler_mnk_per_cta[0],
        mma_tiler_mnk[1],
        mma_tiler_mnk[2],
    )

    # Launch the kernel. Grid is sized from GEMM (M, N), even when D is N/2.
    tile_sched_params, grid_shape = compute_grid(
        m,
        n,
        cta_tile_shape_mnk,
        cluster_shape_mnk,
        max_active_clusters,
    )
    fallback_tile_sched_params = None
    if cutlass.const_expr(
        use_clc_dynamic_scheduler and fallback_cluster_shape_mnk is not None
    ):
        fallback_tile_sched_params, _ = compute_grid(
            m,
            n,
            cta_tile_shape_mnk,
            fallback_cluster_shape_mnk,
            max_active_clusters,
        )

    # Epilogue + one MMA warp + A/B loads.
    num_load_warps = 2
    block_size = cute.arch.WARP_SIZE * (num_epilogue_warps + 1 + num_load_warps)
    if cutlass.const_expr(use_clc_dynamic_scheduler):
        block_size = cute.arch.WARP_SIZE * (num_epilogue_warps + 1 + num_load_warps + 1)
    # Pad block size to warp-group granularity
    block_size = (block_size + 127) // 128 * 128
    if cutlass.const_expr(fallback_cluster_shape_mnk is not None):
        kernel(
            tma_a_desc,
            tma_b_desc,
            tma_c_desc,
            tma_sfa_desc,
            tma_sfb_desc,
            c_kernel,
            mnk,
            tile_sched_params,
            fallback_tile_sched_params,
            bias,
            scale,
            x_scale,
            weight_scale,
            sf_c,
            scale_c,
            scale_gate,
            qkv_scale,
            q_norm_weight,
            k_norm_weight,
            cos_sin_cache,
            positions,
        ).launch(
            grid=grid_shape,
            block=[block_size, 1, 1],
            cluster=cluster_shape_mnk,
            fallback_cluster=fallback_cluster_shape_mnk,
            stream=stream,
            use_pdl=True,
        )
    else:
        kernel(
            tma_a_desc,
            tma_b_desc,
            tma_c_desc,
            tma_sfa_desc,
            tma_sfb_desc,
            c_kernel,
            mnk,
            tile_sched_params,
            fallback_tile_sched_params,
            bias,
            scale,
            x_scale,
            weight_scale,
            sf_c,
            scale_c,
            scale_gate,
            qkv_scale,
            q_norm_weight,
            k_norm_weight,
            cos_sin_cache,
            positions,
        ).launch(
            grid=grid_shape,
            block=[block_size, 1, 1],
            cluster=cluster_shape_mnk,
            stream=stream,
            use_pdl=True,
        )
