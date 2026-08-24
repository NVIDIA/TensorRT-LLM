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

import torch

from ..cuda_tile_utils import IS_CUDA_TILE_AVAILABLE
from ..cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from ..flashinfer_utils import IS_FLASHINFER_AVAILABLE
from .cpp_custom_ops import _register_fake
from .torch_custom_ops import BufferKind, bmm_out
from .trtllm_gen_custom_ops import fp8_block_scale_moe_runner
from .userbuffers_custom_ops import add_to_ub, copy_to_userbuffers, matmul_to_ub

# Attention custom ops are defined in modules.attention, and MLA custom ops are
# defined in modules.mla. They are not re-exported here to avoid circular imports:
# custom_ops must not depend on modules.attention or modules.mla.


def inplace_slice_copy(dest: torch.Tensor, src: torch.Tensor, dim1_start: int,
                       dim1_end: int) -> None:
    torch.ops.trtllm.inplace_slice_copy(dest, src, dim1_start, dim1_end)


__all__ = [
    'IS_FLASHINFER_AVAILABLE',
    '_register_fake',
    'bmm_out',
    'BufferKind',
    'fp8_block_scale_moe_runner',
    'add_to_ub',
    'copy_to_userbuffers',
    'matmul_to_ub',
    'IS_CUTLASS_DSL_AVAILABLE',
    'inplace_slice_copy',
    'TILE_SIZED_GROUPED_GEMM_RUNNERS',
]

# Grouped-GEMM runners whose tactic carries an mma_tiler_mn. cute_dsl_custom_ops
# defines them inside its own ``if IS_CUTLASS_DSL_AVAILABLE:`` block and has no
# else-branch, so without the DSL the names do not exist and importing them
# directly raises. Re-exporting here gives consumers a target whose shape does
# not depend on the environment -- always present, always a tuple, only the
# length varies -- which matters because create_moe imports every MoE backend
# eagerly under _torch.models, so one such import failing takes down the whole
# model-architecture registry rather than just that backend.
TILE_SIZED_GROUPED_GEMM_RUNNERS: tuple[type, ...] = ()

if IS_FLASHINFER_AVAILABLE:
    from .flashinfer_custom_ops import (
        flashinfer_apply_rope_with_cos_sin_cache_inplace,
        flashinfer_fused_add_rmsnorm, flashinfer_gelu_tanh_and_mul,
        flashinfer_gemma_fused_add_rmsnorm, flashinfer_gemma_rmsnorm,
        flashinfer_rmsnorm, flashinfer_silu_and_mul)
    __all__ += [
        'flashinfer_gelu_tanh_and_mul',
        'flashinfer_silu_and_mul',
        'flashinfer_rmsnorm',
        'flashinfer_fused_add_rmsnorm',
        'flashinfer_apply_rope_with_cos_sin_cache_inplace',
        'flashinfer_gemma_fused_add_rmsnorm',
        'flashinfer_gemma_rmsnorm',
    ]

if IS_CUTLASS_DSL_AVAILABLE:
    from . import cute_dsl_custom_ops as _cute_dsl_ops
    from .cute_dsl_custom_ops import (
        cute_dsl_nvfp4_dense_gemm_swiglu_blackwell,
        cute_dsl_nvfp4_dense_gemm_swiglu_fp4out_blackwell,
        cute_dsl_nvfp4_gemm_blackwell)

    # Reached through the module, not imported by name: these four are tuple
    # ingredients rather than re-exports, and binding them would put four more
    # DSL-only attributes on the package -- the env-dependent surface the tuple
    # exists to replace.
    TILE_SIZED_GROUPED_GEMM_RUNNERS = (
        _cute_dsl_ops.Sm100BlockScaledContiguousGroupedGemmRunner,
        _cute_dsl_ops.Sm100BlockScaledContiguousGroupedGemmFinalizeFusionRunner,
        _cute_dsl_ops.Sm100BlockScaledContiguousGroupedGemmSwigluFusionRunner,
        _cute_dsl_ops.Sm100BlockScaledContiguousGatherGroupedGemmActFusionRunner
    )
    __all__ += [
        'cute_dsl_nvfp4_gemm_blackwell',
        'cute_dsl_nvfp4_dense_gemm_swiglu_blackwell',
        'cute_dsl_nvfp4_dense_gemm_swiglu_fp4out_blackwell',
    ]

    # MegaMoE NVFP4 op probes a strict superset of IS_CUTLASS_DSL_AVAILABLE
    # (cutlass.torch + cutlass._mlir + cute_nvgpu MMA atoms + the ported
    # CuteDSL kernel package). The cute_dsl_megamoe_custom_op module
    # sets ``IS_MEGAMOE_OP_AVAILABLE`` based on its own try/except probe;
    # importing the module is safe regardless of the result -- it just
    # logs and leaves ``IS_MEGAMOE_OP_AVAILABLE = False`` on partial
    # cutlass-dsl installs so callers can fall back via the factory.
    from .cute_dsl_megamoe_custom_op import IS_MEGAMOE_OP_AVAILABLE
    if IS_MEGAMOE_OP_AVAILABLE:
        from .cute_dsl_megamoe_custom_op import cute_dsl_megamoe_nvfp4_blackwell
        __all__ += ['cute_dsl_megamoe_nvfp4_blackwell']

if IS_CUTLASS_DSL_AVAILABLE and IS_FLASHINFER_AVAILABLE:
    from .cute_dsl_kimi_k3_custom_ops import kda_prefill
    __all__ += ['kda_prefill']

if IS_CUDA_TILE_AVAILABLE:
    from .cuda_tile_custom_ops import (cuda_tile_rms_norm,
                                       cuda_tile_rms_norm_fuse_residual_)
    __all__ += [
        'cuda_tile_rms_norm',
        'cuda_tile_rms_norm_fuse_residual_',
    ]
