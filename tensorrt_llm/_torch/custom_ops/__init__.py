# Imported first to register the op before the chain via ..modules.attention re-enters this package.
from . import triton_fused_inv_rope_fp8_quant  # noqa: F401  # isort: skip

import torch

from ..cuda_tile_utils import IS_CUDA_TILE_AVAILABLE
from ..cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from ..flashinfer_utils import IS_FLASHINFER_AVAILABLE
from .cpp_custom_ops import _register_fake
from .torch_custom_ops import BufferKind, bmm_out
from .trtllm_gen_custom_ops import fp8_block_scale_moe_runner
from .userbuffers_custom_ops import add_to_ub, copy_to_userbuffers, matmul_to_ub

# Attention custom ops (attn_custom_op_inplace, mla_custom_op_inplace) are defined in
# modules.attention and must be imported from there. They are not re-exported here to
# avoid circular imports: custom_ops must not depend on modules.attention.


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
]

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
    from .cute_dsl_custom_ops import (
        cute_dsl_nvfp4_dense_gemm_swiglu_blackwell,
        cute_dsl_nvfp4_dense_gemm_swiglu_fp4out_blackwell,
        cute_dsl_nvfp4_gemm_blackwell)
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

if IS_CUDA_TILE_AVAILABLE:
    from .cuda_tile_custom_ops import (cuda_tile_rms_norm,
                                       cuda_tile_rms_norm_fuse_residual_)
    __all__ += [
        'cuda_tile_rms_norm',
        'cuda_tile_rms_norm_fuse_residual_',
    ]
