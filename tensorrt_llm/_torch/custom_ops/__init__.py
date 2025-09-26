from ..cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from ..flashinfer_utils import IS_FLASHINFER_AVAILABLE
from ..modules.attention import attn_custom_op_inplace, mla_custom_op_inplace
from .cpp_custom_ops import _register_fake
from .torch_custom_ops import bmm_out
from .trtllm_gen_custom_ops import fp8_block_scale_moe_runner
from .userbuffers_custom_ops import add_to_ub, copy_to_userbuffers, matmul_to_ub

__all__ = [
    'IS_FLASHINFER_AVAILABLE',
    '_register_fake',
    'bmm_out',
    'fp8_block_scale_moe_runner',
    'add_to_ub',
    'copy_to_userbuffers',
    'matmul_to_ub',
    'attn_custom_op_inplace',
    'mla_custom_op_inplace',
    'IS_CUTLASS_DSL_AVAILABLE',
]

if IS_FLASHINFER_AVAILABLE:
    from .flashinfer_custom_ops import (
        flashinfer_apply_rope_with_cos_sin_cache_inplace,
        flashinfer_fused_add_rmsnorm, flashinfer_gemma_fused_add_rmsnorm,
        flashinfer_gemma_rmsnorm, flashinfer_rmsnorm, flashinfer_silu_and_mul)
    __all__ += [
        'flashinfer_silu_and_mul',
        'flashinfer_rmsnorm',
        'flashinfer_fused_add_rmsnorm',
        'flashinfer_apply_rope_with_cos_sin_cache_inplace',
        'flashinfer_gemma_fused_add_rmsnorm',
        'flashinfer_gemma_rmsnorm',
    ]

if IS_CUTLASS_DSL_AVAILABLE:
    from .cute_dsl_custom_ops import cute_dsl_nvfp4_gemm_blackwell
    __all__ += [
        'cute_dsl_nvfp4_gemm_blackwell',
    ]
