from .cpp_custom_ops import _register_fake
from .flashinfer_custom_ops import IS_FLASHINFER_AVAIABLE
from .torch_custom_ops import bmm_out
from .userbuffers_custom_ops import add_to_ub, copy_to_userbuffers, matmul_to_ub

__all__ = [
    'IS_FLASHINFER_AVAIABLE',
    '_register_fake',
    'bmm_out',
    'add_to_ub',
    'copy_to_userbuffers',
    'matmul_to_ub',
]

if IS_FLASHINFER_AVAIABLE:
    from .flashinfer_custom_ops import (
        flashinfer_apply_rope_with_cos_sin_cache_inplace,
        flashinfer_fused_add_rmsnorm, flashinfer_rmsnorm,
        flashinfer_silu_and_mul)
    __all__ += [
        'flashinfer_silu_and_mul',
        'flashinfer_rmsnorm',
        'flashinfer_fused_add_rmsnorm',
        'flashinfer_apply_rope_with_cos_sin_cache_inplace',
    ]
