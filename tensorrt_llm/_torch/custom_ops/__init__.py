from .cpp_custom_ops import _register_fake
from .flashinfer_custom_ops import IS_FLASHINFER_AVAIABLE
from .torch_custom_ops import bmm_out
from .userbuffers_custom_ops import ub_scaled_mm_allreduce_quant_scaled_mm_op

__all__ = [
    'IS_FLASHINFER_AVAIABLE',
    '_register_fake',
    'bmm_out',
    'ub_scaled_mm_allreduce_quant_scaled_mm_op',
]

if IS_FLASHINFER_AVAIABLE:
    from .flashinfer_custom_ops import (flashinfer_apply_rope_inplace,
                                        flashinfer_fused_add_rmsnorm,
                                        flashinfer_rmsnorm,
                                        flashinfer_silu_and_mul)
    __all__ += [
        'flashinfer_silu_and_mul',
        'flashinfer_rmsnorm',
        'flashinfer_fused_add_rmsnorm',
        'flashinfer_apply_rope_inplace',
    ]
