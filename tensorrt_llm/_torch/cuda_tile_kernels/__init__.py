from ..cuda_tile_utils import IS_CUDA_TILE_AVAILABLE

if IS_CUDA_TILE_AVAILABLE:
    from .rms_norm import rms_norm_kernel
    from .rms_norm import rms_norm_kernel_gather
    from .rms_norm import rms_norm_kernel_static_persistent
    from .rms_norm_fuse_residual import rms_norm_fuse_residual_kernel
    from .rms_norm_fuse_residual import rms_norm_fuse_residual_kernel_gather
    from .rms_norm_fuse_residual import rms_norm_fuse_residual_kernel_static_persistent

    __all__ = [
        "rms_norm_kernel",
        "rms_norm_kernel_gather",
        "rms_norm_kernel_static_persistent",
        "rms_norm_fuse_residual_kernel",
        "rms_norm_fuse_residual_kernel_gather",
        "rms_norm_fuse_residual_kernel_static_persistent",
    ]
