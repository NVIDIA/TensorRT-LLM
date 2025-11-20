# Adapted from https://gitlab-master.nvidia.com/dl/tileir/cutile-python/-/blob/main/test/bench_rms_norm.py

import torch

from ..cuda_tile_utils import IS_CUDA_TILE_AVAILABLE

if IS_CUDA_TILE_AVAILABLE:
    import cuda.tile as ct

    from ..cuda_tile_kernels.rms_norm import rms_norm_kernel
    from ..cuda_tile_kernels.rms_norm import rms_norm_kernel_gather
    from ..cuda_tile_kernels.rms_norm import rms_norm_kernel_static_persistent
    from ..cuda_tile_utils import ceil_div
    from ..cuda_tile_utils import next_power_of_2


    @torch.library.custom_op("trtllm::cuda_tile_rms_norm", mutates_args=())
    def cuda_tile_rms_norm(
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        static_persistent: bool,
        gather: bool,
    ) -> torch.Tensor:
        x = x.contiguous()
        weight = weight.contiguous()

        # Allocate output tensor
        y = torch.empty_like(x)
        M, N = x.shape

        if static_persistent:
            NUM_SMS = torch.cuda.get_device_properties(
                "cuda"
            ).multi_processor_count
            TILE_SIZE_M = 4  # Default value, could be made configurable
            TILE_SIZE_N = next_power_of_2(N)

            # Other tile sizes are more optimal when other dimension is too large/too small
            if TILE_SIZE_N <= 1024:
                TILE_SIZE_M = 16
            elif TILE_SIZE_N >= 16384:
                TILE_SIZE_M = 2

            grid_size = min(
                NUM_SMS,
                ceil_div(M, TILE_SIZE_M) * ceil_div(N, TILE_SIZE_N),
            )
            grid = (grid_size,)
            ct.launch(torch.cuda.current_stream(), grid, rms_norm_kernel_static_persistent, (
                x,
                y,
                weight,
                TILE_SIZE_M,
                TILE_SIZE_N,
                eps,
            ))
        else:
            # Standard RMSNorm kernel
            rstd = torch.empty((M,), dtype=torch.float32, device='cuda')
            MAX_FUSED_SIZE = 2048 // x.element_size()
            TILE_SIZE = min(MAX_FUSED_SIZE, next_power_of_2(N))
            grid = (M,)
            kernel = rms_norm_kernel_gather if gather else rms_norm_kernel
            ct.launch(torch.cuda.current_stream(), grid, kernel, (
                x,
                weight,
                y,
                rstd,
                N,
                eps,
                TILE_SIZE,
            ))
        return y.view(*x.shape)

    @cuda_tile_rms_norm.register_fake
    def _(
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        static_persistent: bool,
        gather: bool,
    ) -> torch.Tensor:
        return torch.empty_like(x.contiguous())
