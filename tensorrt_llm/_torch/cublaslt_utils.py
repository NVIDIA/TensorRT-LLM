from ..logger import logger

IS_CUBLASLT_AVAILABLE = False

# Check cuBLASLt availability
try:
    import torch

    # Check if cublas_fp4_scaled_mm is available
    if hasattr(torch.ops.trtllm, 'cublas_fp4_scaled_mm'):
        logger.info(f"cuBLASLt FP4 GEMM is available")
        IS_CUBLASLT_AVAILABLE = True
except ImportError:
    pass
