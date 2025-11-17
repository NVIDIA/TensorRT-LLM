"""Backend abstraction for distributed operations.

This module provides a strategy pattern for selecting between different distributed backends:
- TorchDistBackend: Uses PyTorch's native distributed operations (demollm mode)
- TRTLLMBackend: Uses TensorRT-LLM optimized operations (MPI mode with TRT-LLM)

The backend is auto-detected based on availability and runtime environment, but can
also be explicitly set for testing or specific use cases.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

import torch

from .common import ReduceOp


class DistributedBackend(ABC):
    """Abstract base class for distributed backends.

    All distributed backends must implement these core operations.
    """

    @abstractmethod
    def all_gather(
        self, tensor: torch.Tensor, dim: int = 0, sizes: Optional[List[int]] = None
    ) -> torch.Tensor:
        """All gather followed by concat in the specified dimension.

        Args:
            tensor: Input tensor to gather
            dim: Dimension along which to concatenate gathered tensors
            sizes: Optional list of sizes for uneven splits

        Returns:
            Concatenated tensor containing data from all ranks
        """
        pass

    @abstractmethod
    def all_reduce(self, tensor: torch.Tensor, op: ReduceOp) -> torch.Tensor:
        """All reduce operation across all ranks.

        Args:
            tensor: Input tensor to reduce
            op: Reduction operation (e.g., SUM, MAX)

        Returns:
            Reduced tensor
        """
        pass

    @abstractmethod
    def fused_linear_all_reduce(
        self, input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Fused linear layer followed by all_reduce.

        Args:
            input: Input tensor
            weight: Weight matrix
            bias: Optional bias vector

        Returns:
            Output after linear operation and all_reduce
        """
        pass

    @abstractmethod
    def fused_fp8_linear_all_reduce(
        self,
        input: torch.Tensor,
        weight_fp8: torch.Tensor,
        bias: Optional[torch.Tensor],
        input_scale: Optional[torch.Tensor],
        weight_scale: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Fused FP8 linear layer followed by all_reduce.

        Args:
            input: Input tensor
            weight_fp8: FP8 quantized weight matrix
            bias: Optional bias vector
            input_scale: Optional input scaling factor
            weight_scale: Optional weight scaling factor

        Returns:
            Output after FP8 linear operation and all_reduce
        """
        pass

    @abstractmethod
    def fused_allreduce_residual_rmsnorm(
        self,
        tensor: torch.Tensor,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fused all_reduce, residual add, and RMSNorm.

        Args:
            tensor: Input tensor to all_reduce
            residual: Residual tensor to add
            norm_weight: RMSNorm weight
            eps: RMSNorm epsilon

        Returns:
            Tuple of (normed_output, tensor_with_residual)
        """
        pass


class TorchDistBackend(DistributedBackend):
    """PyTorch distributed backend implementation.

    This backend uses PyTorch's native distributed operations and is used in
    demollm mode or when TRT-LLM ops are not available.
    """

    def all_gather(
        self, tensor: torch.Tensor, dim: int = 0, sizes: Optional[List[int]] = None
    ) -> torch.Tensor:
        from . import common as dist

        tl = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(tl, tensor)
        return torch.cat(tl, dim=dim)

    def all_reduce(self, tensor: torch.Tensor, op: ReduceOp) -> torch.Tensor:
        from . import common as dist

        t_res = tensor.clone()
        dist.all_reduce(t_res, op=op)
        return t_res

    def fused_linear_all_reduce(
        self, input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        from . import common as dist

        output = torch.ops.aten.linear(input, weight, bias)
        dist.all_reduce(output, op=ReduceOp.SUM)
        return output

    def fused_fp8_linear_all_reduce(
        self,
        input: torch.Tensor,
        weight_fp8: torch.Tensor,
        bias: Optional[torch.Tensor],
        input_scale: Optional[torch.Tensor],
        weight_scale: Optional[torch.Tensor],
    ) -> torch.Tensor:
        from . import common as dist

        # Use the FP8 linear op
        out = torch.ops.auto_deploy.torch_quant_fp8_linear(
            input, weight_fp8, bias, input_scale, weight_scale
        )
        dist.all_reduce(out, op=ReduceOp.SUM)
        return out

    def fused_allreduce_residual_rmsnorm(
        self,
        tensor: torch.Tensor,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from .common import all_reduce

        # Fallback: unfused implementation using torch distributed
        # This is used in demollm mode without MPI
        # 1. All-reduce the tensor
        tensor_reduced = tensor.clone()
        all_reduce(tensor_reduced, op=ReduceOp.SUM)

        # 2. Add residual
        tensor_with_residual = tensor_reduced + residual

        # 3. Apply RMSNorm using PyTorch's built-in function
        norm_out = torch.nn.functional.rms_norm(
            tensor_with_residual,
            normalized_shape=(tensor_with_residual.size(-1),),
            weight=norm_weight,
            eps=eps,
        )

        return norm_out, tensor_with_residual


class TRTLLMBackend(DistributedBackend):
    """TensorRT-LLM optimized backend implementation.

    This backend uses TRT-LLM's optimized distributed operations for improved
    tensor parallel performance. Only available when running with MPI.
    """

    def all_gather(
        self, tensor: torch.Tensor, dim: int = 0, sizes: Optional[List[int]] = None
    ) -> torch.Tensor:
        from . import trtllm as trtllm_dist

        return trtllm_dist.trtllm_allgather(tensor, dim=dim, sizes=sizes)

    def all_reduce(self, tensor: torch.Tensor, op: ReduceOp) -> torch.Tensor:
        from . import trtllm as trtllm_dist

        return trtllm_dist.trtllm_allreduce(tensor, op=op)

    def fused_linear_all_reduce(
        self, input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        from . import trtllm as trtllm_dist

        output = torch.ops.aten.linear(input, weight, bias)
        return trtllm_dist.trtllm_allreduce(output, op=ReduceOp.SUM)

    def fused_fp8_linear_all_reduce(
        self,
        input: torch.Tensor,
        weight_fp8: torch.Tensor,
        bias: Optional[torch.Tensor],
        input_scale: Optional[torch.Tensor],
        weight_scale: Optional[torch.Tensor],
    ) -> torch.Tensor:
        from . import trtllm as trtllm_dist

        # Use the FP8 linear op
        out = torch.ops.auto_deploy.torch_quant_fp8_linear(
            input, weight_fp8, bias, input_scale, weight_scale
        )
        return trtllm_dist.trtllm_allreduce(out, op=ReduceOp.SUM)

    def fused_allreduce_residual_rmsnorm(
        self,
        tensor: torch.Tensor,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Import here to avoid circular dependency
        from ...modules.linear import AllReduceFusionOp, AllReduceParams
        from . import trtllm as trtllm_dist

        all_reduce_params = AllReduceParams(
            fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
            bias=None,
            residual=residual,
            norm_weight=norm_weight,
            eps=eps,
        )
        return trtllm_dist.trtllm_allreduce(
            tensor, ReduceOp.SUM, all_reduce_params=all_reduce_params
        )


class BackendManager:
    """Singleton manager for distributed backend selection.

    The backend is lazily initialized on first access. By default, it auto-detects
    the appropriate backend based on TRT-LLM availability and MPI mode. The backend
    can be explicitly set for testing or specific use cases.
    """

    _instance: Optional["BackendManager"] = None
    _backend: Optional[DistributedBackend] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def set_backend(self, backend: DistributedBackend):
        """Explicitly set the backend.

        Args:
            backend: The backend instance to use
        """
        self._backend = backend

    def get_backend(self) -> DistributedBackend:
        """Get the current backend, auto-detecting if not set.

        Returns:
            The active distributed backend
        """
        if self._backend is None:
            self._backend = self._auto_detect_backend()
        return self._backend

    def _auto_detect_backend(self) -> DistributedBackend:
        """Auto-detect which backend to use based on runtime environment.

        Returns:
            TRTLLMBackend if TRT-LLM ops are available and running with MPI,
            otherwise TorchDistBackend
        """
        from .trtllm import is_trtllm_op_available

        if is_trtllm_op_available():
            return TRTLLMBackend()
        return TorchDistBackend()

    def reset(self):
        """Reset backend to force re-detection.

        This is primarily useful for testing.
        """
        self._backend = None


# Global backend manager instance
_backend_manager = BackendManager()


def get_dist_backend() -> DistributedBackend:
    """Get the current distributed backend.

    Returns:
        The active distributed backend instance
    """
    return _backend_manager.get_backend()


def set_dist_backend(backend: DistributedBackend):
    """Set the distributed backend explicitly.

    Args:
        backend: The backend instance to use
    """
    _backend_manager.set_backend(backend)


def reset_dist_backend():
    """Reset the backend to force re-detection.

    This is primarily useful for testing scenarios.
    """
    _backend_manager.reset()
