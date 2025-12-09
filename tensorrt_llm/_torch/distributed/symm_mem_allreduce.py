# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""
Symmetric Memory AllReduce

This module provides PyTorch Symmetric Memory-based allreduce operations,
leveraging MULTIMEM hardware instructions.
"""

from typing import Optional

import torch
import torch.distributed as dist
from torch import nn

from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

try:
    import torch.distributed._symmetric_memory as torch_symm_mem

    SYMM_MEM_AVAILABLE = True
except ImportError:
    SYMM_MEM_AVAILABLE = False
    logger.warning(
        "PyTorch symmetric memory not available. Install PyTorch >= 2.8 for MULTIMEM support."
    )


class SymmetricMemoryAllReduce(nn.Module):
    """
    AllReduce implementation using PyTorch's symmetric memory operations.
    This leverages MULTIMEM hardware instructions for faster allreduce operations.

    Supported configurations (world_size):
    - SM 9.0: 4, 6, 8 GPUs
    - SM 10.0: 6, 8 GPUs

    """

    # World sizes that support MULTIMEM instructions
    _WORLD_SIZES_MULTIMEM = {
        "9.0": [4, 6, 8],
        "10.0": [6, 8],
    }

    MiB = 1024 * 1024
    # Maximum buffer sizes for symmetric memory (bytes)
    _MAX_SIZES = {
        "9.0": {
            2: 64 * MiB,  # 64 MB
            4: 32 * MiB,  # 32 MB
            6: 64 * MiB,  # 64 MB
            8: 64 * MiB,  # 64 MB
        },
        "10.0": {
            2: 8 * MiB,  # 8 MB
            4: 32 * MiB,  # 32 MB
            6: 128 * MiB,  # 128 MB
            8: 128 * MiB,  # 128 MB
        },
    }

    def __init__(
        self,
        mapping: Mapping,
        dtype: torch.dtype = torch.bfloat16,
        group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()

        self.disabled = True
        self.mapping = mapping
        self.dtype = dtype
        self.world_size = mapping.tp_size

        if not SYMM_MEM_AVAILABLE:
            logger.warning("SymmetricMemoryAllReduce: PyTorch symm_mem not available")
            return

        if not torch.cuda.is_available():
            logger.warning("SymmetricMemoryAllReduce: CUDA not available")
            return

        # Get device capability
        device = torch.device(f"cuda:{mapping.tp_rank}")
        capability = torch.cuda.get_device_capability(device)
        self.device_capability = f"{capability[0]}.{capability[1]}"

        # Check if this configuration is supported
        if self.device_capability not in self._MAX_SIZES:
            logger.warning(
                f"SymmetricMemoryAllReduce: Device capability {self.device_capability} not supported"
            )
            return

        if self.world_size not in self._MAX_SIZES[self.device_capability]:
            logger.info(
                f"SymmetricMemoryAllReduce: World size {self.world_size} not supported "
                f"for SM {self.device_capability}"
            )
            return

        # Get max buffer size for this configuration
        self.max_size = self._MAX_SIZES[self.device_capability][self.world_size]

        # Set up process group
        self.group = group
        if self.group is None:
            # Get or create TP group with correct ranks
            # For TP parallelism, we need ranks [0, 1, 2, ..., tp_size-1] globally
            # NOT starting from tp_rank!
            if not dist.is_initialized():
                logger.warning("SymmetricMemoryAllReduce: torch.distributed not initialized")
                self.disabled = True
                return
            # Get actual TP group ranks from mapping (tp_group is a property, not a method)
            tp_group_ranks = mapping.tp_group
            self.group = dist.new_group(tp_group_ranks) if len(tp_group_ranks) > 1 else None

        # Enable symmetric memory for this group
        try:
            # Get group_name - this may fail if ProcessGroup doesn't have group_name set
            if not hasattr(self.group, "group_name"):
                logger.warning(
                    "SymmetricMemoryAllReduce: ProcessGroup does not have group_name attribute"
                )
                self.disabled = True
                return

            group_name_str = str(self.group.group_name)
            torch_symm_mem.enable_symm_mem_for_group(group_name_str)
            logger.debug(
                f"SymmetricMemoryAllReduce: Enabled symmetric memory for group {group_name_str}"
            )
        except Exception as e:
            logger.warning(
                f"SymmetricMemoryAllReduce: Failed to enable symmetric memory for group: {e}"
            )
            self.disabled = True
            return

        # Allocate symmetric memory buffer
        try:
            self.buffer = torch_symm_mem.empty(
                self.max_size // self.dtype.itemsize,
                device=device,
                dtype=self.dtype,
            )
            # Pass group name string
            group_name_str = str(self.group.group_name)
            handle = torch_symm_mem.rendezvous(self.buffer, group_name_str)

            if handle.multicast_ptr == 0:
                logger.warning(
                    "SymmetricMemoryAllReduce: MULTIMEM operations not supported (multicast_ptr is 0)"
                )
                return

            # Only enable if MULTIMEM is supported
            # Otherwise, no benefit over existing TensorRT-LLM strategies
            use_multimem = self.world_size in self._WORLD_SIZES_MULTIMEM.get(
                self.device_capability, []
            )

            if not use_multimem:
                logger.info(
                    f"SymmetricMemoryAllReduce: MULTIMEM not supported for "
                    f"world_size={self.world_size}, SM={self.device_capability}. "
                    f"Falling back to standard allreduce strategies."
                )
                return

            self.disabled = False
            logger.info(
                f"SymmetricMemoryAllReduce (MULTIMEM) initialized: "
                f"world_size={self.world_size}, "
                f"max_size={self.max_size}, "
                f"SM={self.device_capability}"
            )

        except Exception as e:
            logger.warning(f"SymmetricMemoryAllReduce initialization failed: {e}")
            return

    @property
    def process_group(self) -> Optional[dist.ProcessGroup]:
        """Expose the ProcessGroup for use in fallback scenarios."""
        return self.group if not self.disabled else None

    def can_use_symm_mem(self, inp: torch.Tensor) -> bool:
        """Check if symmetric memory can be used for this tensor."""
        if self.disabled:
            return False
        if inp.dtype != self.dtype:
            return False
        inp_size = inp.numel() * inp.element_size()
        if inp_size % 4 != 0:
            return False
        if inp_size >= self.max_size:
            return False
        return True

    def forward(
        self,
        inp: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform allreduce using symmetric memory operations.

        Args:
            inp: Input tensor to reduce
            out: Optional output tensor (if None, will be allocated)

        Returns:
            Reduced tensor
        """
        if not self.can_use_symm_mem(inp):
            return None  # Caller should fall back to other strategy

        if out is None:
            out = torch.empty_like(inp)

        # Copy input to symmetric memory buffer
        self.buffer[: inp.numel()].copy_(inp.view(-1))

        # Perform MULTIMEM allreduce
        # Pass group name string (matching vLLM's implementation)
        group_name_str = str(self.group.group_name)
        torch.ops.symm_mem.multimem_all_reduce_(
            self.buffer[: inp.numel()],
            "sum",
            group_name_str,
        )

        # Copy result back
        out.copy_(self.buffer[: inp.numel()].view(out.shape))

        return out
