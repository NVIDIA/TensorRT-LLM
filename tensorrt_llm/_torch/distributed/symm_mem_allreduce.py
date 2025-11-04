# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""
Symmetric Memory AllReduce for H100+ GPUs

This module provides PyTorch Symmetric Memory-based allreduce operations,
leveraging H100's MULTIMEM hardware instructions for 3x faster performance
compared to custom CUDA kernels on supported configurations.
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

    This leverages H100's MULTIMEM hardware instructions for significantly faster
    allreduce operations compared to software implementations.

    Supported configurations (world_size):
    - SM 9.0 (H100): 4, 6, 8 GPUs
    - SM 10.0 (future): 6, 8 GPUs

    Based on vLLM's implementation but integrated into TensorRT-LLM.
    """

    # World sizes that support MULTIMEM instructions
    _WORLD_SIZES_MULTIMEM = {
        "9.0": [4, 6, 8],  # H100
        "10.0": [6, 8],  # Future architectures
    }

    # Maximum buffer sizes for symmetric memory (bytes)
    _MAX_SIZES = {
        "9.0": {
            4: 8 * 1024 * 1024,  # 8MB for 4 GPUs
            6: 6 * 1024 * 1024,  # 6MB for 6 GPUs
            8: 4 * 1024 * 1024,  # 4MB for 8 GPUs
        },
        "10.0": {
            6: 8 * 1024 * 1024,
            8: 6 * 1024 * 1024,
        }
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
            logger.warning(
                "SymmetricMemoryAllReduce: PyTorch symm_mem not available")
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
                f"for SM {self.device_capability}")
            return

        # Get max buffer size for this configuration
        self.max_size = self._MAX_SIZES[self.device_capability][self.world_size]

        # Set up process group
        if group is None:
            # Get or create TP group with correct ranks
            # For TP parallelism, we need ranks [0, 1, 2, ..., tp_size-1] globally
            # NOT starting from tp_rank!
            if not dist.is_initialized():
                logger.warning(
                    "SymmetricMemoryAllReduce: torch.distributed not initialized"
                )
                self.disabled = True
                return

            # Assume contiguous TP ranks for now
            # TODO: Get actual TP group from mapping if available
            tp_group_ranks = list(range(mapping.tp_size))
            self.group = dist.new_group(tp_group_ranks) if len(
                tp_group_ranks) > 1 else None
        else:
            self.group = group

        if self.group is None:
            logger.warning("SymmetricMemoryAllReduce: No valid process group")
            self.disabled = True
            return

        # Allocate symmetric memory buffer
        try:
            self.buffer = torch_symm_mem.empty(
                self.max_size // self.dtype.itemsize,
                device=device,
                dtype=self.dtype,
            )
            # Pass group_name (string) not the group object
            handle = torch_symm_mem.rendezvous(self.buffer,
                                               self.group.group_name)

            if handle.multicast_ptr == 0:
                logger.warning(
                    "SymmetricMemoryAllReduce: MULTIMEM operations not supported (multicast_ptr is 0)"
                )
                return

            # Determine which algorithm to use
            self.use_multimem = (self.world_size
                                 in self._WORLD_SIZES_MULTIMEM.get(
                                     self.device_capability, []))

            self.disabled = False
            logger.info(f"SymmetricMemoryAllReduce initialized: "
                        f"world_size={self.world_size}, "
                        f"max_size={self.max_size}, "
                        f"SM={self.device_capability}, "
                        f"use_multimem={self.use_multimem}")

        except Exception as e:
            logger.warning(
                f"SymmetricMemoryAllReduce initialization failed: {e}")
            return

    def should_use_symm_mem(self, inp: torch.Tensor) -> bool:
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
        if not self.should_use_symm_mem(inp):
            return None  # Caller should fall back to other strategy

        if out is None:
            out = torch.empty_like(inp)

        # Copy input to symmetric memory buffer
        self.buffer[:inp.numel()].copy_(inp.view(-1))

        # Perform allreduce using appropriate algorithm
        if self.use_multimem:
            # Use MULTIMEM hardware instructions (faster)
            torch.ops.symm_mem.multimem_all_reduce_(
                self.buffer[:inp.numel()],
                "sum",
                self.group.group_name,
            )
        else:
            # Use two-shot algorithm (fallback)
            torch.ops.symm_mem.two_shot_all_reduce_(
                self.buffer[:inp.numel()],
                "sum",
                self.group.group_name,
            )

        # Copy result back
        out.copy_(self.buffer[:inp.numel()].view(out.shape))

        return out
