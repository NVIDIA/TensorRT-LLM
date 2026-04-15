# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""
Symmetric Memory AllGather

This module provides PyTorch Symmetric Memory-based allgather operations,
leveraging MULTIMEM hardware instructions (multimem_all_gather_out) for
low-latency allgather on NVSwitch-connected GPUs.

CUDA Graph compatible: workspace is pre-allocated during init so no
allocations occur during graph capture/replay.
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


class SymmetricMemoryAllGather(nn.Module):
    """
    AllGather implementation using PyTorch's symmetric memory operations.

    Uses ``torch.ops.symm_mem.multimem_all_gather_out`` which leverages
    NVSwitch multicast for single-kernel, hardware-accelerated allgather.

    The gathered output lives inside a pre-allocated symmetric-memory
    workspace so the op is safe to use inside CUDA Graph capture.
    """

    MiB = 1024 * 1024

    # Max workspace sizes (bytes) per device capability and world size.
    # These mirror SymmetricMemoryAllReduce limits; the allgather output
    # buffer lives in the *same* p2p workspace pool so we stay within HW
    # multicast limits.
    _MAX_SIZES = {
        "9.0": {
            2: 64 * MiB,
            4: 32 * MiB,
            6: 64 * MiB,
            8: 64 * MiB,
        },
        "10.0": {
            2: 8 * MiB,
            4: 32 * MiB,
            6: 128 * MiB,
            8: 128 * MiB,
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
        self.group_name_str: Optional[str] = None

        if not SYMM_MEM_AVAILABLE:
            logger.warning("SymmetricMemoryAllGather: PyTorch symm_mem not available")
            return

        if not torch.cuda.is_available():
            logger.warning("SymmetricMemoryAllGather: CUDA not available")
            return

        # Device capability
        device = torch.device(f"cuda:{mapping.tp_rank}")
        capability = torch.cuda.get_device_capability(device)
        self.device_capability = f"{capability[0]}.{capability[1]}"

        if self.device_capability not in self._MAX_SIZES:
            logger.warning(
                f"SymmetricMemoryAllGather: Device capability {self.device_capability} not supported"
            )
            return

        if self.world_size not in self._MAX_SIZES[self.device_capability]:
            logger.info(
                f"SymmetricMemoryAllGather: World size {self.world_size} not supported "
                f"for SM {self.device_capability}"
            )
            return

        self.max_size = self._MAX_SIZES[self.device_capability][self.world_size]

        # Process group
        self.group = group
        if self.group is None:
            if not dist.is_initialized():
                logger.warning("SymmetricMemoryAllGather: torch.distributed not initialized")
                return
            tp_group_ranks = mapping.tp_group
            self.group = dist.new_group(tp_group_ranks) if len(tp_group_ranks) > 1 else None

        try:
            if not hasattr(self.group, "group_name"):
                logger.warning(
                    "SymmetricMemoryAllGather: ProcessGroup does not have group_name attribute"
                )
                return

            self.group_name_str = str(self.group.group_name)
            torch_symm_mem.enable_symm_mem_for_group(self.group_name_str)
        except Exception as e:
            logger.warning(
                f"SymmetricMemoryAllGather: Failed to enable symmetric memory for group: {e}"
            )
            return

        # Pre-allocate workspace so CUDA Graph capture won't trigger allocation.
        # The workspace must hold the *full* gathered output (world_size x shard).
        try:
            torch_symm_mem.get_symm_mem_workspace(
                self.group_name_str,
                min_size=self.max_size,
            )
            self.disabled = False
            logger.info(
                f"SymmetricMemoryAllGather (MULTIMEM) initialized: "
                f"world_size={self.world_size}, "
                f"max_workspace={self.max_size}, "
                f"SM={self.device_capability}"
            )
        except Exception as e:
            logger.warning(f"SymmetricMemoryAllGather workspace pre-allocation failed: {e}")

    def can_use_symm_mem(self, inp: torch.Tensor, dim: int = 0) -> bool:
        """Check whether this tensor can be gathered with symm_mem."""
        if self.disabled:
            return False
        if inp.dtype != self.dtype:
            return False
        # Normalize negative dim.
        ndim = inp.ndim
        if dim < 0:
            dim = ndim + dim
        if dim < 0 or dim >= ndim:
            return False
        # Output = world_size * inp (gathered along the specified dim).
        out_bytes = inp.numel() * self.world_size * inp.element_size()
        if out_bytes > self.max_size:
            return False
        return True

    def _allgather_dim0(self, inp: torch.Tensor) -> torch.Tensor:
        """Core allgather along dim-0 using multimem_all_gather_out."""
        out_shape = list(inp.shape)
        out_shape[0] = inp.shape[0] * self.world_size

        symm_mem = torch_symm_mem.get_symm_mem_workspace(
            self.group_name_str,
            min_size=inp.numel() * self.world_size * inp.element_size(),
        )
        out = symm_mem.get_buffer(
            symm_mem.rank,
            torch.Size(out_shape),
            inp.dtype,
        )
        torch.ops.symm_mem.multimem_all_gather_out(
            inp.contiguous(), self.group_name_str, out
        )
        return out.clone()

    def forward(
        self,
        inp: torch.Tensor,
        dim: int = 0,
    ) -> Optional[torch.Tensor]:
        """
        Perform allgather using multimem_all_gather_out.

        Supports any gather dimension by transposing to dim-0 internally.
        Returns the gathered tensor, or ``None`` if this call cannot be
        served by symm_mem (caller should fall back to NCCL).
        """
        if not self.can_use_symm_mem(inp, dim):
            return None

        # Normalize dim.
        if dim < 0:
            dim = inp.ndim + dim

        if dim == 0:
            return self._allgather_dim0(inp)

        # For dim != 0: move the gather dim to position 0,
        # allgather along dim-0, then move it back.
        inp_t = inp.transpose(0, dim).contiguous()
        gathered = self._allgather_dim0(inp_t)
        return gathered.transpose(0, dim).contiguous()
