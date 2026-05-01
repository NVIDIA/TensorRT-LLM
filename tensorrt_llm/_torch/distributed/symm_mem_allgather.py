# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""
Symmetric Memory AllGather

This module provides PyTorch Symmetric Memory-based allgather operations,
leveraging MULTIMEM hardware instructions (multimem_all_gather_out) for
low-latency allgather on NVSwitch-connected GPUs.

CUDA Graph compatible: the symm_mem workspace is acquired once during init
and reused on every forward, so no allocations occur during graph
capture/replay.
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

    Supported configurations (world_size):
    - SM 9.0: 4, 6, 8 GPUs
    - SM 10.0: 6, 8 GPUs

    Any gather dim is supported via transpose-to-dim-0 in :py:meth:`forward`.
    Large outputs on the dim != 0 path fall back to NCCL via
    ``_SYMM_MEM_THRESHOLD`` because the transpose+contiguous copies
    erode the MULTIMEM latency advantage on prefill-sized tensors.
    """

    MiB = 1024 * 1024

    # World sizes that actually have MULTIMEM hardware support; mirrors
    # SymmetricMemoryAllReduce so the two ops gate identically.
    _WORLD_SIZES_MULTIMEM = {
        "9.0": [4, 6, 8],
        "10.0": [6, 8],
    }

    # Max workspace sizes (bytes) per device capability and world size.
    # Only MULTIMEM-capable world sizes are listed; entries match
    # SymmetricMemoryAllReduce's limits but the workspace is independent
    # (allocated through get_symm_mem_workspace, not the AllReduce buffer).
    _MAX_SIZES = {
        "9.0": {
            4: 32 * MiB,
            6: 64 * MiB,
            8: 64 * MiB,
        },
        "10.0": {
            6: 128 * MiB,
            8: 128 * MiB,
        },
    }

    # Output-bytes threshold above which the dim != 0 (transpose) path falls
    # back to NCCL. The transpose+contiguous copies plus the multimem
    # all-at-once latency stop beating NCCL ring-LL's pipelined bandwidth —
    # and on paths where multi-stream overlap is disabled (e.g. piecewise
    # CG under disable_multi_stream), the multimem call ends up fully
    # exposed on the critical path. Falling back to NCCL in that regime
    # trades ~µs of latency for several ms of saved transpose + better
    # pipelining on prefill-sized tensors.
    # Measured on B200/H100 ws=8 piecewise CG with multi_stream disabled
    # (#13321). Cross-over is weakly ws-dependent (~17% via the (ws-1)/ws
    # factor), so a single scalar covers all measured (cap, ws) configs.
    # The dim == 0 path is unaffected (no transpose), and is gated only
    # by the workspace overflow check (_MAX_SIZES).
    _SYMM_MEM_THRESHOLD = 1 * MiB

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
        self._symm_mem = None

        if not SYMM_MEM_AVAILABLE:
            logger.warning("SymmetricMemoryAllGather: PyTorch symm_mem not available")
            return

        if not torch.cuda.is_available():
            logger.warning("SymmetricMemoryAllGather: CUDA not available")
            return

        # The executor has already pinned this process to the right GPU
        # via torch.cuda.set_device(); use that rather than re-deriving an
        # ordinal from the TP/local rank (which breaks on multi-node).
        capability = torch.cuda.get_device_capability(torch.cuda.current_device())
        self.device_capability = f"{capability[0]}.{capability[1]}"

        if self.device_capability not in self._MAX_SIZES:
            logger.warning(
                f"SymmetricMemoryAllGather: Device capability {self.device_capability} not supported"
            )
            return

        # Gate by MULTIMEM-supported world sizes (mirrors AllReduce).
        if self.world_size not in self._WORLD_SIZES_MULTIMEM.get(self.device_capability, []):
            logger.info(
                f"SymmetricMemoryAllGather: MULTIMEM not supported for "
                f"world_size={self.world_size}, SM={self.device_capability}"
            )
            return

        if self.world_size not in self._MAX_SIZES[self.device_capability]:
            logger.info(
                f"SymmetricMemoryAllGather: World size {self.world_size} not supported "
                f"for SM {self.device_capability}"
            )
            return

        self.max_size = self._MAX_SIZES[self.device_capability][self.world_size]

        # Process group. NOTE: a group created here is left alive for the
        # module's lifetime — torch does not provide a cheap, safe way to
        # tear it down on partial init failure, and the cache in
        # auto_deploy/.../trtllm_dist.py keeps this module long-lived.
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

        # Cross-node rendezvous probe.
        #
        # `get_symm_mem_workspace` below returns a pooled workspace handle
        # whose `multicast_ptr` can read as non-zero locally even when the
        # group spans multiple NVSwitch domains (cross-node TP). The first
        # `multimem_all_gather_out` would then hang on the implicit
        # rendezvous instead of letting the caller fall back to NCCL.
        #
        # Mirror SymmetricMemoryAllReduce: allocate a small symm_mem buffer
        # and call `torch_symm_mem.rendezvous`, which is a blocking
        # cross-rank handshake. If MULTIMEM isn't actually usable the
        # handle's `multicast_ptr` comes back as 0 and we disable here
        # rather than at runtime.
        try:
            probe = torch_symm_mem.empty(
                1,
                device=torch.device(f"cuda:{torch.cuda.current_device()}"),
                dtype=torch.uint8,
            )
            rendezvous_handle = torch_symm_mem.rendezvous(probe, self.group_name_str)
            if rendezvous_handle.multicast_ptr == 0:
                logger.warning(
                    "SymmetricMemoryAllGather: MULTIMEM not available after "
                    "rendezvous (multicast_ptr is 0) — disabling"
                )
                return
        except Exception as e:
            logger.warning(f"SymmetricMemoryAllGather: rendezvous probe failed: {e}")
            return

        # Acquire the workspace once and verify MULTIMEM is actually
        # available. Holding the handle as a member guarantees forward()
        # never re-enters get_symm_mem_workspace(), so there is no chance
        # of reallocation during CUDA Graph capture.
        try:
            self._symm_mem = torch_symm_mem.get_symm_mem_workspace(
                self.group_name_str,
                min_size=self.max_size,
            )
            if getattr(self._symm_mem, "multicast_ptr", 0) == 0:
                logger.warning(
                    "SymmetricMemoryAllGather: MULTIMEM not available "
                    "(multicast_ptr is 0) — disabling"
                )
                self._symm_mem = None
                return

            self.disabled = False
            logger.info(
                f"SymmetricMemoryAllGather (MULTIMEM) initialized: "
                f"world_size={self.world_size}, "
                f"max_workspace={self.max_size}, "
                f"SM={self.device_capability}"
            )
        except Exception as e:
            logger.warning(f"SymmetricMemoryAllGather: workspace pre-allocation failed: {e}")

    @staticmethod
    def _normalize_dim(dim: int, ndim: int) -> Optional[int]:
        """Normalize a possibly-negative dim. Returns ``None`` if out of range."""
        if dim < 0:
            dim = ndim + dim
        if dim < 0 or dim >= ndim:
            return None
        return dim

    def can_use_symm_mem(self, inp: torch.Tensor, dim: int = 0) -> bool:
        """Check whether this tensor can be gathered with symm_mem.

        Non-zero gather dims are supported by transposing the gather dim to
        position 0 in :py:meth:`forward`, performing the dim-0 multimem
        allgather, and transposing back.
        """
        if self.disabled:
            return False
        if inp.dtype != self.dtype:
            return False
        dim = self._normalize_dim(dim, inp.ndim)
        if dim is None:
            return False
        # multimem_all_gather_out requires 4B-aligned input.
        inp_bytes = inp.numel() * inp.element_size()
        if inp_bytes % 4 != 0:
            return False
        # Output = world_size * inp; must fit in pre-allocated workspace.
        # Use >= to match SymmetricMemoryAllReduce's bound.
        out_bytes = inp_bytes * self.world_size
        if out_bytes >= self.max_size:
            return False
        # On the dim != 0 (transpose) path, gate by the perf threshold so
        # that large tensors (typical of prefill) fall back to NCCL where
        # ring-LL pipelining beats multimem-with-transpose. dim == 0 is
        # unaffected.
        if dim != 0 and out_bytes >= self._SYMM_MEM_THRESHOLD:
            return False
        return True

    def _allgather_dim0(self, inp: torch.Tensor) -> torch.Tensor:
        """Core allgather along dim-0 using multimem_all_gather_out."""
        out_shape = list(inp.shape)
        out_shape[0] = inp.shape[0] * self.world_size

        out = self._symm_mem.get_buffer(
            self._symm_mem.rank,
            torch.Size(out_shape),
            inp.dtype,
        )
        torch.ops.symm_mem.multimem_all_gather_out(inp.contiguous(), self.group_name_str, out)
        # Clone so the caller can safely outlive the next forward (which
        # would otherwise overwrite this same workspace slice).
        return out.clone()

    def forward(
        self,
        inp: torch.Tensor,
        dim: int = 0,
    ) -> Optional[torch.Tensor]:
        """
        Perform allgather using multimem_all_gather_out.

        Supports any gather dimension by transposing the gather dim to
        position 0, running the dim-0 multimem allgather, and transposing
        back. Returns the gathered tensor, or ``None`` if this call cannot
        be served by symm_mem (caller should fall back to NCCL).
        """
        if not self.can_use_symm_mem(inp, dim):
            return None

        dim = self._normalize_dim(dim, inp.ndim)
        assert dim is not None, "dim already validated by can_use_symm_mem"

        if dim == 0:
            return self._allgather_dim0(inp)

        # For dim != 0: move the gather dim to position 0,
        # allgather along dim-0, then move it back.
        inp_t = inp.transpose(0, dim).contiguous()
        gathered = self._allgather_dim0(inp_t)
        return gathered.transpose(0, dim).contiguous()
