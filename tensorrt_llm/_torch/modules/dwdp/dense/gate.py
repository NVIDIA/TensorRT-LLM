# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Dense DWDP launch gate: what a dense DWDP launch needs from the host, before the model exists.

``TorchLlmArgs.validate_dwdp_launch`` refuses, when the arguments are built, what the arguments
alone rule out.  ``py_executor_creator`` calls :func:`validate_dense_gate` on each worker, before it
constructs any DWDP or dense DWDP manager and the model, for what only the host can tell, so an
unsupported launch fails before the weights load rather than in setup or in the first forward.
The gate also decides how dense DWDP runs: with expert DWDP (attached) or without it, standalone
on an attention-DP instance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

from tensorrt_llm._utils import mpi_disabled
from tensorrt_llm.logger import logger

from .. import fabric_gather, vmm
from .config import DenseConfig
from .types import LOG_PREFIX, CopyMode, DenseMode

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs


def validate_dense_gate(llm_args: TorchLlmArgs) -> DenseMode:
    """Check what dense DWDP needs from the host, before the model is built; return the mode.

    Called only when ``dwdp_config.enable_dense`` is set, on arguments that
    ``TorchLlmArgs.validate_dwdp_launch`` accepted.  The checks, in this order:

    1. FABRIC peer handles (``vmm.peer_handle_type()``): the row store is shared only through
       FABRIC handles.
    2. Standalone only: the MPI backend (``TLLM_DISABLE_MPI`` unset), since the dense group is an
       MPI sub-communicator over the attention-DP group; and with ``dwdp_config.copy_mode`` gather
       a registered ``trtllm::dwdp_fabric_gather``.  A standalone gather launch that passes logs
       one warning, which names the compute-side defaults setup applies for the gather and the
       setting that picks its kernel.  With expert DWDP, ``copy_mode`` sets the expert transfer,
       which the dense copies follow, and ``DwdpManager`` checks the gather op.

    Args:
        llm_args: The launch configuration.

    Returns:
        ``DenseMode.ATTACHED`` with ``dwdp_config.enable_experts``, else
        ``DenseMode.STANDALONE``.

    Raises:
        ValueError: Naming the first violated requirement.
    """
    handle_type = vmm.peer_handle_type()
    if handle_type != vmm.cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC:
        _reject(
            "FABRIC peer handles (CU_MEM_HANDLE_TYPE_FABRIC)",
            f"this host shares {handle_type.name} handles",
        )
    if llm_args.dwdp_config.enable_experts:
        return DenseMode.ATTACHED
    _check_standalone(DenseConfig.from_dwdp_config(llm_args.dwdp_config).standalone_copy_mode())
    return DenseMode.STANDALONE


def _check_standalone(copy_mode: CopyMode) -> None:
    """The standalone-only requirements (step 2), after the common one."""
    if mpi_disabled():
        _reject(
            "the MPI backend (the standalone dense group is an MPI sub-communicator over the "
            "attention-DP group)",
            "TLLM_DISABLE_MPI=1",
        )
    if copy_mode is not CopyMode.GATHER:
        return
    if fabric_gather.gather_op() is None:
        _reject(
            "trtllm::dwdp_fabric_gather for dwdp_config.copy_mode='gather'",
            "this build does not register it",
        )
    logger.warning(
        f"{LOG_PREFIX} dwdp_config.copy_mode='gather' without expert DWDP: the gather kernel "
        f"holds SMs next to the MoE all-to-all. Setup applies "
        f"TRTLLM_CUTEDSL_DENSE_GEMM_SCHEDULER=clc_dynamic and TLLM_ROUTING_DISABLE_COOP_LAUNCH=1, "
        f"each unless it is set, so that the persistent dense GEMM grids do not run a second wave "
        f"on the SMs the gather holds and cooperative routing does not wait for them; "
        f"dwdp_config.gather_kernel picks the gather kernel"
    )


def _reject(requirement: str, actual: str) -> NoReturn:
    raise ValueError(
        f"Dense DWDP (dwdp_config.enable_dense) requires {requirement}, but {actual}. Change "
        f"the launch configuration or set dwdp_config.enable_dense=False."
    )
