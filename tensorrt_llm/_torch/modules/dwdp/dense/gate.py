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
"""Dense DWDP launch gate: reject what dense DWDP cannot serve before the model exists.

``py_executor_creator`` calls :func:`validate_dense_gate` next to its DWDP checks, before it
constructs any DWDP or dense DWDP manager and the model, so an unsupported launch fails before the
weights load rather than in setup or in the first forward.  The gate also decides how dense DWDP
runs: with expert DWDP (attached) or without it, standalone on an attention-DP instance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

from tensorrt_llm._utils import mpi_disabled
from tensorrt_llm.llmapi.llm_args import PrefillCudaGraphBackend
from tensorrt_llm.logger import logger

from .. import fabric_gather, vmm
from .config import DenseConfig
from .types import LOG_PREFIX, CopyMode, DenseMode

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
    from tensorrt_llm.mapping import Mapping

#: The two launch shapes dense DWDP runs in; every mode rejection names both.
_MODES = (
    "runs either with expert DWDP (dwdp_config.enable_experts=True, dwdp_size > 1, tp_size=1, "
    "disable_overlap_scheduler=True) or without it on an attention-DP instance "
    "(dwdp_config.enable_experts=False, enable_attention_dp=True, tp_size > 1)"
)


def validate_dense_gate(llm_args: TorchLlmArgs, mapping: Mapping) -> DenseMode:
    """Reject launches dense DWDP cannot serve, before the model is built; return the mode.

    Called only when ``dwdp_config.enable_dense`` is set.  The checks, in this order:

    1. The mode.  With ``dwdp_config.enable_experts``, attached: ``dwdp_size > 1`` (the dense
       rows are sharded over the DWDP group), the overlap scheduler off, as expert DWDP requires,
       and ``tp_size == 1``.  Without, standalone: ``enable_attention_dp`` and ``tp_size > 1``;
       every rank of the instance then holds the same dense weights, so the rows are sharded over
       the attention-DP (tp) group, and the overlap scheduler may be on.  Any other launch is
       rejected with a message that names both modes, including expert DWDP over an
       attention-DP instance (``tp_size > 1``), which dense DWDP does not run with.
    2. In both modes:

       * ``cuda_graph_config`` and ``torch_compile_config`` None, ``prefill_cuda_graph_backend``
         disabled: the per-layer hooks need eager single-stream execution, since they keep
         host-side slot state and order the copies and the weight reads with stream events in
         every forward.
       * ``sleep_config`` None: the moved weights live in the staging slots and the row store,
         outside the memory pools that sleep and wake-up manage.
       * ``lora_config`` and ``speculative_config`` None: they read layer weights outside the
         layer's read window.
       * ``enable_locality_domains`` off: locality domains make ``Linear`` and ``MLA`` keep
         ``_locality_domain_*`` shard copies of their weights, which dense DWDP does not move,
         so they would stay replicated.
       * ``mapping.pp_size`` and ``cp_size`` 1: every rank of the group holds every layer.
       * FABRIC peer handles (``vmm.peer_handle_type()``): the row store is shared only through
         FABRIC handles.
    3. Standalone only: the MPI backend (``TLLM_DISABLE_MPI`` unset), since the dense group is an
       MPI sub-communicator over the attention-DP group; and with ``dwdp_config.copy_mode`` gather
       a registered ``trtllm::dwdp_fabric_gather``.  A standalone gather launch that passes logs
       one warning, which names the compute-side defaults setup applies for the gather and the
       variable that picks its kernel.  With expert DWDP, ``copy_mode`` sets the expert transfer,
       which the dense copies follow, and ``DwdpManager`` checks the gather op.

    Args:
        llm_args: The launch configuration.
        mapping: The rank mapping, with the DWDP fields filled in.

    Returns:
        ``DenseMode.ATTACHED`` with ``dwdp_config.enable_experts``, else
        ``DenseMode.STANDALONE``.

    Raises:
        ValueError: Naming the first violated requirement.
    """
    config = DenseConfig.from_dwdp_config(llm_args.dwdp_config)
    mode = _launch_mode(llm_args, mapping)
    _require_unset("cuda_graph_config", llm_args.cuda_graph_config)
    _require_unset("torch_compile_config", llm_args.torch_compile_config)
    prefill_graphs = llm_args.prefill_cuda_graph_backend
    if prefill_graphs != PrefillCudaGraphBackend.DISABLED:
        _reject(
            f"prefill_cuda_graph_backend='{PrefillCudaGraphBackend.DISABLED}'",
            f"prefill_cuda_graph_backend='{prefill_graphs}'",
        )
    _require_unset("sleep_config", llm_args.sleep_config)
    _require_unset("lora_config", llm_args.lora_config)
    _require_unset("speculative_config", llm_args.speculative_config)
    if llm_args.enable_locality_domains:
        _reject("enable_locality_domains=False", "enable_locality_domains=True")
    for name in ("pp_size", "cp_size"):
        size = getattr(mapping, name)
        if size != 1:
            _reject(f"{name}=1", f"{name}={size}")
    handle_type = vmm.peer_handle_type()
    if handle_type != vmm.cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC:
        _reject(
            "FABRIC peer handles (CU_MEM_HANDLE_TYPE_FABRIC)",
            f"this host shares {handle_type.name} handles",
        )
    if mode is DenseMode.STANDALONE:
        _check_standalone(config.standalone_copy_mode())
    return mode


def _launch_mode(llm_args: TorchLlmArgs, mapping: Mapping) -> DenseMode:
    """The mode the launch asks for (step 1); rejects a launch that fits neither."""
    dwdp_config = llm_args.dwdp_config
    if dwdp_config.enable_experts:
        if dwdp_config.dwdp_size <= 1:
            _reject_mode(f"expert DWDP with dwdp_size={dwdp_config.dwdp_size}")
        if mapping.tp_size != 1:
            _reject_mode(f"expert DWDP with tp_size={mapping.tp_size}")
        if not llm_args.disable_overlap_scheduler:
            _reject_mode("expert DWDP with disable_overlap_scheduler=False")
        return DenseMode.ATTACHED
    if not mapping.enable_attention_dp:
        _reject_mode("no expert DWDP and enable_attention_dp=False")
    if mapping.tp_size <= 1:
        _reject_mode(f"no expert DWDP and tp_size={mapping.tp_size}")
    return DenseMode.STANDALONE


def _check_standalone(copy_mode: CopyMode) -> None:
    """The standalone-only requirements (step 3), after the common ones."""
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
        f"TRTLLM_DWDP_GATHER_KERNEL picks the gather kernel"
    )


def _require_unset(name: str, value: object) -> None:
    """Reject a launch that sets ``name``."""
    if value is not None:
        _reject(f"{name}=None", f"{name} is set")


def _reject(requirement: str, actual: str) -> NoReturn:
    raise ValueError(
        f"Dense DWDP (dwdp_config.enable_dense) requires {requirement}, but {actual}. Change "
        f"the launch configuration or set dwdp_config.enable_dense=False."
    )


def _reject_mode(actual: str) -> NoReturn:
    raise ValueError(
        f"Dense DWDP (dwdp_config.enable_dense) {_MODES}, but this launch has {actual}. Change "
        f"the launch configuration or set dwdp_config.enable_dense=False."
    )
