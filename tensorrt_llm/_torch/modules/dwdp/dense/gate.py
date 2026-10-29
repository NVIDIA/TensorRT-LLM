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
constructs the ``DwdpManager`` and the model, so an unsupported launch fails before the weights
load rather than in setup or in the first forward.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

from tensorrt_llm.llmapi.llm_args import PrefillCudaGraphBackend

from .. import vmm
from .config import DENSE_ENV, DenseConfig

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
    from tensorrt_llm.mapping import Mapping


def validate_dense_gate(llm_args: TorchLlmArgs, mapping: Mapping) -> None:
    """Reject launches dense DWDP cannot serve, before the model is built.

    Called only when ``dense_dwdp_requested()``.  The requirements, checked in this order:

    * Valid ``TRTLLM_DWDP_DENSE*`` switches: ``DenseConfig.from_env()`` reads them all, so an
      invalid ``TRTLLM_DWDP_DENSE_SEGMENTS`` fails here rather than in setup P1, and the
      warnings about values it ignores are logged before the load as well.
    * ``dwdp_config`` with ``dwdp_size > 1``: the dense rows are sharded over the DWDP group.
    * ``cuda_graph_config`` and ``torch_compile_config`` None, ``prefill_cuda_graph_backend``
      disabled: the per-layer hooks need eager single-stream execution, since they keep host-side
      slot state and order the copies and the weight reads with stream events in every forward.
    * ``sleep_config`` None: the moved weights live in the staging slots and the row store,
      outside the memory pools that sleep and wake-up manage.
    * ``lora_config`` and ``speculative_config`` None: they read layer weights outside the
      layer's read window.
    * ``enable_ugpu`` off: uGPU makes ``Linear`` and ``MLA`` keep ``_ugpu_*`` shard copies of
      their weights, which dense DWDP does not move, so they would stay replicated.
    * ``disable_overlap_scheduler``: DWDP runs only with the overlap scheduler off.
    * ``mapping.tp_size``, ``pp_size`` and ``cp_size`` all 1: every rank of the group holds the
      same unsharded dense weights of every layer.
    * FABRIC peer handles (``vmm.peer_handle_type()``): the row store is shared only through
      FABRIC handles.

    Args:
        llm_args: The launch configuration.
        mapping: The rank mapping, with the DWDP fields filled in.

    Raises:
        ValueError: Naming the first violated requirement.
    """
    try:
        DenseConfig.from_env()
    except ValueError as exc:
        _reject(f"valid {DENSE_ENV}_* switches", str(exc))
    dwdp_config = llm_args.dwdp_config
    if dwdp_config is None:
        _reject("dwdp_config with dwdp_size > 1", "dwdp_config is not set")
    if dwdp_config.dwdp_size <= 1:
        _reject("dwdp_config with dwdp_size > 1", f"dwdp_size={dwdp_config.dwdp_size}")
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
    if llm_args.enable_ugpu:
        _reject("enable_ugpu=False", "enable_ugpu=True")
    if not llm_args.disable_overlap_scheduler:
        _reject("disable_overlap_scheduler=True", "disable_overlap_scheduler=False")
    for name in ("tp_size", "pp_size", "cp_size"):
        size = getattr(mapping, name)
        if size != 1:
            _reject(f"{name}=1", f"{name}={size}")
    handle_type = vmm.peer_handle_type()
    if handle_type != vmm.cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC:
        _reject(
            "FABRIC peer handles (CU_MEM_HANDLE_TYPE_FABRIC)",
            f"this host shares {handle_type.name} handles",
        )


def _require_unset(name: str, value: object) -> None:
    if value is not None:
        _reject(f"{name}=None", f"{name} is set")


def _reject(requirement: str, actual: str) -> NoReturn:
    raise ValueError(
        f"Dense DWDP ({DENSE_ENV}) requires {requirement}, but {actual}. Change the launch "
        f"configuration or unset {DENSE_ENV}."
    )
