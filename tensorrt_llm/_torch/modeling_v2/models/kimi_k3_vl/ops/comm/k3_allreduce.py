# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Kimi K3 targets' TP all-reduce module: the runtime's ``AllReduce`` with the family's MNNVL all-reduce.

``K3AllReduce(mapping, strategy, dtype)`` is a ``tensorrt_llm._torch.distributed.AllReduce``. Every call runs the
runtime module's ``forward``, which picks the path:

1. one rank, or ``enable_allreduce`` False: the input itself;
2. MPI disabled and the mapping carries a TP process group (Ray): ``torch.distributed.all_reduce``;
3. ``mnnvl_allreduce`` when it is set: a ``K3MnnvlAllReduce`` on the mapping's shared Lamport workspace. A fusion
   it does not run returns None and falls through to step 4;
4. the runtime's own all-reduce on the mapping's IPC workspace (the all-reduce autotuner, or the raw op with
   ``TLLM_DISABLE_ALLREDUCE_AUTOTUNE=1``), with every fusion passed through. The MNNVL strategy takes this step
   with AUTO.

The runtime module is built as any model builds it (``AllReduce.__init__`` with the dtype): where it runs MNNVL
(the TP group spans nodes of one NVLink domain, or the MNNVL strategy is named on an MNNVL-capable host) it builds
its own MNNVL all-reduce on the mapping's shared Lamport workspace. This module then sets the public
``mnnvl_allreduce`` slot: to a ``K3MnnvlAllReduce`` on the same workspace where ``mnnvl_workspace.mnnvl_supported``
holds, else to None. The family's one-shot ceiling is the runtime's until the decode path raises it
(``decode_comm.use_decode_one_shot``).

Step 4 can return NCCL window buffers (NCCL_SYMMETRIC). The engine opens an NCCL window tensor scope around every
model forward and binds CUDA graph captures to their pool. A caller outside the engine (a test) opens
``nccl_window_tensor_scope`` around its calls.
"""

import torch

from tensorrt_llm._torch.distributed import AllReduce
from tensorrt_llm._torch.modeling_v2.models.kimi_k3_vl import engine_settings
from tensorrt_llm._torch.modeling_v2.models.kimi_k3_vl import mnnvl_workspace as mcast
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from .mnnvl_allreduce import K3MnnvlAllReduce

# The strategies the targets take (``engine_settings.ALLREDUCE_STRATEGIES``, which route and the constructors check).
ROUTED_STRATEGIES = tuple(AllReduceStrategy[name] for name in engine_settings.ALLREDUCE_STRATEGIES)


class K3AllReduce(AllReduce):
    """The sum of a tensor over ``mapping``'s TP group (see the module docstring for which kernel runs).

    Construction is collective over the TP group, as the runtime module's is: it builds the mapping's IPC workspace
    and, over MNNVL, the shared Lamport workspace. ``mnnvl_allreduce`` is the family's MNNVL all-reduce, or None
    off MNNVL. ``forward(input, all_reduce_params=...)`` returns ``input`` summed over the group (a new tensor), or
    ``all_reduce_params.fusion_op``'s outputs: the one tensor of a single-output fusion (RMS_NORM), and several
    outputs as a tuple over MNNVL and a list off MNNVL."""

    def __init__(
        self,
        mapping: Mapping,
        strategy: AllReduceStrategy = AllReduceStrategy.AUTO,
        dtype: torch.dtype = torch.bfloat16,
    ):
        strategy = AllReduceStrategy(strategy)
        if strategy not in ROUTED_STRATEGIES:
            raise ValueError(f"K3AllReduce runs the AUTO and MNNVL strategies, not {strategy.name}")
        if mapping.tp_size > 1 and mapping.enable_attention_dp:
            raise ValueError("K3AllReduce reduces over the TP group; attention DP has none")
        super().__init__(mapping=mapping, strategy=strategy, dtype=dtype)
        # The slot holds the family's MNNVL all-reduce or nothing, never the runtime's.
        self.mnnvl_allreduce = None
        self.dtype = dtype
        if mapping.tp_size == 1:
            return
        if mcast.mnnvl_supported(
            mapping, dtype, explicitly_requested=strategy == AllReduceStrategy.MNNVL
        ):
            # Every rank reaches the same outcome: building the shared workspace fails on all ranks or on none.
            try:
                self.mnnvl_allreduce = K3MnnvlAllReduce(mapping, dtype)
            except Exception as error:  # noqa: BLE001 -- the runtime's all-reduce takes every call
                logger.debug(f"MNNVL AllReduce can't be enabled due to {error}.")
                self.mnnvl_allreduce = None
