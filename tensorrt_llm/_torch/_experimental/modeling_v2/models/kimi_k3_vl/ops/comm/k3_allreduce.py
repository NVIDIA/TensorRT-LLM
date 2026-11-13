# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Kimi K3 targets' TP all-reduce module, in place of the built-in ``AllReduce``.

``K3AllReduce(mapping, strategy, dtype)(x, all_reduce_params=...)`` takes the calls the K3 targets make (the sum of a
row-parallel output, or a fusion; ``all_reduce_params`` None or a ``tensorrt_llm.functional.AllReduceParams``) and
runs, for the strategies the targets route (AUTO and MNNVL), the kernel the built-in module runs, decided the same
way and in the same order (``AllReduce.forward``):

1. one rank, or ``enable_allreduce`` False: the input itself;
2. MPI disabled and the mapping carries a TP process group (Ray): ``torch.distributed.all_reduce``, plain sum only;
3. over MNNVL (``mnnvl_workspace.mnnvl_supported``: the TP group spans nodes of one NVLink domain, or
   the MNNVL strategy is named on an MNNVL-capable host): ``mnnvl_allreduce``, a ``K3MnnvlAllReduce`` on the
   mapping's shared Lamport workspace, one-shot up to the built-in threshold until the decode path raises it
   (``decode_comm.use_decode_one_shot``); a fusion it does not run falls through;
4. ``autotuned_allreduce`` when ``autotune_enabled()`` (MPI on, ``TLLM_DISABLE_ALLREDUCE_AUTOTUNE``
   not set; read per call), else ``catalog/comm/allreduce`` with strategy AUTO, both on the mapping's IPC workspace,
   every fusion passed through. The MNNVL strategy also takes this step with AUTO, as the built-in module does.

Not taken: the symmetric-memory strategy (the targets route AUTO and MNNVL only) and the MoE fold.
``uses_nccl_symmetric_memory_window()`` is False: the targets' row-parallel GEMMs write ordinary memory, as the
built-in BF16 ``Linear`` does on the K3 targets (only its ``use_custom_cublas_mm`` path writes a window).

An autotuned call that runs NCCL_SYMMETRIC returns tensors leased from NCCL window buffers. The engine wraps every
model forward in an NCCL window tensor scope and binds CUDA graph captures to their pool, which releases them; a
caller outside the engine (a test) opens ``nccl_window_tensor_scope`` around its calls, as the built-in module's
callers do.
"""

from typing import Optional, Tuple, Union

import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.allreduce import (
    allreduce,
    allreduce_group_args,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl import mnnvl_workspace as mcast
from tensorrt_llm.functional import AllReduceFusionOp, AllReduceStrategy
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from .autotuned_allreduce import autotune_enabled, autotuned_allreduce, autotuned_allreduce_args
from .mnnvl_allreduce import K3MnnvlAllReduce

ROUTED_STRATEGIES = (AllReduceStrategy.AUTO, AllReduceStrategy.MNNVL)


class K3AllReduce(nn.Module):
    """The sum of a tensor over ``mapping``'s TP group (see the module docstring for which kernel runs).

    Construction is collective over the TP group, as the built-in module's is: it builds the mapping's IPC
    workspace (``catalog/comm/allreduce.allreduce_group_args``) and, over MNNVL, the shared Lamport workspace.
    ``mnnvl_allreduce`` is the MNNVL all-reduce, or None off MNNVL."""

    def __init__(
        self,
        mapping: Mapping,
        strategy: AllReduceStrategy = AllReduceStrategy.AUTO,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        strategy = AllReduceStrategy(strategy)
        if strategy not in ROUTED_STRATEGIES:
            raise ValueError(f"K3AllReduce runs the AUTO and MNNVL strategies, not {strategy.name}")
        self.mapping = mapping
        self.strategy = strategy
        self.dtype = dtype
        self.mnnvl_allreduce: Optional[K3MnnvlAllReduce] = None
        self._group: Optional[dict] = None
        self._autotune: Optional[dict] = None
        self._torch_distributed = False
        if mapping.tp_size == 1:
            return
        if mapping.enable_attention_dp:
            raise ValueError("K3AllReduce reduces over the TP group; attention DP has none")
        from tensorrt_llm._utils import mpi_disabled

        # The built-in module's plain torch.distributed path under Ray.
        self._torch_distributed = mpi_disabled() and (
            getattr(mapping, "_use_torch_distributed_allreduce", False)
            or getattr(mapping, "_tp_group_pg", None) is not None
        )
        self._group = allreduce_group_args(mapping, int(AllReduceStrategy.AUTO))
        self._autotune = autotuned_allreduce_args(mapping)
        if mcast.mnnvl_supported(
            mapping, dtype, explicitly_requested=strategy == AllReduceStrategy.MNNVL
        ):
            # Every rank reaches the same outcome: building the shared workspace fails on all ranks or on none.
            try:
                self.mnnvl_allreduce = K3MnnvlAllReduce(mapping, dtype)
            except Exception as error:  # noqa: BLE001 -- the fallback below takes every call
                logger.debug(f"MNNVL AllReduce can't be enabled due to {error}.")
                self.mnnvl_allreduce = None

    def forward(
        self, input: torch.Tensor, *, all_reduce_params=None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """``input`` summed over the TP group (a new tensor), or ``all_reduce_params.fusion_op``'s outputs:
        ``(norm_out, residual_out)`` for ``RESIDUAL_RMS_NORM``. ``input`` itself when the group has one rank or
        ``all_reduce_params.enable_allreduce`` is False."""
        params = all_reduce_params
        if self.mapping.tp_size == 1 or (params is not None and params.enable_allreduce is False):
            return input
        input = input.contiguous()
        fusion_op = AllReduceFusionOp.NONE if params is None else params.fusion_op
        if self._torch_distributed:
            if fusion_op != AllReduceFusionOp.NONE or (
                params is not None
                and any(
                    v is not None
                    for v in (params.residual, params.norm_weight, params.scale, params.bias)
                )
            ):
                raise NotImplementedError(
                    "Torch distributed all-reduce fallback supports only plain all-reduce."
                )
            output = input.clone()
            torch.distributed.all_reduce(output, group=self.mapping.tp_group_pg)
            return output
        if self.mnnvl_allreduce is not None:
            out = self.mnnvl_allreduce(input, all_reduce_params=params)
            if out is not None:
                return out
        args = (
            input,
            None if params is None else params.residual,
            None if params is None else params.norm_weight,
            None if params is None else params.scale,
            None if params is None else params.bias,
        )
        eps = 1e-6 if params is None else params.eps
        trigger = True if params is None else params.trigger_completion_at_end
        if autotune_enabled():
            out = autotuned_allreduce(
                *args,
                op=int(fusion_op),
                eps=eps,
                trigger_completion_at_end=trigger,
                **self._autotune,
            )
        else:
            out = allreduce(
                *args, op=int(fusion_op), eps=eps, trigger_completion_at_end=trigger, **self._group
            )
        # The built-in module returns these ops' Tensor[] as is (``output if len(output) > 1 else output[0]``): a
        # list for a fusion with several outputs. Its MNNVL path returns a tuple, as ``mnnvl_allreduce`` above does.
        return list(out) if isinstance(out, tuple) else out

    def supports_moe_fold(self, num_tokens: int, hidden_size: int) -> bool:
        """The MoE fold (``MoEAllReduceParams``) is never taken."""
        return False

    def gemm_allreduce_norm(
        self, input, weight, weight_scale, input_scale, all_reduce_params, gemm_tactic
    ) -> None:
        """No fused GEMM + all-reduce + norm kernel: the caller runs them apart."""
        return None

    def uses_nccl_symmetric_memory_window(self) -> bool:
        """False: the caller allocates the input in ordinary memory (see the module docstring)."""
        return False

    @property
    def output_buffer_kind(self) -> int:
        """The buffer kind a caller allocates this module's input in: the default one."""
        from tensorrt_llm.bindings.internal.thop import BufferKind

        return int(BufferKind.DEFAULT)
