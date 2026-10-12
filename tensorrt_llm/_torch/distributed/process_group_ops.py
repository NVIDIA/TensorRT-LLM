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
"""torch.compile-traceable wrappers around the ``trtllm::*_pg`` collectives.

With MPI disabled (``TLLM_DISABLE_MPI=1``) the C++ collectives
``trtllm::allreduce_pg``, ``allgather_pg``, ``allgather_list_pg``,
``reducescatter_pg`` and ``reducescatter_list_pg`` take a boxed c10d
``ProcessGroup`` (``__torch__.torch.classes.c10d.ProcessGroup``).  Dynamo can
trace neither ``ProcessGroup.boxed()`` nor a pre-boxed script object flowing
into an op call (no fake class is registered for the c10d ProcessGroup), so
every such call site was a graph break and ``torch.compile(fullgraph=True)``
was impossible for tensor-parallel models.

The ``*_pg_by_name`` ops below take the group's c10d **name** instead -- a
plain ``str``, schema-legal and a constant to dynamo -- and resolve + box the
group inside the opaque op body.  This mirrors how
``torch.distributed._functional_collectives`` identifies groups.  The fake
implementations delegate to the fakes of the MPI-mode ops (``trtllm::allreduce``
etc., registered in ``tensorrt_llm/_torch/custom_ops/cpp_custom_ops.py``) so
output shapes keep a single source of truth.
"""

import torch
from torch.distributed import ProcessGroup
from torch.distributed.distributed_c10d import _resolve_process_group


def process_group_name(pg: ProcessGroup) -> str:
    """Return the c10d registry name of ``pg``.

    Every group created through ``torch.distributed.new_group`` or a
    ``DeviceMesh`` is registered under a unique name that
    ``_resolve_process_group`` maps back to the group.  Reading the attribute
    is traceable by dynamo (it folds to a string constant), unlike
    ``pg.boxed()``.
    """
    name = getattr(pg, "group_name", None)
    if not name:
        raise ValueError(
            "ProcessGroup has no c10d group_name; create groups via "
            "torch.distributed.new_group / DeviceMesh so torch.compile'd "
            "collectives can resolve them by name."
        )
    return str(name)


def _boxed_process_group(group_name: str) -> torch.ScriptObject:
    return _resolve_process_group(group_name).boxed()


@torch.library.custom_op("trtllm::allreduce_pg_by_name", mutates_args=())
def allreduce_pg_by_name(
    input: torch.Tensor,
    residual: torch.Tensor | None,
    norm_weight: torch.Tensor | None,
    scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    workspace: torch.Tensor | None,
    group: list[int],
    rank: int,
    group_name: str,
    strategy: int,
    op: int,
    eps: float,
    trigger_completion_at_end: bool,
) -> list[torch.Tensor]:
    return torch.ops.trtllm.allreduce_pg(
        input,
        residual,
        norm_weight,
        scale,
        bias,
        workspace,
        group,
        rank,
        _boxed_process_group(group_name),
        strategy,
        op,
        eps,
        trigger_completion_at_end,
    )


@allreduce_pg_by_name.register_fake
def _(
    input,
    residual,
    norm_weight,
    scale,
    bias,
    workspace,
    group,
    rank,
    group_name,
    strategy,
    op,
    eps,
    trigger_completion_at_end,
):
    return torch.ops.trtllm.allreduce(
        input,
        residual,
        norm_weight,
        scale,
        bias,
        workspace,
        group,
        strategy,
        op,
        eps,
        trigger_completion_at_end,
    )


@torch.library.custom_op("trtllm::allgather_pg_by_name", mutates_args=())
def allgather_pg_by_name(
    input: torch.Tensor, sizes: list[int] | None, group: list[int], group_name: str
) -> torch.Tensor:
    return torch.ops.trtllm.allgather_pg(input, sizes, group, _boxed_process_group(group_name))


@allgather_pg_by_name.register_fake
def _(input, sizes, group, group_name):
    return torch.ops.trtllm.allgather(input, sizes, group)


@torch.library.custom_op("trtllm::allgather_list_pg_by_name", mutates_args=())
def allgather_list_pg_by_name(
    input_list: list[torch.Tensor], sizes: list[int] | None, group: list[int], group_name: str
) -> list[torch.Tensor]:
    return torch.ops.trtllm.allgather_list_pg(
        input_list, sizes, group, _boxed_process_group(group_name)
    )


@allgather_list_pg_by_name.register_fake
def _(input_list, sizes, group, group_name):
    return torch.ops.trtllm.allgather_list(input_list, sizes, group)


@torch.library.custom_op("trtllm::reducescatter_pg_by_name", mutates_args=())
def reducescatter_pg_by_name(
    input: torch.Tensor, sizes: list[int] | None, group: list[int], group_name: str
) -> torch.Tensor:
    return torch.ops.trtllm.reducescatter_pg(input, sizes, group, _boxed_process_group(group_name))


@reducescatter_pg_by_name.register_fake
def _(input, sizes, group, group_name):
    return torch.ops.trtllm.reducescatter(input, sizes, group)


@torch.library.custom_op("trtllm::reducescatter_list_pg_by_name", mutates_args=())
def reducescatter_list_pg_by_name(
    input_list: list[torch.Tensor], sizes: list[int] | None, group: list[int], group_name: str
) -> list[torch.Tensor]:
    return torch.ops.trtllm.reducescatter_list_pg(
        input_list, sizes, group, _boxed_process_group(group_name)
    )


@reducescatter_list_pg_by_name.register_fake
def _(input_list, sizes, group, group_name):
    # ``trtllm::reducescatter_list`` has no fake of its own; the per-tensor op has.
    return [torch.ops.trtllm.reducescatter(t, sizes, group) for t in input_list]
