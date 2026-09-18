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
"""ReduceOp enum-mapping regression tests.

tensorrt_llm's ReduceOp and torch.distributed.ReduceOp assign different
integers to the same operation (tensorrt_llm MIN=2/MAX=3, torch MIN=3/MAX=4).
Since both hash as plain ints, passing a torch enum into the tensorrt_llm
mapping used to hash-collide into the WRONG entry: torch MIN silently became
a MAX reduction (observed in production as every rank adopting the largest
per-rank KV block count instead of the smallest). The mappers now reject
foreign op types outright.
"""

import pytest
import torch
from mpi4py import MPI

from tensorrt_llm._torch.distributed.communicator import (
    ReduceOp,
    reduce_op_to_mpi,
    reduce_op_to_torch,
)

_TORCH_EXPECTED = {
    ReduceOp.SUM: torch.distributed.ReduceOp.SUM,
    ReduceOp.PRODUCT: torch.distributed.ReduceOp.PRODUCT,
    ReduceOp.MIN: torch.distributed.ReduceOp.MIN,
    ReduceOp.MAX: torch.distributed.ReduceOp.MAX,
    ReduceOp.BAND: torch.distributed.ReduceOp.BAND,
    ReduceOp.BOR: torch.distributed.ReduceOp.BOR,
    ReduceOp.BXOR: torch.distributed.ReduceOp.BXOR,
}

_MPI_EXPECTED = {
    ReduceOp.SUM: MPI.SUM,
    ReduceOp.PRODUCT: MPI.PROD,
    ReduceOp.MIN: MPI.MIN,
    ReduceOp.MAX: MPI.MAX,
    ReduceOp.BAND: MPI.BAND,
    ReduceOp.BOR: MPI.BOR,
    ReduceOp.BXOR: MPI.BXOR,
}


@pytest.mark.parametrize("op", list(ReduceOp), ids=lambda op: op.name)
def test_every_member_maps_to_matching_torch_op(op):
    assert reduce_op_to_torch(op) is _TORCH_EXPECTED[op]


@pytest.mark.parametrize("op", list(ReduceOp), ids=lambda op: op.name)
def test_every_member_maps_to_matching_mpi_op(op):
    assert reduce_op_to_mpi(op) is _MPI_EXPECTED[op]


@pytest.mark.parametrize(
    "foreign",
    [torch.distributed.ReduceOp.MIN, torch.distributed.ReduceOp.MAX, 2, 3],
    ids=["torch_MIN", "torch_MAX", "int_2", "int_3"],
)
def test_foreign_op_rejected_by_torch_mapper(foreign):
    with pytest.raises(TypeError, match="tensorrt_llm ReduceOp"):
        reduce_op_to_torch(foreign)


def test_foreign_op_rejected_by_mpi_mapper():
    with pytest.raises(TypeError, match="tensorrt_llm ReduceOp"):
        reduce_op_to_mpi(torch.distributed.ReduceOp.MIN)


def test_enum_values_disagree_so_type_check_is_load_bearing():
    # The premise of the strict type check: torch's MIN shares its integer
    # value with tensorrt_llm's MAX, so a permissive dict lookup resolves a
    # torch MIN to a MAX reduction. If torch ever renumbers its enum and this
    # assertion fails, revisit whether the type check is still required.
    assert int(torch.distributed.ReduceOp.MIN) == int(ReduceOp.MAX)
