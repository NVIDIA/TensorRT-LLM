# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``K3AllReduce`` without a collective: the strategies it takes, a one-rank group, and the hooks a row-parallel
caller reads. The multi-rank behavior is ``test_modeling_v2_k3_allreduce_op_matrix.py``."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget
from tensorrt_llm.functional import AllReduceParams, AllReduceStrategy
from tensorrt_llm.mapping import Mapping

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.comm.k3_allreduce import (
        K3AllReduce,
    )
except NoTarget as no_target:
    pytest.skip(str(no_target), allow_module_level=True)

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("strategy", [AllReduceStrategy.AUTO, AllReduceStrategy.MNNVL])
def test_one_rank_returns_the_input(strategy):
    module = K3AllReduce(Mapping(), strategy, torch.bfloat16)
    x = torch.randn(3, 16).bfloat16()
    assert module(x) is x
    assert module(x, all_reduce_params=AllReduceParams()) is x
    assert module.mnnvl_allreduce is None


@pytest.mark.parametrize(
    "strategy",
    [s for s in AllReduceStrategy if s not in (AllReduceStrategy.AUTO, AllReduceStrategy.MNNVL)],
)
def test_other_strategies_are_refused(strategy):
    with pytest.raises(ValueError, match="AUTO and MNNVL"):
        K3AllReduce(Mapping(), strategy, torch.bfloat16)


def test_row_parallel_hooks():
    module = K3AllReduce(Mapping())
    assert module.uses_nccl_symmetric_memory_window() is False
    assert module.supports_moe_fold(8, 7168) is False
    assert module.gemm_allreduce_norm(None, None, None, None, None, None) is None
