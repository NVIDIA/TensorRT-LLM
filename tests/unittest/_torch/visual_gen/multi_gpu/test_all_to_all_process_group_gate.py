# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""What ``all_to_all_4d``/``all_to_all_5d`` require, with MPI left enabled.

Both gate on an initialized ``torch.distributed`` process group. The Ulysses
suites set ``TLLM_DISABLE_MPI=1``, so they pass under an env-flag gate too and
cannot tell the two apart. These run on gloo with the flag unset, which is the
combination an MGMN worker actually has.
"""

import os
import socket

import pytest
import torch

from tensorrt_llm._torch.distributed.ops import all_to_all_4d, all_to_all_5d

pytestmark = pytest.mark.cpu_only


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture
def no_mpi_flag():
    had = os.environ.pop("TLLM_DISABLE_MPI", None)
    yield
    if had is not None:
        os.environ["TLLM_DISABLE_MPI"] = had


@pytest.fixture
def single_rank_gloo(no_mpi_flag):
    assert not torch.distributed.is_initialized(), "a prior test leaked a process group"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(_free_port())
    torch.distributed.init_process_group(backend="gloo", init_method="env://", world_size=1, rank=0)
    try:
        yield
    finally:
        torch.distributed.destroy_process_group()


def test_all_to_all_passes_through_a_single_rank_group(single_rank_gloo):
    x4 = torch.randn(2, 4, 8, 16)
    assert torch.equal(all_to_all_4d(x4, scatter_dim=1, gather_dim=2), x4)

    x5 = torch.randn(2, 4, 3, 8, 16)
    assert torch.equal(all_to_all_5d(x5, scatter_dim=1, gather_dim=3), x5)


@pytest.mark.parametrize(
    "call",
    [
        lambda t: all_to_all_4d(t, scatter_dim=1, gather_dim=2),
        lambda t: all_to_all_5d(t.unsqueeze(2).expand(-1, -1, 3, -1, -1), 1, 3),
    ],
    ids=["4d", "5d"],
)
def test_all_to_all_requires_a_process_group(no_mpi_flag, call):
    assert not torch.distributed.is_initialized(), "a prior test leaked a process group"
    with pytest.raises(NotImplementedError, match="process group"):
        call(torch.randn(2, 4, 8, 16))
