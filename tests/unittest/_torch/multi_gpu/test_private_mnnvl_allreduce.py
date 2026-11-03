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
"""A private-workspace MNNVL all-reduce equals MNNVLAllReduce bit for bit, also
while both run concurrently on two streams."""

import os
import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI

import tensorrt_llm
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

pytestmark = pytest.mark.threadleak(enabled=False)

HIDDEN = 7168


def _run(fn, *args):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        fn(rank, *args)
    except Exception:
        traceback.print_exc()
        raise
    return True


@torch.inference_mode()
def _private_allreduce_check(rank, tp):
    os.environ["TLLM_TEST_MNNVL"] = "1"
    from tensorrt_llm._torch.distributed.ops import AllReduceFusionOp, MNNVLAllReduce
    from tensorrt_llm._torch.speculative.private_mnnvl_allreduce import PrivateMNNVLAllReduce
    from tensorrt_llm.functional import AllReduceParams

    mapping = Mapping(world_size=tp, tp_size=tp, rank=rank)
    shared = MNNVLAllReduce(mapping, torch.bfloat16)
    private = PrivateMNNVLAllReduce(mapping, torch.bfloat16, 256, HIDDEN)
    gen = torch.Generator(device="cuda").manual_seed(rank)
    # The same norm weight on every rank.
    cpu_gen = torch.Generator().manual_seed(0)
    gamma = (1 + 0.1 * torch.randn(HIDDEN, generator=cpu_gen)).to(torch.bfloat16).cuda()
    for tokens in (7, 14, 32, 256):
        x = torch.randn(tokens, HIDDEN, device="cuda", generator=gen).to(torch.bfloat16)
        params = AllReduceParams(fusion_op=AllReduceFusionOp.RMS_NORM, norm_weight=gamma, eps=1e-5)
        ref = shared(x, all_reduce_params=params)
        out = private.rms_norm(x, gamma, 1e-5)
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int16), ref.view(torch.int16)), tokens
    # Concurrent: the private reduction on a side stream while the shared one runs.
    side = torch.cuda.Stream()
    xs = [
        torch.randn(14, HIDDEN, device="cuda", generator=gen).to(torch.bfloat16) for _ in range(8)
    ]
    refs = [private.rms_norm(x, gamma, 1e-5) for x in xs]
    torch.cuda.synchronize()
    outs = []
    for x in xs:
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            outs.append(private.rms_norm(x, gamma, 1e-5))
        shared(
            x,
            all_reduce_params=AllReduceParams(
                fusion_op=AllReduceFusionOp.RMS_NORM, norm_weight=gamma, eps=1e-5
            ),
        )
    torch.cuda.synchronize()
    for out, ref in zip(outs, refs):
        assert torch.equal(out.view(torch.int16), ref.view(torch.int16))


def _submit(executor, fn):
    tp = executor.num_workers
    results = executor.map(_run, *zip(*[(fn, tp)] * tp))
    for r in results:
        assert r is True


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_private_mnnvl_allreduce_matches(mpi_pool_executor):
    _submit(mpi_pool_executor, _private_allreduce_check)
