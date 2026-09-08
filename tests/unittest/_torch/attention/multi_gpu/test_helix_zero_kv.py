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
"""Multi-rank coverage for the zero-local-KV rows of Helix post-processing.

A CP rank that owns no KV blocks for a token attends to zero keys, so the
attention kernel normalizes by a zero softmax sum and hands back NaN together
with a finite sentinel in the softmax stats. Those rows have to reach the
combine as an exact no-op, and *where* they are neutralized differs per backend:
inside the compiled pre-alltoall region for NCCL, inside the all-to-all sender
for fifo v2, eagerly for fifo v1. Checking any one of those in isolation would
not say the contract holds, so this drives ``_helix_post_process`` itself.

Nothing else covers this. ``test_helix_postprocess.py`` exercises
``_helix_sanitize_empty_kv`` and ``_helix_zero_kv_mask`` directly but never the
exchange, and ``test_mla_helix.py`` drives all three backends end to end while
handing every rank an equal, non-zero slice of the KV
(``ctx_len_per_gpu = ctx_len // world_size``), so its ``zero_kv_mask`` is all
False and the neutralization never runs.

Masked rows are poisoned with NaN before the exchange, and
``test_zero_kv_negative_control`` withholds the mask to prove that poison does
reach the output. Without that control a PASS here would only mean the test ran.

Scope: this runs at cp_size 2, so it does not cover the sender's entry stride at
larger cp (``entryIdx`` advances by a channel count derived from cp_size). That
was checked out of tree at cp16.
"""

import pickle
import sys

import _torch.attention.multi_gpu.helix_test_utils as helix_utils
import cloudpickle
import pytest
import torch
from _torch.attention.multi_gpu.helix_test_utils import parse_comms_medium, run_single_rank
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
from utils.util import skip_pre_blackwell

import tensorrt_llm
from tensorrt_llm._torch.attention.attention import _helix_post_process
from tensorrt_llm.mapping import CpType, Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
cloudpickle.register_pickle_by_value(helix_utils)
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

WORLD_SIZE = 2
NUM_HEADS = 6  # heads this rank owns after TP/CP splitting
VALUE_DIM = 512  # kv_lora_rank
# bf16 partials through an fp32 combine. Three orders below the ~1.0 scale of
# the data, and loose on purpose: the failure this hunts is NaN, not rounding.
TOLERANCE = 2e-2


def _zero_kv_mask(rank: int, world_size: int, num_tokens: int, device: torch.device):
    """Per-rank mask marking the tokens this rank owns no KV for.

    Deliberately different on every rank, so an implementation that applies some
    other rank's mask fails. Only token ``rank`` plus a strided tail are masked,
    which leaves most tokens combining every rank. An earlier version masked
    ``t % world_size == rank`` and left exactly one contributor per token, so
    every combine weight was 1.0 and the output came back bit-identical to the
    input -- a real number from a test that never combined anything.
    """
    idx = torch.arange(num_tokens, device=device)
    mask = idx == rank
    mask |= (idx >= world_size) & (idx % (2 * world_size + 1) == rank)
    return mask


def _reference(all_o, all_stats, all_mask, my_rank, cp_size):
    """float64 combine over every rank's pre-sanitize tensors.

    ``all_o[r]`` is rank r's partial output laid out
    ``[num_tokens, cp_size, num_heads, value_dim]`` where the cp dimension
    indexes the *destination* rank, so this rank's share is ``[:, my_rank]``.
    """
    num_tokens = all_o[0].shape[0]
    o = torch.stack([t.view(num_tokens, cp_size, NUM_HEADS, VALUE_DIM)[:, my_rank] for t in all_o])
    s = torch.stack([t.view(num_tokens, cp_size, NUM_HEADS, 2)[:, my_rank] for t in all_stats])
    m = torch.stack(all_mask)[:, :, None]  # [ranks, tokens, 1]

    o = torch.where(m[..., None], torch.zeros((), dtype=torch.float64), o.double())
    smax = torch.where(m, torch.full((), float("-inf"), dtype=torch.float64), s[..., 0].double())
    ssum = torch.where(m, torch.zeros((), dtype=torch.float64), s[..., 1].double())

    weight = ssum * torch.exp(smax - smax.max(dim=0).values)
    weight = weight / weight.sum(dim=0)
    return (o * weight[..., None]).sum(dim=0).reshape(num_tokens, NUM_HEADS * VALUE_DIM)


def _zero_kv_rank(rank, world_size, num_tokens, comms_medium, use_cuda_graph, withhold_mask):
    """Rank body: build poisoned inputs, post-process, compare to a reference.

    Returns ``(nan_count, max_abs_error)``. ``max_abs_error`` is NaN when the
    output contains NaN, since the comparison is meaningless there.
    """
    comm = tensorrt_llm.mpi_comm()
    device = torch.device("cuda", torch.cuda.current_device())

    torch.manual_seed(1234 + rank)
    partial_o = torch.randn(
        num_tokens, world_size * NUM_HEADS * VALUE_DIM, device=device, dtype=torch.bfloat16
    )
    stats = torch.empty(num_tokens, world_size * NUM_HEADS, 2, device=device, dtype=torch.float32)
    stats[..., 0].normal_(0.0, 2.0)  # max
    stats[..., 1].uniform_(0.5, 2.0)  # sum, strictly positive

    mask = _zero_kv_mask(rank, world_size, num_tokens, device)
    # Poison exactly what the neutralization is supposed to overwrite. The large
    # finite max makes an unsanitized row win the combine outright rather than
    # being rounded away, and 0 * NaN = NaN means its partial_o then poisons the
    # result on every rank.
    partial_o[mask] = float("nan")
    stats[mask, :, 0] = 1e4
    stats[mask, :, 1] = 7.0

    all_o = [torch.from_numpy(x) for x in comm.allgather(partial_o.float().cpu().numpy())]
    all_s = [torch.from_numpy(x) for x in comm.allgather(stats.cpu().numpy())]
    all_m = [torch.from_numpy(x) for x in comm.allgather(mask.cpu().numpy())]
    reference = _reference(all_o, all_s, all_m, rank, world_size).to(device)

    use_nccl_for_alltoall, fifo_version = parse_comms_medium(comms_medium)
    mapping = Mapping(
        world_size=world_size,
        rank=rank,
        cp_size=world_size,
        cp_config={
            "cp_type": CpType.HELIX,
            "use_nccl_for_alltoall": use_nccl_for_alltoall,
            "fifo_version": fifo_version,
        },
    )
    zero_kv_mask = None if withhold_mask else mask

    if use_cuda_graph:
        # Production captures this region, and the masked branch has only ever
        # been exercised eagerly. Warm up on a side stream first, then capture
        # on every rank at once -- the all-to-all is collective.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                _helix_post_process(
                    partial_o, stats, mapping, NUM_HEADS, VALUE_DIM, zero_kv_mask=zero_kv_mask
                )
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        comm.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = _helix_post_process(
                partial_o, stats, mapping, NUM_HEADS, VALUE_DIM, zero_kv_mask=zero_kv_mask
            )
        torch.cuda.synchronize()
        comm.barrier()
        graph.replay()
        torch.cuda.synchronize()
    else:
        output = _helix_post_process(
            partial_o.clone(),
            stats.clone(),
            mapping,
            NUM_HEADS,
            VALUE_DIM,
            zero_kv_mask=zero_kv_mask,
        )

    output = output.double()
    nan_count = int(torch.isnan(output).sum())
    max_err = float("nan") if nan_count else float((output - reference).abs().max())
    return nan_count, max_err


def _launch(num_tokens, comms_medium, use_cuda_graph=False, withhold_mask=False):
    """Run ``_zero_kv_rank`` on WORLD_SIZE ranks and collect their results."""
    args = (_zero_kv_rank, WORLD_SIZE, num_tokens, comms_medium, use_cuda_graph, withhold_mask)
    with MPIPoolExecutor(max_workers=WORLD_SIZE) as executor:
        return list(executor.map(run_single_rank, *zip(*[args] * WORLD_SIZE)))


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs to run this test")
@skip_pre_blackwell
@pytest.mark.parametrize("num_tokens", [17, 96])
@pytest.mark.parametrize("comms_medium", ["nccl", "fifo_v1", "fifo_v2"])
def test_zero_kv_rows_are_neutral(comms_medium: str, num_tokens: int):
    """Masked rows must contribute nothing, on every backend."""
    for rank, (nan_count, max_err) in enumerate(_launch(num_tokens, comms_medium)):
        assert nan_count == 0, (
            f"rank {rank}: {nan_count} NaN in the output, so a zero-local-KV row "
            f"reached the combine unsanitized"
        )
        assert max_err < TOLERANCE, f"rank {rank}: max|out - ref| = {max_err:.3e}"


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs to run this test")
@skip_pre_blackwell
def test_zero_kv_rows_are_neutral_under_cuda_graph():
    """Same contract inside a CUDA graph, which is how production runs it.

    fifo v2 only: it is the backend that neutralizes inside the all-to-all
    sender, and NCCL collectives under capture are a separate question.
    """
    for rank, (nan_count, max_err) in enumerate(_launch(96, "fifo_v2", use_cuda_graph=True)):
        assert nan_count == 0, f"rank {rank}: {nan_count} NaN in the replayed output"
        assert max_err < TOLERANCE, f"rank {rank}: max|out - ref| = {max_err:.3e}"


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs to run this test")
@skip_pre_blackwell
@pytest.mark.parametrize("comms_medium", ["nccl", "fifo_v1", "fifo_v2"])
def test_zero_kv_negative_control(comms_medium: str):
    """Withholding the mask must produce NaN.

    This is what makes the tests above mean something: it shows the poison in
    the masked rows really does reach the output when nothing neutralizes it.
    """
    results = _launch(96, comms_medium, withhold_mask=True)
    assert any(nan_count > 0 for nan_count, _ in results), (
        "no NaN with the mask withheld, so these tests cannot detect a missing "
        "sanitize and their PASS means nothing"
    )
