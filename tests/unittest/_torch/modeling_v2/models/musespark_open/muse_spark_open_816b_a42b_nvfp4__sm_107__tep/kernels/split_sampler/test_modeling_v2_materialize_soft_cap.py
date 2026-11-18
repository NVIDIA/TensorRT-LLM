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
"""``TargetHead.materialize`` (``kernels/split_sampler/target_head.py``) at this target's head shape.

``materialize`` hands the built-in code (rejection sampling, occurrence penalties, greedy and non-fused sampling)
the processor's FP32 capped logits of the step's pending BF16 row. It computes them in one pass
(``soft_capped_logits``); the oracle is ``head_sampler.reference_capped``, the processor's torch chain statement
for statement. The tests pin, at 1, 8, 64, 65, 96 and 768 rows of the 202,048-entry vocabulary:

* the materialized logits equal the oracle's bit for bit (NaN positions alike), for every BF16 input value and for
  logits of served magnitude, for the processor's constants and two other multiplier / cap pairs; on a row the step
  left whole and on a vocabulary-parallel row (``pending_local``) the head all-gathers first, where the logits
  argument is only the placeholder and must not be read;
* the same holds when ``materialize`` is captured in a CUDA graph and the graph is replayed on new logits;
* ``materialize`` consumes the pending row and returns a new FP32 tensor.
"""

import types

import pytest
import torch

import tensorrt_llm._torch.distributed as trtllm_distributed
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep.kernels.split_sampler import (  # noqa: E501
    target_head,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep.ops.spec import (  # noqa: E501
    head_sampler,
)

VOCAB = 202048
WORLD = 4
SHARD = VOCAB // WORLD
#: One row, one request's verify block, both sides of a 64-row boundary, 96 requests' bonus
#: rows and the wide generation server's 96 requests at K = 7.
ROWS = [1, 8, 64, 65, 96, 768]
#: The processor's (``output_multiplier``, ``final_logit_softcapping``), a second pair, and a pair
#: without a multiplier.
CONSTANTS = [(0.17677669529663687, 20.0), (32.0**-0.5, 30.0), (1.0, 30.0)]
CONSTANT_IDS = ["processor", "cap30", "unit_multiplier"]


def _supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)


pytestmark = pytest.mark.skipif(not _supported(), reason="this target's head runs on SM107")


def every_bf16_value(rows: int) -> torch.Tensor:
    """``[rows, VOCAB]`` BF16 logits whose every row holds each of the 65,536 BF16 bit patterns
    (Inf, NaN and subnormals included), in an order that differs from row to row."""
    # An odd stride walks all 2**16 residues in any 2**16 consecutive columns.
    cols = (torch.arange(VOCAB, dtype=torch.int64, device="cuda") * 40503 & 0xFFFF).to(torch.int32)
    offsets = torch.arange(rows, dtype=torch.int32, device="cuda").unsqueeze(1) * 12347
    bits = ((cols + offsets) & 0xFFFF) - 2**15
    return bits.to(torch.int16).view(torch.bfloat16)


def served_logits(rows: int, seed: int) -> torch.Tensor:
    """``[rows, VOCAB]`` BF16 logits of served magnitude, with a few large outliers."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, VOCAB, device="cuda", generator=gen) * 6.0
    x[:, :: VOCAB // 7] *= 40.0
    return x.to(torch.bfloat16)


VALUES = {"every_bf16": lambda rows, seed: every_bf16_value(rows), "served": served_logits}


def assert_bitwise(got: torch.Tensor, want: torch.Tensor) -> None:
    assert got.dtype == torch.float32 and got.shape == want.shape
    nan = torch.isnan(want)
    assert torch.equal(torch.isnan(got), nan)
    mismatches = int((got[~nan].view(torch.int32) != want[~nan].view(torch.int32)).sum())
    assert mismatches == 0, f"{mismatches} values differ from the torch chain's"


def stub_head(multiplier: float, soft_cap: float) -> target_head.TargetHead:
    """A ``TargetHead`` with only the state ``materialize`` reads (no library, no workspace)."""
    head = target_head.TargetHead.__new__(target_head.TargetHead)
    head.multiplier = float(multiplier)
    head.soft_cap = float(soft_cap)
    head.pending = None
    head.pending_inbox = False
    head.pending_local = None
    head.pending_s2 = False
    head.lm_head = types.SimpleNamespace(mapping=object(), gather_output_sizes=None)
    return head


class FakeGather:
    """The built-in all-gather of a ``[rows, SHARD]`` LM-head output: rank 0's shard (``local``)
    followed by the other ranks' shards of the row ``set_row`` split."""

    def __init__(self, head: target_head.TargetHead):
        self.head = head
        self.others = None
        self.calls = 0

    def set_row(self, full: torch.Tensor) -> torch.Tensor:
        """Splits ``full`` into the ranks' shards; returns rank 0's."""
        shards = [full[:, r * SHARD : (r + 1) * SHARD].contiguous() for r in range(WORLD)]
        self.others = shards[1:]
        return shards[0]

    def __call__(self, local, mapping, sizes=None):
        assert mapping is self.head.lm_head.mapping
        assert sizes is self.head.lm_head.gather_output_sizes
        self.calls += 1
        return torch.cat([local, *self.others], dim=-1)


def pend(head, gather, logits: torch.Tensor, gathered: bool) -> torch.Tensor:
    """Leaves ``logits`` pending as the step's row; returns the object the worker passes to
    ``materialize`` (the row itself, or a vocabulary-parallel step's unwritten placeholder)."""
    if not gathered:
        head.pending = logits
        return logits
    placeholder = torch.full_like(logits, float("nan"))
    head.pending = placeholder
    head.pending_local = gather.set_row(logits)
    return placeholder


@pytest.fixture
def fake_gather(monkeypatch):
    def make(head):
        gather = FakeGather(head)
        monkeypatch.setattr(trtllm_distributed, "allgather", gather)
        return gather

    return make


@pytest.mark.parametrize("gathered", [False, True], ids=["whole", "vocab_parallel"])
@pytest.mark.parametrize("values", sorted(VALUES))
@pytest.mark.parametrize("multiplier,soft_cap", CONSTANTS, ids=CONSTANT_IDS)
@pytest.mark.parametrize("rows", ROWS)
def test_materialize_matches_the_torch_chain(
    rows, multiplier, soft_cap, values, gathered, fake_gather
):
    head = stub_head(multiplier, soft_cap)
    gather = fake_gather(head)
    logits = VALUES[values](rows, rows)
    want = head_sampler.reference_capped(logits, multiplier, soft_cap)
    arg = pend(head, gather, logits, gathered)
    got = head.materialize(arg)
    assert head.pending is None and head.pending_local is None
    assert gather.calls == int(gathered)
    assert got.data_ptr() not in (logits.data_ptr(), arg.data_ptr())
    assert_bitwise(got, want)


@pytest.mark.parametrize("gathered", [False, True], ids=["whole", "vocab_parallel"])
@pytest.mark.parametrize("rows", [1, 65, 96, 768])
def test_materialize_in_a_cuda_graph(rows, gathered, fake_gather):
    multiplier, soft_cap = CONSTANTS[0]
    head = stub_head(multiplier, soft_cap)
    gather = fake_gather(head)
    static = served_logits(rows, 0)

    # Warm up (compiles the kernel) on a side stream, then capture.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        head.materialize(pend(head, gather, static, gathered))
    torch.cuda.current_stream().wait_stream(side)

    arg = pend(head, gather, static, gathered)
    local = head.pending_local
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = head.materialize(arg)
    assert head.pending is None and head.pending_local is None

    for seed, values in ((1, "served"), (2, "every_bf16"), (3, "served")):
        fresh = VALUES[values](rows, seed)
        if gathered:
            # The captured gather reads rank 0's shard and the other ranks' shards in place.
            local.copy_(fresh[:, :SHARD])
            for r, shard in enumerate(gather.others, start=1):
                shard.copy_(fresh[:, r * SHARD : (r + 1) * SHARD])
        else:
            static.copy_(fresh)
        graph.replay()
        torch.cuda.synchronize()
        assert_bitwise(out, head_sampler.reference_capped(fresh, multiplier, soft_cap))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
