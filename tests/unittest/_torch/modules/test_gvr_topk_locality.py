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
"""CPU tests for Rubin locality-domain GVR row sharding."""

import sys
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.locality_domain.gvr_topk import plan_gvr_topk_row_shards
from tensorrt_llm._torch.modules.top_k import TopK, TopKImplementation


def test_plan_keeps_every_next_n_request_group_together() -> None:
    plan = plan_gvr_topk_row_shards(
        num_rows=15,
        next_n=3,
        score_width=4096,
        top_k=512,
        topology=((104, 208), (104, 208)),
        min_total_score_elements=0,
        min_score_elements_per_sm=0,
    )

    assert plan is not None
    assert [(shard.request_start, shard.request_end) for shard in plan.shards] == [
        (0, 3),
        (3, 5),
    ]
    assert [(shard.row_start, shard.row_end) for shard in plan.shards] == [
        (0, 9),
        (9, 15),
    ]
    assert all(shard.row_start % 3 == 0 and shard.row_end % 3 == 0 for shard in plan.shards)


def test_plan_uses_real_asymmetric_partition_sm_counts() -> None:
    plan = plan_gvr_topk_row_shards(
        num_rows=10,
        next_n=1,
        score_width=4096,
        top_k=512,
        topology=((64, 208), (128, 208)),
        min_total_score_elements=0,
        min_score_elements_per_sm=0,
    )

    assert plan is not None
    assert [shard.num_requests for shard in plan.shards] == [3, 7]
    assert [shard.num_sms for shard in plan.shards] == [64, 128]


@pytest.mark.parametrize(
    "kwargs",
    [
        # A single request cannot be split without breaking the local-row ABI.
        {"num_rows": 4, "next_n": 4, "score_width": 1 << 18},
        # The total launch is too small to amortize two launches and events.
        {"num_rows": 2, "next_n": 1, "score_width": 4096},
        # Total work passes, but the smaller shard has too little work per SM.
        {"num_rows": 4, "next_n": 1, "score_width": 1 << 18},
    ],
)
def test_provisional_gain_gate_keeps_unprofitable_shapes_full_device(kwargs) -> None:
    topology = ((8, 208), (200, 208)) if kwargs["num_rows"] == 4 else ((104, 208), (104, 208))
    assert (
        plan_gvr_topk_row_shards(
            top_k=512,
            topology=topology,
            **kwargs,
        )
        is None
    )


@pytest.mark.parametrize(
    "topology,error",
    [
        (((104, 208), (104, 210)), "disagree"),
        (((120, 208), (120, 208)), "overlap"),
    ],
)
def test_invalid_topology_is_rejected_before_launch(topology, error) -> None:
    with pytest.raises(ValueError, match=error):
        plan_gvr_topk_row_shards(
            num_rows=8,
            next_n=1,
            score_width=1 << 18,
            top_k=512,
            topology=topology,
        )


class _FakeRuntime:
    def __init__(self) -> None:
        self.events: list[object] = []
        self.current_partition: int | None = None

    def fork(self) -> None:
        self.events.append("fork")

    @contextmanager
    def partition_context(self, partition_id: int):
        self.events.append(("enter", partition_id))
        self.current_partition = partition_id
        try:
            yield
        finally:
            self.current_partition = None
            self.events.append(("exit", partition_id))

    def join(self) -> None:
        self.events.append("join")


def _install_fake_selfsampling_runner(monkeypatch) -> Mock:
    runner = Mock()
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm._torch.cute_dsl_kernels.blackwell.top_k",
        SimpleNamespace(selfsampling_topk_run_varlen=runner),
    )
    return runner


def test_topk_dispatches_disjoint_request_slices_and_joins(monkeypatch) -> None:
    runtime = _FakeRuntime()
    plan = plan_gvr_topk_row_shards(
        num_rows=6,
        next_n=2,
        score_width=8,
        top_k=2,
        topology=((1, 3), (2, 3)),
        min_total_score_elements=0,
        min_score_elements_per_sm=0,
    )
    assert plan is not None
    launch_key = ("cpu-mock",)
    monkeypatch.setattr(
        TopK,
        "_build_gvr_locality_launch",
        lambda self, scores, next_n, max_seq_len: (runtime, plan, launch_key),
    )
    runner = _install_fake_selfsampling_runner(monkeypatch)

    def run_slice(scores, lengths, output, **kwargs) -> None:
        partition_id = runtime.current_partition
        runtime.events.append(("launch", partition_id, scores.shape[0], lengths.tolist()))
        output.fill_(10 + int(partition_id))

    runner.side_effect = run_slice
    top_k = TopK(
        2,
        decode_implementation=TopKImplementation.CUTE_DSL_GVR,
        compress_ratio=4,
        use_gvr_locality_domain=True,
    )
    scores = torch.randn(6, 8)
    lengths = torch.tensor([32, 48, 64], dtype=torch.int32)
    output = torch.empty(6, 2, dtype=torch.int32)

    result = top_k(
        scores,
        output,
        is_prefill=False,
        sequence_lengths=lengths,
        scan_lengths=lengths,
        next_n=2,
        max_seq_len=16,
    )

    assert result is output
    assert runtime.events == [
        "fork",
        ("enter", 0),
        ("launch", 0, 2, [32]),
        ("exit", 0),
        ("enter", 1),
        ("launch", 1, 4, [48, 64]),
        ("exit", 1),
        "join",
    ]
    assert output[:2].eq(10).all()
    assert output[2:].eq(11).all()
    assert top_k._gvr_locality_ready_launches == {launch_key}
    assert runner.call_count == 2
    for call in runner.call_args_list:
        assert call.kwargs == {"next_n": 2, "compress_ratio": 4, "max_seq_len": 64}


def test_topk_joins_and_does_not_mark_ready_after_shard_failure(monkeypatch) -> None:
    runtime = _FakeRuntime()
    plan = plan_gvr_topk_row_shards(
        num_rows=2,
        next_n=1,
        score_width=8,
        top_k=2,
        topology=((1, 2), (1, 2)),
        min_total_score_elements=0,
        min_score_elements_per_sm=0,
    )
    assert plan is not None
    monkeypatch.setattr(
        TopK,
        "_build_gvr_locality_launch",
        lambda self, scores, next_n, max_seq_len: (runtime, plan, ("failed",)),
    )
    runner = _install_fake_selfsampling_runner(monkeypatch)
    runner.side_effect = RuntimeError("synthetic launch failure")
    top_k = TopK(
        2,
        decode_implementation=TopKImplementation.CUTE_DSL_GVR,
        use_gvr_locality_domain=True,
    )

    with pytest.raises(RuntimeError, match="synthetic"):
        top_k(
            torch.randn(2, 8),
            torch.empty(2, 2, dtype=torch.int32),
            is_prefill=False,
            sequence_lengths=torch.tensor([8, 8], dtype=torch.int32),
            scan_lengths=torch.tensor([8, 8], dtype=torch.int32),
            max_seq_len=8,
        )

    assert runtime.events[-1] == "join"
    assert not top_k._gvr_locality_ready_launches


def test_topk_preserves_launch_failure_if_join_also_fails(monkeypatch) -> None:
    runtime = _FakeRuntime()
    plan = plan_gvr_topk_row_shards(
        num_rows=2,
        next_n=1,
        score_width=8,
        top_k=2,
        topology=((1, 2), (1, 2)),
        min_total_score_elements=0,
        min_score_elements_per_sm=0,
    )
    assert plan is not None
    monkeypatch.setattr(
        TopK,
        "_build_gvr_locality_launch",
        lambda self, scores, next_n, max_seq_len: (runtime, plan, ("failed",)),
    )
    runner = _install_fake_selfsampling_runner(monkeypatch)
    runner.side_effect = RuntimeError("synthetic launch failure")
    runtime.join = Mock(side_effect=RuntimeError("synthetic join failure"))
    top_k = TopK(
        2,
        decode_implementation=TopKImplementation.CUTE_DSL_GVR,
        use_gvr_locality_domain=True,
    )

    with pytest.raises(RuntimeError, match="synthetic launch failure"):
        top_k(
            torch.randn(2, 8),
            torch.empty(2, 2, dtype=torch.int32),
            is_prefill=False,
            sequence_lengths=torch.tensor([8, 8], dtype=torch.int32),
            scan_lengths=torch.tensor([8, 8], dtype=torch.int32),
            max_seq_len=8,
        )

    runtime.join.assert_called_once_with()
    assert not top_k._gvr_locality_ready_launches


@pytest.mark.parametrize(
    "lengths_shape,output_shape,error",
    [
        ((3,), (2, 2), "sequence_lengths"),
        ((2,), (3, 2), "output shape"),
    ],
)
def test_topk_rejects_global_shapes_that_sharding_could_mask(
    monkeypatch, lengths_shape, output_shape, error
) -> None:
    runtime = _FakeRuntime()
    plan = plan_gvr_topk_row_shards(
        num_rows=2,
        next_n=1,
        score_width=8,
        top_k=2,
        topology=((1, 2), (1, 2)),
        min_total_score_elements=0,
        min_score_elements_per_sm=0,
    )
    assert plan is not None
    monkeypatch.setattr(
        TopK,
        "_build_gvr_locality_launch",
        lambda self, scores, next_n, max_seq_len: (runtime, plan, ("invalid",)),
    )
    _install_fake_selfsampling_runner(monkeypatch)
    top_k = TopK(
        2,
        decode_implementation=TopKImplementation.CUTE_DSL_GVR,
        use_gvr_locality_domain=True,
    )

    with pytest.raises(RuntimeError, match=error):
        top_k(
            torch.randn(2, 8),
            torch.empty(output_shape, dtype=torch.int32),
            is_prefill=False,
            sequence_lengths=torch.full(lengths_shape, 8, dtype=torch.int32),
            scan_lengths=torch.full(lengths_shape, 8, dtype=torch.int32),
            max_seq_len=8,
        )

    assert runtime.events == []


def test_cold_capture_fails_before_locality_runtime_initialization(monkeypatch) -> None:
    from tensorrt_llm._torch import locality_domain_utils

    class _FakeCudaScores:
        is_cuda = True
        shape = (4, 1 << 18)

        @staticmethod
        def get_device() -> int:
            return 0

    top_k = TopK(
        2,
        decode_implementation=TopKImplementation.CUTE_DSL_GVR,
        use_gvr_locality_domain=True,
    )
    # Capability discovery is an eager, non-allocating step. Pretend it has
    # already succeeded so this test isolates the capture lifecycle guard.
    top_k._gvr_locality_capability[0] = True
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(locality_domain_utils, "get_current_locality_domain", lambda: None)

    with pytest.raises(RuntimeError, match="resources are cold"):
        top_k._build_gvr_locality_launch(_FakeCudaScores(), next_n=1, max_seq_len=1 << 18)

    assert top_k._gvr_locality_runtime is None


def test_cold_capture_does_not_run_capability_discovery(monkeypatch) -> None:
    from tensorrt_llm._torch import locality_domain_utils

    class _FakeCudaScores:
        is_cuda = True
        shape = (4, 1 << 18)

        @staticmethod
        def get_device() -> int:
            return 0

    top_k = TopK(
        2,
        decode_implementation=TopKImplementation.CUTE_DSL_GVR,
        use_gvr_locality_domain=True,
    )
    properties = Mock(side_effect=AssertionError("must not query properties during capture"))
    supported = Mock(side_effect=AssertionError("must not query driver support during capture"))
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    monkeypatch.setattr(locality_domain_utils, "get_current_locality_domain", lambda: None)
    monkeypatch.setattr(locality_domain_utils, "is_locality_domain_supported", supported)

    with pytest.raises(RuntimeError, match="capability is cold"):
        top_k._build_gvr_locality_launch(_FakeCudaScores(), next_n=1, max_seq_len=1 << 18)

    properties.assert_not_called()
    supported.assert_not_called()


def test_explicit_opt_in_on_unsupported_cpu_uses_one_full_device_call(monkeypatch) -> None:
    runner = _install_fake_selfsampling_runner(monkeypatch)
    top_k = TopK(
        2,
        decode_implementation=TopKImplementation.CUTE_DSL_GVR,
        use_gvr_locality_domain=True,
    )
    scores = torch.randn(4, 8)
    lengths = torch.full((4,), 8, dtype=torch.int32)
    output = torch.empty(4, 2, dtype=torch.int32)

    top_k(
        scores,
        output,
        is_prefill=False,
        sequence_lengths=lengths,
        scan_lengths=lengths,
        max_seq_len=1 << 18,
    )

    runner.assert_called_once_with(
        scores,
        lengths,
        output,
        next_n=1,
        compress_ratio=1,
        max_seq_len=1 << 18,
    )


def test_default_off_does_not_call_locality_helper(monkeypatch) -> None:
    runner = _install_fake_selfsampling_runner(monkeypatch)
    locality_helper = Mock(side_effect=AssertionError("default path must not enter helper"))
    monkeypatch.setattr(TopK, "_run_gvr_locality_domain", locality_helper)
    top_k = TopK(
        2,
        decode_implementation=TopKImplementation.CUTE_DSL_GVR,
    )
    scores = torch.randn(4, 8)
    lengths = torch.full((4,), 8, dtype=torch.int32)
    output = torch.empty(4, 2, dtype=torch.int32)

    top_k(
        scores,
        output,
        is_prefill=False,
        sequence_lengths=lengths,
        scan_lengths=lengths,
        max_seq_len=1 << 18,
    )

    locality_helper.assert_not_called()
    runner.assert_called_once()


def test_temporal_gvr_never_enters_locality_row_sharding(monkeypatch) -> None:
    build_launch = Mock(side_effect=AssertionError("V1 must not shard"))
    monkeypatch.setattr(TopK, "_build_gvr_locality_launch", build_launch)
    temporal_runner = Mock()
    monkeypatch.setattr(
        torch.ops.trtllm,
        "cute_dsl_gvr_topk_decode",
        temporal_runner,
        raising=False,
    )
    top_k = TopK(
        2,
        decode_implementation=TopKImplementation.CUTE_DSL_GVR,
        gvr_self_sampling=False,
        use_gvr_locality_domain=True,
    )
    scores = torch.randn(2, 8)
    lengths = torch.full((2,), 8, dtype=torch.int32)

    top_k(
        scores,
        torch.empty(2, 2, dtype=torch.int32),
        is_prefill=False,
        sequence_lengths=lengths,
        scan_lengths=lengths,
        max_seq_len=1 << 18,
        gvr_ext_kwargs={"gvr_prior_indices": torch.zeros(2, 2, dtype=torch.int32)},
    )

    build_launch.assert_not_called()
    temporal_runner.assert_called_once()
