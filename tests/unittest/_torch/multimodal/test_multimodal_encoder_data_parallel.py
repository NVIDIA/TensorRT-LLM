# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.models.multimodal_encoder_data_parallel import (
    EncoderParallelLayout,
    execute_encoder_dp_items,
    plan_encoder_dp_items,
)
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


def layout(dp_rank: int, dp_size: int = 2) -> EncoderParallelLayout:
    return EncoderParallelLayout(Mapping(), dp_size=dp_size, dp_rank=dp_rank, distributes_items=True)


def test_plan_encoder_dp_items_accepts_an_empty_batch():
    assert plan_encoder_dp_items([], num_ranks=2) == ()


def test_plan_encoder_dp_items_uses_input_cost():
    # LPT: the large item takes rank 0, both small items then go to rank 1.
    assert plan_encoder_dp_items([100, 10, 10], num_ranks=2) == (0, 1, 1)
    # Equal costs alternate ranks; ties go to the lower rank.
    assert plan_encoder_dp_items([1, 1, 1, 1], num_ranks=2) == (0, 1, 0, 1)


def test_execute_encoder_dp_items_reconstructs_global_output():
    costs, row_counts = [3, 4, 2], [2, 1, 2]
    peer_output = torch.tensor([[10.0], [10.0], [30.0], [30.0]])
    encoded_indices: list[int] = []

    def encode_local(indices):
        encoded_indices.extend(indices)
        return torch.cat(
            [torch.full((row_counts[index], 1), float((index + 1) * 10)) for index in indices]
        )

    def allgather(tensor, rank_row_counts):
        assert rank_row_counts == [1, 4]
        assert tensor.shape == (1, 1)
        return torch.cat([tensor, peer_output])

    output = execute_encoder_dp_items(
        costs,
        row_counts,
        layout=layout(dp_rank=0),
        encode_local=encode_local,
        any_rank_failed=lambda local_failed: local_failed,
        allgather=allgather,
        empty_output=torch.empty((0, 1)),
    )

    assert encoded_indices == [1]
    torch.testing.assert_close(
        output,
        torch.tensor([[10.0], [10.0], [20.0], [30.0], [30.0]]),
    )


def test_execute_encoder_dp_items_synchronizes_local_shape_failure():
    collective_dtypes: list[torch.dtype] = []
    statuses: list[bool] = []

    def any_rank_failed(local_failed):
        statuses.append(local_failed)
        return local_failed

    def allgather(tensor, row_counts):
        collective_dtypes.append(tensor.dtype)
        return tensor

    with pytest.raises(RuntimeError, match="rank failed") as error:
        execute_encoder_dp_items(
            [1],
            [2],
            layout=layout(dp_rank=0, dp_size=1),
            encode_local=lambda indices: torch.zeros((1, 1)),
            any_rank_failed=any_rank_failed,
            allgather=allgather,
            empty_output=torch.empty((0, 1)),
        )

    assert isinstance(error.value.__cause__, ValueError)
    assert statuses == [True]
    assert collective_dtypes == []


def test_execute_encoder_dp_items_keeps_empty_rank_in_collectives():
    peer_contribution = torch.full((2, 1), 7.0)
    collective_dtypes: list[torch.dtype] = []
    statuses: list[bool] = []

    def any_rank_failed(local_failed):
        statuses.append(local_failed)
        return local_failed

    def allgather(tensor, row_counts):
        collective_dtypes.append(tensor.dtype)
        assert tensor.shape == (0, 1)
        assert row_counts == [2, 0]
        return peer_contribution

    output = execute_encoder_dp_items(
        [1],
        [2],
        layout=layout(dp_rank=1),
        encode_local=lambda _: pytest.fail("An empty encoder-DP rank must not run the encoder."),
        any_rank_failed=any_rank_failed,
        allgather=allgather,
        empty_output=torch.empty((0, 1)),
    )

    torch.testing.assert_close(output, peer_contribution)
    assert statuses == [False]
    assert collective_dtypes == [torch.float32]


def test_execute_encoder_dp_items_stops_before_output_collective_on_peer_failure():
    def allgather(tensor, row_counts):
        pytest.fail("Output collective must not run after a peer failure.")

    with pytest.raises(RuntimeError, match="peer rank failed"):
        execute_encoder_dp_items(
            [1],
            [1],
            layout=layout(dp_rank=1),
            encode_local=lambda _: torch.empty((0, 1)),
            any_rank_failed=lambda local_failed: True,
            allgather=allgather,
            empty_output=torch.empty((0, 1)),
        )
