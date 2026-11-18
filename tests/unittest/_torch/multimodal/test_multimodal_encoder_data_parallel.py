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

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

import tensorrt_llm._torch.models.modeling_multimodal_mixin as mixin_module
from tensorrt_llm._torch.distributed import communicator
from tensorrt_llm._torch.models.modeling_multimodal_mixin import MultimodalModelMixin
from tensorrt_llm._torch.models.multimodal_encoder_data_parallel import (
    EncoderDpError,
    EncoderDpStatus,
    EncoderParallelLayout,
    MultimodalEncoderContractError,
    encoder_dp_error_guard,
    execute_encoder_dp_items,
    plan_encoder_dp_items,
)
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


def layout(dp_rank: int, dp_size: int = 2) -> EncoderParallelLayout:
    return EncoderParallelLayout(
        Mapping(), dp_size=dp_size, dp_rank=dp_rank, distributes_items=True
    )


def exchange_with_peer(peer_status: int, statuses: list[int] | None = None):
    """A 2-rank status exchange seen from rank 0. The peer reports ``peer_status``."""

    def exchange(values):
        if statuses is not None:
            statuses.append(values[0])
        return [values, [peer_status]]

    return exchange


def own_status(values):
    return [values]


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

    def prepare_local(indices):
        encoded_indices.extend(indices)
        return lambda: torch.cat(
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
        prepare_local=prepare_local,
        exchange_status=own_status,
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
    statuses: list[int] = []

    def allgather(tensor, row_counts):
        collective_dtypes.append(tensor.dtype)
        return tensor

    # A wrong row count is a request error: every rank raises one, and the
    # output gather does not run.
    with pytest.raises(MultimodalEncoderContractError, match="does not match"):
        execute_encoder_dp_items(
            [1],
            [2],
            layout=layout(dp_rank=0, dp_size=1),
            prepare_local=lambda indices: lambda: torch.zeros((1, 1)),
            exchange_status=exchange_with_peer(EncoderDpStatus.OK, statuses),
            allgather=allgather,
            empty_output=torch.empty((0, 1)),
        )

    assert statuses == [EncoderDpStatus.OK, EncoderDpStatus.REQUEST_ERROR]
    assert collective_dtypes == []


def test_execute_encoder_dp_items_keeps_empty_rank_in_collectives():
    peer_contribution = torch.full((2, 1), 7.0)
    collective_dtypes: list[torch.dtype] = []
    statuses: list[int] = []

    def allgather(tensor, row_counts):
        collective_dtypes.append(tensor.dtype)
        assert tensor.shape == (0, 1)
        assert row_counts == [2, 0]
        return peer_contribution

    output = execute_encoder_dp_items(
        [1],
        [2],
        layout=layout(dp_rank=1),
        prepare_local=lambda _: pytest.fail("An empty encoder-DP rank must not run the encoder."),
        exchange_status=exchange_with_peer(EncoderDpStatus.OK, statuses),
        allgather=allgather,
        empty_output=torch.empty((0, 1)),
    )

    torch.testing.assert_close(output, peer_contribution)
    # One exchange after preparing and one after encoding.
    assert statuses == [EncoderDpStatus.OK, EncoderDpStatus.OK]
    assert collective_dtypes == [torch.float32]


@pytest.mark.parametrize(
    "peer_status,expected",
    [
        (EncoderDpStatus.REQUEST_ERROR, MultimodalEncoderContractError),
        (EncoderDpStatus.SYSTEM_ERROR, EncoderDpError),
    ],
)
def test_execute_encoder_dp_items_stops_before_encoding_on_peer_prepare_failure(
    peer_status, expected
):
    # A peer of this replica failed to prepare, so this rank must not start
    # encoder TP work that the peer would never enter.
    encode = Mock()

    with pytest.raises(expected, match=r"rank\(s\) \[1\]"):
        execute_encoder_dp_items(
            [1],
            [1],
            layout=layout(dp_rank=0),
            prepare_local=lambda _: encode,
            exchange_status=exchange_with_peer(peer_status),
            allgather=lambda *_: pytest.fail("Output collective must not run after a failure."),
            empty_output=torch.empty((0, 1)),
        )
    encode.assert_not_called()


def test_guard_shares_a_system_error_as_the_same_error_on_every_rank():
    failure = MemoryError("rank 0 out of memory")
    with pytest.raises(EncoderDpError) as failed_rank:
        with encoder_dp_error_guard(exchange_with_peer(EncoderDpStatus.OK)):
            raise failure
    with pytest.raises(EncoderDpError) as peer_rank:
        with encoder_dp_error_guard(lambda values: [[EncoderDpStatus.SYSTEM_ERROR], values]):
            pass

    assert (
        str(failed_rank.value)
        == str(peer_rank.value)
        == ("Multimodal encoder failed on TP rank(s) [0].")
    )
    assert failed_rank.value.__cause__ is failure
    assert peer_rank.value.__cause__ is None


def test_guard_keeps_a_local_request_error_and_names_failed_peers():
    failure = MultimodalEncoderContractError("bad item")
    with pytest.raises(MultimodalEncoderContractError) as failed_rank:
        with encoder_dp_error_guard(exchange_with_peer(EncoderDpStatus.OK)):
            raise failure
    with pytest.raises(MultimodalEncoderContractError, match=r"rank\(s\) \[1\]"):
        with encoder_dp_error_guard(exchange_with_peer(EncoderDpStatus.REQUEST_ERROR)):
            pass

    assert failed_rank.value is failure


def test_guard_reraises_an_inner_guard_error_without_another_exchange():
    exchange = Mock(side_effect=own_status)
    with pytest.raises(EncoderDpError) as error:
        with encoder_dp_error_guard(exchange):
            with encoder_dp_error_guard(exchange):
                raise MemoryError("encoder OOM")

    # Only the inner guard exchanged; the outer one re-raised its error as is.
    exchange.assert_called_once()
    assert isinstance(error.value.__cause__, MemoryError)


def test_encoder_status_exchange_does_not_need_cuda():
    mapping = Mapping()
    model = SimpleNamespace(model_config=SimpleNamespace(mapping=mapping))
    with patch.object(mixin_module, "Distributed") as distributed:
        MultimodalModelMixin.exchange_encoder_status(model, [1, 0])
    distributed.get.assert_called_once_with(mapping)
    distributed.get.return_value.tp_allgather_int64.assert_called_once_with([1, 0], prefer_cpu=True)

    # TorchDist then uses the TP group's CPU backend, even when the CUDA path exists.
    torch_dist = object.__new__(communicator.TorchDist)
    torch_dist.mapping = SimpleNamespace(tp_group_pg=SimpleNamespace(size=lambda: 2))
    torch_dist._cuda_obj_pg = Mock(return_value=object())
    torch_dist._cuda_allgather_cpu_tensor = Mock(side_effect=AssertionError("used CUDA"))

    def cpu_all_gather(rows, tensor, group):
        assert group is torch_dist.mapping.tp_group_pg
        assert tensor.device.type == "cpu"
        for rank, row in enumerate(rows):
            row.copy_(tensor + rank)

    with patch.object(communicator.dist, "all_gather", cpu_all_gather):
        rows = torch_dist.tp_allgather_int64([1, 0], prefer_cpu=True)
    np.testing.assert_array_equal(rows, [[1, 0], [2, 1]])
