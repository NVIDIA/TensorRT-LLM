# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pickle
import sys

import cloudpickle
import pytest
import torch
from mpi4py import MPI

import tensorrt_llm
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_multimodal_mixin import MultimodalModelMixin
from tensorrt_llm.inputs.multimodal import MultimodalParams
from tensorrt_llm.llmapi.llm_args import MultimodalConfig
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

pytestmark = pytest.mark.threadleak(enabled=False)


class _EncoderDpModel(torch.nn.Module, MultimodalModelMixin):
    def __init__(self, rank: int, *, enable_attention_dp: bool) -> None:
        super().__init__()
        self.model_config = ModelConfig(
            mapping=Mapping(
                world_size=2,
                rank=rank,
                tp_size=2,
                enable_attention_dp=enable_attention_dp,
            ),
            multimodal_config=MultimodalConfig(
                encoder_data_parallel_size=1 if enable_attention_dp else 2,
                encoder_cache_max_bytes=0,
            ),
        )
        self.embedding = torch.nn.Embedding(
            num_embeddings=8,
            embedding_dim=2,
            dtype=torch.bfloat16,
            device="cuda",
        )
        self.encoded_item_ids: list[int] = []

    @property
    def text_embedding_layer(self) -> torch.nn.Embedding:
        return self.embedding

    @property
    def embedding_dim(self) -> int:
        return self.embedding.embedding_dim

    @property
    def embedding_dtype(self) -> torch.dtype:
        return self.embedding.weight.dtype

    def encode_multimodal_inputs(
        self,
        multimodal_params: list[MultimodalParams],
        **encoder_kwargs,
    ) -> torch.Tensor:
        del encoder_kwargs
        rows = []
        for param in multimodal_params:
            item_ids = param.multimodal_data["image"]["pixel_values"][:, 0, 0, 0]
            lengths = param.multimodal_data["multimodal_embedding_lengths"]
            self.encoded_item_ids.extend(int(item_id) for item_id in item_ids)
            for item_id, length in zip(item_ids, lengths, strict=True):
                rows.append(
                    torch.full(
                        (length, self.embedding_dim),
                        float(item_id),
                        dtype=self.embedding_dtype,
                        device="cuda",
                    )
                )
        return torch.cat(rows, dim=0)


def _make_param(item_ids: list[int], lengths: list[int]) -> MultimodalParams:
    return MultimodalParams(
        multimodal_data={
            "image": {
                "pixel_values": torch.tensor(item_ids, dtype=torch.float32).reshape(-1, 1, 1, 1),
                "image_sizes": [[1, 1] for _ in item_ids],
            },
            "multimodal_embedding_lengths": lengths,
        }
    )


@torch.inference_mode()
def _run_encoder_dp_rank(enable_attention_dp: bool):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    model = _EncoderDpModel(rank, enable_attention_dp=enable_attention_dp)

    if enable_attention_dp:
        param = _make_param([10 + rank], [rank + 1])
    else:
        param = _make_param([10, 20, 30], [4, 1, 3])

    output = model._run_multimodal_encoder([param])
    torch.cuda.synchronize()
    return rank, model.encoded_item_ids, output.cpu()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_encoder_dp_partitions_and_reconstructs_with_nccl(mpi_pool_executor):
    results = mpi_pool_executor.map(_run_encoder_dp_rank, [False, False])
    results = sorted(results)

    assert results[0][1] == [10]
    assert results[1][1] == [20, 30]
    expected = (
        torch.tensor(
            [10.0] * 4 + [20.0] + [30.0] * 3,
            dtype=torch.bfloat16,
        )
        .unsqueeze(1)
        .repeat(1, 2)
    )
    for _, _, output in results:
        torch.testing.assert_close(output, expected)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_attention_dp_encodes_rank_local_requests(mpi_pool_executor):
    results = mpi_pool_executor.map(_run_encoder_dp_rank, [True, True])
    results = sorted(results)

    assert results[0][1] == [10]
    assert results[1][1] == [11]
    torch.testing.assert_close(
        results[0][2],
        torch.full((1, 2), 10.0, dtype=torch.bfloat16),
    )
    torch.testing.assert_close(
        results[1][2],
        torch.full((2, 2), 11.0, dtype=torch.bfloat16),
    )
