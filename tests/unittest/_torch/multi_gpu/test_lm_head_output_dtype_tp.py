# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify FP32 LM-head logits through a real tensor-parallel all-gather."""

import pickle
import sys

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

import tensorrt_llm
from tensorrt_llm._torch.modules.embedding import LMHead
from tensorrt_llm._torch.modules.linear import TensorParallelMode
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

# The shared MPI executor keeps its worker thread alive between tests.
pytestmark = pytest.mark.threadleak(enabled=False)


@torch.inference_mode()
def _run_column_lm_head(
    tensor_parallel_size: int, hidden: torch.Tensor, weight: torch.Tensor
) -> bool:
    """Check the gathered vocabulary and FP32 precision on every MPI rank."""
    assert tensorrt_llm.mpi_world_size() == tensor_parallel_size
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    vocab_size, hidden_size = weight.shape
    lm_head = LMHead(
        num_embeddings=vocab_size,
        embedding_dim=hidden_size,
        dtype=weight.dtype,
        mapping=Mapping(world_size=tensor_parallel_size, tp_size=tensor_parallel_size, rank=rank),
        tensor_parallel_mode=TensorParallelMode.COLUMN,
        gather_output=True,
        output_dtype=torch.float32,
    )
    lm_head.load_weights([{"weight": weight}])
    lm_head.cuda()
    hidden = hidden.cuda()

    logits = lm_head(hidden)
    reference = hidden.double() @ weight.cuda().double().t()

    assert lm_head.weight.dtype == hidden.dtype == weight.dtype
    assert logits.dtype == torch.float32
    assert logits.shape == (hidden.shape[0], vocab_size)
    torch.testing.assert_close(logits.double(), reference, rtol=1e-4, atol=1e-4)
    assert not torch.equal(logits, logits.to(weight.dtype).float())
    return True


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
@pytest.mark.parametrize("vocab_size", [128, 127], ids=["balanced", "padded"])
@pytest.mark.parametrize("num_tokens", [1, 7])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_lm_head_float32_column_gather(
    mpi_pool_executor: MPIPoolExecutor,
    vocab_size: int,
    num_tokens: int,
    dtype: torch.dtype,
) -> None:
    """Preserve FP32 logits and vocabulary order across two padded or balanced shards."""
    hidden_size = 128
    generator = torch.Generator().manual_seed(42)
    hidden = torch.randn(num_tokens, hidden_size, generator=generator, dtype=dtype)
    weight = (torch.randn(vocab_size, hidden_size, generator=generator) * 4 / hidden_size**0.5).to(
        dtype
    )
    tensor_parallel_size = mpi_pool_executor.num_workers
    assert tensor_parallel_size == 2
    arguments = [(tensor_parallel_size, hidden, weight)] * tensor_parallel_size
    results = mpi_pool_executor.map(_run_column_lm_head, *zip(*arguments))
    assert list(results) == [True] * tensor_parallel_size
