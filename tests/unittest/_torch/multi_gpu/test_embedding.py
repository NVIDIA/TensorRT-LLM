import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from torch import nn

import tensorrt_llm
from tensorrt_llm._torch.modules.embedding import Embedding, LMHead
from tensorrt_llm._torch.modules.linear import TensorParallelMode
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

# needed since we reuse the mpi executor pool, first test running will leak a thread
pytestmark = pytest.mark.threadleak(enabled=False)


def run_single_rank(tensor_parallel_size, single_rank_forward_func, input,
                    weights, vocab_size, hidden_size, dtype):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        single_rank_forward_func(input, vocab_size, hidden_size, dtype,
                                 tensor_parallel_size, rank, weights)
    except Exception:
        traceback.print_exc()
        raise
    return True


@torch.inference_mode
def column_embedding_forward(x, vocab_size, hidden_size, dtype,
                             tensor_parallel_size, tensor_parallel_rank,
                             weight):

    x = x.cuda()
    embedding = Embedding(
        num_embeddings=vocab_size,
        embedding_dim=hidden_size,
        dtype=dtype,
        mapping=Mapping(
            world_size=tensor_parallel_size,
            tp_size=tensor_parallel_size,
            rank=tensor_parallel_rank,
        ),
        tensor_parallel_mode=TensorParallelMode.COLUMN,
    )
    embedding.load_weights([dict(weight=weight)])
    embedding.cuda()

    output = embedding.forward(x)

    # torch run
    embedding = nn.Embedding(
        num_embeddings=vocab_size,
        embedding_dim=hidden_size,
        dtype=dtype,
    )
    embedding.weight.data.copy_(weight)
    embedding.cuda()

    torch_output = embedding.forward(x)

    # compare
    torch.cuda.synchronize()
    torch.testing.assert_close(output, torch_output)


@torch.inference_mode
def row_embedding_forward(x, vocab_size, hidden_size, dtype,
                          tensor_parallel_size, tensor_parallel_rank, weight):

    x = x.cuda()
    embedding = Embedding(
        num_embeddings=vocab_size,
        embedding_dim=hidden_size,
        dtype=dtype,
        mapping=Mapping(
            world_size=tensor_parallel_size,
            tp_size=tensor_parallel_size,
            rank=tensor_parallel_rank,
        ),
        tensor_parallel_mode=TensorParallelMode.ROW,
        gather_output=True,
    )
    embedding.load_weights([dict(weight=weight)])
    embedding.cuda()

    output = embedding.forward(x)

    # torch run
    embedding = nn.Embedding(
        num_embeddings=vocab_size,
        embedding_dim=hidden_size,
        dtype=dtype,
    )
    embedding.weight.data.copy_(weight)
    embedding.cuda()

    torch_output = embedding.forward(x)

    # compare
    torch.cuda.synchronize()
    torch.testing.assert_close(output, torch_output)


@torch.inference_mode
def column_lm_head_forward(x, vocab_size, hidden_size, dtype,
                           tensor_parallel_size, tensor_parallel_rank, weight):

    x = x.cuda()
    lm_head = LMHead(
        num_embeddings=vocab_size,
        embedding_dim=hidden_size,
        dtype=dtype,
        mapping=Mapping(
            world_size=tensor_parallel_size,
            tp_size=tensor_parallel_size,
            rank=tensor_parallel_rank,
        ),
        tensor_parallel_mode=TensorParallelMode.COLUMN,
        gather_output=True,
    )
    lm_head.load_weights([dict(weight=weight)])
    lm_head.cuda()

    output = lm_head.forward(x)

    # torch run
    lm_head = nn.Linear(
        in_features=hidden_size,
        out_features=vocab_size,
        bias=False,
        dtype=dtype,
    )
    lm_head.weight.data.copy_(weight)
    lm_head.cuda()

    torch_output = lm_head.forward(x)

    # compare
    torch.cuda.synchronize()
    torch.testing.assert_close(output, torch_output)


@torch.inference_mode
def row_lm_head_forward(x, vocab_size, hidden_size, dtype, tensor_parallel_size,
                        tensor_parallel_rank, weight):

    x = x.cuda()
    lm_head = LMHead(
        num_embeddings=vocab_size,
        embedding_dim=hidden_size,
        dtype=dtype,
        mapping=Mapping(
            world_size=tensor_parallel_size,
            tp_size=tensor_parallel_size,
            rank=tensor_parallel_rank,
        ),
        tensor_parallel_mode=TensorParallelMode.ROW,
    )
    lm_head.load_weights([dict(weight=weight)])
    lm_head.cuda()

    xs = torch.chunk(x, 2, dim=-1)
    output = lm_head.forward(xs[tensor_parallel_rank])

    # torch run
    lm_head = nn.Linear(
        in_features=hidden_size,
        out_features=vocab_size,
        bias=False,
        dtype=dtype,
    )
    lm_head.weight.data.copy_(weight)
    lm_head.cuda()

    torch_output = lm_head.forward(x)

    # compare
    torch.cuda.synchronize()
    torch.testing.assert_close(output, torch_output, rtol=0.05, atol=0.05)


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason='needs 2 GPUs to run this test')
@pytest.mark.parametrize("vocab_size", [128, 127],
                         ids=["balanced", "unbalanced"])
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_column_embedding(vocab_size, mpi_pool_executor):
    torch.manual_seed(42)
    seq_len = 10
    hidden_size = 16
    dtype = torch.bfloat16
    tensor_parallel_size = mpi_pool_executor.num_workers
    input = torch.randint(0, vocab_size, (seq_len, ))
    weight = torch.randn((vocab_size, hidden_size), dtype=dtype)
    results = mpi_pool_executor.map(
        run_single_rank,
        *zip(*[(tensor_parallel_size, column_embedding_forward, input, weight,
                vocab_size, hidden_size, dtype)] * 2))
    for r in results:
        assert r is True


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason='needs 2 GPUs to run this test')
@pytest.mark.parametrize("hidden_size", [16, 15],
                         ids=["balanced", "unbalanced"])
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_row_embedding(hidden_size, mpi_pool_executor):
    torch.manual_seed(42)
    seq_len = 2
    vocab_size = 128
    dtype = torch.bfloat16
    tensor_parallel_size = mpi_pool_executor.num_workers
    input = torch.randint(0, vocab_size, (seq_len, ))
    weight = torch.randn((vocab_size, hidden_size), dtype=dtype)
    results = mpi_pool_executor.map(
        run_single_rank,
        *zip(*[(tensor_parallel_size, row_embedding_forward, input, weight,
                vocab_size, hidden_size, dtype)] * 2))
    for r in results:
        assert r is True


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason='needs 2 GPUs to run this test')
@pytest.mark.parametrize("vocab_size", [128, 127],
                         ids=["balanced", "unbalanced"])
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_column_lm_head(vocab_size, mpi_pool_executor):
    torch.manual_seed(42)
    seq_len = 10
    hidden_size = 16
    dtype = torch.bfloat16
    tensor_parallel_size = mpi_pool_executor.num_workers
    input = torch.randn((seq_len, hidden_size), dtype=dtype)
    weight = torch.randn((vocab_size, hidden_size), dtype=dtype)
    results = mpi_pool_executor.map(
        run_single_rank,
        *zip(*[(tensor_parallel_size, column_lm_head_forward, input, weight,
                vocab_size, hidden_size, dtype)] * 2))
    for r in results:
        assert r is True


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason='needs 2 GPUs to run this test')
@pytest.mark.parametrize("hidden_size", [16, 15],
                         ids=["balanced", "unbalanced"])
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_row_lm_head(hidden_size, mpi_pool_executor):
    torch.manual_seed(42)
    seq_len = 2
    vocab_size = 128
    dtype = torch.bfloat16
    tensor_parallel_size = mpi_pool_executor.num_workers
    input = torch.randn((seq_len, hidden_size), dtype=dtype)
    weight = torch.randn((vocab_size, hidden_size), dtype=dtype)
    results = mpi_pool_executor.map(
        run_single_rank,
        *zip(*[(tensor_parallel_size, row_lm_head_forward, input, weight,
                vocab_size, hidden_size, dtype)] * 2))
    for r in results:
        assert r is True
