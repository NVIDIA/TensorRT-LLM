# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify FP32 LM-head logits through a real tensor-parallel all-gather."""

import pickle
import sys
from types import SimpleNamespace

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

import tensorrt_llm
from tensorrt_llm._torch.modules.context_logits import (
    ContextLogitsPlan,
    context_logits_heads,
    context_logits_scope,
)
from tensorrt_llm._torch.modules.embedding import LMHead
from tensorrt_llm._torch.modules.linear import TensorParallelMode
from tensorrt_llm._torch.modules.logits_processor import LogitsProcessor
from tensorrt_llm._torch.pyexecutor.llm_request import PyResult
from tensorrt_llm.executor.result import compute_prompt_logprobs
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

# The shared MPI executor keeps its worker thread alive between tests.
pytestmark = pytest.mark.threadleak(enabled=False)


@torch.inference_mode()
def _check_bounded_context(lm_head: LMHead, hidden: torch.Tensor, rank: int) -> None:
    """Keep TP chunks aligned when only one rank runs out of output or storage memory."""
    model = torch.nn.Module()
    model.lm_head = lm_head
    model.logits_processor = LogitsProcessor()
    assert context_logits_heads(model, lm_head.mapping, is_spec_decode=False) == (lm_head,)
    lm_head.prepare_context_logits_workspace()
    workspace = lm_head.context_logits_workspace
    pointers = [buffer.data_ptr() for buffer in workspace.resident]
    hidden = hidden.repeat(21, 1)
    reference = lm_head(hidden).float()
    vocab_size = lm_head.num_embeddings

    for host_storage in (False, True):
        requests = []
        for length, full, k in ((70, True, None), (70, False, 0), (5, False, None)):
            tokens = [index % vocab_size for index in range(length)]
            requests.append(
                SimpleNamespace(
                    context_chunk_size=length,
                    context_current_position=0,
                    py_prompt_len=length,
                    py_return_context_logits=full or k is not None,
                    py_context_logits_explicit=full,
                    py_prompt_logprobs=k,
                    py_prompt_logprobs_simple_format=True,
                    is_last_context_chunk=True,
                    get_tokens=lambda beam, tokens=tokens: tokens,
                    py_result=PyResult(
                        prompt_len=length,
                        max_new_tokens=2,
                        return_context_logits=full or k is not None,
                    ),
                )
            )
        allocate = workspace._allocate
        empty = torch.empty
        project = workspace.project
        projected_rows = []

        def allocate_output(rows):
            if rank == 1 and not host_storage and rows > 35:
                raise torch.OutOfMemoryError("injected rank-local workspace OOM")
            return allocate(rows)

        def allocate_storage(shape, *args, **kwargs):
            if (
                host_storage
                and rank == 0
                and tuple(shape) == (70, 1, vocab_size)
                and torch.device(kwargs.get("device", "cpu")).type == "cuda"
            ):
                raise torch.OutOfMemoryError("injected rank-local storage OOM")
            return empty(shape, *args, **kwargs)

        def record_projection(head, rows, buffers):
            projected_rows.append(rows.shape[0])
            return project(head, rows, buffers)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(workspace, "_allocate", allocate_output)
            patch.setattr(torch, "empty", allocate_storage)
            patch.setattr(workspace, "project", record_projection)
            batch = SimpleNamespace(context_requests=requests, all_requests=lambda: requests)
            plan = ContextLogitsPlan.for_batch(model, lm_head.mapping, batch, is_spec_decode=False)
            assert plan is not None
            with context_logits_scope(plan):
                sampling = model.logits_processor(hidden, lm_head, None, return_context_logits=True)
            assert plan.handled
        assert projected_rows == ([32, 32, 6, 70, 1, 2] if host_storage else [35, 35] * 2 + [1, 2])
        assert [buffer.data_ptr() for buffer in workspace.resident] == pointers
        # Both returned outputs must survive reuse of the resident collective buffers.
        for buffer in workspace.resident:
            buffer.zero_()
        full_result = requests[0].py_result.context_logits
        assert full_result.device.type == ("cpu" if host_storage and rank == 0 else "cuda")
        torch.testing.assert_close(full_result.cpu(), reference[:70].cpu())
        torch.testing.assert_close(sampling, reference[[69, 139, 144, 145, 146]])
        compact = requests[1].py_result
        assert compact.context_logits is None
        expected = compute_prompt_logprobs(0, reference[70:139].cpu(), list(range(1, 70)), True)
        assert compact.context_prompt_logprobs == pytest.approx(expected.prompt, abs=1e-5)
        torch.testing.assert_close(compact.context_prompt_last_logits, reference[139:140].cpu())


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
    # Reuse one existing padded-vocabulary case for the distributed fallback checks.
    if hidden.shape[0] == 7 and vocab_size == 127 and weight.dtype == torch.bfloat16:
        _check_bounded_context(lm_head, hidden, rank)
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
