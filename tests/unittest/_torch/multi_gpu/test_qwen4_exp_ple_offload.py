# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pickle
import sys
import traceback
from types import SimpleNamespace

import cloudpickle
import pytest
import torch
from mpi4py import MPI

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)
pytestmark = pytest.mark.threadleak(enabled=False)


def _run_pinned_tp2(tp_size: int):
    rank = -1
    try:
        from unittest import mock

        import tensorrt_llm
        from tensorrt_llm._torch import distributed as dist_ops
        from tensorrt_llm._torch.modules import qwen4_exp_ple as ple_impl
        from tensorrt_llm.mapping import Mapping

        rank = tensorrt_llm.mpi_rank()
        assert tp_size == 2
        torch.cuda.set_device(rank)
        mapping = Mapping(world_size=tp_size, tp_size=tp_size, rank=rank)
        print(f"PLE TP2 rank {rank}: initialized", flush=True)

        def _nccl_allreduce(*, mapping, dtype):
            return dist_ops.AllReduce(
                mapping=mapping,
                strategy=dist_ops.AllReduceStrategy.NCCL,
                dtype=dtype,
            )

        for use_fp8 in (False, True):
            excluded = [] if use_fp8 else ["ple.ple_embedding.ngram_embedding"]
            config = SimpleNamespace(
                ngram_size=2,
                heads_per_ngram=1,
                vocab_size=16,
                eos_token_id=2,
                seed=1234,
                ngram_vocab_size_base=3,
                make_ngram_vocab_size_divisible_by=4,
                qwen4_exp_ple_host_offload=True,
                quantization_config={
                    "quant_method": "fp8",
                    "modules_to_not_convert": excluded,
                },
            )
            with mock.patch.object(
                ple_impl,
                "AllReduce",
                side_effect=_nccl_allreduce,
            ):
                module = ple_impl.Qwen4ExpNGramEmbedding(
                    config,
                    embedding_dim=7,
                    dtype=torch.bfloat16,
                    mapping=mapping,
                ).to("cuda")

            assert module.embedding_allreduce.strategy == dist_ops.AllReduceStrategy.NCCL
            weight = module.ngram_embedding.weight
            assert weight.device.type == "cpu" and weight.is_pinned()
            assert tuple(weight.shape) == (2, 7)
            assert (module.vocab_start_index, module.vocab_end_index) == (
                rank * 2,
                (rank + 1) * 2,
            )
            table_ptr = weight.data_ptr()

            full = torch.arange(28, dtype=torch.float32).reshape(4, 7) - 14
            table_dtype = torch.float8_e4m3fn if use_fp8 else torch.bfloat16
            stored = full.to(table_dtype)
            with torch.no_grad():
                weight.copy_(stored[module.vocab_start_index : module.vocab_end_index])
            scale = torch.tensor(0.125, dtype=torch.bfloat16)
            if use_fp8:
                module.configure_fp8_weight_storage(scale, table_dtype)

            ids = torch.tensor(
                [[0, 1, 2, 3], [-1, 4, 3, 0]],
                dtype=torch.long,
                device="cuda",
            )

            partial = module.ngram_embedding.gather(ids)
            expected_partial = torch.zeros_like(partial)
            owned = (ids >= module.vocab_start_index) & (ids < module.vocab_end_index)
            expected_partial[owned] = stored[ids[owned].cpu()].to(torch.bfloat16).cuda()
            torch.testing.assert_close(partial, expected_partial, rtol=0, atol=0)

            print(f"PLE TP2 rank {rank}: all-reduce use_fp8={use_fp8}", flush=True)
            actual = module.embed(ids)
            expected = torch.zeros_like(actual)
            valid = (ids >= 0) & (ids < 4)
            expected[valid] = stored[ids[valid].cpu()].to(torch.bfloat16).cuda()
            if use_fp8:
                expected = expected.float().mul(scale.float().cuda()).to(torch.bfloat16)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert weight.data_ptr() == table_ptr
            torch.cuda.synchronize()
            print(f"PLE TP2 rank {rank}: passed use_fp8={use_fp8}", flush=True)

        return True
    except BaseException:
        return f"rank {rank} failed:\n{traceback.format_exc()}"


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_qwen4_exp_ple_pinned_tp2_nccl(mpi_pool_executor):
    results = list(mpi_pool_executor.map(_run_pinned_tp2, [2, 2]))
    for result in results:
        assert result is True, result
