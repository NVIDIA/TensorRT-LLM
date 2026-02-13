# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI

import tensorrt_llm
from tensorrt_llm._torch.distributed import (
    AllReduce,
    AllReduceFusionOp,
    AllReduceParams,
    AllReduceStrategy,
)
from tensorrt_llm._torch.flashinfer_utils import (
    IS_FLASHINFER_AVAILABLE,
    FlashInferAllReduceWorkspace,
    init_flashinfer_allreduce_workspace,
)
from tensorrt_llm.mapping import Mapping

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

# needed since we reuse the mpi executor pool, first test running will leak a thread
pytestmark = pytest.mark.threadleak(enabled=False)

skip_no_flashinfer = pytest.mark.skipif(
    not IS_FLASHINFER_AVAILABLE, reason="FlashInfer is not installed"
)


def _run_single_rank(tensor_parallel_size, worker_fn, *args):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        return worker_fn(*args, tensor_parallel_size=tensor_parallel_size, rank=rank)
    except Exception:
        traceback.print_exc()
        raise


@torch.inference_mode()
def _flashinfer_allreduce_worker(
    x: torch.Tensor,
    hidden_size: int,
    dtype: torch.dtype,
    *,
    tensor_parallel_size: int,
    rank: int,
):
    x = x.cuda()
    mapping = Mapping(
        world_size=tensor_parallel_size,
        tp_size=tensor_parallel_size,
        rank=rank,
    )

    # Initialize the workspace (idempotent — only first call creates it)
    init_flashinfer_allreduce_workspace(mapping)

    allreduce = AllReduce(
        mapping=mapping,
        strategy=AllReduceStrategy.FLASHINFER,
        dtype=dtype,
    ).cuda()

    # Shard the input across TP ranks (simulates row-parallel linear output)
    xs = torch.chunk(x.clone(), tensor_parallel_size, dim=-1)
    shard = xs[rank]

    # Run FlashInfer custom allreduce
    output = allreduce(shard, all_reduce_params=AllReduceParams())

    # Reference: NCCL allreduce
    ref_allreduce = AllReduce(mapping=mapping, strategy=AllReduceStrategy.AUTO).cuda()
    ref_output = ref_allreduce(shard, all_reduce_params=AllReduceParams())

    torch.testing.assert_close(output, ref_output, rtol=1e-3, atol=1e-3)
    return True


@skip_no_flashinfer
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
@pytest.mark.parametrize("seq_len", [1, 16, 128, 256], ids=lambda x: f"seqlen:{x}")
@pytest.mark.parametrize("hidden_size", [4096, 8192], ids=lambda x: f"hidden:{x}")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=lambda x: f"dtype:{x}")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_flashinfer_allreduce_correctness(seq_len, hidden_size, dtype, mpi_pool_executor):
    """Verify FlashInfer custom allreduce matches NCCL reference."""
    torch.manual_seed(42)
    tp = mpi_pool_executor.num_workers
    x = torch.randn((seq_len, hidden_size), dtype=dtype)
    results = mpi_pool_executor.map(
        _run_single_rank,
        *zip(*[(tp, _flashinfer_allreduce_worker, x, hidden_size, dtype)] * tp),
    )
    for r in results:
        assert r is True


@torch.inference_mode()
def _flashinfer_fallback_worker(
    x: torch.Tensor,
    hidden_size: int,
    dtype: torch.dtype,
    *,
    tensor_parallel_size: int,
    rank: int,
):
    x = x.cuda()
    mapping = Mapping(
        world_size=tensor_parallel_size,
        tp_size=tensor_parallel_size,
        rank=rank,
    )

    init_flashinfer_allreduce_workspace(mapping)

    allreduce = AllReduce(
        mapping=mapping,
        strategy=AllReduceStrategy.FLASHINFER,
        dtype=dtype,
    ).cuda()

    xs = torch.chunk(x.clone(), tensor_parallel_size, dim=-1)
    shard = xs[rank]

    # This input is large enough to exceed the max_size buffer,
    # so should_custom_ar returns False and it falls back to AUTO.
    output = allreduce(shard, all_reduce_params=AllReduceParams())

    # Reference
    ref_allreduce = AllReduce(mapping=mapping, strategy=AllReduceStrategy.AUTO).cuda()
    ref_output = ref_allreduce(shard, all_reduce_params=AllReduceParams())

    torch.testing.assert_close(output, ref_output, rtol=1e-3, atol=1e-3)
    return True


@skip_no_flashinfer
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_flashinfer_allreduce_fallback_large_input(mpi_pool_executor):
    """Verify large inputs fall back to AUTO without crashing."""
    torch.manual_seed(42)
    tp = mpi_pool_executor.num_workers
    dtype = torch.bfloat16
    # seq_len=8192, hidden=8192 => 8192*8192*2 = 128 MB >> max_size
    # This guarantees should_custom_ar returns False
    seq_len = 8192
    hidden_size = 8192
    x = torch.randn((seq_len, hidden_size), dtype=dtype)
    results = mpi_pool_executor.map(
        _run_single_rank,
        *zip(*[(tp, _flashinfer_fallback_worker, x, hidden_size, dtype)] * tp),
    )
    for r in results:
        assert r is True


@skip_no_flashinfer
@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs for workspace init"
)
class TestShouldCustomAR:
    """Unit tests for FlashInferVLLMAllReduce.should_custom_ar (logic only)."""

    @staticmethod
    def _make_module(tp_size=2):
        """Create a FlashInferVLLMAllReduce for testing should_custom_ar logic."""
        from tensorrt_llm._torch.distributed.ops import FlashInferVLLMAllReduce

        mapping = Mapping(world_size=tp_size, tp_size=tp_size, rank=0)
        init_flashinfer_allreduce_workspace(mapping)
        return FlashInferVLLMAllReduce(mapping=mapping, dtype=torch.bfloat16)

    def test_within_buffer(self):
        mod = self._make_module()
        # Small tensor that fits in buffer
        inp = torch.randn(16, 4096, dtype=torch.bfloat16, device="cuda")
        assert mod.should_custom_ar(inp) is True

    def test_exceeds_buffer(self):
        mod = self._make_module()
        # Tensor larger than max_size
        num_elements = (mod.reg_buffer_size // 2) + 1  # +1 element over limit
        inp = torch.randn(num_elements, dtype=torch.bfloat16, device="cuda")
        assert mod.should_custom_ar(inp) is False

    def test_alignment_16_bytes(self):
        mod = self._make_module()
        # 5 bf16 elements = 10 bytes, not a multiple of 16
        inp = torch.randn(5, dtype=torch.bfloat16, device="cuda")
        assert mod.should_custom_ar(inp) is False

    def test_alignment_passes(self):
        mod = self._make_module()
        # 8 bf16 elements = 16 bytes, multiple of 16
        inp = torch.randn(8, dtype=torch.bfloat16, device="cuda")
        assert mod.should_custom_ar(inp) is True

    def test_non_contiguous(self):
        mod = self._make_module()
        inp = torch.randn(16, 4096, dtype=torch.bfloat16, device="cuda")
        non_contig = inp[:, ::2]  # non-contiguous slice
        assert not non_contig.is_contiguous()
        assert mod.should_custom_ar(non_contig) is False

    def test_unsupported_dtype_fp64(self):
        mod = self._make_module()
        inp = torch.randn(16, 4096, dtype=torch.float64, device="cuda")
        assert mod.should_custom_ar(inp) is False

    def test_unsupported_dtype_int(self):
        mod = self._make_module()
        inp = torch.randint(0, 100, (16, 4096), dtype=torch.int32, device="cuda")
        assert mod.should_custom_ar(inp) is False

    def test_supported_dtypes(self):
        mod = self._make_module()
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            inp = torch.randn(8, 1024, dtype=dtype, device="cuda")
            assert mod.should_custom_ar(inp) is True, f"Failed for {dtype}"

    def test_fusion_op_not_none(self):
        mod = self._make_module()
        inp = torch.randn(16, 4096, dtype=torch.bfloat16, device="cuda")
        params = AllReduceParams(
            fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
            residual=torch.randn_like(inp),
            norm_weight=torch.randn(4096, dtype=torch.bfloat16, device="cuda"),
            eps=1e-5,
        )
        assert mod.should_custom_ar(inp, params) is False

    def test_fusion_op_none_passes(self):
        mod = self._make_module()
        inp = torch.randn(16, 4096, dtype=torch.bfloat16, device="cuda")
        params = AllReduceParams(fusion_op=AllReduceFusionOp.NONE)
        assert mod.should_custom_ar(inp, params) is True


@skip_no_flashinfer
@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs for workspace init"
)
class TestGetMaxSize:
    """Unit tests for FlashInferAllReduceWorkspace._get_max_size."""

    def _make_workspace(self, tp_size=2):
        mapping = Mapping(world_size=tp_size, tp_size=tp_size, rank=0)
        init_flashinfer_allreduce_workspace(mapping)
        return FlashInferAllReduceWorkspace(mapping)

    def test_max_size_is_positive(self):
        ws = self._make_workspace()
        assert ws.max_size > 0

    def test_max_size_does_not_exceed_default(self):
        ws = self._make_workspace()
        default = 8 * 1024 * 1024  # 8 MiB
        assert ws.max_size <= default

    def test_max_size_is_power_of_two_aligned(self):
        """max_size should be at least 256 KiB (smallest arch entry)."""
        ws = self._make_workspace()
        assert ws.max_size >= 256 * 1024
