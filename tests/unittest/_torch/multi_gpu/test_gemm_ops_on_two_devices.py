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
"""TRT-LLM GEMM ops called on cuda:0 and then cuda:1 from one thread.

The ops run on the calling thread's PyTorch cuBLAS handle, which PyTorch keeps per thread and
device. Each op runs once with the current device following the inputs, which checks that it
keeps one handle per device, and once with the current device left on cuda:0, which checks that
it switches to the inputs' device.
"""

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

import tensorrt_llm  # noqa: F401  (registers torch.ops.trtllm)

pytestmark = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")

_NUM_TOKENS = 16
_HIDDEN_SIZE = 64
_LORA_RANK = 16
_TOLERANCE = dict(atol=2e-2, rtol=2e-2)


def _randn(*shape, device):
    # Scaled so that the outputs stay near 1, where the absolute tolerance is meaningful.
    return (torch.randn(shape, device=device) * 0.25).to(torch.bfloat16)


def _run_lora_grouped_gemm(device):
    """One context request whose tokens all use one LoRA adapter, so the op takes its cuBLAS
    path."""
    x = _randn(_NUM_TOKENS, _HIDDEN_SIZE, device=device)
    lora_in = _randn(_LORA_RANK, _HIDDEN_SIZE, device=device)
    lora_out = _randn(_HIDDEN_SIZE, _LORA_RANK, device=device)
    # A, B and no DoRA magnitude.
    weight_pointers = torch.tensor(
        [[lora_in.data_ptr(), lora_out.data_ptr(), 0]], dtype=torch.int64
    )
    output = torch.ops.trtllm.lora_grouped_gemm(
        x,
        host_request_types=torch.zeros(1, dtype=torch.int32),
        lora_ranks=[torch.tensor([_LORA_RANK], dtype=torch.int32)],
        lora_weights_pointers=[weight_pointers],
        host_context_lengths=torch.tensor([_NUM_TOKENS], dtype=torch.int32),
        output_hidden_sizes=[_HIDDEN_SIZE],
        transA=False,
        transB=True,
        max_low_rank=_LORA_RANK,
        weight_index=0,
        isRemoveInputPadding=True,
    )[0]
    # The op keeps the rank-sized intermediate in BF16.
    low_rank = (x.float() @ lora_in.float().t()).to(torch.bfloat16)
    return output, low_rank.float() @ lora_out.float().t()


def _run_cublas_mm(device):
    x = _randn(_NUM_TOKENS, _HIDDEN_SIZE, device=device)
    w = _randn(_HIDDEN_SIZE, _HIDDEN_SIZE, device=device)
    output = torch.ops.trtllm.cublas_mm(x, w.t(), bias=None, out_dtype=None)
    return output, x.float() @ w.float().t()


def _run_on_two_devices(run_op, current_device_follows_input):
    for index in range(2):
        device = torch.device("cuda", index)
        with torch.cuda.device(device if current_device_follows_input else 0):
            output, reference = run_op(device)
        torch.testing.assert_close(output.float(), reference, **_TOLERANCE)


@pytest.mark.parametrize(
    "run_op", [_run_lora_grouped_gemm, _run_cublas_mm], ids=["lora_grouped_gemm", "cublas_mm"]
)
@pytest.mark.parametrize(
    "current_device_follows_input",
    [True, False],
    ids=["current_device_follows_input", "current_device_stays_cuda0"],
)
def test_gemm_op_on_two_devices_from_one_thread(monkeypatch, run_op, current_device_follows_input):
    # With LORA_USE_UNIFIED_GEMM=OFF the LoRA op would not use cuBLAS.
    monkeypatch.delenv("LORA_USE_UNIFIED_GEMM", raising=False)
    torch.manual_seed(0)
    # A new thread has no cached wrappers, so the case does not depend on the cases before it.
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(_run_on_two_devices, run_op, current_device_follows_input).result()
