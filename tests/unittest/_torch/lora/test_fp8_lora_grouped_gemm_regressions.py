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

from pathlib import Path

import pytest
import torch

from tensorrt_llm._torch.peft.lora.layer import (
    LoraLayer,
    _validate_fp8_lora_cuda_graph_alignment,
    add_lora_result,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _kernel_source(filename: str) -> str:
    return (_REPO_ROOT / "cpp" / "tensorrt_llm" / "kernels" / filename).read_text()


def _function_block(source: str, start: str, end: str) -> str:
    start_index = source.index(start)
    end_index = source.index(end, start_index)
    return source[start_index:end_index]


def test_fp8_cuda_graph_alignment_accepts_valid_ranks_and_dims():
    min_kn = _validate_fp8_lora_cuda_graph_alignment(
        torch.tensor([0, 16, 32], dtype=torch.int32), 64, [128, 256], 32
    )

    assert min_kn == 16


@pytest.mark.parametrize("use_cuda_graph_mode", [False, True])
def test_lora_layer_converts_fp8_cache_input_and_restores_output_dtype(
    monkeypatch, use_cuda_graph_mode
):
    layer = LoraLayer([], [])
    forwarded = {}

    def fake_forward(x, _lora_params, _layer_idx):
        forwarded["input"] = x
        return x

    method_name = "_forward_cuda_graph_mode" if use_cuda_graph_mode else "_forward_eager_mode"
    monkeypatch.setattr(layer, method_name, fake_forward)

    x = torch.tensor([[-500.0, -1.0, 1.0, 500.0]], dtype=torch.bfloat16)
    result = layer(
        x,
        {
            "data_type": torch.float8_e4m3fn,
            "use_cuda_graph_mode": use_cuda_graph_mode,
        },
        layer_idx=0,
    )

    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    expected = x.clamp(min=-fp8_max, max=fp8_max).to(torch.float8_e4m3fn)
    assert forwarded["input"].dtype == torch.float8_e4m3fn
    torch.testing.assert_close(forwarded["input"], expected)
    assert result.dtype == torch.bfloat16
    torch.testing.assert_close(result, expected.to(torch.bfloat16))


def test_add_lora_result_casts_fp8_delta_to_base_output_dtype():
    output = torch.zeros((1, 4), dtype=torch.bfloat16)
    lora_result = torch.ones((1, 4), dtype=torch.float8_e4m3fn)

    result = add_lora_result(output, lora_result)

    assert result.dtype == torch.bfloat16
    torch.testing.assert_close(result, torch.ones_like(output))
    assert add_lora_result(output, None) is output


def test_lora_layer_rejects_non_fp8_activation_cache_dtype_mismatch():
    layer = LoraLayer([], [])

    with pytest.raises(TypeError, match="must match PEFT cache dtype"):
        layer(
            torch.empty((1, 16), dtype=torch.bfloat16),
            {"data_type": torch.float16},
            layer_idx=0,
        )


@pytest.mark.parametrize(
    "slot_ranks,max_rank,match",
    [
        ([16, 24], 32, "active LoRA ranks"),
        ([16], 24, "max LoRA rank"),
    ],
)
def test_fp8_cuda_graph_alignment_rejects_misaligned_ranks(slot_ranks, max_rank, match):
    with pytest.raises(ValueError, match=match):
        _validate_fp8_lora_cuda_graph_alignment(
            torch.tensor(slot_ranks, dtype=torch.int32), 64, [128], max_rank
        )


@pytest.mark.parametrize("hidden_size,output_hidden_sizes", [(24, [128]), (64, [128, 72])])
def test_fp8_cuda_graph_alignment_rejects_misaligned_hidden_dims(hidden_size, output_hidden_sizes):
    with pytest.raises(ValueError, match="hidden and output sizes"):
        _validate_fp8_lora_cuda_graph_alignment(
            torch.tensor([16], dtype=torch.int32), hidden_size, output_hidden_sizes, 16
        )


def test_fp8_cuda_graph_grouped_gemm_uses_live_device_problem_metadata():
    source = _kernel_source("cuda_graph_grouped_gemm.cu")
    fp8_graph_body = _function_block(
        source, "void fp8CudaGraphGroupedGemm(", "\nvoid cudaGraphGroupedGemm("
    )

    assert "hostMaxProblemSizesPtr" not in fp8_graph_body
    assert "cudaMemcpyHostToDevice" not in fp8_graph_body
    assert "fillFp8CudaGraphGroupedGemmParams" in fp8_graph_body
    assert "problemSizesPtr, problemCount, ldaGpu, ldbGpu, ldcGpu, lddGpu" in fp8_graph_body


@pytest.mark.parametrize(
    "filename,messages",
    [
        ("groupGemm.cu", ["FP8 grouped GEMM requires CUTLASS modifiable TMA support"]),
        (
            "splitkGroupGemm.cu",
            ["FP8 split-K grouped GEMM requires CUTLASS modifiable TMA support"],
        ),
        (
            "cuda_graph_grouped_gemm.cu",
            [
                "FP8 CUDA graph grouped GEMM requires CUTLASS modifiable TMA support",
                "FP8 CUDA graph split-K grouped GEMM requires CUTLASS modifiable TMA support",
            ],
        ),
    ],
)
def test_fp8_grouped_gemm_dispatch_has_explicit_unsupported_cutlass_guard(filename, messages):
    source = _kernel_source(filename)

    assert "#else" in source
    for message in messages:
        assert message in source


@pytest.mark.parametrize("filename", ["groupGemm.cu", "splitkGroupGemm.cu"])
def test_fp8_grouped_gemm_alignment_checks_require_multiples_of_16(filename):
    source = _kernel_source(filename)

    assert "problem.n() % kFp8TmaAlignment == 0" in source
    assert "problem.k() % kFp8TmaAlignment == 0" in source


def test_fp8_cuda_graph_alignment_check_requires_rank_multiple_of_16():
    source = _kernel_source("cuda_graph_grouped_gemm.cu")

    assert "minKN >= kFp8TmaAlignment && minKN % kFp8TmaAlignment == 0" in source
    assert "problem.n() % kFp8TmaAlignment == 0" in source
    assert "problem.k() % kFp8TmaAlignment == 0" in source
