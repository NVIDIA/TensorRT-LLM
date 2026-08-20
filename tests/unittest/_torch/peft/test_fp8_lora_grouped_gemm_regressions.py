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

from tensorrt_llm._torch.peft.lora.cuda_graph_lora_params import CudaGraphLoraParams
from tensorrt_llm._torch.peft.lora.layer import (
    LoraLayer,
    _validate_fp8_lora_cuda_graph_alignment,
    add_lora_result,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_NATIVE_FP8_AVAILABLE = torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


def _kernel_source(filename: str) -> str:
    return (_REPO_ROOT / "cpp" / "tensorrt_llm" / "kernels" / filename).read_text()


def _function_block(source: str, start: str, end: str) -> str:
    start_index = source.index(start)
    end_index = source.index(end, start_index)
    return source[start_index:end_index]


def _make_fp8_lora_problem(batch_size=16, hidden_size=32, rank=16, output_size=32):
    torch.manual_seed(0)
    x = (torch.randn(batch_size, hidden_size, device="cuda") * 0.25).to(torch.float8_e4m3fn)
    lora_in = (torch.randn(rank, hidden_size, device="cuda") * 0.25).to(torch.float8_e4m3fn)
    lora_out = (torch.randn(output_size, rank, device="cuda") * 0.25).to(torch.float8_e4m3fn)
    reference = x.float() @ lora_in.float().T @ lora_out.float().T
    return x, lora_in.contiguous(), lora_out.contiguous(), reference


def _assert_fp8_gemm_matches_reference(actual, reference):
    torch.testing.assert_close(
        actual.float(),
        reference.to(torch.float8_e4m3fn).float(),
        atol=0.25,
        rtol=0.25,
    )


@pytest.mark.skipif(not _NATIVE_FP8_AVAILABLE, reason="Native FP8 LoRA requires SM90")
def test_fp8_eager_grouped_gemm_matches_reference():
    x, lora_in, lora_out, reference = _make_fp8_lora_problem()
    rank = lora_in.shape[0]
    weight_pointers = torch.tensor(
        [[lora_in.data_ptr(), lora_out.data_ptr(), 0]], dtype=torch.int64
    )

    actual = torch.ops.trtllm.lora_grouped_gemm(
        x,
        torch.zeros(1, dtype=torch.int32),
        [torch.tensor([rank], dtype=torch.int32)],
        [weight_pointers],
        torch.tensor([x.shape[0]], dtype=torch.int32),
        [lora_out.shape[0]],
        False,
        True,
        rank,
        0,
        True,
    )[0]

    _assert_fp8_gemm_matches_reference(actual, reference)


@pytest.mark.skipif(not _NATIVE_FP8_AVAILABLE, reason="Native FP8 LoRA requires SM90")
def test_fp8_cuda_graph_grouped_gemm_matches_reference_after_replay():
    batch_size = 16
    x, lora_in, lora_out, _ = _make_fp8_lora_problem(batch_size=batch_size)
    rank = lora_in.shape[0]
    module_id = 0
    layer_key = CudaGraphLoraParams.LoraLayerKey(layer_idx=0, module_ids=(module_id,))
    cuda_graph_params = CudaGraphLoraParams(
        max_batch_size=batch_size,
        max_lora_size=1,
        max_rank=rank,
        layer_info={
            layer_key: CudaGraphLoraParams.LoraLayerInfo(
                module_num=1, output_sizes=[lora_out.shape[0]]
            )
        },
    )
    cuda_graph_params.update_sorted_indices([0] * batch_size)
    cuda_graph_params.update_slots_params([0] * batch_size)
    cuda_graph_params.slot_ranks_host[0] = rank
    cuda_graph_params.slot_ranks.copy_(cuda_graph_params.slot_ranks_host)
    layer_params = cuda_graph_params.get_layer_params(layer_key)
    layer_params.h_b_ptrs[0, 0] = lora_in.data_ptr()
    layer_params.h_b_prime_ptrs[0, 0] = lora_out.data_ptr()
    layer_params.d_b_ptrs.copy_(layer_params.h_b_ptrs)
    layer_params.d_b_prime_ptrs.copy_(layer_params.h_b_prime_ptrs)

    layer = LoraLayer([module_id], [lora_out.shape[0]])
    lora_params = {
        "cuda_graph_params": cuda_graph_params,
        "data_type": torch.float8_e4m3fn,
        "use_cuda_graph_mode": True,
    }
    static_x = x.clone()
    for _ in range(3):
        layer(static_x, lora_params, layer_idx=0)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = layer(static_x, lora_params, layer_idx=0)

    replay_x, _, _, reference = _make_fp8_lora_problem(batch_size=batch_size)
    replay_x = torch.roll(replay_x.float(), shifts=1, dims=0).to(torch.float8_e4m3fn)
    reference = replay_x.float() @ lora_in.float().T @ lora_out.float().T
    static_x.copy_(replay_x)
    graph.replay()
    torch.cuda.synchronize()

    _assert_fp8_gemm_matches_reference(graph_output, reference)


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


def test_fp8_cuda_graph_grouped_gemm_reuses_live_device_metadata():
    source = _kernel_source("cuda_graph_grouped_gemm.cu")
    fp8_graph_body = _function_block(
        source, "void fp8CudaGraphGroupedGemm(", "\nvoid cudaGraphGroupedGemm("
    )

    assert "hostMaxProblemSizesPtr" not in fp8_graph_body
    assert "cudaMemcpyHostToDevice" not in fp8_graph_body
    assert "fillFp8CudaGraphGroupedGemmParams" not in source
    assert "cudaMemcpyDeviceToDevice" not in fp8_graph_body
    assert "reinterpret_cast<UnderlyingProblemShape*>(problemSizesPtr)" in fp8_graph_body
    assert "reinterpret_cast<ElementA**>(ptrAGpu)" in fp8_graph_body
    assert "reinterpret_cast<StrideA*>(ldaGpu)" in fp8_graph_body
    assert "std::is_same_v<StrideA, PackedStride>" in fp8_graph_body


@pytest.mark.parametrize(
    "filename,messages",
    [
        ("groupGemm.cu", ["FP8 grouped GEMM requires CUTLASS modifiable TMA support"]),
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


@pytest.mark.parametrize("filename", ["groupGemm.cu", "cuda_graph_grouped_gemm.cu"])
def test_fp8_grouped_gemm_dispatch_requires_sm90(filename):
    source = _kernel_source(filename)

    assert "getSMVersion()" in source
    assert "smVersion == 90" in source
    assert "requires Hopper (SM90)" in source
    assert "SM120/SM121" in source


def test_fp8_grouped_gemm_alignment_checks_require_multiples_of_16():
    source = _kernel_source("groupGemm.cu")

    assert "problem.n() % kFp8TmaAlignment == 0" in source
    assert "problem.k() % kFp8TmaAlignment == 0" in source


def test_fp8_splitk_grouped_gemm_delegates_to_regular_grouped_gemm():
    source = _kernel_source("splitkGroupGemm.cu")

    assert "fp8SplitkGroupedGemm" not in source
    assert "groupedGemm(problemSizes" in source


def test_fp8_tma_alignment_has_one_cpp_definition():
    kernels_dir = _REPO_ROOT / "cpp" / "tensorrt_llm" / "kernels"
    definitions = sorted(
        path
        for path in kernels_dir.iterdir()
        if path.suffix in {".cu", ".h"} and "kFp8TmaAlignment = 16" in path.read_text()
    )

    assert definitions == [kernels_dir / "groupGemm.h"]


def test_fp8_cuda_graph_alignment_check_requires_rank_multiple_of_16():
    source = _kernel_source("cuda_graph_grouped_gemm.cu")

    assert "minKN >= kFp8TmaAlignment && minKN % kFp8TmaAlignment == 0" in source
    assert "problem.n() % kFp8TmaAlignment == 0" in source
    assert "problem.k() % kFp8TmaAlignment == 0" in source
