# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
FMHA_ROOT = REPO_ROOT / "tensorrt_llm" / "_torch" / "attention_backend" / "fmha"
KERNEL_ROOT = REPO_ROOT / "cpp" / "tensorrt_llm" / "kernels"
THOP_ROOT = REPO_ROOT / "cpp" / "tensorrt_llm" / "thop"


def test_flashinfer_trtllm_gen_rejects_compact_pseudokv_carrier_before_dispatch() -> None:
    source = (FMHA_ROOT / "flashinfer_trtllm_gen.py").read_text(encoding="utf-8")

    guard = 'if fwd.compact_pseudokv is not None:'
    reason = 'return False, "trtllm-gen does not support compact pseudo-KV."'

    assert guard in source
    assert reason in source
    assert source.index(guard) < source.index("if meta.kv_cache_block_offsets is None:")


def test_fallback_fmha_forwards_compact_pseudokv_carrier_to_thop_attention() -> None:
    source = (FMHA_ROOT / "fallback.py").read_text(encoding="utf-8")
    interface_source = (
        REPO_ROOT / "tensorrt_llm" / "_torch" / "attention_backend" / "interface.py"
    ).read_text(encoding="utf-8")

    required_kwargs = [
        "compact_pseudokv_key=forward_args.compact_pseudokv_key",
        "compact_pseudokv_value=forward_args.compact_pseudokv_value",
        "compact_pseudokv_positions=forward_args.compact_pseudokv_positions",
        "compact_pseudokv_causal_mask=forward_args.compact_pseudokv_causal_mask",
        "compact_pseudokv_source_seq_len=forward_args.compact_pseudokv_source_seq_len",
    ]

    assert "compact_pseudokv: Optional[CompactPseudoKvPrediction] = None" in interface_source
    assert "def compact_pseudokv_key" in interface_source
    for kwarg in required_kwargs:
        assert kwarg in source


def test_compact_pseudokv_kernel_handles_fully_masked_rows() -> None:
    source = (KERNEL_ROOT / "compactPseudoKvAttentionKernels.cu").read_text(encoding="utf-8")

    assert "std::numeric_limits<float>::infinity()" in source
    assert "if (!isfinite(maxScore))" in source
    assert "reinterpret_cast<float*>(outputBase)[dim] = 0.0F;" in source


def test_compact_pseudokv_kernel_serializes_attribute_update_and_launch() -> None:
    source = (KERNEL_ROOT / "compactPseudoKvAttentionKernels.cu").read_text(encoding="utf-8")
    attribute_index = source.index("cudaFuncSetAttribute")
    launch_index = source.index("compactPseudoKvAttentionFloatKernel<<<")
    lock_index = source.index("std::lock_guard<std::mutex> const lock")

    assert lock_index < attribute_index < launch_index


def test_compact_pseudokv_thop_validates_device_and_contiguous_layout() -> None:
    source = (THOP_ROOT / "compactPseudoKvAttentionOp.cpp").read_text(encoding="utf-8")

    assert "requires all tensors on the same CUDA device" in source
    assert "q.stride(2) == 1" in source
    assert "compactKey.stride(2) == 1" in source
    assert "compactValue.stride(2) == 1" in source
    assert "causalMask.is_contiguous()" in source
    assert "outputIsRank3 ? output.stride(2) == 1 : output.stride(1) == 1" in source
