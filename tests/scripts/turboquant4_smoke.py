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
"""Import-free TurboQuant4 smoke test.

This script intentionally loads ``tensorrt_llm/_torch/modules/turboquant4.py``
by file path instead of importing ``tensorrt_llm``. It is meant for local
developer environments that have PyTorch but do not have TensorRT installed.

Pass ``--native`` in a built TensorRT-LLM CUDA environment to import the package,
load the native thop extension, and compare the TurboQuant4 CUDA ops against the
same Python reference path used by the default smoke test.
"""

import argparse
import importlib.util
from pathlib import Path

import torch

_NATIVE_OPS = (
    "turboquant4_quantize",
    "turboquant4_dequantize",
    "turboquant4_update_cache",
    "turboquant4_dequantize_cache",
    "turboquant4_attention",
    "turboquant4_batch_attention",
)


def _load_turboquant4_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "tensorrt_llm" / "_torch" / "modules" / "turboquant4.py"
    spec = importlib.util.spec_from_file_location("turboquant4_smoke_target", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _run_reference_smoke(turboquant4) -> None:
    tokens_per_block = 4
    num_heads = 4
    num_kv_heads = 2
    head_dim = 128

    x = torch.randn(2, 3, head_dim)
    transformed = turboquant4.fwht(turboquant4.fwht(x))
    torch.testing.assert_close(transformed, x, rtol=1e-5, atol=1e-5)

    codes, scales = turboquant4.turboquant4_quantize(x)
    assert codes.shape == (2, 3, head_dim // 2)
    assert codes.dtype == torch.uint8
    assert scales.shape == (2, 3, 1)
    assert scales.dtype == torch.float32
    dequantized = turboquant4.turboquant4_dequantize(codes, scales, dtype=x.dtype)
    assert dequantized.shape == x.shape
    assert torch.isfinite(dequantized).all()

    cache = torch.zeros(3, 2, tokens_per_block, num_kv_heads, head_dim // 2, dtype=torch.uint8)
    cache_scales = torch.zeros(3, 2, tokens_per_block, num_kv_heads, 1, dtype=torch.float32)
    q = torch.randn(5, num_heads, head_dim)
    k = torch.randn(5, num_kv_heads, head_dim)
    v = torch.randn(5, num_kv_heads, head_dim)
    block_ids = [[0, 1], [2]]

    turboquant4.turboquant4_update_cache(
        k[:3], cache, cache_scales, block_ids[0], 0, 0, tokens_per_block
    )
    turboquant4.turboquant4_update_cache(
        v[:3], cache, cache_scales, block_ids[0], 1, 0, tokens_per_block
    )
    turboquant4.turboquant4_update_cache(
        k[3:], cache, cache_scales, block_ids[1], 0, 0, tokens_per_block
    )
    turboquant4.turboquant4_update_cache(
        v[3:], cache, cache_scales, block_ids[1], 1, 0, tokens_per_block
    )

    cached_k = turboquant4.turboquant4_dequantize_cache(
        cache, cache_scales, block_ids[0], 0, 3, tokens_per_block, dtype=k.dtype
    )
    assert cached_k.shape == (3, num_kv_heads, head_dim)
    torch.testing.assert_close(
        cached_k,
        turboquant4.turboquant4_quantize_dequantize(k[:3].unsqueeze(0)).squeeze(0),
    )

    q_batch_indices = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32)
    query_positions = torch.tensor([0, 1, 2, 0, 1], dtype=torch.int32)
    seq_lens = torch.tensor([3, 2], dtype=torch.int32)
    output = turboquant4.turboquant4_batch_attention(
        q,
        cache,
        cache_scales,
        block_ids,
        q_batch_indices,
        query_positions,
        seq_lens,
        tokens_per_block,
        1.0,
        True,
        None,
    )
    assert output.shape == (5, num_heads, head_dim)
    assert torch.isfinite(output).all()
    expected_outputs = []
    for query_idx, (batch_idx, query_pos) in enumerate(
        zip(q_batch_indices.tolist(), query_positions.tolist(), strict=True)
    ):
        expected_outputs.append(
            turboquant4.turboquant4_attention(
                q[query_idx : query_idx + 1],
                cache,
                cache_scales,
                block_ids[batch_idx],
                int(seq_lens[batch_idx].item()),
                query_pos,
                tokens_per_block,
                1.0,
                True,
                None,
            )
        )
    torch.testing.assert_close(output, torch.cat(expected_outputs, dim=0))

    try:
        turboquant4.turboquant4_batch_attention(
            q[:3],
            cache,
            cache_scales,
            [[0, -1], [2]],
            torch.tensor([0, 0, 1], dtype=torch.int32),
            torch.tensor([0, 4, 0], dtype=torch.int32),
            torch.tensor([5, 3], dtype=torch.int32),
            tokens_per_block,
            1.0,
            True,
            None,
        )
    except RuntimeError as exc:
        assert "batch 0" in str(exc)
    else:
        raise AssertionError("expected invalid block id rejection")

    print("turboquant4 smoke ok", tuple(output.shape), output.dtype)


def _load_native_ops() -> None:
    missing = [name for name in _NATIVE_OPS if not hasattr(torch.ops.trtllm, name)]
    if not missing:
        return

    repo_root = Path(__file__).resolve().parents[2]
    import_error = None
    try:
        from tensorrt_llm._common import _init

        _init()
    except Exception as exc:  # noqa: BLE001 - this is a diagnostic smoke script.
        import_error = exc

    missing = [name for name in _NATIVE_OPS if not hasattr(torch.ops.trtllm, name)]
    if not missing:
        return

    direct_load_error = None
    try:
        torch.classes.load_library(repo_root / "tensorrt_llm" / "libs" / "libth_common.so")
    except Exception as exc:  # noqa: BLE001 - this is a diagnostic smoke script.
        direct_load_error = exc

    missing = [name for name in _NATIVE_OPS if not hasattr(torch.ops.trtllm, name)]
    if not missing:
        return

    message = (
        "TurboQuant4 native ops are not registered: "
        f"{', '.join(missing)}. Run --native only in a built TensorRT-LLM "
        "environment with libth_common.so available."
    )
    if import_error is not None:
        message += f" TensorRT-LLM initialization failed with: {import_error!r}."
    if direct_load_error is not None:
        message += f" Direct libth_common load failed with: {direct_load_error!r}."
    raise RuntimeError(message)


def _run_native_smoke(turboquant4) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("TurboQuant4 native smoke requires CUDA.")
    _load_native_ops()

    torch.manual_seed(0)
    device = torch.device("cuda")
    tokens_per_block = 16
    num_heads = 4
    num_kv_heads = 2
    head_dim = 128

    x = torch.randn(8, head_dim, device=device) * 0.1
    native_codes, native_scales = turboquant4.turboquant4_quantize(x)
    native_result = turboquant4.turboquant4_dequantize(native_codes, native_scales, dtype=x.dtype)
    ref_codes, ref_scales = turboquant4.turboquant4_quantize(x.cpu())
    ref_result = turboquant4.turboquant4_dequantize(ref_codes, ref_scales, dtype=x.dtype)
    torch.testing.assert_close(native_codes.cpu(), ref_codes)
    torch.testing.assert_close(native_scales.cpu(), ref_scales, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(native_result.cpu(), ref_result, atol=1e-5, rtol=1e-5)

    seq_lens = [260, 130]
    q_lens = [2, 1]
    block_counts = [(seq_len + tokens_per_block - 1) // tokens_per_block for seq_len in seq_lens]
    block_ids_per_request = [
        list(range(block_counts[0])),
        list(range(block_counts[0], block_counts[0] + block_counts[1])),
    ]
    num_blocks = sum(block_counts)

    native_cache = torch.empty(
        num_blocks,
        2,
        tokens_per_block,
        num_kv_heads,
        head_dim // 2,
        dtype=torch.uint8,
        device=device,
    )
    native_scales_cache = torch.empty(
        num_blocks,
        2,
        tokens_per_block,
        num_kv_heads,
        1,
        dtype=torch.float32,
        device=device,
    )
    ref_cache = torch.empty(
        num_blocks, 2, tokens_per_block, num_kv_heads, head_dim // 2, dtype=torch.uint8
    )
    ref_scales_cache = torch.empty(
        num_blocks, 2, tokens_per_block, num_kv_heads, 1, dtype=torch.float32
    )
    q_chunks = []
    q_batch_indices = []
    query_positions = []
    for batch_idx, (seq_len, q_len, block_ids) in enumerate(
        zip(seq_lens, q_lens, block_ids_per_request, strict=True)
    ):
        k = torch.randn(seq_len, num_kv_heads, head_dim, device=device)
        v = torch.randn(seq_len, num_kv_heads, head_dim, device=device)
        turboquant4.turboquant4_update_cache(
            k, native_cache, native_scales_cache, block_ids, 0, 0, tokens_per_block
        )
        turboquant4.turboquant4_update_cache(
            v, native_cache, native_scales_cache, block_ids, 1, 0, tokens_per_block
        )
        turboquant4.turboquant4_update_cache(
            k.cpu(), ref_cache, ref_scales_cache, block_ids, 0, 0, tokens_per_block
        )
        turboquant4.turboquant4_update_cache(
            v.cpu(), ref_cache, ref_scales_cache, block_ids, 1, 0, tokens_per_block
        )
        q_chunks.append(torch.randn(q_len, num_heads, head_dim, device=device))
        q_batch_indices.extend([batch_idx] * q_len)
        query_positions.extend(range(seq_len - q_len, seq_len))

    q = torch.cat(q_chunks, dim=0)
    q_batch_indices_tensor = torch.tensor(q_batch_indices, dtype=torch.int32)
    query_positions_tensor = torch.tensor(query_positions, dtype=torch.int32)
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32)

    native = turboquant4.turboquant4_batch_attention(
        q,
        native_cache,
        native_scales_cache,
        block_ids_per_request,
        q_batch_indices_tensor,
        query_positions_tensor,
        seq_lens_tensor,
        tokens_per_block,
        1.0,
        True,
        None,
    )
    expected = turboquant4.turboquant4_batch_attention(
        q.cpu(),
        ref_cache,
        ref_scales_cache,
        block_ids_per_request,
        q_batch_indices_tensor,
        query_positions_tensor,
        seq_lens_tensor,
        tokens_per_block,
        1.0,
        True,
        None,
    )
    torch.testing.assert_close(native.cpu(), expected, atol=1e-4, rtol=1e-4)

    try:
        turboquant4.turboquant4_update_cache(
            torch.randn(1, 1, 32, device=device),
            torch.empty(1, 2, 4, 1, 16, dtype=torch.uint8, device=device),
            torch.empty(1, 2, 4, 1, 1, dtype=torch.float32, device=device),
            [1],
            0,
            0,
            4,
        )
    except RuntimeError as exc:
        assert "out of range" in str(exc)
    else:
        raise AssertionError("expected native invalid block id rejection")

    print("turboquant4 native smoke ok", tuple(native.shape), native.dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--native",
        action="store_true",
        help="also validate CUDA native trtllm::turboquant4_* ops against the CPU reference",
    )
    args = parser.parse_args()

    turboquant4 = _load_turboquant4_module()
    _run_reference_smoke(turboquant4)
    if args.native:
        _run_native_smoke(turboquant4)


if __name__ == "__main__":
    main()
