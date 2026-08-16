# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for the Triton FlashDecoding kernel with paged KV cache.

Covers the shapes the PyTorch backend routes to Triton because FlashInfer's
paged kernels cannot serve them, in particular Gemma4's head_dim=512
full-attention layers on architectures without trtllm-gen cubins.
"""

import math

import pytest
import torch


def _import_triton_decode():
    """Import triton_decode directly to avoid TRT-LLM C++ bindings."""
    import importlib.util
    import os

    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "..",
        "tensorrt_llm",
        "_torch",
        "attention_backend",
        "triton_decode.py",
    )
    spec = importlib.util.spec_from_file_location("triton_decode", os.path.abspath(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


triton_decode_mod = _import_triton_decode()
triton_decode = triton_decode_mod.triton_decode

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton decode kernel requires a GPU"
)


def _decode_reference(q, k_list, v_list, sm_scale, sliding_window=None):
    """Reference single-token decode attention.

    Args:
        q: [batch, n_heads, head_dim]
        k_list: per-sequence [seq_len, n_kv_heads, head_dim]
        v_list: per-sequence [seq_len, n_kv_heads, head_dim]
        sm_scale: softmax scale
        sliding_window: attend only to the last N tokens (inclusive of current)

    Returns:
        [batch, n_heads, head_dim]
    """
    batch, n_heads, _ = q.shape
    outputs = []
    for b in range(batch):
        k = k_list[b]
        v = v_list[b]
        seq_len, n_kv_heads, _ = k.shape
        if sliding_window is not None and sliding_window > 0:
            start = max(0, seq_len - sliding_window)
            k = k[start:]
            v = v[start:]
        gqa = n_heads // n_kv_heads
        # Broadcast KV heads across their query group.
        k_rep = k.repeat_interleave(gqa, dim=1)  # [win, n_heads, head_dim]
        v_rep = v.repeat_interleave(gqa, dim=1)
        # [n_heads, win]
        scores = torch.einsum("hd,whd->hw", q[b].float(), k_rep.float())
        scores = scores * sm_scale
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("hw,whd->hd", probs, v_rep.float())
        outputs.append(out)
    return torch.stack(outputs, dim=0)


def _build_paged_cache(k_list, v_list, page_size, device, dtype=torch.bfloat16):
    """Pack per-sequence KV into a combined HND paged cache.

    Returns:
        kv_cache: [num_pages, 2, n_kv_heads, page_size, head_dim]
        kv_indices: [total_pages] physical page ids
        kv_indptr: [batch + 1] cumulative page counts
        kv_last_page_len: [batch] valid tokens in each sequence's last page
    """
    n_kv_heads, head_dim = k_list[0].shape[1], k_list[0].shape[2]
    pages_k, pages_v = [], []
    page_counts = [0]
    last_page_lens = []

    for k, v in zip(k_list, v_list):
        seq_len = k.shape[0]
        n_pages = (seq_len + page_size - 1) // page_size
        page_counts.append(page_counts[-1] + n_pages)
        last = seq_len - (n_pages - 1) * page_size
        last_page_lens.append(last)

        for p in range(n_pages):
            start = p * page_size
            end = min(start + page_size, seq_len)
            n_tok = end - start
            page_k = torch.zeros(n_kv_heads, page_size, head_dim, dtype=dtype, device=device)
            page_v = torch.zeros(n_kv_heads, page_size, head_dim, dtype=dtype, device=device)
            page_k[:, :n_tok, :] = k[start:end].transpose(0, 1).to(dtype)
            page_v[:, :n_tok, :] = v[start:end].transpose(0, 1).to(dtype)
            pages_k.append(page_k)
            pages_v.append(page_v)

    k_pages = torch.stack(pages_k, dim=0)
    v_pages = torch.stack(pages_v, dim=0)
    kv_cache = torch.stack([k_pages, v_pages], dim=1)

    kv_indices = torch.arange(k_pages.shape[0], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor(page_counts, dtype=torch.int32, device=device)
    kv_last_page_len = torch.tensor(last_page_lens, dtype=torch.int32, device=device)
    return kv_cache, kv_indices, kv_indptr, kv_last_page_len


def _run(
    seq_lens,
    n_heads,
    n_kv_heads,
    head_dim,
    page_size,
    device,
    sliding_window=None,
    kv_dtype=torch.bfloat16,
    seed=0,
):
    """Build inputs, run the kernel, and return (actual, expected)."""
    torch.manual_seed(seed)
    batch = len(seq_lens)

    q = torch.randn(batch, n_heads, head_dim, dtype=torch.bfloat16, device=device)
    k_list = [
        torch.randn(s, n_kv_heads, head_dim, dtype=torch.bfloat16, device=device) for s in seq_lens
    ]
    v_list = [
        torch.randn(s, n_kv_heads, head_dim, dtype=torch.bfloat16, device=device) for s in seq_lens
    ]

    kv_cache, kv_indices, kv_indptr, kv_last_page_len = _build_paged_cache(
        k_list, v_list, page_size, device, dtype=kv_dtype
    )

    # The kernel reads the cache and casts to the query dtype, so an FP8 cache
    # is lossy. Compare against the values actually stored, not the originals.
    if kv_dtype != torch.bfloat16:
        k_list = [k.to(kv_dtype).to(torch.bfloat16) for k in k_list]
        v_list = [v.to(kv_dtype).to(torch.bfloat16) for v in v_list]

    sm_scale = 1.0 / math.sqrt(head_dim)
    actual = triton_decode(
        q=q,
        kv_cache=kv_cache,
        kv_indices=kv_indices,
        kv_indptr=kv_indptr,
        kv_last_page_len=kv_last_page_len,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
    )
    expected = _decode_reference(q, k_list, v_list, sm_scale, sliding_window=sliding_window)
    return actual.float(), expected


@pytest.fixture
def device():
    return torch.device("cuda:0")


class TestTritonDecodeHeadDims:
    """head_dim coverage, including the Gemma4 512 path."""

    @pytest.mark.parametrize("head_dim", [64, 128, 256, 512])
    def test_head_dims(self, head_dim, device):
        actual, expected = _run(
            [37], n_heads=8, n_kv_heads=4, head_dim=head_dim, page_size=16, device=device
        )
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)

    def test_gemma4_e2b_mqa_hd512(self, device):
        """Gemma4-E2B full-attention layers: MQA, head_dim 512."""
        actual, expected = _run(
            [64], n_heads=8, n_kv_heads=1, head_dim=512, page_size=16, device=device
        )
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)

    def test_gemma4_31b_gqa_hd512(self, device):
        """Gemma4-31B full-attention layers: GQA 32/4, head_dim 512."""
        actual, expected = _run(
            [100], n_heads=32, n_kv_heads=4, head_dim=512, page_size=32, device=device
        )
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


class TestTritonDecodeGQARatios:
    """GQA/MQA head-ratio coverage, including non-power-of-2 ratios."""

    @pytest.mark.parametrize(
        "n_heads,n_kv_heads",
        [
            (8, 8),
            (8, 4),
            (8, 1),
            (16, 2),
            (12, 4),
            (24, 4),
        ],
    )
    def test_head_ratios(self, n_heads, n_kv_heads, device):
        actual, expected = _run(
            [48], n_heads=n_heads, n_kv_heads=n_kv_heads, head_dim=128, page_size=16, device=device
        )
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


class TestTritonDecodePaging:
    """Page-boundary and partial-last-page behaviour."""

    @pytest.mark.parametrize("page_size", [1, 8, 16, 32, 64])
    def test_page_sizes(self, page_size, device):
        actual, expected = _run(
            [53], n_heads=8, n_kv_heads=2, head_dim=128, page_size=page_size, device=device
        )
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)

    def test_exact_page_multiple(self, device):
        """seq_len an exact multiple of page_size: last page is full."""
        actual, expected = _run(
            [64], n_heads=8, n_kv_heads=2, head_dim=128, page_size=16, device=device
        )
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)

    def test_single_token_sequence(self, device):
        actual, expected = _run(
            [1], n_heads=8, n_kv_heads=2, head_dim=128, page_size=16, device=device
        )
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)

    def test_variable_length_batch(self, device):
        """Mixed sequence lengths exercise per-sequence indptr/last_page_len."""
        actual, expected = _run(
            [1, 15, 16, 17, 128, 300],
            n_heads=8,
            n_kv_heads=2,
            head_dim=128,
            page_size=16,
            device=device,
        )
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)

    def test_long_sequence_multi_split(self, device):
        """Long single sequence with small batch forces split-K > 1."""
        actual, expected = _run(
            [4096], n_heads=8, n_kv_heads=1, head_dim=128, page_size=32, device=device
        )
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


class TestTritonDecodeSlidingWindow:
    """Sliding-window masking, as used by Gemma4's VSWA layers."""

    @pytest.mark.parametrize("sliding_window", [1, 8, 64, 1024])
    def test_sliding_window(self, sliding_window, device):
        actual, expected = _run(
            [300],
            n_heads=8,
            n_kv_heads=2,
            head_dim=128,
            page_size=16,
            device=device,
            sliding_window=sliding_window,
        )
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)

    def test_window_larger_than_sequence(self, device):
        """A window wider than the sequence must equal full attention."""
        windowed, _ = _run(
            [32],
            n_heads=8,
            n_kv_heads=2,
            head_dim=128,
            page_size=16,
            device=device,
            sliding_window=4096,
            seed=7,
        )
        full, _ = _run(
            [32],
            n_heads=8,
            n_kv_heads=2,
            head_dim=128,
            page_size=16,
            device=device,
            sliding_window=None,
            seed=7,
        )
        torch.testing.assert_close(windowed, full, atol=2e-2, rtol=2e-2)

    def test_sliding_window_hd512_batch(self, device):
        actual, expected = _run(
            [120, 500],
            n_heads=8,
            n_kv_heads=1,
            head_dim=512,
            page_size=32,
            device=device,
            sliding_window=1024,
        )
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


class TestTritonDecodeFp8Cache:
    """FP8 KV cache is dequantized in-kernel and needs no conversion pass."""

    @pytest.mark.parametrize("head_dim", [128, 512])
    def test_fp8_kv_cache(self, head_dim, device):
        actual, expected = _run(
            [64],
            n_heads=8,
            n_kv_heads=2,
            head_dim=head_dim,
            page_size=16,
            device=device,
            kv_dtype=torch.float8_e4m3fn,
        )
        # Looser bound: e4m3 round-trip dominates the error.
        torch.testing.assert_close(actual, expected, atol=2e-1, rtol=2e-1)

    def test_fp8_sliding_window(self, device):
        actual, expected = _run(
            [256],
            n_heads=8,
            n_kv_heads=2,
            head_dim=128,
            page_size=16,
            device=device,
            sliding_window=64,
            kv_dtype=torch.float8_e4m3fn,
        )
        torch.testing.assert_close(actual, expected, atol=2e-1, rtol=2e-1)


class TestTritonDecodeOutputTensor:
    """The caller-supplied ``out`` tensor is written in place."""

    def test_out_written_in_place(self, device):
        torch.manual_seed(3)
        n_heads, n_kv_heads, head_dim, page_size = 8, 2, 128, 16
        q = torch.randn(1, n_heads, head_dim, dtype=torch.bfloat16, device=device)
        k_list = [torch.randn(40, n_kv_heads, head_dim, dtype=torch.bfloat16, device=device)]
        v_list = [torch.randn(40, n_kv_heads, head_dim, dtype=torch.bfloat16, device=device)]
        kv_cache, kv_indices, kv_indptr, last_page = _build_paged_cache(
            k_list, v_list, page_size, device
        )

        out = torch.zeros_like(q)
        sm_scale = 1.0 / math.sqrt(head_dim)
        returned = triton_decode(
            q=q,
            kv_cache=kv_cache,
            kv_indices=kv_indices,
            kv_indptr=kv_indptr,
            kv_last_page_len=last_page,
            sm_scale=sm_scale,
            out=out,
        )

        assert returned.data_ptr() == out.data_ptr()
        expected = _decode_reference(q, k_list, v_list, sm_scale)
        torch.testing.assert_close(out.float(), expected, atol=2e-2, rtol=2e-2)
