# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage for packed-QKV context FMHA with unequal prompt lengths.

Split out of #18953 so the kernel change in #19698 can be reviewed and reverted
on its own. The packed-QKV path loads ``cu_kv_seqlens`` but consumes nothing
from it, so this is layout coverage, not a regression test for that read.
"""

import pytest
import torch
from backend_case import BackendCase, generate_inputs, run_backend

from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("num_kv_heads", [4, 2], ids=["mha", "gqa"])
@pytest.mark.parametrize("prompt_lens", [[31, 7], [63, 31]], ids=["short", "page_boundary"])
@torch.inference_mode()
def test_packed_qkv_fmha_context_lengths(
    dtype: str, num_kv_heads: int, prompt_lens: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unequal prompt lengths reach native packed FMHA with full-precision KV."""
    monkeypatch.setenv("TLLM_FMHA_LIBS", "fallback")
    original_forward = FallbackFmha.forward
    launches = []

    def checked_forward(self, q, k, v, metadata, forward_args):
        """Pin the layout under test: packed QKV, two contexts, full-precision KV."""
        assert k is None and v is None
        assert metadata.num_contexts == 2
        assert not metadata.use_paged_context_fmha
        assert not self.attn.quant_config.layer_quant_mode.has_kv_cache_quant()
        launches.append(metadata.num_contexts)
        return original_forward(self, q, k, v, metadata, forward_args)

    monkeypatch.setattr(FallbackFmha, "forward", checked_forward)
    case = BackendCase(
        num_heads=4,
        num_kv_heads=num_kv_heads,
        head_dim=128,
        seq_lens=prompt_lens,
        num_cached_tokens=[0, 0],
        num_contexts=2,
        dtype=dtype,
        page_size=64,
    )
    inputs = generate_inputs(case, seed=47)
    expected = run_backend(case, "VANILLA", inputs, kv_dtype=case.compute_dtype, kv_layout="NHD")
    actual = run_backend(case, "TRTLLM", inputs, kv_dtype=case.compute_dtype, kv_layout="HND")
    assert launches
    atol, rtol = (0.04, 0.01) if dtype == "bfloat16" else (0.015, 0.005)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
