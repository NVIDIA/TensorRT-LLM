# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paged-context refusals of the native attention op.

``TrtllmAttentionMetadata.__post_init__`` enables ``use_paged_context_fmha``
whenever chunked prefill, KV block reuse or speculative draft tokens are
configured. Without a fused context FMHA kernel for the configuration, the
context phase would reach the unfused library, whose context path builds K/V
from the current chunk only: the cached prefix would be dropped from attention
and then overwritten by the chunk's write-back. ``unfused_is_supported``
refuses that call with an error instead. The trtllm-gen library serves paged
KV at every page size: the cubin table carries 32-token pages, and NVRTC
generates the kernels of the others.
"""

import pytest
import torch
from fmha_test_utils import native_batch, native_forward_args

from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha
from tensorrt_llm._torch.attention.backends.fmha.trtllm_gen import TrtllmGenFmha
from tensorrt_llm._torch.attention.backends.fmha.unfused import UnfusedFmha
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

# FP32 attention has no fused context kernel, so its context phase runs unfused.
_UNFUSED_DTYPE = torch.float32


def _is_supported(fmha_cls, batch, **overrides) -> bool:
    attn, metadata, qkv = batch
    forward_args = native_forward_args(attn, metadata, qkv, **overrides)
    return fmha_cls(attn).is_supported(qkv, None, None, metadata, forward_args)


def test_paged_context_without_fused_kernel_is_refused():
    with native_batch(
        _UNFUSED_DTYPE, num_contexts=1, num_generations=0, chunked_prefill=True
    ) as batch:
        with pytest.raises(RuntimeError, match="requires a fused context FMHA kernel"):
            _is_supported(FallbackFmha, batch)
        with pytest.raises(RuntimeError, match="requires a fused context FMHA kernel"):
            _is_supported(UnfusedFmha, batch)


def test_unfused_context_without_paged_context_is_admitted():
    with native_batch(_UNFUSED_DTYPE, num_contexts=1, num_generations=0) as batch:
        assert _is_supported(UnfusedFmha, batch)


def test_generation_only_batch_skips_the_paged_context_refusal():
    with native_batch(
        _UNFUSED_DTYPE, num_contexts=0, num_generations=2, chunked_prefill=True
    ) as batch:
        assert _is_supported(FallbackFmha, batch)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not 100 <= get_sm_version() < 110,
    reason="trtllm-gen kernels need an SM100-family GPU",
)
@pytest.mark.parametrize("tokens_per_block", [16, 64, 128])
@pytest.mark.parametrize(
    "num_contexts,num_generations", [(1, 0), (0, 2)], ids=["paged_prefill", "generation"]
)
def test_trtllm_gen_serves_page_sizes_outside_the_cubin_table(
    tokens_per_block, num_contexts, num_generations
):
    with native_batch(
        torch.bfloat16,
        num_contexts=num_contexts,
        num_generations=num_generations,
        chunked_prefill=True,
        tokens_per_block=tokens_per_block,
    ) as batch:
        assert _is_supported(TrtllmGenFmha, batch)
