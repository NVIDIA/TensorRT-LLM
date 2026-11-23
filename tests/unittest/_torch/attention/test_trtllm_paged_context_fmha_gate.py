# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paged-context refusals of the native attention op.

``TrtllmAttentionMetadata.__post_init__`` enables ``use_paged_context_fmha``
whenever chunked prefill, KV block reuse or speculative draft tokens are
configured. Without a fused context FMHA kernel for the configuration, the
context phase would reach the unfused library, whose context path builds K/V
from the current chunk only: the cached prefix would be dropped from attention
and then overwritten by the chunk's write-back. ``unfused_is_supported``
declines that call instead. A page size that only the trtllm-gen
kernels outside the op read is declined by every native library, so selection
continues to those kernels.
"""

<<<<<<< HEAD
=======
from copy import copy
from types import SimpleNamespace
from unittest import mock

>>>>>>> origin/main
import pytest
import torch
from fmha_test_utils import native_batch, native_forward_args

from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha
<<<<<<< HEAD
from tensorrt_llm._torch.attention.backends.fmha.trtllm_gen import TrtllmGenFmha
from tensorrt_llm._torch.attention.backends.fmha.unfused import UnfusedFmha
=======
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionRuntimeFeatures,
    PredefinedAttentionMask,
)
from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheManager
from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal import thop
>>>>>>> origin/main

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

# FP32 attention has no fused context kernel, so its context phase runs unfused.
_UNFUSED_DTYPE = torch.float32


<<<<<<< HEAD
def _is_supported(fmha_cls, batch, **overrides) -> bool:
    attn, metadata, qkv = batch
    forward_args = native_forward_args(attn, metadata, qkv, **overrides)
    return fmha_cls(attn).is_supported(qkv, None, None, metadata, forward_args)
=======
@pytest.mark.cpu_only
@pytest.mark.parametrize("sm_version", [80, 86, 89, 90, 100, 103, 120, 121])
@pytest.mark.parametrize("scratch", [False, True])
def test_swa_scratch_architecture_gate(sm_version, scratch):
    manager = object.__new__(KVCacheManagerV2)
    manager.enable_swa_scratch_reuse = scratch
    with (
        mock.patch(
            "tensorrt_llm._torch.attention.backends.trtllm.get_sm_version",
            return_value=sm_version,
        ),
        mock.patch.object(TrtllmAttentionMetadata, "_post_init_with_buffers") as allocate,
    ):
        # SM120/121 run the fmha_v2 kernels that dereference stale per-layer pages.
        if scratch and sm_version not in (90, 100, 103):
            with pytest.raises(ValueError, match="only on SM90 and SM100-family"):
                TrtllmAttentionMetadata(
                    max_num_requests=1, max_num_tokens=32, kv_cache_manager=manager
                )
            allocate.assert_not_called()
        else:
            TrtllmAttentionMetadata(max_num_requests=1, max_num_tokens=32, kv_cache_manager=manager)
            allocate.assert_called_once()


@pytest.mark.cpu_only
@pytest.mark.parametrize("sm_version", [80, 86, 89, 90, 100, 103, 120, 121])
@pytest.mark.parametrize("scratch", [False, True])
def test_csa2_swa_scratch_architecture_gate(sm_version: int, scratch: bool) -> None:
    manager = object.__new__(CSA2CacheManager)
    manager.enable_swa_scratch_reuse = scratch
    with (
        mock.patch(
            "tensorrt_llm._torch.attention.backends.trtllm.get_sm_version",
            return_value=sm_version,
        ),
        mock.patch(
            "tensorrt_llm._torch.attention.backends.sparse.csa2.metadata.get_sm_version",
            return_value=sm_version,
        ),
        mock.patch.object(TrtllmAttentionMetadata, "_post_init_with_buffers") as allocate,
    ):
        if scratch and sm_version not in (90, 100, 103, 120, 121):
            with pytest.raises(
                ValueError, match=f"CSA2 attention is unsupported on SM{sm_version}"
            ):
                CSA2TrtllmMetadata(max_num_requests=1, max_num_tokens=32, kv_cache_manager=manager)
            allocate.assert_not_called()
        else:
            metadata = CSA2TrtllmMetadata(
                max_num_requests=1, max_num_tokens=32, kv_cache_manager=manager
            )
            allocate.assert_called_once_with(None)
            allocate.reset_mock()

            clone = copy(metadata)
            clone.is_cuda_graph = True
            clone.__post_init__()
            allocate.assert_called_once_with(None)
            assert clone.kv_cache_manager is manager


def _make_metadata(head_dim, features, kv_dtype=_ABSENT, tokens_per_block=32):
    """Construct real metadata with no CUDA buffers.
>>>>>>> origin/main


def test_paged_context_without_fused_kernel_is_declined():
    with native_batch(
        _UNFUSED_DTYPE, num_contexts=1, num_generations=0, chunked_prefill=True
    ) as batch:
        assert not _is_supported(FallbackFmha, batch)
        assert not _is_supported(UnfusedFmha, batch)


def test_unfused_context_without_paged_context_is_admitted():
    with native_batch(_UNFUSED_DTYPE, num_contexts=1, num_generations=0) as batch:
        assert _is_supported(UnfusedFmha, batch)


def test_generation_only_batch_skips_the_paged_context_refusal():
    with native_batch(
        _UNFUSED_DTYPE, num_contexts=0, num_generations=2, chunked_prefill=True
    ) as batch:
        assert _is_supported(FallbackFmha, batch)


@pytest.mark.parametrize("fmha_cls", [FallbackFmha, TrtllmGenFmha, UnfusedFmha])
def test_trtllm_gen_only_page_size_declines_paged_prefill(fmha_cls):
    with native_batch(
        torch.bfloat16,
        num_contexts=1,
        num_generations=0,
        chunked_prefill=True,
        extra_page_sizes=frozenset({32}),
    ) as batch:
        assert not _is_supported(fmha_cls, batch)


def test_trtllm_gen_only_page_size_admits_generation():
    with native_batch(
        torch.bfloat16,
        num_contexts=0,
        num_generations=2,
        chunked_prefill=True,
        extra_page_sizes=frozenset({32}),
    ) as batch:
        assert _is_supported(FallbackFmha, batch)


def test_unfused_declines_fp8_output():
    if torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("FP8 output needs SM89 or newer")
    with native_batch(torch.bfloat16, num_contexts=1, num_generations=0) as batch:
        attn, _, qkv = batch
        output = torch.empty(
            (qkv.shape[0], attn.num_heads * attn.head_dim),
            dtype=torch.float8_e4m3fn,
            device=qkv.device,
        )
        assert not _is_supported(UnfusedFmha, batch, output=output)
