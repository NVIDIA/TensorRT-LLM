# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kernel-presence lookup for paged-context FMHA.

``TrtllmAttentionMetadata`` enables ``use_paged_context_fmha`` whenever chunked
prefill, KV block reuse or speculative draft tokens are configured, and those
features require the context phase to attend to KV already in the cache. Only
the fused context FMHA kernel can do that: ``AttentionOp::initialize()`` ends
with ``mEnableContextFMHA = mIsGenerationMLA || mFmhaDispatcher->isSupported()``,
so a configuration with no compiled kernel silently falls back to the unfused
path, which attends to the current chunk only and then overwrites the cached
prefix.

The ``AttentionOp`` constructor refuses that combination after initialization. This module
covers the ``fused_context_fmha_kernel_exists`` diagnostic that reports what the
running build actually contains, including the case that matters most: that the
lookup is able to answer "no". Every test here calls into the native kernel
table, so all of them need a GPU. Missing bindings on a GPU are a failure, not a
skip.
"""

import pytest
import torch

from tensorrt_llm._torch.attention.backends.fmha.interface import StaticAttentionConfig
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.functional import PositionEmbeddingType

# The paged-KV context FMHA kernels are generated for 32-token pages.
_TOKENS_PER_BLOCK = 32
# A head width the SM100-family dispatcher has no context kernel for. Other
# architectures route to FMHA-v2 kernels with a different supported set, so
# this absence is only asserted within the SM100 family.
_UNSUPPORTED_HEAD_SIZE = 96
# A head width the SM100-family kernel table does carry for 16-bit paged-context
# attention.
_SUPPORTED_HEAD_SIZE = 64

_NO_GPU = not torch.cuda.is_available()
_NO_GPU_REASON = "needs a GPU; missing bindings on a GPU are a test failure"

pytestmark = pytest.mark.skipif(_NO_GPU, reason=_NO_GPU_REASON)


def _is_sm100_family() -> bool:
    return 100 <= get_sm_version() < 110


@pytest.mark.parametrize(
    "head_size,position_embedding_type,error",
    [
        (96, PositionEmbeddingType.learned_absolute, "requires a fused context FMHA kernel"),
        (64, PositionEmbeddingType.relative, "not supported with relative position embedding"),
    ],
    ids=["missing-kernel", "relative-position"],
)
def test_attention_op_rejects_unfused_paged_context(head_size, position_embedding_type, error):
    if not _is_sm100_family():
        pytest.skip("the unsupported head-size case targets the SM100-family dispatcher")
    config = StaticAttentionConfig(
        num_heads=1,
        num_kv_heads=1,
        head_size=head_size,
        tokens_per_block=_TOKENS_PER_BLOCK,
        type=DataType.BF16,
        use_kv_cache=True,
        paged_context_fmha=True,
        position_embedding_type=position_embedding_type,
    )
    with pytest.raises(RuntimeError, match=error):
        thop.AttentionOp(config.to_thop_config())


@pytest.mark.parametrize(
    "head_size,tokens_per_block",
    [(0, 32), (-1, 32), (64, 0), (64, -1)],
)
def test_lookup_rejects_invalid_dimensions(head_size, tokens_per_block):
    """Non-positive dimensions are rejected before any dispatcher query."""
    assert not thop.fused_context_fmha_kernel_exists(
        head_size=head_size,
        kv_cache_dtype=DataType.FP8,
        tokens_per_block=tokens_per_block,
        output_dtype=DataType.BF16,
    )


@pytest.mark.parametrize(
    "kv_cache_dtype,output_dtype",
    [
        (DataType.FLOAT, DataType.BF16),
        (DataType.INT8, DataType.BF16),
        (DataType.FP8, DataType.FLOAT),
        (DataType.FP8, DataType.INT8),
    ],
    ids=["float_kv", "int8_kv", "float_out", "int8_out"],
)
def test_lookup_rejects_unsupported_precisions(kv_cache_dtype, output_dtype):
    """Precisions with no context FMHA representation report absent rather than
    mapping onto some other kernel."""
    assert not thop.fused_context_fmha_kernel_exists(
        head_size=_SUPPORTED_HEAD_SIZE,
        kv_cache_dtype=kv_cache_dtype,
        tokens_per_block=_TOKENS_PER_BLOCK,
        output_dtype=output_dtype,
    )


@pytest.mark.parametrize(
    "kv_cache_dtype,output_dtype",
    [
        (DataType.BF16, DataType.BF16),
        (DataType.HALF, DataType.HALF),
    ],
    ids=["bf16", "fp16"],
)
def test_lookup_reports_present_for_a_built_configuration(kv_cache_dtype, output_dtype):
    """The lookup must be able to answer "yes".

    Without a known-present case, an always-false lookup would pass every other
    test in this module. The SM100-family kernel table carries dense causal
    paged-context kernels at head size 64 for matched 16-bit Q/KV and output.
    """
    if not _is_sm100_family():
        pytest.skip("the known-present configuration is asserted for the SM100 family")
    assert thop.fused_context_fmha_kernel_exists(
        head_size=_SUPPORTED_HEAD_SIZE,
        kv_cache_dtype=kv_cache_dtype,
        tokens_per_block=_TOKENS_PER_BLOCK,
        output_dtype=output_dtype,
    )


def test_lookup_reports_present_for_nvfp4_kv_cache():
    """An NVFP4 KV cache is read by an FP8 Q kernel, not a matched-precision one.

    ``AttentionOp`` requires ``mFP8ContextFMHA`` for an FP4 KV cache and then
    selects an E4M3 Q / E2M1 KV context kernel. The SM100-family kernel table
    carries that combination at head size 64 with FP8 output, so the lookup
    must answer "yes" for it rather than treat NVFP4 as an unknown precision.
    """
    if not _is_sm100_family():
        pytest.skip("E2M1-KV context kernels exist only in the SM100-family kernel set")
    assert thop.fused_context_fmha_kernel_exists(
        head_size=_SUPPORTED_HEAD_SIZE,
        kv_cache_dtype=DataType.NVFP4,
        tokens_per_block=_TOKENS_PER_BLOCK,
        output_dtype=DataType.FP8,
    )


def test_lookup_reports_absent_for_nvfp4_kv_cache_at_fmha_v2_head_size():
    """Head size 72 is routed to FMHA-v2 even on the SM100 family, and that
    runner asserts matched Q/KV precision. The lookup must answer "no" for
    NVFP4 KV there without raising, mirroring the dispatcher's routing rule."""
    if not _is_sm100_family():
        pytest.skip("the head-size-72 routing exception is specific to the SM100-family dispatcher")
    assert not thop.fused_context_fmha_kernel_exists(
        head_size=72,
        kv_cache_dtype=DataType.NVFP4,
        tokens_per_block=_TOKENS_PER_BLOCK,
        output_dtype=DataType.FP8,
    )


def test_lookup_reports_absent_for_nvfp4_kv_cache_outside_sm100_family():
    """Off the SM100 family the lookup must answer "no" for NVFP4 KV without
    raising: the FMHA-v2 dispatcher asserts that Q and KV precision match, and
    the probe has to refuse before constructing it."""
    if _is_sm100_family():
        pytest.skip("the SM100-family kernel set does carry E2M1-KV context kernels")
    assert not thop.fused_context_fmha_kernel_exists(
        head_size=_SUPPORTED_HEAD_SIZE,
        kv_cache_dtype=DataType.NVFP4,
        tokens_per_block=_TOKENS_PER_BLOCK,
        output_dtype=DataType.FP8,
    )


def test_lookup_reports_absent_for_an_unbuilt_head_size():
    """The lookup must be able to answer "no".

    A build whose ``--cuda_architectures`` omits this device's SM has no kernels
    for it at all and reports absent for every query, which this test cannot
    distinguish from a genuine absence. That is the correct answer either way:
    on such a build the runtime would fall back to unfused MHA.
    """
    if not _is_sm100_family():
        pytest.skip("the unsupported head-size case targets the SM100-family dispatcher")
    assert not thop.fused_context_fmha_kernel_exists(
        head_size=_UNSUPPORTED_HEAD_SIZE,
        kv_cache_dtype=DataType.BF16,
        tokens_per_block=_TOKENS_PER_BLOCK,
        output_dtype=DataType.BF16,
    )


def test_lookup_is_page_size_sensitive():
    """Page size is part of the kernel identity, not an ignored argument."""
    if not _is_sm100_family():
        pytest.skip("page-size sensitivity is asserted for the SM100 family")
    present = thop.fused_context_fmha_kernel_exists(
        head_size=_SUPPORTED_HEAD_SIZE,
        kv_cache_dtype=DataType.BF16,
        tokens_per_block=_TOKENS_PER_BLOCK,
        output_dtype=DataType.BF16,
    )
    odd_page = thop.fused_context_fmha_kernel_exists(
        head_size=_SUPPORTED_HEAD_SIZE,
        kv_cache_dtype=DataType.BF16,
        tokens_per_block=_TOKENS_PER_BLOCK + 1,
        output_dtype=DataType.BF16,
    )
    assert present
    assert not odd_page
