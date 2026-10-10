# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""V4 keeps its original ratio semantics and cannot accept a V4.1 variant."""

import pytest

from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import (
    DeepseekV4AttentionType,
    DeepSeekV4MetadataParams,
    DeepSeekV4Params,
    compress_ratio_has_attention,
    is_compress_layer,
    is_overlap_compressor,
    is_sparse_layer,
)


@pytest.mark.parametrize("params", [DeepSeekV4Params, DeepSeekV4MetadataParams])
@pytest.mark.parametrize("variant", ["v41", "V41", "deepseek_v41", "v4.1", ""])
def test_v4_params_reject_variant_dispatch(params, variant):
    with pytest.raises(TypeError, match="variant"):
        params(variant=variant)


def test_v4_ratio_semantics():
    assert is_sparse_layer(4)
    assert not is_sparse_layer(128)
    assert is_overlap_compressor(4)
    assert not is_compress_layer(1)


@pytest.mark.parametrize(
    "ratio,expected",
    [
        (1, {"SWA"}),
        (
            4,
            {
                "SWA",
                "COMPRESSOR_KV",
                "COMPRESSOR_SCORE",
                "INDEXER_COMPRESSOR_KV",
                "INDEXER_COMPRESSOR_SCORE",
                "COMPRESS",
                "INDEXER_COMPRESS",
            },
        ),
        (128, {"SWA", "COMPRESSOR_KV", "COMPRESSOR_SCORE", "COMPRESS"}),
    ],
)
def test_v4_cache_roles(ratio, expected):
    assert {
        role.name for role in DeepseekV4AttentionType if compress_ratio_has_attention(ratio, role)
    } == expected
