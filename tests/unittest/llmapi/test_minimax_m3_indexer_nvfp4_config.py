# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMaxM3SparseAttentionConfig.indexer_kv_dtype='nvfp4' (the NVFP4 index-K of the MiniMax-M3.1-Flash targets)."""

import pytest

from tensorrt_llm.llmapi.llm_args import MiniMaxM3SparseAttentionConfig

pytestmark = pytest.mark.cpu_only

_KILL = "TRTLLM_M31_DISABLE_AGG_NVFP4_INDEXK"


def _config(**kwargs) -> MiniMaxM3SparseAttentionConfig:
    return MiniMaxM3SparseAttentionConfig(
        **{
            "implementation": "msa",
            "fuse_qkv_index_projection": True,
            "indexer_kv_dtype": "nvfp4",
            **kwargs,
        }
    )


def test_nvfp4_is_accepted_on_the_fused_msa_path(monkeypatch):
    monkeypatch.delenv(_KILL, raising=False)
    assert _config().indexer_kv_dtype == "nvfp4"


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"implementation": "triton", "fuse_qkv_index_projection": False}, "'msa'"),
        ({"fuse_qkv_index_projection": False}, "fuse_qkv_index_projection=True"),
        ({"sparse_disable_index_value": False}, "sparse_disable_index_value=True"),
    ],
)
def test_nvfp4_requirements(monkeypatch, kwargs, message):
    monkeypatch.delenv(_KILL, raising=False)
    with pytest.raises(ValueError, match=message):
        _config(**kwargs)


def test_kill_switch_falls_back_to_fp8(monkeypatch):
    monkeypatch.setenv(_KILL, "1")
    assert _config().indexer_kv_dtype == "fp8"
