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
"""Which DSA context attention backend _use_flash_mla_sparse_prefill selects."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.attention.backends.sparse.dsa import module as dsa_module


def _mla(num_heads_tp, *, fp8_kv=False, fp4_kv=False, kv_cache_dtype="auto", cp_size=1):
    return SimpleNamespace(
        num_heads_tp=num_heads_tp,
        kv_cache_dtype=kv_cache_dtype,
        mapping=SimpleNamespace(cp_size=cp_size),
        mqa=SimpleNamespace(has_fp8_kv_cache=fp8_kv, has_fp4_kv_cache=fp4_kv),
    )


@pytest.fixture
def sm100(monkeypatch):
    monkeypatch.setattr(dsa_module, "get_sm_version", lambda: 103)
    monkeypatch.setattr(dsa_module, "flash_mla_sparse_fwd", object())
    monkeypatch.delenv("TRTLLM_DSA_FLASH_MLA_PREFILL", raising=False)


@pytest.mark.parametrize(
    "num_heads_tp,expected",
    [
        (96, True),  # Kimi K3.1 under attention DP: TRTLLM-Gen only has Q32 tiles
        (128, False),  # DeepSeek-V3.2 / GLM attention DP: native Q64 tiles
        (64, False),
        (12, False),  # Kimi K3.1 TP8
        (6, False),  # Kimi K3.1 TP16
    ],
)
def test_auto_takes_flash_mla_only_for_q32_tiled_head_groups(sm100, num_heads_tp, expected):
    assert dsa_module._use_flash_mla_sparse_prefill(_mla(num_heads_tp)) is expected


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(fp8_kv=True),
        dict(fp4_kv=True),
        dict(kv_cache_dtype="fp8_ds_mla"),
        dict(cp_size=2),
    ],
)
def test_needs_bf16_kv_cache_and_no_context_parallelism(sm100, kwargs):
    assert dsa_module._use_flash_mla_sparse_prefill(_mla(96, **kwargs)) is False


def test_env_overrides(sm100, monkeypatch):
    monkeypatch.setenv("TRTLLM_DSA_FLASH_MLA_PREFILL", "0")
    assert dsa_module._use_flash_mla_sparse_prefill(_mla(96)) is False
    monkeypatch.setenv("TRTLLM_DSA_FLASH_MLA_PREFILL", "1")
    assert dsa_module._use_flash_mla_sparse_prefill(_mla(128)) is True
    # More than 128 heads cannot be padded to FlashMLA's 128-head group.
    assert dsa_module._use_flash_mla_sparse_prefill(_mla(256)) is False
    monkeypatch.setenv("TRTLLM_DSA_FLASH_MLA_PREFILL", "yes")
    with pytest.raises(ValueError, match="TRTLLM_DSA_FLASH_MLA_PREFILL"):
        dsa_module._use_flash_mla_sparse_prefill(_mla(96))


@pytest.mark.parametrize("sm", [90, 107, 120])
def test_off_outside_sm100_sm103_or_without_flash_mla(monkeypatch, sm):
    monkeypatch.delenv("TRTLLM_DSA_FLASH_MLA_PREFILL", raising=False)
    monkeypatch.setattr(dsa_module, "flash_mla_sparse_fwd", object())
    monkeypatch.setattr(dsa_module, "get_sm_version", lambda: sm)
    assert dsa_module._use_flash_mla_sparse_prefill(_mla(96)) is False
    monkeypatch.setattr(dsa_module, "get_sm_version", lambda: 103)
    monkeypatch.setattr(dsa_module, "flash_mla_sparse_fwd", None)
    assert dsa_module._use_flash_mla_sparse_prefill(_mla(96)) is False
