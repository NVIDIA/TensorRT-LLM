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
"""`_create_kv_cache_manager` passes `is_disagg` once to the manager it builds.

The attention-only and MLA branches pass it to every manager, including a
`MambaHybridCacheManagerV2` subclass that a model declares
(`RuntimeHooks.kv_cache_manager_cls`) for a configuration the family predicates
do not mark as hybrid. The hybrid branches pass it to the V2 manager only: the
V1 hybrid managers do not take it.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import _util
from tensorrt_llm._torch.pyexecutor.config_utils import MambaKVCacheParams
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import (
    CppMambaHybridCacheManager,
    MambaHybridCacheManagerV2,
)
from tensorrt_llm.llmapi.llm_args import KvCacheConfig

pytestmark = pytest.mark.cpu_only

#: The predicate that selects each branch of `_create_kv_cache_manager`.
_BRANCH_PREDICATES = {
    "kda": "is_kimi_linear",
    "mla": "is_mla",
    "mamba2": "is_nemotron_hybrid",
    "gdn": "is_qwen3_hybrid",
    "attention": None,
}
_HYBRID_BRANCHES = ("kda", "mamba2", "gdn")


def _capturing(manager_cls: type) -> type:
    """A subclass of `manager_cls` that records its arguments instead of building a cache."""

    class Capturing(manager_cls):
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.args = args
            self.kwargs = kwargs

    Capturing.__name__ = Capturing.__qualname__ = f"Capturing{manager_cls.__name__}"
    return Capturing


def _build(monkeypatch, branch: str, manager_cls: type, is_disagg: bool):
    """Build `manager_cls` through the branch of `_create_kv_cache_manager` named `branch`."""
    predicates = (
        "is_kimi_linear",
        "is_mla",
        "is_nemotron_hybrid",
        "is_qwen3_hybrid",
        "is_qwen4_exp",
    )
    for name in predicates:
        monkeypatch.setattr(_util, name, lambda config: False)
    monkeypatch.setattr(_util.config_utils, "is_glm5_next", lambda config: False)
    if _BRANCH_PREDICATES[branch] is not None:
        monkeypatch.setattr(_util, _BRANCH_PREDICATES[branch], lambda config: True)
    mamba_params = MambaKVCacheParams(
        state_size=16,
        conv_kernel=4,
        num_heads=4,
        n_groups=1,
        head_dim=8,
        mamba_layer_mask=[True, False],
        target_full_attention_layer_mask=[False, True],
        num_mamba_layers=1,
        num_draft_layers=0,
        dtype=torch.bfloat16,
        mamba_ssm_cache_dtype=torch.bfloat16,
    )
    monkeypatch.setattr(
        _util, "extract_mamba_kv_cache_params", lambda *args, **kwargs: mamba_params
    )
    pretrained_config = SimpleNamespace(
        vocab_size=1024,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=2,
        kv_lora_rank=32,
        qk_rope_head_dim=16,
    )
    return _util._create_kv_cache_manager(
        model_engine=None,
        kv_cache_manager_cls=manager_cls,
        mapping=SimpleNamespace(enable_attention_dp=False),
        kv_cache_config=KvCacheConfig(),
        tokens_per_block=32,
        max_seq_len=256,
        max_batch_size=2,
        spec_config=None,
        sparse_attention_config=None,
        max_num_tokens=256,
        max_beam_width=1,
        kv_connector_manager=None,
        model_config=SimpleNamespace(pretrained_config=pretrained_config, quant_config=None),
        dtype=torch.bfloat16,
        is_draft=False,
        is_disagg=is_disagg,
    )


@pytest.mark.parametrize("is_disagg", [True, False])
@pytest.mark.parametrize("branch", list(_BRANCH_PREDICATES))
def test_a_hybrid_v2_manager_gets_is_disagg_once(monkeypatch, branch, is_disagg):
    manager = _build(monkeypatch, branch, _capturing(MambaHybridCacheManagerV2), is_disagg)
    assert manager.kwargs["is_disagg"] is is_disagg


@pytest.mark.parametrize("branch", ["attention", "mla"])
def test_an_attention_manager_gets_is_disagg(monkeypatch, branch):
    manager = _build(monkeypatch, branch, _capturing(KVCacheManagerV2), True)
    assert manager.kwargs["is_disagg"] is True


@pytest.mark.parametrize("branch", _HYBRID_BRANCHES)
def test_a_v1_hybrid_manager_gets_no_is_disagg(monkeypatch, branch):
    manager = _build(monkeypatch, branch, _capturing(CppMambaHybridCacheManager), True)
    assert "is_disagg" not in manager.kwargs
