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
"""Standalone DSpark drafter state in disaggregated serving.

The unified draft cache keeps the drafter's KV history in the target's
KVCacheManagerV2 and ships it from the context worker to the generation worker.
Drafters it cannot hold (MLA) keep the legacy drafter state instead of failing
startup, and ``TLLM_DSPARK_DISABLE_UNIFIED_DRAFT_CACHE=1`` forces the legacy
state for every drafter. With legacy state the generation manager owns no draft
layers, so a request that arrives without draft history (a context worker that
runs no drafter) is accepted instead of raising in ``_restore_draft_history``.
"""

from types import SimpleNamespace
from unittest import mock

import pytest

from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm._torch.pyexecutor import _util
from tensorrt_llm._torch.pyexecutor._util import KvCacheCreator
from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode

pytestmark = pytest.mark.cpu_only

_DISABLE_ENV = "TLLM_DSPARK_DISABLE_UNIFIED_DRAFT_CACHE"


def _gqa_drafter():
    """A GQA drafter (RadixArk-style): no MLA latent fields."""
    return SimpleNamespace(
        pretrained_config=SimpleNamespace(
            num_hidden_layers=5,
            hidden_size=7168,
            num_attention_heads=64,
            num_key_value_heads=8,
            head_dim=128,
        )
    )


def _mla_drafter():
    """An MLA drafter (Inferact Kimi-K3 DSpark): is_mla() keys on these two fields."""
    return SimpleNamespace(
        pretrained_config=SimpleNamespace(
            num_hidden_layers=5,
            hidden_size=7168,
            num_attention_heads=64,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
        )
    )


def _creator(draft_config, *, is_disagg=True, kv_cache_manager_v2=True):
    """A KvCacheCreator with only the fields the draft-cache decision reads."""
    creator = object.__new__(KvCacheCreator)
    creator._speculative_config = SimpleNamespace(
        spec_dec_mode=SpeculativeDecodingMode.DSPARK,
        draft_is_embedded_in_target=False,
        _use_shared_kv_cache=False,
    )
    creator._draft_config = draft_config
    creator._is_disagg = is_disagg
    creator._is_kv_cache_manager_v2 = kv_cache_manager_v2
    creator._mapping = SimpleNamespace(enable_attention_dp=True)
    return creator


@pytest.fixture
def separate_draft_manager(monkeypatch):
    """Records whether the separate (legacy) draft-manager decision was reached."""
    calls = []

    def _should_use_separate(spec_config):
        calls.append(spec_config)
        return True

    monkeypatch.setattr(_util, "should_use_separate_draft_kv_cache", _should_use_separate)
    return calls


def test_gqa_drafter_keeps_unified_cache_by_default(monkeypatch, separate_draft_manager):
    monkeypatch.delenv(_DISABLE_ENV, raising=False)
    creator = _creator(_gqa_drafter())

    with mock.patch.object(_util.logger, "warning") as warning:
        creator._validate_standalone_draft_cache()

    assert creator._uses_unified_standalone_draft_cache()
    # The unified cache replaces the separate draft manager.
    assert not creator._should_create_separate_draft_kv_cache()
    assert separate_draft_manager == []
    warning.assert_not_called()


def test_env_forces_legacy_state_for_a_gqa_drafter(monkeypatch, separate_draft_manager):
    monkeypatch.setenv(_DISABLE_ENV, "1")
    creator = _creator(_gqa_drafter())

    with mock.patch.object(_util.logger, "warning") as warning:
        creator._validate_standalone_draft_cache()

    assert not creator._uses_unified_standalone_draft_cache()
    assert creator._should_create_separate_draft_kv_cache()
    assert separate_draft_manager == [creator._speculative_config]
    warning.assert_not_called()


@pytest.mark.parametrize("value", ["0", "", "true"])
def test_only_env_value_1_disables_the_unified_cache(monkeypatch, value):
    monkeypatch.setenv(_DISABLE_ENV, value)

    assert _creator(_gqa_drafter())._uses_unified_standalone_draft_cache()


def test_env_skips_the_disagg_kv_cache_manager_v2_requirement(monkeypatch):
    """Legacy drafter state transfers no draft history, so it needs no KVCMv2."""
    creator = _creator(_gqa_drafter(), kv_cache_manager_v2=False)

    monkeypatch.delenv(_DISABLE_ENV, raising=False)
    with pytest.raises(ValueError, match="use_kv_cache_manager_v2"):
        creator._validate_standalone_draft_cache()

    monkeypatch.setenv(_DISABLE_ENV, "1")
    creator._validate_standalone_draft_cache()
    assert not creator._uses_unified_standalone_draft_cache()


@pytest.mark.parametrize("is_disagg", [True, False], ids=["disagg", "aggregate"])
def test_mla_drafter_falls_back_to_legacy_state(monkeypatch, separate_draft_manager, is_disagg):
    """Without the env an MLA drafter warns and keeps legacy state; startup does not fail."""
    monkeypatch.delenv(_DISABLE_ENV, raising=False)
    creator = _creator(_mla_drafter(), is_disagg=is_disagg)

    with mock.patch.object(_util.logger, "warning") as warning:
        creator._validate_standalone_draft_cache()

    assert not creator._uses_unified_standalone_draft_cache()
    assert creator._should_create_separate_draft_kv_cache()
    assert separate_draft_manager == [creator._speculative_config]
    if is_disagg:
        warning.assert_called_once()
        message = warning.call_args.args[0]
        assert "does not yet support MLA drafters" in message
        assert "legacy drafter state" in message
    else:
        # Aggregate serving transfers nothing, so there is nothing to warn about.
        warning.assert_not_called()


def _restore(manager_draft_layer_ids, history=None):
    receiver = SimpleNamespace(
        _kv_cache_manager=SimpleNamespace(draft_layer_ids=manager_draft_layer_ids)
    )
    request = SimpleNamespace(py_request_id=7, py_draft_transfer_history=history)
    return KvCacheTransceiverV2._restore_draft_history(receiver, request)


def test_generation_without_draft_history_needs_legacy_state():
    """Why the fallback matters at GEN when the context worker runs no drafter.

    A unified-cache generation manager owns draft layers and rejects a request
    without draft history; with legacy state it owns none and accepts it.
    """
    with pytest.raises(ValueError, match="requires draft history"):
        _restore(manager_draft_layer_ids=(61,))

    assert _restore(manager_draft_layer_ids=()) is None
