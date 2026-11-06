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
An MLA drafter keeps the legacy drafter state unless
``TLLM_DSPARK_UNIFIED_MLA_DRAFT_CACHE=1`` opts it in, and
``TLLM_DSPARK_DISABLE_UNIFIED_DRAFT_CACHE=1`` forces the legacy state for every
drafter. With legacy state the generation manager owns no draft layers, so a
request that arrives without draft history (a context worker that runs no
drafter) is accepted instead of raising in ``_restore_draft_history``.

``TLLM_DSPARK_DRAFT_CONTEXT_LIMIT`` makes a generation worker start a request
whose prompt is longer than the limit from an empty drafter context. The
context worker, whose prefix lookup and block commit the transferred draft
history takes part in, ignores it.

``TLLM_DSPARK_DRAFT_CONTEXT_WINDOW`` (with ``TLLM_DSPARK_DRAFT_CONTEXT_SINK``)
makes such a request keep the last tokens (and the first ones) of its own
transferred context instead.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm._torch.pyexecutor import _util, kv_cache_manager_v2
from tensorrt_llm._torch.pyexecutor._util import KvCacheCreator
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import (
    KVCacheManagerV2,
    draft_context_limit_from_env,
)
from tensorrt_llm._torch.pyexecutor.standalone_draft_cache import (
    DraftContextWindow,
    DraftContextWindowMove,
    StandaloneDraftHistory,
    draft_context_window_from_env,
)
from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode
from tensorrt_llm._utils import torch_dtype_to_binding

pytestmark = pytest.mark.cpu_only

_DISABLE_ENV = "TLLM_DSPARK_DISABLE_UNIFIED_DRAFT_CACHE"
_MLA_OPT_IN_ENV = "TLLM_DSPARK_UNIFIED_MLA_DRAFT_CACHE"
_ROLE_ENV = "TRTLLM_DISAGG_ROLE"
_CONTEXT_LIMIT_ENV = "TLLM_DSPARK_DRAFT_CONTEXT_LIMIT"
_CONTEXT_WINDOW_ENV = "TLLM_DSPARK_DRAFT_CONTEXT_WINDOW"
_CONTEXT_SINK_ENV = "TLLM_DSPARK_DRAFT_CONTEXT_SINK"


@pytest.fixture(autouse=True)
def _no_inherited_env_overrides(monkeypatch):
    monkeypatch.delenv(_ROLE_ENV, raising=False)
    monkeypatch.delenv(_CONTEXT_LIMIT_ENV, raising=False)
    monkeypatch.delenv(_CONTEXT_WINDOW_ENV, raising=False)
    monkeypatch.delenv(_CONTEXT_SINK_ENV, raising=False)


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
    """Without the opt-in an MLA drafter warns and keeps legacy state; startup does not fail."""
    monkeypatch.delenv(_DISABLE_ENV, raising=False)
    monkeypatch.delenv(_MLA_OPT_IN_ENV, raising=False)
    creator = _creator(_mla_drafter(), is_disagg=is_disagg)

    with mock.patch.object(_util.logger, "warning") as warning:
        creator._validate_standalone_draft_cache()

    assert not creator._uses_unified_standalone_draft_cache()
    assert creator._should_create_separate_draft_kv_cache()
    assert separate_draft_manager == [creator._speculative_config]
    if is_disagg:
        warning.assert_called_once()
        message = warning.call_args.args[0]
        assert _MLA_OPT_IN_ENV in message
        assert "legacy drafter state" in message
    else:
        # Aggregate serving transfers nothing, so there is nothing to warn about.
        warning.assert_not_called()


@pytest.mark.parametrize("is_disagg", [True, False], ids=["disagg", "aggregate"])
def test_mla_drafter_opt_in_uses_the_unified_cache(monkeypatch, separate_draft_manager, is_disagg):
    monkeypatch.delenv(_DISABLE_ENV, raising=False)
    monkeypatch.setenv(_MLA_OPT_IN_ENV, "1")
    creator = _creator(_mla_drafter(), is_disagg=is_disagg)

    with mock.patch.object(_util.logger, "warning") as warning:
        creator._validate_standalone_draft_cache()

    assert creator._uses_unified_standalone_draft_cache()
    # The unified cache replaces the separate draft manager.
    assert not creator._should_create_separate_draft_kv_cache()
    assert separate_draft_manager == []
    warning.assert_not_called()


@pytest.mark.parametrize("value", ["0", "", "true"])
def test_only_value_1_opts_the_mla_drafter_in(monkeypatch, value):
    monkeypatch.delenv(_DISABLE_ENV, raising=False)
    monkeypatch.setenv(_MLA_OPT_IN_ENV, value)

    assert not _creator(_mla_drafter())._uses_unified_standalone_draft_cache()


def test_disable_env_wins_over_the_mla_opt_in(monkeypatch, separate_draft_manager):
    monkeypatch.setenv(_DISABLE_ENV, "1")
    monkeypatch.setenv(_MLA_OPT_IN_ENV, "1")
    creator = _creator(_mla_drafter())

    with mock.patch.object(_util.logger, "warning") as warning:
        creator._validate_standalone_draft_cache()

    assert not creator._uses_unified_standalone_draft_cache()
    assert creator._should_create_separate_draft_kv_cache()
    warning.assert_not_called()


def test_mla_opt_in_does_not_change_a_gqa_drafter(monkeypatch):
    monkeypatch.delenv(_DISABLE_ENV, raising=False)
    monkeypatch.setenv(_MLA_OPT_IN_ENV, "1")

    assert _creator(_gqa_drafter())._uses_unified_standalone_draft_cache()


def test_mla_opt_in_still_requires_kv_cache_manager_v2(monkeypatch):
    monkeypatch.delenv(_DISABLE_ENV, raising=False)
    monkeypatch.setenv(_MLA_OPT_IN_ENV, "1")
    creator = _creator(_mla_drafter(), kv_cache_manager_v2=False)

    assert not creator._uses_unified_standalone_draft_cache()
    with pytest.raises(ValueError, match="use_kv_cache_manager_v2"):
        creator._validate_standalone_draft_cache()


@pytest.mark.parametrize("opted_in", [True, False], ids=["unified", "legacy"])
def test_draft_kv_budget_cap_is_rejected_only_with_the_unified_cache(monkeypatch, opted_in):
    """The cap sizes the separate draft manager, which the unified cache does not have."""
    monkeypatch.delenv(_DISABLE_ENV, raising=False)
    monkeypatch.setenv("TLLM_DRAFT_KV_GPU_BUDGET_GIB", "8")
    if opted_in:
        monkeypatch.setenv(_MLA_OPT_IN_ENV, "1")
    else:
        monkeypatch.delenv(_MLA_OPT_IN_ENV, raising=False)
    creator = _creator(_mla_drafter())

    if opted_in:
        with pytest.raises(ValueError, match="TLLM_DRAFT_KV_GPU_BUDGET_GIB"):
            creator._validate_standalone_draft_cache()
    else:
        creator._validate_standalone_draft_cache()


@pytest.mark.parametrize("mla_opt_in", [False, True], ids=["default", "mla_opt_in_set"])
@pytest.mark.parametrize("embedded", [False, True], ids=["gqa", "embedded"])
def test_draft_kv_budget_cap_is_ignored_by_the_default_unified_caches(
    monkeypatch, embedded, mla_opt_in
):
    """A GQA or embedded drafter has the unified cache by default and ignored the cap before."""
    monkeypatch.delenv(_DISABLE_ENV, raising=False)
    monkeypatch.setenv("TLLM_DRAFT_KV_GPU_BUDGET_GIB", "8")
    if mla_opt_in:
        monkeypatch.setenv(_MLA_OPT_IN_ENV, "1")
    else:
        monkeypatch.delenv(_MLA_OPT_IN_ENV, raising=False)
    creator = _creator(_gqa_drafter())
    creator._speculative_config.draft_is_embedded_in_target = embedded
    assert creator._uses_unified_standalone_draft_cache()

    creator._validate_standalone_draft_cache()


class _Mapping(SimpleNamespace):
    def has_cp_helix(self):
        return False


def _kwargs_creator(draft_model, draft_config, *, tp_size=8, attention_backend="AUTO"):
    """A KvCacheCreator with only the fields the unified draft kwargs read."""
    creator = _creator(draft_config)
    creator._speculative_config.attention_backend = attention_backend
    creator._speculative_config.max_draft_len = 5
    creator._mapping = _Mapping(enable_attention_dp=False, tp_size=tp_size)
    creator._model_engine = SimpleNamespace(model=SimpleNamespace(draft_model=draft_model))
    return creator


def _mla_draft_model(backend="CUTEDSL", dtype=torch.bfloat16, context_kv_fp8=False):
    """The fields MLADSparkForCausalLM publishes before its buffers are built."""
    return SimpleNamespace(
        _kv_factor=1,
        _num_attn_layers=5,
        _num_kv_heads=1,
        _head_dim=576,
        dflash_attention_backend=backend,
        fc=SimpleNamespace(weight=SimpleNamespace(dtype=dtype)),
        config=SimpleNamespace(max_position_embeddings=1048576),
        context_kv_fp8=context_kv_fp8,
    )


def _gqa_draft_model(backend="VANILLA"):
    return SimpleNamespace(
        dflash_attention_backend=backend,
        fc=SimpleNamespace(weight=SimpleNamespace(dtype=torch.bfloat16)),
        config=SimpleNamespace(max_position_embeddings=65536),
    )


@pytest.mark.parametrize("tp_size", [1, 8])
def test_mla_drafter_unified_kwargs_describe_one_unsharded_latent(tp_size):
    """One latent per token and layer; its page layout does not follow the TP size."""
    creator = _kwargs_creator(_mla_draft_model(), _mla_drafter(), tp_size=tp_size)

    kwargs = creator._get_unified_draft_cache_kwargs()

    assert kwargs["draft_num_layers"] == 5
    assert kwargs["draft_kv_factor"] == 1
    assert kwargs["draft_num_kv_heads"] == 1
    assert kwargs["draft_total_num_kv_heads"] == 1
    assert kwargs["draft_head_dim"] == 576
    assert kwargs["draft_dtype"] == torch_dtype_to_binding(torch.bfloat16)
    assert kwargs["draft_attention_backend"] == "CUTEDSL"
    assert kwargs["draft_scratch_tokens"] == 6
    assert kwargs["draft_max_position_embeddings"] == 1048576


def test_mla_drafter_pool_stays_bf16_under_the_trtllm_backend():
    """FP8 pages are for the GQA TRTLLM drafter; the MLA latent keeps the projection dtype."""
    creator = _kwargs_creator(
        _mla_draft_model(backend="TRTLLM"), _mla_drafter(), attention_backend="TRTLLM"
    )

    assert creator._get_unified_draft_cache_kwargs()["draft_dtype"] == torch_dtype_to_binding(
        torch.bfloat16
    )


def test_mla_drafter_fp8_context_kv_stores_the_same_latent_in_fp8():
    """speculative_config.context_kv_dtype="fp8" changes the storage dtype and nothing else."""
    bf16 = _kwargs_creator(_mla_draft_model(), _mla_drafter())._get_unified_draft_cache_kwargs()
    fp8 = _kwargs_creator(
        _mla_draft_model(context_kv_fp8=True), _mla_drafter()
    )._get_unified_draft_cache_kwargs()

    assert bf16.pop("draft_dtype") == torch_dtype_to_binding(torch.bfloat16)
    assert fp8.pop("draft_dtype") == torch_dtype_to_binding(torch.float8_e4m3fn)
    assert fp8 == bf16


def test_gqa_drafter_unified_kwargs_are_unchanged():
    creator = _kwargs_creator(_gqa_draft_model(), _gqa_drafter(), tp_size=4)

    kwargs = creator._get_unified_draft_cache_kwargs()

    assert kwargs["draft_num_layers"] == 5
    assert kwargs["draft_kv_factor"] == 2
    # 8 KV heads over TP 4.
    assert kwargs["draft_num_kv_heads"] == 2
    assert kwargs["draft_total_num_kv_heads"] == 8
    assert kwargs["draft_head_dim"] == 128
    assert kwargs["draft_dtype"] == torch_dtype_to_binding(torch.bfloat16)


def test_gqa_trtllm_drafter_pool_is_fp8():
    creator = _kwargs_creator(
        _gqa_draft_model(backend="TRTLLM"), _gqa_drafter(), attention_backend="TRTLLM"
    )

    assert creator._get_unified_draft_cache_kwargs()["draft_dtype"] == torch_dtype_to_binding(
        torch.float8_e4m3fn
    )


@pytest.mark.parametrize("setting", [None, "", "  "], ids=["unset", "empty", "blank"])
def test_context_limit_is_off_unless_set(monkeypatch, setting):
    if setting is not None:
        monkeypatch.setenv(_CONTEXT_LIMIT_ENV, setting)

    assert draft_context_limit_from_env() is None


@pytest.mark.parametrize(("setting", "limit"), [("98304", 98304), (" 122880 ", 122880), ("1", 1)])
def test_context_limit_is_a_token_count(monkeypatch, setting, limit):
    monkeypatch.setenv(_CONTEXT_LIMIT_ENV, setting)

    assert draft_context_limit_from_env() == limit


@pytest.mark.parametrize("setting", ["abc", "0", "-5", "1e5", "12.5"])
def test_context_limit_must_be_a_positive_integer(monkeypatch, setting):
    monkeypatch.setenv(_CONTEXT_LIMIT_ENV, setting)

    with pytest.raises(ValueError, match=_CONTEXT_LIMIT_ENV):
        draft_context_limit_from_env()


@pytest.mark.parametrize("drafter", [_gqa_drafter, _mla_drafter], ids=["gqa", "mla"])
def test_context_limit_is_accepted_by_a_disaggregated_server_with_the_unified_cache(
    monkeypatch, drafter
):
    monkeypatch.delenv(_DISABLE_ENV, raising=False)
    monkeypatch.setenv(_MLA_OPT_IN_ENV, "1")
    monkeypatch.setenv(_CONTEXT_LIMIT_ENV, "98304")

    with (
        mock.patch.object(_util.logger, "info") as info,
        mock.patch.object(_util.logger, "warning") as warning,
    ):
        _creator(drafter())._validate_standalone_draft_cache()

    # The effective limit is on record, with the worker it acts on.
    info.assert_called_once()
    message = info.call_args.args[0]
    assert "98304" in message
    assert "generation worker" in message
    warning.assert_not_called()


def _limit_creator(drafter, *, embedded=False, **options):
    creator = _creator(drafter(), **options)
    creator._speculative_config.draft_is_embedded_in_target = embedded
    return creator


@pytest.mark.parametrize(
    ("drafter", "options", "env"),
    [
        (_gqa_drafter, {"is_disagg": False}, {}),
        (_gqa_drafter, {"kv_cache_manager_v2": False}, {}),
        (_gqa_drafter, {}, {_DISABLE_ENV: "1"}),
        (_mla_drafter, {}, {}),
        (_gqa_drafter, {"embedded": True}, {}),
    ],
    ids=[
        "aggregate",
        "no_kv_cache_manager_v2",
        "legacy_state_env",
        "mla_without_opt_in",
        "embedded",
    ],
)
def test_context_limit_is_rejected_where_no_drafter_context_is_transferred(
    monkeypatch, drafter, options, env
):
    """A server that would silently ignore the limit fails at startup instead."""
    monkeypatch.delenv(_DISABLE_ENV, raising=False)
    monkeypatch.delenv(_MLA_OPT_IN_ENV, raising=False)
    monkeypatch.setenv(_CONTEXT_LIMIT_ENV, "98304")
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=_CONTEXT_LIMIT_ENV):
        _limit_creator(drafter, **options)._validate_standalone_draft_cache()


def test_an_invalid_context_limit_fails_startup(monkeypatch):
    monkeypatch.setenv(_CONTEXT_LIMIT_ENV, "0")

    with pytest.raises(ValueError, match=_CONTEXT_LIMIT_ENV):
        _creator(_gqa_drafter())._validate_standalone_draft_cache()


def test_startup_is_unchanged_without_a_context_limit(monkeypatch):
    monkeypatch.delenv(_DISABLE_ENV, raising=False)

    with mock.patch.object(_util.logger, "info") as info:
        _creator(_gqa_drafter())._validate_standalone_draft_cache()
        _creator(_gqa_drafter(), is_disagg=False)._validate_standalone_draft_cache()

    info.assert_not_called()


def _advertised_position_limit(monkeypatch, setting):
    """The draft position limit a context or generation worker builds its manager with."""
    if setting is None:
        monkeypatch.delenv(_CONTEXT_LIMIT_ENV, raising=False)
    else:
        monkeypatch.setenv(_CONTEXT_LIMIT_ENV, setting)
    creator = _kwargs_creator(_mla_draft_model(), _mla_drafter())
    return creator._get_unified_draft_cache_kwargs()["draft_max_position_embeddings"]


def test_context_limit_does_not_change_the_draft_position_limit(monkeypatch):
    """Both workers and the transfer check keep the position limit the drafter advertises."""
    assert _advertised_position_limit(monkeypatch, "98304") == 1048576
    assert _advertised_position_limit(monkeypatch, None) == 1048576


class _ContextManager:
    """The fields of a block-reusing KVCacheManagerV2 that its prefix lookup and commit read.

    The real methods run on it.
    """

    draft_layer_ids = (61,)
    draft_window_size = None
    enable_block_reuse = True
    _can_publish_block_reuse = True
    _has_cp_helix = False
    max_seq_len = 262144
    reuse_match_backoff = 0
    _draft_context_exceeds_limit = KVCacheManagerV2._draft_context_exceeds_limit
    _context_reuse_tokens = KVCacheManagerV2._context_reuse_tokens
    try_commit_blocks = KVCacheManagerV2.try_commit_blocks

    def __init__(self, draft_max_position_embeddings, request_id):
        self.draft_max_position_embeddings = draft_max_position_embeddings
        self.kv_cache = mock.Mock(num_committed_tokens=0)
        self.kv_cache_map = {request_id: self.kv_cache}

    def _reuse_token_source(self, req):
        return list(range(req.prompt_len))

    def _augment_tokens_for_block_reuse(self, tokens, req, start=None, end=None):
        return tokens[start:end]


def _context_request(prompt_len):
    return SimpleNamespace(
        py_request_id=7,
        prompt_len=prompt_len,
        is_dummy_request=False,
        context_current_position=prompt_len,
        context_remaining_length=0,
    )


@pytest.mark.parametrize("setting", [None, "98304"], ids=["no_limit", "limit_set"])
@pytest.mark.parametrize("prompt_len", [98304, 98305, 122881, 262144])
def test_context_worker_keeps_matching_and_committing_every_prompt(
    monkeypatch, setting, prompt_len
):
    """The prompts a generation worker drafts cold are the longest conversations, the ones a
    context worker saves the most prefill on: the limit must not take them out of its prefix
    cache."""
    manager = _ContextManager(_advertised_position_limit(monkeypatch, setting), request_id=7)
    request = _context_request(prompt_len)

    assert not manager._draft_context_exceeds_limit(request)
    # Everything but the last prompt token, which the target recomputes.
    assert len(manager._context_reuse_tokens(request)) == prompt_len - 1
    manager.try_commit_blocks(request)

    ((tokens,), _) = manager.kv_cache.commit.call_args
    assert len(tokens) == prompt_len
    manager.kv_cache.stop_committing.assert_called_once()


@pytest.mark.parametrize(("prompt_len", "skipped"), [(65536, False), (65537, True), (262144, True)])
def test_context_worker_skips_only_prompts_beyond_the_position_limit_the_drafter_advertises(
    prompt_len, skipped
):
    manager = _ContextManager(65536, request_id=7)
    request = _context_request(prompt_len)

    assert manager._draft_context_exceeds_limit(request) is skipped
    assert bool(manager._context_reuse_tokens(request)) is not skipped
    manager.try_commit_blocks(request)
    assert manager.kv_cache.commit.called is not skipped


_REQUEST_ID = 7
_LAYOUT = {"num_layers": 5, "window_size": None}


class _GenerationManager:
    """The fields of a draft-owning KVCacheManagerV2 that restore_draft_history reads.

    Its real draft-history bookkeeping runs on it.
    """

    draft_layer_ids = (61,)
    draft_attention_backend = "VANILLA"
    draft_window_size = None
    draft_max_position_embeddings = 1048576
    enable_block_reuse = False
    _draft_history_uses_kv_history = KVCacheManagerV2._draft_history_uses_kv_history
    get_draft_history = KVCacheManagerV2.get_draft_history
    set_draft_history = KVCacheManagerV2.set_draft_history
    restore_draft_history = KVCacheManagerV2.restore_draft_history
    take_draft_context_move = KVCacheManagerV2.take_draft_context_move

    def __init__(self, draft_context_limit, capacity=300000, draft_context_window=None):
        self.draft_context_limit = draft_context_limit
        self.draft_context_window = draft_context_window
        self._draft_context_moves = {}
        self.draft_history = {}
        self.kv_cache_map = {
            _REQUEST_ID: SimpleNamespace(is_active=True, capacity=capacity, history_length=0)
        }
        self.validated = []

    def get_draft_transfer_identity(self):
        return _LAYOUT

    def get_draft_block_table(self, request_ids, histories=None):
        # Receiver-local allocation is validated against the history about to be published.
        self.validated.append(histories)


def _restore_history(manager, prompt_len, valid_length=None):
    metadata = {
        "layout": _LAYOUT,
        "valid_length": prompt_len if valid_length is None else valid_length,
        "position": prompt_len,
    }
    manager.restore_draft_history(_REQUEST_ID, metadata)
    return manager.get_draft_history(_REQUEST_ID)


@pytest.mark.parametrize("prompt_len", [1, 98304, 98305, 262144])
def test_generation_keeps_the_transferred_context_without_a_limit(prompt_len):
    manager = _GenerationManager(draft_context_limit=None)
    history = StandaloneDraftHistory(prompt_len, prompt_len)

    assert _restore_history(manager, prompt_len) == history
    assert manager.validated == [[history]]


@pytest.mark.parametrize("prompt_len", [1, 98303, 98304])
def test_generation_keeps_the_context_of_a_prompt_up_to_the_limit(prompt_len):
    manager = _GenerationManager(draft_context_limit=98304)

    assert _restore_history(manager, prompt_len) == StandaloneDraftHistory(prompt_len, prompt_len)


@pytest.mark.parametrize("prompt_len", [98305, 122880, 262144])
def test_generation_drafts_a_prompt_above_the_limit_from_an_empty_context(prompt_len):
    manager = _GenerationManager(draft_context_limit=98304)

    history = _restore_history(manager, prompt_len)

    # Nothing is drafted from the transferred pages, but the position still counts the prompt: the
    # draft pages are allocated for it and the history published later follows the sequence.
    assert history == StandaloneDraftHistory(0, prompt_len)
    assert manager.validated == [[history]]


def test_generation_gates_on_the_prompt_length_not_on_the_retained_context():
    manager = _GenerationManager(draft_context_limit=98304)

    # The limit is about the prompt the drafter would attend to, not about how much of its context
    # the context worker retained.
    history = _restore_history(manager, 122880, valid_length=4096)

    assert history == StandaloneDraftHistory(0, 122880)


def test_the_first_gated_prompt_is_logged():
    manager = _GenerationManager(draft_context_limit=98304)

    with mock.patch.object(kv_cache_manager_v2.logger, "info_once") as info_once:
        _restore_history(manager, 98304)
        info_once.assert_not_called()
        _restore_history(manager, 98305)

    info_once.assert_called_once()
    message = info_once.call_args.args[0]
    assert "98304" in message
    assert "98305-token" in message


def test_a_gated_history_is_published_like_an_ungated_one():
    manager = _GenerationManager(draft_context_limit=98304, capacity=98305 + 16)
    _restore_history(manager, 98305)

    # What the worker publishes after a decode step that accepted six tokens.
    manager.set_draft_history(_REQUEST_ID, 6, 98305 + 6)

    assert manager.get_draft_history(_REQUEST_ID) == StandaloneDraftHistory(6, 98311)


def test_a_rolling_drafter_is_not_gated():
    manager = _GenerationManager(draft_context_limit=98304)
    manager.draft_window_size = 4096

    assert _restore_history(manager, 200000, valid_length=4096) == StandaloneDraftHistory(
        4096, 200000
    )


def test_an_embedded_drafter_history_is_not_gated():
    manager = _GenerationManager(draft_context_limit=98304)
    manager.draft_attention_backend = "DSv4"
    manager.draft_window_size = 4096
    manager.kv_cache_map[_REQUEST_ID].history_length = 200000

    assert _restore_history(manager, 200000, valid_length=4096) is None
    assert manager.validated == [None]


def test_the_cold_start_overwrites_the_received_pages_so_they_must_be_private():
    manager = _GenerationManager(draft_context_limit=98304)
    manager.enable_block_reuse = True

    with pytest.raises(ValueError, match=_CONTEXT_LIMIT_ENV):
        _restore_history(manager, 98305)

    assert manager.draft_history == {}
    # A prompt that keeps its transferred context is not written over.
    assert _restore_history(manager, 98304) == StandaloneDraftHistory(98304, 98304)


class _Receiver:
    """The transceiver methods that check and hand over the transferred draft history."""

    _restore_draft_history = KvCacheTransceiverV2._restore_draft_history
    _validate_draft_history_range = KvCacheTransceiverV2._validate_draft_history_range
    _global_prompt_len = KvCacheTransceiverV2._global_prompt_len

    def __init__(self, manager):
        self._kv_cache_manager = manager


@pytest.mark.parametrize(
    ("limit", "expected"),
    [(None, StandaloneDraftHistory(122880, 122880)), (98304, StandaloneDraftHistory(0, 122880))],
    ids=["no_limit", "limit_set"],
)
def test_the_limit_on_one_worker_cannot_fail_the_transfer(limit, expected):
    """The context worker sends the history of the whole prompt whatever the limit is, and the
    generation worker's transfer check expects it: only what it binds differs."""
    request = SimpleNamespace(
        py_request_id=_REQUEST_ID,
        prompt_len=122880,
        py_draft_transfer_history={"layout": _LAYOUT, "valid_length": 122880, "position": 122880},
    )
    manager = _GenerationManager(draft_context_limit=limit)

    _Receiver(manager)._restore_draft_history(request)

    assert manager.get_draft_history(_REQUEST_ID) == expected


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


def _context_server_creator(draft_config, **kwargs):
    """A creator for a context server whose drafter only builds the context KV."""
    creator = _creator(draft_config, **kwargs)
    creator._speculative_config.context_kv_only = True
    return creator


def test_context_kv_only_is_accepted_with_the_unified_cache_of_an_mla_drafter(monkeypatch):
    monkeypatch.delenv(_DISABLE_ENV, raising=False)
    monkeypatch.setenv(_MLA_OPT_IN_ENV, "1")

    _context_server_creator(_mla_drafter())._validate_standalone_draft_cache()


def test_context_kv_only_without_the_mla_opt_in_is_rejected(monkeypatch):
    monkeypatch.delenv(_DISABLE_ENV, raising=False)
    monkeypatch.delenv(_MLA_OPT_IN_ENV, raising=False)

    with pytest.raises(ValueError, match=_MLA_OPT_IN_ENV):
        _context_server_creator(_mla_drafter())._validate_standalone_draft_cache()


def test_context_kv_only_with_the_legacy_state_env_is_rejected(monkeypatch):
    monkeypatch.setenv(_MLA_OPT_IN_ENV, "1")
    monkeypatch.setenv(_DISABLE_ENV, "1")

    with pytest.raises(ValueError, match="context_kv_only"):
        _context_server_creator(_mla_drafter())._validate_standalone_draft_cache()


@pytest.mark.parametrize("setting", [{"is_disagg": False}, {"kv_cache_manager_v2": False}])
def test_context_kv_only_needs_disaggregation_and_the_kv_cache_manager_v2(monkeypatch, setting):
    monkeypatch.delenv(_DISABLE_ENV, raising=False)
    monkeypatch.setenv(_MLA_OPT_IN_ENV, "1")

    with pytest.raises(ValueError, match="context_kv_only"):
        _context_server_creator(_mla_drafter(), **setting)._validate_standalone_draft_cache()


@pytest.mark.parametrize(
    "role,accepted",
    [(None, True), ("", True), ("context", True), ("generation", False)],
    ids=["unset", "empty", "context", "generation"],
)
def test_context_kv_only_is_for_context_servers(monkeypatch, role, accepted):
    """A launcher that names the server's role must not give a generation server this drafter."""
    monkeypatch.delenv(_DISABLE_ENV, raising=False)
    monkeypatch.setenv(_MLA_OPT_IN_ENV, "1")
    if role is not None:
        monkeypatch.setenv(_ROLE_ENV, role)
    creator = _context_server_creator(_mla_drafter())

    if accepted:
        creator._validate_standalone_draft_cache()
    else:
        with pytest.raises(ValueError, match="role is 'generation'"):
            creator._validate_standalone_draft_cache()


def test_a_drafter_without_context_kv_only_is_validated_as_before(monkeypatch):
    monkeypatch.delenv(_DISABLE_ENV, raising=False)
    monkeypatch.setenv(_MLA_OPT_IN_ENV, "1")
    monkeypatch.setenv(_ROLE_ENV, "generation")
    creator = _creator(_mla_drafter(), is_disagg=False)
    creator._speculative_config.context_kv_only = False

    creator._validate_standalone_draft_cache()


def _fp8_context_creator(draft_config, **kwargs):
    """A creator whose drafter stores its context KV as FP8."""
    creator = _creator(draft_config, **kwargs)
    creator._speculative_config.context_kv_dtype = "fp8"
    return creator


@pytest.mark.parametrize("is_disagg", [True, False], ids=["disagg", "aggregate"])
def test_fp8_context_kv_is_accepted_with_the_unified_cache_of_an_mla_drafter(
    monkeypatch, is_disagg
):
    monkeypatch.delenv(_DISABLE_ENV, raising=False)
    monkeypatch.setenv(_MLA_OPT_IN_ENV, "1")

    _fp8_context_creator(_mla_drafter(), is_disagg=is_disagg)._validate_standalone_draft_cache()


def test_fp8_context_kv_without_the_mla_opt_in_is_rejected(monkeypatch):
    """Without the opt-in the drafter keeps the legacy state, which has no draft pages."""
    monkeypatch.delenv(_DISABLE_ENV, raising=False)
    monkeypatch.delenv(_MLA_OPT_IN_ENV, raising=False)

    with pytest.raises(ValueError, match="context_kv_dtype"):
        _fp8_context_creator(_mla_drafter())._validate_standalone_draft_cache()


def test_fp8_context_kv_with_the_legacy_state_env_is_rejected(monkeypatch):
    monkeypatch.setenv(_MLA_OPT_IN_ENV, "1")
    monkeypatch.setenv(_DISABLE_ENV, "1")

    with pytest.raises(ValueError, match="context_kv_dtype"):
        _fp8_context_creator(_mla_drafter())._validate_standalone_draft_cache()


def test_fp8_context_kv_needs_the_kv_cache_manager_v2(monkeypatch):
    monkeypatch.delenv(_DISABLE_ENV, raising=False)
    monkeypatch.setenv(_MLA_OPT_IN_ENV, "1")

    with pytest.raises(ValueError, match="context_kv_dtype"):
        _fp8_context_creator(
            _mla_drafter(), kv_cache_manager_v2=False
        )._validate_standalone_draft_cache()


@pytest.mark.parametrize("opted_in", [True, False], ids=["unified", "legacy"])
def test_the_default_context_kv_dtype_is_validated_as_before(monkeypatch, opted_in):
    monkeypatch.delenv(_DISABLE_ENV, raising=False)
    if opted_in:
        monkeypatch.setenv(_MLA_OPT_IN_ENV, "1")
    else:
        monkeypatch.delenv(_MLA_OPT_IN_ENV, raising=False)
    creator = _creator(_mla_drafter())
    creator._speculative_config.context_kv_dtype = "auto"

    creator._validate_standalone_draft_cache()


# ---------------------------------------------------------------------------
# TLLM_DSPARK_DRAFT_CONTEXT_WINDOW / _SINK: what a prompt above the limit drafts from.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("setting", [None, "", "  "], ids=["unset", "empty", "blank"])
def test_context_window_is_off_unless_set(monkeypatch, setting):
    if setting is not None:
        monkeypatch.setenv(_CONTEXT_WINDOW_ENV, setting)

    assert draft_context_window_from_env() is None


@pytest.mark.parametrize(
    ("window", "sink", "expected"),
    [
        ("65536", None, DraftContextWindow(sink=0, recent=65536)),
        (" 65536 ", "64", DraftContextWindow(sink=64, recent=65536)),
        ("1", "0", DraftContextWindow(sink=0, recent=1)),
    ],
)
def test_context_window_is_a_token_count(monkeypatch, window, sink, expected):
    monkeypatch.setenv(_CONTEXT_WINDOW_ENV, window)
    if sink is not None:
        monkeypatch.setenv(_CONTEXT_SINK_ENV, sink)

    assert draft_context_window_from_env() == expected
    assert expected.length == expected.sink + expected.recent


@pytest.mark.parametrize(
    ("window", "sink", "named"),
    [
        ("0", None, _CONTEXT_WINDOW_ENV),
        ("-4", None, _CONTEXT_WINDOW_ENV),
        ("6e4", None, _CONTEXT_WINDOW_ENV),
        ("abc", None, _CONTEXT_WINDOW_ENV),
        ("65536", "-1", _CONTEXT_SINK_ENV),
        ("65536", "1.5", _CONTEXT_SINK_ENV),
        (None, "64", _CONTEXT_SINK_ENV),
    ],
    ids=["zero", "negative", "float", "text", "negative_sink", "float_sink", "sink_alone"],
)
def test_context_window_settings_must_be_token_counts(monkeypatch, window, sink, named):
    if window is not None:
        monkeypatch.setenv(_CONTEXT_WINDOW_ENV, window)
    if sink is not None:
        monkeypatch.setenv(_CONTEXT_SINK_ENV, sink)

    with pytest.raises(ValueError, match=named):
        draft_context_window_from_env()


def _window_env(monkeypatch, *, limit="98304", window="65536", sink=None):
    monkeypatch.delenv(_DISABLE_ENV, raising=False)
    monkeypatch.setenv(_MLA_OPT_IN_ENV, "1")
    if limit is not None:
        monkeypatch.setenv(_CONTEXT_LIMIT_ENV, limit)
    monkeypatch.setenv(_CONTEXT_WINDOW_ENV, window)
    if sink is not None:
        monkeypatch.setenv(_CONTEXT_SINK_ENV, sink)


@pytest.mark.parametrize(("window", "sink"), [("65536", None), ("98240", "64"), ("98304", None)])
def test_context_window_is_accepted_with_the_limit_on_an_mla_generation_server(
    monkeypatch, window, sink
):
    _window_env(monkeypatch, window=window, sink=sink)

    with mock.patch.object(_util.logger, "info") as info:
        _creator(_mla_drafter())._validate_standalone_draft_cache()

    # One record of what a generation worker drafts a prompt above the limit from, in the form of
    # the limit's own startup record.
    info.assert_called_once()
    message = info.call_args.args[0]
    assert message.startswith(
        "TLLM_DSPARK_DRAFT_CONTEXT_LIMIT=98304: a generation worker drafts prompts longer than"
    )
    assert f"first {sink or 0} and last {window} tokens" in message


def test_context_window_needs_the_limit(monkeypatch):
    _window_env(monkeypatch, limit=None)

    with pytest.raises(ValueError, match=_CONTEXT_LIMIT_ENV):
        _creator(_mla_drafter())._validate_standalone_draft_cache()


def test_context_window_must_fit_below_the_limit(monkeypatch):
    # The window of the shortest prompt above the limit must not overlap its sink.
    _window_env(monkeypatch, window="98304", sink="1")

    with pytest.raises(ValueError, match=_CONTEXT_WINDOW_ENV):
        _creator(_mla_drafter())._validate_standalone_draft_cache()


@pytest.mark.parametrize(
    ("drafter", "env"),
    [(_gqa_drafter, {}), (_mla_drafter, {_MLA_OPT_IN_ENV: "0"})],
    ids=["gqa", "mla_without_opt_in"],
)
def test_context_window_needs_an_mla_drafter_with_the_unified_cache(monkeypatch, drafter, env):
    """The window rebases the RoPE slice of MLA latents; anything else fails at startup."""
    _window_env(monkeypatch)
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=_CONTEXT_WINDOW_ENV):
        _creator(drafter())._validate_standalone_draft_cache()


def _window_manager(window=65536, sink=0, limit=98304, **kwargs):
    return _GenerationManager(
        draft_context_limit=limit,
        draft_context_window=DraftContextWindow(sink=sink, recent=window),
        **kwargs,
    )


@pytest.mark.parametrize("prompt_len", [1, 65536, 98304])
def test_window_keeps_the_whole_context_of_a_prompt_up_to_the_limit(prompt_len):
    manager = _window_manager()

    assert _restore_history(manager, prompt_len) == StandaloneDraftHistory(prompt_len, prompt_len)
    assert manager.take_draft_context_move(_REQUEST_ID) is None


@pytest.mark.parametrize("sink", [0, 64])
@pytest.mark.parametrize("prompt_len", [98305, 131072, 262144])
def test_window_binds_the_sink_and_the_last_tokens_of_a_prompt_above_the_limit(prompt_len, sink):
    manager = _window_manager(sink=sink)

    history = _restore_history(manager, prompt_len)

    # The bound context is the sink plus the window; the position still counts the prompt.
    assert history == StandaloneDraftHistory(sink + 65536, prompt_len)
    assert manager.validated == [[history]]
    # The worker moves the last 65536 latents to right after the sink, once.
    move = manager.take_draft_context_move(_REQUEST_ID)
    assert move == DraftContextWindowMove(
        source_start=prompt_len - 65536, dest_start=sink, length=65536
    )
    assert move.shift == prompt_len - 65536 - sink
    assert manager.take_draft_context_move(_REQUEST_ID) is None


def test_window_of_an_incomplete_transferred_context_starts_empty():
    manager = _window_manager()

    # Only a context that covers the whole prompt has a known last window.
    history = _restore_history(manager, 122880, valid_length=4096)

    assert history == StandaloneDraftHistory(0, 122880)
    assert manager.take_draft_context_move(_REQUEST_ID) is None


def test_the_first_windowed_prompt_is_logged():
    manager = _window_manager(sink=64)

    with mock.patch.object(kv_cache_manager_v2.logger, "info_once") as info_once:
        _restore_history(manager, 98304)
        info_once.assert_not_called()
        _restore_history(manager, 131072)

    info_once.assert_called_once()
    message = info_once.call_args.args[0]
    # The limit's first-hit record, refined.
    assert message.startswith(f"TLLM_DSPARK_DRAFT_CONTEXT_LIMIT=98304: request {_REQUEST_ID} has a")
    assert "131072-token" in message
    assert "first 64 and last 65536 tokens" in message


def test_a_windowed_history_is_published_like_an_ungated_one():
    manager = _window_manager(capacity=131072 + 16)
    _restore_history(manager, 131072)

    # What the worker publishes after a decode step that accepted six tokens.
    manager.set_draft_history(_REQUEST_ID, 65536 + 6, 131072 + 6)

    assert manager.get_draft_history(_REQUEST_ID) == StandaloneDraftHistory(65542, 131078)


def test_the_window_move_rewrites_the_received_pages_so_they_must_be_private():
    manager = _window_manager()
    manager.enable_block_reuse = True

    with pytest.raises(ValueError, match=_CONTEXT_WINDOW_ENV):
        _restore_history(manager, 98305)

    assert manager.draft_history == {} and manager.take_draft_context_move(_REQUEST_ID) is None


@pytest.mark.parametrize(
    ("source_start", "dest_start", "length"),
    [(0, 1, 4), (5, 0, 0), (5, -1, 4), (5.0, 0, 4)],
    ids=["backwards", "empty", "negative", "float"],
)
def test_a_window_move_goes_towards_the_start_of_the_context(source_start, dest_start, length):
    with pytest.raises(ValueError, match="window move"):
        DraftContextWindowMove(source_start=source_start, dest_start=dest_start, length=length)


@pytest.mark.parametrize(
    ("limit", "window", "expected"),
    [
        (None, None, StandaloneDraftHistory(122880, 122880)),
        (98304, None, StandaloneDraftHistory(0, 122880)),
        (98304, 65536, StandaloneDraftHistory(65536, 122880)),
    ],
    ids=["no_limit", "limit_set", "window_set"],
)
def test_the_window_on_one_worker_cannot_fail_the_transfer(limit, window, expected):
    """The context worker sends the history of the whole prompt whatever the generation worker
    keeps of it."""
    request = SimpleNamespace(
        py_request_id=_REQUEST_ID,
        prompt_len=122880,
        py_draft_transfer_history={"layout": _LAYOUT, "valid_length": 122880, "position": 122880},
    )
    manager = _GenerationManager(
        draft_context_limit=limit,
        draft_context_window=None if window is None else DraftContextWindow(0, window),
    )

    _Receiver(manager)._restore_draft_history(request)

    assert manager.get_draft_history(_REQUEST_ID) == expected
