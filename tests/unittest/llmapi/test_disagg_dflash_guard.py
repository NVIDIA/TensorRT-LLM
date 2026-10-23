# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The config-time guard refusing DFlash speculation on a disaggregated engine.

The DFlash drafter's pooled-context K/V is filled only by a local prefill
forward, in worker-local buffers the KV-cache transceiver never transfers. On
a disaggregated generation engine no request ever runs a context forward, so
every request falls back to the drafter's single shared dummy slot and
acceptance length silently collapses while the drafter overhead is still paid.
Because the ctx/gen role is decided per request, not per config, the guard
refuses DFlash on ANY engine configured with a cache transceiver backend; a
context-side engine loses nothing, since it never runs generation steps.
"""

import pytest

from tensorrt_llm.llmapi import (
    CacheTransceiverConfig,
    DFlashDecodingConfig,
    DSparkDecodingConfig,
    NGramDecodingConfig,
    TorchLlmArgs,
    llm_args,
)
from tensorrt_llm.llmapi.llm_args import disagg_dflash_error, disagg_shares_draft_kv

pytestmark = pytest.mark.cpu_only


def _dflash_config():
    return DFlashDecodingConfig(max_draft_len=7)


def _transceiver_config():
    return CacheTransceiverConfig(backend="NIXL")


class TestDisaggDFlashError:
    """The message-producing predicate, over the four config quadrants."""

    def test_disagg_plus_dflash_is_refused(self):
        message = disagg_dflash_error(_dflash_config(), _transceiver_config())
        assert message is not None
        # The message must name both settings and both workarounds.
        assert "DFlash" in message
        assert "cache_transceiver_config" in message
        assert "without speculation" in message
        assert "aggregated" in message

    def test_aggregated_dflash_is_allowed(self):
        assert disagg_dflash_error(_dflash_config(), None) is None

    def test_disagg_without_speculation_is_allowed(self):
        assert disagg_dflash_error(None, _transceiver_config()) is None

    def test_disagg_with_non_dflash_speculation_is_allowed(self):
        ngram = NGramDecodingConfig(max_draft_len=4)
        assert disagg_dflash_error(ngram, _transceiver_config()) is None

    def test_transceiver_without_backend_is_not_disagg(self):
        # Matches the existing is_disagg idiom: a CacheTransceiverConfig with
        # no backend selected does not configure disaggregation.
        config = CacheTransceiverConfig(backend=None)
        assert disagg_dflash_error(_dflash_config(), config) is None


class TestTorchLlmArgsValidation:
    """What TorchLlmArgs.validate_speculative_config does with DFlash on a disaggregated engine."""

    # Keeps the DFlash token-budget rule satisfied so only the disagg guard
    # is under test: max_batch_size * (1 + max_draft_len) <= max_num_tokens.
    _SHAPE = dict(max_batch_size=8, max_num_tokens=8192, max_seq_len=8192)

    @pytest.mark.parametrize(
        "attention_backend,shared_draft_kv,warns",
        [
            ("VANILLA", None, True),
            ("TRTLLM", None, False),
            ("TRTLLM", "1", False),
            ("TRTLLM", "0", True),
            ("AUTO", None, False),
            ("AUTO", "1", False),
            ("AUTO", "0", True),
        ],
    )
    def test_disagg_dflash_warns_unless_the_draft_kv_is_shared(
        self, monkeypatch, attention_backend, shared_draft_kv, warns
    ):
        # The drafter's context reaches a generation engine only through a draft
        # KV shared with the target: the TRTLLM backend, unless
        # TRTLLM_DISAGG_SHARED_DRAFT_KV=0 keeps it private.
        warnings = []
        monkeypatch.setattr(llm_args.logger, "warning", lambda msg, *a, **k: warnings.append(msg))
        if shared_draft_kv is None:
            monkeypatch.delenv("TRTLLM_DISAGG_SHARED_DRAFT_KV", raising=False)
        else:
            monkeypatch.setenv("TRTLLM_DISAGG_SHARED_DRAFT_KV", shared_draft_kv)
        TorchLlmArgs(
            model="dummy",
            speculative_config=DFlashDecodingConfig(
                max_draft_len=7, attention_backend=attention_backend
            ),
            cache_transceiver_config=_transceiver_config(),
            **self._SHAPE,
        )
        assert any("DFlash acceptance is degraded" in str(w) for w in warnings) == warns

    def test_aggregated_dflash_config_is_allowed(self):
        args = TorchLlmArgs(model="dummy", speculative_config=_dflash_config(), **self._SHAPE)
        assert args.cache_transceiver_config is None

    def test_disagg_without_speculation_is_allowed(self):
        args = TorchLlmArgs(
            model="dummy", cache_transceiver_config=_transceiver_config(), **self._SHAPE
        )
        assert args.speculative_config is None


@pytest.mark.parametrize(
    ("decoding_type", "attention_backend", "env", "shares"),
    [
        ("DFlash", "AUTO", None, True),
        ("DFlash", "TRTLLM", None, True),
        ("DFlash", "VANILLA", None, False),
        ("DFlash", "AUTO", "0", False),
        ("DFlash", "VANILLA", "1", True),
        ("DSpark", "AUTO", None, False),
        ("DSpark", "AUTO", "1", True),
    ],
)
def test_disagg_shares_draft_kv(monkeypatch, decoding_type, attention_backend, env, shares):
    """Unset, the variable leaves the choice to a DFlash backend that can share; DSpark needs it."""
    if env is None:
        monkeypatch.delenv("TRTLLM_DISAGG_SHARED_DRAFT_KV", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_DISAGG_SHARED_DRAFT_KV", env)
    config_cls = DFlashDecodingConfig if decoding_type == "DFlash" else DSparkDecodingConfig
    config = config_cls(max_draft_len=7, attention_backend=attention_backend)
    assert disagg_shares_draft_kv(config) is shares
