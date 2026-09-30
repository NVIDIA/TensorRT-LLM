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
    NGramDecodingConfig,
    TorchLlmArgs,
)
from tensorrt_llm.llmapi.llm_args import disagg_dflash_error

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
    """The guard wired into TorchLlmArgs.validate_speculative_config."""

    # Keeps the DFlash token-budget rule satisfied so only the disagg guard
    # is under test: max_batch_size * (1 + max_draft_len) <= max_num_tokens.
    _SHAPE = dict(max_batch_size=8, max_num_tokens=8192, max_seq_len=8192)

    def test_disagg_gen_side_dflash_config_is_refused(self):
        # Both disagg sides carry the same cache_transceiver_config (the
        # gen-side yaml of a ctx/gen pair looks exactly like this), so this
        # is the generation-side shape the guard exists for -- and the
        # context-side shape, refused by design (see disagg_dflash_error).
        with pytest.raises(ValueError, match="DFlash.*disaggregated"):
            TorchLlmArgs(
                model="dummy",
                speculative_config=_dflash_config(),
                cache_transceiver_config=_transceiver_config(),
                **self._SHAPE,
            )

    def test_aggregated_dflash_config_is_allowed(self):
        args = TorchLlmArgs(model="dummy", speculative_config=_dflash_config(), **self._SHAPE)
        assert args.cache_transceiver_config is None

    def test_disagg_without_speculation_is_allowed(self):
        args = TorchLlmArgs(
            model="dummy", cache_transceiver_config=_transceiver_config(), **self._SHAPE
        )
        assert args.speculative_config is None
