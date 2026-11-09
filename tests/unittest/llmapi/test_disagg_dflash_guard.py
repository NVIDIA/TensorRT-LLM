# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DFlash speculation on a disaggregated engine: the config-time contract.

DFlash is no longer refused on a disaggregated engine. The drafter's prompt
window (its context K/V, written by the context worker's prefill) reaches the
generation worker when the draft KV shares the target's KV cache manager: the
draft layers then travel with the target's in the transceiver's session
(``_torch/disaggregation/draft_kv.py``). That sharing is the default for the
AUTO and TRTLLM drafter backends (``disagg_shares_draft_kv``);
``TRTLLM_DISAGG_SHARED_DRAFT_KV=0`` or a VANILLA drafter keeps the draft KV
private, and validation then only warns that acceptance is degraded, since the
target still verifies every draft.

The old ``disagg_dflash_error`` refusal (the drafter's context was worker-local,
so every generation request fell back to one shared dummy slot) is gone; the
tests here pin its replacement.
"""

import pytest

from tensorrt_llm.llmapi import (
    CacheTransceiverConfig,
    DFlashDecodingConfig,
    DSparkDecodingConfig,
    TorchLlmArgs,
    llm_args,
)
from tensorrt_llm.llmapi.llm_args import disagg_shares_draft_kv

pytestmark = pytest.mark.cpu_only


def _dflash_config():
    return DFlashDecodingConfig(max_draft_len=7)


def _transceiver_config():
    return CacheTransceiverConfig(backend="NIXL")


def test_the_old_refusal_is_gone():
    """DFlash on a disaggregated engine is validated, not refused, so the refusal helper is gone."""
    assert not hasattr(llm_args, "disagg_dflash_error")


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
        degraded = [str(w) for w in warnings if "DFlash acceptance is degraded" in str(w)]
        assert bool(degraded) == warns
        for message in degraded:
            # The warning names the fix, not a benchmarking overlay.
            assert "TRTLLM_DISAGG_SHARED_DRAFT_KV" in message
            assert "overlay" not in message

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
