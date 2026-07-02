# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-safe tests for the vanilla MTP rejection-sampling config guard.

These exercise ``TorchLlmArgs.validate_speculative_config`` directly (no GPU,
no model load) to verify:
  - vanilla MTP may enable rejection sampling (classified by spec-dec mode),
  - Eagle3 is unchanged,
  - unsupported combinations (SA, relaxed-thinking acceptance, TP/CP/ADP,
    guided decoding) disable it on default and raise on explicit opt-in,
  - the default stays False.
"""

import pytest

import tensorrt_llm._torch.flashinfer_utils as flashinfer_utils
from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode
from tensorrt_llm.llmapi.llm_args import (
    Eagle3DecodingConfig,
    MTPDecodingConfig,
    PARDDecodingConfig,
    SAEnhancerConfig,
    TorchLlmArgs,
)

MODEL = "dummy/model-path"


@pytest.fixture(autouse=True)
def _force_flashinfer_available(monkeypatch):
    # The rejection-sampling guard fails fast when FlashInfer is unavailable.
    # Force it available so these config tests are CPU-safe and independent of
    # whether FlashInfer is installed in the test environment. The dedicated
    # FlashInfer-absent test overrides this.
    monkeypatch.setattr(flashinfer_utils, "IS_FLASHINFER_AVAILABLE", True)


def _make_args(spec_config, **kwargs):
    # model is required but never loaded for config validation.
    return TorchLlmArgs(model=MODEL, speculative_config=spec_config, **kwargs)


def test_vanilla_mtp_rejection_allowed():
    spec = MTPDecodingConfig(use_mtp_vanilla=True, use_rejection_sampling=True)
    assert spec.spec_dec_mode == SpeculativeDecodingMode.MTP
    args = _make_args(spec)
    assert args.speculative_config.use_rejection_sampling is True


def test_eagle3_rejection_still_allowed():
    # Eagle3DecodingConfig requires max_draft_len > 0 and a draft model dir
    # (never loaded here -- config validation only).
    spec = Eagle3DecodingConfig(
        max_draft_len=3, speculative_model=MODEL, use_rejection_sampling=True
    )
    args = _make_args(spec)
    assert args.speculative_config.use_rejection_sampling is True


def test_flashinfer_absent_raises_on_explicit_mtp(monkeypatch):
    monkeypatch.setattr(flashinfer_utils, "IS_FLASHINFER_AVAILABLE", False)
    spec = MTPDecodingConfig(use_mtp_vanilla=True, use_rejection_sampling=True)
    with pytest.raises(ValueError):
        _make_args(spec)


def test_mtp_rejection_default_stays_false():
    spec = MTPDecodingConfig(use_mtp_vanilla=True)
    args = _make_args(spec)
    assert args.speculative_config.use_rejection_sampling is False


def test_mtp_eagle_one_model_not_reclassified_as_vanilla():
    # MTP-Eagle one-model shares MTPDecodingConfig but is NOT the vanilla MTP
    # mode; explicit rejection opt-in must raise (it is not on the supported
    # vanilla-MTP path).
    spec = MTPDecodingConfig(
        use_mtp_vanilla=False, mtp_eagle_one_model=True, use_rejection_sampling=True
    )
    assert spec.spec_dec_mode == SpeculativeDecodingMode.MTP_EAGLE_ONE_MODEL
    with pytest.raises(ValueError):
        _make_args(spec)


def test_mtp_relaxed_acceptance_explicit_raises():
    spec = MTPDecodingConfig(
        use_mtp_vanilla=True, use_relaxed_acceptance_for_thinking=True, use_rejection_sampling=True
    )
    with pytest.raises(ValueError):
        _make_args(spec)


def test_mtp_sa_rejection_explicit_raises():
    # Explicit opt-in with SA must raise, not be silently pre-cleared by the
    # base-config validator before the centralized gate runs.
    spec = MTPDecodingConfig(
        use_mtp_vanilla=True, sa_config=SAEnhancerConfig(), use_rejection_sampling=True
    )
    with pytest.raises(ValueError):
        _make_args(spec)


def test_mtp_sa_rejection_default_silently_disabled():
    # Not an explicit opt-in: SA present + default rejection -> silently disabled.
    spec = MTPDecodingConfig(use_mtp_vanilla=True, sa_config=SAEnhancerConfig())
    args = _make_args(spec)
    assert args.speculative_config.use_rejection_sampling is False


def test_mtp_tp_explicit_raises():
    spec = MTPDecodingConfig(use_mtp_vanilla=True, use_rejection_sampling=True)
    with pytest.raises(ValueError):
        _make_args(spec, tensor_parallel_size=2)


def test_mtp_guided_decoding_explicit_raises():
    spec = MTPDecodingConfig(use_mtp_vanilla=True, use_rejection_sampling=True)
    with pytest.raises(ValueError):
        _make_args(spec, guided_decoding_backend="xgrammar")


def test_mtp_tp_default_silently_disabled():
    # Not an explicit opt-in (value omitted): TP-active vanilla MTP silently
    # disables rejection instead of raising.
    spec = MTPDecodingConfig(use_mtp_vanilla=True)
    args = _make_args(spec, tensor_parallel_size=2)
    assert args.speculative_config.use_rejection_sampling is False


def test_pard_rejection_allowed():
    # PARD is a newly enabled one-model rejection path; allowed without SA etc.
    spec = PARDDecodingConfig(max_draft_len=4, speculative_model=MODEL, use_rejection_sampling=True)
    assert spec.spec_dec_mode == SpeculativeDecodingMode.PARD
    args = _make_args(spec)
    assert args.speculative_config.use_rejection_sampling is True


def test_pard_sa_rejection_explicit_raises():
    # PARD + SA: SA overrides emitted draft tokens after their probs are stored,
    # violating the proposal-distribution invariant -> explicit opt-in raises.
    spec = PARDDecodingConfig(
        max_draft_len=4,
        speculative_model=MODEL,
        sa_config=SAEnhancerConfig(),
        use_rejection_sampling=True,
    )
    with pytest.raises(ValueError):
        _make_args(spec)


def test_pard_guided_decoding_explicit_raises():
    spec = PARDDecodingConfig(max_draft_len=4, speculative_model=MODEL, use_rejection_sampling=True)
    with pytest.raises(ValueError):
        _make_args(spec, guided_decoding_backend="xgrammar")


def test_pard_tp_default_silently_disabled():
    # Not an explicit opt-in: TP-active PARD silently disables rejection.
    spec = PARDDecodingConfig(max_draft_len=4, speculative_model=MODEL)
    args = _make_args(spec, tensor_parallel_size=2)
    assert args.speculative_config.use_rejection_sampling is False


def test_pard_sa_rejection_default_silently_disabled():
    # Symmetry with the explicit PARD+SA raise: SA present + default (non-explicit)
    # rejection is silently disabled rather than raising.
    spec = PARDDecodingConfig(
        max_draft_len=4, speculative_model=MODEL, sa_config=SAEnhancerConfig()
    )
    args = _make_args(spec)
    assert args.speculative_config.use_rejection_sampling is False


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
