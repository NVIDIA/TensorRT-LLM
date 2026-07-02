# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-safe proof: rejection gates fire before any partial state is set up.

`TorchLlmArgs.validate_speculative_config` runs at args construction, before any
executor / model / worker setup. These tests instrument the rejection setup and
sampling entrypoints (`SpecMetadata.prepare_rejection_sampling_buffers`,
`SpecWorkerBase.sample_draft` / `sample_draft_block` /
`_draft_sampler_advanced_for_rejection`) so they raise if ever reached, then
verify that explicit unsupported MTP rejection opt-ins (FlashInfer-absent / TP /
CP / ADP / guided) raise `ValueError` at construction WITHOUT touching any of
those entrypoints, that default/non-explicit TP silently disables rejection, and
that a FlashInfer-available single-GPU config still keeps rejection enabled.
"""

import pytest

import tensorrt_llm._torch.flashinfer_utils as flashinfer_utils
import tensorrt_llm._torch.speculative.interface as iface
from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig, TorchLlmArgs

MODEL = "dummy/model-path"


class _ReachedSetupError(AssertionError):
    """Raised if a rejection setup/sampling entrypoint runs during config."""


@pytest.fixture(autouse=True)
def _trip_wires(monkeypatch):
    """Make every rejection setup/sampling entrypoint fatal if reached."""

    def _boom(*args, **kwargs):
        raise _ReachedSetupError("a rejection setup/sampling entrypoint was reached during config")

    monkeypatch.setattr(iface.SpecMetadata, "prepare_rejection_sampling_buffers", _boom)
    monkeypatch.setattr(iface.SpecWorkerBase, "sample_draft", _boom)
    monkeypatch.setattr(iface.SpecWorkerBase, "sample_draft_block", _boom)
    monkeypatch.setattr(iface.SpecWorkerBase, "_draft_sampler_advanced_for_rejection", _boom)


def _mtp(**kw):
    return MTPDecodingConfig(max_draft_len=1, use_mtp_vanilla=True, **kw)


def _flashinfer(monkeypatch, available: bool):
    monkeypatch.setattr(flashinfer_utils, "IS_FLASHINFER_AVAILABLE", available)


# --- Explicit opt-in must raise at construction (before any setup) ----------


def test_flashinfer_absent_explicit_raises(monkeypatch):
    _flashinfer(monkeypatch, False)
    with pytest.raises(ValueError):
        TorchLlmArgs(model=MODEL, speculative_config=_mtp(use_rejection_sampling=True))


def test_tp_explicit_raises(monkeypatch):
    _flashinfer(monkeypatch, True)
    with pytest.raises(ValueError):
        TorchLlmArgs(
            model=MODEL,
            speculative_config=_mtp(use_rejection_sampling=True),
            tensor_parallel_size=2,
        )


def test_cp_explicit_raises(monkeypatch):
    _flashinfer(monkeypatch, True)
    with pytest.raises(ValueError):
        TorchLlmArgs(
            model=MODEL,
            speculative_config=_mtp(use_rejection_sampling=True),
            context_parallel_size=2,
        )


def test_adp_explicit_raises(monkeypatch):
    _flashinfer(monkeypatch, True)
    with pytest.raises(ValueError):
        TorchLlmArgs(
            model=MODEL,
            speculative_config=_mtp(use_rejection_sampling=True),
            enable_attention_dp=True,
        )


def test_guided_explicit_raises(monkeypatch):
    _flashinfer(monkeypatch, True)
    with pytest.raises(ValueError):
        TorchLlmArgs(
            model=MODEL,
            speculative_config=_mtp(use_rejection_sampling=True),
            guided_decoding_backend="xgrammar",
        )


# --- Default / non-explicit must silently disable, never set up -------------


def test_default_tp_silently_disables(monkeypatch):
    _flashinfer(monkeypatch, True)
    # use_rejection_sampling not set -> inherited default; TP active -> disabled.
    args = TorchLlmArgs(model=MODEL, speculative_config=_mtp(), tensor_parallel_size=2)
    assert args.speculative_config.use_rejection_sampling is False


# --- Positive control: supported single-GPU config keeps rejection on -------


def test_single_gpu_flashinfer_available_keeps_rejection(monkeypatch):
    _flashinfer(monkeypatch, True)
    args = TorchLlmArgs(model=MODEL, speculative_config=_mtp(use_rejection_sampling=True))
    assert args.speculative_config.use_rejection_sampling is True
