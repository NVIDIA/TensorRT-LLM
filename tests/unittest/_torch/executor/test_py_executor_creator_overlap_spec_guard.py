# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The overlap-scheduler KV-length-correction guard in the executor creator.

With the overlap scheduler on, generation rows are prepared before acceptance
is known, so their KV lengths have to be corrected afterwards. One correction
hook, ``apply_spec_decode_kv_lens_offsets``, only runs for speculation configs
that share the target KV cache; for the others it returns immediately and the
engine decodes against KV slots that were never committed, producing wrong
tokens with no error. The creator refuses that combination.
"""

import pytest

from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE
from tensorrt_llm._torch.pyexecutor import py_executor_creator
from tensorrt_llm._torch.pyexecutor.py_executor_creator import (
    ALLOW_UNCORRECTED_OVERLAP_SPEC_ENV_VAR,
    _enforce_overlap_spec_kv_correction,
    _overlap_spec_kv_lengths_uncorrected,
)
from tensorrt_llm.llmapi import (
    DFlashDecodingConfig,
    MiniMaxM3SparseAttentionConfig,
    MTPDecodingConfig,
    NGramDecodingConfig,
    SADecodingConfig,
)

pytestmark = pytest.mark.cpu_only


@pytest.fixture(autouse=True)
def _clear_escape_hatch(monkeypatch):
    """Isolate every test from an inherited opt-in env value.

    The escape-hatch tests set their own required value explicitly, so this
    only guarantees the default-refusal tests see the env var unset.
    """
    monkeypatch.delenv(ALLOW_UNCORRECTED_OVERLAP_SPEC_ENV_VAR, raising=False)


class _OptInCorrectionMetadata:
    """Stands in for metadata whose KV-length correction is opt-in."""

    def apply_spec_decode_kv_lens_offsets(self, *args, **kwargs):
        raise NotImplementedError


class _UnconditionalCorrectionMetadata:
    """Stands in for metadata the model engine corrects via kv_lens_cuda."""


def _use_stub_backend(monkeypatch, metadata_cls):
    """Resolve every backend name to a stub exposing ``metadata_cls``.

    The guard has to behave the same wherever it runs, including hosts with no
    FlashInfer install, so the logic is pinned against stub metadata and the
    real FlashInfer coupling is pinned separately below.
    """

    class _Backend:
        Metadata = metadata_cls

    monkeypatch.setattr(
        py_executor_creator,
        "get_attention_backend",
        lambda backend_name, sparse_params=None: _Backend,
    )


def _sa_config():
    return SADecodingConfig(max_draft_len=4)


def _shared_kv_config():
    """The one combination the correction is actually engaged for.

    ``update_spec_config_from_model_config`` sets ``_use_shared_kv_cache`` for
    one-model MTP-Eagle on the target architectures that share the KV cache.
    """
    config = MTPDecodingConfig(num_nextn_predict_layers=1)
    config._use_shared_kv_cache = True
    return config


def test_suffix_automaton_with_overlap_is_refused(monkeypatch):
    _use_stub_backend(monkeypatch, _OptInCorrectionMetadata)
    config = _sa_config()
    assert config.spec_dec_mode.use_one_engine()
    assert not config._use_shared_kv_cache
    assert _overlap_spec_kv_lengths_uncorrected("FLASHINFER", config, False)
    with pytest.raises(ValueError) as excinfo:
        _enforce_overlap_spec_kv_correction("FLASHINFER", config, False)
    # The message has to name the remedy, not just the failure.
    assert "disable_overlap_scheduler=True" in str(excinfo.value)
    assert ALLOW_UNCORRECTED_OVERLAP_SPEC_ENV_VAR in str(excinfo.value)


def test_suffix_automaton_without_overlap_is_admitted(monkeypatch):
    _use_stub_backend(monkeypatch, _OptInCorrectionMetadata)
    config = _sa_config()
    assert not _overlap_spec_kv_lengths_uncorrected("FLASHINFER", config, True)
    _enforce_overlap_spec_kv_correction("FLASHINFER", config, True)


def test_dflash_with_overlap_is_refused(monkeypatch):
    """The guard is keyed on the correction, not on a list of modes.

    DFlash on FlashInfer is separately refused by the one-engine gate above
    it, so this pins the guard's own verdict rather than the creator's.
    """
    _use_stub_backend(monkeypatch, _OptInCorrectionMetadata)
    config = DFlashDecodingConfig(max_draft_len=7)
    assert config.spec_dec_mode.use_one_engine()
    assert _overlap_spec_kv_lengths_uncorrected("FLASHINFER", config, False)
    with pytest.raises(ValueError):
        _enforce_overlap_spec_kv_correction("FLASHINFER", config, False)


def test_shared_kv_cache_with_overlap_is_admitted(monkeypatch):
    _use_stub_backend(monkeypatch, _OptInCorrectionMetadata)
    config = _shared_kv_config()
    assert config.spec_dec_mode.use_one_engine()
    assert not _overlap_spec_kv_lengths_uncorrected("FLASHINFER", config, False)
    _enforce_overlap_spec_kv_correction("FLASHINFER", config, False)


def test_escape_hatch_warns_instead_of_refusing(monkeypatch):
    _use_stub_backend(monkeypatch, _OptInCorrectionMetadata)
    monkeypatch.setenv(ALLOW_UNCORRECTED_OVERLAP_SPEC_ENV_VAR, "1")
    config = _sa_config()
    # The configuration is still reported as uncorrected; only the reaction
    # changes.
    assert _overlap_spec_kv_lengths_uncorrected("FLASHINFER", config, False)
    warnings = []
    monkeypatch.setattr(
        py_executor_creator.logger,
        "warning_once",
        lambda message, **kwargs: warnings.append((message, kwargs)),
    )
    # Returning without raising is not enough: the corruption warning has to
    # actually fire so the operator is told the outputs are unreliable.
    _enforce_overlap_spec_kv_correction("FLASHINFER", config, False)
    assert len(warnings) == 1
    message, kwargs = warnings[0]
    assert kwargs.get("key") == "uncorrected_overlap_spec"
    assert ALLOW_UNCORRECTED_OVERLAP_SPEC_ENV_VAR in message


def test_escape_hatch_requires_the_exact_opt_in(monkeypatch):
    _use_stub_backend(monkeypatch, _OptInCorrectionMetadata)
    monkeypatch.setenv(ALLOW_UNCORRECTED_OVERLAP_SPEC_ENV_VAR, "0")
    with pytest.raises(ValueError):
        _enforce_overlap_spec_kv_correction("FLASHINFER", _sa_config(), False)


def test_backends_correcting_kv_lengths_unconditionally_are_not_gated(monkeypatch):
    """Metadata without the opt-in hook is corrected by the model engine."""
    _use_stub_backend(monkeypatch, _UnconditionalCorrectionMetadata)
    assert not _overlap_spec_kv_lengths_uncorrected("TRTLLM", _sa_config(), False)


def test_external_drafters_are_not_gated(monkeypatch):
    _use_stub_backend(monkeypatch, _OptInCorrectionMetadata)
    config = NGramDecodingConfig(max_draft_len=4, max_matching_ngram_size=2)
    assert not config.spec_dec_mode.use_one_engine()
    assert not _overlap_spec_kv_lengths_uncorrected("FLASHINFER", config, False)


def test_no_speculation_is_not_gated(monkeypatch):
    _use_stub_backend(monkeypatch, _OptInCorrectionMetadata)
    assert not _overlap_spec_kv_lengths_uncorrected("FLASHINFER", None, False)


def test_sparse_attention_judges_the_effective_backend(monkeypatch):
    """A sparse config replaces the dense backend in its slot at runtime.

    The guard has to resolve with the lowered sparse params like the engine
    does, otherwise it judges dense metadata that will never run and refuses
    combinations the sparse backend corrects on its own.
    """
    received = []

    class _DenseBackend:
        Metadata = _OptInCorrectionMetadata

    class _SparseBackend:
        Metadata = _UnconditionalCorrectionMetadata

    def _resolve(backend_name, sparse_params=None):
        received.append(sparse_params)
        return _DenseBackend if sparse_params is None else _SparseBackend

    monkeypatch.setattr(py_executor_creator, "get_attention_backend", _resolve)
    sparse_config = MiniMaxM3SparseAttentionConfig(implementation="triton")
    config = _sa_config()
    assert not _overlap_spec_kv_lengths_uncorrected(
        "FLASHINFER", config, False, sparse_attention_config=sparse_config
    )
    _enforce_overlap_spec_kv_correction(
        "FLASHINFER", config, False, sparse_attention_config=sparse_config
    )
    assert received and all(params is not None for params in received)


def test_unlowerable_sparse_config_is_not_refused(monkeypatch):
    """Lowering can need checkpoint fields that do not exist at this point.

    The engine lowers with pretrained_config later; when that makes the
    effective backend unresolvable here, the guard must stand down rather
    than refuse (or crash) on metadata it cannot identify.
    """
    _use_stub_backend(monkeypatch, _OptInCorrectionMetadata)

    class _NeedsCheckpoint:
        def to_sparse_params(self, **kwargs):
            raise ValueError("resolved from a checkpoint config")

    assert not _overlap_spec_kv_lengths_uncorrected(
        "FLASHINFER", _sa_config(), False, sparse_attention_config=_NeedsCheckpoint()
    )


@pytest.mark.skipif(not IS_FLASHINFER_AVAILABLE, reason="requires the FlashInfer backend")
def test_minimax_m3_sparse_attention_with_overlap_is_admitted():
    """Real-coupling pin for the sparse resolution path.

    MiniMax-M3 under the FLASHINFER slot selects a TrtllmAttention-based
    sparse metadata that the engine corrects through ``kv_lens_cuda``, so
    overlap plus one-engine speculation must be admitted.
    """
    sparse_config = MiniMaxM3SparseAttentionConfig(implementation="triton")
    assert not _overlap_spec_kv_lengths_uncorrected(
        "FLASHINFER", _sa_config(), False, sparse_attention_config=sparse_config
    )


@pytest.mark.skipif(not IS_FLASHINFER_AVAILABLE, reason="requires the FlashInfer backend")
def test_flashinfer_is_the_backend_the_guard_is_aimed_at():
    """Without this, the stub tests above could drift off the real backend."""
    config = _sa_config()
    assert _overlap_spec_kv_lengths_uncorrected("FLASHINFER", config, False)
    assert not _overlap_spec_kv_lengths_uncorrected("TRTLLM", config, False)
