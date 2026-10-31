# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FlashInfer qualification is independent of DFlash cache ownership.

DFlash remains refused for every draft backend, including those using private
buffers. Other modes retain the existing separate-manager qualification gate.
"""

import pytest

from tensorrt_llm._torch.pyexecutor.py_executor_creator import _flashinfer_one_engine_spec_supported
from tensorrt_llm._torch.speculative.interface import should_use_separate_draft_kv_cache
from tensorrt_llm.llmapi import (
    DFlashDecodingConfig,
    MTPDecodingConfig,
    NGramDecodingConfig,
    SADecodingConfig,
)

pytestmark = pytest.mark.cpu_only


def test_no_speculation_is_admitted():
    assert _flashinfer_one_engine_spec_supported("FLASHINFER", None)


def test_suffix_automaton_is_admitted_on_flashinfer():
    config = SADecodingConfig(max_draft_len=4)
    assert config.spec_dec_mode.use_one_engine()
    assert not config._use_shared_kv_cache
    assert _flashinfer_one_engine_spec_supported("FLASHINFER", config)


@pytest.mark.parametrize(
    ("backend", "uses_managed_pool"),
    [("VANILLA", False), ("FA4", False), ("TRTLLM", True)],
)
def test_dflash_is_refused_on_flashinfer(backend: str, uses_managed_pool: bool) -> None:
    config = DFlashDecodingConfig(max_draft_len=7, attention_backend=backend)
    assert should_use_separate_draft_kv_cache(config) is uses_managed_pool
    assert not _flashinfer_one_engine_spec_supported("FLASHINFER", config)
    assert _flashinfer_one_engine_spec_supported("TRTLLM", config)


def test_default_dflash_is_refused_without_a_separate_manager() -> None:
    config = DFlashDecodingConfig(max_draft_len=7)
    assert not should_use_separate_draft_kv_cache(config)
    assert not _flashinfer_one_engine_spec_supported("FLASHINFER", config)


@pytest.mark.parametrize("backend", ["VANILLA", "FA4", "TRTLLM"])
def test_dflash_cache_opt_out_does_not_lift_flashinfer_gate(backend: str) -> None:
    config = DFlashDecodingConfig(max_draft_len=7, attention_backend=backend)
    config._allow_separate_draft_kv_cache = False
    assert not should_use_separate_draft_kv_cache(config)
    assert not _flashinfer_one_engine_spec_supported("FLASHINFER", config)


def test_external_drafters_are_admitted_on_flashinfer():
    config = NGramDecodingConfig(max_draft_len=4, max_matching_ngram_size=2)
    assert not config.spec_dec_mode.use_one_engine()
    assert _flashinfer_one_engine_spec_supported("FLASHINFER", config)


def test_one_engine_drafters_with_a_draft_model_are_refused_on_flashinfer():
    config = MTPDecodingConfig(num_nextn_predict_layers=1)
    assert config.spec_dec_mode.use_one_engine()
    assert not _flashinfer_one_engine_spec_supported("FLASHINFER", config)


def test_shared_kv_cache_is_admitted_on_flashinfer():
    config = MTPDecodingConfig(num_nextn_predict_layers=1)
    config._use_shared_kv_cache = True
    assert _flashinfer_one_engine_spec_supported("FLASHINFER", config)


def test_other_backends_are_not_gated():
    config = MTPDecodingConfig(num_nextn_predict_layers=1)
    assert _flashinfer_one_engine_spec_supported("TRTLLM", config)
