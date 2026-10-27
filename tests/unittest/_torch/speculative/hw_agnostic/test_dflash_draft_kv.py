# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Separate-draft-KV policy follows the DFlash backend's storage ownership.

TRTLLM borrows the managed paged pool. VANILLA and FA4 own private context
buffers and must not allocate a redundant manager or be gated on its capacity.
"""

import json

import pytest

from tensorrt_llm._torch.speculative.interface import should_use_separate_draft_kv_cache
from tensorrt_llm.llmapi import DFlashDecodingConfig, EagleDecodingConfig

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    ("backend", "uses_managed_pool"),
    [("VANILLA", False), ("FA4", False), ("TRTLLM", True)],
)
def test_dflash_draft_cache_matches_backend_storage(backend: str, uses_managed_pool: bool) -> None:
    config = DFlashDecodingConfig(max_draft_len=7, attention_backend=backend)
    assert config.spec_dec_mode.use_one_engine()
    assert not config._use_shared_kv_cache
    assert config._allow_separate_draft_kv_cache
    assert should_use_separate_draft_kv_cache(config) is uses_managed_pool


_WINDOWED = {"layer_types": ["sliding_attention"] * 2, "sliding_window": 16}
_UNWINDOWED = {"layer_types": ["full_attention"] * 2}


@pytest.mark.parametrize(
    ("drafter", "unavailable", "resolved"),
    [(_WINDOWED, None, "TRTLLM"), (_WINDOWED, "no", "VANILLA"), (_UNWINDOWED, None, "VANILLA")],
)
def test_default_dflash_resolves_auto_before_choosing_its_draft_cache(
    tmp_path, monkeypatch, drafter, unavailable, resolved
) -> None:
    """AUTO is pinned from the checkpoint by the DFlash rule before the cache decision reads it."""
    from tensorrt_llm._torch.speculative import dflash
    from tensorrt_llm._torch.speculative.utils import get_num_spec_layers
    from tensorrt_llm.llmapi import llm_args

    monkeypatch.setattr(llm_args.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(dflash, "dflash_trtllm_gen_unavailability_reason", lambda: unavailable)
    (tmp_path / "config.json").write_text(json.dumps(drafter))
    config = DFlashDecodingConfig(max_draft_len=7, speculative_model=str(tmp_path))
    assert config.attention_backend == "AUTO"
    config.resolve_from_checkpoint()
    assert config.attention_backend == resolved
    assert should_use_separate_draft_kv_cache(config) is (resolved == "TRTLLM")
    assert get_num_spec_layers(config) == 0


@pytest.mark.parametrize(
    ("transceiver", "shared", "modeling_v2", "resolved"),
    [
        ("NIXL", "0", "off", "VANILLA"),
        ("NIXL", "1", "off", "AUTO"),
        ("NIXL", None, "off", "AUTO"),
        (None, "0", "auto", "VANILLA"),
        (None, "0", "off", "AUTO"),
    ],
)
def test_auto_keeps_vanilla_where_trtllm_is_refused_or_bypassed(
    tmp_path, monkeypatch, transceiver, shared, modeling_v2, resolved
) -> None:
    """Workers keeping a private draft KV refuse TRTLLM; modeling_v2 DFlash compositions need VANILLA."""
    from tensorrt_llm.llmapi import CacheTransceiverConfig, TorchLlmArgs, llm_args

    # No GPU: whatever stays AUTO is left to the ranks' windowed-and-probe rule.
    monkeypatch.setattr(llm_args.torch.cuda, "is_available", lambda: False)
    if shared is None:
        monkeypatch.delenv("TRTLLM_DISAGG_SHARED_DRAFT_KV", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_DISAGG_SHARED_DRAFT_KV", shared)
    (tmp_path / "config.json").write_text(json.dumps(_WINDOWED))
    args = TorchLlmArgs(
        model="dummy",
        speculative_config=DFlashDecodingConfig(max_draft_len=7, speculative_model=str(tmp_path)),
        cache_transceiver_config=transceiver and CacheTransceiverConfig(backend=transceiver),
        modeling_v2=modeling_v2,
        max_batch_size=8,
        max_num_tokens=8192,
        max_seq_len=8192,
    )
    assert args.speculative_config.attention_backend == resolved


def test_rank_pins_auto_before_choosing_its_draft_cache(tmp_path, monkeypatch) -> None:
    """An AUTO a GPU-less frontend left open is pinned on the rank; DSpark keeps its own AUTO."""
    from tensorrt_llm._torch.pyexecutor.py_executor_creator import (
        _pin_dflash_auto_attention_backend,
    )
    from tensorrt_llm._torch.speculative import dflash
    from tensorrt_llm.llmapi import DSparkDecodingConfig, llm_args

    monkeypatch.setattr(llm_args.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(dflash, "dflash_trtllm_gen_unavailability_reason", lambda: None)
    (tmp_path / "config.json").write_text(json.dumps(_WINDOWED))
    config = DFlashDecodingConfig(max_draft_len=7, speculative_model=str(tmp_path))
    _pin_dflash_auto_attention_backend(config)
    assert config.attention_backend == "TRTLLM"
    assert should_use_separate_draft_kv_cache(config) is True
    dspark = DSparkDecodingConfig(max_draft_len=7)
    _pin_dflash_auto_attention_backend(dspark)
    assert dspark.attention_backend == "AUTO"


@pytest.mark.parametrize("backend", ["VANILLA", "FA4", "TRTLLM"])
def test_dflash_explicit_cache_opt_out_is_preserved(backend: str) -> None:
    config = DFlashDecodingConfig(max_draft_len=7, attention_backend=backend)
    config._allow_separate_draft_kv_cache = False
    assert should_use_separate_draft_kv_cache(config) is False


def test_eagle3_one_model_still_gets_a_separate_draft_kv_cache():
    """The DFlash exemption must not leak into other one-engine drafters."""
    config = EagleDecodingConfig(max_draft_len=3, speculative_model="/tmp/eagle3-draft")
    assert config.spec_dec_mode.is_eagle3_one_model()
    assert should_use_separate_draft_kv_cache(config) is True
