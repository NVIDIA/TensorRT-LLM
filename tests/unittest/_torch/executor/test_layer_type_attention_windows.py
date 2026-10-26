# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for deriving per-layer attention windows from an HF ``layer_types`` list.

``_derive_layer_type_attention_windows`` turns a config that interleaves
``sliding_attention`` and ``full_attention`` layers around a single
``sliding_window`` value into the per-layer window vector the V2 KV cache
manager needs to build one pool group per window size. It returns ``None``
whenever the single-window default is already correct, which leaves the
caller's ``KvCacheConfig`` untouched.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import tensorrt_llm._torch.pyexecutor._util as util
from tensorrt_llm._torch.pyexecutor._util import (
    _create_kv_cache_manager,
    _derive_layer_type_attention_windows,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.llmapi.llm_args import KvCacheConfig

pytestmark = pytest.mark.cpu_only

MAX_SEQ_LEN = 8192
SLIDING_WINDOW = 1024


def _config(**kwargs):
    config = {
        "num_hidden_layers": 4,
        "layer_types": ["sliding_attention"] * 3 + ["full_attention"],
        "sliding_window": SLIDING_WINDOW,
    }
    config.update(kwargs)
    return SimpleNamespace(**config)


def test_mixed_layer_types_produce_per_layer_windows():
    """A mixed schedule yields one window per layer, in global layer order."""
    windows = _derive_layer_type_attention_windows(_config(), MAX_SEQ_LEN)
    assert windows == [SLIDING_WINDOW] * 3 + [MAX_SEQ_LEN]


def test_windows_cover_every_layer_in_global_order():
    """The vector spans the whole model, not a pipeline-parallel shard.

    The KV cache manager treats ``max_attention_window`` as a pattern anchored
    at global layer 0 and projects it onto the local layers of each PP rank,
    so the helper must not apply any layer offset itself.
    """
    layer_types = ["full_attention", "sliding_attention"] * 8
    windows = _derive_layer_type_attention_windows(
        _config(num_hidden_layers=16, layer_types=layer_types), MAX_SEQ_LEN
    )
    assert windows == [MAX_SEQ_LEN, SLIDING_WINDOW] * 8


def test_sliding_window_larger_than_max_seq_len_is_clamped():
    windows = _derive_layer_type_attention_windows(
        _config(sliding_window=MAX_SEQ_LEN * 2), MAX_SEQ_LEN
    )
    assert windows is None


def test_all_full_attention_returns_none():
    """No sliding layer means no per-layer windows to derive."""
    assert (
        _derive_layer_type_attention_windows(
            _config(layer_types=["full_attention"] * 4), MAX_SEQ_LEN
        )
        is None
    )


def test_all_sliding_attention_returns_none():
    """A uniform window is already covered by the single-window default."""
    assert (
        _derive_layer_type_attention_windows(
            _config(layer_types=["sliding_attention"] * 4), MAX_SEQ_LEN
        )
        is None
    )


@pytest.mark.parametrize("layer_types", [None, []])
def test_missing_layer_types_returns_none(layer_types):
    assert (
        _derive_layer_type_attention_windows(_config(layer_types=layer_types), MAX_SEQ_LEN) is None
    )


def test_missing_sliding_window_returns_none():
    assert _derive_layer_type_attention_windows(_config(sliding_window=None), MAX_SEQ_LEN) is None


def test_use_sliding_window_disabled_returns_none():
    assert (
        _derive_layer_type_attention_windows(_config(use_sliding_window=False), MAX_SEQ_LEN) is None
    )


def test_unknown_layer_type_is_treated_as_full_attention():
    """Unrecognized layer types do not raise; only names containing "sliding"
    are bounded, everything else attends to the full sequence."""
    windows = _derive_layer_type_attention_windows(
        _config(
            layer_types=[
                "sliding_attention",
                "linear_attention",
                "full_attention",
                "some_future_attention",
            ]
        ),
        MAX_SEQ_LEN,
    )
    assert windows == [SLIDING_WINDOW, MAX_SEQ_LEN, MAX_SEQ_LEN, MAX_SEQ_LEN]


def test_unsupported_window_metadata_falls_back_to_default():
    """A per-layer ``sliding_window`` list is not supported: warn, not raise."""
    assert (
        _derive_layer_type_attention_windows(
            _config(sliding_window=[SLIDING_WINDOW, None, None, None]), MAX_SEQ_LEN
        )
        is None
    )


@pytest.mark.parametrize("num_hidden_layers", [None, 0, -1])
def test_missing_num_hidden_layers_returns_none(num_hidden_layers):
    assert (
        _derive_layer_type_attention_windows(
            _config(num_hidden_layers=num_hidden_layers), MAX_SEQ_LEN
        )
        is None
    )


def test_invalid_sliding_window_falls_back_to_default():
    """``sliding_window=0`` is rejected by the per-layer resolver."""
    assert _derive_layer_type_attention_windows(_config(sliding_window=0), MAX_SEQ_LEN) is None


# ---------------------------------------------------------------------------
# Integration coverage: the derived vector reaching ``_create_kv_cache_manager``.
#
# The tests above exercise the pure helper. These drive the real
# ``_create_kv_cache_manager`` with the heavy manager construction replaced by a
# recording stub, so they assert on the ``KvCacheConfig`` that would reach the
# constructed manager without touching a GPU or allocating any KV cache. They
# are still CPU-only: the ``torch`` import above is already an implicit
# dependency of ``_util`` (the module under test imports it at load time).
# ---------------------------------------------------------------------------

_FAMILY_PREDICATES = (
    "is_kimi_linear",
    "is_mla",
    "is_nemotron_hybrid",
    "is_qwen3_hybrid",
    "is_qwen4_exp",
)


def _make_recording_manager(base):
    """A ``base`` subclass whose constructor records its config and does no work.

    ``base`` is ``KVCacheManagerV2`` or ``KVCacheManager`` so the ``issubclass``
    dispatch inside ``_create_kv_cache_manager`` (which decides whether to derive
    per-layer windows) sees the real class hierarchy. ``__init__`` deliberately
    skips ``super().__init__`` so no pools are allocated; the two abstract
    ``BaseResourceManager`` methods are stubbed only so the class is
    instantiable.
    """
    captured = {}

    class _Recording(base):
        def __init__(self, kv_cache_config, *args, **kwargs):
            captured["kv_cache_config"] = kv_cache_config
            captured["args"] = args
            captured["kwargs"] = kwargs

        def get_max_resource_count(self):
            return 0

        def get_needed_resource_to_completion(self, request):
            return 0

    return _Recording, captured


def _pretrained_config(**overrides):
    config = {
        "num_hidden_layers": 4,
        "layer_types": ["sliding_attention"] * 3 + ["full_attention"],
        "sliding_window": SLIDING_WINDOW,
        "hidden_size": 64,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "vocab_size": 1000,
    }
    config.update(overrides)
    return SimpleNamespace(**config)


def _run_create_kv_cache_manager(
    monkeypatch, manager_cls, kv_cache_config, *, is_draft, config_overrides=None
):
    # Force the generic (non-hybrid-family) construction path so the recording
    # stub is what gets built; the derivation under test runs before this
    # dispatch and is independent of the model family.
    for name in _FAMILY_PREDICATES:
        monkeypatch.setattr(util, name, lambda _config: False)

    pretrained_config = _pretrained_config(**(config_overrides or {}))
    model_config = SimpleNamespace(pretrained_config=pretrained_config, quant_config=None)
    _create_kv_cache_manager(
        model_engine=None,
        kv_cache_manager_cls=manager_cls,
        mapping=MagicMock(),
        kv_cache_config=kv_cache_config,
        tokens_per_block=32,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=8,
        spec_config=None,
        sparse_attention_config=None,
        max_num_tokens=MAX_SEQ_LEN,
        max_beam_width=1,
        kv_connector_manager=None,
        model_config=model_config,
        dtype=torch.float16,
        is_draft=is_draft,
    )


def test_derived_windows_reach_v2_manager(monkeypatch):
    """(a) Mixed schedule + unset window + V2 -> derived vector is installed."""
    manager_cls, captured = _make_recording_manager(KVCacheManagerV2)
    kv_cache_config = KvCacheConfig()
    assert kv_cache_config.max_attention_window is None

    _run_create_kv_cache_manager(monkeypatch, manager_cls, kv_cache_config, is_draft=False)

    installed = captured["kv_cache_config"].max_attention_window
    assert installed == [SLIDING_WINDOW] * 3 + [MAX_SEQ_LEN]
    # The caller's config is left untouched: derivation copies before writing.
    assert kv_cache_config.max_attention_window is None


def test_explicit_window_is_left_unchanged(monkeypatch):
    """(b) An explicit user window disables derivation and is passed through."""
    manager_cls, captured = _make_recording_manager(KVCacheManagerV2)
    explicit = [4096]
    kv_cache_config = KvCacheConfig(max_attention_window=explicit)

    _run_create_kv_cache_manager(monkeypatch, manager_cls, kv_cache_config, is_draft=False)

    assert captured["kv_cache_config"].max_attention_window == explicit


def test_v1_manager_skips_derivation(monkeypatch):
    """(c) V1 manager -> no derivation, window stays unset."""
    manager_cls, captured = _make_recording_manager(KVCacheManager)
    assert not issubclass(manager_cls, KVCacheManagerV2)
    kv_cache_config = KvCacheConfig()

    _run_create_kv_cache_manager(monkeypatch, manager_cls, kv_cache_config, is_draft=False)

    assert captured["kv_cache_config"].max_attention_window is None


def test_draft_manager_skips_derivation(monkeypatch):
    """(d) is_draft=True -> derivation skipped even on a V2 manager (Change A)."""
    manager_cls, captured = _make_recording_manager(KVCacheManagerV2)
    kv_cache_config = KvCacheConfig()

    _run_create_kv_cache_manager(monkeypatch, manager_cls, kv_cache_config, is_draft=True)

    assert captured["kv_cache_config"].max_attention_window is None


# ---------------------------------------------------------------------------
# Step-5: a vision-language composite whose decoder fields live under
# ``text_config``; 69 sliding layers at a 512-token window and 23 full-attention
# layers (3, 7, ..., 91) that also hold the sparse indexer's state.
# ---------------------------------------------------------------------------

STEP5_LAYERS = 92
STEP5_WINDOW = 512
STEP5_MAX_SEQ_LEN = 9216
STEP5_FULL_LAYERS = tuple(range(3, STEP5_LAYERS, 4))


def _step5_config():
    from tensorrt_llm._torch.configs import Step5Config

    # The checkpoint lists the three MTP layers after the decoder layers.
    layer_types = (["sliding_attention"] * 3 + ["full_attention"]) * 23 + ["sliding_attention"] * 3
    return Step5Config(
        text_config=dict(
            num_hidden_layers=STEP5_LAYERS,
            layer_types=layer_types,
            sliding_window=STEP5_WINDOW,
            num_attention_heads=64,
            num_attention_groups=4,
            head_dim=192,
            sparse_config={
                "enabled": True,
                "proxy_dim": 256,
                "sparse_indexer_num_k_heads": 1,
                "region_block_size": 8,
                "apply_to_layer_types": ["full_attention"],
            },
        )
    )


def _step5_windows():
    return [
        STEP5_MAX_SEQ_LEN if layer in STEP5_FULL_LAYERS else STEP5_WINDOW
        for layer in range(STEP5_LAYERS)
    ]


def test_step5_windows_come_from_its_text_config():
    """The composite carries no layer_types of its own; the windows come from its
    decoder's, and the MTP entries past layer 91 are not cache layers."""
    from tensorrt_llm._torch.attention.backends.sparse.step5 import Step5CacheManager

    model_config = SimpleNamespace(pretrained_config=_step5_config())
    windows = util._derive_v2_layer_type_attention_windows(
        KvCacheConfig(), Step5CacheManager, model_config, STEP5_MAX_SEQ_LEN
    )
    assert windows == _step5_windows()


def _step5_model_config():
    return SimpleNamespace(
        pretrained_config=_step5_config(),
        sparse_attention_config=None,
        quant_config=None,
        attn_backend="TRTLLM",
        mapping=None,
    )


def test_step5_selects_its_v2_manager():
    from tensorrt_llm._torch.attention.backends.sparse.step5 import Step5CacheManager

    assert (
        util.get_kv_cache_manager_cls(_step5_model_config(), KvCacheConfig()) is Step5CacheManager
    )


def test_step5_refuses_the_v1_manager():
    """The indexer state is a V2 buffer: a V1 request fails instead of serving
    the full-attention layers without it."""
    with pytest.raises(ValueError, match="Step-5 requires the KV cache manager V2"):
        util.get_kv_cache_manager_cls(
            _step5_model_config(), KvCacheConfig(use_kv_cache_manager_v2=False)
        )


@pytest.mark.parametrize(("tp_size", "attention_dp", "kv_heads"), [(4, False, 1), (4, True, 4)])
def test_step5_static_cost_charges_padded_kv_and_indexer(tp_size, attention_dp, kv_heads):
    """Per token, each full-attention layer costs its K/V at heads padded to 256
    plus 1056 bytes of indexer state ([k | z] bf16 at proxy_dim 256, and one fp8
    256-wide summary per 8 tokens); the sliding layers are a per-request cost."""
    from tensorrt_llm._torch.attention.backends.sparse.step5 import Step5CacheManager
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(world_size=tp_size, rank=0, tp_size=tp_size, enable_attention_dp=attention_dp)
    model_config = ModelConfig(pretrained_config=_step5_config(), mapping=mapping)
    slope, fixed = Step5CacheManager.get_cache_size_per_token(
        model_config,
        mapping,
        tokens_per_block=32,
        kv_cache_config=KvCacheConfig(max_attention_window=_step5_windows()),
        max_seq_len=STEP5_MAX_SEQ_LEN,
        max_batch_size=128,
        max_num_tokens=8192,
    )
    bf16_kv_per_token = 2 * kv_heads * 256 * 2
    assert slope == len(STEP5_FULL_LAYERS) * (bf16_kv_per_token + 1056)
    assert fixed > 0
