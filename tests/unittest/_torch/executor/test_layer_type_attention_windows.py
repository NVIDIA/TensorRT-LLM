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

import pytest

from tensorrt_llm._torch.pyexecutor._util import _derive_layer_type_attention_windows

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
