# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify that the ULL step begin launches the held spec-head prep.

``RubinUllDSAMetadata.defer_spec_head_prep`` only holds the prep on a fused-Helix-metadata step.
While it is held,
``on_update_kv_lens`` defers to ``run_held_step_begin``, whose launch takes the prep and the
metadata launch's own arguments, exactly once. The engine's ``_fused_spec_head_prep`` then skips
its launch. The kernel's bits are ``ops/embedding/test_rubin_ull_step_begin.py``'s.
"""

import importlib

import pytest

from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import DSAtrtllmAttentionMetadata

_FAMILY = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31"
_META = f"{_FAMILY}.ops.attention.rubin_ull_dsa_metadata"


@pytest.fixture
def meta(monkeypatch):
    """A ``RubinUllDSAMetadata`` without its engine state: the engine refresh is a stand-in that
    launches the fused Helix metadata through the hook with marker arguments; the engine's own
    launch records itself."""
    module = importlib.import_module(_META)
    calls = []

    def refresh(self):
        calls.append("refresh")
        self._launch_fused_dsa_helix_decode_metadata("seq_lens", num_tokens=8)

    monkeypatch.setattr(DSAtrtllmAttentionMetadata, "on_update_kv_lens", refresh)
    monkeypatch.setattr(
        DSAtrtllmAttentionMetadata,
        "_launch_fused_dsa_helix_decode_metadata",
        lambda self, *a, **k: calls.append("plain launch"),
    )
    md = object.__new__(module.RubinUllDSAMetadata)
    md._helix_fused_meta_applies = lambda: True
    md.key_shard_world = 1
    md.num_generations = 1
    return md, calls


def test_nothing_is_held_on_a_step_without_the_fused_metadata(meta) -> None:
    md, calls = meta
    md._helix_fused_meta_applies = lambda: False
    assert md.defer_spec_head_prep({"num_pos": 8}) is False
    md.on_update_kv_lens()
    assert calls == ["refresh", "plain launch"]
    assert md.run_held_step_begin(lambda *a: None) is False


def test_the_held_refresh_runs_in_the_step_begin(meta) -> None:
    md, calls = meta
    prep = {"num_pos": 8}
    assert md.defer_spec_head_prep(prep) is True
    md.on_update_kv_lens()  # the engine's refresh, deferred
    assert calls == []
    launched = []
    assert md.run_held_step_begin(lambda *a: launched.append(a)) is True
    assert calls == ["refresh"]
    assert launched == [(prep, ("seq_lens",), {"num_tokens": 8})]
    assert md._held_prep is None and md._held_launch is None
    md.on_update_kv_lens()  # later refreshes of the step run as before
    assert calls == ["refresh", "refresh", "plain launch"]


def test_a_held_prep_that_is_never_launched_is_loud(meta) -> None:
    md, calls = meta
    assert md.defer_spec_head_prep({}) is True
    with pytest.raises(RuntimeError, match="never launched"):
        md.defer_spec_head_prep({})


def test_a_refresh_without_the_metadata_launch_is_loud(meta, monkeypatch) -> None:
    md, calls = meta
    monkeypatch.setattr(DSAtrtllmAttentionMetadata, "on_update_kv_lens", lambda self: None)
    assert md.defer_spec_head_prep({}) is True
    with pytest.raises(RuntimeError, match="launched no Helix metadata"):
        md.run_held_step_begin(lambda *a: None)
    assert md._held_prep is None
