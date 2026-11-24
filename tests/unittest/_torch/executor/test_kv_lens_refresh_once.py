# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""``kv_lens_refresh_once``: a metadata class that declares it gets one ``on_update_kv_lens`` per
``preprocess_inputs``, after the last correction.

CPU stand-ins (no GPU): ``DecoderInputPreparer.preprocess_inputs`` runs on DSA metadata
stand-ins whose ``on_update_kv_lens`` records the inputs it would read. Along every path (no
speculation; the overlap correction; the Helix verify-group recompute; the fused spec head that
refreshes itself, or declines) a class that declares the attribute gets exactly one refresh, and
it sees the same inputs as the last refresh of a class that does not, so the metadata ends in the
same state. That holds whether the class inherits ``on_update_kv_lens`` or defines its own. A
class without the attribute keeps every call.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import DSAtrtllmAttentionMetadata
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import inputs as inputs_mod

NUM_GEN, NEXT_N = 3, 4


class _PlainMeta(DSAtrtllmAttentionMetadata):
    """Inherits the DSA ``on_update_kv_lens`` (patched per test to a recorder); no attribute."""


class _OnceMeta(DSAtrtllmAttentionMetadata):
    """Inherits the DSA ``on_update_kv_lens`` and declares the attribute."""

    kv_lens_refresh_once = True


class _OwnMeta(DSAtrtllmAttentionMetadata):
    """Defines its own ``on_update_kv_lens``; no attribute."""

    def on_update_kv_lens(self):
        self.calls.append(_snapshot(self))


class _OwnOnceMeta(_OwnMeta):
    """Defines its own ``on_update_kv_lens`` and declares the attribute, as a model's metadata
    subclass does."""

    kv_lens_refresh_once = True


def _snapshot(md):
    return (
        md.kv_lens_cuda.clone(),
        md.helix_position_offsets.clone(),
        md.helix_kv_bounds.clone(),
        md.position_ids.clone(),
    )


def _meta(cls, helix):
    md = object.__new__(cls)
    md.calls = []
    md.kv_cache_manager = object()
    md._seq_lens = torch.full((NUM_GEN,), NEXT_N, dtype=torch.int32)
    md._num_contexts = 0
    md._num_generations = NUM_GEN
    md._num_ctx_tokens = 0
    md._num_tokens = NUM_GEN * NEXT_N
    md.num_chunked_ctx_requests = 0
    md.kv_lens_cuda = torch.tensor([100, 200, 300], dtype=torch.int32)
    md._helix_spec_tokens_valid = helix
    md.helix_position_offsets = torch.arange(NUM_GEN * NEXT_N, dtype=torch.int32) + 1000
    md.helix_kv_bounds = torch.zeros(NUM_GEN * NEXT_N, dtype=torch.int32)

    def recompute(num_tokens, width):
        # Like populate_helix_spec_buffers: derive bounds and the gen kv lens from positions.
        md.helix_kv_bounds[:num_tokens] = md.helix_position_offsets[:num_tokens] // 7
        md.kv_lens_cuda[:NUM_GEN] = md.helix_kv_bounds[:num_tokens].view(NUM_GEN, width)[:, -1]

    md.recompute_helix_spec_buffers = recompute
    return md


def _preparer(overlap, helix):
    prep = object.__new__(inputs_mod.DecoderInputPreparer)
    prep.guided_decoder = None
    prep._config = SimpleNamespace(disable_overlap_scheduler=not overlap)
    prep.mapping = SimpleNamespace(has_cp_helix=lambda: helix)
    prep._buffers = SimpleNamespace(
        previous_pos_id_offsets_cuda=torch.arange(NUM_GEN * NEXT_N, dtype=torch.int32) % 3,
        previous_kv_lens_offsets_cuda=torch.tensor([-1, -2, -3], dtype=torch.int32),
    )
    prep.get_runtime_tokens_per_gen_step = lambda draft_len: NEXT_N
    return prep


def _run(monkeypatch, *, cls, overlap, helix, spec, fusion):
    monkeypatch.setattr(
        DSAtrtllmAttentionMetadata,
        "on_update_kv_lens",
        lambda self: self.calls.append(_snapshot(self)),
    )
    monkeypatch.setattr(inputs_mod, "spec_head_fusion_enabled", lambda: fusion is not None)
    if fusion is not None:

        def fused(self, inputs, runtime_draft_len):
            if not fusion:
                return False  # declines, having changed nothing
            md = inputs["attn_metadata"]
            md.kv_lens_cuda[:NUM_GEN] += 5
            inputs["position_ids"][0] += 1
            md.on_update_kv_lens()
            return True

        monkeypatch.setattr(inputs_mod.DecoderInputPreparer, "_fused_spec_head_prep", fused)
    md = _meta(cls, helix)
    position_ids = torch.arange(NUM_GEN * NEXT_N, dtype=torch.int32).view(1, -1)
    md.position_ids = position_ids
    inputs = {
        "attn_metadata": md,
        "input_ids": torch.zeros(NUM_GEN * NEXT_N, dtype=torch.int32),
        "position_ids": position_ids,
    }
    _preparer(overlap, helix).preprocess_inputs(
        inputs, enable_spec_decode=spec, runtime_draft_len=NEXT_N - 1
    )
    return md


CASES = {
    "no_spec": dict(overlap=True, helix=True, spec=False, fusion=None),
    "overlap": dict(overlap=True, helix=False, spec=True, fusion=None),
    "helix": dict(overlap=False, helix=True, spec=True, fusion=None),
    "overlap_helix": dict(overlap=True, helix=True, spec=True, fusion=None),
    "fused_head": dict(overlap=True, helix=True, spec=True, fusion=True),
    "fused_head_declines": dict(overlap=True, helix=True, spec=True, fusion=False),
}
CALLS_EVERY = {
    "no_spec": 1,
    "overlap": 2,
    "helix": 2,
    "overlap_helix": 3,
    "fused_head": 2,
    "fused_head_declines": 3,
}
# (class that declares the attribute, the same refresh without it)
PAIRS = {"inherited": (_OnceMeta, _PlainMeta), "own": (_OwnOnceMeta, _OwnMeta)}


@pytest.mark.parametrize("pair", list(PAIRS))
@pytest.mark.parametrize("case", list(CASES))
def test_one_refresh_with_the_last_inputs(monkeypatch, case, pair):
    once_cls, every_cls = PAIRS[pair]
    every = _run(monkeypatch, cls=every_cls, **CASES[case])
    once = _run(monkeypatch, cls=once_cls, **CASES[case])
    assert len(every.calls) == CALLS_EVERY[case]
    assert len(once.calls) == 1
    for a, b in zip(every.calls[-1], once.calls[0]):
        assert torch.equal(a, b)
    # The inputs themselves end the same, and the single refresh saw their final values.
    for a, b in zip(_snapshot(every), _snapshot(once)):
        assert torch.equal(a, b)
    for a, b in zip(_snapshot(once), once.calls[0]):
        assert torch.equal(a, b)


@pytest.mark.parametrize("cls", [_PlainMeta, _OwnMeta], ids=["inherited", "own"])
@pytest.mark.parametrize("case", list(CASES))
def test_without_the_attribute_every_call_stays(monkeypatch, case, cls):
    md = _run(monkeypatch, cls=cls, **CASES[case])
    assert len(md.calls) == CALLS_EVERY[case]


def test_the_class_decides():
    assert inputs_mod._kv_lens_refresh_once(None) is False
    assert inputs_mod._kv_lens_refresh_once(object.__new__(_PlainMeta)) is False
    assert inputs_mod._kv_lens_refresh_once(object.__new__(_OnceMeta)) is True
    assert inputs_mod._kv_lens_refresh_once(object.__new__(_OwnOnceMeta)) is True
    # An instance attribute does not opt in: the choice is per class.
    md = object.__new__(_PlainMeta)
    md.kv_lens_refresh_once = True
    assert inputs_mod._kv_lens_refresh_once(md) is False
