# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DecoderInputPreparer.preprocess_inputs re-derives the KV-length metadata once,
after the overlap scheduler's KV-length correction, on exactly the steps that correct it, and leaves the inputs as
before."""

import types

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.inputs import DecoderInputPreparer
from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks


class _Metadata:
    """The fields preprocess_inputs reads; on_update_kv_lens records the kv_lens it saw."""

    def __init__(self, gen: int, tokens_per_gen: int, with_kv_cache: bool = True):
        self.kv_cache_manager = object() if with_kv_cache else None
        self.num_seqs = gen
        self.num_contexts = 0
        self.num_generations = gen
        self.num_ctx_tokens = 0
        self.num_chunked_ctx_requests = 0
        self.kv_lens_cuda = torch.arange(100, 100 + gen, dtype=torch.int32)
        self.seen: list[list[int]] = []

    def on_update_kv_lens(self) -> None:
        self.seen.append(self.kv_lens_cuda.tolist())


def _preparer(disable_overlap: bool = False) -> DecoderInputPreparer:
    r = object.__new__(DecoderInputPreparer)
    r.runtime_hooks = RuntimeHooks()
    r._config = types.SimpleNamespace(disable_overlap_scheduler=disable_overlap)
    r._buffers = types.SimpleNamespace(
        previous_pos_id_offsets_cuda=torch.full((64,), -2, dtype=torch.int32),
        previous_kv_lens_offsets_cuda=torch.full((16,), -3, dtype=torch.int32),
    )
    r.mapping = types.SimpleNamespace(has_cp_helix=lambda: False)
    r.decode_prologue = None
    r.guided_decoder = None
    return r


def _inputs(md: _Metadata, tokens: int) -> dict:
    return {
        "attn_metadata": md,
        "input_ids": torch.zeros(tokens, dtype=torch.int32),
        "position_ids": torch.arange(tokens, dtype=torch.int32).view(1, -1),
    }


@pytest.mark.parametrize("gen", [1, 4])
def test_corrected_step_updates_once_with_the_same_inputs(gen) -> None:
    t = 4
    md = _Metadata(gen, t)
    _preparer().preprocess_inputs(
        _inputs(md, gen * t), enable_spec_decode=True, runtime_draft_len=t - 1
    )
    corrected = [100 + i - 3 for i in range(gen)]
    # One update, after the correction, for every model.
    assert md.seen == [corrected]


@pytest.mark.parametrize(
    "spec, disable_overlap, with_kv_cache",
    [(False, False, True), (True, True, True), (True, False, False)],
    ids=["no spec decode", "no overlap scheduler", "no kv cache manager"],
)
def test_uncorrected_steps_keep_their_update(spec, disable_overlap, with_kv_cache) -> None:
    """Steps that do not correct the lengths still re-derive them once (the per-forward invalidation), from the
    uncorrected lengths."""
    md = _Metadata(2, 4, with_kv_cache)
    inputs = _preparer(disable_overlap).preprocess_inputs(
        _inputs(md, 8), enable_spec_decode=spec, runtime_draft_len=3
    )
    assert md.seen == [[100, 101]]
    assert inputs["position_ids"].tolist() == [list(range(8))]
