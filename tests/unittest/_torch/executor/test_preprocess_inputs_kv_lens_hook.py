# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""_preprocess_inputs calls an idempotent on_update_kv_lens once instead of twice.

Under the overlap scheduler with speculative decoding the engine corrects
kv_lens_cuda and calls the hook again right after. A hook marked with
idempotent_kv_lens_hook is then called once, after the correction; an unmarked
hook, or a run without a correction, keeps the original calls.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.interface import (
    idempotent_kv_lens_hook,
    kv_lens_hook_is_idempotent,
)
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import DecoderRunner

KV_LENS = torch.tensor([10, 20, 30], dtype=torch.int32)
KV_LENS_OFFSETS = torch.tensor([1, 2, 0], dtype=torch.int32)
TOKENS_PER_REQUEST = 3
NUM_TOKENS = KV_LENS.numel() * TOKENS_PER_REQUEST


class _Metadata:
    """The slice of TrtllmAttentionMetadata that _preprocess_inputs touches."""

    def __init__(self):
        self.kv_cache_manager = object()
        self.num_seqs = KV_LENS.numel()
        self.num_contexts = 0
        self.num_generations = KV_LENS.numel()
        self.num_ctx_tokens = 0
        self.num_chunked_ctx_requests = 0
        self.kv_lens_cuda = KV_LENS.clone()
        # kv_lens_cuda as seen by each on_update_kv_lens() call.
        self.kv_lens_at_hook = []

    def on_update_kv_lens(self):
        self.kv_lens_at_hook.append(self.kv_lens_cuda.clone())


class _IdempotentMetadata(_Metadata):
    @idempotent_kv_lens_hook
    def on_update_kv_lens(self):
        super().on_update_kv_lens()


class _UnmarkedSubclassMetadata(_IdempotentMetadata):
    """Overrides a marked hook without the marker (DeepSeek-V4 over DSA)."""

    def on_update_kv_lens(self):
        super().on_update_kv_lens()


def _run_preprocess_inputs(metadata, *, spec_decode, overlap):
    engine = object.__new__(DecoderRunner)
    engine._config = SimpleNamespace(disable_overlap_scheduler=not overlap)
    engine.previous_pos_id_offsets_cuda = torch.arange(NUM_TOKENS, dtype=torch.int32)
    engine.previous_kv_lens_offsets_cuda = KV_LENS_OFFSETS.clone()
    engine.mapping = SimpleNamespace(has_cp_helix=lambda: False)
    engine.guided_decoder = None
    inputs = {
        "input_ids": torch.zeros(NUM_TOKENS, dtype=torch.int32),
        "position_ids": torch.zeros(1, NUM_TOKENS, dtype=torch.int32),
        "attn_metadata": metadata,
    }
    engine._preprocess_inputs(inputs, enable_spec_decode=spec_decode, runtime_draft_len=0)
    return inputs


def test_kv_lens_hook_is_idempotent_reads_the_dispatched_hook():
    assert not kv_lens_hook_is_idempotent(_Metadata())
    assert kv_lens_hook_is_idempotent(_IdempotentMetadata())
    assert not kv_lens_hook_is_idempotent(_UnmarkedSubclassMetadata())


@pytest.mark.parametrize(
    "metadata_cls", [_Metadata, _IdempotentMetadata, _UnmarkedSubclassMetadata]
)
def test_correction_repeats_the_hook_only_when_it_is_not_idempotent(metadata_cls):
    metadata = metadata_cls()
    inputs = _run_preprocess_inputs(metadata, spec_decode=True, overlap=True)
    corrected = KV_LENS + KV_LENS_OFFSETS
    # The correction itself is untouched.
    assert torch.equal(metadata.kv_lens_cuda, corrected)
    assert torch.equal(inputs["position_ids"][0], torch.arange(NUM_TOKENS, dtype=torch.int32))
    if kv_lens_hook_is_idempotent(metadata):
        # Only the call after the correction survives.
        assert len(metadata.kv_lens_at_hook) == 1
        assert torch.equal(metadata.kv_lens_at_hook[0], corrected)
    else:
        # Once before and once after the correction.
        assert len(metadata.kv_lens_at_hook) == 2
        assert torch.equal(metadata.kv_lens_at_hook[0], KV_LENS)
        assert torch.equal(metadata.kv_lens_at_hook[1], corrected)


@pytest.mark.parametrize("spec_decode,overlap", [(False, True), (True, False), (False, False)])
def test_single_hook_call_is_kept_when_no_correction_follows(spec_decode, overlap):
    metadata = _IdempotentMetadata()
    _run_preprocess_inputs(metadata, spec_decode=spec_decode, overlap=overlap)
    assert len(metadata.kv_lens_at_hook) == 1
    assert torch.equal(metadata.kv_lens_at_hook[0], KV_LENS)
    assert torch.equal(metadata.kv_lens_cuda, KV_LENS)
