# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""_preprocess_inputs calls on_update_kv_lens once per step.

Under the overlap scheduler with speculative decoding the runner corrects
kv_lens_cuda and calls the hook after the correction; the call before it is
dropped. Without a correction the single call before it is kept.
"""

from types import SimpleNamespace

import pytest
import torch

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


@pytest.mark.parametrize(
    "spec_decode,overlap,corrected",
    [(True, True, True), (True, False, False), (False, True, False)],
)
def test_hook_runs_once_per_step(spec_decode, overlap, corrected):
    metadata = _Metadata()
    _run_preprocess_inputs(metadata, spec_decode=spec_decode, overlap=overlap)
    expected = KV_LENS + KV_LENS_OFFSETS if corrected else KV_LENS
    assert len(metadata.kv_lens_at_hook) == 1
    assert torch.equal(metadata.kv_lens_at_hook[0], expected)
