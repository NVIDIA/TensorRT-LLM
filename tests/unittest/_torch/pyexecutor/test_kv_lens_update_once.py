# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DecoderInputPreparer.preprocess_inputs corrects only the generation positions.

The overlap scheduler's KV-length correction reaches the attention metadata as
BatchInputs.kv_lens_offsets and is applied in its prepare, so preprocess_inputs
neither changes the device KV lengths nor calls the metadata.
"""

import types

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.inputs import DecoderInputPreparer
from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks

pytestmark = pytest.mark.cpu_only


class _Metadata:
    """The fields preprocess_inputs reads; any method call is recorded."""

    supports_kv_lens_offsets = True

    def __init__(self, gen: int, with_kv_cache: bool = True):
        self.kv_cache_manager = object() if with_kv_cache else None
        self.num_seqs = gen
        self.num_contexts = 0
        self.num_generations = gen
        self.num_ctx_tokens = 0
        self.num_chunked_ctx_requests = 0
        self.kv_lens_cuda = torch.arange(100, 100 + gen, dtype=torch.int32)
        self.calls: list[str] = []

    def __getattr__(self, name: str):
        def record(*args, **kwargs) -> None:
            self.calls.append(name)

        return record


def _preparer(hooks: RuntimeHooks, disable_overlap: bool = False) -> DecoderInputPreparer:
    preparer = object.__new__(DecoderInputPreparer)
    preparer.runtime_hooks = hooks
    preparer._config = types.SimpleNamespace(disable_overlap_scheduler=disable_overlap)
    preparer._buffers = types.SimpleNamespace(
        previous_pos_id_offsets_cuda=torch.full((64,), -2, dtype=torch.int32),
        previous_kv_lens_offsets_cuda=torch.full((16,), -3, dtype=torch.int32),
    )
    preparer.mapping = types.SimpleNamespace(has_cp_helix=lambda: False)
    preparer.decode_prologue = None
    preparer.guided_decoder = None
    return preparer


def _inputs(metadata: _Metadata, tokens: int) -> dict:
    return {
        "attn_metadata": metadata,
        "input_ids": torch.zeros(tokens, dtype=torch.int32),
        "position_ids": torch.arange(tokens, dtype=torch.int32).view(1, -1),
    }


@pytest.mark.parametrize("gen", [1, 4])
def test_overlap_step_corrects_positions_only(gen) -> None:
    hooks = RuntimeHooks()
    tokens_per_gen = 4
    metadata = _Metadata(gen)
    inputs = _preparer(hooks).preprocess_inputs(
        _inputs(metadata, gen * tokens_per_gen),
        enable_spec_decode=True,
        runtime_draft_len=tokens_per_gen - 1,
    )
    num_tokens = gen * tokens_per_gen
    assert inputs["position_ids"].tolist() == [[i - 2 for i in range(num_tokens)]]
    assert metadata.kv_lens_cuda.tolist() == [100 + i for i in range(gen)]
    assert metadata.calls == []


@pytest.mark.parametrize(
    "spec, disable_overlap, with_kv_cache",
    [(False, False, True), (True, True, True), (True, False, False)],
    ids=["no spec decode", "no overlap scheduler", "no kv cache manager"],
)
def test_uncorrected_steps_leave_the_inputs(spec, disable_overlap, with_kv_cache) -> None:
    metadata = _Metadata(2, with_kv_cache)
    inputs = _preparer(RuntimeHooks(), disable_overlap).preprocess_inputs(
        _inputs(metadata, 8), enable_spec_decode=spec, runtime_draft_len=3
    )
    assert inputs["position_ids"].tolist() == [list(range(8))]
    assert metadata.kv_lens_cuda.tolist() == [100, 101]
    assert metadata.calls == []
