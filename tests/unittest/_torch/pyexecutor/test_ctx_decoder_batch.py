# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Context-tail batching requires a drafter that rebuilds its final window."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm.llmapi.llm_args import ContextChunkingPolicy

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    "has_spec,final_window_draft,eligible",
    [(False, False, True), (True, False, False), (True, True, True)],
    ids=["target-only", "full-history-draft", "final-window-draft"],
)
def test_context_decoder_batching_requires_a_complete_final_window_draft(
    has_spec, final_window_draft, eligible
):
    from tensorrt_llm._torch.pyexecutor.model_engine import ModelEngine
    from tensorrt_llm._torch.pyexecutor.scheduler.ctx_decoder_batch import install_ctx_decoder_batch
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler

    engine = object.__new__(ModelEngine)
    engine.model = SimpleNamespace(
        model=SimpleNamespace(decoder_replay_window=140, decoder_replay_split=20),
        draft_model=SimpleNamespace(
            supports_final_window_context_replay=lambda _: final_window_draft
        ),
    )
    engine.spec_config = object() if has_spec else None
    scheduler = Mock(
        spec=KVCacheV2Scheduler,
        chunking_enabled=True,
        chunking_policy=ContextChunkingPolicy.FIRST_COME_FIRST_SERVED,
        kv_cache_manager=object(),
    )
    executor = SimpleNamespace(
        scheduler=scheduler,
        model_engine=engine,
        enable_attention_dp=True,
        kv_cache_transceiver=object(),
        dist=SimpleNamespace(tp_size=2, pp_size=1),
    )
    batcher = install_ctx_decoder_batch(executor)
    assert (batcher is not None) == eligible
