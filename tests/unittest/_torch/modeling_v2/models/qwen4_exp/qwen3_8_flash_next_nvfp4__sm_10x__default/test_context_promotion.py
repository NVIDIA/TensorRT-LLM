# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The default target's one-token prompt stays in prefill.

PLE seeds a fresh request's n-gram history and conv state in the context forward only, so the
decoder runner must not move the target's final one-token context into the decode CUDA graph.
"""

from types import SimpleNamespace

import pytest
import torch
from _torch.pyexecutor.engine.runners.decoder.utils import decoder_config, decoder_graphs
from _torch.pyexecutor.engine_test_utils import _make_request_stub

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_10x__default.modeling import (  # noqa: E501
    ModelingV2Qwen38FlashNextNvfp4Sm10xDefault,
)
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests

pytestmark = pytest.mark.cpu_only


def _promoted_request_ids(model: torch.nn.Module) -> frozenset[int]:
    graphs = decoder_graphs(decoder_config(), model=model)
    graphs.cuda_graph_runner.enabled = True
    batch = ScheduledRequests()
    batch.context_requests_last_chunk = [_make_request_stub(10, prompt_len=1)]
    _, promoted_ids = graphs.promote_final_contexts(
        batch,
        SimpleNamespace(context_swa_layer_limit=None),
        enable_spec_decode=False,
        runtime_draft_len=0,
        ced_enabled=False,
    )
    return promoted_ids


def test_the_runner_keeps_a_one_token_prompt_in_prefill() -> None:
    model = ModelingV2Qwen38FlashNextNvfp4Sm10xDefault.__new__(
        ModelingV2Qwen38FlashNextNvfp4Sm10xDefault
    )
    torch.nn.Module.__init__(model)

    assert _promoted_request_ids(torch.nn.Identity()) == frozenset({10})
    assert _promoted_request_ids(model) == frozenset()
