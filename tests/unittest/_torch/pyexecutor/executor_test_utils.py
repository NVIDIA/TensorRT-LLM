# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.request_delivery import ResponseChannel
from tensorrt_llm._torch.pyexecutor.scheduler.waiting_queue import FCFSWaitingQueue
from tensorrt_llm.bindings.internal.batch_manager import ReqIdsSet
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs


def make_partial_executor(**attrs: Any) -> PyExecutor:
    """Build a PyExecutor with encoder-path state and caller-provided overrides."""
    executor = object.__new__(PyExecutor)
    defaults = TorchLlmArgs.model_fields
    state = dict(
        device_id=0,
        global_rank=0,
        dist=SimpleNamespace(rank=0, tp_size=1, pp_size=1, world_size=1),
        enable_attention_dp=False,
        model_engine=None,
        _num_embeddings=None,
        draft_model_engine=None,
        sampler=Mock(),
        resource_manager=Mock(),
        kv_cache_manager=None,
        _is_kv_manager_v2=False,
        kv_cache_transceiver=None,
        kv_connector_manager=None,
        _disagg_coordinator=None,
        _disagg_pp_termination_handler=None,
        active_requests=[],
        waiting_queue=FCFSWaitingQueue(),
        canceled_req_ids=[],
        inflight_req_ids=ReqIdsSet(),
        previous_batch=None,
        _pending_recompute_pause_ids=set(),
        _prefetched_request_ids=set(),
        _response_channel=ResponseChannel(),
        gather_all_responses=False,
        is_benchmark_disagg=False,
        is_encoder_decoder=False,
        encoder_launch_executor=None,
        pending_encoder_steps=[],
        encoder_batch_wait_iters_count=0,
        execution_stream=Mock(),
        encoder_stream=Mock(),
        _mm_encoder_item_scheduling_enabled=False,
        llm_args=None,
        max_batch_size=defaults["max_batch_size"].default,
        max_num_tokens=defaults["max_num_tokens"].default,
        max_input_len=defaults["max_input_len"].default,
        max_beam_width=defaults["max_beam_width"].default,
        batch_wait_timeout_iters=defaults["batch_wait_timeout_iters"].default,
        batch_wait_max_tokens_ratio=defaults["batch_wait_max_tokens_ratio"].default,
    )
    state.update(attrs)
    for name, value in state.items():
        setattr(executor, name, value)
    return executor
