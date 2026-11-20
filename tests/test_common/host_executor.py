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

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager
from tensorrt_llm.mapping import Mapping


def make_host_executor(*, max_num_sequences: int = 8) -> PyExecutor:
    """Initialize a stopped, single-rank executor for a host orchestration test."""
    model_engine = SimpleNamespace(
        model=torch.nn.Identity(),
        uses_torch_compile=False,
        mm_encoder_item_scheduling_enabled=False,
        enable_in_graph_sampling=False,
        enable_attention_dp=False,
        enable_overlap_headroom=False,
        is_spec_decode=False,
        spec_config=None,
        get_max_num_sequences=lambda: max_num_sequences,
        warmup_scope=nullcontext,
        warmup=Mock(),
        llm_args=SimpleNamespace(
            kv_connector_config=None,
            max_stats_len=16,
            max_num_tokens=1024,
            print_iter_log=False,
            enable_client_io_during_gpu_wait=False,
            enable_iter_perf_stats=False,
            enable_iter_req_stats=False,
            iter_perf_stats_interval=1,
            stream_interval=1,
            return_perf_metrics=False,
            attention_dp_config=None,
            batch_wait_timeout_ms=0,
            batch_wait_timeout_iters=0,
            batch_wait_max_tokens_ratio=0,
        ),
    )
    dist = Mock(rank=0, tp_rank=0, pp_size=1, tp_size=1, cp_size=1, world_size=1, mapping=Mapping())
    with (
        patch.object(torch.cuda, "current_device", return_value=0),
        patch.object(torch.cuda, "current_stream", return_value=Mock()),
        patch.object(torch.cuda, "Stream", side_effect=Mock),
        patch.object(torch.cuda, "Event", side_effect=Mock),
        patch.object(torch.cuda, "stream", wraps=nullcontext),
        patch.object(PyExecutor, "_set_global_steady_clock_offset"),
    ):
        return PyExecutor(
            ResourceManager({}),
            scheduler=Mock(ctx_demote_uncached_tokens=0),
            model_engine=model_engine,
            sampler=None,
            dist=dist,
            max_num_sequences=max_num_sequences,
            start_worker=False,
        )
