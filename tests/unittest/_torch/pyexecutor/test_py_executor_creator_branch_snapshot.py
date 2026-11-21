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

from unittest.mock import Mock, patch

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManagerV2
from tensorrt_llm._torch.pyexecutor.py_executor_creator import _snapshot_ctx_chunk_config
from tensorrt_llm.llmapi.llm_args import ContextChunkingPolicy

pytestmark = pytest.mark.cpu_only

FCFS = ContextChunkingPolicy.FIRST_COME_FIRST_SERVED
FORCE = ContextChunkingPolicy.FORCE_CHUNK


RECORDS = Mock(
    spec=KVCacheManagerV2, records_branch_snapshots=True, chunks_at_snapshot_points=False
)
NO_RECORDS = Mock(
    spec=KVCacheManagerV2, records_branch_snapshots=False, chunks_at_snapshot_points=False
)
# A recurrent-state manager under block reuse, also one a model declares for a configuration
# `is_hybrid_linear` does not recognize.
RECURRENT = Mock(
    spec=KVCacheManagerV2, records_branch_snapshots=False, chunks_at_snapshot_points=True
)


@pytest.mark.parametrize(
    ("manager", "ctx_chunk_config", "expected"),
    [
        # Branch points need forced boundaries; the SWA chunk unit is kept.
        (RECORDS, (FCFS, 256), (FORCE, 256)),
        (RECORDS, None, (FORCE, 32)),
        # Without branch points, the configured chunking is kept, including
        # none at all. This covers joint target/draft reuse, models without SWA
        # layers, a V1 manager, and no manager.
        (NO_RECORDS, (FCFS, 256), (FCFS, 256)),
        (NO_RECORDS, None, None),
        (Mock(records_branch_snapshots=True), (FCFS, 256), (FCFS, 256)),
        (None, None, None),
        # Recurrent-state snapshots need forced boundaries as well.
        (RECURRENT, (FCFS, 256), (FORCE, 256)),
        (RECURRENT, None, (FORCE, 32)),
    ],
)
def test_snapshot_ctx_chunk_config(
    manager: object, ctx_chunk_config: tuple | None, expected: tuple | None
) -> None:
    assert _snapshot_ctx_chunk_config(ctx_chunk_config, manager, 32) == expected


@pytest.mark.parametrize(
    ("ctx_chunk_config", "configured_policy", "warned"),
    [
        # Chunked prefill was off, so forcing chunks is reported.
        (None, None, True),
        # A user-set policy is replaced and reported.
        ((FCFS, 256), FCFS, True),
        # The default policy and an explicit FORCE_CHUNK are not reported.
        ((FCFS, 256), None, False),
        ((FORCE, 32), FORCE, False),
    ],
)
@pytest.mark.parametrize(
    ("manager", "reason"),
    [(RECORDS, "enable_branch_snapshot"), (RECURRENT, "block reuse of recurrent state")],
    ids=["branch_points", "recurrent_state"],
)
def test_snapshot_ctx_chunk_config_reports_overrides(
    ctx_chunk_config: tuple | None,
    configured_policy: ContextChunkingPolicy | None,
    warned: bool,
    manager: object,
    reason: str,
) -> None:
    with patch("tensorrt_llm._torch.pyexecutor.py_executor_creator.logger") as logger:
        _snapshot_ctx_chunk_config(ctx_chunk_config, manager, 32, configured_policy)
    assert logger.warning_once.called == warned
    if warned:
        assert reason in logger.warning_once.call_args.args[0]


def test_only_attention_layers_need_no_chunking_at_snapshot_points() -> None:
    manager = object.__new__(KVCacheManagerV2)
    assert not manager.chunks_at_snapshot_points


@pytest.mark.parametrize(
    ("enable_block_reuse", "is_draft", "mamba_layer_mask", "expected"),
    [
        (True, False, [True, False], True),
        (False, False, [True, False], False),
        # A draft pool's recurrent layers belong to its target.
        (True, True, [True, False], False),
        (True, False, [False, False], False),
    ],
)
def test_a_hybrid_manager_chunks_at_snapshot_points_under_block_reuse(
    enable_block_reuse: bool, is_draft: bool, mamba_layer_mask: list, expected: bool
) -> None:
    manager = object.__new__(MambaHybridCacheManagerV2)
    manager.enable_block_reuse = enable_block_reuse
    manager.is_draft = is_draft
    manager._mamba_layer_mask = mamba_layer_mask
    assert manager.chunks_at_snapshot_points == expected
