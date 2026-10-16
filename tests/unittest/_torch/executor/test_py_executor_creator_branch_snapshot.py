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

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.py_executor_creator import _branch_snapshot_ctx_chunk_config
from tensorrt_llm.llmapi.llm_args import ContextChunkingPolicy

pytestmark = pytest.mark.cpu_only

FCFS = ContextChunkingPolicy.FIRST_COME_FIRST_SERVED
FORCE = ContextChunkingPolicy.FORCE_CHUNK


RECORDS = SimpleNamespace(records_branch_snapshots=True)
NO_RECORDS = SimpleNamespace(records_branch_snapshots=False)


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
        (object(), (FCFS, 256), (FCFS, 256)),
        (None, None, None),
    ],
)
def test_branch_snapshot_ctx_chunk_config(
    manager: object, ctx_chunk_config: tuple | None, expected: tuple | None
) -> None:
    assert _branch_snapshot_ctx_chunk_config(ctx_chunk_config, manager, 32) == expected
