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

from tensorrt_llm._torch.attention.backends.sparse.params import (
    get_indexer_mqa_logits_elem_budget,
    get_indexer_mqa_logits_workspace_bytes,
)


def test_workspace_bytes_match_runtime_element_budget(monkeypatch):
    monkeypatch.setenv("TLLM_INDEXER_MQA_LOGITS_ELEM_BUDGET", "1024")
    assert get_indexer_mqa_logits_elem_budget() == 1024
    assert get_indexer_mqa_logits_workspace_bytes() == 4096


def test_workspace_bytes_are_bounded_by_reachable_request_shape(monkeypatch):
    monkeypatch.setenv("TLLM_INDEXER_MQA_LOGITS_ELEM_BUDGET", str(1 << 31))
    assert get_indexer_mqa_logits_workspace_bytes(4096, 4096) == 64 * 1024 * 1024
