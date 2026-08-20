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

import inspect

from tensorrt_llm._torch.attention_backend import prims_ts


def test_block_sparse_public_api_uses_live_runtime_metadata() -> None:
    expected_exports = {
        "BlockSparseTSWrapper",
        "BlockSparsePagedTSWrapper",
        "block_sparse_attention",
        "block_sparse_attention_with_paged_kv_cache",
    }
    assert expected_exports <= set(prims_ts.__all__)

    plan_parameters = inspect.signature(prims_ts.BlockSparseTSWrapper.plan).parameters
    assert {
        "device",
        "max_blocks_per_row",
        "use_kv_valid_bits",
    } <= plan_parameters.keys()
    assert {
        "block_indptr",
        "block_indices",
        "kv_valid_bits",
    }.isdisjoint(plan_parameters)

    run_parameters = inspect.signature(prims_ts.BlockSparseTSWrapper.run).parameters
    assert {
        "block_indptr",
        "block_indices",
        "kv_valid_bits",
    } <= run_parameters.keys()
