# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import torch

from tensorrt_llm._torch.auto_deploy.custom_ops.attention.torch_backend_attention import (
    _update_kv_cache,
)


def test_update_kv_cache():
    K_D_HEAD = 4
    V_D_HEAD = 2
    MAX_BATCH_SIZE = 2
    MAX_SEQ_LEN = 4
    seq_length = 4
    n_heads = 3
    batch_size = 1

    # Initialize KV cache
    k_cache = torch.zeros(MAX_BATCH_SIZE, MAX_SEQ_LEN, n_heads, K_D_HEAD)
    v_cache = torch.zeros(MAX_BATCH_SIZE, MAX_SEQ_LEN, n_heads, V_D_HEAD)

    # Generate q,k,v test vectors
    k = torch.ones(batch_size, seq_length, n_heads, K_D_HEAD)
    v = torch.ones(batch_size, seq_length, n_heads, V_D_HEAD)

    print("k_cache: " + str(k_cache))
    print("v_cache: " + str(v_cache))
    print("input_pos: " + str(torch.tensor([0, 0])))
    print("slot_idx: " + str(torch.tensor([0, 1])))
    print("seq_start: " + str(torch.tensor([0, 3])))

    _update_kv_cache(
        k.view(batch_size * seq_length, n_heads, K_D_HEAD),
        v.view(batch_size * seq_length, n_heads, V_D_HEAD),
        k_cache,
        v_cache,
        torch.tensor([3, 1]).long(),
        torch.tensor([0, 0]),
        slot_idx=torch.tensor([0, 1]),
        seq_start=torch.tensor([0, 3]).long(),
    )

    print("k_cache: " + str(k_cache))
    print("v_cache: " + str(v_cache))
