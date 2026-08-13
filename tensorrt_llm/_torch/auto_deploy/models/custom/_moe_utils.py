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

"""Utilities for custom MoE model checkpoint compatibility."""


def unpack_packed_expert_weights(state_dict, prefix: str, num_experts: int):
    """Unpack transformers packed expert tensors into per-expert module weights.

    Some transformers MoE models store routed experts as:
      * experts.gate_up_proj: [num_experts, 2 * intermediate_size, hidden_size]
      * experts.down_proj: [num_experts, hidden_size, intermediate_size]

    The custom AD models keep per-expert Linear modules for torch_moe dispatch.
    """

    gate_up_key = prefix + "experts.gate_up_proj"
    if gate_up_key in state_dict:
        gate_up_proj = state_dict.pop(gate_up_key)
        for expert_idx in range(num_experts):
            gate_proj, up_proj = gate_up_proj[expert_idx].chunk(2, dim=0)
            state_dict[prefix + f"experts.{expert_idx}.gate_proj.weight"] = gate_proj
            state_dict[prefix + f"experts.{expert_idx}.up_proj.weight"] = up_proj

    down_key = prefix + "experts.down_proj"
    if down_key in state_dict:
        down_proj = state_dict.pop(down_key)
        for expert_idx in range(num_experts):
            state_dict[prefix + f"experts.{expert_idx}.down_proj.weight"] = down_proj[expert_idx]
