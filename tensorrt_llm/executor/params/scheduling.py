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
from dataclasses import dataclass
from typing import List, Optional, Tuple

AgentHierarchy = List[Tuple[str, int]]


@dataclass(slots=True, kw_only=True)
class SchedulingParams:
    """Schedule parameters.

    Args:
        attention_dp_rank (int): The rank of target attention dp
        attention_dp_relax (bool): Whether to allow the request to be scheduled to other attention dp for better
            throughput. Defaults to True.
        agent_hierarchy (AgentHierarchy): Path of (agent_type, node_id) tuples
            identifying this request's position in an agent execution tree.
            Used by the batch scheduler for hierarchy-aware scheduling.
    """

    attention_dp_rank: Optional[int] = None
    attention_dp_relax: bool = True
    agent_hierarchy: Optional[AgentHierarchy] = None
