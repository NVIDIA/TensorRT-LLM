# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""SimDistributed: mock Distributed for simulation mode.

Implements the Distributed ABC with no-op communication. All broadcast,
allgather, and barrier calls return immediately. This allows simulation
mode to run in a single process even with TP>1 mappings.
"""

from tensorrt_llm._torch.distributed.communicator import Distributed, ReduceOp
from tensorrt_llm.mapping import Mapping


class SimDistributed(Distributed):
    """Mock Distributed that no-ops all communication for sim mode.

    Properties (rank, tp_size, pp_size, etc.) are inherited from the
    Distributed base class which reads them from the mapping.
    Only communication methods are overridden to be no-ops.
    """

    def __init__(self, mapping: Mapping):
        super().__init__(mapping)

    def barrier(self):
        pass

    def tp_barrier(self):
        pass

    def broadcast(self, obj, root=0):
        return obj

    def allgather(self, obj, root=0):
        return [obj]

    def allreduce(self, obj, op: ReduceOp = ReduceOp.SUM):
        return obj

    def tp_broadcast(self, obj, root=0, **kwargs):
        return obj

    def cp_broadcast(self, obj, root=0, **kwargs):
        return obj

    def tp_allgather(self, obj):
        return [obj]

    def cp_allgather(self, obj):
        return [obj]

    def recv_object(self, src, tag=0):
        raise NotImplementedError(
            "SimDistributed does not support PP recv_object")

    def send_object(self, obj, dest, tag=0):
        raise NotImplementedError(
            "SimDistributed does not support PP send_object")

    def isend_object(self, obj, dest, tag=0):
        raise NotImplementedError(
            "SimDistributed does not support PP isend_object")
