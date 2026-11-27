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
"""Host entry of the preparation of the sm_120 FP8-PV kernel: raw BF16 q, k, v into the operands of the attention.

The chain of ../prep/chain.py (make_host: its launches) with the V scan and V pass of this kernel (v_pass.py): the
smoothed E4M3 V, its means and its non-finite flags, next to Q', K' and their scales.
"""

import torch

from ...inputs import Compensation
from ..prep import chain
from .softmax import MIN_SCORE_LOG2
from .v_pass import V_THREADS, basis_kernel, v_kernel

_host = chain.make_host(basis_kernel, v_kernel, V_THREADS, 4)
_compiled = {}

# the buffers of a call whose addresses the host function takes after q, k, v, the rows, the basis addresses and
# the rotation, in its order
_HOST_BUFFERS = (
    "q4",
    "k4",
    "q_sf",
    "k_sf",
    "score_scale",
    "v8",
    "v_mean",
    "v8_scale",
    "v_flags",
    "vmax",
    "gmax",
    "prov",
    "done",
)


# the scratch of the chain at a shape on a device of nsm SMs for a rank and a bases kind, name -> (dtype, shape):
# that of both chains
scratch = chain.scratch


def run(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    buffers: dict[str, torch.Tensor],
    lens: torch.Tensor | None,
    comp: Compensation,
    scale_rule: str,
) -> None:
    """Fill the operands of the attention from raw q, k, v: Q'/K', score_scale (>= 2^MIN_SCORE_LOG2), smoothed V.
    lens, comp, scale_rule: as in chain.run."""
    chain.run(
        _host, _compiled, _HOST_BUFFERS, MIN_SCORE_LOG2, q, k, v, buffers, lens, comp, scale_rule
    )
