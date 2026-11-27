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
"""Host entry of the preparation of the sm_120 NVFP4-PV kernel: raw BF16 q, k, v into the operands of kernel.py.

The chain of ../prep/chain.py (make_host: its launches) with the V scan and V pass of this kernel (v_pass.py); K', V',
their scales and the means go to the tile images of the attention (layout.py).
"""

import torch

from ..prep import chain
from .layout import SK_OFF, SKSF_OFF, STAGE
from .softmax import MIN_SCORE_LOG2
from .v_pass import V_THREADS, basis_kernel, v_kernel

_host = chain.make_host(basis_kernel, v_kernel, V_THREADS, 2, k_image=(SK_OFF, SKSF_OFF, STAGE))
_compiled = {}

# the buffers of a call whose addresses the host function takes after q, k, v, the rows and the rotation, in its order
_HOST_BUFFERS = (
    "q4",
    "img",
    "q_sf",
    "img",
    "score_scale",
    "img",
    "v4_scale",
    None,
    None,
    "ut",
    "vmax",
    "gmax",
    "prov",
    "done",
)


# the scratch of the chain at a shape on a device of nsm SMs, name -> (dtype, shape): that of both chains
scratch = chain.scratch


def run(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rows: torch.Tensor,
    buffers: dict[str, torch.Tensor],
    lens: torch.Tensor | None = None,
    map_rows: bool = False,
) -> None:
    """Fill the operands of the attention kernel from raw BF16 q, k, v [B, S, H, 128], in the buffers of the call.

    * Q' (q4, q_sf), score_scale (floored at 2^MIN_SCORE_LOG2), K', V' and the means (img, v4_scale).
    * rows: the sampled tokens of the Q/K basis (inputs.sample_rows); lens, map_rows: as in chain.run.
    """
    chain.run(
        _host, _compiled, _HOST_BUFFERS, MIN_SCORE_LOG2, q, k, v, rows, buffers, lens, map_rows
    )
