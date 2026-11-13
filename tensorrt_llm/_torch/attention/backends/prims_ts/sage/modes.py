# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Names for how a Sage softmax obtains and applies the ``sfK`` scales.

Each attention kernel picks one transport and one scaling per plan and keeps
its own selection, resources and code (the context kernel's
``sage_k_transport`` and ``sage_score_scaling`` in
``kernels/fmha_context/sage_scales.py``, the decode kernel's
``FmhaDecodeConfig.sage_score_scaling``); only the names are shared.
"""

import enum


class SageKTransport(enum.Enum):
    """Where the softmax reads a K tile's ``sfK`` words from.

    The choice follows the number of words one K tile carries. The context
    kernel reads them from global memory when a 128-token tile has at most 32
    words (K blocks of four tokens or more) and stages the 128 words of
    one-token blocks in shared memory (``sage_k_transport``). The decode
    kernel stages every tile's words in shared memory, where its load warp
    also gathers the atoms of block-sparse routes (``SageKScalesResource``).
    """

    # The softmax lanes load the words from the prepared buffer in global
    # memory before the score wait.
    GMEM = "gmem"
    # The load warp stages the words in a shared-memory ring.
    SMEM = "smem"


class SageScoreScaling(enum.Enum):
    """How the softmax applies ``sfK`` to a K tile's scores."""

    # The scores stay raw; each block's maximum and exp2 multiplier carry
    # ``sfQ * sfK``.
    BLOCK = "block"
    # The max pass multiplies the scores by ``sfK`` in place.
    DEQUANTIZED = "dequantized"
