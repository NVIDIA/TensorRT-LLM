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
"""oproj_weight_staging: weight staging inside already-waiting kernels, plus cache-policy hints.

Site ``oproj_weight_staging`` (attention window, hosted by attention_output_tail's o_proj): the
served attention_output_tail/attention_output_into_hop tail streams part of the local ``W_o`` into
shared memory before its PDL wait and the rest after it; oproj_weight_staging stages the after-wait
rows before the wait, from a kernel that is already resident and waiting (the o_proj itself, or
attention_output_tail's W_UV kernel), with L2 cache-policy hints (staging prefetch normal or
``evict_last``, the consumer's copies optionally ``evict_first``). The
arithmetic is unchanged, so the partial is bit-identical to attention_output_tail's. No launch
and no stream is added.

Registered entry points (``dispatch.get_impl("oproj_weight_staging", name)`` after this
package is imported):

* ``wrap_attn_tail(mla, tail) -> AttnTail | None``: wraps the tail that attention_output_tail's
  ``make_attn_tail`` created for an MLA module (see ``tail.py``).
* ``stage_tail(...)``: the functional tail (see ``kernels.py``).
"""

from ... import dispatch
from .kernels import DEFAULT_STAGE, StageConfig, stage_tail
from .tail import FEATURE_ID, StagedAttnTail, wrap_attn_tail

dispatch.register(FEATURE_ID, "wrap_attn_tail", wrap_attn_tail)
dispatch.register(FEATURE_ID, "stage_tail", stage_tail)

__all__ = [
    "DEFAULT_STAGE",
    "FEATURE_ID",
    "StageConfig",
    "StagedAttnTail",
    "stage_tail",
    "wrap_attn_tail",
]
