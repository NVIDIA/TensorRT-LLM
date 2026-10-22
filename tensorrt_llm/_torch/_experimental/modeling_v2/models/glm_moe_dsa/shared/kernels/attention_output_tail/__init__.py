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
"""attention_output_tail: local attention output tail (the attention_output_tail contract).

Replaces the GEN-phase W_UV BMM and the local ``o_proj`` GEMM of an MLA layer
with attention_output_tail-owned kernels (a Triton W_UV kernel and a CuTe DSL o_proj kernel that
streams its weights around its PDL wait) that keep the native BF16 rounding
of the intermediate and emit the ordinary rank-local BF16 partial
``[R, hidden]``. The native all-reduce consumes that partial unchanged; attention_output_tail
hides no communication.

Registered entry points (``dispatch.get_impl("attention_output_tail", name)``):

* ``make_attn_tail(mla) -> AttnTail | None``: per-layer adapter used by the
  MLA call sites (see ``tail.py``).
* ``attn_tail(latent, w_uv, w_o, num_rows, out_rows=None)``: the functional
  tail (see ``kernels.py``).
"""

from ... import dispatch
from .kernels import ROW_TILE, attn_tail, check_tail_shapes, check_weights
from .tail import AttnTail, make_attn_tail

dispatch.register("attention_output_tail", "make_attn_tail", make_attn_tail)
dispatch.register("attention_output_tail", "attn_tail", attn_tail)

__all__ = [
    "ROW_TILE",
    "AttnTail",
    "attn_tail",
    "check_tail_shapes",
    "check_weights",
    "make_attn_tail",
]
