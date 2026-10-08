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
"""fp32 dense reference for attention over a causal K/V cache.

Independent of the cache: it keeps every token ever committed in plain tensors and
selects the visible ones by position, so a cache that reads one key too many or too
few disagrees with it.
"""

import torch
import torch.nn.functional as F


def reference_attention(q: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Dense fp32 attention of ``q`` over an already visible-filtered key/value set.

    ``q`` is ``[T, H, D]``; ``keys``/``values`` are ``[S, H_kv, D]``. Grouped-query heads
    are expanded; the math runs in fp32 and the result comes back in ``q``'s dtype.
    """
    rep = q.shape[1] // keys.shape[1]
    kx = keys.repeat_interleave(rep, dim=1)
    vx = values.repeat_interleave(rep, dim=1)
    out = F.scaled_dot_product_attention(
        q.transpose(0, 1).float().unsqueeze(0),
        kx.transpose(0, 1).float().unsqueeze(0),
        vx.transpose(0, 1).float().unsqueeze(0),
    )
    return out.squeeze(0).transpose(0, 1).to(q.dtype)


def exact_reference(q, pk, pv, hk, hv, k, v, start, end, window):
    """Attention of staged tokens ``[start, end)`` over exactly what the window allows.

    Visible: the pinned prefix ``pk``/``pv``, the ``window`` keys before ``start`` and
    the staged tokens ``k``/``v`` up to ``end``. ``hk``/``hv`` are every history token
    committed so far, oldest first.
    """
    keys = torch.cat([pk, hk, k[:end]])
    values = torch.cat([pv, hv, v[:end]])
    pos = torch.arange(keys.shape[0], device=keys.device)
    visible = (pos < pk.shape[0]) | (pos >= pk.shape[0] + hk.shape[0] + start - window)
    return reference_attention(q[start:end], keys[visible], values[visible])
