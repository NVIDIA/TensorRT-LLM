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
"""Inkling's KV cache manager: paged KV plus the short-conv state pool.

Lives with the model's attention package rather than under ``pyexecutor``,
matching ``sparse/minimax_m3/cache_manager.py``.

There is deliberately no shared conv-state protocol. ``BaseMambaCacheManager``
is the closest existing one, but it mandates SSM state and replay metadata
Inkling cannot back, and its one-tensor-per-layer accessor cannot express
Inkling's four convs per layer at two different widths. If a second short-conv
model appears, widen that hook rather than adding another beside it.
"""

import torch

from ...pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from .conv_state import InklingConvRuntime, InklingConvStateCache


class InklingHybridCacheManager(KVCacheManagerV2):
    """Paged KV (V2, per-layer geometry) + the short-conv state pool.

    Folding the pool into the cache manager -- the shape
    ``CppMambaHybridCacheManager`` uses for mamba conv/SSM state -- lets it reach
    the model through the standard ``attn_metadata.kv_cache_manager`` field and
    be released by the manager's own ``free_resources``. The conv rows are then
    freed by the same call that frees the request's KV blocks, so the two views
    cannot drift apart.

    The cost is that the pool is also allocated for the throwaway manager built
    during KV-cache size estimation, and freed along with it.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pretrained_config = kwargs["pretrained_config"]
        mapping = kwargs["mapping"]
        max_batch_size = kwargs["max_batch_size"]
        # Not kwargs["dtype"] -- that is the KV cache dtype, a C++ binding type
        # torch.zeros rejects. The conv pool holds pre-conv activations, so it
        # takes the model's compute dtype from the (text) config.
        text_config = getattr(pretrained_config, "text_config", pretrained_config)
        conv_dtype = getattr(text_config, "torch_dtype", None)
        if not isinstance(conv_dtype, torch.dtype):
            conv_dtype = torch.bfloat16
        # The conv pool's k/v width follows the attention kv-head split, so it
        # takes the attention TP, not the global one -- the same rule
        # KVCacheManagerV2 applies to the paged pool. Dividing by the global
        # tp_size would allocate narrow conv rows for full-width convs.
        attn_tp_size = 1 if mapping.enable_attention_dp else mapping.tp_size
        # +1 row for the CUDA-graph padding / dummy-request slot (the mamba
        # pattern): a padded decode batch admits up to max_batch_size real
        # requests plus a shared dummy row.
        self._conv_cache = InklingConvStateCache(
            pretrained_config,
            attn_tp_size,
            max_batch_size + 1,
            torch.device("cuda", torch.cuda.current_device()),
            conv_dtype,
        )

    # ---- model-facing -----------------------------------------------------
    def prepare_conv_runtime(self, attn_metadata):
        return self._conv_cache, InklingConvRuntime.build(attn_metadata, self._conv_cache)

    def free_conv_state(self, request_ids) -> None:
        self._conv_cache.free(list(request_ids))

    # ---- KVCacheManagerV2 -----------------------------------------------------
    def free_resources(self, request, *args, **kwargs):
        """Release the conv row with the request's KV blocks.

        This is what lets the model engine's warmup/estimation dummy-batch
        cleanup drop its Inkling-specific branch: it already calls
        ``kv_cache_manager.free_resources(req)`` for every dummy request, and a
        leaked conv row would later be reused, with stale state, by a real
        request whose id collides with a dummy id.
        """
        rid = getattr(request, "py_request_id", None)
        if rid is not None:
            self.free_conv_state([rid])
        return super().free_resources(request, *args, **kwargs)
