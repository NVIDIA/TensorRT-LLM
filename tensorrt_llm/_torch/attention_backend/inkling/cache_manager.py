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
matching ``sparse/minimax_m3/cache_manager.py``. Nothing about it is installed
into shared framework directories: ``_util`` selects this class the same way it
selects ``MiniMaxM3KVCacheManagerV2``, and ``InklingAttentionMetadata``
type-tests it directly.

There is deliberately no ``BaseConvStateManager`` protocol. Per-request
short-conv state is NOT new -- ``BaseMambaCacheManager`` already declares
``get_conv_states(layer_idx)`` and nemotron_h / qwen3_next / qwen3_5 implement
it -- but that protocol also mandates ``get_ssm_states``, ``is_speculative``,
``mamba_layer_cache`` and replay metadata, none of which Inkling can back, and
its one-tensor-per-layer accessor cannot express Inkling's four convs per layer
at two different widths (k/v follow the TP-sharded kv split; the post-attention
and post-MLP convs run replicated on the full residual stream).

A parallel protocol was tried and removed: both of its useful methods returned
Inkling's own pool and runtime types, so it abstracted nothing while putting an
Inkling-specific file under ``pyexecutor``. If a second short-conv model ever
appears, widen the framework's existing hook rather than inventing another one
beside it.
"""

import torch

from ...pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2


class InklingHybridCacheManager(KVCacheManagerV2):
    """Paged KV (V2, per-layer geometry) + the short-conv state pool.

    The pool used to be a separate ResourceManager registered under its own
    ResourceManagerType, published from three call sites inside
    ``PyTorchModelEngine``. Making it part of the cache manager -- the shape
    ``CppMambaHybridCacheManager`` uses for mamba conv/SSM state -- removes all
    of that: the pool reaches the model through
    ``attn_metadata.kv_cache_manager``, which is a standard AttentionMetadata
    field, and rows are released by the manager's own ``free_resources``, which
    every caller (including the warmup/estimation dummy-batch path) already
    invokes.

    It also removes a whole class of bug rather than just some code. A pool that
    lives beside the cache manager can disagree with it about block reuse or
    request lifetime and nothing forces the two views back together; here the
    conv rows are freed by the same call that frees the request's KV blocks, so
    they cannot drift apart.

    Note: because the manager owns the pool, the pool is now also allocated for
    the throwaway manager built during KV-cache size estimation. That is roughly
    66 layers x 4 convs x (max_batch+1) rows -- tens of MB, freed with the
    estimation manager -- and it buys the lifetime coupling above.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Imported here, not at module scope: modeling_inkling imports from
        # _torch.attention_backend and _torch.modules, and a top-level import
        # would close a cycle back through pyexecutor at model-load time.
        from ...models.modeling_inkling import InklingConvStateCache

        pretrained_config = kwargs["pretrained_config"]
        mapping = kwargs["mapping"]
        max_batch_size = kwargs["max_batch_size"]
        # NOT kwargs["dtype"]: that is the KV cache dtype, a C++
        # ``tensorrt_llm.bindings.DataType``, and torch.zeros rejects it. The
        # conv pool holds pre-conv activations, so it takes the model's compute
        # dtype from the (text) config.
        text_config = getattr(pretrained_config, "text_config", pretrained_config)
        conv_dtype = getattr(text_config, "torch_dtype", None)
        if not isinstance(conv_dtype, torch.dtype):
            conv_dtype = torch.bfloat16
        # +1 row for the CUDA-graph padding / dummy-request slot (the mamba
        # pattern): a padded decode batch admits up to max_batch_size real
        # requests plus a shared dummy row.
        self._conv_cache = InklingConvStateCache(
            pretrained_config,
            mapping.tp_size,
            max_batch_size + 1,
            torch.device("cuda", torch.cuda.current_device()),
            conv_dtype,
        )

    # ---- model-facing -----------------------------------------------------
    def prepare_conv_runtime(self, attn_metadata):
        from ...models.modeling_inkling import InklingConvRuntime

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
