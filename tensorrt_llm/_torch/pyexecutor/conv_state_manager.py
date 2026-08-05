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
"""Capability protocol for per-request short-convolution state.

A model whose layers carry a short causal convolution needs the previous
``kernel_size - 1`` pre-conv inputs per request, across decode steps. That is
per-request state with the KV cache's lifetime, so the cache manager is where
it belongs -- the same conclusion ``BaseMambaCacheManager`` reached for mamba
conv/SSM state.

This is a separate protocol from ``BaseMambaCacheManager`` on purpose. That one
mandates ``get_ssm_states``, ``is_speculative`` and ``mamba_layer_cache``; a
model with convolutions but no selective-scan state would have to stub all
three, and a consumer that type-tested for it would be asserting something
untrue. Two capabilities, two protocols.

Consumers should test the capability, not the model:

    if isinstance(attn_metadata.kv_cache_manager, BaseConvStateManager):
        ...

which is the shape ``AttentionMetadata._prepare_mamba_metadata`` already uses,
so any manager that implements this protocol works without another branch.
"""

from abc import ABC, abstractmethod
from typing import Any, Sequence

import torch

from .kv_cache_manager_v2 import KVCacheManagerV2


class BaseConvStateManager(ABC):
    """Per-request short-conv state carried with the KV-cache request lifetime.

    Implementations are expected to be cache managers (``is-a`` KVCacheManager
    or KVCacheManagerV2), so request admission, per-request row allocation and
    release all follow the KV cache rather than running beside it. That is the
    point of the protocol: a pool that lives outside the cache manager can
    disagree with it about block reuse or request lifetime, and nothing forces
    the two views back into agreement.
    """

    @abstractmethod
    def get_conv_state_cache(self) -> Any:
        """The pool object the model's convolutions read and update in place.

        Returned rather than exposed per layer because the fused
        ``causal_conv1d_*`` ops mutate the buffers at the per-request slot
        indices, so the model needs the pool itself, not a copy.
        """

    @abstractmethod
    def prepare_conv_runtime(self, attn_metadata):
        """Publish this batch's pool rows and return ``(pool, runtime)``.

        Must write the resolved rows into a STABLE device buffer: a captured
        CUDA graph aliases that pointer, so a fresh allocation per step would
        strand the capture. Must run from the pre-forward metadata hook, never
        from inside a captured ``model.forward``, so every replay reads the
        current batch's rows rather than the capture-time ones.
        """

    @abstractmethod
    def free_conv_state(self, request_ids: Sequence[int]) -> None:
        """Release the pool rows owned by ``request_ids``.

        Called from the manager's own ``free_resources`` so a conv row cannot
        outlive the KV blocks of the same request -- a leaked row is later
        reused, with stale state, by whatever request next takes that slot.
        """


class InklingHybridCacheManager(KVCacheManagerV2, BaseConvStateManager):
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
        from ..models.modeling_inkling import InklingConvStateCache

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

    # ---- BaseConvStateManager -------------------------------------------------
    def get_conv_state_cache(self):
        return self._conv_cache

    def prepare_conv_runtime(self, attn_metadata):
        from ..models.modeling_inkling import InklingConvRuntime

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
