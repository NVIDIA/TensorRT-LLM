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

from ....logger import logger
from ...pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from .conv_state import InklingConvState, InklingConvStateCache


def _resolve_conv_dtype(pretrained_config) -> torch.dtype:
    """The compute dtype the short-conv pool holds.

    Not the manager's ``dtype`` argument -- that is the KV cache dtype, a C++
    binding type ``torch.zeros`` rejects, and it is ``nvfp4``/``fp8`` on
    quantized releases while the conv pool holds unquantized pre-conv
    activations.

    HuggingFace configs carry ``torch_dtype`` as either a ``torch.dtype`` or its
    name (``"bfloat16"``), so both are accepted. An unresolvable value raises:
    the previous silent fall back to bfloat16 turned an fp16 checkpoint into a
    pool of the wrong dtype, which reaches the conv kernels as a dtype mismatch
    far from its cause.
    """
    config = getattr(pretrained_config, "text_config", pretrained_config)
    dtype = getattr(config, "torch_dtype", None)
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        resolved = getattr(torch, dtype, None)
        if isinstance(resolved, torch.dtype):
            return resolved
    raise ValueError(
        f"Inkling short-conv pool needs the model's compute dtype, but "
        f"torch_dtype={dtype!r} on {type(config).__name__} is not a torch dtype"
    )


class InklingHybridCacheManager(KVCacheManagerV2):
    """Paged KV (V2, per-layer geometry) + the short-conv state pool.

    Folding the pool into the cache manager -- the shape
    ``CppMambaHybridCacheManager`` uses for mamba conv/SSM state -- lets it reach
    the model through the standard ``attn_metadata.kv_cache_manager`` field and
    be released by the manager's own ``free_resources``. The conv rows are then
    freed by the same call that frees the request's KV blocks, so the two views
    cannot drift apart.

    The cost is that the pool is also allocated for the throwaway manager built
    during KV-cache size estimation, and freed along with it. That is what makes
    the pool's bytes show up in the peak-memory reading
    ``configure_kv_cache_capacity`` takes -- the pool is a plain torch
    allocation and no V2 byte quota knows about it -- so the budget handed to
    the serving manager already has one pool's worth subtracted. The accounting
    holds exactly because both pools are the same fixed size; see
    :class:`InklingConvStateCache`.
    """

    def __init__(self, *args, pretrained_config, mapping, max_batch_size, **kwargs):
        # The three arguments the pool needs are declared, not read back out of
        # ``**kwargs``. KVCacheManagerV2 takes ``mapping`` / ``max_batch_size``
        # keyword-only and absorbs ``pretrained_config`` into ``**kwargs``
        # without storing it, so subscripting kwargs worked only as long as
        # every caller passed all three by keyword: omitting one surfaced as a
        # bare KeyError from inside this constructor rather than as a TypeError
        # naming the parameter.
        super().__init__(
            *args,
            pretrained_config=pretrained_config,
            mapping=mapping,
            max_batch_size=max_batch_size,
            **kwargs,
        )
        # The conv pool's k/v width follows the attention kv-head split, so it
        # takes the attention TP, not the global one -- the same rule
        # KVCacheManagerV2 applies to the paged pool. Dividing by the global
        # tp_size would allocate narrow conv rows for full-width convs.
        attn_tp_size = 1 if mapping.enable_attention_dp else mapping.tp_size
        # One row per sequence that can be resident at once. Pipeline stages
        # each hold a microbatch, so the bound is max_batch_size * pp_size --
        # the same count MambaHybridCacheManagerV2 calls
        # ``_max_resident_sequences``. The padding and attention-DP rows are
        # reserved on top of this by the pool itself.
        num_request_slots = max_batch_size * mapping.pp_size
        spec_config = kwargs.get("spec_config")
        max_draft_len = int(getattr(spec_config, "max_draft_len", 0) or 0)
        self._conv_cache = InklingConvStateCache(
            pretrained_config,
            attn_tp_size,
            num_request_slots,
            torch.device("cuda", torch.cuda.current_device()),
            _resolve_conv_dtype(pretrained_config),
            reserve_attention_dp_slot=mapping.enable_attention_dp,
            max_draft_len=max_draft_len,
        )
        logger.info(
            f"Inkling short-conv state pool: {self._conv_cache.num_slots} rows "
            f"({num_request_slots} request + reserved), "
            f"{self._conv_cache.conv_state_bytes() / (1 << 20):.1f} MiB"
        )

    # ---- model-facing -----------------------------------------------------
    @property
    def conv_state_cache(self) -> InklingConvStateCache:
        """The short-conv state pool, for the metadata's per-step publication."""
        return self._conv_cache

    def get_conv_states(self, layer_idx: int) -> InklingConvState:
        """The four short-conv state buffers of ``layer_idx``.

        Named after ``BaseMambaCacheManager.get_conv_states`` on purpose: this
        is the same question asked of the same kind of manager. It cannot
        *implement* that hook, which returns one tensor per layer and cannot
        express Inkling's four convs at two widths -- widening the shared hook
        is the move if a second short-conv model appears.
        """
        return self._conv_cache.layer_state(layer_idx)

    def get_state_indices(self) -> torch.Tensor:
        """Pool rows of the current batch, in packed batch order."""
        return self._conv_cache.state_indices

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
