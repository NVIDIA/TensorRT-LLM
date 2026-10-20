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
"""The cache kinds, one class per cache variant.

Each class claims its manager classes (``managers``) and overrides only what its cache does
differently from :class:`~._base.StandardSpec`. Managers no kind claims by class are described as
``attention``; so are subclasses of a claimed manager that add nothing the transfer needs to know
(DSA's indexer-K is an index-key cache, replicated by default; GLM-5 Next is a ``mamba`` manager).
"""

from typing import FrozenSet, Optional

from tensorrt_llm._torch.disaggregation.base import CacheKind

from ..registry import register
from . import _paged, _state, _virtual_layers
from ._base import StandardSpec


@register
class AttentionSpec(StandardSpec):
    """Paged attention: MHA / GQA / MLA, full or sliding window, in a standard ``KVCacheManagerV2``.

    Every life cycle is one paged space. K/V are sharded by KV head and stored head-major, the
    layout the TRTLLM attention kernels write; index-key caches are replicated and move whole from
    one elected sender. MLA is the one-head, key-only case of the same layout.
    """

    name = "attention"
    managers = ("tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2.KVCacheManagerV2",)


@register
class MiniMaxM3Spec(AttentionSpec):
    """MiniMax M3: paged K/V whose layout follows the attention backend, plus replicated index-K.

    Ordinary K/V is stored head-major or token-major as the manager's backend selects
    (``_main_kv_layout_name``). V2 storage may coalesce index-K into the K/V pool when per-block
    sizes coincide; regions are per ``(layer, role)``, so each keeps its own layout either way.
    """

    name = "minimax_m3"
    managers = (
        "tensorrt_llm._torch.attention.backends.sparse.minimax_m3.cache_manager."
        "MiniMaxM3KVCacheManagerV2",
    )

    def kv_layout(self, manager) -> str:
        return manager._main_kv_layout_name()


@register
class Fp4MlaSpec(AttentionSpec):
    """FP4 MLA: NVFP4 latent cache in token-major blocks plus a replicated high-precision tail.

    K and its block scales are stored token-major; the high-precision tail is identical on every
    rank, so it is replicated. The physical ownership bridge for NVFP4 key-only caches applies to
    every V2 kind (``_ownership``).
    """

    name = "fp4_mla"
    managers = (
        "tensorrt_llm._torch.attention.backends.fp4_mla.cache_manager.Fp4MlaKVCacheManagerV2",
    )

    def replicated_roles(self, manager) -> FrozenSet[str]:
        from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role

        return super().replicated_roles(manager) | {str(Role.MLA_HP_TAIL)}

    def kv_layout(self, manager) -> str:
        return _paged.TOKEN_MAJOR


@register
class DeepseekV4Spec(AttentionSpec):
    """DeepSeek-V4: several attention caches (SWA, compressed, indexer) per model layer.

    Layers are virtual (see ``_virtual_layers``). NVFP4 alignment pages are storage-only roles the
    manager declares ignored: they keep their slot offsets but are not transferred.
    """

    name = "deepseek_v4"
    managers = (
        "tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.cache_manager."
        "DeepseekV4CacheManager",
    )

    def global_layer_ids(self, manager, cycle):
        return _virtual_layers.global_layer_ids(manager, cycle)

    def window_size(self, manager, cycle):
        return _virtual_layers.window_size(manager, cycle.internal_layer_ids[0])


@register
class Csa2Spec(AttentionSpec):
    """CSA2: compressed sparse attention caches, every role replicated across TP.

    Every CSA2 buffer (SWA, global, index, compressor state) is computed identically on every TP
    rank, so each one moves whole from one elected sender.

    Layers are virtual (see ``_virtual_layers``). A request may run in remote-tail mode, which the
    executor marks on the request:

    * ``py_csa2_remote_tail_start``: only the prompt up to this token is transferred (the
      generation side recomputes the tail), so it bounds the moved history;
    * ``py_csa2_remote_tail_split``: decoder SWA state of model layers at or past this layer is
      rebuilt by the generation side, so those regions are not moved;
    * ``py_csa2_remote_tail_mode == "source"``: the context side generates no first token to
      forward.
    """

    name = "csa2"
    managers = (
        "tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager.CSA2CacheManager",
    )

    def role_buffers(self, manager, unit, role, nbytes, *, heads, tokens_per_block, context):
        return (_paged.replicated(nbytes),)

    def global_layer_ids(self, manager, cycle):
        return _virtual_layers.global_layer_ids(manager, cycle)

    def window_size(self, manager, cycle):
        return _virtual_layers.window_size(manager, cycle.internal_layer_ids[0])

    def transfer_length(self, request) -> int:
        start = getattr(request, "py_csa2_remote_tail_start", None)
        return request.prompt_len if start is None else start

    def sends_first_tokens(self, request) -> bool:
        return getattr(request, "py_csa2_remote_tail_mode", None) != "source"

    def excluded_regions(self, manager, page_table, space_index, request) -> FrozenSet[int]:
        split_layer = getattr(request, "py_csa2_remote_tail_split", None)
        if split_layer is None:
            return frozenset()
        identities = {
            (int(layer_id), str(role)): identity
            for (layer_id, role), identity in manager._physical_roles.items()
        }
        excluded = set()
        for index, region in enumerate(page_table.spaces[space_index].regions):
            identity = identities.get((region.local_layer, region.role))
            if identity is None:
                continue
            model_layer, _ratio, cache_role = identity
            # ``cache_role`` is a ``CSA2CacheRole``; only decoder SWA state is rebuilt remotely.
            if cache_role is type(cache_role).SWA and model_layer >= split_layer:
                excluded.add(index)
        return frozenset(excluded)


@register
class MambaSpec(StandardSpec):
    """Hybrid recurrent models (Mamba2, GDN, KDA): attention life cycles plus recurrent state.

    Attention life cycles are paged as in ``attention``. Recurrent life cycles are state spaces: a
    request owns one slot holding every recurrent layer's state, moved once after the last chunk:

    * convolution state is sectioned (``[x | B | C]`` for Mamba2, ``[Q | K | V]`` for GDN), each
      section sharded by channel over the state grid on its own;
    * SSM state is sharded by head;
    * PLE side state (short-convolution state and n-gram context) is computed from replicated
      inputs, so every rank holds the same bytes;
    * index-key caches of hybrid sparse models (GLM-5 Next k-pool) are replicated, as for attention;
    * any other state role is held whole.
    """

    name = "mamba"
    managers = (
        "tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.MambaHybridCacheManagerV2",
    )

    def life_cycle_unit(self, manager, cycle) -> CacheKind:
        # Recurrent and attention layers never share a life cycle: their buffer configs differ,
        # so V2 groups them apart.
        recurrent = [manager._is_local_mamba_layer(int(lid)) for lid in cycle.internal_layer_ids]
        if any(recurrent) and not all(recurrent):
            raise ValueError(
                f"life cycle with layers {cycle.internal_layer_ids} mixes recurrent and attention "
                "layers"
            )
        return CacheKind.STATE if any(recurrent) else CacheKind.PAGED

    def replicated_roles(self, manager) -> FrozenSet[str]:
        from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaRole

        return super().replicated_roles(manager) | {
            str(MambaRole.PLE_CONV_STATE),
            str(MambaRole.PLE_NGRAM_CONTEXT),
        }

    def state_buffers(self, manager, role: str, nbytes: int, context: str):
        from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaRole

        if role == str(MambaRole.CONV_STATE):
            d_conv_m1 = int(manager.conv_state_shape[1])
            elem = manager.conv_state_dtype.itemsize
            sections = [int(dim) * d_conv_m1 * elem for dim in manager.conv_section_dims]
            return _state.sectioned(sections, nbytes, context)
        if role == str(MambaRole.SSM_STATE):
            _, head_dim, d_state = manager.ssm_state_shape
            bytes_per_head = int(head_dim) * int(d_state) * manager.ssm_state_dtype.itemsize
            return _state.by_head(nbytes, bytes_per_head, context)
        return super().state_buffers(manager, role, nbytes, context)

    def state_slot(self, manager, request) -> Optional[int]:
        if manager.local_num_mamba_layers == 0:
            return None
        return manager._request_id_to_state_index[request.py_request_id]


@register
class QsaSpec(MambaSpec):
    """QSA hybrid models: ``mamba`` plus per-request index positions.

    Index positions are request-wide coordinates, identical on every rank, so they are replicated
    and one elected sender moves them. They carry no head axis: sharding them by KV head would only
    line up when both peers shard heads the same way.
    """

    name = "qsa"
    managers = (
        "tensorrt_llm._torch.attention.backends.sparse.qsa.cache_manager."
        "QSAMambaHybridCacheManagerV2",
    )

    def replicated_roles(self, manager) -> FrozenSet[str]:
        from tensorrt_llm._torch.attention.backends.sparse.qsa.cache_manager import (
            QSA_INDEX_POSITION,
        )

        return super().replicated_roles(manager) | {str(QSA_INDEX_POSITION)}
