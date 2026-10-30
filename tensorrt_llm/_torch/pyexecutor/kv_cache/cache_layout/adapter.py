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
"""A cache manager's page table and what a transfer of a request moves, read off the manager
(``CacheLayoutAdapter``), and the adapter of each manager class (``adapter_for``)."""

from __future__ import annotations

import importlib
from dataclasses import replace
from typing import TYPE_CHECKING, Iterable, Optional

from ..kv_cache_manager_v2 import Role
from .page_layout import (
    BufferDesc,
    LayerGroupDesc,
    PageLayout,
    Tensor,
    content_shard,
    grid_place,
    head_major,
    held_bytes,
    owned,
    token_major,
)
from .page_table import (
    DevicePool,
    ManagerLayout,
    buffer_expansions,
    has_sparse_buffers,
    is_non_required_layer,
    is_private_layer,
    is_recurrent_layer,
    layer_configs_by_id,
    rank_topology,
    virtual_layers,
)

if TYPE_CHECKING:
    from tensorrt_llm.runtime.kv_cache_manager_v2 import AttentionLayerConfig, SsmLayerConfig

    from ...llm_request import LlmRequest
    from ..kv_cache_manager_v2 import KVCacheManagerV2

# The roles the base manager registers and splits by KV head (``buffer_tensors``).
_HEAD_SPLIT_ROLES = frozenset(
    str(role) for role in (Role.KEY, Role.VALUE, Role.KEY_BLOCK_SCALE, Role.VALUE_BLOCK_SCALE)
)


class CacheLayoutAdapter:
    """What a ``KVCacheManagerV2``'s pages hold and what a transfer of a request moves, read off
    the manager for whoever lends or moves its pages (the lender, the disaggregated transfer), which
    then never look into the manager.

    This adapter describes the base manager. A manager whose caches lay out or move otherwise has
    an adapter, in ``adapters``, overriding the methods whose answer differs, and an entry naming it
    in ``adapter_for``'s table.
    """

    def __init__(self, manager: KVCacheManagerV2) -> None:
        self.manager = manager

    # -- the page table -------------------------------------------------------------------------

    def page_table(self) -> ManagerLayout:
        """The manager's page table: its page layout, device pools and the facts of its layer
        groups.

        Built from the manager's runtime config and storage and the methods below: how each
        buffer's bytes spread over ranks (``buffer_tensors``), how layers are named
        (``layer_ids``), the KV heads (``kv_heads``), which roles never travel because they hold
        nothing a peer needs (``untransferred_roles``) or the receiver rebuilds them
        (``derived_roles``), which it rebuilds per request instead of receiving them
        (``request_rebuilt_roles``, with ``untransferred_buffers``) and the quantization
        (``quant_mode``). ``ValueError`` if the manager's layer groups, layer configs and these
        declarations do not agree."""
        m = self.manager
        tokens_per_block = int(m.tokens_per_block)
        layer_cfg = self.layer_configs()
        grouping = [[int(layer) for layer in group] for group in m.impl.layer_grouping]
        pg_of_lg = [int(g) for g in m.impl.get_life_cycle_pool_group_indices()]
        if len(pg_of_lg) != len(grouping):
            raise ValueError(f"{len(grouping)} layer groups but {len(pg_of_lg)} pool-group indices")
        layer_ids = self.layer_ids()
        missing = sorted({lid for group in grouping for lid in group} - set(layer_ids))
        if missing:
            raise ValueError(f"the adapter names no global layer for internal layers {missing}")
        global_of = {lid: int(ids[0]) for lid, ids in layer_ids.items()}
        model_of = {lid: int(ids[1]) for lid, ids in layer_ids.items()}
        stage_layers = {int(layer) for layer in m.pp_layers}
        unheld = sorted({model_of[lid] for group in grouping for lid in group} - stage_layers)
        if unheld:
            raise ValueError(
                f"the adapter names model layers {unheld}, which this pipeline stage does not "
                f"hold (pp_layers {sorted(stage_layers)})"
            )
        kv_heads = {lid: self.kv_heads(lid) for group in grouping for lid in group}

        state_groups = set()
        for lg, group in enumerate(grouping):
            if not group or group[0] not in layer_cfg:
                raise ValueError(f"layer group {lg} has no layer config (layers {group})")
            if is_recurrent_layer(layer_cfg[group[0]]):
                state_groups.add(lg)

        derived = self.derived_roles()
        untransferred = self.untransferred_roles() | derived
        expansions = {
            lid: buffer_expansions(cfg, tokens_per_block) for lid, cfg in layer_cfg.items()
        }

        topology = rank_topology(m.mapping)
        cp_rank, cp_size = int(m._helix_cp_rank), int(m._helix_cp_size)

        device_pools: dict[int, tuple[DevicePool, ...]] = {}
        buffers: list[BufferDesc] = []
        # (global layer, role) -> the internal layer whose buffer it names.
        named: dict[tuple[int, str], int] = {}
        group_buffers: dict[int, list[BufferDesc]] = {lg: [] for lg in range(len(grouping))}
        for pg in m.impl.pool_group_descs:
            g = int(pg.pool_group_index)
            pools = tuple(
                DevicePool(
                    g, int(p.pool_index), int(p.base_address), int(p.slot_bytes), int(pg.num_slots)
                )
                for p in pg.pools
            )
            device_pools[g] = tuple(sorted(pools, key=lambda p: p.index))
            # One variant per layer group drawing from this pool group; pool ``i`` of a slot holds
            # the ``i``-th coalesced buffer, whose members sit back to back.
            for variant in pg.slot_desc.variants:
                lg = int(variant.layer_group_id)
                state = lg in state_groups
                for pool_index, coalesced in enumerate(variant.coalesced_buffers):
                    size = int(coalesced.single_buffer_size)
                    for j, buffer_id in enumerate(coalesced.buffer_ids):
                        lid = int(buffer_id.layer_id)
                        role = str(buffer_id.role)
                        transfer = role not in untransferred
                        expansion = expansions.get(lid, {}).get(role, 1)
                        try:
                            tensors = self.buffer_tensors(
                                lid, role, size, tokens_per_block // expansion, state
                            )
                            declared = sum(held_bytes(t, topology) for t in tensors)
                        except ValueError as error:
                            raise ValueError(f"buffer ({lid}, {role}): {error}") from error
                        if declared != size:
                            # Bytes no tensor describes would never move, or be moved twice.
                            raise ValueError(
                                f"buffer ({lid}, {role}): its tensors hold {declared} bytes on "
                                f"this rank, not the {size} bytes of its slot"
                            )
                        if (
                            expansion != 1
                            and transfer
                            and role not in derived
                            and any(t.split and t.name != "own" for t in tensors)
                        ):
                            # Its tensors describe one page; ranks holding parts of several
                            # sub-pages would pair bytes of different sub-pages. A rank's own
                            # share pairs only with the same rank's, whole.
                            raise ValueError(
                                f"buffer ({lid}, {role}): {expansion} sub-pages per block that "
                                "move are declarable only replicated or as each rank's own share"
                            )
                        owner = named.setdefault((global_of.get(lid, lid), role), lid)
                        if owner != lid:
                            raise ValueError(
                                f"internal layers {owner} and {lid} both name the {role} buffer "
                                f"of global layer {global_of.get(lid, lid)}"
                            )
                        desc = BufferDesc(
                            layer=global_of.get(lid, lid),
                            role=role,
                            pool_group=g,
                            pool_index=pool_index,
                            slot_offset=j * size,
                            size=size,
                            tensors=tensors,
                            transferred=transfer,
                            expansion=expansion,
                        )
                        buffers.append(desc)
                        group_buffers[lg].append(desc)

        rebuilt = sorted({b.role for b in buffers if b.role in derived})
        if rebuilt:
            self._check_rebuild(rebuilt)
        per_request = sorted(self.request_rebuilt_roles())
        if (
            per_request
            and type(self).untransferred_buffers is CacheLayoutAdapter.untransferred_buffers
        ):
            raise ValueError(
                f"{type(self).__name__} rebuilds its {per_request} buffers per request but does "
                "not override untransferred_buffers, which names the requests it rebuilds them "
                "for, so a peer lacking them would leave them unwritten"
            )

        layer_groups: list[LayerGroupDesc] = []
        total_kv_heads: dict[int, int] = {}
        for lg, group in enumerate(grouping):
            first = layer_cfg[group[0]]
            is_state = lg in state_groups
            window = read = None
            sink_tokens = 0
            if not is_state:
                window = getattr(first, "window_size", None)
                window = None if window is None else int(window)
                read = m._attention_read_window(lg)
                read = None if read is None else int(read)
                if (read is None) != (window is None) or (
                    read is not None and not 0 < read <= window
                ):
                    raise ValueError(
                        f"layer group {lg} reads a window of {read} tokens, which its kept window "
                        f"of {window} tokens does not hold"
                    )
                sink_tokens = int(getattr(first, "num_sink_tokens", None) or 0)
                for lid in group:
                    total = kv_heads[lid][1]
                    if total:
                        total_kv_heads[global_of[lid]] = total
            # Layer groups with equal slot sizes may share a pool group, each drawing its own
            # slots.
            g = pg_of_lg[lg]
            if g not in device_pools:
                raise ValueError(f"layer group {lg} maps to pool group {g}, which has no pools")
            named_layers = sorted((global_of[lid], model_of[lid]) for lid in group)
            layer_groups.append(
                LayerGroupDesc(
                    kind="state" if is_state else "attention",
                    kept_window=window,
                    num_sink_blocks=-(-sink_tokens // tokens_per_block),
                    pool_group=g,
                    layers=tuple(layer for layer, _ in named_layers),
                    # The lender lends derived buffers too, so its names cover their bytes.
                    shard=content_shard(
                        [
                            replace(b, transferred=True) if b.role in derived else b
                            for b in group_buffers[lg]
                        ],
                        topology,
                    ),
                    read_window=read,
                    model_layers=tuple(model_layer for _, model_layer in named_layers),
                )
            )

        page_layout = PageLayout(
            tokens_per_page=tokens_per_block,
            pool_slot_bytes={
                g: tuple(p.slot_bytes for p in pools) for g, pools in device_pools.items()
            },
            layer_groups=tuple(layer_groups),
            buffers=tuple(buffers),
            topology=topology,
            total_kv_heads_by_layer=total_kv_heads,
            dtype=self.dtype(),
            quant_mode=self.quant_mode(),
            rebuilds_untransferred=bool(rebuilt),
            request_rebuilt_roles=self.request_rebuilt_roles(),
        )
        return ManagerLayout(
            page_layout=page_layout,
            tokens_per_block=tokens_per_block * cp_size,
            pool_group_of=tuple(pg_of_lg),
            windows=tuple(desc.kept_window for desc in layer_groups),
            sink_blocks=tuple(desc.num_sink_blocks for desc in layer_groups),
            recurrent=tuple(desc.kind == "state" for desc in layer_groups),
            sparse=tuple(has_sparse_buffers(layer_cfg[group[0]]) for group in grouping),
            layers=tuple(desc.layers for desc in layer_groups),
            shards=tuple((desc.shard.count, desc.shard.index) for desc in layer_groups),
            page_bytes={g: sum(p.slot_bytes for p in pools) for g, pools in device_pools.items()},
            device_pools=device_pools,
            cp=(cp_rank, cp_size),
            read_windows=tuple(desc.read_window for desc in layer_groups),
            named_groups=tuple(
                not any(is_private_layer(layer_cfg[lid]) for lid in group) for group in grouping
            ),
            prefix_required=tuple(
                not any(is_non_required_layer(layer_cfg[lid]) for lid in group)
                for group in grouping
            ),
            local_layers=tuple(tuple((lid, global_of[lid]) for lid in group) for group in grouping),
            kv_heads=tuple(kv_heads[group[0]][0] for group in grouping),
        )

    def _check_rebuild(self, rebuilt: list[str]) -> None:
        """Refuse derived buffers no ``rebuild_imported`` of this adapter rebuilds: the base's
        rebuilds nothing, and one from a parent above the adapter adding derived roles does not
        know the roles it adds."""
        if type(self).rebuild_imported is CacheLayoutAdapter.rebuild_imported:
            raise ValueError(
                f"{type(self).__name__} derives its {rebuilt} buffers from the transferred ones "
                "but does not override rebuild_imported, so an import would leave them stale"
            )
        rebuild_owner = _defined_by(type(self), "rebuild_imported")
        roles_owner = _defined_by(type(self), "derived_roles")
        if roles_owner is not rebuild_owner and issubclass(roles_owner, rebuild_owner):
            added = sorted(set(rebuilt) - set(super(roles_owner, self).derived_roles()))
            if added:
                raise ValueError(
                    f"{roles_owner.__name__} derives its {added} buffers, but its rebuild comes "
                    f"from {rebuild_owner.__name__}, which does not know them; override "
                    "rebuild_imported with them"
                )

    def layer_configs(self) -> dict[int, AttentionLayerConfig | SsmLayerConfig]:
        """Internal layer id -> the layer's config, as the manager configured its runtime."""
        return layer_configs_by_id(self.manager.kv_cache_manager_py_config)

    def layer_ids(self) -> dict[int, tuple[int, int]]:
        """Internal layer id -> (global layer id, model layer), the same under any pipeline split:
        the model layer itself, or ``model_layer * num_types + type`` for a manager keeping several
        caches per model layer (``virtual_layers``). An adapter for a manager whose internal layers
        are numbered otherwise overrides this."""
        m = self.manager
        internal = sorted({int(lid) for group in m.impl.layer_grouping for lid in group})
        virtual = self.virtual_layers()
        if virtual is None:
            unnamed = [lid for lid in internal if not 0 <= lid < len(m.pp_layers)]
            if unnamed:
                raise ValueError(
                    f"internal layers {unnamed} are no local layers, and the adapter names no "
                    "model layer for them (virtual_layers, layer_ids)"
                )
            # Internal layer i is the stage's i-th model layer, as layer_offsets numbers it.
            offsets = getattr(m, "layer_offsets", None)
            if offsets is not None:
                renumbered = [lid for lid in internal if offsets.get(int(m.pp_layers[lid])) != lid]
                if renumbered:
                    raise ValueError(
                        f"internal layers {renumbered} are not the local index layer_offsets gives "
                        "their model layers; an adapter for a manager numbering its layers "
                        "otherwise names them (_layer_attn_to_layer_id, layer_ids)"
                    )
            return {lid: (int(m.pp_layers[lid]),) * 2 for lid in internal}
        types_of, num_types = virtual
        unnamed = [lid for lid in internal if lid not in types_of]
        if unnamed:
            raise ValueError(
                f"internal layers {unnamed} have no (model layer, type) in virtual_layers"
            )
        return {
            lid: (types_of[lid][0] * num_types + types_of[lid][1], types_of[lid][0])
            for lid in internal
        }

    def virtual_layers(self) -> tuple[dict[int, tuple[int, int]], int] | None:
        """(internal id -> (model layer, attention type), type count) of a manager keeping several
        caches per model layer, read from its (model layer, type) -> internal id map
        ``_layer_attn_to_layer_id`` (``page_table.virtual_layers``): the base manager's retained
        precision twins keep one, as DeepSeek-V4 and CSA2 do. None: one cache per model layer."""
        return virtual_layers(getattr(self.manager, "_layer_attn_to_layer_id", None))

    def kv_heads(self, layer: int) -> tuple[int, int | None]:
        """(KV heads on this rank, KV heads over all tensor-parallel ranks or None if unknown) of
        internal layer ``layer``: those of its model layer (``layer_ids``), 0 heads on this rank
        for a model layer this pipeline stage does not hold."""
        m = self.manager
        model_layer = int(self.layer_ids()[layer][1])
        pp_layers = [int(layer) for layer in m.pp_layers]
        per_layer = list(m.num_kv_heads_per_layer or ())
        index = pp_layers.index(model_layer) if model_layer in pp_layers else len(per_layer)
        per_rank = int(per_layer[index] or 0) if index < len(per_layer) else 0
        total = m.num_kv_heads
        if isinstance(total, int):
            return per_rank, int(total)
        if total is None or model_layer >= len(total) or total[model_layer] is None:
            return per_rank, None
        return per_rank, int(total[model_layer])

    def buffer_tensors(
        self, layer_id: int, role: str, size: int, tokens: int, state: bool
    ) -> tuple[Tensor, ...]:
        """How the bytes of internal layer ``layer_id``'s ``role`` buffer spread over ranks: the
        tensors whose local pieces its ``size`` bytes hold, for ``tokens`` tokens of a page, in a
        layer of recurrent ``state`` or of attention.

        The base adapter declares the roles the base manager registers: K/V and their block scales
        head-major (HND) by the layer's KV heads, as the TRTLLM attention kernels write them. A
        buffer that never moves (``untransferred_roles``, ``derived_roles``) is this rank's own. An
        adapter for a manager registering any other role, ``Role.INDEX_KEY`` included, or laying
        these out otherwise, overrides this method, building tensors with ``page_layout``'s
        ``head_major``, ``token_major``, ``sectioned`` and ``replicated`` or declaring its own.

        Raises:
            ValueError: For a role that moves and that the adapter does not declare: guessing how
                its bytes spread would transfer them wrongly between ranks of different parallel
                sizes.
        """
        if role in _HEAD_SPLIT_ROLES:
            return self.head_split(layer_id, size, tokens, state)
        if role in self.untransferred_roles() | self.derived_roles():
            split = self.state_split() if state else ("tp",)
            return owned(size, grid_place(rank_topology(self.manager.mapping), split)[0], split)
        raise ValueError(
            f"{type(self.manager).__name__} registers role {role!r} but {type(self).__name__} does "
            "not declare how its bytes spread over ranks; override buffer_tensors (or list the "
            "role in untransferred_roles or derived_roles if it never moves)"
        )

    def head_split(
        self,
        layer_id: int,
        size: int,
        tokens: int,
        state: bool,
        *,
        head_bytes: Optional[int] = None,
        total: Optional[int] = None,
        token_major_: bool = False,
        dtype: Optional[str] = None,
    ) -> tuple[Tensor, ...]:
        """A buffer split by head over the ranks attention splits over (tensor ranks) or recurrent
        state does (``state_split``): ``size // head_bytes`` heads on this rank of ``total`` over
        all those ranks, head-major or, with ``token_major_``, token-major, of elements of type
        ``dtype``. By default an attention buffer's heads are the layer's KV heads, of their total,
        and a recurrent state's split evenly. An attention buffer with heads of its own
        (``head_bytes``) states their total, which the layer's KV heads do not tell. A buffer whose
        heads are unknown is this rank's own share."""
        split = self.state_split() if state else ("tp",)
        ranks, _ = grid_place(rank_topology(self.manager.mapping), split)
        if head_bytes is None and not state:
            kv_heads, kv_total = self.kv_heads(layer_id)
            if kv_heads > 0 and size % kv_heads == 0:
                head_bytes, total = size // kv_heads, kv_total
        elif head_bytes is not None and not state and total is None:
            raise ValueError(
                f"heads of {head_bytes} bytes are an attention buffer's own; state how many lie "
                "over all tensor ranks (total)"
            )
        if head_bytes is None or size % head_bytes:
            return owned(size, ranks, split)
        heads = size // head_bytes
        total = total or heads * ranks
        if token_major_:
            return token_major(size, tokens, heads, total, split, dtype=dtype)
        return head_major(size, heads, total, split, dtype=dtype)

    def state_split(self) -> tuple[str, ...]:
        """The parallel dimensions recurrent state splits over, as its pools are sized
        (``mamba_effective_tp_size``): tensor times context ranks under Helix, else tensor ranks,
        which attention data parallelism makes one rank alone (``grid_place``)."""
        m = self.manager
        if m._has_cp_helix and not getattr(m.mapping, "enable_attention_dp", False):
            return ("tp", "cp")
        return ("tp",)

    def untransferred_roles(self) -> frozenset[str]:
        """Roles whose buffers hold no state a peer needs, such as storage-only buffers that
        satisfy a pool's layout: they stay in local copies of whole pages but are never moved.
        None by default. Buffers the receiver rebuilds instead are ``derived_roles``."""
        return frozenset()

    def derived_roles(self) -> frozenset[str]:
        """Roles whose buffers derive from the transferred ones: a transfer leaves them for
        ``rebuild_imported`` to rebuild after the import. The lender, which rebuilds nothing,
        lends them with the rest; where the source holds none of them, the importer rebuilds them.
        None by default; an adapter naming any overrides that rebuild, or its page table is
        refused."""
        return frozenset()

    def request_rebuilt_roles(self) -> frozenset[str]:
        """Roles whose buffers the manager, receiving, rebuilds for some requests rather than
        receiving them (``untransferred_buffers``), so a peer may hold none of them for a model
        layer it holds: none by default. An adapter naming any overrides
        ``untransferred_buffers``, or its page table is refused. A peer missing any other role a
        model layer both hold transfers is refused, rather than leaving the buffer unwritten."""
        return frozenset()

    def dtype(self) -> str | None:
        """The element type of the KV data by name, such as ``"BF16"`` or ``"NVFP4"``; None for a
        manager declaring none."""
        dtype = getattr(self.manager, "dtype", None)
        return None if dtype is None else str(getattr(dtype, "name", dtype))

    def quant_mode(self) -> str | None:
        """How the KV data is quantized: None unquantized, else ``"fp8"`` or ``"nvfp4"``; an
        adapter for another format (DeepSeek-V4's ``"fp8_ds_mla"``) overrides this."""
        return {"FP8": "fp8", "NVFP4": "nvfp4"}.get(self.dtype())

    # -- what a transfer of one request moves ---------------------------------------------------
    # The transceiver asks these of every request; an adapter for a manager whose caches move
    # differently for some requests overrides them, so the transfer itself knows no manager.

    def transfer_length(self, request: LlmRequest) -> int:
        """The prompt tokens whose pages a transfer of the request moves, of this rank's prompt
        (``request.prompt_len``, under Helix its slice): the whole prompt. An adapter whose
        receiver recomputes a tail of the prompt overrides this, and is then refused under Helix,
        where a transfer moves the whole slice. The recompute is the executor's, not the
        transfer's: today only CSA2's remote-tail replay runs one."""
        return request.prompt_len

    def untransferred_buffers(self, request: LlmRequest) -> frozenset[tuple[int, str]]:
        """The buffers, as (global layer id, role), a transfer of this request leaves for the
        receiver to rebuild: none. The receiver's executor rebuilds them, as only CSA2's
        remote-tail replay does today; the transfer does not. Buffers that never move for any
        request are ``untransferred_roles`` and ``derived_roles``."""
        return frozenset()

    def sends_first_tokens(self, request: LlmRequest) -> bool:
        """Whether a transfer of the request carries its first generated and draft tokens: yes. A
        receiver given none computes them itself, as only CSA2's remote-tail replay does today."""
        return True

    def state_slot(self, request: LlmRequest, layer_group: int) -> Optional[int]:
        """The slot of layer group ``layer_group``'s recurrent state that a transfer of the request
        moves: the one the request's cache holds in place, or None when it holds none."""
        kv_cache = self.manager.kv_cache_map.get(request.py_request_id)
        if kv_cache is None:
            return None
        slot = int(kv_cache.get_ssm_block_base_index(layer_group))
        return slot if slot >= 0 else None

    def pipelined_transfer_blocker(self) -> Optional[str]:
        """Why the manager's pages cannot move in pieces while the prompt is still computed, or
        None when they can: recurrent state is one checkpoint at the end of the prompt, which no
        piece carries."""
        if self.model_has_recurrent_state:
            return "recurrent state, one checkpoint at the end of the prompt that no piece carries"
        return None

    @property
    def model_has_recurrent_state(self) -> bool:
        """Whether the model keeps recurrent state on any pipeline stage, whichever layers this
        stage holds: this stage's layer configs say so; an adapter for a manager whose recurrent
        layers may all lie on other stages overrides this."""
        return any(is_recurrent_layer(config) for config in self.layer_configs().values())

    def rebuild_imported(self, requests: Iterable[LlmRequest]) -> None:
        """Derive, before their first generation step, the caches of requests whose import just
        completed that the import did not carry: nothing. An adapter with such caches, the
        buffers of its ``derived_roles`` among them, overrides this and calls it through
        ``super()``."""


def _defined_by(cls: type, name: str) -> type:
    """The first class of ``cls``'s MRO defining ``name``."""
    return next(c for c in cls.__mro__ if name in vars(c))


_PKG = "tensorrt_llm._torch"

# Manager class (module.qualname) -> its adapter, as (module under ``adapters``, class name) or the
# class itself. A manager takes the adapter of the first class of its MRO with an entry, after the
# attention layout policy installed on it, if any; a manager none of whose classes has one has no
# adapter.
_ADAPTERS: dict[str, tuple[str, str] | type[CacheLayoutAdapter]] = {
    f"{_PKG}.pyexecutor.kv_cache.kv_cache_manager_v2.KVCacheManagerV2": CacheLayoutAdapter,
    f"{_PKG}.pyexecutor.kv_cache.mamba_cache_manager.MambaHybridCacheManagerV2": (
        "mamba",
        "MambaLayoutAdapter",
    ),
    f"{_PKG}.attention.backends.fp4_mla.cache_manager.Fp4MlaV2CacheLayoutPolicy": (
        "fp4_mla",
        "Fp4MlaLayoutAdapter",
    ),
    f"{_PKG}.attention.backends.sparse.csa2.cache_manager.CSA2CacheManager": (
        "csa2",
        "CSA2LayoutAdapter",
    ),
    f"{_PKG}.attention.backends.sparse.deepseek_v4.cache_manager.DeepseekV4CacheManager": (
        "deepseek_v4",
        "DeepseekV4LayoutAdapter",
    ),
    f"{_PKG}.attention.backends.sparse.dsa.cache_manager.DSACacheManagerV2": (
        "dsa",
        "DSALayoutAdapter",
    ),
    f"{_PKG}.attention.backends.sparse.glm_kpool.cache_manager.Glm5NextCacheManager": (
        "glm_kpool",
        "Glm5NextLayoutAdapter",
    ),
    f"{_PKG}.attention.backends.sparse.minimax_m3.cache_manager.MiniMaxM3KVCacheManagerV2": (
        "minimax_m3",
        "MiniMaxM3LayoutAdapter",
    ),
    f"{_PKG}.attention.backends.sparse.qsa.cache_manager.QSAMambaHybridCacheManagerV2": (
        "qsa",
        "QSALayoutAdapter",
    ),
}


def adapter_class_for(manager_class: type) -> type[CacheLayoutAdapter] | None:
    """The adapter class of managers of ``manager_class``, or None for a class none of whose MRO
    has one (a manager of another kind than ``KVCacheManagerV2``)."""
    for cls in manager_class.__mro__:
        entry = _ADAPTERS.get(f"{cls.__module__}.{cls.__qualname__}")
        if entry is None:
            continue
        if isinstance(entry, type):
            return entry
        module, name = entry
        return getattr(importlib.import_module(f"{__package__}.adapters.{module}"), name)
    return None


def adapter_for(manager: object) -> CacheLayoutAdapter | None:
    """The adapter of ``manager``: that of the attention layout policy installed on it, else that
    of its class (``adapter_class_for``); None for a manager no adapter describes."""
    policy = getattr(manager, "_attention_cache_layout_policy", None)
    for cls in (policy, type(manager)):
        if isinstance(cls, type):
            adapter = adapter_class_for(cls)
            if adapter is not None:
                return adapter(manager)
    return None
