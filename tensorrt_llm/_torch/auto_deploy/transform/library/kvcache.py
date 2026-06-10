# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


"""Graph transformation to automatically add kv cache into fused MHA op.

The transform runs in two passes so that KV grouping is driven by a single
source of truth — ``KVPagedResourceHandler.__eq__``:

    Pass 1: Walk each source attention node, ask the backend descriptor for the
    KV handler via ``get_cache_initializers``, and assign a ``group_idx`` by
    comparing the handler against previously-seen handlers (`_find_or_add_group`
    semantics).  Same equivalence class → same group.  Because the handler's
    equality includes ``sliding_window``, same-head-dim layers with different
    windows automatically land in different groups.

    Pass 2: For multi-group models, publish the per-group window sizes to the
    interface (``set_kv_groups``) and to SequenceInfo
    (``register_window_groups``), allocate per-group metadata placeholders for
    groups ``1..N-1``, and insert cached attention ops with their layer's
    group's metadata nodes.

After the transform, ``group_idx == pool_idx`` holds everywhere: the executor
reads per-pool windows from ``cache_seq_interface.kv_group_windows`` and
issues per-window queries (``get_cache_indices(request, window_size=W)``)
against the unified ``KVCacheManager``; the C++ side routes each call to the
correct pool via its ``mLayerToWindowSize`` map.
"""

import inspect
import operator
from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
from pydantic import Field
from torch.fx import GraphModule, Node

from ...custom_ops.attention_interface import (
    AttentionDescriptor,
    AttentionRegistry,
    Constant,
    KVPagedResourceHandler,
    PrepareMetadataCallable,
    ResourceHandler,
    SpeculativeOnly,
)
from ...custom_ops.semantic_mask_registry import SemanticMaskRegistry
from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface, SpeculativeDecodingModelArgs
from ...utils._graph import add_graph_input
from ...utils.cuda_mem_tracker import get_mem_info
from ...utils.logger import ad_logger
from ...utils.node_utils import extract_op_args, get_op_schema, is_op
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)


class InsertCachedAttentionConfig(TransformConfig):
    """Configuration for the insert cached attention transform."""

    backend: Optional[str] = Field(default=None, description="The attention backend to use.")


class _InsertCachedOperator(BaseTransform):
    """A generic base transform to insert cached operators into the graph module."""

    config: InsertCachedAttentionConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return InsertCachedAttentionConfig

    @property
    def attn_descriptor(self) -> Type[AttentionDescriptor]:
        return AttentionRegistry.get(self.config.backend)

    def _process_metadata_std(self, gm: GraphModule, cm: CachedSequenceInterface) -> List[Node]:
        """Process the standard metadata nodes."""
        return [
            self._add_or_retrieve_input(gm, cm, arg_name)
            for arg_name in self.attn_descriptor.get_standard_metadata_args()
        ]

    def _process_semantic_mask(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        backend: str,
        meta_nodes_std: List[Node],
        attn_node: Node,
        semantic_mask_cache: Dict[Node, Node],
    ) -> Optional[Node]:
        """Lower a semantic attn_mask node into a backend-prepared cached mask node."""
        # Skip ops that don't have an attn_mask argument (e.g., SSM, gated delta rule)
        schema = get_op_schema(attn_node.target)
        if not any(a.name == "attn_mask" for a in schema.arguments):
            return None
        attn_mask = extract_op_args(attn_node, "attn_mask")[0]
        source_semantic_op = SemanticMaskRegistry.get_source_op(attn_mask)
        spec = SemanticMaskRegistry.get(attn_mask, backend)
        if spec is None:
            if source_semantic_op is not None:
                supported_backends = ", ".join(
                    SemanticMaskRegistry.get_supported_backends(attn_mask)
                )
                raise RuntimeError(
                    f"Cached attention backend {backend!r} does not support lowering semantic "
                    f"mask op {source_semantic_op!s}. Supported backends: {supported_backends}."
                )
            return attn_mask

        if attn_mask in semantic_mask_cache:
            return semantic_mask_cache[attn_mask]

        std_meta_by_name = dict(
            zip(self.attn_descriptor.get_standard_metadata_args(), meta_nodes_std, strict=True)
        )
        prep_args = []
        for arg in spec.prepare_op._schema.arguments:
            input_name = arg.name
            if input_name in std_meta_by_name:
                prep_args.append(std_meta_by_name[input_name])
            elif input_name in cm.info.available_args or gm.graph.find_nodes(
                op="placeholder", target=input_name
            ):
                prep_args.append(self._add_or_retrieve_input(gm, cm, input_name))
            elif arg.has_default_value():
                prep_args.append(arg.default_value)
            else:
                raise ValueError(
                    f"Semantic mask prep op expects unavailable input {input_name!r} for "
                    f"backend={backend!r}."
                )

        node_last_input = gm.graph.find_nodes(op="placeholder", sort=True)[-1]
        with gm.graph.inserting_before(node_last_input.next):
            ret_node = gm.graph.call_function(
                spec.prepare_op,
                args=(*prep_args, *spec.const_args),
            )
            if spec.num_outputs == 1:
                prepared_mask = ret_node
            else:
                prepared_mask = gm.graph.call_function(operator.getitem, args=(ret_node, 0))

        semantic_mask_cache[attn_mask] = prepared_mask
        return prepared_mask

    def _insert_extra_metadata_op(
        self,
        gm: GraphModule,
        prep_meta_op: PrepareMetadataCallable,
        inputs_for_prep_meta: List[Node],
        const_args: List[Constant],
        num_meta_out: int,
    ) -> List[Node]:
        # add the computed extra metadata nodes to the graph and add to meta for cached attention op
        meta_nodes_extra = []
        node_last_input = gm.graph.find_nodes(op="placeholder", sort=True)[-1]
        with gm.graph.inserting_before(node_last_input.next):
            ret_node = gm.graph.call_function(
                prep_meta_op, args=(*inputs_for_prep_meta, *const_args)
            )
            for idx in range(num_meta_out):
                meta_extra_node = gm.graph.call_function(operator.getitem, args=(ret_node, idx))
                meta_nodes_extra.append(meta_extra_node)

        return meta_nodes_extra

    def _process_metadata_extra(
        self, gm: GraphModule, cm: CachedSequenceInterface, any_source_attn_node: Node
    ) -> List[Node]:
        """Process the get_metadata function into an op and return node references."""
        # get the metadata op for extra metadata and number of return values
        prep_meta_op, num_meta_out, const_args = (
            self.attn_descriptor.get_prepare_extra_metadata_info(any_source_attn_node)
        )

        # if there is no extra metadata op or no return values, we can return early
        if prep_meta_op is None or num_meta_out == 0:
            return []

        # check what inputs the extra metadata op expects
        inputs_for_prep_meta = [
            self._add_or_retrieve_input(gm, cm, arg.name)
            for arg in get_op_schema(prep_meta_op).arguments
            if arg.name in cm.info.available_args
        ]

        return self._insert_extra_metadata_op(
            gm, prep_meta_op, inputs_for_prep_meta, const_args, num_meta_out
        )

    def _process_metadata_host(
        self,
        cm: CachedSequenceInterface,
        handler_groups: Optional[List[KVPagedResourceHandler]] = None,
    ):
        """Process the host-side prepare metadata function.

        When ``handler_groups`` has 2+ entries and the backend's host-prep
        declares ``pool_window_left``, registers the host-prep once per pool:
        pool 0 uses unsuffixed arg names, pools 1..N-1 use ``_g{i}`` /
        ``_g{i}_host`` suffixed names plus a closure that remaps those
        suffixed kwargs back to the function's canonical parameter names and
        binds the pool's ``pool_window_left``.  Each invocation only refreshes
        cached wrappers belonging to its pool.
        """
        prep_meta_host_op = self.attn_descriptor.get_host_prepare_metadata_function()
        if prep_meta_host_op is None:
            return

        sig = inspect.signature(prep_meta_host_op)
        sig_arg_names = list(sig.parameters.keys())

        # Backend opts in to per-pool dispatch by declaring `pool_window_left`.
        backend_supports_per_pool = "pool_window_left" in sig.parameters
        num_groups = len(handler_groups) if handler_groups is not None else 0

        if not backend_supports_per_pool or num_groups < 2:
            # Register the host-side prepare metadata function with SequenceInfo.
            # Arg availability is validated by require_copy() inside register_host_prepare.
            cm.info.register_host_prepare_for_attention_forward(
                prep_meta_host_op,
                [name for name in sig_arg_names if name != "pool_window_left"],
            )
            return

        # VSWA: register one host-prep per pool with per-pool args + bound
        # pool_window_left.  Mirrors the per-group prepare_extra_metadata wiring
        # below (vswa_swappable_bases) to keep the two routing paths consistent.
        #
        # The framework's run_host_prepare_for_attention_forward calls each
        # registered function as ``host_function(**{arg: get_arg(arg)})`` — i.e.
        # the registered arg names become the *kwarg keys* at call time.  For
        # pool > 0 the source tensors live under suffixed names
        # (``cache_loc_g1_host`` etc.) but the underlying host-prep function
        # only declares the unsuffixed parameter names.  Each per-pool
        # registration therefore wraps the host-prep in a closure that remaps
        # suffixed kwargs back to the function's canonical parameter names
        # before delegating, and binds this pool's ``pool_window_left``.
        swappable_bases = {
            "cache_loc",
            "cu_num_pages",
            "last_page_len",
            "seq_len_with_cache",
        }
        host_suffix = "_host"

        def _make_pool_host_prep(canonical_op, pool_window_left, kwarg_remap):
            """Closure: rename suffixed kwargs → canonical params, bind pool_window_left."""

            def _invoke(**kwargs):
                canonical = {kwarg_remap.get(k, k): v for k, v in kwargs.items()}
                return canonical_op(**canonical, pool_window_left=pool_window_left)

            return _invoke

        for gi, handler in enumerate(handler_groups):
            per_group_args: List[str] = []
            kwarg_remap: Dict[str, str] = {}  # suffixed → canonical
            for arg_name in sig_arg_names:
                if arg_name == "pool_window_left":
                    continue  # bound by closure
                if gi == 0:
                    per_group_args.append(arg_name)
                    continue
                base = arg_name.removesuffix(host_suffix)
                if base in swappable_bases:
                    is_host = arg_name.endswith(host_suffix)
                    suffixed = f"{base}_g{gi}{host_suffix if is_host else ''}"
                    per_group_args.append(suffixed)
                    kwarg_remap[suffixed] = arg_name
                else:
                    per_group_args.append(arg_name)

            # FlashInfer's _to_flashinfer_window_left convention:
            # window_left = sliding_window - 1 for SWA, -1 for full attention.
            sw = handler.sliding_window
            pool_window_left = sw - 1 if isinstance(sw, int) and sw > 0 else -1

            bound = _make_pool_host_prep(prep_meta_host_op, pool_window_left, kwarg_remap)
            cm.info.register_host_prepare_for_attention_forward(bound, per_group_args)

    def _process_cache_node(self, gm: GraphModule, cache_name: str) -> Node:
        """Process the cache nodes by inserting a cached attention replacement op."""
        return add_graph_input(gm, cache_name)

    def _insert_cached_attn_node(
        self,
        gm: GraphModule,
        attn_node: Node,
        cached_attn_op,
        qkv_nodes: List[Node],
        meta_nodes_std: List[Node],
        meta_nodes_extra: List[Node],
        cache_nodes: List[Node],
        constants: List[Constant],
        prepared_attn_mask: Optional[Node] = None,
    ):
        """Insert a cached attention node into the graph."""
        with gm.graph.inserting_before(attn_node):
            args = (
                *qkv_nodes,
                *meta_nodes_std,
                *meta_nodes_extra,
                *cache_nodes,
                *constants,
            )
            if prepared_attn_mask is not None:
                args = (*args, prepared_attn_mask)
            cached_attn_node = gm.graph.call_function(cached_attn_op, args=args)
        attn_node.replace_all_uses_with(cached_attn_node)
        gm.graph.erase_node(attn_node)

    @staticmethod
    def _suppress_spec_handlers_maybe(
        resource_handler: Optional[ResourceHandler], spec_config: Optional[object]
    ) -> Optional[ResourceHandler]:
        """Drop a speculative-only resource to the None sentinel when spec decoding is off.

        Handlers carrying the ``SpeculativeOnly`` trait are read only on the speculative extend
        path and are never bound by the cache manager without ``spec_config``; registering them
        would leak an unmanaged per-layer allocation. Returns ``None`` for such handlers when
        ``spec_config`` is None, otherwise returns ``resource_handler`` unchanged (``isinstance``
        is None-safe, so the existing None sentinel passes through untouched).
        """
        if spec_config is None and isinstance(resource_handler, SpeculativeOnly):
            return None
        return resource_handler

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        """Replace uncached source attention node with corresponding cached attn node."""
        attn_descriptor = self.attn_descriptor

        # look for relevant source attention nodes
        source_op = attn_descriptor.get_source_attention_op()
        source_attn_nodes = [n for n in gm.graph.nodes if is_op(n, source_op)]

        if not source_attn_nodes:
            # If there are no nodes for kv cache insertion found, return current graph
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        # Record whether this backend's kernel applies the sliding-window mask
        # itself (cyclic KV indexing, e.g. trtllm). The executor uses this to
        # decide between passing the full per-window block table + global KV
        # lengths (cyclic) and host-slicing to the live window (triton/flashinfer).
        cm.set_kernel_handles_cyclic_swa(attn_descriptor.kernel_handles_cyclic_swa())

        # get standard metadata nodes for all source attention nodes
        meta_nodes_std = self._process_metadata_std(gm, cm)

        # insert metadata computation and extract each argument as a node
        meta_nodes_extra = self._process_metadata_extra(gm, cm, source_attn_nodes[0])

        # Seed KvCacheConfig.max_attention_window from per-layer sliding_window
        # annotations so the rest of the KV-cache config plumbing (e.g.
        # downstream C++ validators that inspect the vector) sees the intended
        # per-layer windows.  The per-group KVCacheManager pools are configured
        # separately in _prepare_kv_cache_config, using each group's reference
        # handler.
        per_layer_sliding_windows = []
        for attn_node in source_attn_nodes:
            (sw,) = extract_op_args(attn_node, "sliding_window")
            per_layer_sliding_windows.append(sw)

        has_any_sliding_window = any(
            isinstance(sw, int) and sw > 0 for sw in per_layer_sliding_windows
        )
        if cm.kv_cache_config.max_attention_window is None and has_any_sliding_window:
            max_attention_window = [
                sw if isinstance(sw, int) and sw > 0 else cm.info.max_seq_len
                for sw in per_layer_sliding_windows
            ]
            cm.update_kv_cache_config(max_attention_window=max_attention_window)

        # --- Pass 1: register resources and assign per-layer group_idx ---
        # Group identity comes from KVPagedResourceHandler.__eq__ (which
        # includes sliding_window).  A group IS a pool IS a metadata set.
        handler_groups: list[KVPagedResourceHandler] = []
        per_layer_group_idx: list[int] = []
        group_idx_by_layer_idx: dict[int, int] = {}

        num_cached_attn_replacements = 0
        cache_nodes_by_layer_idx = {}
        semantic_mask_cache: Dict[Node, Node] = {}
        # Collect per-layer info for the second pass (node insertion).
        # Tuple layout:
        #   (attn_node, qkv, cache_in_nodes, constants, group_idx, prepared_mask)
        layer_infos: list[tuple] = []

        for attn_node in source_attn_nodes:
            qkv = attn_node.args[: attn_descriptor.get_num_qkv_args()]
            layer_idx = attn_descriptor.get_layer_idx(attn_node)
            shared_kv_source_layer_idx = attn_descriptor.get_shared_kv_source_layer_idx(attn_node)

            if shared_kv_source_layer_idx is not None:
                if not attn_descriptor.supports_shared_kv():
                    raise RuntimeError(
                        f"Backend '{self.config.backend}' does not support shared-KV attention."
                    )
                if layer_idx is None:
                    raise RuntimeError(
                        "Shared-KV attention node is missing layer_idx metadata required for "
                        "cache aliasing."
                    )
                if shared_kv_source_layer_idx == layer_idx:
                    raise RuntimeError(f"Layer {layer_idx} cannot share its own KV cache.")
                if shared_kv_source_layer_idx not in cache_nodes_by_layer_idx:
                    raise RuntimeError(
                        f"Missing shared-KV source layer {shared_kv_source_layer_idx}."
                    )
                cache_in_nodes = cache_nodes_by_layer_idx[shared_kv_source_layer_idx]
                # Shared-KV layers inherit their source layer's group
                group_idx = group_idx_by_layer_idx.get(shared_kv_source_layer_idx, 0)
            else:
                if layer_idx is not None and layer_idx in cache_nodes_by_layer_idx:
                    raise RuntimeError(
                        f"Duplicate KV cache owner detected for layer {layer_idx}. "
                        "Each non-shared attention layer must own exactly one cache."
                    )
                cache_in_nodes = []
                group_idx = 0
                for k, resource_handler in attn_descriptor.get_cache_initializers(
                    attn_node, cm.kv_cache_config
                ).items():
                    # Speculative-only resources (intermediate SSM/conv state and replay
                    # buffers) are never bound by the cache manager when spec decoding is off
                    # (see CachedSequenceInterface._create_and_assign_state_views), so
                    # allocating them would waste a full per-layer state buffer and OOM. Drop
                    # them to the None sentinel instead of registering an unmanaged resource.
                    resource_handler = self._suppress_spec_handlers_maybe(
                        resource_handler, cm._spec_config
                    )
                    if resource_handler is None:
                        # None sentinel: pass literal None positionally, no resource allocated.
                        cache_in_nodes.append(None)
                    else:
                        # A window that can't slide within max_seq_len is functionally
                        # full attention; normalize it to 0 so the layer shares the
                        # full-attention pool instead of forking a redundant
                        # single-window pool.
                        if (
                            isinstance(resource_handler, KVPagedResourceHandler)
                            and resource_handler.sliding_window >= cm.info.max_seq_len
                        ):
                            resource_handler.sliding_window = 0
                        resource_name = cm.add_resource(k, resource_handler)
                        node = self._process_cache_node(gm, resource_name)
                        cache_in_nodes.append(node)
                        # Determine group from handler equality
                        if isinstance(resource_handler, KVPagedResourceHandler):
                            for gi, ref in enumerate(handler_groups):
                                if resource_handler == ref:
                                    group_idx = gi
                                    break
                            else:
                                group_idx = len(handler_groups)
                                handler_groups.append(resource_handler)
                if layer_idx is not None:
                    cache_nodes_by_layer_idx[layer_idx] = cache_in_nodes
                    group_idx_by_layer_idx[layer_idx] = group_idx

            per_layer_group_idx.append(group_idx)

            attn_descriptor.prepare_node_for_cache_insertion(gm, attn_node)

            prepared_mask = self._process_semantic_mask(
                gm,
                cm,
                self.config.backend,
                meta_nodes_std,
                attn_node,
                semantic_mask_cache,
            )
            constants = attn_descriptor.get_constants(attn_node)

            layer_infos.append(
                (attn_node, qkv, cache_in_nodes, constants, group_idx, prepared_mask)
            )

        # --- Group setup: register metadata groups and create graph placeholders ---
        # Register every group (including the single-pool case) so downstream
        # consumers see a consistent representation: num_pools == num_groups.
        # Per-group graph placeholders are still only created for groups 1..N-1
        # because group 0 reuses the legacy unsuffixed tensors.
        num_groups = len(handler_groups)
        is_multi_group = num_groups >= 2
        vswa_group_nodes: dict[int, dict[str, "Node"]] = {}
        # Per-group extra-metadata: group 0 reuses the previously-built nodes;
        # groups 1..N-1 get their own prepare-extra call wired to per-group
        # swappable inputs.  This is required because some backends' prepare-extra
        # ops (e.g. Triton's, which consumes seq_len_with_cache) would otherwise
        # produce group-0 outputs that silently feed every layer.
        meta_nodes_extra_by_group: Dict[int, List[Node]] = {0: meta_nodes_extra}
        host_suffix = "_host"

        has_unmanaged_paged = num_groups == 0 and any(
            h.is_paged for h in cm._resource_lookup.values()
        )

        if num_groups >= 1:
            group_windows = [
                h.sliding_window if h.sliding_window > 0 else cm.info.max_seq_len
                for h in handler_groups
            ]
            cm.info.register_window_groups(group_windows)
            cm.set_kv_groups(group_windows)
            self._process_metadata_host(cm, handler_groups=handler_groups)
        elif has_unmanaged_paged:
            # MLA-only (and any future paged-handler family that does not contribute
            # to handler_groups): no KVPagedResourceHandler entries, but at least one
            # paged cache still consumes cache_loc/cu_num_pages metadata. Register a
            # single full-attention pool so ad_executor.py:740 stages those tensors
            # instead of passing None to nest_sequences.
            # TODO: unify paged-handler grouping (KV + MLA) under a shared abstraction
            # so the predicate at line 361 covers all paged handlers naturally.
            group_windows = [cm.info.max_seq_len]
            cm.info.register_window_groups(group_windows)
            cm.set_kv_groups(group_windows)
            self._process_metadata_host(cm, handler_groups=[])
        else:
            # No paged caches at all (cache-less or state-only models like Mamba
            # without attention). nest_sequences will correctly skip cache_loc
            # staging because no kernel in the graph consumes it. Intentional no-op.
            ad_logger.debug(
                "kvcache transform: no paged KV resources registered; "
                "cache_loc staging will be skipped. Expected for state-only or "
                "cache-less models."
            )
            self._process_metadata_host(cm, handler_groups=[])

        if is_multi_group:
            # Names that are routed per-group end-to-end.  seq_len_with_cache is
            # in this set because under SWA front-eviction the live window's
            # length diverges from the global (input_pos + seq_len); the kernel,
            # the prepare-extra op, and the host prepare all need the
            # window-capped value.
            vswa_swappable_bases = {
                "cache_loc",
                "cu_num_pages",
                "last_page_len",
                "seq_len_with_cache",
            }

            # Create per-group graph placeholders for groups 1..N-1
            std_arg_names = self.attn_descriptor.get_standard_metadata_args()
            for gi in range(1, num_groups):
                vswa_group_nodes[gi] = {}
                for arg_name in std_arg_names:
                    base = arg_name.removesuffix(host_suffix)
                    if base in vswa_swappable_bases:
                        is_host = arg_name.endswith(host_suffix)
                        group_arg = f"{base}_g{gi}{host_suffix if is_host else ''}"
                        vswa_group_nodes[gi][arg_name] = self._add_or_retrieve_input(
                            gm, cm, group_arg
                        )

            # Per-group prepare-extra-metadata: re-invoke the op with this
            # group's swappable inputs so each non-zero group's downstream
            # consumers (e.g. update_paged_kv_cache write positions) reflect
            # the window-capped view.  Skipped when the backend has no extra-op.
            prep_meta_op, num_meta_out, const_args_extra = (
                self.attn_descriptor.get_prepare_extra_metadata_info(source_attn_nodes[0])
            )
            if prep_meta_op is not None and num_meta_out > 0:
                op_arg_names = [
                    arg.name
                    for arg in get_op_schema(prep_meta_op).arguments
                    if arg.name in cm.info.available_args
                ]
                for gi in range(1, num_groups):
                    group_inputs: List[Node] = []
                    for arg_name in op_arg_names:
                        base = arg_name.removesuffix(host_suffix)
                        if base in vswa_swappable_bases:
                            is_host = arg_name.endswith(host_suffix)
                            group_arg = f"{base}_g{gi}{host_suffix if is_host else ''}"
                            group_inputs.append(self._add_or_retrieve_input(gm, cm, group_arg))
                        else:
                            group_inputs.append(self._add_or_retrieve_input(gm, cm, arg_name))
                    meta_nodes_extra_by_group[gi] = self._insert_extra_metadata_op(
                        gm, prep_meta_op, group_inputs, const_args_extra, num_meta_out
                    )
            else:
                # Backend has no extra-op: every group gets an empty list (which
                # is what the original meta_nodes_extra would have been anyway).
                for gi in range(1, num_groups):
                    meta_nodes_extra_by_group[gi] = []

        # --- Pass 2: insert cached attention nodes with correct group metadata ---
        for (
            attn_node,
            qkv,
            cache_in_nodes,
            constants,
            group_idx,
            prepared_mask,
        ) in layer_infos:
            layer_meta_nodes_std = meta_nodes_std
            if is_multi_group and group_idx > 0:
                std_arg_names = self.attn_descriptor.get_standard_metadata_args()
                layer_meta_nodes_std = list(meta_nodes_std)
                for arg_pos, arg_name in enumerate(std_arg_names):
                    if arg_name in vswa_group_nodes.get(group_idx, {}):
                        layer_meta_nodes_std[arg_pos] = vswa_group_nodes[group_idx][arg_name]

            layer_meta_nodes_extra = meta_nodes_extra_by_group.get(group_idx, meta_nodes_extra)

            self._insert_cached_attn_node(
                gm,
                attn_node,
                attn_descriptor.get_cached_attention_op(),
                qkv,
                layer_meta_nodes_std,
                layer_meta_nodes_extra,
                cache_in_nodes,
                constants,
                prepared_mask,
            )

        num_cached_attn_replacements = len(layer_infos)

        info = TransformInfo(
            skipped=False,
            num_matches=num_cached_attn_replacements,
            is_clean=False,
            has_valid_shapes=False,
        )

        return gm, info


@TransformRegistry.register("insert_cached_attention")
class InsertCachedAttention(_InsertCachedOperator):
    """A transform to insert cached attention into the graph module."""


@TransformRegistry.register("insert_cached_mla_attention")
class InsertCachedMLAAttention(_InsertCachedOperator):
    """A transform to insert cached MLA attention into the graph module."""

    @staticmethod
    def _get_mla_dims(source_attn_node: Node) -> Tuple[int, int]:
        compressed_kv_fake = source_attn_node.args[2].meta["val"]
        kpe_fake = source_attn_node.args[3].meta["val"]
        return compressed_kv_fake.shape[-1], kpe_fake.shape[-1]

    @classmethod
    def resolve_backend_for_node(
        cls,
        requested_backend: Optional[str],
        source_attn_node: Node,
    ) -> str:
        """Resolve the MLA backend for a node based on shape and local GPU support.

        AutoDeploy's current FlashInfer MLA integration is the Path 1
        ``BatchMLAPagedAttentionWrapper`` route. That path is only validated for the
        DeepSeek-style shape contract on Hopper+ today, so unsupported MLA variants
        must fall back to the torch backend before cache insertion.
        """
        backend = requested_backend or "torch_mla"
        if backend != "flashinfer_mla":
            return backend

        kv_lora_rank, qk_rope_head_dim = cls._get_mla_dims(source_attn_node)
        if not torch.cuda.is_available():
            ad_logger.warning(
                "Falling back from flashinfer_mla to torch_mla because CUDA is unavailable."
            )
            return "torch_mla"

        capability = torch.cuda.get_device_capability()
        if capability < (9, 0):
            ad_logger.warning(
                "Falling back from flashinfer_mla to torch_mla because compute capability %s "
                "is below Hopper.",
                capability,
            )
            return "torch_mla"

        if kv_lora_rank != 512 or qk_rope_head_dim != 64:
            if capability >= (10, 0) and kv_lora_rank == 256 and qk_rope_head_dim == 64:
                ad_logger.warning(
                    "Switching MLA backend from flashinfer_mla to flashinfer_trtllm_mla for "
                    "Blackwell rank-256 decode support (kv_lora_rank=%d, qk_rope_head_dim=%d, "
                    "compute capability=%s).",
                    kv_lora_rank,
                    qk_rope_head_dim,
                    capability,
                )
                return "flashinfer_trtllm_mla"

            ad_logger.warning(
                "Falling back from flashinfer_mla to torch_mla for unsupported MLA shape "
                "(kv_lora_rank=%d, qk_rope_head_dim=%d) on compute capability %s. "
                "The current AutoDeploy FlashInfer MLA path only supports kv_lora_rank=512 "
                "and qk_rope_head_dim=64.",
                kv_lora_rank,
                qk_rope_head_dim,
                capability,
            )
            return "torch_mla"

        return backend

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        if self.config.backend == "flashinfer_mla":
            source_op = AttentionRegistry.get("torch_mla").get_source_attention_op()
            source_attn_nodes = [n for n in gm.graph.nodes if is_op(n, source_op)]
            if source_attn_nodes:
                resolved_backend = self.resolve_backend_for_node(
                    self.config.backend, source_attn_nodes[0]
                )
                if resolved_backend != self.config.backend:
                    self.config.backend = resolved_backend

        return super()._apply(gm, cm, factory, shared_config)


@TransformRegistry.register("resize_kv_cache")
class ResizeKVCache(BaseTransform):
    """Resize the KV cache to occupy available GPU memory.

    This implements the two-phase approach:
    1. Run a forward pass to allocate intermediate memory (activations, workspaces, etc.)
    2. Call resize_kv_cache_manager() to recreate KVCacheManager with optimal capacity
    """

    def _apply_to_full_model(
        self,
        mod: nn.Module,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[nn.Module, TransformInfo]:
        # check if we need a resize or not
        if not cm.needs_resize():
            return mod, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        # Run a forward pass to get the extra memory usage
        cm.info.set_max_num_tokens_sample()
        try:
            if cm._spec_config is not None:
                # Resize runs before the executor's resource managers exist, so there is no
                # SAManager yet; spec_dec_args.sa_manager stays None and SA enhancement no-ops.
                mod(
                    **cm.named_args,
                    spec_dec_args=SpeculativeDecodingModelArgs(cache_seq_interface=cm),
                )
            else:
                mod(**cm.named_args)
        except torch.OutOfMemoryError as e:
            self._log_info(
                f"OutOfMemoryError in forward pass while trying to resize the kv-cache:\n{e}"
            )
            raise e

        # NOTE: use fragmented memory without empty cache (peak forward memory + fragmented memory)
        # as a proxy for the memory reserved for the forward pass. This is a rough estimate and
        # may not be accurate.
        *_, mem_reserved_for_forward = get_mem_info(empty_cache=False, unit="B")

        # Resize - KVCacheManager will compute optimal capacity based on free memory
        cm.resize_kv_cache_manager(mem_reserved_for_forward)

        info = TransformInfo(
            skipped=False,
            num_matches=0,
            is_clean=True,
            has_valid_shapes=True,
        )

        return mod, info


@TransformRegistry.register("initialize_cache")
class InitializeCache(BaseTransform):
    """Initialize KV caches using KVCacheManager.

    Gets kv_cache_config from shared_config.ad_config and creates the KVCacheManager
    in estimation mode with conservative capacity. The ResizeKVCache transform will
    later recreate it with optimal capacity after measuring memory usage.
    """

    def _apply_to_full_model(
        self,
        mod: nn.Module,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[nn.Module, TransformInfo]:
        # Initialize with estimation mode
        # This allows resize_kv_cache to recreate with correct capacity after measuring memory
        num_caches = cm.initialize_resources()

        info = TransformInfo(
            skipped=False, num_matches=num_caches, is_clean=True, has_valid_shapes=True
        )
        return mod, info
