"""A simple wrapper transform to build a model via the model factory."""

import types
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.fx as fx
from torch.fx import GraphModule, Node
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ...custom_ops.attention_interface import AttentionDescriptor
from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry
from .kvcache import InsertCachedAttention


def fake_profiler_mha(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    profiling_metadata: Dict[str, Any],
    **kwargs,
):
    """Fake attn to populated attention nodes of each layer."""
    # logic adopted from
    # https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/sdpa_attention.py#L73
    if "is_causal" in kwargs:
        is_causal = kwargs["is_causal"]
    else:
        is_causal = (
            query.shape[2] > 1 and attention_mask is None and getattr(module, "is_causal", True)
        )

    # construct kwargs for bsnd_grouped_sdpa
    node_kwargs = {"attn_mask": attention_mask, "is_causal": is_causal}
    kwargs_to_op = {
        "dropout": "dropout_p",
        "scaling": "scale",
        "scale": "scale",
        "s_aux": "sinks",
        "sinks": "sinks",
        "sliding_window": "sliding_window",
        "logit_cap": "logit_cap",
    }
    for k_kwargs, k_op_kwargs in kwargs_to_op.items():
        if k_kwargs in kwargs:
            node_kwargs[k_op_kwargs] = kwargs[k_kwargs]

    # store new fake attention node in graph module
    graph: fx.Graph = profiling_metadata["gm"].graph
    q_fake = graph.placeholder(name="q_fake")
    q_fake.meta["val"] = torch.empty_like(query.transpose(1, 2), device="meta", dtype=query.dtype)
    k_fake = graph.placeholder(name="k_fake")
    k_fake.meta["val"] = torch.empty_like(key.transpose(1, 2), device="meta", dtype=key.dtype)
    v_fake = graph.placeholder(name="v_fake")
    v_fake.meta["val"] = torch.empty_like(value.transpose(1, 2), device="meta", dtype=value.dtype)

    module._node_ref = graph.call_function(
        torch.ops.auto_deploy.torch_attention_bsnd_grouped_sdpa,
        args=(q_fake, k_fake, v_fake),
        kwargs=node_kwargs,
    )

    # update num matches
    profiling_metadata["num_matches"] += 1

    # Return fake outputs
    attn_output = torch.empty_like(query.transpose(1, 2).contiguous())
    attn_weights = None

    return attn_output, attn_weights


@TransformRegistry.register("detect_hf_attn_layers")
class DetectHFAttnLayers(BaseTransform):
    """Detect the number of attn layers in the model and store a node-like reference for them.

    This is achieved by running a single forward pass to profile the model.
    """

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        model = gm.factory_model

        # Register profiler attn operator
        ALL_ATTENTION_FUNCTIONS.register("ad_profile_mha", fake_profiler_mha)
        attn_implementation = model.config._attn_implementation
        model.config._attn_implementation = "ad_profile_mha"

        # update the graph module with the fake attn nodes during the profiling run
        profiling_metadata = {"gm": gm, "num_matches": 0}
        model.forward(*cm.args, profiling_metadata=profiling_metadata)

        # switch back to original attn implementation
        model.config._attn_implementation = attn_implementation

        info = TransformInfo(
            skipped=False,
            num_matches=profiling_metadata["num_matches"],
            is_clean=True,
            has_valid_shapes=True,
        )

        return gm, info


def get_cached_attn(
    attn_descriptor: Type[AttentionDescriptor],
):
    """
    Returns the cached attn operator that can be called with HF attn inputs and outputs.

    Args:
        ad_cached_attn_op: the cached attn operator to call.
    Returns:
        cached_attn: the cached attn operator that can be called with HF attn inputs and outputs.
    """

    def cached_attn(
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        **kwargs,
    ):
        # check if we need to transpose the inputs
        # incoming layout is bnsd in HF attn interface
        attention_layout = attn_descriptor.get_attention_layout()
        if attention_layout == "bsnd":
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
        elif attention_layout != "bnsd":
            raise ValueError(f"Unsupported attention layout: {attention_layout}")

        attn_output = attn_descriptor.get_cached_attention_op()(
            query,
            key,
            value,
            # metadata+caches+buffers+constants as constructed in forward_cached
            *kwargs["cached_attn_args_lookup"][module._node_ref],
        )

        # check if we need to transpose the outputs, outgoing layout is bsnd in HF attn interface
        if attention_layout == "bnsd":
            attn_output = attn_output.transpose(1, 2)
        elif attention_layout != "bsnd":
            raise ValueError(f"Unsupported attention layout: {attention_layout}")

        # Cached attn ops does not return attn weights in general
        attn_weights = None

        return attn_output, attn_weights

    return cached_attn


def forward_cached(gm: GraphModule, *cm_args):
    """Pre-process cached arguments for attention and then run regular factory forward."""
    # retrieve uncached args
    uncached_args = cm_args[: gm.num_uncached_args]

    cached_attn_args_lookup: Dict[Node, List[Any]] = defaultdict(list)

    # check if there is any cached attn nodes and if yes, compute metadata
    if gm.node_to_cache_buffer_indices:
        metadata = gm.get_metadata(
            *[cm_args[i] for i in gm.prepare_metadata_args_index_map],
            *gm.prepare_metadata_const_args,
        )

    for node, c_b_indices in gm.node_to_cache_buffer_indices.items():
        cached_attn_args_lookup[node].extend(metadata)
        cached_attn_args_lookup[node].extend([cm_args[i] for i in c_b_indices])
        cached_attn_args_lookup[node].extend(gm.node_to_constants[node])

    return gm.factory_model.forward(*uncached_args, cached_attn_args_lookup=cached_attn_args_lookup)


# TODO: how running different kv cache transforms look like? This one below wouldn't work if we
# had multiple types attention to replace...
@TransformRegistry.register("transformers_replace_cached_attn")
class HFReplaceCachedAttn(InsertCachedAttention):
    """Replace cached attention for the factory model, update inputs and outputs, and patch the gm forward."""

    def _process_get_metadata(
        self, gm: GraphModule, m_args: List[str], const_args: List[Any]
    ) -> List[Node]:
        """Store get metadata function as reference and simply return."""
        get_metadata, num_ret_metadata = self.attn_descriptor.get_prepare_metadata_op()
        gm.get_metadata = get_metadata
        gm.prepare_metadata_arg_names = m_args
        gm.prepare_metadata_const_args = const_args
        return [f"metadata_{i}" for i in range(num_ret_metadata)]  # we don't need actual nodes...

    def _process_cache_node(self, gm: GraphModule, cache_name: str) -> Node:
        """We don't need to actually do anything here, just return the cache name."""
        return cache_name

    def _insert_cached_attn_node(
        self,
        gm: GraphModule,
        attn_node: Node,
        qkv_nodes: List[Node],
        meta_nodes: List[Node],
        cache_nodes: List[Node],
        buffer_nodes: List[Node],
        constants: List[Any],
    ):
        """Here we now need to actually do the correct mapping of the cached attn nodes."""
        # store reference to metadata, caches, buffers, and constants for this attn node
        gm.node_to_cache_buffer_names[attn_node] = (*cache_nodes, *buffer_nodes)
        gm.node_to_constants[attn_node] = constants

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        # patch gm so we can run the insert cached attn transform
        gm.node_to_cache_buffer_names: Dict[Node, List[str]] = {}
        gm.node_to_constants: Dict[Node, List[Any]] = {}

        # switch to to cached inputs now as expected by insert cached attn transform
        gm.num_uncached_args = len(cm.args)
        cm.info.switch_to_cached_attn_inputs()

        # run actual insert cached attn transform
        gm, info = super()._apply(gm, cm, factory, shared_config)

        # we retrieve a key to index map for cm.args
        cm_keys_to_index = {k: idx for idx, k in enumerate(cm.all_future_arg_names)}

        # store index map for prepare_metadata relevant args
        gm.prepare_metadata_args_index_map = [
            cm_keys_to_index[k] for k in gm.prepare_metadata_arg_names
        ]

        # store a map from node to cache_buffer indices
        gm.node_to_cache_buffer_indices: Dict[Node, List[int]] = {}
        for node, c_b_names in gm.node_to_cache_buffer_names.items():
            gm.node_to_cache_buffer_indices[node] = [cm_keys_to_index[k] for k in c_b_names]

        # switch to cached forward which requires the above computed index maps
        ALL_ATTENTION_FUNCTIONS.register("ad_cached_mha", get_cached_attn(self.attn_descriptor))
        gm.factory_model.config._attn_implementation = "ad_cached_mha"
        gm.forward = types.MethodType(forward_cached, gm)

        # we assume graph is clean again by definition
        info_dict = info.model_dump()
        info_dict["is_clean"] = True
        info_dict["has_valid_shapes"] = True
        info = TransformInfo(**info_dict)

        return gm, info
