"""A simple wrapper transform to build a model via the model factory."""

import types
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.fx as fx
import torch.nn as nn
from torch.fx import Graph, GraphModule, Node
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ...custom_ops.attention_interface import AttentionDescriptor, Constant
from ...export.library.unified_attn import HF_ATTN_KWARGS_MAPPING
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

    for k_kwargs, k_op_kwargs in HF_ATTN_KWARGS_MAPPING.items():
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

    node_kwargs["layout"] = "bsnd"
    module._node_ref = graph.call_function(
        torch.ops.auto_deploy.torch_attention,
        args=(q_fake, k_fake, v_fake),
        kwargs=node_kwargs,
    )

    # update num matches
    profiling_metadata["num_matches"] += 1

    # Return fake outputs
    attn_output = torch.empty_like(query.transpose(1, 2).contiguous())
    attn_weights = None

    return attn_output, attn_weights


@contextmanager
def switch_attn_implementation(config: PretrainedConfig, attn_implementation: str):
    """Temporarily switch the attn implementation of the model."""
    # store original attn implementation including from sub_configs
    attn_implementations: Dict[Optional[str], str] = {}
    attn_implementations[None] = config._attn_implementation
    for sub_config_key in config.sub_configs:
        sub_config = getattr(config, sub_config_key, None)
        if sub_config is not None:
            attn_implementations[sub_config_key] = sub_config._attn_implementation

    # override attn implementation for all configs/sub_configs
    config._attn_implementation = attn_implementation

    yield

    # restore original attn implementation for all configs/sub_configs
    for sub_config_key, sub_attn_implementation in attn_implementations.items():
        sub_config = config if sub_config_key is None else getattr(config, sub_config_key)
        sub_config._attn_implementation = sub_attn_implementation


@TransformRegistry.register("detect_hf_attn_layers")
class DetectHFAttnLayers(BaseTransform):
    """Detect the number of attn layers in the model and store a node-like reference for them.

    This is achieved by running a single forward pass to profile the model.
    """

    def _apply_to_full_model(
        self,
        mod: nn.Module,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[nn.Module, TransformInfo]:
        # Register profiler attn operator
        ALL_ATTENTION_FUNCTIONS.register("ad_profile_mha", fake_profiler_mha)

        # let's start a fake graph module for making tracing/profiling easier
        mod._gm = GraphModule(nn.Module(), Graph())

        # run the forward pass with the profiling function
        with switch_attn_implementation(mod.config, "ad_profile_mha"):
            # update the graph module with the fake attn nodes during the profiling run
            profiling_metadata = {"gm": mod._gm, "num_matches": 0}
            mod.forward(**cm.named_args, profiling_metadata=profiling_metadata)

        info = TransformInfo(
            skipped=False,
            num_matches=profiling_metadata["num_matches"],
            is_clean=True,
            has_valid_shapes=True,
        )

        return mod, info


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
            # metadata+caches+buffers with name lookup set up during kvcache transform
            *[kwargs[k] for k in module._node_ref.meta["metadata_cache_buffer_keys"]],
            # constants set up during kvcache transform
            *module._node_ref.meta["constants"],
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


def forward_with_prepare_metadata(mod: nn.Module, **cm_kwargs):
    """Run prepare_metadata as pre-processing step, add to kwargs, and then run regular forward."""
    gm = mod._gm
    if hasattr(gm, "_prepare_metadata_info"):
        # collect args+constant args
        args = [cm_kwargs[k] for k in gm._prepare_metadata_info["arg_names"]]
        const_args = gm._prepare_metadata_info["const_args"]

        # run prepare_metadata function and add to kwargs
        metadata = gm._prepare_metadata_info["get_metadata"](*args, *const_args)
        return_names = gm._prepare_metadata_info["return_names"]
        cm_kwargs.update({k: v for k, v in zip(return_names, metadata)})

    return mod._original_forward(**cm_kwargs)


# TODO: how running different kv cache transforms look like? This one below wouldn't work if we
# had multiple types attention to replace...
@TransformRegistry.register("transformers_replace_cached_attn")
class HFReplaceCachedAttn(InsertCachedAttention):
    """Replace cached attention for the factory model, update inputs and outputs, and patch the gm forward."""

    def _process_get_metadata(
        self, gm: GraphModule, m_args: List[str], const_args: List[Constant]
    ) -> List[Node]:
        """Store get metadata function as reference and simply return."""
        get_metadata, num_ret_metadata = self.attn_descriptor.get_prepare_metadata_op()
        gm._prepare_metadata_info = {
            "get_metadata": get_metadata,
            "arg_names": m_args,
            "const_args": const_args,
            "return_names": [f"metadata_{i}" for i in range(num_ret_metadata)],
        }
        return gm._prepare_metadata_info["return_names"]  # we don't need actual nodes...

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
        constants: List[Constant],
    ):
        """Here we now need to actually do the correct mapping of the cached attn nodes."""
        # store reference to metadata, caches, buffers, and constants for this attn node
        attn_node.meta["metadata_cache_buffer_keys"] = (*meta_nodes, *cache_nodes, *buffer_nodes)
        attn_node.meta["constants"] = constants

    def _apply_to_full_model(
        self,
        mod: nn.Module,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[nn.Module, TransformInfo]:
        # switch to cached attn inputs from now
        cm.info.switch_to_cached_attn_inputs()

        # run actual insert cached attn transform with fake graph module
        mod._gm, info = super()._apply(mod._gm, cm, factory, shared_config)

        # register cached attn operator and switch to cached forward function
        ALL_ATTENTION_FUNCTIONS.register("ad_cached_mha", get_cached_attn(self.attn_descriptor))
        mod._original_forward = mod.forward
        mod.forward = types.MethodType(forward_with_prepare_metadata, mod)

        # switch to cached attn implementation but _only_ for submodules/configs that have a cached
        # attn node (we don't want to switch to cached attn implementation for all modules)
        for submod in mod.modules():
            if hasattr(submod, "_node_ref"):
                submod.config._attn_implementation = "ad_cached_mha"

        # we assume graph is clean again by definition
        info_dict = info.model_dump()
        info_dict["is_clean"] = True
        info_dict["has_valid_shapes"] = True
        info = TransformInfo(**info_dict)

        return mod, info
