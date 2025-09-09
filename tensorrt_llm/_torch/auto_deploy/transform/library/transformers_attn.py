"""A simple wrapper transform to build a model via the model factory."""

import types
from typing import Callable, Optional, Tuple, Type

import torch
import torch.fx as fx
from pydantic import Field
from torch.fx import GraphModule
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ...custom_ops.attention_interface import AttentionRegistry
from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)


def hf_cached_attn_wrapper(
    ad_cached_attn_op: Callable,
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
        attn_output = ad_cached_attn_op(
            # QKV in blhd layout
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            # metadata
            *(kwargs.get("metadata")),
            # caches
            *(kwargs.get("all_layers_kv_cache")[module._layer_idx]),
            # buffers
            *(kwargs.get("buffers")),
            # constants, harded coded for flashinfer.
            *(kwargs.get("all_layers_attn_constants")[module._layer_idx]),
        )

        # Cached attn ops does not return attn weights in general
        attn_weights = None

        return attn_output, attn_weights

    return cached_attn


def forward_cached(gm: GraphModule, *cm_args):
    """
    Run transformers module with cm inputs and cached attn.
    Used to patch the original HFCausalLM.forward method.
    """
    # TODO: differentiate between standard args and extra args instead of hard-coding them here.
    args_map = gm.cm_args_index_map
    input_ids, position_ids, *extra_args = cm_args[args_map["ids_and_position_ids"]]

    # cached_attn_args: seq_len, input_pos, cache_loc, pages_per_seq
    cached_attn_args = cm_args[args_map["cached_attn_args"]]
    metadata = gm.get_metadata(
        input_ids,
        position_ids,
        *cached_attn_args,
        *gm.const_args_for_prepare_metadata,
    )

    # organize kv cache into one list per layer,
    # e.g. [[k_cache_0, v_cache_0], [k_cache_1, v_cache_1], ...]
    all_layers_kv_cache = [cm_args[slice] for slice in args_map["kvcache_args"]]

    # buffers: e.g. (workspace_buffer,)
    buffers = cm_args[args_map["buffer_args"]] if "buffer_args" in args_map else ()

    # get attention constants
    all_layers_attn_constants = gm.attention_constants

    # pass additional inputs with kwargs
    hf_input_kwargs = {
        "metadata": metadata,
        "all_layers_kv_cache": all_layers_kv_cache,
        "buffers": buffers,
        "all_layers_attn_constants": all_layers_attn_constants,
    }

    return gm.factory_model.forward(input_ids, position_ids, *extra_args, **hf_input_kwargs)


def fake_profiler_mha(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    **kwargs,
):
    """Fake attn to populated attention nodes of each layer."""
    # Set layer idx to current attention module
    module._layer_idx = len(kwargs["source_attn_nodes"])

    # logic adopted from
    # https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/sdpa_attention.py#L73
    if "is_causal" in kwargs:
        is_causal = kwargs["is_causal"]
    else:
        is_causal = (
            query.shape[2] > 1 and attention_mask is None and getattr(module, "is_causal", True)
        )

    # Append new attn node detected at current layer
    kwargs["source_attn_nodes"].append(
        _get_fake_attn_node(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            module.config.torch_dtype,
            node_kwargs={
                "attn_mask": attention_mask,
                "dropout_p": kwargs.get("dropout"),
                "is_causal": is_causal,
                "scale": kwargs.get("scaling"),
            },
        )
    )

    # Return fake outputs
    attn_output = torch.empty_like(query.transpose(1, 2).contiguous())
    attn_weights = None

    return attn_output, attn_weights


def _get_fake_attn_node(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dtype: torch.dtype,
    node_kwargs: dict = {},
):
    """Return a standard attention node with correct info for the cache initializations."""

    fake_graph = fx.Graph()
    q_fake = fake_graph.placeholder(name="q_fake")
    q_fake.meta["val"] = torch.empty_like(query, device="meta", dtype=dtype)
    k_fake = fake_graph.placeholder(name="k_fake")
    k_fake.meta["val"] = torch.empty_like(key, device="meta", dtype=dtype)
    v_fake = fake_graph.placeholder(name="v_fake")
    v_fake.meta["val"] = torch.empty_like(value, device="meta", dtype=dtype)

    dummy_node = fake_graph.call_function(
        torch.ops.auto_deploy.torch_attention_bsnd_grouped_sdpa,
        args=(q_fake, k_fake, v_fake),
        kwargs=node_kwargs,
    )
    return dummy_node


class HFReplaceCachedAttnConfig(TransformConfig):
    """Configuration for the transformers cached attention transform."""

    attn_backend: Optional[str] = Field(default=None, description="The attention backend to use.")


@TransformRegistry.register("transformers_replace_cached_attn")
class HFReplaceCachedAttn(BaseTransform):
    """Replace cached attention for the factory model, update inputs and outputs, and patch the gm forward."""

    config: HFReplaceCachedAttnConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return HFReplaceCachedAttnConfig

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        attn_descriptor = AttentionRegistry.get(self.config.attn_backend)
        model = gm.factory_model
        cache_config = factory.get_cache_config()

        # First, we run a profile to get the number of attns and per-rank kv cache shapes.
        # We do this to avoid any assumptions about params naming or tp plan of the hf model.

        # Register fake profiler attn.
        ALL_ATTENTION_FUNCTIONS.register("ad_cached_mha", fake_profiler_mha)
        model.config._attn_implementation = "ad_cached_mha"

        # we set the standard example sequence WITHOUT extra_args to set them to None so that
        # only the text portion of the model gets called.
        cm.info.set_example_sequence()

        # Create an empty list of source attn nodes and populate with profiling forward run
        source_attn_nodes = []
        model.forward(*cm.args, source_attn_nodes=source_attn_nodes)

        # Ends profiling, switch back to real attn operator.
        ALL_ATTENTION_FUNCTIONS.register(
            "ad_cached_mha", hf_cached_attn_wrapper(attn_descriptor.get_cached_attention_op())
        )

        # Next, add cm args and build an index_map for positional args
        cm_args_index_map = {}

        # TODO: differentiate between standard args and extra args
        # Cm has 2 inputs to start with, input_ids and position_ids.
        num_cm_args = len(cm.args)
        cm_args_index_map["ids_and_position_ids"] = slice(0, num_cm_args)

        # Add cached attn args and update index map
        num_cached_attn_args = len(cm.info.switch_to_cached_attn_inputs())
        cm_args_index_map["cached_attn_args"] = slice(
            num_cm_args, num_cm_args + num_cached_attn_args
        )
        num_cm_args += num_cached_attn_args

        # Record kv cache index as a list of slices, one per layer.
        per_layer_kvcache_slice = []
        for layer_idx, attn_node in enumerate(source_attn_nodes):
            cache_initializers = attn_descriptor.get_cache_initializers(attn_node, cache_config)
            for k_or_v, get_cache in cache_initializers.items():
                cm.add_cache(f"{k_or_v}_{layer_idx}", get_cache)
            per_layer_kvcache_slice.append(
                slice(num_cm_args, num_cm_args + len(cache_initializers))
            )
            num_cm_args += len(cache_initializers)
        cm_args_index_map["kvcache_args"] = per_layer_kvcache_slice

        # Add buffer and update map
        existing_buffers = set()
        for attn_node in source_attn_nodes:
            for k, get_buffer in attn_descriptor.get_global_buffer_initializers(attn_node).items():
                if k in existing_buffers:
                    continue
                cm.add_cache(k, get_buffer)
                existing_buffers.add(k)
                cm_args_index_map["buffer_args"] = slice(num_cm_args, num_cm_args + 1)
                num_cm_args += 1

        # Add constants as model attributes
        gm.attention_constants = []
        for attn_node in source_attn_nodes:
            constants = attn_descriptor.get_constants(attn_node)
            gm.attention_constants.append(constants)

        # prepare get_metadata function
        get_metadata, _ = attn_descriptor.get_prepare_metadata_op()
        gm.get_metadata = get_metadata
        gm.const_args_for_prepare_metadata = cm.info.const_args_for_prepare_metadata

        # Save index map for the module to fetch args from cm input
        gm.cm_args_index_map = cm_args_index_map

        # Finally, we patch the forward method of the gm.
        gm.forward = types.MethodType(forward_cached, gm)

        # by convention, we say this fake graph module is always clean
        info = TransformInfo(
            skipped=False,
            num_matches=len(source_attn_nodes),
            is_clean=True,
            has_valid_shapes=True,
        )

        return gm, info
