"""A simple wrapper transform to build a model via the model factory."""

import types
from functools import partial
from typing import Optional, Tuple

import torch
import torch.fx as fx
from torch.fx import GraphModule
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ...custom_ops.attention_interface import AttentionRegistry
from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry


def cached_mha_for_hf(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    **kwargs,
):
    """
    A wrapper that accept HF attn inputs and outputs but calls flashinfer operator.
    Used to patch attn_interface in transformers.
    Args:
        *args: same as HF attention_interface.
        **kwargs: need to contain metadata, all_layers_kv_cache, and buffers from cm.
    Returns:
        attn_output: same as HF attention_interface.
        attn_weights: set to None.
    """

    # First we extract cached attn inputs from kwargs
    try:
        metadata = kwargs["metadata"]
        all_layers_kv_cache = kwargs["all_layers_kv_cache"]
        buffers = kwargs["buffers"]
    except KeyError as e:
        raise KeyError(f"Missing required kwarg: {e.args[0]}")

    # reshape to flashinfer input format
    # TODO:(hg) might want to optimize this later.
    query = torch.einsum("bhld->blhd", query)
    key = torch.einsum("bhld->blhd", key)
    value = torch.einsum("bhld->blhd", value)

    # Get the current layer's kv cache.
    layer_kv_cache = all_layers_kv_cache[module._layer_idx]

    attn_output = torch.ops.auto_deploy.flashinfer_attention_mha_with_cache(
        # QKV
        query,
        key,
        value,
        # metadata
        *metadata,
        # caches
        *layer_kv_cache,
        # buffers
        *buffers,
        # constants, harded coded for flashinfer.
        scaling,
        k_scale=1,
        v_scale=1,
    )

    attn_weights = None

    return attn_output, attn_weights


def forward_cached(module: torch.nn.Module, *cm_args):
    """
    Run transformers module with cm inputs and cached attn.
    Used to patch the original HFCausalLM.forward method.
    """
    args_map = module.cm_args_index_map
    input_ids, position_ids = cm_args[args_map["ids_and_position_ids"]]

    # cached_attn_args: seq_len, input_pos, cache_loc, pages_per_seq
    cached_attn_args = cm_args[args_map["cached_attn_args"]]
    metadata = module.get_attn_metadata(input_ids, position_ids, *cached_attn_args)

    # organize kv cache into one list per layer,
    # e.g. [[k_cache_0, v_cache_0], [k_cache_1, v_cache_1], ...]
    all_layers_kv_cache = [cm_args[slice] for slice in args_map["kvcache_args"]]

    # buffers: (workspace_buffer,)
    buffers = cm_args[args_map["buffer_args"]]

    # pass additional inputs with kwargs
    hf_input_kwargs = {
        "metadata": metadata,
        "all_layers_kv_cache": all_layers_kv_cache,
        "buffers": buffers,
    }

    return module.original_forward(
        input_ids=input_ids, position_ids=position_ids, **hf_input_kwargs
    )


def fake_profiler_mha(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    **kwargs,
):
    """Fake attn to count number of attn calls and record per-rank kv shape of each layer."""

    # Get pointer to the factory model.
    factory_model_ptr = kwargs["factory_model_ptr"]

    # Set layer idx to current attention module, so it can fetch its kv cache from cm input args later.
    module._layer_idx = len(factory_model_ptr.kv_cache_shapes)

    # Record kv cache shape for this layer, for initializing kv cache.
    factory_model_ptr.kv_cache_shapes.append(
        {
            "n_q_heads": query.shape[1],
            "n_kv_heads": key.shape[1],
            "head_dim": key.shape[-1],
        }
    )

    # Return fake outputs.
    attn_output = torch.empty_like(torch.einsum("bhld->blhd", query).contiguous())
    attn_weights = None

    return attn_output, attn_weights


def _get_fake_attn_node(num_heads: int, num_kv_heads: int, head_dim: int, dtype: torch.dtype):
    """Return a fake attn node with correct shape info for the cache initializers."""
    q_fake = fx.Node(
        graph=fx.Graph(),
        name="q_fake",
        op="call_function",
        target=lambda x: x,
        args=(1,),
        kwargs={},
    )
    q_fake.meta["val"] = torch.empty(1, 1, num_heads, head_dim, device="meta", dtype=dtype)
    k_fake = fx.Node(
        graph=fx.Graph(),
        name="k_fake",
        op="call_function",
        target=lambda x: x,
        args=(1,),
        kwargs={},
    )
    k_fake.meta["val"] = torch.empty(1, 1, num_kv_heads, head_dim, device="meta", dtype=dtype)

    dummpy_node = fx.Node(
        graph=fx.Graph(),
        name="dummy",
        op="call_function",
        target=lambda x: x,
        args=(q_fake, k_fake),
        kwargs={},
    )
    return dummpy_node


@TransformRegistry.register("transformers_replace_cached_attn")
class HFReplaceCachedAttn(BaseTransform):
    """Replace cached attention for the factory model, update inputs and outputs, and patch the gm forward."""

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        # TODO:(hg) Hard-coded attn_backend=flashinfer now.
        attn_descriptor = AttentionRegistry.get("flashinfer")
        model = gm.factory_model
        cache_config = factory.get_cache_config()

        # First, we run a profile to get the number of attns and per-rank kv cache shapes.
        # We do this to avoid any assumptions about params naming or tp plan of the hf model.

        # Register fake profiler attn.
        ALL_ATTENTION_FUNCTIONS.register("ad_cached_mha", fake_profiler_mha)
        model.config._attn_implementation = "ad_cached_mha"

        # Add an empty list as attribute for profiler to write to.
        model.kv_cache_shapes = []

        # Run forward with fake attn and dummy inputs.
        dummy_input_ids = model.dummy_inputs["input_ids"].to(model.device)
        model.original_forward(input_ids=dummy_input_ids, factory_model_ptr=model)
        num_layers = len(model.kv_cache_shapes)

        # Ends profiling, switch back to real attn operator.
        ALL_ATTENTION_FUNCTIONS.register("ad_cached_mha", cached_mha_for_hf)

        # Next, we gradually add cm args and build an index_map along the way, to
        # let each attention layer find their inputs from all cm args.
        cm_args_index_map = {}

        # Cm has 2 inputs to start with, input_ids and position_ids.
        num_cm_args = len(cm.args)
        cm_args_index_map["ids_and_position_ids"] = slice(0, num_cm_args)

        # Add cached attn args and update index map
        num_cached_attn_args = len(cm.info.switch_to_cached_attn_inputs())
        cm_args_index_map["cached_attn_args"] = slice(
            num_cm_args, num_cm_args + num_cached_attn_args
        )
        num_cm_args += num_cached_attn_args

        # We record kv cache index as a list of slices, one per layer.
        per_layer_kvcache_slice = []
        for layer_idx in range(num_layers):
            # Allowing each layer to have potentially different kv cache shapes.
            dummy_attn_node = _get_fake_attn_node(
                model.kv_cache_shapes[layer_idx]["n_q_heads"],
                model.kv_cache_shapes[layer_idx]["n_kv_heads"],
                model.kv_cache_shapes[layer_idx]["head_dim"],
                model.dtype,
            )
            cache_initializers = attn_descriptor.get_cache_initializers(
                dummy_attn_node, cache_config
            )
            for k_or_v, get_cache in cache_initializers.items():
                cm.add_cache(f"{k_or_v}_{layer_idx}", get_cache)
            per_layer_kvcache_slice.append(
                slice(num_cm_args, num_cm_args + len(cache_initializers))
            )
            num_cm_args += len(cache_initializers)
        cm_args_index_map["kvcache_args"] = per_layer_kvcache_slice

        # Add buffer and update map
        existing_buffers = set()
        for layer_idx in range(num_layers):
            for k, get_buffer in attn_descriptor.get_global_buffer_initializers(
                dummy_attn_node
            ).items():
                if k in existing_buffers:
                    continue
                cm.add_cache(k, get_buffer)
                existing_buffers.add(k)
                cm_args_index_map["buffer_args"] = slice(num_cm_args, num_cm_args + 1)
                num_cm_args += 1

        # prepare get_metadata function
        get_metadata, _ = attn_descriptor.get_prepare_metadata_op()
        model.get_attn_metadata = partial(get_metadata, page_size=cm.info.page_size)

        # Save index map for the module to fetch args from cm input
        model.cm_args_index_map = cm_args_index_map

        # Finally, we patch the forward method of facotry_model and the whole gm.
        model.forward = types.MethodType(forward_cached, model)
        gm.forward = model.forward

        # by convention, we say this fake graph module is always clean
        info = TransformInfo(skipped=False, num_matches=1, is_clean=True, has_valid_shapes=True)

        return gm, info
