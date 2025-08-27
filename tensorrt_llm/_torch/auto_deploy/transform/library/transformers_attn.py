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


def _get_fake_attn_node(num_heads: int, num_kv_heads: int, head_dim: int, dtype: torch.dtype):
    """Return a fake attn node for the cache initializers."""
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
        cache_config = factory.get_cache_config()
        hf_config = gm.factory_model.config
        # TODO:(hg) Hard-coded attn_backend=flashinfer now.
        attn_descriptor = AttentionRegistry.get("flashinfer")

        # Register cached attn in HF.
        ALL_ATTENTION_FUNCTIONS.register("ad_cached_mha", cached_mha_for_hf)
        gm.factory_model.config._attn_implementation = "ad_cached_mha"

        # Each attn module save their layer idx to find their cache from cm input args later.
        for layer_idx in range(hf_config.num_hidden_layers):
            gm.factory_model.model.layers[layer_idx].self_attn._layer_idx = layer_idx

        cm_args_index_map = {}
        # cm have 2 inputs by default, input_ids and position_ids.
        num_cm_args = len(cm.args)
        cm_args_index_map["ids_and_position_ids"] = slice(0, num_cm_args)

        # Add cached attn args and update index map
        num_cached_attn_args = len(cm.info.switch_to_cached_attn_inputs())
        cm_args_index_map["cached_attn_args"] = slice(
            num_cm_args, num_cm_args + num_cached_attn_args
        )
        num_cm_args += num_cached_attn_args

        # Add cache and update map
        dummy_attn_node = _get_fake_attn_node(
            hf_config.num_attention_heads,
            hf_config.num_key_value_heads,
            hf_config.head_dim,
            gm.factory_model.dtype,
        )
        cache_initializers = attn_descriptor.get_cache_initializers(dummy_attn_node, cache_config)
        per_layer_kvcache_slice = []
        for layer_idx in range(hf_config.num_hidden_layers):
            for k_or_v, get_cache in cache_initializers.items():
                cm.add_cache(f"{k_or_v}_{layer_idx}", get_cache)
            per_layer_kvcache_slice.append(
                slice(num_cm_args, num_cm_args + len(cache_initializers))
            )
            num_cm_args += len(cache_initializers)
        cm_args_index_map["kvcache_args"] = per_layer_kvcache_slice

        # Add buffer and update map
        existing_buffers = set()
        for layer_idx in range(hf_config.num_hidden_layers):
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
        gm.factory_model.get_attn_metadata = partial(get_metadata, page_size=cm.info.page_size)

        # Save index map for the module to fetch args from cm input
        gm.factory_model.cm_args_index_map = cm_args_index_map

        # Finally, we patch the forward method of facotry_model and the whole gm.
        gm.factory_model.forward = types.MethodType(forward_cached, gm.factory_model)
        gm.forward = gm.factory_model.forward

        # by convention, we say this fake graph module is always clean
        info = TransformInfo(skipped=False, num_matches=1, is_clean=True, has_valid_shapes=True)

        return gm, info
