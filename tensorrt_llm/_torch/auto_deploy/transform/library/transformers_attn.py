"""A simple wrapper transform to build a model via the model factory."""

import types
from functools import partial
from typing import Optional, Tuple

import torch
from torch.fx import GraphModule
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ...custom_ops.attention_interface import AttentionRegistry
from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface, SequenceInfo
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

    # cm pass cache of all layers. we need to get the current layer's kv cache.
    layer_idx = module._layer_idx
    layer_kv_cache = (all_layers_kv_cache[layer_idx * 2], all_layers_kv_cache[layer_idx * 2 + 1])

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


def forward_cached(module: torch.nn.Module, *cm_args, **cm_kwargs):
    """
    Run transformers module with cm inputs and cached attn.
    Used to patch the original HFCausalLM.forward method.
    """
    expected_num_args = (
        2 + module.num_cached_attn_args + module.num_kvcache_args + module.num_buffer_args
    )
    assert len(cm_args) == expected_num_args, (
        f"Number of cm input args ({len(cm_args)}) mismatch. Expected {expected_num_args}."
    )

    input_ids, position_ids = cm_args[:2]

    # cached_attn_args: seq_len, input_pos, cache_loc, pages_per_seq
    cached_attn_args = cm_args[2 : 2 + module.num_cached_attn_args]
    metadata = module.get_attn_metadata(input_ids, position_ids, *cached_attn_args)
    assert len(metadata) == module.num_metadata, (
        f"Number of metadata ({len(metadata)}) mismatch. Expected {module.num_metadata}."
    )

    # kv_cache_args: k_cache_0, v_cache_0, k_cache_1, v_cache_1, ...
    kv_cache_args = cm_args[
        2 + module.num_cached_attn_args : 2 + module.num_cached_attn_args + module.num_kvcache_args
    ]

    # buffers: (workspace_buffer,)
    buffers = cm_args[2 + module.num_cached_attn_args + module.num_kvcache_args :]

    # pass additional inputs with kwargs
    hf_input_kwargs = {
        "metadata": metadata,
        "all_layers_kv_cache": kv_cache_args,
        "buffers": buffers,
    }

    return module.forward_with_kwargs(input_ids, position_ids, **hf_input_kwargs)


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

        # switch cm to cached attn inputs
        num_cached_attn_args = len(cm.info.switch_to_cached_attn_inputs())

        # Each attn module save their layer idx to find their cache from cm input args later.
        for layer_idx in range(hf_config.num_hidden_layers):
            gm.factory_model.model.layers[layer_idx].self_attn._layer_idx = layer_idx

        # Add cache
        # equivalent to attn_descriptor.get_cache_initializers()
        def _get_cache(si: SequenceInfo):
            # use torch.empty
            return torch.zeros(
                si.num_pages,
                si.page_size,
                hf_config.num_key_value_heads,
                hf_config.head_dim,
                device=si.device,
                dtype=cache_config.dtype or hf_config.torch_dtype,
            )

        cache_initializers = {
            "k_cache": _get_cache,
            "v_cache": _get_cache,
        }

        num_kvcache_args = 0
        for layer_idx in range(hf_config.num_hidden_layers):
            for k_or_v, get_cache in cache_initializers.items():
                k_indexed = f"{k_or_v}_{layer_idx}"
                cm.add_cache(k_indexed, get_cache)
                num_kvcache_args += 1

        # Add buffer
        # equivalent to attn_descriptor.get_global_buffer_initializers
        num_buffer_args = 0

        def _init_workspace(si: SequenceInfo) -> torch.Tensor:
            # NOTE (lucaslie): avoid OOM for many cudagraphs,
            # see https://github.com/NVIDIA/TensorRT-LLM/pull/3686
            buffer = torch.empty(320 * 1024 * 1024, dtype=torch.uint8, device=si.device)
            attn_descriptor._get_planner().init_workspace(buffer)
            return buffer

        cm.add_cache("workspace_buffer", _init_workspace)
        num_buffer_args += 1

        # prepare get_metadata function
        get_metadata, num_metadata = attn_descriptor.get_prepare_metadata_op()

        # Save attributes for the module to use during forward
        # TODO: consider make an attribute class
        gm.factory_model.get_attn_metadata = partial(get_metadata, page_size=cm.info.page_size)
        gm.factory_model.num_cached_attn_args = num_cached_attn_args
        gm.factory_model.num_kvcache_args = num_kvcache_args
        gm.factory_model.num_buffer_args = num_buffer_args
        gm.factory_model.num_metadata = num_metadata

        # Finally, we patch the forward method of facotry_model and the whole gm.
        gm.factory_model.forward = types.MethodType(
            partial(forward_cached, get_metadata=get_metadata), gm.factory_model
        )
        gm.forward = gm.factory_model.forward

        # by convention, we say this fake graph module is always clean
        info = TransformInfo(skipped=False, num_matches=1, is_clean=True, has_valid_shapes=True)

        return gm, info
