"""High-level entrypoint to transform a model into an efficient inference model."""

import gc

import torch
import torch.nn as nn

from ..compile import compile_and_capture
from ..custom_ops.attention_interface import AttentionRegistry
from ..llm_args import AutoDeployConfig
from ..models.factory import ModelFactory
from ..shim.interface import CachedSequenceInterface
from ..transform.optimizer import InferenceOptimizer as ModularInferenceOptimizer
from ..utils.logger import ad_logger
from .library import (
    fuse_allreduce_residual_rmsnorm,
    fuse_collectives,
    fuse_rmsnorm,
    insert_cached_attention,
    resize_kv_cache,
    update_in_out_nodes,
)


class InferenceOptimizer:
    def __init__(self, factory: ModelFactory, ad_config: AutoDeployConfig):
        self.factory = factory
        self.ad_config = ad_config

    def __call__(self, cm: CachedSequenceInterface) -> nn.Module:
        """Transform a model into an optimized inference model.

        Args:
            model: The model to transform.
            cp: The cache pool to use for caching.
            args: Example inputs to the model.
            dynamic_shapes: Dynamic shapes to use. Defaults to None.
            poe_config: The config for positional encoding. Defaults to None.
            quantization: The quantization method to use. Defaults to None.

        Returns:
            A nn.Module representing the optimized inference model.
        """
        ############################################################################################
        # RUN MODULAR INFERENCE OPTIMIZER FOR ALREADY-MIGRATED TRANSFORMS
        ############################################################################################
        # TODO (hg): default values that are not representable in YAML.
        # move to the optimizer
        if "match_attention_layout" in self.ad_config.transforms:
            self.ad_config.transforms[
                "match_attention_layout"
            ].attention_op = AttentionRegistry.get(self.ad_config.attn_backend)
        if "match_rope_layout" in self.ad_config.transforms:
            self.ad_config.transforms["match_rope_layout"].expected_layout = AttentionRegistry.get(
                self.ad_config.attn_backend
            ).get_attention_layout()

        new_optimizer = ModularInferenceOptimizer(self.factory, self.ad_config.transforms)

        # TODO (hg): similar to above.
        if "load_weights" in new_optimizer.config:
            new_optimizer.config[
                "load_weights"
            ].checkpoint_device = self.ad_config.checkpoint_device
            new_optimizer.config["load_weights"].device = cm.device

        egm = new_optimizer(cm)

        # TODO (lucaslie): continue moving legacy transforms to the new optimizer
        ############################################################################################
        # RUN POST-LOAD FUSION AND OPTIMIZATIONS
        ############################################################################################

        # run MoE fusion
        # TODO: https://github.com/NVIDIA/TensorRT-LLM/issues/4674 this is causing OOMs
        # fuse_moe(egm)

        # run GEMM fusion
        # TODO: https://github.com/NVIDIA/TensorRT-LLM/issues/4674 this is causing OOMs
        # fuse_gemms(egm)

        # check if we can fuse allreduce, residual and rmsnorm
        fuse_allreduce_residual_rmsnorm(egm)

        # check if we can fuse collectives
        fuse_collectives(egm)

        # TODO (lucaslie): add backend selection as part of configurable inference optimizers
        # check if we can fuse rmsnorm
        fuse_rmsnorm(egm, "flashinfer")

        # visualize the final graph
        if self.ad_config.visualize:
            try:
                from .library import visualize_namespace

                visualize_namespace(egm, args=cm.args, dynamic_shapes=cm.dynamic_shapes)
                ad_logger.warning(
                    "Please run `pip install -r examples/auto_deploy/requirements.txt` to visualize"
                    " the graph."
                )
            except ImportError:
                pass

        ############################################################################################
        # SWITCH TO CACHED+FLATTENED ATTENTION + INITIALIZE CACHES
        ############################################################################################

        update_in_out_nodes(egm, cm)

        # detect attention op and replace with cache-aware op
        for a_backend in [self.ad_config.attn_backend, self.ad_config.mla_backend]:
            attn_descriptor = AttentionRegistry.get(a_backend)
            insert_cached_attention(egm, cm, attn_descriptor, self.factory.get_cache_config())

        # initialize cache on correct device
        cm.initialize_caches()

        # resize kv cache to occupy the available GPU memory up to free_mem_ratio
        resize_kv_cache(egm, cm, free_mem_ratio=self.ad_config.free_mem_ratio)

        ############################################################################################
        # COMPILE MODEL
        ############################################################################################

        cm.info.set_generate_only_batch()
        compiler_kwargs = {
            "cuda_graph_batch_sizes": self.ad_config.cuda_graph_batch_sizes,
            "num_batched_inputs": 2,  # TODO (lucaslie): improve once we have a config system...
        }
        egm_compiled = compile_and_capture(
            egm,
            self.ad_config.compile_backend,
            args=cm.args,
            dynamic_shapes=cm.dynamic_shapes,
            compiler_kwargs=compiler_kwargs,
        )
        cm.info.reset()

        torch.cuda.empty_cache()
        gc.collect()
        return egm_compiled
