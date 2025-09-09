"""High-level entrypoint to transform a model into an efficient inference model."""

import gc

import torch
import torch.nn as nn

from ..custom_ops.attention_interface import AttentionRegistry
from ..llm_args import AutoDeployConfig
from ..models.factory import ModelFactory
from ..shim.interface import CachedSequenceInterface
from ..transform.optimizer import InferenceOptimizer as ModularInferenceOptimizer


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
            self.ad_config.transforms["match_attention_layout"]["attention_op"] = (
                AttentionRegistry.get(self.ad_config.attn_backend)
            )
        if "match_rope_layout" in self.ad_config.transforms:
            self.ad_config.transforms["match_rope_layout"]["expected_layout"] = (
                AttentionRegistry.get(self.ad_config.attn_backend).get_attention_layout()
            )

        if "load_weights" in self.ad_config.transforms:
            self.ad_config.transforms["load_weights"]["checkpoint_device"] = (
                self.ad_config.checkpoint_device
            )
            self.ad_config.transforms["load_weights"]["device"] = cm.device

        if "build_and_load_factory_model" in self.ad_config.transforms:
            self.ad_config.transforms["build_and_load_factory_model"]["device"] = cm.device

        if "move_cm_to_device" in self.ad_config.transforms:
            self.ad_config.transforms["move_cm_to_device"]["checkpoint_device"] = (
                self.ad_config.checkpoint_device
            )
            self.ad_config.transforms["move_cm_to_device"]["device"] = cm.device

        if "resize_kv_cache" in self.ad_config.transforms:
            self.ad_config.transforms["resize_kv_cache"]["free_mem_ratio"] = (
                self.ad_config.free_mem_ratio
            )
        if "insert_cached_attention" in self.ad_config.transforms:
            self.ad_config.transforms["insert_cached_attention"]["attn_backend"] = (
                self.ad_config.attn_backend
            )
        if "insert_cached_mla_attention" in self.ad_config.transforms:
            self.ad_config.transforms["insert_cached_mla_attention"]["attn_backend"] = (
                self.ad_config.mla_backend
            )

        # TODO: (hg)Missing MLA here. Figure out how to add MLA since duplicate transforms are not allowed.
        # Old code:
        # detect attention op and replace with cache-aware op
        # for a_backend in [self.ad_config.attn_backend, self.ad_config.mla_backend]:
        #     attn_descriptor = AttentionRegistry.get(a_backend)
        #     insert_cached_attention(egm, cm, attn_descriptor, self.factory.get_cache_config())

        if "compile_model" in self.ad_config.transforms:
            self.ad_config.transforms["compile_model"]["cuda_graph_batch_sizes"] = (
                self.ad_config.cuda_graph_batch_sizes
            )
            self.ad_config.transforms["compile_model"]["compile_backend"] = (
                self.ad_config.compile_backend
            )

        new_optimizer = ModularInferenceOptimizer(self.factory, self.ad_config.transforms)
        # TODO: (hg) move this. let match_rope_layout and match_atten_layout use this shared config
        new_optimizer.shared_config.attn_backend = self.ad_config.attn_backend

        egm = new_optimizer(cm)

        # NOTE: (hg)Disabled visualization since compiled gm is a CapturedGraph instead of GraphModule.
        # We can add a new stage in the optimizer to visualize the intermediate gm.
        # if self.ad_config.visualize:
        #     try:
        #         from .library import visualize_namespace

        #         visualize_namespace(egm, args=cm.args, dynamic_shapes=cm.dynamic_shapes)
        #         ad_logger.warning(
        #             "Please run `pip install -r examples/auto_deploy/requirements.txt` to visualize"
        #             " the graph."
        #         )
        #     except ImportError:
        #         pass

        torch.cuda.empty_cache()
        gc.collect()
        return egm
