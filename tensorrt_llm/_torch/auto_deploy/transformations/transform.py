"""High-level entrypoint to transform a model into an efficient inference model."""

import gc

import torch
from torch.fx import GraphModule

from ..compile import compile_and_capture
from ..custom_ops.attention_interface import AttentionRegistry
from ..distributed import common as dist_ad
from ..models.factory import ModelFactory
from ..shim.interface import AutoDeployConfig, CachedSequenceInterface
from ..utils.logger import ad_logger
from ._graph import canonicalize_graph, move_to_device
from .export import torch_export_to_gm
from .library import (
    check_in_out_nodes,
    column_row_shard,
    eliminate_redundant_transposes,
    ep_shard,
    fuse_allreduce_residual_rmsnorm,
    fuse_collectives,
    fuse_gemms,
    fuse_moe,
    insert_cached_attention,
    match_attention_layout,
    match_causal_attn_mask,
    match_eager_attention,
    match_grouped_attention,
    match_moe_pattern,
    match_repeat_kv,
    match_rope_v1,
    match_rope_v2,
    quantize,
    resize_kv_cache,
)


class InferenceOptimizer:
    def __init__(
        self,
        factory: ModelFactory,
        *,  # TODO: temporary until we have a better config system
        ad_config: AutoDeployConfig,
        visualize: bool = False,
    ):
        self.factory = factory
        self.attn_backend = ad_config.attn_backend
        self.mla_backend = ad_config.mla_backend
        # TODO (lliebenwein): let's split up the compile backend to separately handle cuda graph
        # and torch compile so we can follow the PyTorchConfig here and enable it separately.
        self.ad_config = ad_config
        if ad_config.use_cuda_graph or ad_config.torch_compile_enabled:
            compile_backend = "torch-opt"
        else:
            compile_backend = "torch-simple"
        self.compile_backend = compile_backend
        self.visualize = visualize

        # look up attention op
        self.attention_op = AttentionRegistry.get(self.attn_backend)
        self.mla_op = AttentionRegistry.get(self.mla_backend)

    def __call__(self, cm: CachedSequenceInterface) -> GraphModule:
        """Transform a model into an optimized inference model.

        Args:
            model: The model to transform.
            cp: The cache pool to use for caching.
            args: Example inputs to the model.
            dynamic_shapes: Dynamic shapes to use. Defaults to None.
            poe_config: The config for positional encoding. Defaults to None.
            quantization: The quantization method to use. Defaults to None.

        Returns:
            A GraphModule representing the optimized inference model.
        """
        ############################################################################################
        # INITIALIZE MODEL
        ############################################################################################
        model = self.factory.build_model(device="meta")

        ############################################################################################
        # EXPORT MODEL TO GRAPH MODULE
        ############################################################################################

        cm.info._set_example_sequence()
        egm = torch_export_to_gm(
            model, args=cm.args_original, dynamic_shapes=cm.original_dynamic_shapes
        )
        del model
        ad_logger.debug("original graph: " + str(egm))
        local_rank, world_size = dist_ad.get_rank_world_size()

        ############################################################################################
        # RUN PATTERN MATCHER TRANSFORMATIONS TO STANDARDIZE GRAPH REPRESENTATION
        ############################################################################################

        # quantization
        egm = quantize(egm, self.factory.get_quant_config())

        # Match MoE pattern
        egm = match_moe_pattern(egm)

        # Match repeat_kv pattern
        egm = match_repeat_kv(egm)

        # Match eager attention pattern
        egm = match_eager_attention(egm)

        # Match grouped attention pattern
        egm = match_grouped_attention(egm)

        # Match and optimize causal attention masks
        egm = match_causal_attn_mask(egm)

        # Match attention layout expected by our backend
        egm = match_attention_layout(egm, self.attention_op)

        # Match rope
        # TODO (lucaslie): let's move this to perf optimization once TP sharding is improved
        # see https://github.com/NVIDIA/TensorRT-LLM/pull/3668#discussion_r2052714528
        egm = match_rope_v1(egm)
        egm = match_rope_v2(egm)

        ############################################################################################
        # RUN TRANSFORMATIONS ON STANDARDIZED GRAPH REPRESENTATION
        ############################################################################################

        # eliminate redundant transpose operations
        egm = eliminate_redundant_transposes(egm)

        # run TP sharding across ranks
        egm = column_row_shard(egm, local_rank, world_size)

        # run EP sharding across ranks
        egm = ep_shard(egm, local_rank, world_size)

        # let's run a shape propagation pass to update the graph with correct meta values for
        # subsequent optimization passes
        egm = canonicalize_graph(egm, shape_prop=True)

        ############################################################################################
        # MOVE MODEL AND LOAD WEIGHTS
        ############################################################################################

        # load weights
        self.factory.load_or_random_init(egm, mmap=True, map_location=cm.device)

        # move remaining parts to device
        move_to_device(egm, cm.device)
        cm.to(cm.device)

        ############################################################################################
        # RUN POST-LOAD FUSION AND OPTIMIZATIONS
        ############################################################################################

        # run MoE fusion
        egm = fuse_moe(egm)

        # run GEMM fusion
        egm = fuse_gemms(egm)

        # check if we can fuse allreduce, residual and rmsnorm
        egm = fuse_allreduce_residual_rmsnorm(egm)

        # check if we can fuse collectives
        egm = fuse_collectives(egm)

        # visualize the final graph
        if self.visualize:
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
        # HANDLE CACHES
        ############################################################################################

        input_nodes = check_in_out_nodes(egm)

        # detect attention op and replace with cache-aware op
        for attn_descriptor in [self.attention_op, self.mla_op]:
            egm = insert_cached_attention(
                egm, cm, attn_descriptor, self.factory.get_cache_config(), input_nodes
            )

        # initialize cache on correct device
        cm.initialize_caches()

        # Free memory ratio is hardcoded to 0.8 for now to ensure we have enough memory for graph
        # capture.
        resize_kv_cache(egm, cm, free_mem_ratio=0.8)

        ############################################################################################
        # COMPILE MODEL
        ############################################################################################

        cm.info._set_generate_only_batch()
        compiler_kwargs = {
            "cuda_graph_batch_sizes": self.ad_config.cuda_graph_batch_sizes,
            "num_batched_inputs": 2,  # TODO (lucaslie): improve once we have a config system...
        }
        egm_compiled = compile_and_capture(
            egm,
            self.compile_backend,
            args=cm.args,
            dynamic_shapes=cm.dynamic_shapes,
            compiler_kwargs=compiler_kwargs,
        )
        cm.info.reset()

        torch.cuda.empty_cache()
        gc.collect()
        return egm_compiled
