"""High-level entrypoint to transform a model into an efficient inference model."""

from torch.fx import GraphModule

from ..compile import compile_and_capture
from ..custom_ops.attention_interface import AttentionRegistry
from ..distributed import common as dist_ad
from ..models.factory import ModelFactory
from ..shim.interface import CachedSequenceInterface
from ..utils.logger import ad_logger
from ._graph import move_to_device
from .export import torch_export_to_gm
from .library import (
    column_row_shard,
    ep_shard,
    fuse_allreduce_residual_rmsnorm,
    fuse_collectives,
    fuse_gemms,
    fuse_moe,
    identify_and_fuse_mha,
    insert_mha_with_kv_cache,
    match_moe_pattern,
    quantize,
)


class InferenceOptimizer:
    def __init__(
        self,
        factory: ModelFactory,
        *,  # TODO (lliebenwein): temporary until we have a better config system
        attn_backend: str,
        compile_backend: str,
        visualize: bool = False,
    ):
        self.factory = factory
        self.attn_backend = attn_backend
        self.compile_backend = compile_backend
        self.visualize = visualize

        # look up attention op
        self.attention_op = AttentionRegistry.get(self.attn_backend)

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
        egm = torch_export_to_gm(model, args=cm.args[:1], dynamic_shapes=cm.dynamic_shapes[:1])
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

        # identify MHA patterns
        egm = identify_and_fuse_mha(egm, self.factory.get_positional_embedding_config())

        ############################################################################################
        # RUN TRANSFORMATIONS ON STANDARDIZED GRAPH REPRESENTATION
        ############################################################################################

        # insert MHA with KV cache
        egm = insert_mha_with_kv_cache(egm, cm, self.attention_op, self.factory.get_cache_config())

        # run TP sharding across ranks
        egm = column_row_shard(egm, local_rank, world_size)

        # run EP sharding across ranks
        egm = ep_shard(egm, local_rank, world_size)

        ############################################################################################
        # SETUP CACHES AND LOAD WEIGHTS
        ############################################################################################

        # initialize caches, load weights, and map to correct device
        cm.initialize_caches()

        self.factory.load_or_random_init(egm, mmap=True, map_location=cm.device)
        move_to_device(egm, cm.device)

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
        # COMPILE MODEL
        ############################################################################################

        cm.info._set_generate_only_batch()
        egm_compiled = compile_and_capture(
            egm, self.compile_backend, args=cm.args, dynamic_shapes=cm.dynamic_shapes
        )
        cm.info.reset()

        return egm_compiled
