"""High-level entrypoint to transform a model into an efficient inference model."""

import torch
from torch.fx import GraphModule

from ..compile import compile_and_capture
from ..custom_ops.attention_interface import AttentionRegistry
from ..distributed import common as dist_ad
from ..models.factory import ModelFactory
from ..shim.interface import CachedSequenceInterface
from ..utils.logger import ad_logger
from ..utils.quantization_utils import is_quantized_graph
from ._graph import move_to_device
from .export import torch_export_to_gm
from .library import (
    column_row_shard_matmul_v3,
    fuse_allreduce_residual_rmsnorm,
    fuse_collectives,
    fuse_gemms,
    insert_mha_with_kv_cache,
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
        poe_config = self.factory.get_positional_encoding_config(model)

        ############################################################################################
        # EXPORT MODEL TO GRAPH MODULE
        ############################################################################################

        cm.info._set_example_sequence()
        egm = torch_export_to_gm(model, args=cm.args[:1], dynamic_shapes=cm.dynamic_shapes[:1])
        del model
        ad_logger.debug("original graph: " + str(egm))
        local_rank, world_size = dist_ad.get_rank_world_size()

        ############################################################################################
        # RUN OPTIMIZATIONS ONE-BY-ONE
        ############################################################################################

        # quantization
        quantization = self.factory.get_quantization()
        quantized_graph = is_quantized_graph(egm)
        if quantization or quantized_graph:
            egm = quantize(
                egm,
                quantization.get("quant_algo", None) if quantization else None,
                skip=quantization.get("exclude_modules", []) if quantization else [],
                is_quantized_graph=quantized_graph,
            )

        # run sharding across ranks
        # NOTE (lliebenwein): sharding will cause the graph's metadata to be out-of-sync.
        # Specifically, torch.export tracks the metadata of each node in the graph and stores it in
        # `node.meta`. We are re-exporting the graph after sharding to update the metadata!
        egm = column_row_shard_matmul_v3(egm, local_rank, world_size)

        egm = torch_export_to_gm(egm, args=cm.args[:1], dynamic_shapes=cm.dynamic_shapes[:1])

        # setup kv-cache (we are adding buffers - hence there might be missing keys)
        rope_theta = (poe_config or {}).get("rope_theta")
        if quantization is not None and "kv_cache_quant_algo" in quantization.keys():
            kv_cache_format = quantization.get("kv_cache_quant_algo", None)
            if kv_cache_format is not None:
                assert kv_cache_format == "FP8", (
                    f"KV cache quantization format {kv_cache_format} is not supported."
                )
            kv_cache_dtype = torch.float8_e4m3fn if kv_cache_format is not None else None
        else:
            kv_cache_dtype = None

        egm = insert_mha_with_kv_cache(egm, cm, self.attention_op, rope_theta, kv_cache_dtype)

        ############################################################################################
        # SETUP CACHES AND LOAD WEIGHTS
        ############################################################################################
        cm.initialize_caches()
        self.factory.load_or_random_init(egm, mmap=True, map_location=cm.device)
        move_to_device(egm, cm.device)

        ############################################################################################
        # RUN POST-LOAD OPTIMIZATIONS
        ############################################################################################

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

        egm_compiled = compile_and_capture(
            egm, self.compile_backend, args=cm.args, dynamic_shapes=cm.dynamic_shapes
        )
        cm.info.reset()
        return egm_compiled
