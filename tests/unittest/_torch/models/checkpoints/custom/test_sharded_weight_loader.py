import glob
import os
from pathlib import Path
from typing import Any, List

import psutil
import pytest
import tqdm
from utils.llm_data import llm_models_root

from tensorrt_llm._torch.models.checkpoints import HfWeightLoader
from tensorrt_llm._torch.models.checkpoints.hf.checkpoint_loader import HfCheckpointLoader
from tensorrt_llm._torch.models.checkpoints.hf.config_loader import HfConfigLoader
from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import HfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import (
    register_checkpoint_weight_loader,
    register_mapper,
    run_concurrently,
)
from tensorrt_llm._torch.modules.linear import TensorParallelMode, load_weight_shard
from tensorrt_llm._utils import local_mpi_barrier
from tensorrt_llm.llmapi.llm import LLM
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping


@register_checkpoint_weight_loader("DUMMY_FORMAT")
class ShardedWeightLoader(HfWeightLoader):
    def load_weights(self, checkpoint_dir: str, mapping: Mapping) -> dict[str, Any]:
        weight_files = glob.glob(f"{checkpoint_dir}/*.safetensors")
        # Some model checkpoint directories contain not only the sharded safetensors, but one
        # consolidated tensor. In the presence of both, we favor the former, as there really is no need
        # to prefetch the (usually) ridiculously large consolidated tensor into memory in such a case.
        filtered_weight_files = [
            x for x in weight_files if "consolidated" not in os.path.split(x)[1]
        ]
        if len(filtered_weight_files) > 0:
            weight_files = filtered_weight_files
        if weight_files:
            # Prefetch the weight files to CPU memory if the size is less than 90% of the available memory.
            # This is a heuristic to avoid prefetching files that are too large and causing file cache thrashing.
            prefetch_size = sum(os.path.getsize(file) for file in weight_files)
            # If the layer number is overridden, it indicates that only a subset of layers are loaded.
            # Prefetching all layers is unnecessary.
            num_layers = int(os.environ.get("TLLM_OVERRIDE_LAYER_NUM", "0"))
            enable_prefetch = (
                prefetch_size < psutil.virtual_memory().available * 0.9 and num_layers == 0
            )
            if enable_prefetch:
                logger.info(f"Prefetching {prefetch_size / (1024**3):.2f}GB checkpoint files.")
                self.prefetch_files(weight_files)
                # Ensure that all local ranks have finished prefetching before loading weights
                local_mpi_barrier()

            return self._load_weights_in_parallel(
                weight_files,
                self._load_safetensors_file,
                mapping,
                "Loading safetensors weights in parallel",
            )

        weight_files = glob.glob(f"{checkpoint_dir}/*.bin")
        if not weight_files:
            weight_files = glob.glob(f"{checkpoint_dir}/*.pth")

        if weight_files:
            return self._load_weights_in_parallel(
                weight_files,
                self._load_bin_or_path_file,
                mapping,
                "Loading bin weights in parallel",
            )

        raise RuntimeError(f"No weight files found in {checkpoint_dir}.")

    def _shard_weights(self, weights: dict[str, Any], mapping: Mapping) -> dict[str, Any]:
        tp_size = mapping.tp_size
        tp_rank = mapping.tp_rank

        logger.info(f"Sharding weights for TP size: {tp_size}, TP rank: {tp_rank}")

        for k, v in weights.items():
            tp_mode = None
            if "embed_tokens" in k:
                tp_mode = TensorParallelMode.COLUMN
            elif "lm_head" in k:
                tp_mode = TensorParallelMode.COLUMN
            # elif "norm" in k:
            #     tp_mode = None
            elif "self_attn.k_proj" in k:
                tp_mode = TensorParallelMode.COLUMN
            elif "self_attn.v_proj" in k:
                tp_mode = TensorParallelMode.COLUMN
            elif "self_attn.q_proj" in k:
                tp_mode = TensorParallelMode.COLUMN
            elif "self_attn.o_proj" in k:
                tp_mode = TensorParallelMode.ROW
            elif "mlp.gate_proj" in k:
                tp_mode = TensorParallelMode.COLUMN
            elif "mlp.up_proj" in k:
                tp_mode = TensorParallelMode.COLUMN
            elif "mlp.down_proj" in k:
                tp_mode = TensorParallelMode.ROW

            original_shape = v.shape
            weights[k] = load_weight_shard(v, tp_size, tp_rank, tp_mode)
            logger.info(f"Weight {k} has shape {original_shape} -> {weights[k].shape}")
        return weights

    def _load_weights_in_parallel(
        self, weight_files: List[str], load_func, mapping: Mapping, description: str
    ) -> dict[str, Any]:
        """
        Load weight files in parallel using the specified loading function.

        Args:
            weight_files: List of weight file paths
            load_func: Function to load individual weight files
            description: Description for the progress bar

        Returns:
            Dictionary containing all loaded weights
        """
        weights = {}
        pbar = tqdm.tqdm(total=len(weight_files), desc=description)

        # Note that the function is called with a tuple of arguments,
        # hence we need to wrap the arguments in a tuple via [(w,) for w in weight_files]
        # specifically the comma right after the w is important to make it a tuple.
        run_concurrently(
            load_func, [(w,) for w in weight_files], reduce_func=weights.update, pbar=pbar
        )

        logger.info(f"Loaded {len(weights)} weights")

        return self._shard_weights(weights, mapping)


@register_mapper("DUMMY_FORMAT")
class ShardedWeightMapper(HfWeightMapper):
    already_sharded: bool = True


def test_sharded_safetensors_checkpoint_loader():
    """Test that the checkpoint loader can load a sharded safetensors checkpoint."""

    model_dir = Path(llm_models_root()) / "llama-3.2-models/Llama-3.2-1B"

    # Create LLM with the provided model
    llm = LLM(
        model=model_dir,
        backend="pytorch",
        tensor_parallel_size=2,
        checkpoint_loader=HfCheckpointLoader(
            weight_loader=ShardedWeightLoader(),
            weight_mapper=ShardedWeightMapper(),
            config_loader=HfConfigLoader(),
        ),
    )
    llm.shutdown()


if __name__ == "__main__":
    pytest.main([__file__])
