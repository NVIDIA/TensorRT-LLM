"""Weight loader for diffusion models."""

import json
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Union

import psutil
import torch
import tqdm

from tensorrt_llm._torch.models.checkpoints.base_weight_loader import BaseWeightLoader
from tensorrt_llm._torch.visual_gen.pipeline_registry import PipelineComponent
from tensorrt_llm._utils import local_mpi_barrier, local_mpi_rank, local_mpi_size
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

_PREFETCH_CHUNK_SIZE = 16 * 1024 * 1024


def _prefetch_file(file_name: str) -> None:
    if not os.path.exists(file_name):
        return

    logger.info(f"Prefetching checkpoint file {file_name} to host page cache...")
    with open(file_name, "rb") as f:
        while f.read(_PREFETCH_CHUNK_SIZE):
            pass
    logger.info(f"Finished prefetching checkpoint file {file_name}.")


def _prefetch_safetensors_files(file_names: List[str]) -> None:
    """Warm safetensors files in host page cache across local ranks."""
    paths: List[str] = []
    seen = set()
    for file_name in file_names:
        path = os.path.abspath(file_name)
        if path not in seen:
            paths.append(path)
            seen.add(path)

    if not paths:
        return

    try:
        prefetch_size = sum(os.path.getsize(path) for path in paths if os.path.exists(path))
        available_memory = psutil.virtual_memory().available
        if prefetch_size >= available_memory * 0.9:
            logger.info(
                "Skipping visual-gen checkpoint prefetch because files require "
                f"{prefetch_size / (1024**3):.2f}GB and available host memory is "
                f"{available_memory / (1024**3):.2f}GB."
            )
            return

        local_paths = paths[local_mpi_rank() :: local_mpi_size()]
        if local_paths:
            logger.info(
                f"Prefetching {prefetch_size / (1024**3):.2f}GB visual-gen "
                "checkpoint files across local ranks."
            )
            max_workers = min(multiprocessing.cpu_count() * 2, 16, len(local_paths))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                list(executor.map(_prefetch_file, local_paths))
    finally:
        local_mpi_barrier()


class WeightLoader(BaseWeightLoader):
    """
    Weight loader for diffusion models.

    Loads weights from safetensors/bin files, similar to HfWeightLoader
    but simpler (no parallel loading optimization for now).

    Supports loading multiple components (e.g., transformer and transformer_2):
        loader = WeightLoader(components=["transformer", "transformer_2"])
        weights = loader.load_weights(ckpt_dir, mapping)
        # Returns: {"transformer": {...}, "transformer_2": {...}}
    """

    def __init__(self, components: Union[str, List[str]] = PipelineComponent.TRANSFORMER):
        """
        Args:
            components: Component(s) to load weights for. Can be:
                - Single string: "transformer" (returns flat dict)
                - List of strings: ["transformer", "transformer_2"] (returns nested dict)
        """
        if isinstance(components, str):
            self.components = [components]
            self.single_component = True
        else:
            self.components = components
            self.single_component = False

    def load_weights(
        self,
        checkpoint_dir: str,
        mapping: Mapping,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Load weights from checkpoint directory.

        Args:
            checkpoint_dir: Path to checkpoint (pipeline root or component dir)
            mapping: Distributed mapping (for future TP/PP support)

        Returns:
            - If single component: Dict mapping weight names to tensors
            - If multiple components: Dict mapping component names to weight dicts
              Example: {"transformer": {...}, "transformer_2": {...}}
        """
        checkpoint_path = Path(checkpoint_dir)

        # Check if this is a pipeline (has model_index.json)
        model_index = checkpoint_path / "model_index.json"
        is_pipeline = model_index.exists()

        # Load weights for each component
        all_weights = {}
        for component in self.components:
            if is_pipeline:
                # Pipeline format: load from component subdirectory
                component_dir = checkpoint_path / component
                if not component_dir.exists():
                    raise ValueError(f"Component '{component}' not found in {checkpoint_dir}")
                weight_dir = component_dir
            else:
                # Standalone model (only valid for single component)
                if len(self.components) > 1:
                    raise ValueError(
                        f"Multiple components specified but {checkpoint_dir} is not a pipeline "
                        "(no model_index.json found)"
                    )
                weight_dir = checkpoint_path

            # Find weight files
            weight_files = self._find_weight_files(weight_dir)
            if not weight_files:
                raise ValueError(f"No weight files found in {weight_dir}")

            if all(wf.endswith(".safetensors") for wf in weight_files):
                _prefetch_safetensors_files(weight_files)

            component_weights = self._load_weight_files(weight_files, component, is_pipeline)
            all_weights[component] = component_weights

        # Return flat dict for single component (backward compatibility)
        if self.single_component:
            return all_weights[self.components[0]]

        # Return nested dict for multiple components
        return all_weights

    def _load_weight_files(
        self, weight_files: List[str], component: str, is_pipeline: bool
    ) -> Dict[str, Any]:
        desc = f"Loading {component}" if is_pipeline else "Loading checkpoint"
        if len(weight_files) <= 1:
            component_weights = {}
            for wf in tqdm.tqdm(weight_files, desc=desc):
                component_weights.update(self._load_file(wf))
            return component_weights

        workers = min(4, len(weight_files))

        logger.info(f"Loading {len(weight_files)} {component} shard files with {workers} workers")
        component_weights = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self._load_file, wf): wf for wf in weight_files}
            for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc=desc):
                wf = futures[future]
                try:
                    loaded = future.result()
                except Exception as exc:
                    raise RuntimeError(f"Failed to load weight file {wf}") from exc
                component_weights.update(loaded)

        return component_weights

    def _find_weight_files(self, weight_dir) -> List[str]:
        """Find safetensors or bin weight files.

        Handles:
        - Single safetensors file
        - Sharded safetensors with index.json
        - PyTorch bin/pth files
        """
        weight_dir = Path(weight_dir)

        # Check for sharded safetensors index
        index_file = weight_dir / "diffusion_pytorch_model.safetensors.index.json"
        if not index_file.exists():
            index_file = weight_dir / "model.safetensors.index.json"

        if index_file.exists():
            # Sharded safetensors: read index to get all shard files
            with open(index_file) as f:
                index = json.load(f)
            shard_files = set(index.get("weight_map", {}).values())
            return sorted([str(weight_dir / f) for f in shard_files])

        # Single safetensors file
        files = list(weight_dir.glob("*.safetensors"))
        if files:
            # Filter out consolidated if multiple files exist
            if len(files) > 1:
                files = [f for f in files if "consolidated" not in f.name]
            return sorted([str(f) for f in files])

        # Fallback to bin
        files = list(weight_dir.glob("*.bin"))
        if files:
            return sorted([str(f) for f in files])

        # Fallback to pth
        files = list(weight_dir.glob("*.pth"))
        return sorted([str(f) for f in files])

    def _load_file(self, filepath: str) -> Dict[str, Any]:
        """Load weights from a single file."""
        logger.debug(f"Loading {filepath}")
        if filepath.endswith(".safetensors"):
            from safetensors.torch import load_file

            return load_file(filepath)
        else:
            return torch.load(filepath, map_location="cpu", weights_only=True)
