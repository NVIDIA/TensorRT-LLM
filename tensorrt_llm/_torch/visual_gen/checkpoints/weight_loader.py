"""Weight loader for diffusion models."""

import json
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
import tqdm

from tensorrt_llm._torch.models.checkpoints.base_weight_loader import BaseWeightLoader
from tensorrt_llm._torch.visual_gen.config import PipelineComponent
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping


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

        Supports two layouts:

        * **Pipeline (diffusers)** -- ``model_index.json`` with component
          sub-directories, each containing weight files.
        * **Monolithic safetensors** -- no ``model_index.json``; weight files
          sit in the root directory with prefixed keys (e.g.
          ``transformer.``, ``video_vae.``).  Only keys matching the
          requested component prefix are returned (prefix stripped).

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
                component_dir = checkpoint_path / component
                if not component_dir.exists():
                    raise ValueError(f"Component '{component}' not found in {checkpoint_dir}")
                weight_dir = component_dir
                prefix_filter = None
            else:
                weight_dir = checkpoint_path
                prefix_filter = f"{component}."

            weight_files = self._find_weight_files(weight_dir)
            if not weight_files:
                raise ValueError(f"No weight files found in {weight_dir}")

            component_weights = {}
            desc = f"Loading {component}"
            for wf in tqdm.tqdm(weight_files, desc=desc):
                raw = self._load_file(wf)
                if prefix_filter is not None:
                    for key, tensor in raw.items():
                        if key.startswith(prefix_filter):
                            component_weights[key[len(prefix_filter) :]] = tensor
                else:
                    component_weights.update(raw)

            all_weights[component] = component_weights

        if self.single_component:
            return all_weights[self.components[0]]
        return all_weights

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
