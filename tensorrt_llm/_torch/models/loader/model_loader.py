import glob
import os
from typing import Any, Dict, List, Optional

from tensorrt_llm.logger import logger

from .file_system_weights_loader_interface import \
    FileSystemWeightsLoaderInterface
from .weights_mapper_interface import TRTLLMStateDict, WeightsMapperInterface


class ModelLoader:
    """
    Loads a model from a checkpoint path using a specified weights loader and mapper.
    """

    def __init__(self,
                 checkpoint_path: str,
                 weights_loader: FileSystemWeightsLoaderInterface,
                 weights_mapper: Optional[WeightsMapperInterface] = None):
        """
        Args:
            checkpoint_path: Path to the model checkpoint directory or file.
            weights_loader: An instance of a FileSystemWeightsLoaderInterface to load raw weights.
            weights_mapper: An optional instance of a WeightsMapperInterface to transform weights.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint path {checkpoint_path} does not exist.")

        self.checkpoint_path = checkpoint_path
        self.weights_loader = weights_loader
        self.weights_mapper = weights_mapper

    def _find_weight_files(self) -> List[str]:
        """
        Finds all relevant weight files in the checkpoint_path.
        Currently looks for .safetensors files if path is a directory,
        or returns the path itself if it's a file.
        This can be expanded to support .bin, .pth etc.
        """
        if os.path.isdir(self.checkpoint_path):
            # Adjust glob pattern as needed for other file types
            files = glob.glob(
                os.path.join(self.checkpoint_path, "*.safetensors"))
            # Could add other patterns: glob.glob(os.path.join(self.checkpoint_path, "*.bin"))
            if not files:
                logger.warning(
                    f"No weight files (e.g., *.safetensors) found in directory: {self.checkpoint_path}"
                )
            return sorted(
                files)  # Sorting ensures consistent loading order if important
        elif os.path.isfile(self.checkpoint_path):
            return [self.checkpoint_path]
        else:
            logger.error(
                f"Checkpoint path {self.checkpoint_path} is not a valid file or directory."
            )
            return []

    def load(self) -> TRTLLMStateDict:
        """
        Loads the model weights.
        1. Finds relevant weight files.
        2. Uses the weights_loader to load them from the file system.
        3. If a weights_mapper is provided, it uses it to map/transform the weights.
        4. Returns the final state dictionary.
        """
        logger.info(f"Starting model loading from: {self.checkpoint_path}")

        weight_files = self._find_weight_files()
        if not weight_files:
            logger.error("No weight files found. Cannot proceed with loading.")
            return {}

        logger.info(f"Found weight files: {weight_files}")
        raw_weights: Dict[str,
                          Any] = self.weights_loader.load_weights(weight_files)

        if not raw_weights:
            logger.warning("Weights loader returned an empty dictionary.")
            # Depending on strictness, could return here or proceed

        if self.weights_mapper:
            logger.info("Applying weights mapper...")
            # If apply_transformations was used to collect callbacks on the mapper:
            # The mapper's map_weights method should internally apply these.
            # For mappers that might require pre-application of transformations before map_weights:
            # if hasattr(self.weights_mapper, 'apply_all_registered_transformations'):
            #     self.weights_mapper.apply_all_registered_transformations(raw_weights) # Or similar pattern

            mapped_weights = self.weights_mapper.map_weights(raw_weights)
            logger.info("Finished applying weights mapper.")
            return mapped_weights
        else:
            logger.info("No weights mapper provided. Returning raw weights.")
            # Ensure the return type matches TRTLLMStateDict,
            # which might just be Dict[str, Any] if no mapping is done.
            return raw_weights
