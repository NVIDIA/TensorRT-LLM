from typing import Callable

from .weights_mapper_interface import (StateDict, TRTLLMStateDict,
                                       WeightsMapperInterface)


class MistralHFWeightsMapper(WeightsMapperInterface):
    """
    Maps weights from a Hugging Face Mistral model checkpoint to a TRT-LLM compatible format.
    """

    def __init__(self):
        super().__init__()
        # Initialize any specific configurations or transformation callbacks if needed
        self._transformation_callbacks = []

    def map_weights(self, hf_state_dict: StateDict) -> TRTLLMStateDict:
        """
        Performs the actual weight mapping and applies any registered transformations.

        Args:
            hf_state_dict: The Hugging Face model's state dictionary.

        Returns:
            A TRT-LLM compatible state dictionary.
        """
        trtllm_state_dict = {}
        # Placeholder for Mistral-specific mapping logic
        # This is where you'd iterate through hf_state_dict, rename keys,
        # reshape tensors, split/merge weights for tensor parallelism, etc.
        # Example (very simplified):
        # for key, value in hf_state_dict.items():
        #     new_key = key # replace with actual renaming logic
        #     new_value = value # replace with actual tensor transformation
        #     trtllm_state_dict[new_key] = new_value

        # Apply registered transformations
        # current_dict = hf_state_dict # Or a copy if transformations shouldn't modify original input
        current_dict = hf_state_dict.copy()  # Work on a copy

        for callback in self._transformation_callbacks:
            current_dict = callback(current_dict)

        # The final mapping to trtllm_state_dict would happen here, potentially using
        # the transformed current_dict or combining it with direct mapping logic.
        # For now, this is a placeholder as the detailed mapping logic is TBD.
        # This part needs to be filled based on how Mistral HF weights map to TRT-LLM structure.
        trtllm_state_dict.update(
            current_dict
        )  # Simplified: assumes callbacks transform to final format

        if not trtllm_state_dict and hf_state_dict:
            # If no mapping logic is implemented yet but input was provided,
            # print a warning or log that it's a pass-through for now.
            print(
                "Warning: MistralHFWeightsMapper.map_weights is not yet fully implemented. "
                "Returning a copy of the input state_dict for now.")

        return trtllm_state_dict

    def apply_transformations(
            self, callback: Callable[[StateDict], StateDict]) -> None:
        """
        Registers a transformation callback to be applied during the map_weights call.
        """
        self._transformation_callbacks.append(callback)
