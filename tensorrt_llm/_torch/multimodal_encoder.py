from tensorrt_llm.inputs.data import PromptInputs
from tensorrt_llm.llmapi.llm import BaseLLM, _TorchLLM
from typing import Any, Union
from pathlib import Path


class MultimodalEncoder(_TorchLLM):
    """MultimodalEncoder class is the main class for running a multimodal encoder model using PyTorch backend.

    Parameters:
"""

    def __init__(self,
                 model: Union[str, Path],
                 trust_remote_code: bool = False,
                 tensor_parallel_size: int = 1,
                 dtype: str = "auto",
                 **kwargs: Any) -> None:

        # Validate that users don't pass LLM-specific or TRT-specific arguments
        self._validate_mm_args_for_torch_backend(kwargs)

        super().__init__(model,
                         trust_remote_code=trust_remote_code,
                         tensor_parallel_size=tensor_parallel_size,
                         dtype = dtype,
                         **kwargs)

    def _build_model(self):
        super()._build_model()
        self._executor_config.mm_encoder_only = True

    def _validate_mm_args_for_torch_backend(self, kwargs: dict) -> None:
        """Validate that users don't pass LLM-specific arguments when using MultimodalEncoder (PyTorch).
        Placeholder for now.
        """
        pass

    def generate(
        self,
        inputs: PromptInputs,
    ) -> RequestOutput:
        """Generate embeddings (and other multimodal encoder outputs) for multiple multimodal requests in parallel.

        Args:
            mm_requests: List of multimodal requests to process

        Returns:
            List of generation results
        """
        async def _process_requests():
            # Submit all requests first
            futures = []
            for request in mm_requests:
                future = await self.generate_async(request)
                futures.append(future)

            # Then wait for all results
            results = []
            for future in futures:
                result = await future.aresult()
                results.append(result)
            return results

        # Run the async operations in an event loop
        return asyncio.run(_process_requests())

    @nvtx_range_debug("MM_encoder.generate_async", color="green", category="VisionEncoder")
    def generate_async(
        self,
        inputs: PromptInputs,
        sampling_params: Optional[SamplingParams] = None,
    ) -> RequestOutput:
        """Generate output for the given multimodal request in the asynchronous mode.
        Asynchronous generation accepts single multimodal request only.

        Returns:
            tensorrt_llm.llmapi.RequestOutput: The output data of the completion request to the LLM.
        """
        # TODO: possible preprocess the input
        result = BaseLLM.generate_async(self, inputs, sampling_params)
        # TODO: possible postprocess the result
        return result


