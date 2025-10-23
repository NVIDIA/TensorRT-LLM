from pathlib import Path
from typing import Any, List, Literal, Optional, Sequence, Union

from tqdm import tqdm

from tensorrt_llm._utils import nvtx_range_debug
from tensorrt_llm.inputs import create_input_processor, prompt_inputs
from tensorrt_llm.inputs.data import PromptInputs
from tensorrt_llm.sampling_params import SamplingParams

from .llm import BaseLLM, RequestOutput, _TorchLLM
from .llm_args import TorchLlmArgs
from .mpi_session import external_mpi_comm_available


class MultimodalEncoder(_TorchLLM):
    """MultimodalEncoder class is the main class for running a multimodal encoder model using PyTorch backend.
"""

    def __init__(self,
                 model: Union[str, Path],
                 trust_remote_code: bool = False,
                 tensor_parallel_size: int = 1,
                 dtype: Literal["auto", "float16", "float32",
                                "bfloat16"] = "auto",
                 **kwargs: Any) -> None:

        # Validate that users don't pass LLM-specific or TRT-specific arguments
        self._validate_mm_args_for_torch_backend(kwargs)

        # Set higher default max_num_tokens for multimodal encoder (16384 vs 8192 default)
        # Vision encoders can handle more tokens than text-only models
        # TODO: Make this adaptive based on model-specific max_mm_token_length (see _test_llm_multimodal_general)
        if 'max_num_tokens' not in kwargs or kwargs['max_num_tokens'] is None:
            kwargs['max_num_tokens'] = 16384

        super().__init__(model,
                         trust_remote_code=trust_remote_code,
                         tensor_parallel_size=tensor_parallel_size,
                         dtype=dtype,
                         **kwargs)

    def _build_model(self):
        BaseLLM._build_model(self)
        assert self._engine_dir is None

        # Tokenizer loading should be after calling model_loader(), since model_loader() may download the model from HF hub.
        # It should also be before bindings ExecutorConfig, which may depend on tokenizer info.
        self._tokenizer = self._try_load_tokenizer()

        # Multimodal special handling:
        # 1. Default load_tokenizer may fail because MM has different tokenizer configuration. Hence we initialize it inside input processor
        # 2. May need to modify model weights for MM (e.g., resize vocab embedding). We must do such operation via input processor's __init__
        checkpoint_format = getattr(self.args, "checkpoint_format", None)
        self.input_processor = create_input_processor(self._hf_model_dir,
                                                      self.tokenizer,
                                                      checkpoint_format)
        self._tokenizer = self.input_processor.tokenizer

        assert isinstance(self.args, TorchLlmArgs)
        self.args.mm_encoder_only = True

        self._executor = self._executor_cls.create(
            self._engine_dir,
            executor_config=None,
            model_world_size=self.args.parallel_config.world_size,
            mpi_session=self.mpi_session,
            reuse_mpi_comm=external_mpi_comm_available(
                self.args.parallel_config.world_size),
            is_llm_executor=True,  # TODO: check if this is correct or needed
            hf_model_dir=self._hf_model_dir,
            tokenizer=self.tokenizer,
            llm_args=self.args)

    def _validate_mm_args_for_torch_backend(self, kwargs: dict) -> None:
        """Validate that users don't pass LLM-specific arguments when using MultimodalEncoder (PyTorch).
        Placeholder for now.
        """

    def generate(
        self,
        inputs: Union[PromptInputs, Sequence[PromptInputs]],
        use_tqdm: bool = True,
    ) -> Union[RequestOutput, List[RequestOutput]]:
        """Generate output for the given prompts in the synchronous mode.
        Synchronous generation accepts either single prompt or batched prompts.

        Args:
            inputs (tensorrt_llm.inputs.data.PromptInputs, Sequence[tensorrt_llm.inputs.data.PromptInputs]): The prompt text or token ids.
                It can be single prompt or batched prompts.
        Returns:
            Union[tensorrt_llm.llmapi.RequestOutput, List[tensorrt_llm.llmapi.RequestOutput]]: The output data of the completion request to the LLM.
        """
        unbatched = not isinstance(inputs, list)
        if not unbatched:
            if isinstance(inputs[0], int):
                unbatched = True

        if unbatched:
            inputs = [inputs]

        inputs = [prompt_inputs(i) for i in inputs]

        def _item_at(maybe_batched: Union[Any, Sequence[Any]], pos: int) -> Any:
            if isinstance(maybe_batched, list):
                return maybe_batched[pos]
            else:
                return maybe_batched

        futures = []
        for i, request_inputs in enumerate(inputs):
            future = self.generate_async(request_inputs)
            futures.append(future)

        for future in tqdm(futures,
                           desc="Processed requests",
                           dynamic_ncols=True,
                           disable=not use_tqdm):
            future.result()

        if unbatched:
            futures = futures[0]

        return futures

    @nvtx_range_debug("MM_encoder.generate_async",
                      color="green",
                      category="VisionEncoder")
    def generate_async(
        self,
        inputs: PromptInputs,
        sampling_params: Optional[SamplingParams] = None,
    ) -> RequestOutput:
        """Generate output for the given multimodal request in the asynchronous mode.
        Asynchronous generation accepts single multimodal request only.

        Returns:
            Future that resolves to tensorrt_llm.llmapi.RequestOutput containing mm_embeddings
        """
        result = super().generate_async(inputs, sampling_params)
        # TODO: possible postprocess the result for disaggregated serving
        return result
