# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Union, cast

from tqdm import tqdm
from typing_extensions import NotRequired, TypedDict

from tensorrt_llm._utils import nvtx_range_debug
from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.inputs import create_input_processor
from tensorrt_llm.inputs.data import PromptInputs
from tensorrt_llm.sampling_params import SamplingParams

from .llm import BaseLLM, PreprocessedInputs, RequestOutput, _TorchLLM
from .llm_args import TorchLlmArgs
from .mpi_session import external_mpi_comm_available
from .utils import set_api_status


class MediaOnlyInput(TypedDict):
    """Media to encode independently of a final prompt."""

    multi_modal_data: Dict[str, Any]
    multi_modal_uuids: NotRequired[Dict[str, List[Any]]]
    mm_processor_kwargs: NotRequired[Dict[str, Any]]
    mm_item_order: NotRequired[List[Dict[str, Any]]]


EncoderInput = Union[PromptInputs, MediaOnlyInput]


class MultimodalEncoder(_TorchLLM):
    """MultimodalEncoder class is the main class for running a multimodal encoder model using PyTorch backend."""

    def __init__(self,
                 model: Union[str, Path],
                 trust_remote_code: bool = False,
                 tensor_parallel_size: int = 1,
                 dtype: Literal["auto", "float16", "float32",
                                "bfloat16"] = "auto",
                 **kwargs: Any) -> None:

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
        # Tokenizer loading should be after calling model_loader(), since model_loader() may download the model from HF hub.
        # It should also be before bindings ExecutorConfig, which may depend on tokenizer info.
        self._tokenizer = self._try_load_tokenizer()

        # Multimodal special handling:
        # 1. Default load_tokenizer may fail because MM has different tokenizer configuration. Hence we initialize it inside input processor
        # 2. May need to modify model weights for MM (e.g., resize vocab embedding). We must do such operation via input processor's __init__
        checkpoint_format = getattr(self.args, "checkpoint_format", None)
        trust_remote_code = self.args.trust_remote_code
        input_processor_kwargs = {}
        video_pruning_rate = self.args.multimodal_config.video_pruning_rate
        if video_pruning_rate is not None:
            input_processor_kwargs["video_pruning_rate"] = video_pruning_rate
        if self.args.multimodal_config.vision_gpu_normalize:
            input_processor_kwargs["vision_gpu_normalize"] = True
        self.input_processor = create_input_processor(
            self._hf_model_dir,
            self.tokenizer,
            checkpoint_format,
            trust_remote_code=trust_remote_code,
            **input_processor_kwargs)
        self._tokenizer = self.input_processor.tokenizer

        assert isinstance(self.args, TorchLlmArgs)
        self.args.mm_encoder_only = True

        self._executor = self._executor_cls.create(
            model_world_size=self.args.parallel_config.world_size,
            mpi_session=self.mpi_session,
            reuse_mpi_comm=external_mpi_comm_available(
                self.args.parallel_config.world_size),
            is_llm_executor=True,
            hf_model_dir=self._hf_model_dir,
            tokenizer=self.tokenizer,
            llm_args=self.args)

    def _validate_mm_args_for_torch_backend(self, kwargs: dict) -> None:
        """Validate that users don't pass LLM-specific arguments when using MultimodalEncoder (PyTorch)."""
        if kwargs.get("encode_only") is True:
            raise ValueError(
                "MultimodalEncoder does not support encode_only=True. "
                "It uses mm_encoder_only execution internally.")

    def _validate_args_for_torch_backend(self, kwargs: dict) -> None:
        """Run multimodal validation inside the tracked constructor boundary."""
        self._validate_mm_args_for_torch_backend(kwargs)
        super()._validate_args_for_torch_backend(kwargs)

    def generate(
        self,
        inputs: Union[EncoderInput, Sequence[EncoderInput]],
        use_tqdm: bool = True,
    ) -> Union[RequestOutput, List[RequestOutput]]:
        """Generate output for the given prompts in the synchronous mode.
        Synchronous generation accepts either single prompt or batched prompts.

        Args:
            inputs: One prompt or media-only request, or a sequence of requests.
        Returns:
            Union[tensorrt_llm.llmapi.RequestOutput, List[tensorrt_llm.llmapi.RequestOutput]]: The output data of the completion request to the LLM.
        """
        unbatched = self._is_unbatched_inputs(inputs)
        requests = ([cast(EncoderInput, inputs)] if unbatched else cast(
            Sequence[EncoderInput], inputs))
        futures = [self.generate_async(request) for request in requests]

        for future in tqdm(futures,
                           desc="Processed requests",
                           dynamic_ncols=True,
                           disable=not use_tqdm):
            future.result()

        return futures[0] if unbatched else futures

    @set_api_status("prototype")
    def preprocess(
        self,
        inputs: EncoderInput,
        sampling_params: Optional[SamplingParams] = None,
        disaggregated_params: Optional[DisaggregatedParams] = None,
    ) -> PreprocessedInputs:
        """Prepare a prompted or media-only request for generate_async."""
        # The shared implementation accepts media alone in encoder-only mode.
        return super().preprocess(cast(PromptInputs, inputs), sampling_params,
                                  disaggregated_params)

    @nvtx_range_debug("MM_encoder.generate_async",
                      color="green",
                      category="VisionEncoder")
    def generate_async(
        self,
        inputs: Union[EncoderInput, PreprocessedInputs],
        sampling_params: Optional[SamplingParams] = None,
    ) -> RequestOutput:
        """Generate output for the given multimodal request in the asynchronous mode.
        Asynchronous generation accepts single multimodal request only.

        For models whose input processor sets `supports_mm_item_handoff`, the
        request may hold only `multi_modal_data` (and optionally
        `mm_processor_kwargs`, `multi_modal_uuids` and `mm_item_order`), with
        no prompt. The processor checks the media and sampling settings for
        independent encoding. The result carries one embedding
        handle per item, in placeholder order (`mm_item_order`, else the order
        of `multi_modal_data`), each with the item's metadata and hash, but no
        prompt layout or mRoPE positions. A prefill worker builds those from the
        real prompt and the item metadata.

        Returns:
            Future that resolves to tensorrt_llm.llmapi.RequestOutput containing mm_embeddings
        """
        return super().generate_async(
            cast(Union[PromptInputs, PreprocessedInputs], inputs),
            sampling_params)
