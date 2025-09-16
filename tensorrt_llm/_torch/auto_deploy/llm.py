import types
from typing import Any, Dict, List, Optional, Tuple

import torch

from ...executor.result import CompletionOutput
from ...inputs.registry import DefaultInputProcessor, ExtraProcessedInputs
from ...llmapi.llm import RequestOutput, _TorchLLM
from ...llmapi.tokenizer import TokenizerBase, TransformersTokenizer, tokenizer_factory
from ...sampling_params import SamplingParams
from .distributed import common as dist_ad
from .llm_args import LlmArgs
from .models.factory import ModelFactory
from .shim.demollm import DemoGenerationExecutor


class ADInputProcessor(DefaultInputProcessor):
    """Input processor for AutoDeploy backend.

    This is a wrapper to either support standard TRT-LLM text-only input processing or use HF's
    message chat template system to process multimodal inputs.
    """

    def __init__(self, tokenizer: Optional[TokenizerBase], processor: Optional[Any] = None):
        super().__init__(model_path=None, model_config=None, tokenizer=tokenizer)
        # NOTE: HF's tokenizer/processor that has the apply_chat_template method
        self.processor = processor or getattr(tokenizer, "tokenizer", None)

    def __call__(
        self, inputs: Dict[str, Any], sampling_params: SamplingParams
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        if self.processor is None:
            raise ValueError("processor is required to tokenize inputs")

        # construct kwargs to reflect DefaultInputProcessor
        kwargs = {
            "add_special_tokens": sampling_params.add_special_tokens,
        }
        if sampling_params.truncate_prompt_tokens is not None:
            kwargs = {
                "truncation": True,
                "max_length": sampling_params.truncate_prompt_tokens,
            }

        # check for messages field and if yes, use the apply_chat_template method
        if "messages" in inputs:
            # TODO: we don't really need this but it makes for a good sanity check. Consider
            # removing this in the future if we need to speed things up.
            prompt = self.processor.apply_chat_template(
                inputs["messages"],
                add_generation_prompt=True,
                tokenize=False,
            )
            inputs["prompt"] = prompt

            all_args = self.processor.apply_chat_template(
                inputs["messages"],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=False,  # there shouldn't be a need for padding ever...
                return_attention_mask=False,
                **kwargs,
            )
            # TODO: is there a more reliable way to avoid the attention_mask here?
            all_args.pop("attention_mask", None)

            # TODO: can we avoid the extra tolist() here eventually?
            token_ids = all_args.pop("input_ids")
            assert token_ids.shape[0] == 1, "messages should be unbatched at this point."
            if all_args:
                extra_processed_inputs = {"multimodal_data": all_args}
            else:
                extra_processed_inputs = None
            return token_ids[0].tolist(), extra_processed_inputs
        elif "multi_modal_data" in inputs:
            return self._mistral_process_inputs(inputs)
        else:
            token_ids = self.tokenizer.encode(inputs["prompt"], **kwargs)
            extra_processed_inputs = None
            if "multi_modal_data" in inputs:
                extra_processed_inputs = {"multimodal_data": inputs["multi_modal_data"]}
            return token_ids, extra_processed_inputs

    def _mistral_process_inputs(self, inputs):
        images = inputs.get("multi_modal_data", {}).get("image")
        do_rescale = self.processor.image_processor.do_rescale
        if images is not None and isinstance(images[0], torch.Tensor):
            # The default multimodal input loader will normalize images to [0, 1] when the requested
            # format is "pt" (pytorch tensors), but not for "pil" (PIL images).
            do_rescale = False

        processed = self.processor(
            text=inputs["prompt"],
            images=images,
            do_rescale=do_rescale,
        )
        input_ids = processed.pop("input_ids").tolist()[0]
        # Remaining in `processed`:
        # * "attention_mask": [B, num_input_tokens]
        # * "pixel_values": [B, C, H, W]
        # * "image_sizes": [B, 2]
        extra_processed_inputs = None
        pixel_values = processed.get("pixel_values")
        if pixel_values is not None:
            # We have no use for the `attention_mask`.
            processed.pop("attention_mask")
            # `image_sizes` is a `[B, 2]` tensor indicating the height and width of each image in the
            # request. If we keep it as a regular tensor, it would get converted to a CUDA tensor before
            # reaching the model forward. Since its values are used to infer the amount of padding
            # + slice the patch embeddings, this would incur a D2H copy. We therefore convert it to a
            # list here to avoid this.
            # processed["image_sizes"] = processed["image_sizes"].tolist()
            # NOTE: `processed` is a dict-like object, but not actually a dict.
            extra_processed_inputs = {
                "multimodal_data": {
                    **processed
                    # "image": {
                    #     **processed
                    # }
                }
            }

        return input_ids, extra_processed_inputs


class LLM(_TorchLLM):
    """LLM class is the main class for running an LLM model using AutoDeploy backend."""

    args: LlmArgs
    _factory: ModelFactory

    @property
    def factory(self) -> ModelFactory:
        if not getattr(self, "_factory", None):
            self._factory = self.args.create_factory()
        return self._factory

    def __init__(self, *args, **kwargs):
        kwargs["backend"] = "_autodeploy"
        super().__init__(*args, **kwargs)

    def _try_load_tokenizer(self) -> Optional[TokenizerBase]:
        if self.args.skip_tokenizer_init:
            return None

        return tokenizer_factory(self.factory.init_tokenizer())

    def _validate_args_for_torch_backend(self, kwargs: dict) -> None:
        """We don't need to validate args for AutoDeploy backend for now."""
        pass

    def _create_input_processor(self) -> ADInputProcessor:
        return ADInputProcessor(self.tokenizer, self.factory.init_processor())

    def _prefetch_model(self):
        """Prefetch the model for the LLM."""
        self.factory.prefetch_checkpoint()

    def _build_model(self):
        """Build the model for the LLM.

        This is a wrapper around the regular build model method that prefetches the model with the
        factory.
        """
        # prefetch model with factory
        self._prefetch_model()

        # NOTE (lucaslie): do regular build model, we bypass the regular LLM CachedModelLoader in
        # _autodeploy backend.
        super()._build_model()

        # now correct input processor
        assert isinstance(self.input_processor, DefaultInputProcessor)
        assert self.tokenizer is None or isinstance(self.tokenizer, TransformersTokenizer)
        self.input_processor = self._create_input_processor()


class DemoLLM(LLM):
    """A simple LLM class to demo the LLM interface while debugging the e2e workflow.

    This is a very simple implementation of an LLM class that can be hacked and used for debugging.
    """

    def __init__(self, **kwargs):
        self.args: LlmArgs = LlmArgs.from_kwargs(**kwargs)

        self.mpi_session = None
        self.runtime_context = None

        # prefetch model and load tokenizer
        self._prefetch_model()
        self._tokenizer = self._try_load_tokenizer()
        self.input_processor = self._create_input_processor()

        # construct demo executor + engine
        self._executor = DemoGenerationExecutor(
            world_size=self.args.world_size,
            tokenizer=self.tokenizer,
            ad_config=self.args.get_pytorch_backend_config(),
        )

    def __del__(self):
        """Ensure proper cleanup of distributed resources."""
        if hasattr(self, "_executor") and self._executor is not None:
            self._executor.shutdown()
        # Call cleanup to ensure process group is properly destroyed
        dist_ad.cleanup()

    @staticmethod
    def _handle_response(request_output: RequestOutput, response: List[CompletionOutput]):
        request_output._done = True
        gen_request = request_output._generation_request
        for i, out in enumerate(response):
            out.text = request_output.tokenizer.decode(
                out.token_ids,
                skip_special_tokens=gen_request.sampling_params.skip_special_tokens,
                spaces_between_special_tokens=gen_request.sampling_params.spaces_between_special_tokens,
            )
            request_output._context_logits = out._postprocess_result["context_logits"]
            request_output._outputs[i] = out

    def generate_async(self, *args, **kwargs) -> RequestOutput:
        request_output = super().generate_async(*args, **kwargs)

        # patch the handle_output method for our use case
        request_output._handle_response = types.MethodType(self._handle_response, request_output)

        return request_output
