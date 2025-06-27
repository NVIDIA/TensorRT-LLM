import types
from typing import List, Optional

from ...executor.result import CompletionOutput
from ...inputs.registry import create_input_processor
from ...llmapi.llm import BaseLLM, RequestOutput
from ...llmapi.tokenizer import TokenizerBase, tokenizer_factory
from .distributed import common as dist_ad
from .llm_args import LlmArgs
from .shim.demollm import DemoGenerationExecutor


class LLM(BaseLLM):
    """LLM class is the main class for running an LLM model using AutoDeploy backend."""

    args: LlmArgs

    def __init__(self, *args, **kwargs):
        kwargs["backend"] = "_autodeploy"
        super().__init__(*args, **kwargs)

    def _try_load_tokenizer(self) -> Optional[TokenizerBase]:
        if self.args.skip_tokenizer_init:
            return None

        factory = self.args.create_factory()
        return tokenizer_factory(factory.init_tokenizer())

    @classmethod
    def from_args(cls, args: LlmArgs) -> "LLM":
        """Initialize an AutoDeploy LLM from an AutoDeploy LlmArgs object directly.

        We temporarily patch from_kwargs to correctly return the correct args object.
        """
        # TODO: finish this
        model = args.model
        return cls(model)

    def _prefetch_model(self):
        """Prefetch the model for the LLM."""
        self.args.create_factory().prefetch_checkpoint()

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
        self.input_processor = create_input_processor(None, self.tokenizer)

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
