from typing import (Any, Callable, Dict, List, Optional, Protocol, Tuple, Type,
                    TypeVar)

from torch import nn

from .._utils import nvtx_range_debug
from ..logger import logger
from ..sampling_params import SamplingParams
from .data import TextPrompt
from .multimodal import (MultimodalInput, apply_mm_hashes, default_hasher,
                         find_mm_token_lengths, find_mm_token_positions,
                         hexdigest_to_int32, validate_mm_inputs)
from .utils import ALL_SUPPORTED_MULTIMODAL_MODELS

N = TypeVar("N", bound=Type[nn.Module])

ExtraProcessedInputs = Dict[str, Any]


class InputProcessor(Protocol):
    """
    Protocol for InputProcessor classes.
    InputProcessor's functions are more relevant to multimodal use cases:
        - Preprocess: extra steps to manipulate the prompts.
        - Forward: the main logic to process the inputs. In multimodal cases, this may run a multimodal encoder model.
        - Postprocess: extra steps to manipulate the outputs
    Model-specific implementation should:
        - Inherit this class and implement the forward() method.
        - Register the inherited class to the model class using @register_input_processor(...)
    """

    model_path: any
    model_config: any
    tokenizer: any

    def __call__(
        self, inputs: TextPrompt, sampling_params: SamplingParams
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        ...


class DefaultInputProcessor(InputProcessor):
    """Preprocess the inputs to the model."""

    def __init__(self,
                 model_path,
                 model_config,
                 tokenizer,
                 trust_remote_code: bool = True) -> None:
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.model_path = model_path

    def __call__(
        self, inputs: TextPrompt, sampling_params: SamplingParams
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        """The default input processor handles only tokenization."""
        if self.tokenizer is None:
            raise ValueError("tokenizer is required to tokenize string prompt")
        kwargs = {}
        if sampling_params.truncate_prompt_tokens is not None:
            kwargs = dict(truncation=True,
                          max_length=sampling_params.truncate_prompt_tokens)
        toktoken_special_tokens = {
            "<|startoftext|>",
            "<|endoftext|>",
            "<|reserved_200000|>",
            "<|reserved_200001|>",
            "<|return|>",
            "<|constrain|>",
            "<|reserved_200004|>",
            "<|channel|>",
            "<|start|>",
            "<|end|>",
            "<|message|>",
            "<|reserved_200009|>",
            "<|reserved_200010|>",
            "<|reserved_200011|>",
            "<|call|>",
            "<|reserved_200013|>",
        }
        with nvtx_range_debug("tokenize prompt"):
            try:
                token_ids = self.tokenizer.encode(
                    inputs["prompt"],
                    add_special_tokens=sampling_params.add_special_tokens,
                    **kwargs)
            except:
                # Tiktoken path
                token_ids = self.tokenizer.encode(
                    inputs["prompt"], allowed_special=toktoken_special_tokens)

        if "query" in inputs:
            with nvtx_range_debug("tokenize query"):
                try:
                    query_token_ids = self.tokenizer.encode(
                        inputs["query"],
                        add_special_tokens=sampling_params.add_special_tokens,
                        **kwargs)
                except:
                    # Tiktoken path
                    query_token_ids = self.tokenizer.encode(
                        inputs["query"],
                        allowed_special=toktoken_special_tokens)

            return token_ids, {"query_token_ids": query_token_ids}

        return token_ids, None


class InputProcessorRegistry:

    def __init__(self) -> None:
        self._input_processors_cls_by_model_type: Dict[
            Type[nn.Module], Type[InputProcessor]] = {}


INPUT_PROCESSOR_REGISTRY = InputProcessorRegistry()


def register_input_processor(processor_cls: Type[InputProcessor],
                             model_type: str,
                             out_of_tree: bool = False):
    """
    Register an input processor to a model class.
    NOTE:
        1. Since this API is only used for multimodal models, we are checking
           the model type only for that.
        2. If this is used for other models in the future, this logic needs to be
           updated e.g. adding another version of this API without the model_type.
        3. If the model is not in the tree, user needs to set out_of_tree to True
           to bypass the model type check and provide their own input preparation.
    """

    def wrapper(model_cls: N) -> N:
        INPUT_PROCESSOR_REGISTRY._input_processors_cls_by_model_type[
            model_cls] = processor_cls
        if not out_of_tree:
            assert model_type in ALL_SUPPORTED_MULTIMODAL_MODELS, \
                f"Model type {model_type} not in {ALL_SUPPORTED_MULTIMODAL_MODELS}.\n" \
                "Please see the tensorrt_llm/inputs/utils.py file for more information."

        return model_cls

    return wrapper


def create_input_processor(model_path_or_dir: str, tokenizer):
    """
    Create an input processor for a specific model.
    """
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models import get_model_architecture

    model_config = None
    try:
        config = ModelConfig.from_pretrained(model_path_or_dir,
                                             trust_remote_code=True)
        model_config = config.pretrained_config
    except (ValueError, EnvironmentError):
        config = None

    if model_config is not None:
        try:
            model_cls, _ = get_model_architecture(model_config)
            input_processor_cls = INPUT_PROCESSOR_REGISTRY._input_processors_cls_by_model_type \
                .get(model_cls)
        except RuntimeError:  # unregistered model
            logger.info("Unregistered model, using DefaultInputProcessor")
            input_processor_cls = None
        if input_processor_cls is not None:
            return input_processor_cls(model_path_or_dir,
                                       model_config,
                                       tokenizer,
                                       trust_remote_code=True)

    return DefaultInputProcessor(None, None, tokenizer)


def create_input_processor_with_hash(
    input_processor: InputProcessor,
    hash_lib=default_hasher,
) -> Callable[[TextPrompt, SamplingParams], Tuple[
        List[int], Optional[ExtraProcessedInputs]]]:
    """Creates a modified processor that applies additional logic like (hashing, find mm chunk positions) to the input processor

    Args:
        original_processor: The original input processor to wrap.
        hash_lib: hasher to use (default: blake3)

    Returns:
        A wrapped processor that modifies prompts before processing.
    """

    def input_processor_wrapper(
        inputs: TextPrompt, sampling_params: SamplingParams
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        try:
            assert 'multi_modal_data' in inputs, "multi_modal_data must be provided for hashing support."
            mm_data = inputs['multi_modal_data']
            num_mm_tokens = find_mm_token_lengths(mm_data, input_processor)
            if len(num_mm_tokens) > 0:
                mm_hashes = apply_mm_hashes(mm_data, hash_lib)
                prompt_token_ids, extra_processed_inputs = input_processor(
                    inputs, sampling_params)
                start_positions = find_mm_token_positions(
                    input_ids=prompt_token_ids,  # token sequence
                    num_mm_tokens=
                    num_mm_tokens,  # list of lengths of each chunk of visual tokens
                    vocab_size=input_processor.model_config.vocab_size,
                )
                # flatten the hashes from dict to a single list
                mm_hashes = [h for hashes in mm_hashes.values() for h in hashes]
                validate_mm_inputs(prompt_token_ids, mm_hashes, start_positions,
                                   num_mm_tokens)
                mm_hashes_int32 = [hexdigest_to_int32(h) for h in mm_hashes
                                   ]  # nested list w/ multiple int32 per hash

                extra_processed_inputs[
                    "multimodal_input"] = MultimodalInput.from_components(
                        mm_hashes_int32, start_positions, num_mm_tokens)
                return prompt_token_ids, extra_processed_inputs
            else:
                return input_processor(inputs, sampling_params)
        except Exception as e:
            # Fall back to basic input processor if multimodal processing fails
            logger.warning(
                f"Multimodal hashing failed: {e}. Falling back to basic input processor."
            )
            return input_processor(inputs, sampling_params)

    return input_processor_wrapper
