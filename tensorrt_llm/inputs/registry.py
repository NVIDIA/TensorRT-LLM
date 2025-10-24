import enum
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, List, Optional, Protocol, Tuple, Type,
                    TypeVar)

from PIL import Image
from torch import Tensor, nn

from .._utils import nvtx_range_debug
from ..logger import logger
from ..sampling_params import SamplingParams
from .data import TextPrompt
from .multimodal import (MultimodalInput, apply_mm_hashes, default_hasher,
                         find_mm_token_lengths, find_mm_token_positions,
                         hexdigest_to_int32, validate_mm_inputs)

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
    multimodal_hashing_supported: Optional[bool] = None

    def __call__(
        self, inputs: TextPrompt, sampling_params: SamplingParams
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        ...


class BaseDummyInputsBuilder:
    """
    Base class for generating dummy inputs. Specially for profiling
    """

    def get_dummy_prompt(self, input_seq_len: int):
        raise NotImplementedError(
            "Please ensure this method is implemented in your inherited class")


class BaseMultimodalInputProcessor:
    """
    Base class for multimodal input processors with default implementations
    of get_num_tokens_per_image and get_num_tokens_per_video methods.

    This class provides default implementations that work with most AutoProcessor-based
    models. Specific processors can override these methods if they need custom logic.
    """

    def get_processor(self) -> Optional[Any]:
        """Return the processor object if available; otherwise raise NotImplementedError.
        """
        if not hasattr(self, 'processor') and not hasattr(self, '_processor'):
            raise NotImplementedError(
                f"cannot find processor in {self.__class__.__name__}. "
                "Please ensure the processor is stored under self.processor or self._processor."
            )
        return getattr(self, 'processor', getattr(self, '_processor', None))

    def get_vocab_size(self) -> Optional[int]:
        """Return the tokenizer/model vocabulary size if available; otherwise None.

        Resolution order:
        1) self.model_config.vocab_size
        2) self.tokenizer.vocab_size
        """
        # 1) Model config
        if hasattr(self, 'model_config') and getattr(
                self.model_config, 'vocab_size', None) is not None:
            return int(self.model_config.vocab_size)

        # 2) Direct tokenizer on self
        if hasattr(self, 'tokenizer') and getattr(self.tokenizer, 'vocab_size',
                                                  None) is not None:
            return int(self.tokenizer.vocab_size)

        logger.debug(
            f"Cannot determine vocab_size from {self.__class__.__name__}. "
            "Please override this method to provide the vocabulary size. ")
        return None

    def get_mm_token_ids(self) -> Optional[Tensor]:
        """Return multimodal token IDs if available; otherwise None.

        The token IDs filtered by this method should be contiguous for each multimodal item, i.e. special tokens if any should be included.
        """
        processor = self.get_processor()
        if processor is not None and getattr(processor, 'mm_token_ids',
                                             None) is not None:
            return processor.mm_token_ids

        logger.debug(
            f"Cannot determine mm_token_ids from {self.__class__.__name__}. "
            "If needed, please override this method to return multimodal token ids. "
        )
        return None

    def get_mm_special_token_ids(self) -> Optional[Tensor]:
        """
        Return multimodal special token IDs if available; otherwise None.

        Special tokens refer to multimodal-related tokens (e.g. <image_end>, <image_break>) that are not part
        of the ViT output but come from text embeddings. Some VLMs
        (e.g., Mistral3, LLaMA4) mix special tokens with multimodal tokens,
        so they need to be returned separately.
        """
        processor = self.get_processor()
        return getattr(processor, "mm_special_token_ids",
                       None) if processor else None

    @property
    def get_num_multimodal_tokens(self):
        """
        Get the Hugging Face processor's '_get_num_multimodal_tokens' method.
        """
        processor = self.get_processor()
        if processor is not None and hasattr(processor,
                                             '_get_num_multimodal_tokens'):
            return processor._get_num_multimodal_tokens
        else:
            raise NotImplementedError(
                f"get_num_multimodal_tokens not implemented for {self.__class__.__name__}. "
                "Please override this method or ensure the processor has _get_num_multimodal_tokens method."
            )

    def get_num_tokens_per_image(
        self,
        *,
        image: Image.Image,
        **kwargs,
    ):
        """
        Calculate the number of tokens generated for an image.

        This (default) method delegates to the Hugging Face processor's '_get_num_multimodal_tokens' method.
        Returns the token count for the given image.

        Subclasses can override this method to provide custom logic to calculate the number of tokens.
        """
        image_height = image.height
        image_width = image.width
        image_size = (image_height, image_width)
        return self.get_num_multimodal_tokens([image_size],
                                              **kwargs)["num_image_tokens"][0]

    def get_num_tokens_per_video(
        self,
        *,
        video: List[Image.Image],
        **kwargs,
    ):
        """
        Calculate the number of tokens generated for a video.

        This (default) method delegates to the Hugging Face processor's '_get_num_multimodal_tokens' method.
        Returns the token count for the given video.

        Subclasses can override this method to provide custom logic to calculate the number of tokens.
        """
        video_width = video[0].width
        video_height = video[0].height
        num_frames = len(video)
        video_size = (num_frames, video_height, video_width)
        try:
            num_video_tokens = self.get_num_multimodal_tokens(
                video_sizes=[video_size], **kwargs)["num_video_tokens"][0]
            return num_video_tokens
        except Exception:
            # Fallback: treat video as sequence of frames
            num_tokens_per_frame = self.get_num_tokens_per_image(image=video[0],
                                                                 **kwargs)
            temporal_patch_size = self.temporal_patch_size if hasattr(
                self, 'temporal_patch_size') else 1
            return num_tokens_per_frame * num_frames // temporal_patch_size


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
        self.multimodal_hashing_supported = None

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


class MultimodalPlaceholderPlacement(enum.Enum):
    """
    The placement of the multimodal placeholder in the prompt. Valid values are:
        - BEFORE_TEXT: the placeholders are placed before the text prompt.
        - AFTER_TEXT: the placeholders are placed after the text prompt.
    """
    INVALID = -1
    BEFORE_TEXT = 0
    AFTER_TEXT = 1


@dataclass(frozen=True)
class MultimodalPlaceholderMetadata:
    """
    Metadata for the multimodal placeholder. It has 3 components:
        - placeholder_map:
            A mapping from modality to placeholder string.
            Modality can be "image", "video", "audio", etc.
        - placeholder_placement:
            The placement of the placeholders, e.g. before or after the text prompt.
        - placeholders_separator:
            The separator between the placeholders, e.g. some models use "\n" to separate the placeholders.
    """
    placeholder_map: Dict[str, str] = field(default_factory=dict)
    placeholder_placement: MultimodalPlaceholderPlacement = MultimodalPlaceholderPlacement.AFTER_TEXT
    placeholders_separator: str = "\n"


class MultimodalPlaceholderRegistry:
    """
    Registry for the multimodal models to keep track of the placeholder information.
    """

    def __init__(self) -> None:
        self._multimodal_placeholder_by_model_type: Dict[
            str, MultimodalPlaceholderMetadata] = {}

    def __str__(self) -> str:
        s = ""
        for model_type, placeholder_metadata in self._multimodal_placeholder_by_model_type.items(
        ):
            s += "-" * 100 + "\n"
            s += f"Model type: {model_type}\n"
            s += f"Placeholder map: {placeholder_metadata.placeholder_map}\n"
            s += f"Placeholder placement: {placeholder_metadata.placeholder_placement}\n"
            s += f"Placeholders separator: \"{placeholder_metadata.placeholders_separator}\"\n"
            s += "-" * 80 + "\n"
        return s

    def set_placeholder_metadata(
            self, model_type: str,
            placeholder_metadata: MultimodalPlaceholderMetadata):
        self._multimodal_placeholder_by_model_type[
            model_type] = placeholder_metadata

    def remove_placeholder_metadata(self, model_type: str):
        if model_type not in self._multimodal_placeholder_by_model_type:
            raise ValueError(f"Model type '{model_type}' is not registered")
        del self._multimodal_placeholder_by_model_type[model_type]

    def is_valid(self, model_type: str, modality: str) -> bool:
        return model_type in self._multimodal_placeholder_by_model_type and \
            modality in self._multimodal_placeholder_by_model_type[model_type].placeholder_map

    def get_placeholder_metadata(
            self, model_type: str) -> MultimodalPlaceholderMetadata:
        if model_type not in self._multimodal_placeholder_by_model_type:
            raise ValueError(
                f"Model type {model_type} is not registered in MultimodalPlaceholderRegistry"
            )
        return self._multimodal_placeholder_by_model_type[model_type]

    def get_placeholder(self, model_type: str, modality: str) -> str:
        if not self.is_valid(model_type, modality):
            raise ValueError(
                f"Model type '{model_type}' with modality '{modality}' is not registered."
            )
        return self._multimodal_placeholder_by_model_type[
            model_type].placeholder_map[modality]

    def get_placeholder_placement(
            self, model_type: str) -> MultimodalPlaceholderPlacement:
        if model_type not in self._multimodal_placeholder_by_model_type:
            raise ValueError(f"Model type '{model_type}' is not registered")
        return self._multimodal_placeholder_by_model_type[
            model_type].placeholder_placement

    def get_placeholders_separator(self, model_type: str) -> str:
        if model_type not in self._multimodal_placeholder_by_model_type:
            raise ValueError(f"Model type '{model_type}' is not registered")
        return self._multimodal_placeholder_by_model_type[
            model_type].placeholders_separator

    def get_registered_image_model_types(self) -> Tuple[str, ...]:
        return (
            model_type
            for model_type in self._multimodal_placeholder_by_model_type
            if "image" in self.
            _multimodal_placeholder_by_model_type[model_type].placeholder_map)

    def get_registered_video_model_types(self) -> Tuple[str, ...]:
        return (
            model_type
            for model_type in self._multimodal_placeholder_by_model_type
            if "video" in self.
            _multimodal_placeholder_by_model_type[model_type].placeholder_map)

    def get_registered_audio_model_types(self) -> Tuple[str, ...]:
        return (
            model_type
            for model_type in self._multimodal_placeholder_by_model_type
            if "audio" in self.
            _multimodal_placeholder_by_model_type[model_type].placeholder_map)

    def get_registered_model_types(self) -> Tuple[str, ...]:
        return tuple(self._multimodal_placeholder_by_model_type.keys())


MULTIMODAL_PLACEHOLDER_REGISTRY = MultimodalPlaceholderRegistry()


class InputProcessorRegistry:

    def __init__(self) -> None:
        self._input_processors_cls_by_model_type: Dict[
            Type[nn.Module], Type[InputProcessor]] = {}


INPUT_PROCESSOR_REGISTRY = InputProcessorRegistry()


def support_multimodal_disaggregated(model_cls: Type[nn.Module]):
    """
    Model-class decorator to declare support for multimodal disaggregated inputs.

    Apply this to a model class AFTER its input processor has been registered via
    @register_input_processor. The decorator will locate the processor class,
    validate requirements, and set `supports_multimodal_disagg = True` on both
    the processor class and the model class.
    """
    processor_cls = INPUT_PROCESSOR_REGISTRY._input_processors_cls_by_model_type.get(
        model_cls)
    if processor_cls is None:
        raise RuntimeError(
            f"No input processor registered for {model_cls.__name__}; ensure @register_input_processor is applied closer to the class than @supports_multimodal_disagg."
        )
    if not issubclass(processor_cls, BaseMultimodalInputProcessor):
        raise TypeError(
            f"{processor_cls.__name__} must inherit from BaseMultimodalInputProcessor to support multimodal disagg"
        )
    method = getattr(processor_cls, "get_prompt_token_ids", None)
    if method is None or not callable(method):
        raise TypeError(
            f"{processor_cls.__name__} must implement a callable method `get_prompt_token_ids` to support multimodal disagg"
        )

    setattr(processor_cls, "support_mm_disagg", True)
    setattr(model_cls, "support_mm_disagg", True)
    return model_cls


def register_input_processor(
        processor_cls: Type[InputProcessor],
        model_type: str,
        placeholder_metadata: MultimodalPlaceholderMetadata = None):
    """
    Register an input processor to a model class.
    NOTE:
        1. Since this API is only used for multimodal models, we are checking
           the model type only for that.
        2. If this is used for other models in the future, this logic needs to be
           updated e.g. adding another version of this API without the model_type.
    """

    def wrapper(model_cls: N) -> N:
        INPUT_PROCESSOR_REGISTRY._input_processors_cls_by_model_type[
            model_cls] = processor_cls
        if placeholder_metadata is None:
            raise ValueError(
                f"A valid placeholder_metadata must be provided but got {placeholder_metadata}"
            )

        MULTIMODAL_PLACEHOLDER_REGISTRY.set_placeholder_metadata(
            model_type, placeholder_metadata)

        return model_cls

    return wrapper


def create_input_processor(
    model_path_or_dir: str,
    tokenizer,
    checkpoint_format: Optional[str] = "HF",
) -> InputProcessor:
    """Create an input processor for a specific model.

    Args:
        model_path_or_dir: Path or repo id used to locate pretrained config/tokenizer.
        tokenizer: Tokenizer instance.
        checkpoint_format: Checkpoint format identifier. "HF" uses Hugging Face-style
            config loading; any other value skips HF config loading. Default is "HF".

    Returns:
        An InputProcessor implementation (model-specific if registered; otherwise DefaultInputProcessor).
    """
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models import get_model_architecture

    model_config = None

    if checkpoint_format == "HF":
        try:
            config = ModelConfig.from_pretrained(model_path_or_dir,
                                                 trust_remote_code=True)
            model_config = config.pretrained_config
        except (ValueError, EnvironmentError) as e:
            config = None
            logger.debug(
                f"Unable to load HF config from {model_path_or_dir}: {e}. Falling back."
            )
    else:
        logger.debug(
            f"checkpoint_format={checkpoint_format}; skipping HF config load.")

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

    def multimodal_hashing_process(
        inputs: TextPrompt, sampling_params: SamplingParams
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        """
        Process the multinmodal hashing for media tokens if possible.
        """
        assert 'multi_modal_data' in inputs, "multi_modal_data must be provided for hashing support."
        mm_data = inputs['multi_modal_data']
        mm_hashes = apply_mm_hashes(mm_data, hash_lib)
        prompt_token_ids, extra_processed_inputs = input_processor(
            inputs, sampling_params)

        num_mm_tokens = find_mm_token_lengths(mm_data, input_processor)
        # TODO: here we assume there is only one modality for now
        num_mm_tokens = next(iter(num_mm_tokens.values()))
        if len(num_mm_tokens) <= 0:
            return [], None

        vocab_size = input_processor.get_vocab_size()
        mm_ids = input_processor.get_mm_token_ids()
        mm_special_token_ids = input_processor.get_mm_special_token_ids()
        if vocab_size is None and mm_ids is None:
            raise ValueError(
                "Cannot locate vocab_size or mm_token_ids for multimodal token preprocessing"
            )
        start_positions, start_special_token_positions = find_mm_token_positions(
            input_ids=prompt_token_ids,  # token sequence
            num_mm_tokens=
            num_mm_tokens,  # list of lengths of each chunk of visual tokens
            vocab_size=vocab_size,
            mm_token_ids=mm_ids,
            mm_special_token_ids=mm_special_token_ids,
        )
        # Store special token offsets if available
        if len(start_special_token_positions
               ) > 0 and mm_special_token_ids is not None:
            extra_processed_inputs["multimodal_data"][
                "special_token_offsets"] = start_special_token_positions
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

    def input_processor_wrapper(
        inputs: TextPrompt, sampling_params: SamplingParams
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        try_multimodal_hashing = False  # only used for first time
        use_multimodal_hashing = False  # used for subsequent calls
        modalities = list(set(inputs['multi_modal_data'].keys())
                          ) if 'multi_modal_data' in inputs else []
        if len(modalities) > 0:
            # TODO: support multimodal hashing for multiple modalities within the same request
            # TODO: add audio support
            if len(modalities) == 1 and modalities[0] in ['image', 'video']:
                # only try multimodal hashing if the inputs only contain image data
                if input_processor.multimodal_hashing_supported is not None:
                    use_multimodal_hashing = input_processor.multimodal_hashing_supported
                else:
                    # we need to try the multimodal hashing for the first time to determine if it is supported
                    try_multimodal_hashing = True

        if try_multimodal_hashing or use_multimodal_hashing:
            try:
                prompt_token_ids, extra_processed_inputs = multimodal_hashing_process(
                    inputs, sampling_params)
                if try_multimodal_hashing:
                    # if trying for first time, set the flag to True
                    input_processor.multimodal_hashing_supported = True
                return prompt_token_ids, extra_processed_inputs
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.warning(f"Multimodal hashing failed: {e}.")
                if try_multimodal_hashing:
                    # if trying for first time, fall back to basic input processor
                    # and set the flag to False so that we don't try again
                    input_processor.multimodal_hashing_supported = False
                    logger.warning("Falling back to basic input processor.")
                    try:
                        return input_processor(inputs, sampling_params)
                    except Exception as e2:
                        import traceback
                        traceback.print_exc()
                        logger.warning(f"Basic input processor failed: {e}.")
                        raise e2
                else:
                    raise e
        else:
            try:
                return input_processor(inputs, sampling_params)
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.warning(f"Basic input processor failed: {e}.")
                raise e

    return input_processor_wrapper
