import enum
import random
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, List, Optional, Protocol, Tuple, Type,
                    TypeVar, Union)

import torch
from PIL import Image
from torch import Tensor, nn
from transformers import (AutoProcessor, PretrainedConfig,
                          PreTrainedTokenizerBase)

import tensorrt_llm

from .._utils import nvtx_range_debug
from ..logger import logger
from ..sampling_params import SamplingParams
from .content_format import ContentFormat
from .data import TextPrompt
from .multimodal import (MultimodalInput, _as_cpu_tensor, _compute_mm_masks,
                         _find_mm_token_start_pos_from_masks, apply_mm_hashes,
                         default_hasher, find_mm_token_lengths,
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
    config: any
    tokenizer: any

    def __call__(
        self, inputs: TextPrompt, sampling_params: SamplingParams
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        ...


class DefaultInputProcessor(InputProcessor):
    """Preprocess the inputs to the model."""

    def __init__(self,
                 model_path,
                 config,
                 tokenizer,
                 trust_remote_code: bool = True) -> None:
        self.tokenizer = tokenizer
        self.config = config
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


class BaseMultimodalInputProcessor(ABC):
    """
    Base class for multimodal input processors with default implementations
    of get_num_tokens_per_image and get_num_tokens_per_video methods.

    This class provides default implementations that work with most AutoProcessor-based
    models. Specific processors can override these methods if they need custom logic.

    Optional tokenized+MM fast path: to support prompt_token_ids + multi_modal_data
    without detokenizing, implement get_text_with_mm_placeholders(mm_counts) and
    expand_prompt_token_ids_for_mm(prompt_token_ids, num_mm_tokens_per_placeholder, ...).
    If these are not implemented, the pipeline detokenizes the text prompt first and then
    processes the multimodal inputs.
    """

    def __init__(self,
                 model_path,
                 config,
                 tokenizer,
                 trust_remote_code: bool = True,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self._config = config
        self._model_path = model_path
        self._tokenizer = tokenizer
        self._use_fast: bool = kwargs.get('use_fast', True)
        self._multimodal_hashing_supported: Optional[bool] = None

    def attach_multimodal_embeddings(
        self,
        inputs: TextPrompt,
        multimodal_embedding: Dict[str, List[torch.Tensor]],
        sampling_params: SamplingParams,
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        """
        Handle externally provided multimodal input embeddings.

        While inputs["multi_modal_data"] is handled by __call__, this method is intended to process
        inputs["multi_modal_embeddings"].
        """
        raise NotImplementedError(
            "Input processor does not support multimodal embedding input")

    @property
    @abstractmethod
    def processor(self) -> AutoProcessor:
        """The HF AutoProcessor for this model."""
        ...

    @property
    @abstractmethod
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """The HF tokenizer for this model."""
        return self._tokenizer

    @property
    @abstractmethod
    def config(self) -> PretrainedConfig:
        """The HF pretrained config for this model."""
        return self._config

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        """The dtype for this model."""
        ...

    @abstractmethod
    def __call__(
        self, inputs: TextPrompt, sampling_params: SamplingParams
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        ...

    @property
    def use_fast(self) -> bool:
        """
        Whether to use fast tokenizer for AutoProcessor.
        Default is True for most multimodal models.
        """
        return self._use_fast

    @property
    def multimodal_hashing_supported(self) -> Optional[bool]:
        """
        Whether multimodal hashing is supported for this processor.

        Returns None if unknown (will be detected at runtime),
        True if supported, False if not supported.
        """
        return self._multimodal_hashing_supported

    @multimodal_hashing_supported.setter
    def multimodal_hashing_supported(self, value: Optional[bool]) -> None:
        """Set the multimodal hashing support status (used for runtime detection)."""
        self._multimodal_hashing_supported = value

    def get_vocab_size(self) -> Optional[int]:
        """Return the tokenizer/model vocabulary size if available; otherwise None.

        Resolution order:
        1) self.config.vocab_size
        2) self.tokenizer.vocab_size
        """
        # 1) Model config
        if hasattr(self.config, 'vocab_size'):
            return int(self.config.vocab_size)

        # 2) Direct tokenizer on self
        if hasattr(self.tokenizer, 'vocab_size'):
            return int(self.tokenizer.vocab_size)

        logger.debug(
            f"Cannot determine vocab_size from {self.__class__.__name__}. "
            "Please override this method to provide the vocabulary size. ")
        return None

    def get_mm_token_ids(self) -> Optional[Tensor]:
        """Token IDs for a logical multimodal unit; include framing tokens for one contiguous span per unit.

        Framing tokens must also be in `get_mm_special_token_ids`. Example (Mistral):
        `[IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END]` → `{IMG, IMG_BREAK, IMG_END}`
        = 1 span; `{IMG}` alone fragments into 3.
        Return value is a 1-D tensor of token IDs; these are token values, not
        prompt positions or per-image counts.
        """
        if hasattr(self.processor, "mm_token_ids"):
            return self.processor.mm_token_ids

        logger.debug(
            f"Cannot find mm_token_ids in {self.__class__.__name__}.processor. "
            "If needed, please override this method to return multimodal token ids. "
        )
        return None

    def get_mm_special_token_ids(self) -> Optional[Tensor]:
        """IDs for in-prompt framing tokens inside a multimodal unit that carry no vision embedding.

        Found in e.g. Mistral3/LLaMA4 (`image_break`, `image_end`). Example: for
        `[IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END]` return `{IMG_BREAK, IMG_END}`
        — subtracted from the embed mask so embed-slot count stays accurate.
        Return value is a 1-D tensor of token IDs; these tokens are excluded
        from the embedding-row mask.
        """
        return getattr(self.processor, "mm_special_token_ids", None)

    @property
    def get_num_multimodal_tokens(self):
        """
        Get the Hugging Face processor's '_get_num_multimodal_tokens' method.
        """
        if hasattr(self.processor, '_get_num_multimodal_tokens'):
            return self.processor._get_num_multimodal_tokens
        else:
            raise NotImplementedError(
                f"get_num_multimodal_tokens not implemented for {self.__class__.__name__}. "
                "Please override this method or ensure the processor has _get_num_multimodal_tokens method."
            )

    def get_num_tokens_per_image(
        self,
        *,
        image: Union[Image.Image, torch.Tensor],
        **kwargs,
    ):
        """
        Calculate the number of tokens generated for an image.

        This (default) method delegates to the Hugging Face processor's '_get_num_multimodal_tokens' method.
        Accepts either a PIL Image or a CHW `torch.Tensor` — the hashing path
        in `find_mm_token_lengths` feeds tensors directly to avoid a costly
        ToPIL round-trip, while existing direct callers may still pass PIL.

        Returns the token count for the given image.
        Example: for Mistral, this count includes IMG placeholders plus row
        break/end framing tokens, matching the prompt-side logical MM unit.

        Subclasses can override this method to provide custom logic to calculate the number of tokens.
        """
        if isinstance(image, torch.Tensor):
            image_size = tuple(image.shape[-2:])
        else:
            image_size = (image.height, image.width)
        return self.get_num_multimodal_tokens([image_size],
                                              **kwargs)["num_image_tokens"][0]

    def get_num_tokens_per_video(
        self,
        *,
        video: List[Union[Image.Image, torch.Tensor]],
        **kwargs,
    ):
        """
        Calculate the number of tokens generated for a video.

        This (default) method delegates to the Hugging Face processor's '_get_num_multimodal_tokens' method.
        Accepts a list of PIL Images or CHW `torch.Tensor` frames.

        Returns the token count for the given video.
        Example: for a video item, return the prompt-side token count for that
        one video unit, not the number of video frames.

        Subclasses can override this method to provide custom logic to calculate the number of tokens.
        """
        num_frames = len(video)
        first_frame = video[0]
        if isinstance(first_frame, torch.Tensor):
            frame_h = int(first_frame.shape[-2])
            frame_w = int(first_frame.shape[-1])
        else:
            frame_h, frame_w = first_frame.height, first_frame.width
        video_size = (num_frames, frame_h, frame_w)
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


class BaseMultimodalDummyInputsBuilder(ABC):
    """
    Base class for generating dummy inputs. Specially for profiling
    """

    DEFAULT_IMAGE_MAX_DIM = 16384
    DEFAULT_IMAGE_MIN_DIM = 128

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image_max_dim = kwargs.get('image_max_dim',
                                        self.DEFAULT_IMAGE_MAX_DIM)
        self.image_min_dim = kwargs.get('image_min_dim',
                                        self.DEFAULT_IMAGE_MIN_DIM)

    @property
    @abstractmethod
    def tokenizer(self) -> PreTrainedTokenizerBase:
        ...

    @property
    @abstractmethod
    def config(self) -> PretrainedConfig:
        ...

    @property
    @abstractmethod
    def model_path(self) -> str:
        ...

    def get_dummy_image(self, max_width: int, max_height: int) -> Image.Image:
        image = Image.new("RGB", (max_width, max_height),
                          color=random.randint(0, 256))
        return image

    def get_dummy_prompt(self, input_seq_len: int):
        # TODO(yechank): We use the max resolution as starting point and keep reducing the resolution until the prompt length is less than the input sequence length.
        # Need to find better way to calculate the dummy prompt length as this iteration may not be efficient.

        # Use the registered model_type from the decorator if available,
        # otherwise fall back to HuggingFace config's model_type.
        # This ensures consistency between placeholder registration and lookup.
        registered_model_type = getattr(self.__class__,
                                        '_registered_model_type', None)
        config_model_type = self.config.model_type
        model_type = registered_model_type or config_model_type

        logger.debug(
            f"[get_dummy_prompt] registered_model_type={registered_model_type}, "
            f"config.model_type={config_model_type}, using model_type={model_type}"
        )

        while self.image_max_dim >= self.image_min_dim:
            image = self.get_dummy_image(max_width=self.image_max_dim,
                                         max_height=self.image_max_dim)

            test_mm_prompt = tensorrt_llm.inputs.utils.default_multimodal_input_loader(
                tokenizer=self.tokenizer,
                model_dir=self.model_path,
                model_type=model_type,
                modality="image",
                prompts=[""],
                media=[[image]],
                image_data_format="pt")[0]

            prompt_token_ids_single_img, _ = self(test_mm_prompt, None)

            if len(prompt_token_ids_single_img) <= input_seq_len:
                return test_mm_prompt

            # reduce img resolution
            self.image_max_dim = self.image_max_dim >> 1

        return None


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
    Metadata for the multimodal placeholder. It has 5 components:
        - placeholder_map:
            A mapping from modality to placeholder string.
            Modality can be "image", "video", "audio", etc.
        - placeholder_placement:
            The placement of the placeholders, e.g. before or after the text prompt.
            Only used when interleave_placeholders is False (the default).
            Ignored when interleave_placeholders is True.
        - placeholders_separator:
            The separator between the placeholders, e.g. some models use "\n" to separate the placeholders.
        - content_format:
            Optional override for the content format expected by the chat template.
            ContentFormat.OPENAI means the template handles multimodal content dicts natively.
            ContentFormat.STRING means the template expects plain string content.
            ContentFormat.PASSTHROUGH skips chat template rendering entirely.
            None means auto-detect at runtime via Jinja AST analysis.
        - interleave_placeholders:
            When True and content_parts is available, placeholders are inserted
            at the exact media positions within the text (interleaved).
            In this mode, placeholder_placement is ignored - the position of
            each placeholder is determined by where the media appears in the
            user's message.
            When False (default), placeholders are bulk-prepended or appended
            according to placeholder_placement.
    """
    placeholder_map: Dict[str, str] = field(default_factory=dict)
    placeholder_placement: MultimodalPlaceholderPlacement = MultimodalPlaceholderPlacement.AFTER_TEXT
    placeholders_separator: str = "\n"
    content_format: Optional[ContentFormat] = None
    interleave_placeholders: bool = False


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

    def get_interleave_placeholders(self, model_type: str) -> bool:
        """Return whether the model opts in to interleaved placeholder insertion."""
        if model_type not in self._multimodal_placeholder_by_model_type:
            return False
        return self._multimodal_placeholder_by_model_type[
            model_type].interleave_placeholders

    def get_content_format(self, model_type: str) -> Optional[ContentFormat]:
        """Get the content format override for a model type, or None for auto-detect."""
        if model_type not in self._multimodal_placeholder_by_model_type:
            return None
        return self._multimodal_placeholder_by_model_type[
            model_type].content_format

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

        # Store model_type on processor class for use in get_dummy_prompt
        processor_cls._registered_model_type = model_type

        return model_cls

    return wrapper


def create_input_processor(
    model_path_or_dir: str,
    tokenizer,
    checkpoint_format: Optional[str] = "HF",
    **kwargs,
) -> Union[InputProcessor, BaseMultimodalInputProcessor]:
    """Create an input processor for a specific model.

    Args:
        model_path_or_dir: Path or repo id used to locate pretrained config/tokenizer.
        tokenizer: Tokenizer instance.
        checkpoint_format: Checkpoint format identifier. "HF" uses Hugging Face-style
            config loading; any other value skips HF config loading. Default is "HF".
        **kwargs: Additional arguments passed to input processor constructors
            (e.g., video_pruning_rate for multimodal models).

    Returns:
        An InputProcessor implementation (model-specific if registered; otherwise DefaultInputProcessor).
    """
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models import get_model_architecture

    config = None

    if checkpoint_format == "HF":
        try:
            model_config = ModelConfig.from_pretrained(model_path_or_dir,
                                                       trust_remote_code=True)
            config = model_config.pretrained_config
        except (ValueError, EnvironmentError) as e:
            logger.debug(
                f"Unable to load HF config from {model_path_or_dir}: {e}. Falling back."
            )
    elif checkpoint_format in ("mistral", "mistral_large_3"):
        logger.debug(f"Detected checkpoint_format={checkpoint_format}.")
        from tensorrt_llm._torch.models.checkpoints.mistral.config_loader import \
            MistralConfigLoader
        model_config = MistralConfigLoader().load(model_path_or_dir)
        config = model_config.pretrained_config
    else:
        logger.debug(
            f"checkpoint_format={checkpoint_format}; skipping HF config load.")

    if config is not None:
        try:
            model_cls, _ = get_model_architecture(config)
            input_processor_cls = INPUT_PROCESSOR_REGISTRY._input_processors_cls_by_model_type \
                .get(model_cls)
        except RuntimeError:  # unregistered model
            logger.info("Unregistered model, using DefaultInputProcessor")
            input_processor_cls = None
        if input_processor_cls is not None:
            return input_processor_cls(model_path_or_dir,
                                       config,
                                       tokenizer,
                                       trust_remote_code=True,
                                       **kwargs)

    return DefaultInputProcessor(None, None, tokenizer)


def _mm_data_to_counts(mm_data: Dict[str, Any]) -> Dict[str, int]:
    """Normalize multimodal data to per-key counts (each value as list length)."""
    mm_items = {
        k: (v if isinstance(v, list) else [v])
        for k, v in mm_data.items()
    }
    return {k: len(v) for k, v in mm_items.items()}


def _process_multimodal_with_dummy_placeholders(
    input_processor: BaseMultimodalInputProcessor,
    mm_data: Dict[str, Any],
    mm_counts: Dict[str, int],
    mm_processor_kwargs: Optional[Dict[str, Any]],
    sampling_params: SamplingParams,
) -> ExtraProcessedInputs:
    """Run input_processor with dummy text placeholders for multi-modal slots; return extra processed inputs."""
    dummy_text = input_processor.get_text_with_mm_placeholders(mm_counts)
    dummy_inputs = TextPrompt(
        prompt=dummy_text,
        multi_modal_data=mm_data,
        mm_processor_kwargs=mm_processor_kwargs,
    )
    # input_processor runs the HF processor / vision encoder on mm_data (e.g. images).
    # extra_processed_inputs contains the processed MM data keyed under "multimodal_data";
    # it is reused later with the real token IDs so we do not run the vision encoder again.
    _, extra_processed_inputs = input_processor(dummy_inputs, sampling_params)
    if extra_processed_inputs is None:
        return {}
    return extra_processed_inputs


def _get_single_mm_token_lengths(
    mm_data: Dict[str, Any],
    input_processor: BaseMultimodalInputProcessor,
    *,
    multimodal_data: Optional[Dict[str, Any]] = None,
) -> Optional[List[int]]:
    """Get the single set of MM token lengths (first value from find_mm_token_lengths). Returns None if empty."""
    num_mm_tokens_by_key = find_mm_token_lengths(
        mm_data, input_processor, multimodal_data=multimodal_data)
    if not num_mm_tokens_by_key:
        return None
    # find_mm_token_lengths returns Dict[modality, List[int]], e.g. {"image": [2928, 2928]}.
    # We need the list of per-item lengths (for _find_mm_token_start_pos_from_masks). We take
    # the first modality's list; multi-modality is not yet supported
    # (see TODO in multimodal_hashing_process).
    num_mm_tokens = next(iter(num_mm_tokens_by_key.values()))
    if len(num_mm_tokens) <= 0:
        return None
    return num_mm_tokens


def maybe_compute_mm_embed_cumsum(
    prompt_token_ids: List[int],
    extra_processed_inputs: Optional[ExtraProcessedInputs],
    input_processor: BaseMultimodalInputProcessor,
) -> None:
    """Ensure `multimodal_embed_mask_cumsum` is present in `extra_processed_inputs`.

    Silently skipped if the processor provides neither `vocab_size` nor
    `mm_token_ids`.

    Idempotent: no-op when the cumsum is already populated. Otherwise
    classifies every prompt position via the processor's `mm_token_ids` /
    `vocab_size` predicate (with specials subtracted), takes the int64
    prefix sum, and stores the flat `int64[prompt_len]` tensor at
    `extra_processed_inputs["multimodal_data"]["multimodal_embed_mask_cumsum"]`.

    """
    if extra_processed_inputs is None:
        return
    mm_data = extra_processed_inputs.get("multimodal_data")
    if mm_data is None:
        return
    if "multimodal_embed_mask_cumsum" in mm_data:
        return

    vocab_size = input_processor.get_vocab_size()
    mm_token_ids = input_processor.get_mm_token_ids()
    if vocab_size is None and mm_token_ids is None:
        logger.debug(
            "maybe_compute_mm_embed_cumsum: processor provides neither "
            "vocab_size nor mm_token_ids — skipping cumsum computation.")
        return

    input_ids = _as_cpu_tensor(prompt_token_ids)
    _, embed_mask, _ = _compute_mm_masks(
        input_ids,
        vocab_size=vocab_size,
        mm_token_ids=mm_token_ids,
        mm_special_token_ids=input_processor.get_mm_special_token_ids(),
    )
    # Cache the int64 cumsum; request-invariant, read once per chunk.
    mm_data["multimodal_embed_mask_cumsum"] = embed_mask.cumsum(
        0, dtype=torch.int64)


def create_input_processor_with_hash(
    input_processor: BaseMultimodalInputProcessor,
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

    def tokenized_multimodal_process(
        inputs: TextPrompt, sampling_params: SamplingParams
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        """
        Process prompt_token_ids and multi_modal_data without detokenizing.

        Runs the input processor with dummy text placeholders for multi-modal slots,
        then replaces placeholder token IDs with the actual feature token IDs and
        delegates to multimodal_hashing_process.

        Args:
            inputs: TextPrompt with "prompt_token_ids" and "multi_modal_data" (and optional "mm_processor_kwargs").
            sampling_params: Sampling parameters for the input processor.

        Returns:
            (prompt_token_ids, extra_processed_inputs) from multimodal_hashing_process.
            ([], None) if multi-modal token lengths cannot be determined.
        """
        prompt_token_ids = inputs["prompt_token_ids"]
        mm_data = inputs["multi_modal_data"]
        mm_counts = _mm_data_to_counts(mm_data)
        extra_processed_inputs = _process_multimodal_with_dummy_placeholders(
            input_processor,
            mm_data,
            mm_counts,
            inputs.get("mm_processor_kwargs"),
            sampling_params,
        )
        num_mm_tokens = _get_single_mm_token_lengths(
            mm_data,
            input_processor,
            multimodal_data=(extra_processed_inputs
                             or {}).get("multimodal_data"),
        )
        if num_mm_tokens is None:
            raise ValueError(
                "tokenized_multimodal_process: find_mm_token_lengths returned "
                "no token lengths for the provided multi_modal_data.")

        expanded_ids = input_processor.expand_prompt_token_ids_for_mm(
            prompt_token_ids,
            num_mm_tokens,
            hf_processor_mm_kwargs=inputs.get("mm_processor_kwargs"))
        return multimodal_hashing_process(
            inputs,
            sampling_params,
            precomputed_token_ids=expanded_ids,
            precomputed_extra=extra_processed_inputs,
            precomputed_num_mm_tokens=num_mm_tokens,
        )

    def multimodal_hashing_process(
        inputs: TextPrompt,
        sampling_params: SamplingParams,
        *,
        precomputed_token_ids: Optional[List[int]] = None,
        precomputed_extra: Optional[ExtraProcessedInputs] = None,
        precomputed_num_mm_tokens: Optional[List[int]] = None,
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        """
        Process multimodal hashing for media tokens if possible.

        precomputed_token_ids and precomputed_extra must be provided together or
        both be None. When both are provided (tokenized+MM path), skips the
        input_processor call and uses them; when both are None, calls input_processor.

        precomputed_num_mm_tokens is optional and independent: if provided, skips
        the otherwise-duplicate find_mm_token_lengths call (the tokenized+MM path
        already computes it upstream to expand the prompt).

        Supports optional user-provided UUIDs via 'multi_modal_uuids' in inputs.
        When a UUID is provided for a multimodal item, it will be used as the
        cache identifier and returned in KV cache events instead of the content hash.
        """
        assert 'multi_modal_data' in inputs, "multi_modal_data must be provided for hashing support."
        mm_data = inputs['multi_modal_data']

        # Extract optional UUIDs (can be None, or dict with same structure as mm_data)
        mm_uuids = inputs.get('multi_modal_uuids', None)

        mm_hashes, mm_uuid_list = apply_mm_hashes(mm_data, mm_uuids, hash_lib)

        if precomputed_token_ids is not None and precomputed_extra is not None:
            prompt_token_ids = precomputed_token_ids
            extra_processed_inputs = precomputed_extra
        elif precomputed_token_ids is None and precomputed_extra is None:
            prompt_token_ids, extra_processed_inputs = input_processor(
                inputs, sampling_params)
        else:
            raise ValueError(
                "precomputed_token_ids and precomputed_extra must be provided "
                "together or both be None; got one without the other.")

        if precomputed_num_mm_tokens is not None:
            num_mm_tokens = precomputed_num_mm_tokens
        else:
            # TODO: here we assume there is only one modality for now
            num_mm_tokens_by_key = find_mm_token_lengths(
                mm_data,
                input_processor,
                multimodal_data=(extra_processed_inputs
                                 or {}).get("multimodal_data"),
            )
            if not num_mm_tokens_by_key:
                return [], None
            num_mm_tokens = next(iter(num_mm_tokens_by_key.values()))
        if len(num_mm_tokens) <= 0:
            return [], None

        vocab_size = input_processor.get_vocab_size()
        mm_ids = input_processor.get_mm_token_ids()
        mm_special_token_ids = input_processor.get_mm_special_token_ids()
        if vocab_size is None and mm_ids is None:
            raise ValueError(
                "Cannot locate vocab_size or mm_token_ids for multimodal token preprocessing"
            )
        # Compute all three masks once here and reuse downstream. The embed
        # cumsum is stashed into extra_processed_inputs so the wrapper's
        # subsequent maybe_compute_mm_embed_cumsum call short-circuits via
        # its idempotency guard, avoiding a second full-sequence isin pass.
        input_ids_tensor = _as_cpu_tensor(prompt_token_ids)
        if input_ids_tensor.numel() == 0:
            start_positions, start_special_token_positions = [], []
        else:
            mm_mask, embed_mask, special_mask = _compute_mm_masks(
                input_ids_tensor,
                vocab_size=vocab_size,
                mm_token_ids=mm_ids,
                mm_special_token_ids=mm_special_token_ids,
            )
            extra_processed_inputs["multimodal_data"].setdefault(
                "multimodal_embed_mask_cumsum",
                embed_mask.cumsum(0, dtype=torch.int64))
            start_positions, start_special_token_positions = (
                _find_mm_token_start_pos_from_masks(mm_mask, special_mask,
                                                    num_mm_tokens))
        # Store special token offsets if available
        if len(start_special_token_positions
               ) > 0 and mm_special_token_ids is not None:
            extra_processed_inputs["multimodal_data"][
                "special_token_offsets"] = start_special_token_positions
        # flatten the hashes from dict to a single list
        mm_hashes_flat = [h for hashes in mm_hashes.values() for h in hashes]
        validate_mm_inputs(prompt_token_ids, mm_hashes_flat, start_positions,
                           num_mm_tokens)
        mm_hashes_int32 = [hexdigest_to_int32(h) for h in mm_hashes_flat
                           ]  # nested list w/ multiple int32 per hash

        extra_processed_inputs[
            "multimodal_input"] = MultimodalInput.from_components(
                mm_hashes_int32, start_positions, num_mm_tokens, mm_uuid_list)
        return prompt_token_ids, extra_processed_inputs

    def process_tokenized_prompt_maybe_hash(
        inputs: TextPrompt, sampling_params: SamplingParams
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        try:
            return tokenized_multimodal_process(inputs, sampling_params)
        except Exception as e:
            logger.warning(f"Tokenized+MM path failed: {e}")
            raise

    def process_prompt_maybe_hash(
        inputs: TextPrompt, sampling_params: SamplingParams
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        try_multimodal_hashing = False  # only used for first time
        use_multimodal_hashing = False  # used for subsequent calls
        modalities = list(set(inputs['multi_modal_data'].keys())
                          ) if 'multi_modal_data' in inputs else []
        if len(modalities) > 0:
            # TODO: support multimodal hashing for multiple modalities within the same request.
            if len(modalities) == 1 and modalities[0] in [
                    'image', 'video', 'audio'
            ]:
                # only try multimodal hashing if the inputs only contain a single modality.
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
            except Exception as e:
                logger.warning(f"Multimodal hashing failed: {e}.")
                if try_multimodal_hashing:
                    # if trying for first time, fall back to basic input processor
                    # and set the flag to False so that we don't try again
                    input_processor.multimodal_hashing_supported = False
                    logger.warning("Falling back to basic input processor.")
                    try:
                        prompt_token_ids, extra_processed_inputs = input_processor(
                            inputs, sampling_params)
                    except Exception as e2:
                        logger.warning(f"Basic input processor failed: {e}.")
                        logger.debug(traceback.format_exc())
                        raise e2
                else:
                    raise e
        else:
            try:
                prompt_token_ids, extra_processed_inputs = input_processor(
                    inputs, sampling_params)
            except Exception as e:
                logger.warning(f"Basic input processor failed: {e}.")
                logger.debug(traceback.format_exc())
                raise e

        return prompt_token_ids, extra_processed_inputs

    def input_processor_wrapper(
        inputs: TextPrompt, sampling_params: SamplingParams
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        # Tokenized prompt + multi_modal_data path: requires the optional hooks.
        # If the processor lacks them, fall through to the regular prompt path.
        has_tokenized_multimodal_prompt = (
            inputs.get("prompt_token_ids") is not None
            and inputs.get("multi_modal_data") is not None
            and inputs.get("prompt") is None
            and hasattr(input_processor, "get_text_with_mm_placeholders")
            and hasattr(input_processor, "expand_prompt_token_ids_for_mm"))

        if has_tokenized_multimodal_prompt:
            prompt_token_ids, extra_processed_inputs = (
                process_tokenized_prompt_maybe_hash(inputs, sampling_params))
        else:
            prompt_token_ids, extra_processed_inputs = (
                process_prompt_maybe_hash(inputs, sampling_params))

        maybe_compute_mm_embed_cumsum(prompt_token_ids, extra_processed_inputs,
                                      input_processor)
        return prompt_token_ids, extra_processed_inputs

    return input_processor_wrapper
