"""
Base test class for multimodal model testing with TensorRT-LLM.

This module provides a comprehensive testing framework for multimodal models,
supporting various scenarios including CUDA graphs, chunked prefill, and KV cache reuse.

Key Features:
    - Template-based testing framework using abstract base class pattern
    - Automatic output comparison between TensorRT-LLM and HuggingFace models
    - Support for multiple modalities: image, video, text, and mixed inputs
    - Advanced features: CUDA graphs, chunked prefill, KV cache reuse
    - Configurable tolerances, cache settings, and test scenarios
    - Detailed logging and error diagnostics
    - Clean resource management with automatic cleanup

Quick Start:
    1. Create a subclass of TestModelingMultimodal
    2. Implement required abstract methods (model config, classes, etc.)
    3. Optionally override methods to customize behavior
    4. Run with: python -m unittest test_modeling_mymodel.TestMyModel.test_all

Example:
    See the TestModelingMultimodal class docstring for a complete example.

Test Scenarios:
    The framework tests various scenarios by default:
    - Modality tests: image, video, multiple images
    - CUDA graph optimization
    - Chunked prefill (processing long sequences in chunks)
    - KV cache reuse (reusing cached key-value pairs)

Environment:
    Set LLM_MODELS_ROOT environment variable to specify model directory.
    Default: /scratch.trt_llm_data/llm-models

Author: TensorRT-LLM Team
"""

import os
import unittest
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import torch
from _torch.helpers import create_mock_engine
from transformers import (AutoProcessor, AutoTokenizer, PretrainedConfig,
                          PreTrainedModel)
from utils.llm_data import llm_models_root

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.interface import \
    AttentionRuntimeFeatures
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.inputs import (create_input_processor,
                                 create_input_processor_with_hash,
                                 default_multimodal_input_loader, prompt_inputs)
from tensorrt_llm.inputs.multimodal import (MultimodalParams,
                                            MultimodalRuntimeData)
from tensorrt_llm.mapping import Mapping


@dataclass(repr=False)
class Scenario:
    """Test scenario configuration for multimodal tests.

    This class defines a test scenario with specific features and modality.
    Each scenario represents a unique combination of testing parameters.

    Attributes:
        modality: Type of multimodal input
            Valid values: "image", "video", "text", "multiple_image", "mixture_text_image"
        use_cuda_graph: Whether to use CUDA graph optimization
            Enables CUDA graph capture and replay for generation phase
        chunked_prefill: Whether to use chunked prefill
            Processes long sequences in chunks to reduce memory usage
        kv_cache_reuse: Whether to test KV cache reuse
            Tests reusing cached key-value pairs across iterations

    Example:
        >>> scenario = Scenario(modality="image", use_cuda_graph=True)
        >>> print(scenario)
        modality:image-cuda_graph
    """
    modality: str = "image"
    use_cuda_graph: bool = False
    chunked_prefill: bool = False
    kv_cache_reuse: bool = False

    def __post_init__(self):
        """Validate scenario configuration after initialization."""
        valid_modalities = [
            "image", "video", "text", "multiple_image", "mixture_text_image"
        ]
        if self.modality not in valid_modalities:
            raise ValueError(f"Invalid modality '{self.modality}'. "
                             f"Valid modalities: {', '.join(valid_modalities)}")

    def __repr__(self) -> str:
        """Generate a human-readable string representation of the scenario."""
        features = []
        features.append(f"modality:{self.modality.lower()}")
        if self.use_cuda_graph:
            features.append("cuda_graph")
        if self.chunked_prefill:
            features.append("chunked_prefill")
        if self.kv_cache_reuse:
            features.append("kv_cache_reuse")
        return "-".join(features)


class TestModelingMultimodal(unittest.TestCase, ABC):
    """
    Base test class for multimodal model testing with TensorRT-LLM.

    This class provides a comprehensive testing framework for multimodal models,
    including support for various features like CUDA graphs, chunked prefill,
    KV cache reuse, and multiple modalities (image, video, text).

    Architecture:
        The test framework follows a template pattern where subclasses provide
        model-specific implementations while the base class handles the testing logic.

    Required Abstract Methods:
        - get_model_config(): Return model configuration dictionary
        - get_trtllm_model_class(): Return TensorRT-LLM model class
        - get_hf_model_class(): Return HuggingFace model class
        - get_weight_mapper_class(): Return weight mapper class
        - get_model_type(): Return model type string
        - get_model_config_class(): Return HuggingFace config class

    Optional Methods to Override:
        - get_dtype(): Customize data type (default: from config)
        - get_tolerance(): Customize comparison tolerances (default: 0.4, 0.4)
        - get_kv_cache_config(): Customize KV cache settings
        - get_max_num_tokens(): Customize token limits
        - get_scenarios(): Customize test scenarios
        - get_raw_test_inputs(): Customize test inputs
        - get_trtllm_inputs(): Customize input preparation (e.g., for special position IDs)
        - init(): Customize initialization logic

    Test Flow:
        1. init() - Initialize models and configurations
        2. For each scenario in get_scenarios():
           a. Initialize KV cache manager
           b. Initialize attention metadata
           c. Get test inputs
           d. Run context phase (with optional chunking/cache reuse)
           e. Compare outputs with HuggingFace
           f. Run generation phase
           g. Compare outputs with HuggingFace
           h. Cleanup
        3. Report results


    Features:
        - Automatic comparison of TensorRT-LLM vs HuggingFace outputs
        - Support for multiple modalities (image, video, text, mixed)
        - CUDA graph optimization testing
        - Chunked prefill testing
        - KV cache reuse testing
        - Configurable tolerances and cache settings
        - Detailed error reporting and diagnostics
        - Clean resource management

    Attributes:
        device: CUDA device for testing
        dtype: Model data type
        config: Model configuration dictionary
        hf_config: HuggingFace configuration object
        hf_model: HuggingFace model instance
        trtllm_model: TensorRT-LLM model instance
        model_config: TensorRT-LLM model config
        runtime_features: Attention runtime features
        kv_cache_manager: KV cache manager
        attn_metadata: Attention metadata
    """

    @abstractmethod
    def get_model_config(self) -> Dict:
        """Return the model configuration dictionary."""

    @abstractmethod
    def get_trtllm_model_class(self) -> Type:
        """Return the TensorRT-LLM model class.

        Returns:
            Class type for the TensorRT-LLM model implementation
        """

    @abstractmethod
    def get_hf_model_class(self) -> Type:
        """Return the HuggingFace model class.

        Returns:
            Class type for the HuggingFace model implementation
        """

    @abstractmethod
    def get_weight_mapper_class(self) -> Type:
        """Return the weight mapper class.

        Returns:
            Class type for the weight mapper
        """

    @abstractmethod
    def get_model_type(self) -> str:
        """Return the model type string.

        Returns:
            Model type string (default: extracted from config)
        """

    @abstractmethod
    def get_model_config_class(self) -> Type:
        """Return the model configuration class.

        Returns:
            Class type for the model configuration
        """

    def get_dtype(self) -> torch.dtype:
        """Return the model data type.

        Returns:
            torch.dtype: Model data type (default: extracted from config)

        Note:
            Override this method if you need a different dtype than specified in config.
            Default behavior: reads from config's torch_dtype field.
        """
        config = self.get_model_config()
        dtype_str = config.get('torch_dtype', 'bfloat16')

        # Handle both string and torch.dtype types
        if isinstance(dtype_str, torch.dtype):
            return dtype_str

        return str_dtype_to_torch(dtype_str)

    # Test input generation methods
    def get_raw_test_inputs(self, modality: str) -> Tuple[List[str], List[str]]:
        """Get test inputs for a given modality.

        Args:
            modality: The modality type ("image", "video", "text", etc.)

        Returns:
            Tuple[List[str], List[str]]: (prompts, media_paths)

        Note:
            Override this method to provide custom test inputs for your model.
            Default test data should be placed in: {LLM_MODELS_ROOT}/multimodals/test_data/
        """
        test_data_root = Path(
            os.path.join(llm_models_root(), "multimodals", "test_data"))

        # Define test cases for each modality
        test_cases = {
            "image": (["Describe the natural environment in the image."],
                      [str(test_data_root / "seashore.png")]),
            "multiple_image":
            (["Describe the difference between the two images."], [
                str(test_data_root / "inpaint.png"),
                str(test_data_root / "61.jpg")
            ]),
            "video": (["Tell me what you see in the video briefly."],
                      [str(test_data_root / "OAI-sora-tokyo-walk.mp4")]),
            "mixture_text_image": ([
                "Describe the scene in the image briefly.",
                "Who invented the internet?"
            ], [[str(test_data_root / "inpaint.png")], []]),
            "text": (["Who invented the internet?"], []),
        }

        if modality not in test_cases:
            raise ValueError(
                f"Invalid modality: {modality}. "
                f"Supported modalities: {', '.join(test_cases.keys())}")

        prompts, media = test_cases[modality]

        return prompts, media

    def create_hf_config(self) -> PretrainedConfig:
        config_dict = deepcopy(self.get_model_config())
        config_class = self.get_model_config_class()
        hf_config = config_class.from_dict(config_dict)
        return hf_config

    # Model creation and management methods
    def create_trtllm_model(self,
                            load_weights: bool = False,
                            hf_model_state_dict: Optional[Dict] = None,
                            **kwargs) -> Tuple[PreTrainedModel, ModelConfig]:
        """Create a TensorRT-LLM model instance.

        Args:
            load_weights: Whether to load weights
            hf_model_state_dict: HuggingFace model state dictionary
            **kwargs: Additional arguments for model initialization

        Returns:
            Tuple of (model, model_config)

        Raises:
            ValueError: If hf_config is not initialized or weight loading fails
        """

        model_config = ModelConfig(pretrained_config=self.hf_config)
        model_class = self.get_trtllm_model_class()
        model = model_class(model_config, **kwargs).to('cuda')

        if load_weights:
            weight_mapper_class = self.get_weight_mapper_class()
            if weight_mapper_class is not None:
                weight_mapper = weight_mapper_class()
                weight_mapper.init_model_and_config(model, self.hf_config)
                model.load_weights(hf_model_state_dict, weight_mapper)
            else:
                model.load_weights(hf_model_state_dict)

        return model, model_config

    def create_hf_model(self,
                        pretrained_config: PretrainedConfig) -> PreTrainedModel:
        """Create a HuggingFace model instance.

        Args:
            pretrained_config: Model configuration dictionary
            dtype: Model data type
            device: Target device

        Returns:
            HuggingFace model instance
        """
        hf_model_class = self.get_hf_model_class()
        hf_model = hf_model_class(pretrained_config).to(self.device).to(
            self.get_dtype())
        hf_model.eval()

        return hf_model

    # Utility methods for comparisons
    def get_tolerance(self) -> Tuple[float, float]:
        """Get tolerance values for output comparison.

        Returns:
            Tuple[float, float]: (atol, rtol) - absolute and relative tolerances

        Note:
            Override this method to provide custom tolerance values for your model.
            Default values are suitable for most multimodal models.
        """
        return 0.4, 0.4  # (atol, rtol)

    def compare_outputs(self,
                        trtllm_output: torch.Tensor,
                        hf_output: torch.Tensor,
                        rtol: Optional[float] = None,
                        atol: Optional[float] = None) -> bool:
        """Compare TensorRT-LLM and HuggingFace outputs.

        Args:
            trtllm_output: Output from TensorRT-LLM model
            hf_output: Output from HuggingFace model
            rtol: Relative tolerance (uses get_tolerance() if None)
            atol: Absolute tolerance (uses get_tolerance() if None)

        Returns:
            bool: True if outputs are close within tolerance

        Raises:
            AssertionError: If outputs don't match within tolerance
        """
        if atol is None or rtol is None:
            default_atol, default_rtol = self.get_tolerance()
            atol = atol if atol is not None else default_atol
            rtol = rtol if rtol is not None else default_rtol

        try:
            torch.testing.assert_close(trtllm_output,
                                       hf_output,
                                       atol=atol,
                                       rtol=rtol)
            return True
        except AssertionError:
            # Add more context to the error message
            max_diff = torch.max(torch.abs(trtllm_output - hf_output)).item()
            mean_diff = torch.mean(torch.abs(trtllm_output - hf_output)).item()
            print(f"Output comparison failed:")
            print(f"  Max absolute difference: {max_diff}")
            print(f"  Mean absolute difference: {mean_diff}")
            print(f"  Tolerance: atol={atol}, rtol={rtol}")
            raise

    def get_kv_cache_config(self, scenario: Scenario) -> Dict:
        """Get KV cache configuration for a test scenario.

        Args:
            scenario: Test scenario configuration

        Returns:
            Dict: Configuration dictionary with tokens_per_block, max_seq_len, batch_size

        Note:
            Override this method to customize KV cache settings for your model.
        """
        tokens_per_block = 128
        batch_size = 16

        # Adjust max_seq_len based on modality
        if scenario.modality == "video":
            max_seq_len = 16384
        else:
            max_seq_len = 8192

        return {
            'tokens_per_block': tokens_per_block,
            'max_seq_len': max_seq_len,
            'batch_size': batch_size
        }

    def get_kv_cache_manager(self, dtype: torch.dtype, config: PretrainedConfig,
                             tokens_per_block: int, max_seq_len: int,
                             batch_size: int, num_blocks: int):
        """Get KV cache manager for testing.

        Args:
            dtype: Data type for KV cache
            config: Model configuration
            tokens_per_block: Number of tokens per cache block
            max_seq_len: Maximum sequence length
            batch_size: Batch size
            num_blocks: Number of cache blocks

        Returns:
            KVCacheManager: Initialized KV cache manager

        Raises:
            ValueError: If dtype is not supported
        """
        # Map torch dtype to TensorRT-LLM dtype
        dtype_map = {
            torch.half: tensorrt_llm.bindings.DataType.HALF,
            torch.float16: tensorrt_llm.bindings.DataType.HALF,
            torch.bfloat16: tensorrt_llm.bindings.DataType.BF16,
        }

        if dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype: {dtype}. "
                             f"Supported dtypes: {list(dtype_map.keys())}")

        kv_cache_dtype = dtype_map[dtype]
        mapping = Mapping(world_size=1, tp_size=1, rank=0)
        kv_cache_config = KvCacheConfig(max_tokens=num_blocks *
                                        tokens_per_block)

        kv_cache_manager = KVCacheManager(
            kv_cache_config,
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=batch_size,
            mapping=mapping,
            dtype=kv_cache_dtype,
        )
        return kv_cache_manager

    def get_input_processor(self, model_path: str):
        """Get input processor for the model.

        Args:
            model_path: Path to the model

        Returns:
            Input processor instance
        """
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        input_processor = create_input_processor(model_path,
                                                 tokenizer=tokenizer)
        return input_processor

    def get_raw_trtllm_inputs(self, modality: str, prompt: List[str],
                              media: List[str]):
        """Get inputs formatted for TensorRT-LLM model.

        Args:
            modality: Input modality type
            prompt: List of text prompts
            media: List of media file paths

        Returns:
            Tuple of (input_ids, multimodal_params_list)

        Raises:
            ValueError: If inputs are invalid or model is not initialized
        """
        model_path = getattr(
            self.hf_model.config, '_name_or_path',
            getattr(self.hf_model, 'model_config',
                    {}).get('pretrained_config', {}).get('_name_or_path', ''))
        processor = self.get_input_processor(model_path)
        inputs = default_multimodal_input_loader(
            tokenizer=processor.tokenizer,
            model_dir=model_path,
            model_type=self.get_model_type(),
            modality=modality,
            prompts=prompt,
            media=media,
            image_data_format="pt",
            num_frames=8,
            device="cpu")
        inputs = [prompt_inputs(i) for i in inputs]

        input_ids = []
        # context_sequence_lengths = []
        multimodal_params_list = []
        for input in inputs:
            input_processor_with_hash = create_input_processor_with_hash(
                processor)
            prompt_token_ids, extra_processed_inputs = input_processor_with_hash(
                input, sampling_params=None)
            input_ids.extend(prompt_token_ids)
            multimodal_params = MultimodalParams(
                multimodal_data=extra_processed_inputs.get('multimodal_data'),
                multimodal_input=extra_processed_inputs.get('multimodal_input'))
            multimodal_params.to_device(
                "multimodal_data",
                self.device,
                pin_memory=True,
                target_keywords=getattr(self.trtllm_model,
                                        "multimodal_data_device_paths", None))
            multimodal_params_list.append(multimodal_params)
        input_ids = torch.tensor(input_ids,
                                 dtype=torch.int32,
                                 device=self.device)
        return input_ids, multimodal_params_list

    def get_trtllm_inputs(
            self,
            input_ids,
            multimodal_params_list,
            is_gen: bool = False,
            num_cached_tokens_per_seq: Optional[List[int]] = None):
        """Prepare inputs for TensorRT-LLM model forward pass.

        Args:
            input_ids: Input token IDs
            multimodal_params_list: List of multimodal parameters
            is_gen: Whether this is a generation phase
            num_cached_tokens_per_seq: Number of cached tokens per sequence

        Returns:
            Dictionary of inputs for model forward pass

        Raises:
            ValueError: If attn_metadata is not initialized
        """
        if self.attn_metadata is None:
            raise ValueError(
                "attn_metadata must be initialized before calling get_trtllm_inputs"
            )

        request_ids = [1]
        prompt_lens = [input_ids.size(-1)]
        if is_gen:
            gen_input_ids = torch.tensor([900], dtype=torch.int, device='cuda')
            num_cached_tokens_per_seq = [input_ids.size(-1)]
            seq_lens = torch.tensor([gen_input_ids.size(-1)],
                                    dtype=torch.int,
                                    pin_memory=True)
            num_contexts = 0
            position_ids = [
                torch.arange(input_ids.size(-1),
                             input_ids.size(-1) + gen_input_ids.size(-1),
                             dtype=torch.int32)
            ]
            multimodal_params_list = []
            input_ids = gen_input_ids
        else:
            num_cached_tokens_per_seq = num_cached_tokens_per_seq or [0]
            seq_lens = torch.tensor([input_ids.size(-1)],
                                    dtype=torch.int,
                                    pin_memory=True)
            num_contexts = 1
            position_ids = [
                torch.arange(0, input_ids.size(-1), dtype=torch.int32)
            ]
            multimodal_runtime = MultimodalRuntimeData(
                mm_token_lengths=multimodal_params_list[0].multimodal_input.
                multimodal_lengths,
                mm_token_positions=multimodal_params_list[0].multimodal_input.
                multimodal_positions,
                past_seen_token_num=num_cached_tokens_per_seq[0],
                chunk_end_pos=num_cached_tokens_per_seq[0] + input_ids.size(-1),
                special_token_offsets=[],
            ) if multimodal_params_list[0].multimodal_input is not None else None
            multimodal_params_list[0].multimodal_runtime = multimodal_runtime

        self.attn_metadata.seq_lens = seq_lens
        self.attn_metadata.beam_width = 1
        self.attn_metadata.request_ids = request_ids
        self.attn_metadata.prompt_lens = prompt_lens
        self.attn_metadata.num_contexts = num_contexts
        self.attn_metadata.num_chunked_ctx_requests = 0
        self.attn_metadata.kv_cache_params = KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=num_cached_tokens_per_seq,
        )
        position_ids = torch.cat(position_ids).unsqueeze(0).cuda()

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attn_metadata": self.attn_metadata,
            "multimodal_params": multimodal_params_list
        }

    def get_hf_inputs(self, modality: str, prompt: List[str], media: List[str]):
        """Get inputs formatted for HuggingFace model.

        Args:
            modality: Input modality type
            prompt: List of text prompts
            media: List of media file paths

        Returns:
            Processed inputs for HuggingFace model
        """
        model_path = getattr(
            self.hf_model.config, '_name_or_path',
            getattr(self.hf_model, 'model_config',
                    {}).get('pretrained_config', {}).get('_name_or_path', ''))
        hf_processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        inputs = default_multimodal_input_loader(
            tokenizer=hf_processor.tokenizer,
            model_dir=model_path,
            model_type=self.get_model_type(),
            modality=modality,
            prompts=prompt,
            media=media,
            image_data_format="pt",
            num_frames=8,
            device="cpu")
        inputs = [prompt_inputs(i) for i in inputs]

        images = []
        videos = None

        if modality in ["image", "multiple_image"]:
            images = [input['multi_modal_data']['image'] for input in inputs]
        elif modality == "mixture_text_image":
            for input in inputs:
                if input.get('multi_modal_data', {}).get('image',
                                                         None) is not None:
                    images.append(input['multi_modal_data']['image'])
        elif modality == "video":
            images = None
            videos = [
                input['multi_modal_data'][f'{modality}'][0].frames
                for input in inputs
            ]
        elif modality == "text":
            # For text-only modality, no images or videos needed
            pass
        else:
            raise ValueError(f"Invalid modality: {modality}")
        processor_inputs = hf_processor(
            text=[input['prompt'] for input in inputs],
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
            do_rescale=False,
        ).to(self.device)
        return processor_inputs

    def run_trtllm_forward(self, trtllm_inputs, use_cuda_graph: bool = False):
        """Run forward pass on TensorRT-LLM model.

        Args:
            trtllm_inputs: Dictionary of inputs for model
            use_cuda_graph: Whether to use CUDA graph optimization

        Returns:
            Model logits output

        """
        if not use_cuda_graph:
            trtllm_inputs["attn_metadata"].prepare()
            return self.trtllm_model.forward(**trtllm_inputs)
        else:
            mock_engine = create_mock_engine(1)
            graph_runner = CUDAGraphRunner(mock_engine)
            trtllm_inputs["attn_metadata"] = trtllm_inputs[
                "attn_metadata"].create_cuda_graph_metadata(1)

            # Prepare metadata before capture (like in working Qwen2.5-VL test)
            trtllm_inputs["attn_metadata"].prepare()

            key = (1, 0, False)
            graph_runner.capture(
                key=key,
                forward_fn=lambda inputs: self.trtllm_model.forward(**inputs),
                initial_inputs=trtllm_inputs)
            for _ in range(2):
                # Run it twice. This helps us catch problems if buffers are accidentally reallocated in prepare().
                trtllm_inputs["attn_metadata"].prepare()
                logits = graph_runner.replay(key=key,
                                             current_inputs=trtllm_inputs)
            return logits.clone()

    def init_kv_cache_manager(self, scenario: Scenario):
        """Initialize KV cache manager for a test scenario.

        Args:
            scenario: Test scenario configuration

        Note:
            This method uses get_kv_cache_config() to obtain configuration.
            Override get_kv_cache_config() to customize cache settings.
        """
        # Get cache configuration from the configurable method
        cache_config = self.get_kv_cache_config(scenario)
        tokens_per_block = cache_config['tokens_per_block']
        max_seq_len = cache_config['max_seq_len']
        batch_size = cache_config['batch_size']

        num_blocks = (max_seq_len + tokens_per_block - 1) // tokens_per_block

        self.kv_cache_manager = self.get_kv_cache_manager(
            dtype=self.model_config.pretrained_config.torch_dtype,
            config=self.model_config.pretrained_config,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            num_blocks=num_blocks)

        self.kv_cache_manager.add_dummy_requests(request_ids=[1],
                                                 token_nums=[max_seq_len])

    def get_max_num_tokens(self, scenario: Scenario) -> int:
        """Get maximum number of tokens for attention metadata.

        Args:
            scenario: Test scenario configuration

        Returns:
            int: Maximum number of tokens

        Note:
            Override this method to customize token limits for your model.
        """
        if scenario.chunked_prefill:
            return 128
        elif scenario.modality == "video":
            return 16384
        else:
            return 8192

    def init_attn_metadata(self, scenario: Scenario):
        """Initialize attention metadata for a test scenario.

        Args:
            scenario: Test scenario configuration
        """

        metadata_cls = get_attention_backend(
            self.model_config.attn_backend).Metadata
        max_num_tokens = self.get_max_num_tokens(scenario)

        self.attn_metadata = metadata_cls(
            max_num_requests=16,
            max_num_sequences=1,
            max_num_tokens=max_num_tokens,
            kv_cache_manager=self.kv_cache_manager,
            runtime_features=self.runtime_features,
        )

    def run_scenario_test(self, scenario: Scenario) -> bool:
        """Run a complete test scenario including context and generation phases.

        Args:
            scenario: Test scenario configuration

        Returns:
            bool: True if test passes, False otherwise

        Note:
            This method orchestrates the entire test flow:
            1. Get test inputs
            2. Initialize KV cache and attention metadata
            3. Run context phase (with optional chunked prefill or cache reuse)
            4. Compare context outputs
            5. Run generation phase
            6. Compare generation outputs
            7. Cleanup
        """
        # Get test inputs
        prompts, media = self.get_raw_test_inputs(scenario.modality)

        # Initialize resources
        self.init_kv_cache_manager(scenario)
        self.init_attn_metadata(scenario)

        # Get raw TensorRT-LLM inputs
        trtllm_input_ids, multimodal_params_list = self.get_raw_trtllm_inputs(
            scenario.modality, prompts, media)

        result = True

        # ===== Context Phase =====
        print(f"  Running context phase...")
        with torch.inference_mode():
            if scenario.chunked_prefill:
                # Chunked prefill: process input in chunks
                chunk_size = 128
                for i in range(0, len(trtllm_input_ids), chunk_size):
                    ctx_trtllm_inputs = self.get_trtllm_inputs(
                        trtllm_input_ids[i:i + chunk_size],
                        multimodal_params_list,
                        is_gen=False,
                        num_cached_tokens_per_seq=[i])
                    logits = self.run_trtllm_forward(ctx_trtllm_inputs,
                                                     use_cuda_graph=False)

            elif scenario.kv_cache_reuse:
                # KV cache reuse: run twice with different cache states
                first = True
                for iteration in range(2):
                    num_cached_tokens_per_seq = 0 if first else [
                        trtllm_input_ids.size(-1) - 1
                    ]
                    current_trtllm_input_ids = trtllm_input_ids if first else trtllm_input_ids[
                        -1:]
                    ctx_trtllm_inputs = self.get_trtllm_inputs(
                        current_trtllm_input_ids,
                        multimodal_params_list,
                        is_gen=False,
                        num_cached_tokens_per_seq=num_cached_tokens_per_seq)
                    logits = self.run_trtllm_forward(ctx_trtllm_inputs,
                                                     use_cuda_graph=False)
                    first = False

            else:
                # Standard context processing
                ctx_trtllm_inputs = self.get_trtllm_inputs(
                    trtllm_input_ids, multimodal_params_list, is_gen=False)
                logits = self.run_trtllm_forward(ctx_trtllm_inputs,
                                                 use_cuda_graph=False)

            # Compare context outputs
            hf_inputs = self.get_hf_inputs(scenario.modality, prompts, media)
            ref = self.hf_model.forward(**hf_inputs, use_cache=True)

            try:
                self.compare_outputs(logits, ref.logits[:, -1].float())
                print(f"  ✓ Context phase passed")
            except AssertionError:
                print(f"  ✗ Context phase failed")
                result = False

        # ===== Generation Phase =====
        print(f"  Running generation phase...")
        gen_trtllm_inputs = self.get_trtllm_inputs(trtllm_input_ids,
                                                   multimodal_params_list,
                                                   is_gen=True)
        gen_hf_inputs = {
            "input_ids": gen_trtllm_inputs["input_ids"].unsqueeze(0),
            "position_ids": gen_trtllm_inputs["position_ids"],
            "past_key_values": ref.past_key_values,
            "use_cache": True
        }

        with torch.inference_mode():
            logits = self.run_trtllm_forward(gen_trtllm_inputs,
                                             scenario.use_cuda_graph)
            ref = self.hf_model.forward(**gen_hf_inputs)

            try:
                self.compare_outputs(logits, ref.logits[:, -1].float())
                print(f"  ✓ Generation phase passed")
            except AssertionError:
                print(f"  ✗ Generation phase failed")
                result = False

        # Cleanup
        self.kv_cache_manager.shutdown()
        self.attn_metadata = None

        return result

    def get_scenarios(self) -> List[Scenario]:
        """Get all scenarios to test.

        Returns:
            List[Scenario]: List of test scenarios to run

        Note:
            Override this method to customize test scenarios for your model.
            Default scenarios cover:
            - Basic modality tests (image, video, multiple_image)
            - CUDA graph optimization
            - Chunked prefill
            - KV cache reuse
        """
        scenarios = [
            # ==== Modality Sanity Checks ====
            Scenario(modality="image",
                     use_cuda_graph=False,
                     chunked_prefill=False,
                     kv_cache_reuse=False),
            Scenario(modality="video",
                     use_cuda_graph=False,
                     chunked_prefill=False,
                     kv_cache_reuse=False),
            Scenario(modality="multiple_image",
                     use_cuda_graph=False,
                     chunked_prefill=False,
                     kv_cache_reuse=False),

            # ==== CUDA Graph Scenarios ====
            Scenario(modality="image",
                     use_cuda_graph=True,
                     chunked_prefill=False,
                     kv_cache_reuse=False),

            # ==== Chunked Prefill Scenarios ====
            Scenario(modality="image",
                     use_cuda_graph=False,
                     chunked_prefill=True,
                     kv_cache_reuse=False),

            # ==== KV Cache Reuse Scenarios ====
            Scenario(modality="image",
                     use_cuda_graph=False,
                     chunked_prefill=False,
                     kv_cache_reuse=True),
        ]
        return scenarios

    def setUp(self):
        """Initialize models and configurations for testing."""
        torch.random.manual_seed(0)

        # TODO: Add multi-GPU support
        self.device = torch.device("cuda:0")

        self.hf_config = self.create_hf_config()
        self.hf_model = self.create_hf_model(self.hf_config)
        self.trtllm_model, self.model_config = self.create_trtllm_model(
            load_weights=True, hf_model_state_dict=self.hf_model.state_dict())
        self.runtime_features = AttentionRuntimeFeatures()

    def tearDown(self):
        """Cleanup resources and reset state.

        This method can be called to free GPU memory and reset test state.
        Useful when running multiple tests or debugging.
        """
        if self.kv_cache_manager is not None:
            try:
                self.kv_cache_manager.shutdown()
            except Exception as e:
                print(f"Warning: Error during KV cache manager shutdown: {e}")
            self.kv_cache_manager = None

        self.attn_metadata = None

        # Force garbage collection to free GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def setup_scenario(self, scenario: Scenario):
        # Update runtime features based on scenario
        if scenario.chunked_prefill:
            self.runtime_features = AttentionRuntimeFeatures(
                chunked_prefill=True, chunk_size=8192)
        elif scenario.kv_cache_reuse:
            self.runtime_features = AttentionRuntimeFeatures(cache_reuse=True,
                                                             chunk_size=8192)

    def test_all(self) -> None:
        """Comprehensive test covering multiple scenario types in one batch.
        The test combines regular inference, CUDA graph, chunked prefill,
        KV cache reuse, and different modalities to provide thorough coverage
        while minimizing model loading overhead.

        Raises:
            AssertionError: If any scenario fails
        """
        scenarios = self.get_scenarios()
        for scenario in scenarios:
            with self.subTest(scenario=scenario):
                self.setup_scenario(scenario)
                print(f"\n========== Testing scenario: {scenario} ==========")
                result = self.run_scenario_test(scenario)
                self.assertTrue(
                    result,
                    f"========== Scenario failed: {scenario} ==========\n")
                print(f"========== Scenario passed: {scenario} ==========\n")
