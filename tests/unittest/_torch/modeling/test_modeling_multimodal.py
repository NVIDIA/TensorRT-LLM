import gc
import os
import unittest
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import torch
from _torch.helpers import create_mock_engine
from transformers import AutoProcessor, AutoTokenizer, PretrainedConfig, PreTrainedModel
from utils.llm_data import llm_models_root

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.interface import AttentionRuntimeFeatures
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.inputs import (
    create_input_processor,
    create_input_processor_with_hash,
    default_multimodal_input_loader,
    prompt_inputs,
)
from tensorrt_llm.inputs.multimodal import MultimodalParams, MultimodalRuntimeData
from tensorrt_llm.mapping import Mapping


@dataclass(repr=False)
class MultimodalScenario:
    """Configuration for a multimodal test scenario.

    Attributes:
        modality: Input type ("image", "video", "text", "multiple_image", "mixture_text_image")
        use_cuda_graph: Enable CUDA graph optimization for generation
        chunked_prefill: Process sequences in chunks to reduce memory
        kv_cache_reuse: Test KV cache reuse across iterations
    """

    modality: str = "image"
    use_cuda_graph: bool = False
    chunked_prefill: bool = False
    kv_cache_reuse: bool = False

    def __post_init__(self):
        """Validate scenario configuration."""
        valid_modalities = ["image", "video", "text", "multiple_image", "mixture_text_image"]
        if self.modality not in valid_modalities:
            raise ValueError(
                f"Invalid modality '{self.modality}'. "
                f"Valid modalities: {', '.join(valid_modalities)}"
            )

    def __repr__(self) -> str:
        """Return string representation of the scenario."""
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
    @abstractmethod
    def get_model_config(self) -> Dict:
        """Return the model configuration dictionary."""

    @abstractmethod
    def get_trtllm_model_class(self) -> Type:
        """Return the TensorRT-LLM model class."""

    @abstractmethod
    def get_hf_model_class(self) -> Type:
        """Return the HuggingFace model class."""

    @abstractmethod
    def get_weight_mapper_class(self) -> Type:
        """Return the weight mapper class."""

    @abstractmethod
    def get_model_type(self) -> str:
        """Return the model type string."""

    @abstractmethod
    def get_model_config_class(self) -> Type:
        """Return the model configuration class."""

    def get_dtype(self) -> torch.dtype:
        """Return the model data type (default: from config's torch_dtype field)."""
        config = self.get_model_config()
        dtype_str = config.get("torch_dtype", "bfloat16")

        if isinstance(dtype_str, torch.dtype):
            return dtype_str
        return str_dtype_to_torch(dtype_str)

    def get_raw_test_inputs(self, modality: str) -> Tuple[List[str], List[str]]:
        """Get test inputs (prompts, media_paths) for a given modality.

        Override to provide custom test inputs. Default test data location:
        {LLM_MODELS_ROOT}/multimodals/test_data/
        """
        test_data_root = Path(os.path.join(llm_models_root(), "multimodals", "test_data"))

        if modality == "image":
            prompts = ["Describe the natural environment in the image."]
            media = [str(test_data_root / "seashore.png")]
        elif modality == "multiple_image":
            prompts = ["Describe the difference between the two images."]
            media = [str(test_data_root / "inpaint.png"), str(test_data_root / "61.jpg")]
        elif modality == "video":
            prompts = ["Tell me what you see in the video briefly."]
            media = [str(test_data_root / "OAI-sora-tokyo-walk.mp4")]
        elif modality == "mixture_text_image":
            prompts = ["Describe the scene in the image briefly.", "Who invented the internet?"]
            media = [[str(test_data_root / "inpaint.png")], []]
        elif modality == "text":
            prompts = ["Who invented the internet?"]
            media = []
        else:
            raise ValueError(
                f"Invalid modality: {modality}. "
                f"Supported modalities: image, multiple_image, video, mixture_text_image, text"
            )

        return prompts, media

    def create_hf_config(self) -> PretrainedConfig:
        config_dict = deepcopy(self.get_model_config())
        config_class = self.get_model_config_class()
        hf_config = config_class.from_dict(config_dict)
        return hf_config

    def create_trtllm_model(
        self, load_weights: bool = False, hf_model_state_dict: Optional[Dict] = None, **kwargs
    ) -> Tuple[PreTrainedModel, ModelConfig]:
        """Create a TensorRT-LLM model instance."""

        model_config = ModelConfig(pretrained_config=self.hf_config)
        model_class = self.get_trtllm_model_class()
        model = model_class(model_config, **kwargs).to("cuda")

        if load_weights:
            weight_mapper_class = self.get_weight_mapper_class()
            if weight_mapper_class is not None:
                weight_mapper = weight_mapper_class()
                weight_mapper.init_model_and_config(model, self.hf_config)
                model.load_weights(hf_model_state_dict, weight_mapper)
            else:
                model.load_weights(hf_model_state_dict)

        return model, model_config

    def create_hf_model(self, pretrained_config: PretrainedConfig) -> PreTrainedModel:
        """Create a HuggingFace model instance."""
        hf_model_class = self.get_hf_model_class()
        hf_model = hf_model_class(pretrained_config).to(self.device).to(self.get_dtype())
        hf_model.eval()

        return hf_model

    def get_tolerance(self) -> Tuple[float, float]:
        """Get tolerance values (atol, rtol) for output comparison."""
        return 0.4, 0.4

    def compare_outputs(
        self,
        trtllm_output: torch.Tensor,
        hf_output: torch.Tensor,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
    ) -> bool:
        """Compare TensorRT-LLM and HuggingFace outputs within tolerance."""
        if atol is None or rtol is None:
            default_atol, default_rtol = self.get_tolerance()
            atol = atol if atol is not None else default_atol
            rtol = rtol if rtol is not None else default_rtol

        try:
            torch.testing.assert_close(trtllm_output, hf_output, atol=atol, rtol=rtol)
            return True
        except AssertionError:
            max_diff = torch.max(torch.abs(trtllm_output - hf_output)).item()
            mean_diff = torch.mean(torch.abs(trtllm_output - hf_output)).item()
            print("Output comparison failed:")
            print(f"  Max absolute difference: {max_diff}")
            print(f"  Mean absolute difference: {mean_diff}")
            print(f"  Tolerance: atol={atol}, rtol={rtol}")
            raise

    def get_kv_cache_config(self, scenario: MultimodalScenario) -> Dict:
        """Get KV cache configuration (tokens_per_block, max_seq_len, batch_size)."""
        tokens_per_block = 128
        batch_size = 16

        if scenario.modality == "video":
            max_seq_len = 16384
        else:
            max_seq_len = 8192

        return {
            "tokens_per_block": tokens_per_block,
            "max_seq_len": max_seq_len,
            "batch_size": batch_size,
        }

    def get_kv_cache_manager(
        self,
        dtype: torch.dtype,
        config: PretrainedConfig,
        tokens_per_block: int,
        max_seq_len: int,
        batch_size: int,
        num_blocks: int,
    ):
        """Get KV cache manager for testing."""
        dtype_map = {
            torch.half: tensorrt_llm.bindings.DataType.HALF,
            torch.float16: tensorrt_llm.bindings.DataType.HALF,
            torch.bfloat16: tensorrt_llm.bindings.DataType.BF16,
        }

        if dtype not in dtype_map:
            raise ValueError(
                f"Unsupported dtype: {dtype}. Supported dtypes: {list(dtype_map.keys())}"
            )

        kv_cache_dtype = dtype_map[dtype]
        mapping = Mapping(world_size=1, tp_size=1, rank=0)
        kv_cache_config = KvCacheConfig(max_tokens=num_blocks * tokens_per_block)

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
        """Get input processor for the model."""
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        input_processor = create_input_processor(model_path, tokenizer=tokenizer)
        return input_processor

    def get_raw_trtllm_inputs(self, modality: str, prompt: List[str], media: List[str]):
        """Get inputs formatted for TensorRT-LLM model."""
        model_path = getattr(
            self.hf_model.config,
            "_name_or_path",
            getattr(self.hf_model, "model_config", {})
            .get("pretrained_config", {})
            .get("_name_or_path", ""),
        )
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
            device="cpu",
        )
        inputs = [prompt_inputs(i) for i in inputs]

        input_ids = []
        multimodal_params_list = []
        for input in inputs:
            input_processor_with_hash = create_input_processor_with_hash(processor)
            prompt_token_ids, extra_processed_inputs = input_processor_with_hash(
                input, sampling_params=None
            )
            input_ids.extend(prompt_token_ids)
            multimodal_params = MultimodalParams(
                multimodal_data=extra_processed_inputs.get("multimodal_data"),
                multimodal_input=extra_processed_inputs.get("multimodal_input"),
            )
            multimodal_params.to_device(
                "multimodal_data",
                self.device,
                pin_memory=True,
                target_keywords=getattr(self.trtllm_model, "multimodal_data_device_paths", None),
            )
            multimodal_params_list.append(multimodal_params)
        input_ids = torch.tensor(input_ids, dtype=torch.int32, device=self.device)
        return input_ids, multimodal_params_list

    def get_trtllm_inputs(
        self,
        input_ids,
        multimodal_params_list,
        is_gen: bool = False,
        num_cached_tokens_per_seq: Optional[List[int]] = None,
    ):
        """Prepare inputs for TensorRT-LLM model forward pass."""
        if self.attn_metadata is None:
            raise ValueError("attn_metadata must be initialized before calling get_trtllm_inputs")

        request_ids = [1]
        prompt_lens = [input_ids.size(-1)]
        if is_gen:
            gen_input_ids = torch.tensor([900], dtype=torch.int, device="cuda")
            num_cached_tokens_per_seq = [input_ids.size(-1)]
            seq_lens = torch.tensor([gen_input_ids.size(-1)], dtype=torch.int, pin_memory=True)
            num_contexts = 0
            position_ids = [
                torch.arange(
                    input_ids.size(-1),
                    input_ids.size(-1) + gen_input_ids.size(-1),
                    dtype=torch.int32,
                )
            ]
            multimodal_params_list = []
            input_ids = gen_input_ids
        else:
            num_cached_tokens_per_seq = num_cached_tokens_per_seq or [0]
            seq_lens = torch.tensor([input_ids.size(-1)], dtype=torch.int, pin_memory=True)
            num_contexts = 1
            position_ids = [torch.arange(0, input_ids.size(-1), dtype=torch.int32)]
            multimodal_runtime = (
                MultimodalRuntimeData(
                    mm_token_lengths=multimodal_params_list[0].multimodal_input.multimodal_lengths,
                    mm_token_positions=multimodal_params_list[
                        0
                    ].multimodal_input.multimodal_positions,
                    past_seen_token_num=num_cached_tokens_per_seq[0],
                    chunk_end_pos=num_cached_tokens_per_seq[0] + input_ids.size(-1),
                    special_token_offsets=[],
                )
                if multimodal_params_list[0].multimodal_input is not None
                else None
            )
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
            "multimodal_params": multimodal_params_list,
        }

    def get_hf_inputs(self, modality: str, prompt: List[str], media: List[str]):
        """Get inputs formatted for HuggingFace model."""
        model_path = getattr(
            self.hf_model.config,
            "_name_or_path",
            getattr(self.hf_model, "model_config", {})
            .get("pretrained_config", {})
            .get("_name_or_path", ""),
        )
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
            device="cpu",
        )
        inputs = [prompt_inputs(i) for i in inputs]

        images = []
        videos = None

        if modality in ["image", "multiple_image"]:
            images = [input["multi_modal_data"]["image"] for input in inputs]
        elif modality == "mixture_text_image":
            for input in inputs:
                if input.get("multi_modal_data", {}).get("image", None) is not None:
                    images.append(input["multi_modal_data"]["image"])
        elif modality == "video":
            images = None
            videos = [input["multi_modal_data"][f"{modality}"][0].frames for input in inputs]
        elif modality == "text":
            # For text-only modality, no images or videos needed
            pass
        else:
            raise ValueError(f"Invalid modality: {modality}")
        processor_inputs = hf_processor(
            text=[input["prompt"] for input in inputs],
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
            do_rescale=False,
        ).to(self.device)
        return processor_inputs

    def run_trtllm_forward(self, trtllm_inputs, use_cuda_graph: bool = False):
        """Run forward pass on TensorRT-LLM model."""
        if not use_cuda_graph:
            trtllm_inputs["attn_metadata"].prepare()
            return self.trtllm_model.forward(**trtllm_inputs)
        else:
            mock_engine = create_mock_engine(1)
            graph_runner = CUDAGraphRunner(mock_engine)
            trtllm_inputs["attn_metadata"] = trtllm_inputs[
                "attn_metadata"
            ].create_cuda_graph_metadata(1)

            # Prepare metadata before capture (like in working Qwen2.5-VL test)
            trtllm_inputs["attn_metadata"].prepare()

            key = (1, 0, False)
            graph_runner.capture(
                key=key,
                forward_fn=lambda inputs: self.trtllm_model.forward(**inputs),
                initial_inputs=trtllm_inputs,
            )
            for _ in range(2):
                # Run it twice. This helps us catch problems if buffers are accidentally reallocated in prepare().
                trtllm_inputs["attn_metadata"].prepare()
                logits = graph_runner.replay(key=key, current_inputs=trtllm_inputs)
            return logits.clone()

    def init_kv_cache_manager(self, scenario: MultimodalScenario):
        """Initialize KV cache manager for a test scenario.

        Args:
            scenario: Test scenario configuration

        Note:
            This method uses get_kv_cache_config() to obtain configuration.
            Override get_kv_cache_config() to customize cache settings.
        """
        # Get cache configuration from the configurable method
        cache_config = self.get_kv_cache_config(scenario)
        tokens_per_block = cache_config["tokens_per_block"]
        max_seq_len = cache_config["max_seq_len"]
        batch_size = cache_config["batch_size"]

        num_blocks = (max_seq_len + tokens_per_block - 1) // tokens_per_block

        self.kv_cache_manager = self.get_kv_cache_manager(
            dtype=self.model_config.pretrained_config.torch_dtype,
            config=self.model_config.pretrained_config,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            num_blocks=num_blocks,
        )

        self.kv_cache_manager.add_dummy_requests(request_ids=[1], token_nums=[max_seq_len])

    def get_max_num_tokens(self, scenario: MultimodalScenario) -> int:
        """Get maximum number of tokens for attention metadata."""
        if scenario.chunked_prefill:
            return 128
        elif scenario.modality == "video":
            return 16384
        else:
            return 8192

    def init_attn_metadata(self, scenario: MultimodalScenario):
        """Initialize attention metadata for a test scenario."""
        metadata_cls = get_attention_backend(self.model_config.attn_backend).Metadata
        max_num_tokens = self.get_max_num_tokens(scenario)

        self.attn_metadata = metadata_cls(
            max_num_requests=16,
            max_num_sequences=1,
            max_num_tokens=max_num_tokens,
            kv_cache_manager=self.kv_cache_manager,
            runtime_features=self.runtime_features,
        )

    def run_scenario_test(self, scenario: MultimodalScenario) -> bool:
        """Run a complete test scenario including context and generation phases."""
        prompts, media = self.get_raw_test_inputs(scenario.modality)
        trtllm_input_ids, multimodal_params_list = self.get_raw_trtllm_inputs(
            scenario.modality, prompts, media
        )

        result = True

        # Context Phase
        print("  Running context phase...")
        with torch.inference_mode():
            if scenario.chunked_prefill:
                # Chunked prefill: process input in chunks
                chunk_size = 128
                for i in range(0, len(trtllm_input_ids), chunk_size):
                    ctx_trtllm_inputs = self.get_trtllm_inputs(
                        trtllm_input_ids[i : i + chunk_size],
                        multimodal_params_list,
                        is_gen=False,
                        num_cached_tokens_per_seq=[i],
                    )
                    logits = self.run_trtllm_forward(ctx_trtllm_inputs, use_cuda_graph=False)

            elif scenario.kv_cache_reuse:
                # KV cache reuse: run twice with different cache states
                first = True
                for iteration in range(2):
                    num_cached_tokens_per_seq = 0 if first else [trtllm_input_ids.size(-1) - 1]
                    current_trtllm_input_ids = trtllm_input_ids if first else trtllm_input_ids[-1:]
                    ctx_trtllm_inputs = self.get_trtllm_inputs(
                        current_trtllm_input_ids,
                        multimodal_params_list,
                        is_gen=False,
                        num_cached_tokens_per_seq=num_cached_tokens_per_seq,
                    )
                    logits = self.run_trtllm_forward(ctx_trtllm_inputs, use_cuda_graph=False)
                    first = False

            else:
                # Standard context processing
                ctx_trtllm_inputs = self.get_trtllm_inputs(
                    trtllm_input_ids, multimodal_params_list, is_gen=False
                )
                logits = self.run_trtllm_forward(ctx_trtllm_inputs, use_cuda_graph=False)

            # Compare context outputs
            hf_inputs = self.get_hf_inputs(scenario.modality, prompts, media)
            ref = self.hf_model.forward(**hf_inputs, use_cache=True)

            try:
                self.compare_outputs(logits, ref.logits[:, -1].float())
                print("  ✓ Context phase passed")
            except AssertionError:
                print("  ✗ Context phase failed")
                result = False

        # Generation Phase
        print("  Running generation phase...")
        gen_trtllm_inputs = self.get_trtllm_inputs(
            trtllm_input_ids, multimodal_params_list, is_gen=True
        )
        gen_hf_inputs = {
            "input_ids": gen_trtllm_inputs["input_ids"].unsqueeze(0),
            "position_ids": gen_trtllm_inputs["position_ids"],
            "past_key_values": ref.past_key_values,
            "use_cache": True,
        }

        with torch.inference_mode():
            logits = self.run_trtllm_forward(gen_trtllm_inputs, scenario.use_cuda_graph)
            ref = self.hf_model.forward(**gen_hf_inputs)

            try:
                self.compare_outputs(logits, ref.logits[:, -1].float())
                print("  ✓ Generation phase passed")
            except AssertionError:
                print("  ✗ Generation phase failed")
                result = False

        self.kv_cache_manager.shutdown()
        self.attn_metadata = None

        return result

    def get_scenarios(self) -> List[MultimodalScenario]:
        """Get all scenarios to test. Override to customize test scenarios."""
        scenarios = [
            # ==== Modality Sanity Checks ====
            MultimodalScenario(
                modality="image", use_cuda_graph=False, chunked_prefill=False, kv_cache_reuse=False
            ),
            MultimodalScenario(
                modality="video", use_cuda_graph=False, chunked_prefill=False, kv_cache_reuse=False
            ),
            MultimodalScenario(
                modality="multiple_image",
                use_cuda_graph=False,
                chunked_prefill=False,
                kv_cache_reuse=False,
            ),
            # ==== CUDA Graph Scenarios ====
            MultimodalScenario(
                modality="image", use_cuda_graph=True, chunked_prefill=False, kv_cache_reuse=False
            ),
            # ==== Chunked Prefill Scenarios ====
            MultimodalScenario(
                modality="image", use_cuda_graph=False, chunked_prefill=True, kv_cache_reuse=False
            ),
            # ==== KV Cache Reuse Scenarios ====
            MultimodalScenario(
                modality="image", use_cuda_graph=False, chunked_prefill=False, kv_cache_reuse=True
            ),
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
            load_weights=True, hf_model_state_dict=self.hf_model.state_dict()
        )
        self.runtime_features = AttentionRuntimeFeatures()

    def tearDown(self):
        """Cleanup resources and free GPU memory."""
        if self.kv_cache_manager is not None:
            try:
                self.kv_cache_manager.shutdown()
            except Exception as e:
                print(f"Warning: Error during KV cache manager shutdown: {e}")
            self.kv_cache_manager = None

        self.attn_metadata = None
        self.runtime_features = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def setup_scenario(self, scenario: MultimodalScenario):
        """Update runtime features based on scenario."""
        if scenario.chunked_prefill:
            self.runtime_features = AttentionRuntimeFeatures(chunked_prefill=True, chunk_size=8192)
        elif scenario.kv_cache_reuse:
            self.runtime_features = AttentionRuntimeFeatures(cache_reuse=True, chunk_size=8192)
        self.init_kv_cache_manager(scenario)
        self.init_attn_metadata(scenario)

    def test_all(self) -> None:
        """Test all scenarios defined in get_scenarios()."""
        scenarios = self.get_scenarios()
        for scenario in scenarios:
            with self.subTest(scenario=scenario):
                self.setup_scenario(scenario)
                print(f"\n========== Testing scenario: {scenario} ==========")
                result = self.run_scenario_test(scenario)
                self.assertTrue(result, f"========== Scenario failed: {scenario} ==========\n")
                print(f"========== Scenario passed: {scenario} ==========\n")
