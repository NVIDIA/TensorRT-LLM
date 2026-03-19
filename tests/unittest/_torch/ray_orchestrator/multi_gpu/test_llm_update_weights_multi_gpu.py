import base64
import pickle
import re
from typing import Callable, List, Optional

import pytest
import torch
from _torch.ray_orchestrator.single_gpu.test_llm_update_weights import (
    RefHFModelWithIPCHandles,
    compare_logits,
    run_generate,
)
from torch.multiprocessing.reductions import reduce_tensor
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from utils.llm_data import llm_models_root
from utils.torch_ref import RefHFModel
from utils.util import skip_pre_blackwell

from tensorrt_llm import LLM
from tensorrt_llm._torch.auto_deploy.custom_ops.quantization.torch_quant import _quantize_nvfp4
from tensorrt_llm._torch.utils import get_device_uuid
from tensorrt_llm.llmapi import KvCacheConfig, SamplingParams


@skip_pre_blackwell
@pytest.mark.parametrize(
    "model_dir, fp8_model_dir",
    [
        ("Qwen3/Qwen3-8B", "Qwen3/Qwen3-8B-FP8"),
        ("Qwen3/Qwen3-30B-A3B", "Qwen3/Qwen3-30B-A3B-FP8"),
    ],
)
def test_llm_update_weights_fp8(model_dir, fp8_model_dir):
    model_dir = str(llm_models_root() / model_dir)
    fp8_model_dir = str(llm_models_root() / fp8_model_dir)
    additional_kwargs = {}
    if "Qwen3/Qwen3-30B-A3B" in model_dir:
        additional_kwargs["moe_config"] = {
            "backend": "DEEPGEMM",
        }
    num_hidden_layers = 1
    hf_model = RefHFModelWithIPCHandles(fp8_model_dir, num_hidden_layers=num_hidden_layers)
    tokenizer = AutoTokenizer.from_pretrained(fp8_model_dir)
    kv_cache_config = KvCacheConfig(enable_block_reuse=True, free_gpu_memory_fraction=0.1)
    llm = LLM(
        model=model_dir,
        ray_worker_extension_cls="tensorrt_llm.llmapi.rlhf_utils.WorkerExtension",
        tensor_parallel_size=2,
        load_format="dummy",
        pipeline_parallel_size=1,
        kv_cache_config=kv_cache_config,
        model_kwargs={
            "num_hidden_layers": num_hidden_layers,
            "quantization_config": {
                "activation_scheme": "dynamic",
                "fmt": "e4m3",
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
            },
        },
        **additional_kwargs,
    )

    # Generate texts from the prompts.
    prompts_texts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    prompts = [tokenizer.encode(prompt) for prompt in prompts_texts]
    del tokenizer
    sampling_params = SamplingParams(temperature=0, return_generation_logits=True, max_tokens=1024)

    ipc_handles = hf_model.get_weight_ipc_handles_serialized([0, 1])

    llm._collective_rpc("update_weights", (ipc_handles,))
    # Finalize the update weights
    llm._collective_rpc("update_weights", (None,))

    llm_logits, ref_logits = run_generate(llm, hf_model, prompts, sampling_params)
    compare_logits(llm_logits, ref_logits)


@skip_pre_blackwell
@pytest.mark.parametrize(
    "model_dir, fp8_model_dir",
    [
        ("Qwen3/Qwen3-8B", "Qwen3/Qwen3-8B-FP8"),
        ("Qwen3/Qwen3-30B-A3B", "Qwen3/Qwen3-30B-A3B-FP8"),
    ],
)
def test_llm_partial_update_weights_fp8(model_dir, fp8_model_dir):
    model_dir = str(llm_models_root() / model_dir)
    fp8_model_dir = str(llm_models_root() / fp8_model_dir)
    additional_kwargs = {}
    if "Qwen3/Qwen3-30B-A3B" in model_dir:
        additional_kwargs["moe_config"] = {
            "backend": "DEEPGEMM",
        }
    num_hidden_layers = 1
    hf_model = RefHFModelWithIPCHandles(fp8_model_dir, num_hidden_layers=num_hidden_layers)
    tokenizer = AutoTokenizer.from_pretrained(fp8_model_dir)
    kv_cache_config = KvCacheConfig(enable_block_reuse=True, free_gpu_memory_fraction=0.1)
    llm = LLM(
        model=model_dir,
        ray_worker_extension_cls="tensorrt_llm.llmapi.rlhf_utils.WorkerExtension",
        tensor_parallel_size=2,
        load_format="dummy",
        pipeline_parallel_size=1,
        kv_cache_config=kv_cache_config,
        model_kwargs={
            "num_hidden_layers": num_hidden_layers,
            "quantization_config": {
                "activation_scheme": "dynamic",
                "fmt": "e4m3",
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
            },
        },
        **additional_kwargs,
    )

    # Generate texts from the prompts.
    prompts_texts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    prompts = [tokenizer.encode(prompt) for prompt in prompts_texts]
    del tokenizer

    sampling_params = SamplingParams(temperature=0, return_generation_logits=True, max_tokens=1024)

    def common_filter(filter_name: str) -> Callable[[str], bool]:
        def filter_fn(name: str) -> bool:
            return name.endswith(filter_name)

        return filter_fn

    # Generate filter_list from model weight keys by removing layer prefix
    # e.g., "model.layers.41.input_layernorm.weight" -> "input_layernorm.weight"
    layer_prefix_pattern = re.compile(r"^model\.layers\.\d+\.")
    filter_set = set()
    for name, _ in hf_model.all_weights[hf_model.device_id]:
        suffix = layer_prefix_pattern.sub("", name)
        filter_set.add(suffix)
    filter_list = list(filter_set)

    for filter_name in filter_list:
        weight_filter = common_filter(filter_name=filter_name)
        ipc_handles = hf_model.get_weight_ipc_handles_serialized(
            [0, 1], weight_filter=weight_filter
        )
        llm._collective_rpc("update_weights", (ipc_handles,))
    # Finalize the update weights
    llm._collective_rpc("update_weights", (None,))

    llm_logits, ref_logits = run_generate(llm, hf_model, prompts, sampling_params)
    compare_logits(llm_logits, ref_logits)


class RefNVFP4ModelWithIPCHandles(RefHFModel):
    """Reference model that loads bf16 weights from HuggingFace and quantizes
    them to NVFP4 format for use in update_weights tests.

    Since HuggingFace does not host NVFP4 checkpoints, this class:
    1. Loads the bf16 model from HF
    2. Quantizes each linear weight to NVFP4 using _quantize_nvfp4
    3. Provides IPC handles with NVFP4 weight keys (weight, weight_scale, weight_scale_2)
    """

    NVFP4_BLOCK_SIZE = 16

    # Patterns for determining which parameters to quantize
    EXCLUDE_PATTERNS = [
        "embed_tokens",
        "lm_head",
        "layernorm",
        "norm",
        "ln_",
        "embeddings",
        "mlp.gate.weight",  # MoE router
    ]
    INCLUDE_PATTERNS = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "fc1",
        "fc2",
        "mlp",
    ]

    # Fusion groups: projections that TRT-LLM fuses together must share weight_scale_2.
    FUSION_GROUPS = (
        ("q_proj", "k_proj", "v_proj"),
        ("gate_proj", "up_proj"),
    )
    _PROJ_TO_GROUP = {proj: group for group in FUSION_GROUPS for proj in group}

    def __init__(self, model_dir: str, device_id: int = 0, num_hidden_layers: Optional[int] = None):
        self.device_id = device_id
        config = AutoConfig.from_pretrained(model_dir)
        if num_hidden_layers is not None:
            config.num_hidden_layers = num_hidden_layers
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, config=config, torch_dtype=torch.bfloat16, attn_implementation="eager"
        ).to(f"cuda:{device_id}")
        self.all_weights = {}
        self.device_uuid = [get_device_uuid(i) for i in range(torch.cuda.device_count())]
        self._quantize_and_replicate_weights()

    def _quantize_and_replicate_weights(self):
        """Quantize linear weights to NVFP4 and replicate across devices.

        Projections in the same fusion group (qkv, gate_up) share a unified
        weight_scale_2 computed from their joint amax, matching the behavior
        of verl's TRTLLMNVFP4QuantizerHelper.
        """
        # First pass: collect all params, buffer fusion groups
        all_params = [
            (name, param.detach().clone()) for name, param in self.model.named_parameters()
        ]

        # Buffer: {(layer_prefix, group): {proj_type: (name, tensor)}}
        fusion_buffer: dict[tuple, dict] = {}
        model_weights = []

        for name, p in all_params:
            if not self._should_quantize(name):
                model_weights.append((name, p))
                continue

            proj_type, group = self._get_proj_info(name)
            if proj_type is not None:
                layer_prefix = name[: name.index(proj_type)].rstrip(".")
                buf_key = (layer_prefix, group)
                fusion_buffer.setdefault(buf_key, {})[proj_type] = (name, p)

                # Flush when all members of the group are collected
                if all(proj in fusion_buffer[buf_key] for proj in group):
                    projs = fusion_buffer.pop(buf_key)
                    model_weights.extend(self._quantize_fusion_group(group, projs))
            else:
                model_weights.extend(self._quantize_single_weight(name, p))

        assert not fusion_buffer, f"Incomplete fusion groups: {list(fusion_buffer.keys())}"

        self.all_weights[self.device_id] = model_weights
        for i in range(torch.cuda.device_count()):
            if i != self.device_id:
                self.all_weights[i] = [(n, p.to(f"cuda:{i}")) for n, p in model_weights]

    @classmethod
    def _should_quantize(cls, name: str) -> bool:
        """Determine whether to quantize a parameter to NVFP4."""
        if not name.endswith(".weight"):
            return False
        name_lower = name.lower()
        for pattern in cls.EXCLUDE_PATTERNS:
            if pattern in name_lower:
                return False
        for pattern in cls.INCLUDE_PATTERNS:
            if pattern in name_lower:
                return True
        return False

    @classmethod
    def _get_proj_info(cls, name: str):
        """Return (proj_type, fusion_group) if param belongs to a fusion group."""
        for proj, group in cls._PROJ_TO_GROUP.items():
            if proj in name:
                return proj, group
        return None, None

    def _quantize_single_weight(self, name: str, weight: torch.Tensor) -> List[tuple]:
        """Quantize a single (non-fused) weight to NVFP4."""
        return list(self._do_quantize(name, weight))

    def _quantize_fusion_group(self, group: tuple, projs: dict) -> List[tuple]:
        """Quantize a fusion group with unified weight_scale_2."""
        # Compute joint amax across all members
        all_amaxes = [v.float().abs().amax() for _, v in projs.values()]
        unified_scale_2 = torch.stack(all_amaxes).amax() / (6.0 * 448.0)

        results = []
        for proj_type in group:
            if proj_type in projs:
                name, weight = projs[proj_type]
                results.extend(self._do_quantize(name, weight, unified_scale_2))
        return results

    def _do_quantize(
        self,
        name: str,
        weight: torch.Tensor,
        weight_scale_2: Optional[torch.Tensor] = None,
    ) -> List[tuple]:
        """Core NVFP4 quantization. Returns [(name, packed), (name_scale, bs), (name_scale_2, s2)]."""
        weight_float = weight.float()
        if weight_scale_2 is None:
            weight_scale_2 = weight_float.abs().amax().float() / (6.0 * 448.0)

        packed_weight, block_scale = _quantize_nvfp4(
            weight_float, self.NVFP4_BLOCK_SIZE, weight_scale_2
        )
        return [
            (name, packed_weight.to(torch.uint8)),
            (name + "_scale", block_scale.to(torch.float8_e4m3fn)),
            (name + "_scale_2", weight_scale_2),
        ]

    def get_weight_ipc_handles_serialized(
        self,
        device_ids: Optional[List[int]] = None,
        weight_filter: Optional[Callable[[str], bool]] = None,
    ):
        ret = {}
        device_list = list(range(torch.cuda.device_count())) if device_ids is None else device_ids

        for device in device_list:
            all_handles = []
            for item in self.all_weights[device]:
                name, p = item
                if weight_filter is not None and not weight_filter(name):
                    continue
                handle = reduce_tensor(p)
                all_handles.append((name, handle))

            serialized = base64.b64encode(pickle.dumps(all_handles)).decode("utf-8")
            ret[self.device_uuid[device]] = serialized

        return ret


@skip_pre_blackwell
@pytest.mark.parametrize(
    "model_dir",
    [
        "Qwen3/Qwen3-30B-A3B",
    ],
)
def test_llm_update_weights_nvfp4(model_dir):
    model_dir = str(llm_models_root() / model_dir)
    hf_model = RefNVFP4ModelWithIPCHandles(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    kv_cache_config = KvCacheConfig(enable_block_reuse=True, free_gpu_memory_fraction=0.1)
    llm = LLM(
        model=model_dir,
        ray_worker_extension_cls="tensorrt_llm.llmapi.rlhf_utils.WorkerExtension",
        tensor_parallel_size=2,
        load_format="dummy",
        pipeline_parallel_size=1,
        kv_cache_config=kv_cache_config,
        force_dynamic_quantization=True,
        model_kwargs={
            "quantization_config": {
                "quant_method": "nvfp4",
                "group_size": 16,
            },
        },
    )

    prompts_texts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    prompts = [tokenizer.encode(prompt) for prompt in prompts_texts]
    del tokenizer
    sampling_params = SamplingParams(temperature=0, return_generation_logits=True, max_tokens=1024)

    ipc_handles = hf_model.get_weight_ipc_handles_serialized([0, 1])
    llm._collective_rpc("update_weights", (ipc_handles,))
    llm._collective_rpc("update_weights", (None,))

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    sampling_params = SamplingParams(temperature=0, max_tokens=64)
    outputs = llm.generate(prompts, sampling_params)
    for i, output in enumerate(outputs):
        text = tokenizer.decode(output.outputs[0].token_ids)
        print(f"[NVFP4 update] Prompt {i}: {prompts_texts[i]!r} -> {text!r}")


@skip_pre_blackwell
@pytest.mark.parametrize(
    "model_dir",
    [
        "Qwen3/Qwen3-30B-A3B",
    ],
)
def test_llm_partial_update_weights_nvfp4(model_dir):
    model_dir = str(llm_models_root() / model_dir)
    num_hidden_layers = 1
    hf_model = RefNVFP4ModelWithIPCHandles(model_dir, num_hidden_layers=num_hidden_layers)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    kv_cache_config = KvCacheConfig(enable_block_reuse=True, free_gpu_memory_fraction=0.1)
    llm = LLM(
        model=model_dir,
        ray_worker_extension_cls="tensorrt_llm.llmapi.rlhf_utils.WorkerExtension",
        tensor_parallel_size=2,
        load_format="dummy",
        pipeline_parallel_size=1,
        kv_cache_config=kv_cache_config,
        force_dynamic_quantization=True,
        model_kwargs={
            "num_hidden_layers": num_hidden_layers,
            "quantization_config": {
                "quant_method": "nvfp4",
                "group_size": 16,
            },
        },
    )

    prompts_texts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    prompts = [tokenizer.encode(prompt) for prompt in prompts_texts]

    sampling_params = SamplingParams(temperature=0, max_tokens=64)

    def common_filter(filter_name: str) -> Callable[[str], bool]:
        def filter_fn(name: str) -> bool:
            return name.endswith(filter_name)

        return filter_fn

    # Generate filter_list from model weight keys by removing layer prefix
    layer_prefix_pattern = re.compile(r"^model\.layers\.\d+\.")
    filter_set = set()
    for name, _ in hf_model.all_weights[hf_model.device_id]:
        suffix = layer_prefix_pattern.sub("", name)
        filter_set.add(suffix)
    filter_list = list(filter_set)

    for filter_name in filter_list:
        weight_filter = common_filter(filter_name=filter_name)
        ipc_handles = hf_model.get_weight_ipc_handles_serialized(
            [0, 1], weight_filter=weight_filter
        )
        llm._collective_rpc("update_weights", (ipc_handles,))
    llm._collective_rpc("update_weights", (None,))

    outputs = llm.generate(prompts, sampling_params)
    for i, output in enumerate(outputs):
        text = tokenizer.decode(output.outputs[0].token_ids)
        print(f"[NVFP4 partial] Prompt {i}: {prompts_texts[i]!r} -> {text!r}")
