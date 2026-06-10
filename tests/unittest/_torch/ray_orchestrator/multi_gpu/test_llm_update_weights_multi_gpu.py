import base64
import gc
import pickle
import re
from typing import Callable, List, Optional, Tuple

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
from utils.util import skip_pre_blackwell, skip_pre_hopper

from tensorrt_llm import LLM
from tensorrt_llm._torch.auto_deploy.custom_ops.quantization.torch_quant import (
    _dequantize_nvfp4,
    _quantize_nvfp4,
)
from tensorrt_llm._torch.utils import get_device_uuid
from tensorrt_llm.evaluate import GSM8K, MMLU
from tensorrt_llm.llmapi import KvCacheConfig, MoeConfig, SamplingParams


@pytest.mark.part0
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


@pytest.mark.part1
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
    """Reference model that loads bf16 weights from HuggingFace, quantizes
    them to NVFP4 format, and keeps a round-tripped bf16 model for reference
    inference.

    Since HuggingFace cannot run NVFP4 inference natively, this class:
    1. Loads the bf16 model from HF
    2. Quantizes each linear weight to NVFP4 using _quantize_nvfp4
    3. Dequantizes back to bf16 and replaces model parameters (round-trip),
       so the HF model can serve as a reference for logits comparison
    4. Provides IPC handles with NVFP4 weight keys (weight, weight_scale, weight_scale_2)
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
        self._dequantized_weights = {}
        self.device_uuid = [get_device_uuid(i) for i in range(torch.cuda.device_count())]
        self._quantize_and_replicate_weights()

    @staticmethod
    def _unfuse_moe_expert_params(
        all_params: List[tuple],
    ) -> Tuple[List[tuple], dict]:
        """Unfuse HF transformers >=5.x fused MoE expert parameters.

        HF stores MoE experts as fused 3D ``nn.Parameter``s whose names do
        not end in ``.weight`` (so they would silently bypass
        ``_should_quantize``):

            ...mlp.experts.gate_up_proj  -> [num_experts, 2*intermediate, hidden]
            ...mlp.experts.down_proj     -> [num_experts, hidden, intermediate]

        Split them into per-expert names that match the TRT-LLM weight
        loader convention so quantization and IPC weight upload route them
        correctly:

            ...mlp.experts.{i}.gate_proj.weight
            ...mlp.experts.{i}.up_proj.weight
            ...mlp.experts.{i}.down_proj.weight

        Also returns a ``refuse_map`` that records, per unfused-param name,
        the originating fused-param name plus how to slice it back. This
        lets the dequantize-and-copy-back step write per-expert dequantized
        weights into the HF model's fused parameter tensors, keeping the HF
        reference and the TRT-LLM weights numerically aligned.
        """
        unfused: List[tuple] = []
        # name -> (fused_param_name, expert_index, "gate" | "up" | "down")
        refuse_map: dict = {}
        for name, p in all_params:
            if name.endswith(".experts.gate_up_proj") and p.dim() == 3:
                prefix = name[: -len("gate_up_proj")]
                num_experts = p.shape[0]
                half = p.shape[1] // 2
                for i in range(num_experts):
                    g_name = f"{prefix}{i}.gate_proj.weight"
                    u_name = f"{prefix}{i}.up_proj.weight"
                    unfused.append((g_name, p[i, :half, :].clone()))
                    unfused.append((u_name, p[i, half:, :].clone()))
                    refuse_map[g_name] = (name, i, "gate")
                    refuse_map[u_name] = (name, i, "up")
            elif name.endswith(".experts.down_proj") and p.dim() == 3:
                prefix = name[: -len("down_proj")]
                num_experts = p.shape[0]
                for i in range(num_experts):
                    d_name = f"{prefix}{i}.down_proj.weight"
                    unfused.append((d_name, p[i].clone()))
                    refuse_map[d_name] = (name, i, "down")
            else:
                unfused.append((name, p))
        return unfused, refuse_map

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
        # transformers >=5.x ships fused 3D MoE expert parameters; split them
        # back into per-expert ``.weight`` entries so the quantize loop and
        # the TRT-LLM weight loader can match them.
        all_params, moe_refuse_map = self._unfuse_moe_expert_params(all_params)

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

        with torch.no_grad():
            param_dict = dict(self.model.named_parameters())
            for name, dequant_weight in self._dequantized_weights.items():
                if name in param_dict:
                    param_dict[name].copy_(dequant_weight)
                elif name in moe_refuse_map:
                    # Per-expert dequantized weight that came from an
                    # unfused 3D MoE param: write back into the fused
                    # tensor's expert slice.
                    fused_name, expert_idx, kind = moe_refuse_map[name]
                    fused = param_dict[fused_name]
                    if kind == "gate":
                        half = fused.shape[1] // 2
                        fused[expert_idx, :half, :].copy_(dequant_weight)
                    elif kind == "up":
                        half = fused.shape[1] // 2
                        fused[expert_idx, half:, :].copy_(dequant_weight)
                    else:  # "down"
                        fused[expert_idx].copy_(dequant_weight)
        del self._dequantized_weights

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
        packed_uint8 = packed_weight.to(torch.uint8)
        block_scale_fp8 = block_scale.to(torch.float8_e4m3fn)

        self._dequantized_weights[name] = _dequantize_nvfp4(
            packed_uint8,
            block_scale_fp8,
            weight_scale_2,
            weight_float.shape,
            torch.bfloat16,
        )

        return [
            (name, packed_uint8),
            (name + "_scale", block_scale_fp8),
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


@pytest.mark.part2
@skip_pre_blackwell
@pytest.mark.parametrize(
    "model_dir",
    [
        "Qwen3/Qwen3-30B-A3B",
        "Qwen3/Qwen3-8B",
    ],
)
@pytest.mark.parametrize(
    "kv_cache_dtype",
    [
        "auto",
        "fp8",
    ],
)
def test_llm_update_weights_nvfp4(model_dir, kv_cache_dtype):
    model_dir = str(llm_models_root() / model_dir)
    num_hidden_layers = 1
    hf_model = RefNVFP4ModelWithIPCHandles(model_dir, num_hidden_layers=num_hidden_layers)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=True, free_gpu_memory_fraction=0.1, dtype=kv_cache_dtype
    )
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

    with llm:
        prompts_texts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]
        prompts = [tokenizer.encode(prompt) for prompt in prompts_texts]
        del tokenizer
        sampling_params = SamplingParams(
            temperature=0, return_generation_logits=True, max_tokens=1024
        )

        ipc_handles = hf_model.get_weight_ipc_handles_serialized([0, 1])
        llm._collective_rpc("update_weights", (ipc_handles,))
        llm._collective_rpc("update_weights", (None,))

        llm_logits, ref_logits = run_generate(llm, hf_model, prompts, sampling_params)
        # Use a looser threshold because NVFP4 logits are compared against a BF16 reference.
        compare_logits(llm_logits, ref_logits, threshold=0.8)


@pytest.mark.part3
@skip_pre_blackwell
@pytest.mark.parametrize(
    "model_dir",
    [
        "Qwen3/Qwen3-30B-A3B",
        "Qwen3/Qwen3-8B",
    ],
)
@pytest.mark.parametrize(
    "kv_cache_dtype",
    [
        "auto",
        "fp8",
    ],
)
def test_llm_partial_update_weights_nvfp4(model_dir, kv_cache_dtype):
    model_dir = str(llm_models_root() / model_dir)
    num_hidden_layers = 1
    hf_model = RefNVFP4ModelWithIPCHandles(model_dir, num_hidden_layers=num_hidden_layers)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=True, free_gpu_memory_fraction=0.1, dtype=kv_cache_dtype
    )
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

    with llm:
        prompts_texts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]
        prompts = [tokenizer.encode(prompt) for prompt in prompts_texts]
        del tokenizer

        sampling_params = SamplingParams(
            temperature=0, return_generation_logits=True, max_tokens=1024
        )

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

        llm_logits, ref_logits = run_generate(llm, hf_model, prompts, sampling_params)
        # Use a looser threshold because NVFP4 logits are compared against a BF16 reference.
        compare_logits(llm_logits, ref_logits, threshold=0.8)


@pytest.mark.part4
@skip_pre_hopper
def test_llm_partial_update_weights_nemotron_h_bf16_accuracy():
    """Partial update_weights + MMLU/GSM8K for the full Nemotron-H 30B BF16 model.

    End-to-end accuracy check on the RLHF update_weights path:
    1. Load HF on cuda:0 + replicate to cuda:1..3 (full 52 layers, no truncation).
    2. Bring up TRT-LLM with load_format="dummy".
    3. Apply update_weights once per weight-name suffix (mimics the streaming
       RLHF pattern); finalize once at the end.
    4. Free HF + replicas to recover memory before evaluation.
    5. Run GSM8K (exercises long decode) and MMLU (short-answer).

    Compared with `compare_logits`, this is robust to numerical reordering of
    low-probability tokens: only the argmax of each step matters for task-level
    accuracy. Needs 4x >=180GB GPUs (e.g., GB200).
    """
    model_dir = str(llm_models_root() /
                    "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    # Full model — read num_hidden_layers from the on-disk config so nothing is truncated.
    src_config = AutoConfig.from_pretrained(model_dir)
    num_hidden_layers = src_config.num_hidden_layers
    hf_model = RefHFModelWithIPCHandles(model_dir,
                                        num_hidden_layers=num_hidden_layers)
    # DEBUG: temporarily drop mamba_ssm_cache_dtype="float32" to verify whether
    # the default (bf16) SSM cache still passes accuracy.
    kv_cache_config = KvCacheConfig(enable_block_reuse=False,
                                    free_gpu_memory_fraction=0.25)
    moe_config = MoeConfig(backend="CUTLASS")
    llm = LLM(
        model=model_dir,
        ray_worker_extension_cls=
        "tensorrt_llm.llmapi.rlhf_utils.WorkerExtension",
        tensor_parallel_size=4,
        load_format="dummy",
        pipeline_parallel_size=1,
        kv_cache_config=kv_cache_config,
        moe_config=moe_config,
        max_batch_size=32,
    )

    with llm:
        # Group weights by trailing suffix (e.g. "input_layernorm.weight",
        # "mamba.A"); send one filtered batch per suffix.
        layer_prefix_pattern = re.compile(r"^model\.layers\.\d+\.")
        filter_set = set()
        for name, _ in hf_model.all_weights[hf_model.device_id]:
            filter_set.add(layer_prefix_pattern.sub("", name))
        filter_list = sorted(filter_set)

        def common_filter(filter_name: str) -> Callable[[str], bool]:
            def filter_fn(name: str) -> bool:
                return name.endswith(filter_name)
            return filter_fn

        for fname in filter_list:
            ipc_handles = hf_model.get_weight_ipc_handles_serialized(
                weight_filter=common_filter(fname))
            llm._collective_rpc("update_weights", (ipc_handles,))
        # Finalize once to trigger post_load_weights on all modules.
        llm._collective_rpc("update_weights", (None,))

        # Free HF + replicas to make room for evaluation activations.
        del hf_model
        gc.collect()
        torch.cuda.empty_cache()

        chat_template_kwargs = dict(enable_thinking=False)

        # GSM8K — long-form generation exercises the decode path.
        gsm8k = GSM8K(
            dataset_path=str(llm_models_root() / "datasets/openai/gsm8k"),
            chat_template_kwargs=chat_template_kwargs,
        )
        gsm8k_sp = SamplingParams(max_tokens=1024,
                                  truncate_prompt_tokens=4094,
                                  temperature=0)
        gsm8k_acc = gsm8k.evaluate(llm, sampling_params=gsm8k_sp)
        # Use the same hypothesis-testing threshold as the official
        # TestNemotronV3Nano accuracy test: ref=69.370, threshold=66.167
        # (alpha=0.05, beta=0.2, sigma=50, N=4096).
        assert gsm8k_acc >= 66.167, (
            f"GSM8K accuracy after partial update_weights = {gsm8k_acc:.2f} "
            f"is below threshold 66.167 (reference 69.370)")

        # MMLU — short-answer single-token classification.
        mmlu = MMLU(
            dataset_path=str(llm_models_root() / "datasets/mmlu"),
            random_seed=0,
            chat_template_kwargs=chat_template_kwargs,
        )
        mmlu_sp = SamplingParams(max_tokens=2,
                                 truncate_prompt_tokens=4094,
                                 temperature=0)
        mmlu_acc = mmlu.evaluate(llm, sampling_params=mmlu_sp)
        # Use the same hypothesis-testing threshold as the official
        # TestNemotronV3Nano accuracy test: ref=73.850, threshold=72.033
        # (alpha=0.05, beta=0.2, sigma=50, N=4096).
        assert mmlu_acc >= 72.033, (
            f"MMLU accuracy after partial update_weights = {mmlu_acc:.2f} "
            f"is below threshold 72.033 (reference 73.850)")


@pytest.mark.part4
@skip_pre_hopper
def test_llm_update_weights_nemotron_h_full():
    """Nemotron-H 30B partial update_weights test on a 7-layer slice.

    Nemotron-H is a hybrid model. With the on-disk hybrid_override_pattern
    "MEMEM*EMEMEM*...", the first 7 layers are M E M E M * E — covering all
    three layer flavors (3x Mamba, 3x MoE, 1x Attention) so the partial-update
    code path is exercised on every block type.

    Streams weights to the LLM one suffix-bucket at a time (mimics the RLHF
    streaming pattern), then finalizes once. Exercises the CUDA-graph decode
    path on Mamba layers: max_tokens=32 forces multiple decode steps so
    _A_expanded / _dt_bias_expanded / _D_expanded are actually read by the
    captured graph after update_weights.
    """
    model_dir = str(llm_models_root() /
                    "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    num_hidden_layers = 7
    hf_model = RefHFModelWithIPCHandles(model_dir,
                                        num_hidden_layers=num_hidden_layers)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    # Nemotron-H's Mamba state dominates the cache budget; 0.25 of free memory
    # leaves enough room for HF (resident on cuda:0 + replicas on cuda:1..3)
    # plus the TRT-LLM model shard. BF16 model -> CUTLASS MoE backend.
    # mamba_ssm_cache_dtype="float32" matches the official Nemotron-Nano
    # accuracy test (TestNemotronV3Nano::test_auto_dtype) — BF16 SSM cache
    # loses precision in long-decode selective_state_update.
    kv_cache_config = KvCacheConfig(enable_block_reuse=True,
                                    free_gpu_memory_fraction=0.25,
                                    mamba_ssm_cache_dtype="float32")
    moe_config = MoeConfig(backend="CUTLASS")
    with LLM(
            model=model_dir,
            ray_worker_extension_cls=
            "tensorrt_llm.llmapi.rlhf_utils.WorkerExtension",
            tensor_parallel_size=4,
            load_format="dummy",
            pipeline_parallel_size=1,
            kv_cache_config=kv_cache_config,
            moe_config=moe_config,
            max_batch_size=4,
            model_kwargs={"num_hidden_layers": num_hidden_layers},
    ) as llm:
        prompts_texts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]
        prompts = [tokenizer.encode(p) for p in prompts_texts]
        del tokenizer
        sampling_params = SamplingParams(temperature=0,
                                         return_generation_logits=True,
                                         max_tokens=32)

        # Warm the KV-cache prefix reuse store with the *dummy* weights before
        # update_weights. If update_weights' finalize path fails to invalidate
        # the prefix cache, the second generation would reuse cached KV blocks
        # produced by random weights and the logits comparison would fail —
        # a stricter check of enable_block_reuse=True correctness than just
        # running once after update_weights.
        llm.generate(prompts, sampling_params)

        # Group weights by trailing suffix (e.g. "input_layernorm.weight",
        # "mixer.A_log"); send one filtered batch per suffix.
        layer_prefix_pattern = re.compile(r"^model\.layers\.\d+\.")
        filter_set = set()
        for name, _ in hf_model.all_weights[hf_model.device_id]:
            filter_set.add(layer_prefix_pattern.sub("", name))
        filter_list = sorted(filter_set)

        def common_filter(filter_name: str) -> Callable[[str], bool]:
            def filter_fn(name: str) -> bool:
                return name.endswith(filter_name)
            return filter_fn

        for fname in filter_list:
            ipc_handles = hf_model.get_weight_ipc_handles_serialized(
                weight_filter=common_filter(fname))
            llm._collective_rpc("update_weights", (ipc_handles,))
        # Finalize once to trigger post_load_weights on all modules.
        llm._collective_rpc("update_weights", (None,))

        llm_logits, ref_logits = run_generate(llm, hf_model, prompts,
                                              sampling_params)
        compare_logits(llm_logits, ref_logits)
