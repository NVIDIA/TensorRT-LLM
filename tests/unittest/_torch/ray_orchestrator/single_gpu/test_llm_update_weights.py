import base64
import gc
import pickle
import re
from typing import Callable, List, Optional, Tuple

import pytest
import torch
from torch.multiprocessing.reductions import reduce_tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.llm_data import llm_models_root
from utils.torch_ref import RefHFModel
from utils.util import getSMVersion, skip_pre_hopper

from tensorrt_llm import LLM
from tensorrt_llm._torch.utils import get_device_uuid
from tensorrt_llm.llmapi import KvCacheConfig, MoeConfig, SamplingParams

# Ray-backed LLM teardown spawns the executor main-loop, GC and log/error
# listener threads in ray-core. These are torn down only when ``ray.shutdown()``
# fires, which only runs from RayExecutor.shutdown() — itself only called when
# the LLM object is explicitly closed. The transformers 5.5.x import graph
# enlarges the live reference set, delaying GC of the test-local LLM past
# pytest-threadleak's post-teardown snapshot. Disable the leak check for this
# file (matches the sibling pattern at
# tests/unittest/_torch/ray_orchestrator/multi_gpu/test_executor.py).
pytestmark = pytest.mark.threadleak(enabled=False)


@pytest.fixture(autouse=True)
def release_shared_cuda_memory():
    """Reclaim producer-side CUDA IPC memory between parametrize IDs."""
    yield
    # Break reference cycles so the test-local hf_model actually dies now.
    gc.collect()
    # Free sent IPC storages whose consumers have already closed them.
    torch.cuda.ipc_collect()
    # Return freed cached segments to the driver so other processes
    # (the next test's Ray workers) can allocate them.
    torch.cuda.empty_cache()


class RefHFModelWithIPCHandles(RefHFModel):
    def __init__(self, model_dir: str, device_id: int = 0, **model_kwargs):
        self.device_id = device_id
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            **model_kwargs,
        ).to(f"cuda:{device_id}")
        # Hybrid configs (e.g. NemotronH) derive num_hidden_layers from
        # ``layers_block_type`` and silently ignore the num_hidden_layers
        # override; callers must pass a truncated ``layers_block_type`` for
        # such models. Catch a silently ignored override loudly here.
        num_hidden_layers = model_kwargs.get("num_hidden_layers")
        if num_hidden_layers is not None:
            assert self.model.config.num_hidden_layers == num_hidden_layers
        self.all_weights = {}
        self.device_uuid = [get_device_uuid(i) for i in range(torch.cuda.device_count())]
        self._replicate_weights()

    def _replicate_weights(self):
        model_weights = []
        for n, p in self.model.named_parameters():
            model_weights.append((n, p.detach().clone()))

        # Only populate the owning device. Extra replicas are materialized
        # lazily by ``get_weight_ipc_handles_serialized`` so that GPUs never
        # asked for via IPC (e.g. cuda:2/3 on a 4-GPU runner when a TP=2
        # test only requests device_ids=[0, 1]) don't hold weight copies
        # that persist across parametrize IDs.
        self.all_weights[self.device_id] = model_weights

    def get_weight_ipc_handles_serialized(
        self,
        device_ids: Optional[List[int]] = None,
        weight_filter: Optional[Callable[[str], bool]] = None,
    ):
        """
        Get base64-encoded serialized IPC handles for model weights.

        Args:
            device_ids: List of CUDA device indices to get weights from
            weight_filter: Optional function that takes weight name and returns True if weight should be included

        Returns:
            ret: Dictionary mapping device UUIDs to base64-encoded pickled handles
        """
        ret = {}
        device_list = list(range(torch.cuda.device_count())) if device_ids is None else device_ids

        for device in device_list:
            if device not in self.all_weights:
                src = self.all_weights[self.device_id]
                self.all_weights[device] = [(n, p.to(f"cuda:{device}")) for n, p in src]

            all_handles = []
            for item in self.all_weights[device]:
                name, p = item
                # Apply filter if provided
                if weight_filter is not None and not weight_filter(name):
                    continue
                handle = reduce_tensor(p)
                all_handles.append((name, handle))

            # Serialize with base64-encoded pickle
            serialized = base64.b64encode(pickle.dumps(all_handles)).decode("utf-8")
            ret[self.device_uuid[device]] = serialized

        return ret


def compare_logits(
    logits_list: List[torch.Tensor],
    ref_logits_list: List[torch.Tensor],
    topk: int = 20,
    threshold: float = 0.9,
):
    assert len(logits_list) == len(ref_logits_list)
    for i in range(len(logits_list)):
        assert logits_list[i].shape == ref_logits_list[i].shape
        lhs_idx = torch.topk(logits_list[i], topk, dim=-1).indices
        rhs_idx = torch.topk(ref_logits_list[i], topk, dim=-1).indices
        # Token wise comparison
        ratios = []
        for j in range(lhs_idx.shape[0]):
            lhs_idx_j = lhs_idx[j].tolist()
            rhs_idx_j = rhs_idx[j].tolist()
            overlap = set(lhs_idx_j) & set(rhs_idx_j)
            ratios.append(len(overlap) / len(lhs_idx_j))

        mean_ratio = sum(ratios) / len(ratios)
        assert mean_ratio > threshold, (
            f"Prompt {i}: overlap ratio: {mean_ratio:.2%} is less than {threshold:.2%}"
        )


def run_generate(
    llm: LLM,
    hf_model: RefHFModel,
    prompts: List[List[int]],
    sampling_params: SamplingParams,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    llm_responses = []
    llm_logits = []
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        llm_logits.append(output.outputs[0].generation_logits)
        llm_responses.append(output.outputs[0].token_ids)
    input_ids, attention_mask, position_ids = RefHFModel.pad_data(prompts, llm_responses)
    ref_logits = hf_model.generate_batch_with_padding(
        input_ids, attention_mask, position_ids, llm_responses, return_logits=True
    )
    return llm_logits, ref_logits


@skip_pre_hopper
@pytest.mark.parametrize(
    "model_dir",
    [
        "llama-models-v2/TinyLlama-1.1B-Chat-v1.0",
        "Qwen2.5-0.5B-Instruct",
        "Qwen3/Qwen3-8B",
        "Qwen3/Qwen3-30B-A3B",
        "Qwen3/Qwen3-8B-FP8",
        "Qwen3/Qwen3-30B-A3B-FP8",
    ],
)
@pytest.mark.part0
def test_llm_update_weights(model_dir):
    model_dir = str(llm_models_root() / model_dir)
    num_hidden_layers = 1
    hf_model = RefHFModelWithIPCHandles(model_dir, num_hidden_layers=num_hidden_layers)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    kv_cache_config = KvCacheConfig(enable_block_reuse=True, free_gpu_memory_fraction=0.1)
    moe_config = MoeConfig(backend="DEEPGEMM" if getSMVersion() >= 100 else "CUTLASS")
    with LLM(
        model=model_dir,
        ray_worker_extension_cls="tensorrt_llm.llmapi.rlhf_utils.WorkerExtension",
        tensor_parallel_size=1,
        load_format="dummy",
        pipeline_parallel_size=1,
        kv_cache_config=kv_cache_config,
        model_kwargs={"num_hidden_layers": num_hidden_layers},
        moe_config=moe_config,
    ) as llm:
        # Generate texts from the prompts.
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

        ipc_handles = hf_model.get_weight_ipc_handles_serialized([0])

        llm._collective_rpc("update_weights", (ipc_handles,))
        # Finalize the update weights
        llm._collective_rpc("update_weights", (None,))

        llm_logits, ref_logits = run_generate(llm, hf_model, prompts, sampling_params)
        compare_logits(llm_logits, ref_logits)

    del hf_model


@skip_pre_hopper
@pytest.mark.parametrize(
    "model_dir",
    [
        "llama-models-v2/TinyLlama-1.1B-Chat-v1.0",
        "Qwen2.5-0.5B-Instruct",
        "Qwen3/Qwen3-8B",
        "Qwen3/Qwen3-30B-A3B",
        "Qwen3/Qwen3-8B-FP8",
        "Qwen3/Qwen3-30B-A3B-FP8",
    ],
)
@pytest.mark.part1
def test_llm_partial_update_weights(model_dir):
    model_dir = str(llm_models_root() / model_dir)
    num_hidden_layers = 1
    hf_model = RefHFModelWithIPCHandles(model_dir, num_hidden_layers=num_hidden_layers)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    kv_cache_config = KvCacheConfig(enable_block_reuse=True, free_gpu_memory_fraction=0.1)
    moe_config = MoeConfig(backend="DEEPGEMM" if getSMVersion() >= 100 else "CUTLASS")
    with LLM(
        model=model_dir,
        ray_worker_extension_cls="tensorrt_llm.llmapi.rlhf_utils.WorkerExtension",
        tensor_parallel_size=1,
        load_format="dummy",
        pipeline_parallel_size=1,
        kv_cache_config=kv_cache_config,
        model_kwargs={"num_hidden_layers": num_hidden_layers},
        moe_config=moe_config,
    ) as llm:
        # Generate texts from the prompts.
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
                [0], weight_filter=weight_filter
            )
            llm._collective_rpc("update_weights", (ipc_handles,))
        # Finalize the update weights
        llm._collective_rpc("update_weights", (None,))

        llm_logits, ref_logits = run_generate(llm, hf_model, prompts, sampling_params)
        compare_logits(llm_logits, ref_logits)

    del hf_model


@skip_pre_hopper
@pytest.mark.parametrize(
    "model_dir, fp8_model_dir",
    [
        ("Qwen3/Qwen3-8B", "Qwen3/Qwen3-8B-FP8"),
        ("Qwen3/Qwen3-30B-A3B", "Qwen3/Qwen3-30B-A3B-FP8"),
    ],
)
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.part2
def test_llm_update_weights_with_quant_config(model_dir, fp8_model_dir, kv_cache_dtype):
    model_dir = str(llm_models_root() / model_dir)
    fp8_model_dir = str(llm_models_root() / fp8_model_dir)
    num_hidden_layers = 1
    hf_model = RefHFModelWithIPCHandles(fp8_model_dir, num_hidden_layers=num_hidden_layers)
    tokenizer = AutoTokenizer.from_pretrained(fp8_model_dir)
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=True, free_gpu_memory_fraction=0.1, dtype=kv_cache_dtype
    )
    moe_config = MoeConfig(backend="DEEPGEMM" if getSMVersion() >= 100 else "CUTLASS")
    with LLM(
        model=model_dir,
        ray_worker_extension_cls="tensorrt_llm.llmapi.rlhf_utils.WorkerExtension",
        tensor_parallel_size=1,
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
        moe_config=moe_config,
    ) as llm:
        # Generate texts from the prompts.
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

        ipc_handles = hf_model.get_weight_ipc_handles_serialized([0])

        llm._collective_rpc("update_weights", (ipc_handles,))
        # Finalize the update weights
        llm._collective_rpc("update_weights", (None,))

        llm_logits, ref_logits = run_generate(llm, hf_model, prompts, sampling_params)
        compare_logits(llm_logits, ref_logits)

    del hf_model
