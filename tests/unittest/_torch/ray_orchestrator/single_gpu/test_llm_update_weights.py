from typing import Callable, List, Optional

import pytest
import torch
from torch.multiprocessing.reductions import reduce_tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import KvCacheConfig, SamplingParams


class HFModel:
    def __init__(self, model_name: str):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        ).to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.cuda_device = torch.cuda.current_device()
        self.all_weights = {}
        self.device_uuid = [HFModel.get_device_uuid(i) for i in range(torch.cuda.device_count())]
        self._replicate_weights()

    @staticmethod
    def get_device_uuid(cuda_device: int):
        from tensorrt_llm._torch.utils import get_device_uuid

        return get_device_uuid(cuda_device)

    def _replicate_weights(self):
        model_weights = []
        for n, p in self.model.named_parameters():
            model_weights.append((n, p.detach().clone()))

        self.all_weights[self.cuda_device] = model_weights
        for i in range(torch.cuda.device_count()):
            if i != self.cuda_device:
                cur_weights = []
                for n, p in self.all_weights[self.cuda_device]:
                    cur_weights.append((n, p.to("cuda:" + str(i))))
                self.all_weights[i] = cur_weights

    def get_weight_ipc_handles(
        self,
        cuda_device: Optional[List[int]] = None,
        weight_filter: Optional[Callable[[str], bool]] = None,
    ):
        """
        Get IPC handles for model weights with flexible filtering.

        Args:
            cuda_device: List of CUDA device indices to get weights from
            weight_filter: Optional function that takes weight name and returns True if weight should be included

        Returns:
            ret: Dictionary containing weight handles
        """
        ret = {}
        device_list = list(range(torch.cuda.device_count())) if cuda_device is None else cuda_device

        for device in device_list:
            all_handles = []
            for item in self.all_weights[device]:
                name, p = item
                # Apply filter if provided
                if weight_filter is not None and not weight_filter(name):
                    continue
                handle = reduce_tensor(p)
                all_handles.append((name, handle))
            ret[self.device_uuid[device]] = all_handles

        return ret

    def generate_batch_incremental(
        self, original_prompts: List[str], generated_token_ids_list: List[List[int]]
    ):
        """
        Generate tokens incrementally for each prompt in the batch: [prompt, prompt+token0, prompt+token0+token1, ...]
        """
        logits_list = []

        for i in range(len(original_prompts)):
            base_token_ids = self.tokenizer.encode(original_prompts[i], return_tensors="pt")[0].to(
                "cuda"
            )
            cur_logits = []
            for j in range(len(generated_token_ids_list[i])):
                if j > 0:
                    cur_gen_tokens = torch.tensor(generated_token_ids_list[i][:j]).to("cuda")
                    cur_token_ids = torch.cat([base_token_ids, cur_gen_tokens], dim=-1)
                else:
                    cur_token_ids = base_token_ids

                ret = self.model.generate(
                    input_ids=cur_token_ids.unsqueeze(0).cuda(),
                    max_new_tokens=1,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

                cur_logits.append(ret["scores"][0])
            cur_logits = torch.stack(cur_logits, dim=0)
            logits_list.append(cur_logits.squeeze(1))

        return logits_list


def extract_tokens_from_outputs(outputs):
    """Extract individual tokens from LLM outputs using token IDs directly"""
    tokens_list = []
    for output in outputs:
        # Get token IDs directly from the output
        token_ids = output.outputs[0].token_ids
        tokens_list.append(token_ids)
    return tokens_list


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


def run_generate(llm, hf_model, prompts, sampling_params):
    outputs = llm.generate(prompts, sampling_params)
    llm_logits = []
    for output in outputs:
        llm_logits.append(output.outputs[0].generation_logits)

    generated_token_ids_list = extract_tokens_from_outputs(outputs)
    ref_logits = hf_model.generate_batch_incremental(prompts, generated_token_ids_list)
    return llm_logits, ref_logits


@pytest.mark.parametrize(
    "model_dir",
    ["Qwen2.5-0.5B-Instruct", "Qwen3/Qwen3-8B", "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"],
)
def test_llm_update_weights(model_dir):
    model_dir = str(llm_models_root() / model_dir)
    kv_cache_config = KvCacheConfig(enable_block_reuse=True, free_gpu_memory_fraction=0.1)

    hf_model = HFModel(model_dir)

    llm = LLM(
        model=model_dir,
        ray_worker_extension_cls="tensorrt_llm.llmapi.rlhf_utils.WorkerExtension",
        tensor_parallel_size=1,
        load_format="dummy",
        pipeline_parallel_size=1,
        kv_cache_config=kv_cache_config,
    )

    # Generate texts from the prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    sampling_params = SamplingParams(temperature=0, return_generation_logits=True)

    ipc_handles = hf_model.get_weight_ipc_handles([0])

    llm._collective_rpc("update_weights", (ipc_handles,))
    # Finalize the update weights
    llm._collective_rpc("update_weights", (None,))

    llm_logits, ref_logits = run_generate(llm, hf_model, prompts, sampling_params)
    compare_logits(llm_logits, ref_logits)


@pytest.mark.parametrize(
    "model_dir",
    ["Qwen2.5-0.5B-Instruct", "Qwen3/Qwen3-8B", "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"],
)
def test_llm_partial_update_weights(model_dir):
    model_dir = str(llm_models_root() / model_dir)
    kv_cache_config = KvCacheConfig(enable_block_reuse=True, free_gpu_memory_fraction=0.1)

    hf_model = HFModel(model_dir)

    llm = LLM(
        model=model_dir,
        ray_worker_extension_cls="tensorrt_llm.llmapi.rlhf_utils.WorkerExtension",
        tensor_parallel_size=1,
        load_format="dummy",
        pipeline_parallel_size=1,
        kv_cache_config=kv_cache_config,
    )

    # Generate texts from the prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    sampling_params = SamplingParams(temperature=0, return_generation_logits=True)

    ipc_handles = hf_model.get_weight_ipc_handles([0])

    def common_filter(filter_name: str) -> Callable[[str], bool]:
        def filter_fn(name: str) -> bool:
            return filter_name in name

        return filter_fn

    filter_list = [
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "o_proj.weight",
        "gate_proj.weight",
        "up_proj.weight",
        "down_proj.weight",
        "norm.weight",
        "embed_tokens.weight",
        "lm_head.weight",
    ]
    if "Qwen2.5" in model_dir or "Qwen2" in model_dir:
        filter_list.extend(
            [
                "q_proj.bias",
                "k_proj.bias",
                "v_proj.bias",
            ]
        )
    for filter_name in filter_list:
        weight_filter = common_filter(filter_name=filter_name)
        ipc_handles = hf_model.get_weight_ipc_handles([0], weight_filter=weight_filter)
        llm._collective_rpc("update_weights", (ipc_handles,), non_block=True)
    # Finalize the update weights
    llm._collective_rpc("update_weights", (None,))

    llm_logits, ref_logits = run_generate(llm, hf_model, prompts, sampling_params)
    compare_logits(llm_logits, ref_logits)
