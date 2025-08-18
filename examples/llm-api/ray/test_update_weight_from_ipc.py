from typing import List

import ray
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import KvCacheConfig, SamplingParams


class HFModel:

    def __init__(self, model_name: str):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16).to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.cuda_device = torch.cuda.current_device()
        # Set seed for reproducible random initialization (same as TrainingInstance)
        torch.manual_seed(32)
        self.all_weights = {}
        self.device_uuid = [
            HFModel.get_device_uuid(i) for i in range(torch.cuda.device_count())
        ]

    @staticmethod
    def get_device_uuid(cuda_device: int):
        from tensorrt_llm._torch.utils import get_device_uuid
        return get_device_uuid(cuda_device)

    def flip_weights(self):
        # Initialize all the parameters with random values (same as TrainingInstance)
        for _, p in self.model.named_parameters():
            p.data = -p.data

        self._replicate_weights()

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

    def get_weight_ipc_handles(self, cuda_device: int = None):
        from torch.multiprocessing.reductions import reduce_tensor

        ret = {}
        device_list = list(range(
            torch.cuda.device_count())) if cuda_device is None else [
                cuda_device
            ]
        for device in device_list:
            all_handles = []
            for item in self.all_weights[device]:
                name, p = item
                handle = reduce_tensor(p)
                all_handles.append((name, handle))
            ret[self.device_uuid[device]] = all_handles
        return ret

    def get_weights(self):
        return dict(self.all_weights[self.cuda_device])

    def generate(self, inputs: List[torch.Tensor], max_new_tokens: int = 50):
        generated_texts = []
        generated_token_ids = []
        for input_ids in inputs:
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)

            input_ids = input_ids.to(self.model.device)

            ret = self.model.generate(input_ids=input_ids,
                                      max_new_tokens=max_new_tokens,
                                      use_cache=True)

            new_tokens = ret[0][input_ids.shape[1]:]
            generated_token_ids.append(new_tokens)
            generated_texts.append(
                self.tokenizer.decode(new_tokens, skip_special_tokens=True))
        return generated_texts, generated_token_ids

    def generate_batch_incremental(self, original_prompts: List[str],
                                   generated_token_ids_list: List[List[int]]):
        """
        Generate tokens incrementally for each prompt in the batch: [prompt, prompt+token0, prompt+token0+token1, ...]
        """
        logits_list = []

        for i in range(len(original_prompts)):
            base_token_ids = self.tokenizer.encode(
                original_prompts[i], return_tensors="pt")[0].to("cuda")
            cur_logits = []
            for j in range(len(generated_token_ids_list[i])):
                if j > 0:
                    cur_gen_tokens = torch.tensor(
                        generated_token_ids_list[i][:j]).to("cuda")
                    cur_token_ids = torch.cat([base_token_ids, cur_gen_tokens],
                                              dim=-1)
                else:
                    cur_token_ids = base_token_ids

                ret = self.model.generate(
                    input_ids=cur_token_ids.unsqueeze(0).cuda(),
                    max_new_tokens=1,
                    return_dict_in_generate=True,
                    output_scores=True)

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


def compare_logits(logits_list: List[torch.Tensor],
                   ref_logits_list: List[torch.Tensor],
                   topk: int = 20,
                   threshold: float = 0.9):
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
        print(f"Prompt {i}: overlap ratio: {mean_ratio:.2%}")
        assert mean_ratio > threshold, f"Prompt {i}: overlap ratio: {mean_ratio:.2%} is less than {threshold:.2%}"


def run_generate(llm, hf_model, prompts, sampling_params):
    outputs = llm.generate(prompts, sampling_params)
    llm_logits = []
    llm_texts = []
    print("*" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        llm_logits.append(output.outputs[0].generation_logits)
        llm_texts.append(generated_text)
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("*" * 50)

    generated_token_ids_list = extract_tokens_from_outputs(outputs)
    ref_logits = hf_model.generate_batch_incremental(prompts,
                                                     generated_token_ids_list)
    return llm_texts, llm_logits, ref_logits


if __name__ == "__main__":
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tp_size = 1
    pp_size = 1

    kv_cache_config = KvCacheConfig(enable_block_reuse=False,
                                    free_gpu_memory_fraction=0.1)

    hf_model = HFModel(model_name)

    llm = LLM(model=model_name,
              tensor_parallel_size=tp_size,
              pipeline_parallel_size=pp_size,
              executor_type="ray",
              kv_cache_config=kv_cache_config)

    # Generate texts from the prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    sampling_params = SamplingParams(temperature=0,
                                     return_generation_logits=True)

    results = []

    print("-" * 20 + "Stage 1: Generate with original model" + "-" * 20)
    results.append(run_generate(llm, hf_model, prompts, sampling_params))
    llm_texts, llm_logits, ref_logits = results[0]
    compare_logits(llm_logits, ref_logits)

    print("-" * 20 + "Stage 2: Test update with flipped weights" + "-" * 20)
    hf_model.flip_weights()
    ipc_handles = hf_model.get_weight_ipc_handles()
    llm.collective_rpc("update_weights_from_ipc_handles", (ipc_handles, ))

    results.append(run_generate(llm, hf_model, prompts, sampling_params))
    llm_texts, llm_logits, ref_logits = results[1]

    # Compare the logits for this phase since output should be random
    compare_logits(llm_logits, ref_logits)

    print("-" * 20 +
          "Stage 3: Update with flipped weights with full tensor API" +
          "-" * 20)
    hf_model.flip_weights()
    llm.collective_rpc("update_weights", (hf_model.get_weights(), ))

    results.append(run_generate(llm, hf_model, prompts, sampling_params))
    llm_texts, llm_logits, ref_logits = results[2]

    # This stage, the weights should be the same as the original model, compare the logits and texts.
    compare_logits(llm_logits, ref_logits)
    for i in range(len(llm_texts)):
        assert llm_texts[i] == results[0][0][
            i], f"Stage 3 texts should be the same as stage 1, while {llm_texts[i]} != {results[0][i]}"

    del llm
    # TODO: should fix ray shutdown inside LLM
    ray.shutdown()
