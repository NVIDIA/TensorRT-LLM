import unittest
from typing import Optional

import pytest
import torch
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.speculative.drafter import Drafter
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, UserProvidedDecodingConfig


class PromptLookupDrafter(Drafter):
    """Minimal host-side prompt-lookup drafter used to exercise the user-provided path.

    Proposes the tokens that followed the latest earlier occurrence of the sequence's suffix
    (up to ``max_matching_ngram_size`` tokens), the way the retired host NGram drafter did.
    """

    # Uses TorchSampler, whose rewind is computed from the padded draft length.
    _needs_padding_kv_extension = True

    def __init__(self, max_draft_len: int, max_matching_ngram_size: int = 2) -> None:
        super().__init__(max_draft_len=max_draft_len, max_total_draft_tokens=max_draft_len)
        self.max_matching_ngram_size = max_matching_ngram_size

    def prepare_draft_tokens(
        self,
        scheduled_requests: ScheduledRequests,
        resource_manager: Optional[ResourceManager] = None,
    ) -> None:
        for request in scheduled_requests.generation_requests:
            request.py_draft_tokens = self._lookup(list(request.get_tokens(0)))

    def _lookup(self, tokens: list[int]) -> list[int]:
        for size in range(min(self.max_matching_ngram_size, len(tokens) - 1), 0, -1):
            pattern = tokens[-size:]
            for start in range(len(tokens) - size - 1, -1, -1):
                if tokens[start : start + size] == pattern:
                    return tokens[start + size : start + size + self.max_draft_len]
        return []


@pytest.mark.cpu_only
def test_prompt_lookup_drafter_prefers_longest_then_latest_match():
    drafter = PromptLookupDrafter(max_draft_len=3, max_matching_ngram_size=3)
    #          0  1  2  3  4  5  6  7  8  9
    tokens = [1, 2, 3, 7, 1, 2, 3, 8, 2, 3]
    assert drafter._lookup(tokens) == [8, 2, 3]  # suffix [2, 3], latest occurrence ends at 6
    assert drafter._lookup([1, 2, 3, 4, 1, 2, 3]) == [4, 1, 2]  # 3-gram match
    assert drafter._lookup([5]) == []
    assert drafter._lookup([5, 6]) == []


# TODO: add disable_overlap_scheduler=False
@pytest.mark.parametrize(
    "disable_overlap_scheduler,use_cuda_graph,attn_backend",
    [[True, False, "TRTLLM"], [True, True, "TRTLLM"], [True, False, "FLASHINFER"]],
)
def test_llama_user_provided(
    disable_overlap_scheduler: bool, use_cuda_graph: bool, attn_backend: str
):
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_mem_gb < 20:
        pytest.skip("Not enough memory to load target model")

    max_batch_size = 2
    max_draft_len = 4
    kv_cache_config = KvCacheConfig(enable_block_reuse=False, max_tokens=8192)
    cuda_graph_config = CudaGraphConfig(batch_sizes=[1]) if use_cuda_graph else None

    llm_common_config = dict(
        model=llm_models_root() / "llama-3.1-model" / "Meta-Llama-3.1-8B",
        backend="pytorch",
        attn_backend=attn_backend,
        disable_overlap_scheduler=disable_overlap_scheduler,
        cuda_graph_config=cuda_graph_config,
        max_batch_size=max_batch_size,
        kv_cache_config=kv_cache_config,
        max_num_tokens=2048,
    )

    drafter = PromptLookupDrafter(max_draft_len=max_draft_len, max_matching_ngram_size=2)

    spec_config = UserProvidedDecodingConfig(
        max_draft_len=max_draft_len,
        drafter=drafter,
    )

    prompts = [
        "The capital of France is",
        "The president of the United States is",
    ]
    sampling_params = SamplingParams(max_tokens=32)

    llm_spec = LLM(**llm_common_config, speculative_config=spec_config)
    results_spec = llm_spec.generate(prompts, sampling_params)
    generated_text_spec = [result.outputs[0].text for result in results_spec]
    llm_spec.shutdown()

    llm_ref = LLM(**llm_common_config)
    results_ref = llm_ref.generate(prompts, sampling_params)
    generated_text_ref = [result.outputs[0].text for result in results_ref]
    llm_ref.shutdown()

    for text_spec, text_ref in zip(generated_text_spec, generated_text_ref):
        # The spec decode algorithm currently guarantees identical results
        assert text_spec == text_ref


if __name__ == "__main__":
    unittest.main()
