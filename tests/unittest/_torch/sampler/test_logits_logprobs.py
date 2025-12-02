import os

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.llm_data import llm_models_root
from utils.util import force_ampere

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi.llm_utils import KvCacheConfig

prompts = ["A B C"]
global_kvcache_config = KvCacheConfig(
    max_tokens=10000,
    enable_block_reuse=True,
)


@pytest.fixture(scope="module", params=[False, True])
def gather_generation_logits_fixture(request) -> bool:
    return request.param


@pytest.fixture(scope="module", params=[False, True])
def gather_context_logits_fixture(request) -> bool:
    return request.param


@pytest.fixture(scope="module", params=[False, True])
def disable_overlap_scheduler_fixture(request) -> bool:
    return request.param


@pytest.fixture(scope="module", params=["TRTLLMSampler", "TorchSampler"])
def sampler_type_fixture(request) -> str:
    return request.param


class CacheSalter:

    _salt = 0

    @classmethod
    def get_salt_unique(cls) -> str:
        cls._salt += 1
        return str(cls._salt)

    @classmethod
    def get_salt_shared(cls) -> str:
        return str(0)

    @classmethod
    def get_salt(cls, reuse_cache: bool) -> str:
        if reuse_cache:
            salt = cls.get_salt_shared()
        else:
            salt = cls.get_salt_unique()
        return salt


@pytest.fixture(scope="module")
def llm(
    gather_context_logits_fixture: bool,
    gather_generation_logits_fixture: bool,
    sampler_type_fixture: str,
    disable_overlap_scheduler_fixture: bool,
):
    gather_generation_logits = gather_generation_logits_fixture
    sampler_type = sampler_type_fixture
    disable_overlap_scheduler = disable_overlap_scheduler_fixture

    llm = LLM(
        model=os.path.join(llm_models_root(), "llama-models-v2",
                           "TinyLlama-1.1B-Chat-v1.0"),
        kv_cache_config=global_kvcache_config,
        gather_generation_logits=gather_generation_logits,
        max_batch_size=
        128,  # reduce buffer sizes, specially for generation logits
        sampler_type=sampler_type,
        disable_overlap_scheduler=disable_overlap_scheduler,
    )

    # FIXME: Sometimes LLM shutdown hangs, might be related to https://nvbugs/5577178.
    #        Remove patch below once fixed.
    old_exit = LLM.__exit__

    def _exit_with_xfail_on_timeout(self, exc_type, exc_value,
                                    traceback) -> bool:
        import _pytest.outcomes
        try:
            return old_exit(self, exc_type, exc_value, traceback)
        except _pytest.outcomes.Failed as e:
            if e.msg and "pytest-timeout" in e.msg.lower():
                pytest.xfail(
                    "Known LLM shutdown issue (https://nvbugs/5577178).")
            else:
                raise

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(LLM, "__exit__", _exit_with_xfail_on_timeout)

        with llm:
            yield llm


@force_ampere  # Save H100 resource
@pytest.mark.parametrize("reuse_cache", [False, True])
@pytest.mark.parametrize("return_log_probs", [False, True])
# FIXME: sometimes LLM shutdown hangs, might be related to https://nvbugs/5577178
# NB: Timeout covers fixtures https://github.com/pytest-dev/pytest-timeout/issues/134
@pytest.mark.timeout(120, method="signal")
@pytest.mark.threadleak(enabled=False)
def test_generate_with_return_logits(
    llm,
    gather_context_logits_fixture: bool,
    gather_generation_logits_fixture: bool,
    reuse_cache: bool,
    return_log_probs: bool,
):
    gather_context_logits = gather_context_logits_fixture
    gather_generation_logits = gather_generation_logits_fixture

    if not (gather_context_logits or gather_generation_logits
            or return_log_probs):  # prune space
        pytest.skip("Nothing to test")

    sampling_params = SamplingParams(
        max_tokens=8,
        return_context_logits=gather_context_logits,
        return_generation_logits=gather_generation_logits,
        logprobs=return_log_probs,
    )

    for output in llm.generate(
            prompts,
            sampling_params=sampling_params,
            cache_salt=[CacheSalter.get_salt(reuse_cache) for _ in prompts],
    ):
        if gather_context_logits:
            assert output.context_logits is not None
            # NOTE: prompt_token_ids of "A B C" becomes [1, 319, 350, 315]
            expected_len = len(prompts[0].split()) + 1
            try:
                assert expected_len == output.context_logits.shape[0]
            except AssertionError:
                # FIXME: Remove this once the bug has been fixed
                if gather_context_logits and reuse_cache:
                    pytest.xfail("Known bug: https://nvbugs/5577178")
                raise
        else:
            assert output.context_logits is None

        for sequence in output.outputs:
            assert sequence.length == sampling_params.max_tokens

            if gather_generation_logits:
                gen_logits = sequence.generation_logits
                assert gen_logits is not None
                assert gen_logits.ndim == 2
                assert gen_logits.shape[0] == sampling_params.max_tokens
                assert torch.argmax(gen_logits,
                                    dim=1).tolist() == sequence.token_ids
            else:
                assert sequence.generation_logits is None

            if return_log_probs:
                assert len(sequence.logprobs) == sampling_params.max_tokens
            else:
                assert len(sequence.logprobs) == 0


@force_ampere  # Save H100 resource
@pytest.mark.parametrize("reuse_cache", [False, True])
@pytest.mark.parametrize("return_log_probs", [False, True])
# FIXME: sometimes LLM shutdown hangs, might be related to https://nvbugs/5577178
# NB: Timeout covers fixtures https://github.com/pytest-dev/pytest-timeout/issues/134
@pytest.mark.timeout(120, method="signal")
@pytest.mark.threadleak(enabled=False)
def test_generate_async_with_return_logits(
    llm,
    gather_context_logits_fixture: bool,
    gather_generation_logits_fixture: bool,
    reuse_cache: bool,
    return_log_probs: bool,
):
    gather_context_logits = gather_context_logits_fixture
    gather_generation_logits = gather_generation_logits_fixture

    if not (gather_context_logits or gather_generation_logits
            or return_log_probs):  # prune space
        pytest.skip("Nothing to test")

    sampling_params = SamplingParams(
        max_tokens=8,
        return_context_logits=gather_context_logits,
        return_generation_logits=gather_generation_logits,
        logprobs=return_log_probs)

    for idx, output in enumerate(
            llm.generate_async(
                prompts[0],
                sampling_params=sampling_params,
                streaming=True,
                cache_salt=CacheSalter.get_salt(reuse_cache),
            )):
        if gather_context_logits:
            assert output.context_logits is not None
            # NOTE: prompt_token_ids of "A B C" becomes [1, 319, 350, 315]
            expected_len = len(prompts[0].split()) + 1
            try:
                assert expected_len == output.context_logits.shape[0]
            except AssertionError:
                # FIXME: Remove this once the bug has been fixed
                if gather_context_logits and reuse_cache:
                    pytest.xfail("Known bug: https://nvbugs/5577178")
                raise
        else:
            assert output.context_logits is None

        for sequence in output.outputs:
            assert sequence.length == idx + 1

            if gather_generation_logits:
                gen_logits = sequence.generation_logits
                assert gen_logits is not None
                assert gen_logits.ndim == 2
                assert gen_logits.shape[0] == 1
                try:
                    assert torch.argmax(
                        gen_logits, dim=1).tolist()[0] == sequence.token_ids[-1]
                except AssertionError:
                    # FIXME: Remove xfail once the bug is fixed
                    pytest.xfail("Known bug: https://nvbugs/5573238")
            else:
                assert sequence.generation_logits is None

            if return_log_probs:
                assert len(sequence.logprobs) == idx + 1
            else:
                assert len(sequence.logprobs) == 0


@pytest.mark.parametrize("logprobs_k", [0, 1, 3],
                         ids=["top_0", "top_1", "top_3"])
def test_sampled_token_always_in_logprobs(logprobs_k: int):
    """Two scenarios:
        - logprobs=0: Returns only sampled token (1 element)
        - logprobs=K (K>0): Returns top-K tokens + sampled token if not in top-K (up to K+1 elements)
    """
    llm = LLM(model=os.path.join(llm_models_root(), "llama-models-v2",
                                 "TinyLlama-1.1B-Chat-v1.0"), )

    sampling_params = SamplingParams(
        max_tokens=8,
        temperature=0.7,
        top_p=0.9,
        logprobs=logprobs_k,
    )

    for output in llm.generate(["The future of AI is"],
                               sampling_params=sampling_params):
        print(f"\n{'='*80}")
        print(f"Generated text: {output.outputs[0].text!r}")
        print(f"Generated token IDs: {output.outputs[0].token_ids}")

        logprobs = output.outputs[0].logprobs
        token_ids = output.outputs[0].token_ids

        assert len(logprobs) == sampling_params.max_tokens, \
            f"Expected {sampling_params.max_tokens} logprob entries, got {len(logprobs)}"

        for token_idx, (sampled_token_id,
                        token_logprobs) in enumerate(zip(token_ids, logprobs)):
            print(
                f"\n  Token {token_idx}: ID={sampled_token_id}, Text={llm.tokenizer.decode([sampled_token_id])!r}"
            )

            assert sampled_token_id in token_logprobs, \
                f"Token {token_idx}: Sampled token ID {sampled_token_id} not in logprobs dict: {token_logprobs.keys()}"

            if logprobs_k == 0:
                assert len(token_logprobs) == 1, \
                    f"Token {token_idx}: Expected 1 logprob (sampled only), got {len(token_logprobs)}"
            else:
                assert len(token_logprobs) <= logprobs_k + 1, \
                    f"Token {token_idx}: Expected at most {logprobs_k + 1} logprobs, got {len(token_logprobs)}"
                assert len(token_logprobs) >= 1

            sorted_tokens_by_prob = sorted(token_logprobs.items(),
                                           key=lambda x: x[1].logprob,
                                           reverse=True)

            if logprobs_k > 0:
                sampled_token_rank = token_logprobs[sampled_token_id].rank
                sampled_in_topk = sampled_token_rank <= logprobs_k

                if not sampled_in_topk:
                    assert sorted_tokens_by_prob[-1][0] == sampled_token_id, \
                        f"Token {token_idx}: Sampled token (ID={sampled_token_id}, rank={sampled_token_rank}) not in top-{logprobs_k}, " \
                        f"should be last in sorted list, but last token is ID={sorted_tokens_by_prob[-1][0]}"

            for rank_idx, (token_id,
                           logprob_obj) in enumerate(sorted_tokens_by_prob,
                                                     start=1):
                token_text = llm.tokenizer.decode([token_id])
                is_sampled = "← SAMPLED" if token_id == sampled_token_id else ""
                print(f"    • Token {token_id:5d} ({token_text:15s}): "
                      f"logprob={logprob_obj.logprob:8.4f}, "
                      f"rank={logprob_obj.rank} {is_sampled}")

                if logprobs_k > 0 and sampled_in_topk:
                    assert logprob_obj.rank == rank_idx, \
                        f"Token {token_idx}: Token {token_id} rank mismatch. " \
                        f"Expected rank {rank_idx} (by sorted position), got {logprob_obj.rank}"

        print(f"{'='*80}\n")


@pytest.mark.parametrize("logprobs_k", [0, 2], ids=["top_0", "top_2"])
def test_logprobs_with_grouped_samplings_strategies(logprobs_k: int):
    """Test logprobs when requests are reordered by sampling strategy grouping"""
    llm = LLM(
        model=os.path.join(llm_models_root(), "llama-models-v2",
                           "TinyLlama-1.1B-Chat-v1.0"),
        max_batch_size=8,
    )

    test_prompts = [
        "The capital of France is",
        "The future of AI is",
        "Hello, my name is",
        "Write a short story about a cat",
    ]

    # Causes reordering: [0,1,2,3] → [0,2,1,3]
    sampling_params_list = [
        SamplingParams(max_tokens=6,
                       temperature=0.8,
                       top_k=50,
                       logprobs=logprobs_k,
                       return_generation_logits=True),
        SamplingParams(max_tokens=6,
                       temperature=0.8,
                       top_p=0.9,
                       logprobs=logprobs_k,
                       return_generation_logits=True),
        SamplingParams(max_tokens=6,
                       temperature=0.8,
                       top_k=50,
                       logprobs=logprobs_k,
                       return_generation_logits=True),
        SamplingParams(max_tokens=6,
                       temperature=0.8,
                       top_p=0.9,
                       logprobs=logprobs_k,
                       return_generation_logits=True),
    ]

    outputs = list(
        llm.generate(test_prompts, sampling_params=sampling_params_list))

    for req_idx, output in enumerate(outputs):
        generation_logits = output.outputs[0].generation_logits
        token_ids = output.outputs[0].token_ids
        logprobs = output.outputs[0].logprobs

        assert generation_logits is not None
        assert len(logprobs) == len(token_ids), "Logprobs length mismatch"

        # generation_logits might be shorter than token_ids
        num_logits = len(generation_logits)

        for token_idx, (sampled_token_id, token_logprobs_dict) in enumerate(
                zip(token_ids[:num_logits], logprobs[:num_logits])):
            returned_logprob = token_logprobs_dict[sampled_token_id].logprob

            logits_for_token = generation_logits[token_idx]
            expected_logprobs = torch.nn.functional.log_softmax(
                logits_for_token.to(dtype=torch.float32), dim=-1)
            expected_logprob = expected_logprobs[sampled_token_id].item()

            logprob_diff = abs(returned_logprob - expected_logprob)
            print(
                f"Req {req_idx}, Token {token_idx}: returned={returned_logprob:.6f}, expected={expected_logprob:.6f}, diff={logprob_diff:.6f}"
            )

            assert logprob_diff < 1e-4, \
                f"Req {req_idx}, Token {token_idx}: Logprob mismatch! " \
                f"Returned {returned_logprob:.6f} but expected {expected_logprob:.6f} " \
                f"(diff={logprob_diff:.6f}). This indicates the logprob might be extracted from " \
                f"the wrong token position."

@force_ampere
@pytest.mark.gpu2
def test_logprobs_match_hf_tp2():
    model_path = os.path.join(llm_models_root(), "llama-models-v2",
                              "TinyLlama-1.1B-Chat-v1.0")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=2,
    )

    prompts = ["The future of the AI is"]

    sampling_params = SamplingParams(
        max_tokens=10,
        temperature=1.0,
        logprobs=0,
    )

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16).to("cuda")
    hf_tokenizer = AutoTokenizer.from_pretrained(model_path)

    output = list(llm.generate(prompts, sampling_params=sampling_params))[0]

    trtllm_token_ids = output.outputs[0].token_ids
    trtllm_logprobs = torch.tensor(
        [list(lp.values())[0].logprob for lp in output.outputs[0].logprobs])

    base_ids = hf_tokenizer.encode(prompts[0], return_tensors="pt").to("cuda")
    hf_logprobs = []

    for i, token_id in enumerate(trtllm_token_ids):
        if i > 0:
            prev_tokens = torch.tensor([trtllm_token_ids[:i]], device="cuda")
            input_ids = torch.cat([base_ids, prev_tokens], dim=1)
        else:
            input_ids = base_ids
        with torch.no_grad():
            logits = hf_model(input_ids).logits[0, -1, :]
        hf_logprobs.append(torch.log_softmax(logits, dim=-1)[token_id].item())

    hf_logprobs = torch.tensor(hf_logprobs)

    print(f"\nTensorRT-LLM logprobs: {trtllm_logprobs}")
    print(f"HuggingFace logprobs:  {hf_logprobs}")
    print(f"Diff: {(trtllm_logprobs - hf_logprobs).abs()}")

    max_diff = (trtllm_logprobs - hf_logprobs).abs().max().item()
    assert max_diff < 0.15, f"Max logprob diff {max_diff:.4f} exceeds threshold"
