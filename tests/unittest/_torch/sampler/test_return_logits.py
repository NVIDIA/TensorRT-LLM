import os

import pytest
import torch
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
    sampler_type_fixture: str,
    disable_overlap_scheduler_fixture: bool,
):
    sampler_type = sampler_type_fixture
    disable_overlap_scheduler = disable_overlap_scheduler_fixture

    llm = LLM(
        model=os.path.join(llm_models_root(), "llama-models-v2",
                           "TinyLlama-1.1B-Chat-v1.0"),
        kv_cache_config=global_kvcache_config,
        max_batch_size=
        128,  # reduce buffer sizes, specially for generation logits
        sampler_type=sampler_type,
        disable_overlap_scheduler=disable_overlap_scheduler,
    )

    # FIXME: Sometimes LLM shutdown hangs, might be related to https://nvbugs/5577178.
    #        Remove patch below once fixed.
    # NB: Xfails are resolved. A passing test should no longer hang on shutdown. Keep the patch for now to avoid regressions.
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


def check_generated_output(gather_context_logits, gather_generation_logits,
                           sampling_params, reuse_cache, return_log_probs, idx,
                           output, streaming):
    if gather_context_logits:
        assert output.context_logits is not None
        # NOTE: prompt_token_ids of "A B C" becomes [1, 319, 350, 315]
        expected_len = len(prompts[0].split()) + 1
        if reuse_cache:
            # Either all tokens are calculated (expected_len) or all tokens are reused (1)
            assert (output.context_logits.shape[0]
                    == expected_len) or (output.context_logits.shape[0] == 1)
            # "Known bug: https://nvbugs/5577178" Reuse may cause the context logits to be shorter than expected.
        else:
            assert expected_len == output.context_logits.shape[0]
    else:
        assert output.context_logits is None

    for sequence in output.outputs:
        if streaming:
            assert sequence.length == idx + 1
        else:
            assert sequence.length == sampling_params.max_tokens

        if gather_generation_logits:
            gen_logits = sequence.generation_logits
            assert gen_logits is not None
            assert gen_logits.ndim == 2
            if streaming:
                assert gen_logits.shape[0] == 1
                assert torch.argmax(gen_logits,
                                    dim=1).tolist()[0] == sequence.token_ids[-1]
            else:
                assert gen_logits.shape[0] == sampling_params.max_tokens
                assert torch.argmax(gen_logits,
                                    dim=1).tolist() == sequence.token_ids
        else:
            assert sequence.generation_logits is None
        if return_log_probs:
            assert len(sequence.logprobs) == sequence.length
        else:
            assert len(sequence.logprobs) == 0


@force_ampere  # Save H100 resource
@pytest.mark.parametrize("reuse_cache", [False, True])
@pytest.mark.parametrize("return_log_probs", [False, True])
@pytest.mark.parametrize("gather_generation_logits", [False, True])
@pytest.mark.parametrize("gather_context_logits", [False, True])
@pytest.mark.parametrize("async_generation", [False, True])
# FIXME: sometimes LLM shutdown hangs, might be related to https://nvbugs/5577178
# NB: Timeout covers fixtures https://github.com/pytest-dev/pytest-timeout/issues/134
@pytest.mark.timeout(120, method="signal")
@pytest.mark.threadleak(enabled=False)
def test_generation_with_return_logits(
    llm,
    gather_context_logits: bool,
    gather_generation_logits: bool,
    reuse_cache: bool,
    return_log_probs: bool,
    async_generation: bool,
):
    if not (gather_context_logits or gather_generation_logits
            or return_log_probs):  # prune space
        pytest.skip("Nothing to test")

    sampling_params = SamplingParams(
        max_tokens=8,
        return_context_logits=gather_context_logits,
        return_generation_logits=gather_generation_logits,
        logprobs=return_log_probs,
    )

    if async_generation:
        for idx, output in enumerate(
                llm.generate_async(
                    prompts[0],
                    sampling_params=sampling_params,
                    streaming=True,
                    cache_salt=CacheSalter.get_salt(reuse_cache),
                )):
            check_generated_output(
                gather_context_logits=gather_context_logits,
                gather_generation_logits=gather_generation_logits,
                sampling_params=sampling_params,
                reuse_cache=reuse_cache,
                return_log_probs=return_log_probs,
                idx=idx,
                output=output,
                streaming=True)
    else:
        for idx, output in enumerate(
                llm.generate(
                    prompts,
                    sampling_params=sampling_params,
                    cache_salt=[
                        CacheSalter.get_salt(reuse_cache) for _ in prompts
                    ],
                )):
            check_generated_output(
                gather_context_logits=gather_context_logits,
                gather_generation_logits=gather_generation_logits,
                sampling_params=sampling_params,
                reuse_cache=reuse_cache,
                return_log_probs=return_log_probs,
                idx=idx,
                output=output,
                streaming=False)
