import os

import pytest
import torch
from utils.llm_data import llm_models_root
from utils.util import force_ampere

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi.llm_utils import BuildConfig, KvCacheConfig

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
    gather_context_logits = gather_context_logits_fixture
    gather_generation_logits = gather_generation_logits_fixture
    sampler_type = sampler_type_fixture
    disable_overlap_scheduler = disable_overlap_scheduler_fixture

    build_config = BuildConfig()
    build_config.gather_context_logits = gather_context_logits

    llm = LLM(
        model=os.path.join(llm_models_root(), "llama-models-v2",
                           "TinyLlama-1.1B-Chat-v1.0"),
        kv_cache_config=global_kvcache_config,
        build_config=build_config,
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

        if gather_generation_logits:
            gen_logits = output.outputs[0].generation_logits
            assert gen_logits is not None
            assert gen_logits.ndim == 2
            assert gen_logits.shape[0] == sampling_params.max_tokens
            assert torch.argmax(gen_logits,
                                dim=1).tolist() == output.outputs[0].token_ids
        else:
            assert output.outputs[0].generation_logits is None

        if return_log_probs:
            assert len(output.outputs[0].logprobs) == sampling_params.max_tokens
        else:
            assert len(output.outputs[0].logprobs) == 0


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

        if gather_generation_logits:
            gen_logits = output.outputs[0].generation_logits
            assert gen_logits is not None
            assert gen_logits.ndim == 2
            assert gen_logits.shape[0] == 1
            try:
                assert torch.argmax(
                    gen_logits,
                    dim=1).tolist()[0] == output.outputs[0].token_ids[-1]
            except AssertionError:
                # FIXME: Remove xfail once the bug is fixed
                pytest.xfail("Known bug: https://nvbugs/5573238")
        else:
            assert output.outputs[0].generation_logits is None

        if return_log_probs:
            assert len(output.outputs[0].logprobs) == idx + 1
        else:
            assert len(output.outputs[0].logprobs) == 0
