# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pathlib

import pytest
import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.configuration_utils import PretrainedConfig
from utils.util import force_ampere

from tensorrt_llm import LLM
from tensorrt_llm._torch.model_config import ModelConfig as _ModelConfig
from tensorrt_llm._torch.models.checkpoints import \
    HfCheckpointLoader as _HfCheckpointLoader
from tensorrt_llm._torch.models.checkpoints.base_config_loader import \
    BaseConfigLoader as _BaseConfigLoader
from tensorrt_llm._torch.models.checkpoints.base_weight_loader import \
    BaseWeightLoader as _BaseWeightLoader
from tensorrt_llm._torch.models.modeling_utils import \
    register_auto_model as _register_auto_model
from tensorrt_llm._torch.models.modeling_utils import \
    register_checkpoint_weight_loader as _register_checkpoint_weight_loader
from tensorrt_llm._torch.models.modeling_utils import \
    register_config_loader as _register_config_loader
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.llmapi.llm_args import NGramDecodingConfig
from tensorrt_llm.sampling_params import SamplingParams

from .llm_test_utils import (global_kvcache_config, llm_test_harness, prompts,
                             run_llm_abort_request,
                             sampling_params_for_aborting_request,
                             tinyllama_logits_processor_test_harness)
from .model_paths import llama_model_path


@force_ampere
@pytest.mark.parametrize("enable_chunked_prefill", [False, True])
def test_tinyllama_logits_processor(enable_chunked_prefill):
    tinyllama_logits_processor_test_harness(
        backend="pytorch", enable_chunked_prefill=enable_chunked_prefill)


@force_ampere
@pytest.mark.mpi_ray_parity
@pytest.mark.parametrize("sampling_params",
                         sampling_params_for_aborting_request)
def test_llm_abort_request(sampling_params):
    llm_kwargs = {}
    if sampling_params.use_beam_search:
        if sampling_params.best_of is None:
            llm_kwargs["max_beam_width"] = sampling_params.n
        else:
            llm_kwargs["max_beam_width"] = sampling_params.best_of
    llm = LLM(model=llama_model_path,
              kv_cache_config=global_kvcache_config,
              **llm_kwargs)
    run_llm_abort_request(llm=llm, sampling_params=sampling_params)


def test_embedding_bias_with_torch_sampler_strategies():
    """Test embedding bias application in TorchSampler."""
    tokenizer = AutoTokenizer.from_pretrained(llama_model_path)
    biased_word_id = tokenizer.encode("Z", add_special_tokens=False)[-1]
    vocab_size_padded = AutoConfig.from_pretrained(llama_model_path).vocab_size
    embedding_bias = torch.zeros(vocab_size_padded)
    embedding_bias[biased_word_id] = torch.finfo(torch.float32).max

    max_tokens = 6
    sampling_kwargs = {
        "max_tokens": max_tokens,
        "embedding_bias": embedding_bias,
    }

    # All test cases use greedy sampling for simplicity

    sampling_params = SamplingParams(**sampling_kwargs)

    # The biased token's decoded text (e.g. with or without a leading space)
    # is tokenizer-specific, so derive the expected repeated-token output
    # rather than hardcoding a particular tokenizer's rendering.
    expected_output = tokenizer.decode([biased_word_id] * max_tokens)

    llm_test_harness(
        llama_model_path,
        prompts,
        [expected_output],
        sampling_params=sampling_params,
        backend="pytorch",
    )


@pytest.mark.parametrize("use_speculative", [True, False])
def test_min_tokens(use_speculative: bool):
    """Check min_tokens is respected."""
    llm_common_config = dict(
        model=llama_model_path,
        max_batch_size=2,
        kv_cache_config=global_kvcache_config,
        max_num_tokens=2048,
    )

    if use_speculative:
        spec_config = NGramDecodingConfig(
            max_draft_len=4,
            max_matching_ngram_size=2,
            is_keep_all=True,
            is_use_oldest=True,
            is_public_pool=True,
        )
        llm = LLM(**llm_common_config, speculative_config=spec_config)
    else:
        llm = LLM(**llm_common_config)

    output_len = 2000
    sampling_params = SamplingParams(max_tokens=output_len,
                                     min_tokens=output_len,
                                     temperature=1)
    res = llm.generate("The end.", sampling_params=sampling_params)

    assert len(res.outputs) == 1
    assert len(res.outputs[0].token_ids) == output_len


def test_min_tokens_long_prompt():
    """Check min_tokens is respected when prompt is longer than min_tokens.

    Regression test for NVBug 5823135: the min-length EOS suppression compared
    total token count (prompt + generated) against the raw min_tokens value
    instead of comparing generated token count only.  When prompt_len >=
    min_tokens the EOS suppression was never activated, allowing early
    termination.
    """
    min_tok = 50
    max_tok = 100
    # Prompt long enough so that prompt_len > min_tok.  "Hello " tokenises
    # to ~1-2 tokens with most tokenizers, so 200 repetitions ≈ 200-400
    # tokens >> min_tok.
    long_prompt = "Hello " * 200

    llm = LLM(
        model=llama_model_path,
        max_batch_size=2,
        kv_cache_config=global_kvcache_config,
        max_num_tokens=2048,
    )

    sampling_params = SamplingParams(
        max_tokens=max_tok,
        min_tokens=min_tok,
        temperature=1,
    )
    res = llm.generate(long_prompt, sampling_params=sampling_params)

    assert len(res.outputs) == 1
    generated_len = len(res.outputs[0].token_ids)
    assert generated_len >= min_tok, (
        f"Generated only {generated_len} tokens with min_tokens={min_tok} "
        f"and a long prompt.  Bug 5823135 regression.")


_LAST_VOCAB_TOKEN_ID = 255

_LAST_VOCAB_EOS_TOKEN_ID = 2


class _LastVocabConfig(PretrainedConfig):

    def __init__(self):
        self.architectures = ["_LastVocabModel"]
        self.torch_dtype = torch.float16
        self.num_key_value_heads = 4
        self.num_attention_heads = 4
        self.hidden_size = 64
        self.vocab_size = 256
        self.num_hidden_layers = 1
        self.eos_token_id = _LAST_VOCAB_EOS_TOKEN_ID

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads


@_register_auto_model("_LastVocabModel")
class _LastVocabModel(torch.nn.Module):
    """Dummy model whose logits always favour the last vocab token (id 255).

    The original bug made logits[..., -1] suppress this token via Python
    negative indexing when ignore_eos + min_tokens were both set.
    """

    def __init__(self, model_config: _ModelConfig):
        super().__init__()
        self.model_config = model_config

    def infer_max_seq_len(self):
        return 2048

    @property
    def config(self):
        return self.model_config.pretrained_config

    def forward(self, *, input_ids, attn_metadata, **kwargs):
        num_batch_tokens = input_ids.size(0)
        last_tokens = torch.cumsum(
            attn_metadata.seq_lens_cuda, dim=0, dtype=torch.long) - 1

        logits = torch.zeros((num_batch_tokens, self.config.vocab_size),
                             device="cuda",
                             dtype=self.config.torch_dtype)
        logits[:, _LAST_VOCAB_TOKEN_ID] = 100.0

        if not kwargs.get("return_context_logits", False):
            logits = logits[last_tokens]
        return {"logits": logits}

    def load_weights(self, weights, weight_mapper=None, skip_modules=None):
        pass


@_register_checkpoint_weight_loader("_LAST_VOCAB_FMT")
class _LastVocabWeightLoader(_BaseWeightLoader):

    def load_weights(self, checkpoint_dir, **kwargs):
        return {}


@_register_config_loader("_LAST_VOCAB_FMT")
class _LastVocabConfigLoader(_BaseConfigLoader):

    def load(self, checkpoint_dir, **kwargs):
        return _ModelConfig(pretrained_config=_LastVocabConfig())


def test_min_tokens_with_ignore_eos():
    """Check ignore_eos + min_tokens does not suppress the last vocab token.

    The original bug: ignore_eos sets end_id to -1. The min_length penalty
    used that value, so logits[..., -1] suppressed the last vocab token
    (id 255) instead of the actual EOS token.

    Uses a dummy model that always outputs the last vocab token (id 255).
    Verifies that token still appears in the output when ignore_eos + min_tokens
    are both set.
    """
    max_tok = 20

    llm = LLM(
        model=pathlib.Path("dummy_last_vocab_path"),
        backend="pytorch",
        max_batch_size=2,
        max_seq_len=128,
        max_num_tokens=32,
        kv_cache_config=KvCacheConfig(enable_block_reuse=False),
        disable_overlap_scheduler=True,
        checkpoint_loader=_HfCheckpointLoader(
            weight_loader=_LastVocabWeightLoader(),
            config_loader=_LastVocabConfigLoader()),
    )

    sampling_params = SamplingParams(
        max_tokens=max_tok,
        min_tokens=10,
        ignore_eos=True,
        temperature=0.0,
        end_id=_LAST_VOCAB_EOS_TOKEN_ID,
    )

    with llm:
        res = llm.generate([[1, 2, 3]], sampling_params=sampling_params)

    assert len(res) == 1
    token_ids = res[0].outputs[0].token_ids

    # With ignore_eos, generation must run for exactly max_tokens.
    assert len(token_ids) == max_tok, (
        f"Generated {len(token_ids)} tokens, expected {max_tok}.")

    # The dummy model always outputs the last vocab token (id 255).
    # The original bug suppressed logits[..., -1] which is this token.
    # After the fix, the last vocab token must NOT be suppressed.
    assert all(t == _LAST_VOCAB_TOKEN_ID for t in token_ids), (
        f"Expected all tokens to be {_LAST_VOCAB_TOKEN_ID}, got {token_ids}. "
        f"logits[..., -1] may be incorrectly suppressing the last vocab token.")
