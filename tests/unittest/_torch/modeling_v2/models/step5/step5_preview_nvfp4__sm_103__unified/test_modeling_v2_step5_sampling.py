# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Step-5 ByteLevel tokenizer repair and request EOS semantics on CPU.

Synthetic tokenizer assets keep these regressions independent of model weights.
"""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
from transformers import GenerationConfig, PreTrainedTokenizerFast

from tensorrt_llm._torch.pyexecutor.llm_request import executor_request_to_llm_request
from tensorrt_llm._torch.pyexecutor.sampler.sampler_features import handle_stop_criteria
from tensorrt_llm.bindings.executor import FinishReason
from tensorrt_llm.executor.base_worker import BaseWorker
from tensorrt_llm.executor.request import GenerationRequest
from tensorrt_llm.llmapi.llm import BaseLLM
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.tokenizer import (
    load_hf_tokenizer,
    maybe_fix_byte_level_tokenizer,
    tokenizer_factory,
)

pytestmark = pytest.mark.cpu_only


@pytest.fixture(scope="module")
def small_tokenizer_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp_path = tmp_path_factory.mktemp("bytelevel")
    specials = ["<bos>", "<base_eos>", "<pad>", "<|im_end|>"]
    vocab = {
        token: i for i, token in enumerate(specials + sorted(pre_tokenizers.ByteLevel.alphabet()))
    }
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)]
    )
    backend.decoder = decoders.ByteLevel()
    backend.post_processor = processors.TemplateProcessing(
        single="<bos> $A", special_tokens=[("<bos>", 0)]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<bos>",
        eos_token="<|im_end|>",
        pad_token="<pad>",
        clean_up_tokenization_spaces=False,
        chat_template="{{ bos_token }}{% for m in messages %}{{ m['content'] }}{% endfor %}",
    )
    tokenizer.save_pretrained(tmp_path)
    config_path = tmp_path / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    config.update(tokenizer_class="LlamaTokenizerFast", add_bos_token=True, add_eos_token=False)
    config_path.write_text(json.dumps(config))
    return tmp_path


def _step5_config(eos_ids=(1, 2), *, composite: bool = True):
    text = SimpleNamespace(model_type="step5", eos_token_id=eos_ids)
    return SimpleNamespace(model_type="step5_vl", text_config=text) if composite else text


class _SamplingLLM(BaseLLM):
    """Use production request setup without constructing a model or executor."""

    def __init__(self, tokenizer, config, generation_config=None) -> None:
        self.args = TorchLlmArgs(model="unused", skip_tokenizer_init=True, modeling_v2="require")
        self._tokenizer = tokenizer
        self._hf_model_config = config
        self._generation_config = generation_config
        self._generation_config_explicit_values = {}


def _executor_request(params: SamplingParams):
    # The enqueue sink is the only stub: construct the real native Request via
    # BaseWorker, including its ignore_eos and stop-word translation.
    engine = SimpleNamespace(enqueue_request=Mock(return_value=7))
    worker = SimpleNamespace(_lora_manager=None, max_seq_len=9216, engine=engine)
    request = GenerationRequest(prompt_token_ids=[0, 42], sampling_params=params)
    request.id = 7
    assert BaseWorker._enqueue_request(worker, request) == 7
    return engine.enqueue_request.call_args.args[0]


def _stop_request(params: SamplingParams, tokens: list[int]):
    request = executor_request_to_llm_request(
        7, _executor_request(params), child_req_ids=[], exclude_last_generation_logits=False
    )
    # The stop predicate sees a request after token append. Supply a short token
    # history via the request's native append operation, without sampling logits.
    for token in tokens:
        request.add_new_token(token, 0)
    request.finish_by = Mock(wraps=request.finish_by)
    return request


@pytest.mark.parametrize("loader", [tokenizer_factory, load_hf_tokenizer])
def test_factory_preserves_serialized_byte_level(small_tokenizer_dir: Path, loader) -> None:
    tokenizer = loader(small_tokenizer_dir, trust_remote_code=False, local_files_only=True)
    expected = PreTrainedTokenizerFast.from_pretrained(small_tokenizer_dir, local_files_only=True)
    assert json.loads(tokenizer.tokenizer.backend_tokenizer.to_str()) == json.loads(
        expected.backend_tokenizer.to_str()
    )
    text = "  hello world\n你好 café 😀\tend  "
    assert tokenizer.encode(text) == expected.encode(text)
    assert tokenizer.decode(tokenizer.encode(text), skip_special_tokens=True) == text
    assert tokenizer.get_chat_template() == expected.get_chat_template()


@pytest.mark.parametrize("changed", ["pre_tokenizer", "decoder", "model", "normalizer"])
def test_repair_detects_each_backend_override(small_tokenizer_dir: Path, changed: str) -> None:
    from tokenizers import normalizers

    original = PreTrainedTokenizerFast.from_pretrained(small_tokenizer_dir, local_files_only=True)
    broken = deepcopy(original)
    if changed == "pre_tokenizer":
        broken.backend_tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    elif changed == "decoder":
        # Keep the correct ByteLevel pre-tokenizer: the old Metaspace-only
        # detector missed this source of visible byte markers.
        broken.backend_tokenizer.decoder = decoders.Fuse()
    elif changed == "model":
        broken.backend_tokenizer.model = models.BPE(
            vocab=broken.get_vocab(), merges=[], fuse_unk=True
        )
    else:
        broken.backend_tokenizer.normalizer = normalizers.Lowercase()
    repaired = maybe_fix_byte_level_tokenizer(
        broken, small_tokenizer_dir, local_files_only=True, padding_side="right"
    )
    assert repaired is not broken
    assert json.loads(repaired.backend_tokenizer.to_str()) == json.loads(
        original.backend_tokenizer.to_str()
    )
    assert repaired.padding_side == "right"
    assert repaired.chat_template == original.chat_template
    assert repaired.special_tokens_map == original.special_tokens_map
    text = "Hello  world\n你好 😀"
    assert repaired.decode(repaired.encode(text), skip_special_tokens=True) == text


def test_matching_and_non_byte_level_tokenizers_are_unchanged(small_tokenizer_dir: Path) -> None:
    tokenizer = PreTrainedTokenizerFast.from_pretrained(small_tokenizer_dir, local_files_only=True)
    assert maybe_fix_byte_level_tokenizer(tokenizer, small_tokenizer_dir) is tokenizer
    data_path = small_tokenizer_dir / "tokenizer.json"
    original = data_path.read_text()
    data = json.loads(original)
    try:
        data["model"].pop("dropout")
        data_path.write_text(json.dumps(data))
        assert maybe_fix_byte_level_tokenizer(tokenizer, small_tokenizer_dir) is tokenizer
        data["pre_tokenizer"] = {"type": "Whitespace"}
        data_path.write_text(json.dumps(data))
        assert maybe_fix_byte_level_tokenizer(tokenizer, small_tokenizer_dir) is tokenizer
    finally:
        data_path.write_text(original)


@pytest.mark.parametrize("eos_id", [1, 2, 3])
@pytest.mark.parametrize("composite", [False, True])
def test_resolved_eos_reaches_engine_stop_criteria(
    small_tokenizer_dir: Path, eos_id: int, composite: bool
) -> None:
    tokenizer = tokenizer_factory(
        small_tokenizer_dir, local_files_only=True, trust_remote_code=False
    )
    params = _SamplingLLM(tokenizer, _step5_config(composite=composite))._prepare_sampling_params()
    assert params.end_id == tokenizer.eos_token_id == 3
    assert params.stop_token_ids == [1, 2]
    request = _stop_request(params, [eos_id])
    assert handle_stop_criteria(request, eos_id, max_seq_len=9216, beam_idx=0)
    expected = FinishReason.END_ID if eos_id == 3 else FinishReason.STOP_WORDS
    request.finish_by.assert_called_once_with(expected, 0)


@pytest.mark.parametrize("ignore_eos", [False, True])
def test_explicit_sampling_and_request_stops(small_tokenizer_dir: Path, ignore_eos: bool) -> None:
    tokenizer = tokenizer_factory(
        small_tokenizer_dir, local_files_only=True, trust_remote_code=False
    )
    user_stops = [99]
    params = SamplingParams(
        temperature=0.7,
        top_k=11,
        top_p=0.85,
        seed=42,
        max_tokens=256,
        truncate_prompt_tokens=4096,
        pad_id=77,
        stop=["Question:", "</s>", "<|im_end|>"],
        stop_token_ids=user_stops,
        ignore_eos=ignore_eos,
        include_stop_str_in_output=True,
    )
    llm = _SamplingLLM(tokenizer, _step5_config())
    assert llm._prepare_sampling_params(params) is params
    assert user_stops == [99]
    assert params.stop_token_ids == [99, 1, 2]
    assert params.pad_id == 77
    assert (params.temperature, params.top_k, params.top_p, params.seed) == (0.7, 11, 0.85, 42)
    assert (params.max_tokens, params.truncate_prompt_tokens) == (256, 4096)
    assert params.include_stop_str_in_output
    assert params._get_stop_words() == [[99], [1], [2]] + [
        tokenizer.encode(s, add_special_tokens=False) for s in params.stop
    ]
    # Setup is idempotent; no duplicate EOS ids or mutation of caller lists.
    params._setup(tokenizer, _step5_config(), None)
    assert params.stop_token_ids == [99, 1, 2]
    native = _executor_request(params)
    assert native.end_id == (-1 if ignore_eos else 3)
    assert native.stop_words == ([] if ignore_eos else params._get_stop_words())
    for eos_id in [1, 2, 3]:
        request = _stop_request(params, [eos_id])
        assert handle_stop_criteria(request, eos_id, max_seq_len=9216, beam_idx=0) is not ignore_eos


def test_explicit_end_id_overrides_model_defaults(small_tokenizer_dir: Path) -> None:
    tokenizer = tokenizer_factory(
        small_tokenizer_dir, local_files_only=True, trust_remote_code=False
    )
    params = SamplingParams(end_id=91, pad_id=92, stop_token_ids=[93])
    _SamplingLLM(tokenizer, _step5_config())._prepare_sampling_params(params)
    assert params.end_id == 91 and params.pad_id == 92 and params.stop_token_ids == [93]
    for eos_id in [1, 2, 3]:
        request = _stop_request(params, [eos_id])
        assert not handle_stop_criteria(request, eos_id, max_seq_len=9216, beam_idx=0)
    for token in [91, 93]:
        assert handle_stop_criteria(
            _stop_request(params, [token]), token, max_seq_len=9216, beam_idx=0
        )


@pytest.mark.parametrize("eos_ids", [1, [1, 2, 1], (1, 2), {1, 2}, None])
def test_eos_shapes_and_shared_generation_config(small_tokenizer_dir: Path, eos_ids) -> None:
    tokenizer = tokenizer_factory(
        small_tokenizer_dir, local_files_only=True, trust_remote_code=False
    )
    generation = GenerationConfig(eos_token_id=4)
    params = SamplingParams()._setup(tokenizer, _step5_config(eos_ids), generation)
    expected = {3, 4}
    expected.update([] if eos_ids is None else [eos_ids] if isinstance(eos_ids, int) else eos_ids)
    assert {params.end_id, *params.stop_token_ids} == expected
    assert generation.eos_token_id == 4


@pytest.mark.parametrize("eos_ids", [True, -1, 1.5, "1", [1, False], [1, -2]])
def test_invalid_config_eos_is_rejected(small_tokenizer_dir: Path, eos_ids) -> None:
    tokenizer = tokenizer_factory(
        small_tokenizer_dir, local_files_only=True, trust_remote_code=False
    )
    with pytest.raises(ValueError, match="non-negative integer token ids"):
        SamplingParams()._setup(tokenizer, _step5_config(eos_ids), None)


def test_eos_absence_and_dictionary_text_config(small_tokenizer_dir: Path) -> None:
    tokenizer = tokenizer_factory(
        small_tokenizer_dir, local_files_only=True, trust_remote_code=False
    )
    tokenizer.tokenizer.eos_token = None
    config = SimpleNamespace(model_type="step5_vl", text_config={"eos_token_id": [1, 2]})
    params = SamplingParams()._setup(tokenizer, config, None)
    assert params.end_id == 1 and params.stop_token_ids == [2]
    with pytest.raises(ValueError, match="generation has no EOS token ids"):
        SamplingParams()._setup(tokenizer, _step5_config(None), None)


def test_other_models_keep_existing_eos_defaults(small_tokenizer_dir: Path) -> None:
    tokenizer = tokenizer_factory(
        small_tokenizer_dir, local_files_only=True, trust_remote_code=False
    )
    config = SimpleNamespace(model_type="llama", eos_token_id=[1, 2])
    params = SamplingParams()._setup(tokenizer, config, None)
    assert params.end_id == 3 and params.stop_token_ids is None
    generation = GenerationConfig(eos_token_id=[1, 2])
    params = SamplingParams()._setup(tokenizer, config, generation)
    assert params.end_id == 3 and params.stop_token_ids == [1, 2]
