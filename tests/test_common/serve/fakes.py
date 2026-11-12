# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fake generators with scripted generation for CPU tests of trtllm-serve.

The fakes subclass the real generator classes, so every ``isinstance`` check in
``tensorrt_llm/serve`` passes without patching module attributes, but they never
run the real constructors. ``generate_async`` records every call and replays a
queued :class:`ScriptedOutput` through the same ``RequestOutput`` surface the
request handlers consume, so handler tests run through production code on CPU.

The fakes fail loudly on purpose: a handler that reads a generator attribute the
fakes lack raises ``AttributeError``, which is the signal to extend the fakes
here (with a self-test in ``tests/unittest/serve/test_serve_test_helpers.py``),
not to catch it in the test.

Not scripted: sampling, logits processors, guided decoding, logprobs, perf
metrics, multimodal data. A test that needs one of those adds it to these fakes
with a self-test, not as a private fake in its own test file.
"""

import asyncio
import collections
import itertools
from concurrent.futures import Future
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, AsyncIterator, Optional, Sequence, Union

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from tensorrt_llm._torch.async_llm import AsyncLLM
from tensorrt_llm.executor.result import GenerationResultBase
from tensorrt_llm.inputs.registry import DefaultInputProcessor
from tensorrt_llm.llmapi import MultimodalEncoder
from tensorrt_llm.llmapi.llm import LLM, PreprocessedInputs
from tensorrt_llm.llmapi.tokenizer import TransformersTokenizer
from tensorrt_llm.sampling_params import SamplingParams

__all__ = [
    "FAKE_CHAT_TEMPLATE",
    "FakeAsyncLLM",
    "FakeCheckpointExecutor",
    "FakeEncodeOnlyLLM",
    "FakeEncoderExecutor",
    "FakeExecutor",
    "FakeLLM",
    "FakeLlmArgs",
    "FakeMultimodalEncoder",
    "GenerateCall",
    "PreprocessCall",
    "ScriptedOutput",
    "ScriptedRequestOutput",
    "ScriptedStep",
    "UserProbeMiddleware",
    "make_fake_tokenizer",
]

# Renders each message as <|role|>content and appends <|assistant|> when
# add_generation_prompt is true, so chat requests render offline through the
# production template machinery.
FAKE_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "<|{{ message['role'] }}|>{{ message['content'] }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)


def make_fake_tokenizer() -> TransformersTokenizer:
    """An in-memory tokenizer with a working chat template, built offline.

    Gives the handlers ``.tokenizer``, ``encode``, ``decode`` and a chat
    template without reading any file. The vocabulary only needs to exist;
    unknown words map to ``<unk>``.
    """
    vocab = {
        "<unk>": 0,
        "</s>": 1,
        "<|system|>": 2,
        "<|user|>": 3,
        "<|assistant|>": 4,
        "hi": 5,
        "hello": 6,
        "ok": 7,
    }
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        eos_token="</s>",
        chat_template=FAKE_CHAT_TEMPLATE,
    )
    return TransformersTokenizer(hf_tokenizer)


@dataclass
class FakeLlmArgs:
    """The LLM args fields ``OpenAIServer`` reads at construction, in its lifespan and in its handlers.

    Defaults are the ``TorchLlmArgs`` defaults. A new direct read of a field
    missing here fails loudly with ``AttributeError``; add the field.
    """

    model: str  # path of the fake checkpoint directory
    backend: str = "pytorch"
    dtype: str = "auto"
    trust_remote_code: bool = False
    checkpoint_format: Optional[str] = None
    num_postprocess_workers: int = 0
    post_processor_hook: Optional[str] = None
    return_perf_metrics: bool = False
    perf_metrics_output_dir: Optional[str] = None
    per_request_spec_decode_stats: bool = False
    speculative_config: Any = None
    decoding_config: Any = None
    prometheus_metrics_config: Any = None
    quant_config: Any = None
    max_seq_len: Optional[int] = None
    max_batch_size: Optional[int] = 2048  # get_server_info reads it
    parallel_config: Any = field(
        default_factory=lambda: SimpleNamespace(
            tp_size=1, pp_size=1, cp_size=1, world_size=1, moe_ep_size=None
        )
    )
    kv_cache_config: Any = None
    cache_transceiver_config: Any = None
    enable_energy_metrics: bool = False
    enable_iter_perf_stats: bool = False
    iter_stats_max_iterations: Optional[int] = None
    encode_only: bool = False
    sleep_config: Any = None
    num_serve_frontends: int = 1
    env_overrides: dict[str, str] | None = None
    max_beam_width: int = 1
    reasoning_parser: Optional[str] = None  # chat, completions, Responses
    gather_generation_logits: bool = False  # to_sampling_params in chat and completions
    guided_decoding_backend: Optional[str] = None  # the required-tool grammar check


class FakeExecutor:
    """Stands in for the executor where ``OpenAIServer`` reads it."""

    def __init__(self, *, resource_governor_queue: object = None) -> None:
        # register_routes reads it to decide on the resource-governor routes.
        self.resource_governor_queue = resource_governor_queue
        self.healthy = True
        self._fatal_error: Optional[BaseException] = None  # read by the /health handler
        self.doing_shutdown = False  # read by the /health handler

    @property
    def fatal_error(self) -> Optional[BaseException]:
        """Public name of ``_fatal_error``; nothing reads it yet."""
        return self._fatal_error

    def check_health(self) -> bool:
        # Like the real executor: healthy unless a fatal error was recorded.
        return self.healthy and self._fatal_error is None

    def shutdown(self) -> None:
        self.doing_shutdown = True


class FakeCheckpointExecutor(FakeExecutor):
    """Supply startup profile evidence, not native lifecycle mutations, on CPU."""

    def collective_rpc(
        self,
        method: str,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
        unique_reply_rank: int | None = None,
        target_ranks: int | list[int] | None = None,
    ) -> list:
        """Return the fixed profile used to construct checkpoint HTTP routes.

        Args:
            method: Only the read-only checkpoint profile query is supported.
            args: Empty profile-query arguments.
            kwargs: Empty profile-query keyword arguments.
            non_block: Whether to return a completed future as the RPC bridge does.
            unique_reply_rank: Optional rank-zero reply selection.
            target_ranks: Optional rank-zero target selection.

        Returns:
            One profile, wrapped in a future for a nonblocking call.

        Raises:
            NotImplementedError: If a test attempts native mutations or other ranks.
        """
        if (
            method != "checkpoint_profile"
            or args
            or kwargs
            or unique_reply_rank not in (None, 0)
            or target_ranks not in (None, 0, [0])
        ):
            raise NotImplementedError("The fake only supports the rank-zero checkpoint profile.")
        profile = {"profile_digest": "a" * 64, "cohort_digest": "b" * 64, "ranks": [0]}
        if non_block:
            future = Future()
            future.set_result(profile)
            return [future]
        return [profile]


class FakeEncoderExecutor:
    """Stands in for the encoder executor of an encode-only LLM."""

    def __init__(self) -> None:
        # The resolved limits the embedding server sizes its encode batcher from.
        self.max_batch_size = 8
        self.max_num_tokens = 8192
        self.max_seq_len = 2048

    def shutdown(self) -> None:
        pass


@dataclass(frozen=True)
class ScriptedStep:
    """One engine response: what it appends to the outputs."""

    text: str = ""
    token_ids: tuple = ()
    finish_reason: Optional[str] = None  # set on the last step: "stop", "length", ...
    stop_reason: Optional[Union[int, str]] = None
    index: Optional[int] = None  # output index; None applies the step to every output


@dataclass(frozen=True)
class ScriptedOutput:
    """What one ``generate_async`` call produces."""

    steps: tuple = (ScriptedStep(text="ok", token_ids=(7,), finish_reason="stop"),)
    raise_on_call: Optional[BaseException] = None  # raised after the call is recorded
    raise_after_steps: Optional[BaseException] = None  # raised once the steps are replayed
    hold_after_steps: bool = False  # wait until release(), abort() or cancellation
    prompt_token_ids: Optional[tuple] = None  # default: derived from the inputs


@dataclass
class GenerateCall:
    """One recorded ``generate_async`` call."""

    inputs: Any  # exactly what the handler passed
    kwargs: dict  # every other argument by name, including sampling_params
    result: Optional["ScriptedRequestOutput"]  # None when raise_on_call fired

    @property
    def sampling_params(self) -> Optional[SamplingParams]:
        return self.kwargs.get("sampling_params")


@dataclass
class PreprocessCall:
    """One recorded ``preprocess`` call."""

    inputs: Any
    sampling_params: Any
    disaggregated_params: Any


def _derive_prompt_fields(inputs: Any, tokenizer: TransformersTokenizer) -> tuple:
    """(prompt token ids, prompt text or None) the way the engine would see them."""
    if isinstance(inputs, PreprocessedInputs):
        return list(inputs.prompt_token_ids), None
    if isinstance(inputs, dict):
        if inputs.get("prompt_token_ids") is not None:  # TokensPrompt
            return list(inputs["prompt_token_ids"]), None
        if inputs.get("prompt") is not None:  # TextPrompt
            return list(tokenizer.encode(inputs["prompt"])), inputs["prompt"]
    if isinstance(inputs, str):
        return list(tokenizer.encode(inputs)), inputs
    if isinstance(inputs, Sequence) and all(isinstance(t, int) for t in inputs):
        return list(inputs), None
    raise TypeError(f"cannot derive prompt token ids from inputs of type {type(inputs)!r}")


class ScriptedRequestOutput(GenerationResultBase):
    """Replays a :class:`ScriptedOutput` with the ``RequestOutput`` surface handlers use."""

    def __init__(
        self,
        *,
        request_id: int,
        script: ScriptedOutput,
        inputs: Any,
        sampling_params: Optional[SamplingParams],
        tokenizer: TransformersTokenizer,
        postproc_params: Any = None,
    ) -> None:
        GenerationResultBase.__init__(
            self,
            request_id,
            sampling_params if sampling_params is not None else SamplingParams(),
            postproc_params=postproc_params,
        )
        self._script = script
        self._fake_tokenizer = tokenizer
        if script.prompt_token_ids is not None:
            self.prompt_token_ids = list(script.prompt_token_ids)
            try:
                _, self.prompt = _derive_prompt_fields(inputs, tokenizer)
            except TypeError:
                self.prompt = None
        else:
            self.prompt_token_ids, self.prompt = _derive_prompt_fields(inputs, tokenizer)
        self._pending = collections.deque(script.steps)
        self._release_event = asyncio.Event()
        self._after_steps_raised = False

    @property
    def request_id(self) -> int:
        return self.id

    @property
    def finished(self) -> bool:
        return self._done

    def release(self) -> None:
        """End a ``hold_after_steps`` wait normally."""
        self._release_event.set()

    def abort(self) -> None:
        # An aborted engine request ends with finish_reason "cancelled".
        for output in self._outputs:
            output.finish_reason = "cancelled"
        super().abort()
        self._done = True
        self._release_event.set()

    def _replay_step(self, step: ScriptedStep) -> None:
        targets = self._outputs if step.index is None else [self._outputs[step.index]]
        for output in targets:
            # Update the diff markers the way the result handler does, so
            # text_diff / token_ids_diff work for streaming post-processors.
            output._last_text_len = len(output.text)
            output._last_token_ids_len = len(output.token_ids)
            output._last_logprobs_len = len(output.logprobs)
            output.text += step.text
            output.token_ids = list(output.token_ids) + list(step.token_ids)
            if step.finish_reason is not None:
                output.finish_reason = step.finish_reason
                output.stop_reason = step.stop_reason
        if not self._pending and not self._script.hold_after_steps:
            self._done = True
        if self.postproc_params is not None:
            # As the postprocess worker does: fill _postprocess_result after
            # each response, with the tokenizer bound on the args.
            args = self.postproc_params.postproc_args
            args.tokenizer = self._fake_tokenizer
            self._outputs[0]._postprocess_result = self.postproc_params.post_processor(self, args)

    def _finish_after_steps(self) -> None:
        if self._script.raise_after_steps is not None and not self._after_steps_raised:
            self._after_steps_raised = True
            self._done = True
            raise self._script.raise_after_steps

    async def aresult(self) -> "ScriptedRequestOutput":
        while self._pending:
            self._replay_step(self._pending.popleft())
        self._finish_after_steps()
        if self._script.hold_after_steps and not self._done:
            await self._release_event.wait()
        self._done = True
        return self

    def __await__(self):
        return self.aresult().__await__()

    def __aiter__(self) -> "ScriptedRequestOutput":
        return self

    async def __anext__(self) -> "ScriptedRequestOutput":
        if self._pending:
            self._replay_step(self._pending.popleft())
            return self
        self._finish_after_steps()
        if self._script.hold_after_steps and not self._done:
            await self._release_event.wait()
            self._done = True
        raise StopAsyncIteration


class _FakeGeneratorMixin:
    """Shared state and scripted generation of every fake generator."""

    def _setup_fake(
        self,
        *,
        args: FakeLlmArgs,
        executor: Optional[FakeExecutor],
        encode_only: bool = False,
    ) -> None:
        self.args = args
        self._hf_model_dir = args.model  # read by OpenAIServer._init_llm
        self._tokenizer = make_fake_tokenizer()
        # BaseLLM.tokenizer returns input_processor.tokenizer; completions call
        # the input processor for text prompts. The real DefaultInputProcessor
        # needs only a tokenizer.
        self.input_processor = DefaultInputProcessor(
            model_path=args.model, config=None, tokenizer=self._tokenizer
        )
        self._executor = executor  # BaseLLM._check_health and /health read it
        self._encode_only = encode_only  # BaseLLM._check_health reads it
        self._llm_id = "fake-llm"
        self.generate_calls: list = []
        self.preprocess_calls: list = []
        self._scripts: collections.deque = collections.deque()
        self._request_ids = itertools.count(1)

    @property
    def llm_id(self) -> str:
        return self._llm_id

    def script(self, *outputs: ScriptedOutput) -> None:
        """Queue outputs for the next ``generate_async`` calls."""
        self._scripts.extend(outputs)

    def preprocess(
        self,
        inputs: Any,
        sampling_params: Any = None,
        disaggregated_params: Any = None,
    ) -> PreprocessedInputs:
        # Replaces the inherited BaseLLM.preprocess, which needs
        # args.generation_config and a model.
        self.preprocess_calls.append(PreprocessCall(inputs, sampling_params, disaggregated_params))
        token_ids, _ = _derive_prompt_fields(inputs, self._tokenizer)
        return PreprocessedInputs(prompt_token_ids=token_ids)

    def generate_async(
        self,
        inputs: Any,
        sampling_params: Optional[SamplingParams] = None,
        **kwargs: Any,
    ) -> ScriptedRequestOutput:
        # Record first, so a scripted failure still leaves the call inspectable.
        call = GenerateCall(
            inputs=inputs,
            kwargs={"sampling_params": sampling_params, **kwargs},
            result=None,
        )
        self.generate_calls.append(call)
        script = self._scripts.popleft() if self._scripts else ScriptedOutput()
        if script.raise_on_call is not None:
            raise script.raise_on_call
        postproc_params = kwargs.get("_postproc_params")
        promise = ScriptedRequestOutput(
            request_id=next(self._request_ids),
            script=script,
            inputs=inputs,
            sampling_params=sampling_params,
            tokenizer=self._tokenizer,
            postproc_params=postproc_params,
        )
        if postproc_params is not None:
            # As GenerationExecutor.generate_async does before handing the
            # request to the postprocess worker.
            postproc_params.postproc_args.num_prompt_tokens = len(promise.prompt_token_ids)
        call.result = promise
        return promise

    async def get_stats_async(self, timeout: Optional[float] = None) -> AsyncIterator[dict]:
        return
        yield  # pragma: no cover - makes this an async generator

    def get_data_transceiver_state(self) -> None:
        return None

    def shutdown(self) -> None:
        # Overrides BaseLLM.shutdown, which BaseLLM.__del__ calls.
        for name in ("_executor", "_encoder_executor"):
            executor = getattr(self, name, None)
            if executor is not None:
                executor.shutdown()


class FakeLLM(_FakeGeneratorMixin, LLM):
    """LLM generator fake: the default, CONTEXT, GENERATION and runtime-control roles."""

    def __init__(self, *, args: FakeLlmArgs, executor: Optional[FakeExecutor] = None) -> None:
        self._setup_fake(args=args, executor=executor or FakeExecutor())


class FakeAsyncLLM(_FakeGeneratorMixin, AsyncLLM):
    """AsyncLLM generator fake: the RL-control role gate requires AsyncLLM."""

    def __init__(self, *, args: FakeLlmArgs, executor: Optional[FakeExecutor] = None) -> None:
        self._setup_fake(args=args, executor=executor or FakeExecutor())


class FakeMultimodalEncoder(_FakeGeneratorMixin, MultimodalEncoder):
    """MultimodalEncoder generator fake: the MM_ENCODER role gate."""

    def __init__(self, *, args: FakeLlmArgs, executor: Optional[FakeExecutor] = None) -> None:
        self._setup_fake(args=args, executor=executor or FakeExecutor())


class FakeEncodeOnlyLLM(FakeLLM):
    """Encode-only LLM fake for the EMBEDDING role.

    A production encode-only LLM has no generation executor: ``_build_model``
    returns before creating one and ``_check_health`` reads only
    ``_encoder_executor`` in encode-only mode.
    """

    def __init__(self, *, args: FakeLlmArgs) -> None:
        self._setup_fake(args=args, executor=None, encode_only=True)
        self._encoder_executor = FakeEncoderExecutor()

    def encode(self, token_ids_batch: Any) -> Any:
        raise NotImplementedError(
            "FakeEncodeOnlyLLM does not encode; script it when a test needs it."
        )


class UserProbeMiddleware:
    """Pure ASGI no-op, used to check where ``--middleware`` lands in the stack."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        await self.app(scope, receive, send)
