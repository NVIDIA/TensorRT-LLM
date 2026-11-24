# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""trtllm-serve ``prompt_embeds``: embeddings of a prompt's tokens generate what the tokens do."""

import base64
import io
import os
from types import SimpleNamespace
from typing import Any, Dict, List

import openai
import pytest
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from ..test_llm import get_model_path
from .openai_server import RemoteOpenAIServer

pytestmark = pytest.mark.threadleak(enabled=False)

MAX_TOKENS = 16


@pytest.fixture(scope="module")
def model_path() -> str:
    """Qwen3-0.6B checkpoint directory."""
    return get_model_path("Qwen3/Qwen3-0.6B")


@pytest.fixture(scope="module")
def server(model_path: str) -> RemoteOpenAIServer:
    """A server with prompt_embeds enabled."""
    with RemoteOpenAIServer(model_path, ["--enable_prompt_embeds"]) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def client(server: RemoteOpenAIServer) -> openai.OpenAI:
    """OpenAI client for the server."""
    return server.get_client()


@pytest.fixture(scope="module")
def tokenizer(model_path: str) -> Any:
    """The model's tokenizer."""
    return AutoTokenizer.from_pretrained(model_path)


@pytest.fixture(scope="module")
def embed_table(model_path: str) -> torch.Tensor:
    """The model's input embedding table."""
    with safe_open(os.path.join(model_path, "model.safetensors"), framework="pt") as f:
        return f.get_tensor("model.embed_tokens.weight")


def _b64(tensor: torch.Tensor) -> str:
    """Encode a tensor as a prompt_embeds payload."""
    buf = io.BytesIO()
    torch.save(tensor.contiguous(), buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _greedy() -> Dict[str, Any]:
    """Sampling arguments for a deterministic comparison."""
    return {"max_tokens": MAX_TOKENS, "temperature": 0.0}


def _complete(client: openai.OpenAI, model: str, **kwargs: Any) -> Any:
    """Call /v1/completions, passing prompt_embeds through extra_body."""
    extra = {}
    if "prompt_embeds" in kwargs:
        extra["prompt_embeds"] = kwargs.pop("prompt_embeds")
        kwargs.setdefault("prompt", None)
    return client.completions.create(model=model, extra_body=extra, **_greedy(), **kwargs)


def test_completion_matches_token_prompt(
    client: openai.OpenAI, model_path: str, tokenizer: Any, embed_table: torch.Tensor
) -> None:
    """A whole-prompt payload of token embeddings reproduces the token prompt."""
    ids: List[int] = tokenizer.encode("The capital of France is", add_special_tokens=False)
    by_ids = client.completions.create(model=model_path, prompt=ids, **_greedy())
    by_embeds = _complete(client, model_path, prompt_embeds=_b64(embed_table[ids]))
    assert by_embeds.choices[0].text == by_ids.choices[0].text
    assert by_embeds.usage.prompt_tokens == len(ids)


# In a batch, embedding and token prompts take different bf16 kernel paths
# (measured drift up to ~0.11 over 16 steps; alone they match exactly), which
# can also flip an exact tie between two greedy candidates.
LOGPROB_ATOL = 0.25


def _log_margins(reference: Any, candidate: Any) -> None:
    """Print each position's tokens and the reference's top-2 logprob margin."""
    for step, (ref_tok, cand_tok) in enumerate(
        zip(reference.logprobs.tokens, candidate.logprobs.tokens)
    ):
        top = sorted(reference.logprobs.top_logprobs[step].values(), reverse=True)
        margin = top[0] - top[1] if len(top) > 1 else float("inf")
        print(f"step {step:2d} ref={ref_tok!r} cand={cand_tok!r} margin={margin:.3f}")


def _assert_same_greedy_path(reference: Any, candidate: Any) -> None:
    """Assert two greedy choices agree token by token, except across a near-tie.

    Up to the first differing token the chosen tokens' logprobs must match; at
    that token both picks must be near-tied in the reference distribution. The
    candidate must have finished, with every token when it ran to the limit.
    """
    ref_lp, cand_lp = reference.logprobs, candidate.logprobs
    assert candidate.finish_reason is not None
    if candidate.finish_reason == "length":
        assert len(cand_lp.tokens) == MAX_TOKENS, len(cand_lp.tokens)
    for step, (ref_tok, cand_tok) in enumerate(zip(ref_lp.tokens, cand_lp.tokens)):
        ref_top = ref_lp.top_logprobs[step]
        if ref_tok != cand_tok:
            assert cand_tok in ref_top, (step, ref_tok, cand_tok, ref_top)
            gap = abs(ref_top[ref_tok] - ref_top[cand_tok])
            assert gap <= LOGPROB_ATOL, (step, ref_tok, cand_tok, ref_top)
            return
        drift = abs(ref_lp.token_logprobs[step] - cand_lp.token_logprobs[step])
        assert drift <= LOGPROB_ATOL, (step, ref_tok, drift)
    assert reference.text == candidate.text


def test_completion_batch_and_stream(
    client: openai.OpenAI, model_path: str, tokenizer: Any, embed_table: torch.Tensor
) -> None:
    """Several payloads in one request follow their token prompts; streaming serves them all."""
    texts = ["Water boils at a temperature of", "The largest planet in the solar system is"]
    id_lists = [tokenizer.encode(t, add_special_tokens=False) for t in texts]
    payloads = [_b64(embed_table[ids]) for ids in id_lists]
    by_ids = client.completions.create(model=model_path, prompt=id_lists, logprobs=2, **_greedy())
    by_embeds = _complete(client, model_path, prompt_embeds=payloads, logprobs=2)
    expected = sorted(by_ids.choices, key=lambda c: c.index)
    got = sorted(by_embeds.choices, key=lambda c: c.index)
    assert [c.index for c in got] == [0, 1]
    for reference, candidate in zip(expected, got):
        _log_margins(reference, candidate)
        _assert_same_greedy_path(reference, candidate)

    stream = client.completions.create(
        model=model_path,
        prompt=None,
        stream=True,
        logprobs=2,
        extra_body={"prompt_embeds": payloads},
        **_greedy(),
    )
    streamed = [
        SimpleNamespace(
            text="",
            finish_reason=None,
            logprobs=SimpleNamespace(tokens=[], token_logprobs=[], top_logprobs=[]),
        )
        for _ in payloads
    ]
    for chunk in stream:
        for choice in chunk.choices:
            out = streamed[choice.index]
            out.text += choice.text
            out.finish_reason = choice.finish_reason or out.finish_reason
            if choice.logprobs is not None:
                out.logprobs.tokens += choice.logprobs.tokens
                out.logprobs.token_logprobs += choice.logprobs.token_logprobs
                out.logprobs.top_logprobs += choice.logprobs.top_logprobs
    # Streaming must follow the same greedy path as the non-streamed request of
    # the same batch; a split is allowed only at a near-tie (top-2 margin).
    for reference, candidate in zip(got, streamed):
        _log_margins(reference, candidate)
        _assert_same_greedy_path(reference, candidate)


def test_chat_part_matches_text(
    client: openai.OpenAI, model_path: str, tokenizer: Any, embed_table: torch.Tensor
) -> None:
    """A chat part holding a message's token embeddings reproduces the text message."""
    question = "What is the capital of Japan? Answer in one word."
    ids = tokenizer.encode(question, add_special_tokens=False)
    by_text = client.chat.completions.create(
        model=model_path, messages=[{"role": "user", "content": question}], **_greedy()
    )
    part = {"type": "prompt_embeds", "prompt_embeds": {"data": _b64(embed_table[ids])}}
    by_embeds = client.chat.completions.create(
        model=model_path, messages=[{"role": "user", "content": [part]}], **_greedy()
    )
    assert by_embeds.choices[0].message.content == by_text.choices[0].message.content
    assert by_embeds.usage.prompt_tokens == by_text.usage.prompt_tokens


def test_completion_wrong_hidden_size(client: openai.OpenAI, model_path: str) -> None:
    """A payload of the wrong width is a client error."""
    with pytest.raises(openai.BadRequestError, match="hidden size"):
        _complete(client, model_path, prompt_embeds=_b64(torch.zeros(4, 7, dtype=torch.bfloat16)))
