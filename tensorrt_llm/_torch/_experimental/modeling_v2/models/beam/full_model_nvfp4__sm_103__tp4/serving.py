# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Text serving for the Beam NVFP4 full-model target: the channel parser and a native launch.

The model writes its reply as channel messages after the chat template's bare
``<|start|>assistant`` generation prefix::

    <|channel|>analysis<|message|>...<|end|><|start|>assistant<|channel|>final<|message|>...<|return|>

The delimiters are ordinary (non-special) vocabulary tokens, so they reach the detokenized text
whatever ``skip_special_tokens`` says. ``BeamHarmonyTextParser`` (reasoning parser key
``beam_harmony_text``) routes analysis bodies to ``reasoning_content`` and final bodies to
``content`` and drops every header and terminator. A reply without channel headers is final
text. It is incremental: a possible delimiter split across stream chunks is held back until it
is resolved, and ``finish`` releases a held tail that is ordinary text. ``parse`` runs the same
machine over a whole reply, so streamed and non-streamed fields agree. Tool calls are not
supported; a message on any other channel is ignored.

``main`` launches the existing OpenAI-compatible server (``launch_server``) on the supported
deployment: four ranks (attention TP4, MoE TP4/EP1, PP1), ``modeling_v2="require"``, TRTLLM
attention, ``KVCacheManagerV2`` with block reuse, the native TorchSampler, eager execution
(no CUDA graphs, no overlap scheduler, no chunked prefill) and one frontend. The native
scheduler batches up to ``--max-batch-size`` requests per engine step (default 1: one sequence
at a time). It owns no scheduler, sampler, route or evaluator::

    python -m tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4.serving \\
        --checkpoint <dir> [--draft-checkpoint <dir>] [--port 8000] [--max-seq-len 69632] \\
        [--max-num-tokens 8192] [--max-batch-size 1] [--iter-stats]

``--draft-checkpoint`` adds DFlash speculative decoding with that DFlash 2 drafter: the native
one-model DFlash worker with ``max_draft_len`` 7 (a verification block of eight rows), TRTLLM
draft attention over the engine's paged draft KV pool and strict greedy verification; the
one-model speculative sampler replaces the TorchSampler and only target-verified tokens reach
the client. ``--iter-stats`` enables the engine's iteration statistics (``GET /metrics``:
per-step scheduled, context and generation request counts, latency and memory).
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Optional

from tensorrt_llm.llmapi.reasoning_parser import (
    BaseReasoningParser,
    ReasoningParserResult,
    register_reasoning_parser,
)

START, END, MESSAGE, CHANNEL, CONSTRAIN, RETURN, CALL = (
    "<|start|>",
    "<|end|>",
    "<|message|>",
    "<|channel|>",
    "<|constrain|>",
    "<|return|>",
    "<|call|>",
)
TAGS = (START, END, MESSAGE, CHANNEL, CONSTRAIN, RETURN, CALL)
ANALYSIS, FINAL = "analysis", "final"
PARSER_KEY = "beam_harmony_text"


def _held_tail(text: str) -> int:
    """Length of the longest suffix of ``text`` that is a proper prefix of a delimiter."""
    for n in range(min(len(text), max(len(t) for t in TAGS) - 1), 0, -1):
        tail = text[-n:]
        if any(t.startswith(tail) and t != tail for t in TAGS):
            return n
    return 0


@register_reasoning_parser(PARSER_KEY)
class BeamHarmonyTextParser(BaseReasoningParser):
    """Split a Beam channel reply into ``reasoning_content`` (analysis) and ``content`` (final)."""

    needs_raw_special_tokens = True
    reasoning_start = CHANNEL + ANALYSIS + MESSAGE
    reasoning_end = END

    def __init__(self, *, chat_template_kwargs: Optional[dict[str, Any]] = None) -> None:
        super().__init__(chat_template_kwargs=chat_template_kwargs)
        self._reset()

    def _reset(self) -> None:
        # "header": before a channel's <|message|>; "body": inside a message; "between": after
        # <|end|> until the next header; "done": after <|return|> or <|call|>.
        self._state = "header"
        self._channel: Optional[str] = None
        self._header = ""
        self._buffer = ""
        self._saw_header = False

    def _emit(self, text: str, out: list) -> None:
        if not text:
            return
        if self._channel == ANALYSIS:
            out[1].append(text)
        elif self._channel == FINAL or (self._channel is None and not self._saw_header):
            out[0].append(text)

    def _feed(self, text: str) -> ReasoningParserResult:
        out = ([], [])
        data = self._buffer + text
        self._buffer = ""
        while data:
            if self._state == "done":
                data = ""
                break
            hits = [(data.find(t), t) for t in TAGS if t in data]
            pos, tag = min(hits) if hits else (-1, None)
            if tag is None:
                hold = _held_tail(data)
                chunk, self._buffer = data[: len(data) - hold], data[len(data) - hold :]
                data = ""
            else:
                chunk, data = data[:pos], data[pos + len(tag) :]
            if self._state == "body":
                self._emit(chunk, out)
            elif self._state == "header":
                if self._header or self._saw_header or tag in (CHANNEL, MESSAGE, CONSTRAIN):
                    self._header += chunk
                else:
                    # Text before any header: a reply without channels is final text.
                    self._emit(chunk, out)
            if tag is None:
                break
            if tag == CHANNEL:
                self._state, self._header, self._saw_header = "header", "", True
            elif tag == MESSAGE:
                if self._state == "header":
                    self._channel = self._header.split(CONSTRAIN)[0].strip().split(" ")[0] or None
                    self._header = ""
                self._state = "body"
            elif tag == CONSTRAIN:
                self._header += CONSTRAIN
            elif tag in (END, START):
                self._state, self._channel, self._header = "between", None, ""
            elif tag in (RETURN, CALL):
                self._state, self._channel = "done", None
        return ReasoningParserResult(content="".join(out[0]), reasoning_content="".join(out[1]))

    def parse(self, text: str) -> ReasoningParserResult:
        self._reset()
        result = self._feed(text)
        tail = self.finish()
        return ReasoningParserResult(
            content=result.content + tail.content,
            reasoning_content=result.reasoning_content + tail.reasoning_content,
        )

    def parse_delta(self, delta_text: str) -> ReasoningParserResult:
        return self._feed(delta_text)

    def finish(self) -> ReasoningParserResult:
        held, self._buffer = self._buffer, ""
        out = ([], [])
        if self._state == "body" and held and held not in TAGS:
            self._emit(held, out)
        elif self._state == "header" and not self._saw_header and not self._header and held:
            self._emit(held, out)
        return ReasoningParserResult(content="".join(out[0]), reasoning_content="".join(out[1]))


#: The supported deployment's settings (one source for the launcher and its records).
DEPLOYMENT = {
    "tensor_parallel_size": 4,
    "pipeline_parallel_size": 1,
    "moe_tensor_parallel_size": 4,
    "moe_expert_parallel_size": 1,
    "enable_attention_dp": False,
    "attn_backend": "TRTLLM",
    "modeling_v2": "require",
    "disable_overlap_scheduler": True,
    "enable_chunked_prefill": False,
}


def load_tokenizer(checkpoint: str):
    """The checkpoint's fast tokenizer, loaded by its concrete class.

    ``AutoTokenizer`` would first parse config.json as a generic transformers configuration,
    which rejects Beam's per-layer-type RoPE parameters; the tokenizer itself needs none of it.
    """
    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast.from_pretrained(checkpoint)


def dflash_config(draft_checkpoint: str):
    """The DFlash speculative config of the supported deployment for ``draft_checkpoint``.

    ``max_draft_len`` is the one verification width the target serves (``routing``); the
    capture layers and mask token come from the drafter's own ``dflash_config``.
    """
    from tensorrt_llm.llmapi.llm_args import DFlashDecodingConfig

    from ..routing import DFLASH_MAX_DRAFT_LEN

    if not os.path.isfile(os.path.join(draft_checkpoint, "config.json")):
        raise ValueError(f"the drafter checkpoint {draft_checkpoint!r} has no config.json")
    return DFlashDecodingConfig(
        max_draft_len=DFLASH_MAX_DRAFT_LEN,
        speculative_model=draft_checkpoint,
        attention_backend="TRTLLM",
    )


def llm_args(
    checkpoint: str,
    *,
    max_seq_len: int = 69632,
    max_num_tokens: int = 8192,
    max_batch_size: int = 1,
    kv_cache_max_tokens: Optional[int] = None,
    iter_stats: bool = False,
    draft_checkpoint: Optional[str] = None,
) -> dict:
    """Native LLM API arguments of the supported deployment for ``checkpoint``.

    ``max_batch_size`` bounds the requests the native scheduler runs in one engine step;
    ``kv_cache_max_tokens`` (default ``2 * max_seq_len``) sizes the shared KV cache.
    ``draft_checkpoint`` adds DFlash speculative decoding with that drafter
    (``dflash_config``); its draft KV pool is the engine's separate draft manager.
    """
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig

    from .model import MAX_NUM_TOKENS
    from .weights import BeamNvfp4CheckpointLoader

    if not 1 <= max_num_tokens <= MAX_NUM_TOKENS:
        raise ValueError(f"max_num_tokens must be in [1, {MAX_NUM_TOKENS}], got {max_num_tokens}")
    args = {
        "model": checkpoint,
        "tokenizer": load_tokenizer(checkpoint),
        "backend": "pytorch",
        **DEPLOYMENT,
        "max_batch_size": max_batch_size,
        "max_seq_len": max_seq_len,
        "max_num_tokens": max_num_tokens,
        "cuda_graph_config": None,
        "kv_cache_config": KvCacheConfig(
            enable_block_reuse=True,
            enable_partial_reuse=False,
            use_kv_cache_manager_v2=True,
            max_tokens=kv_cache_max_tokens or 2 * max_seq_len,
        ),
        "checkpoint_loader": BeamNvfp4CheckpointLoader(),
        "reasoning_parser": PARSER_KEY,
        "enable_iter_perf_stats": iter_stats,
    }
    if draft_checkpoint is not None:
        args["speculative_config"] = dflash_config(draft_checkpoint)
    return args


def main(argv: Optional[list] = None) -> None:
    ap = argparse.ArgumentParser(description="Serve the Beam NVFP4 full model (OpenAI API).")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--served-model-name", default=None)
    ap.add_argument(
        "--chat-template", default=None, help="default: <checkpoint>/chat_template.jinja"
    )
    ap.add_argument("--max-seq-len", type=int, default=69632)
    ap.add_argument("--max-num-tokens", type=int, default=8192)
    ap.add_argument("--max-batch-size", type=int, default=1)
    ap.add_argument("--kv-cache-max-tokens", type=int, default=None)
    ap.add_argument("--iter-stats", action="store_true")
    ap.add_argument(
        "--draft-checkpoint",
        default=None,
        help="DFlash 2 drafter directory; without it, speculation is off",
    )
    args = ap.parse_args(argv)

    from tensorrt_llm.commands.serve import launch_server

    # The capacity is the explicit max_tokens above; no estimation pass on a dry-run cache.
    os.environ.setdefault("TRTLLM_SKIP_KV_CACHE_ESTIMATION", "1")
    template = args.chat_template or os.path.join(args.checkpoint, "chat_template.jinja")
    launch_server(
        args.host,
        args.port,
        llm_args(
            args.checkpoint,
            max_seq_len=args.max_seq_len,
            max_num_tokens=args.max_num_tokens,
            max_batch_size=args.max_batch_size,
            kv_cache_max_tokens=args.kv_cache_max_tokens,
            iter_stats=args.iter_stats,
            draft_checkpoint=args.draft_checkpoint,
        ),
        chat_template=template,
        served_model_name=args.served_model_name,
        multi_frontend_enabled=False,
    )


if __name__ == "__main__":
    main()
