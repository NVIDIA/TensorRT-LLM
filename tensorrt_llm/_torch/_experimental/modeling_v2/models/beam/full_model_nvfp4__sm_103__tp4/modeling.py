# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Routed ModelingV2 target: the Beam NVFP4 full model on the native PyTorch runtime.

``ModelingV2BeamFullModelNvfp4Sm103Tp4`` is what ``AutoModelForCausalLM`` builds for a
``BeamForCausalLM`` NVFP4 checkpoint on four sm_103 ranks (attention TP4, MoE TP4/EP1, PP1, no
attention DP, no speculative decoding) when ``modeling_v2`` routes it. It owns exactly one
parameter set: the full-model component (``model.BeamFullModelNvfp4Sm103Tp4``), constructed
from the checkpoint's own ``config.json`` and filled by ``weights.BeamNvfp4CheckpointLoader``
through the component's transactional ``load``; there is no second embedding, head or expert
copy.

Each engine step is the native runtime's batch of one or more requests (up to the engine's
``max_batch_size``), context requests first, dispatched once on the phase:

* ``PrefillTarget``: a step with at least one context request (its prompt tokens, after any
  reused prefix), possibly followed by generation requests; the logits of each request's
  last row, or of every row when the engine asks for context logits;
* ``DecodeTarget``: a step of generation requests, one token each; their next-token logits.

Both return FP32 softcapped logits ``[rows, 151936]`` in request order; the engine's
TorchSampler selects tokens from them. The KV cache is the engine's ``KVCacheManagerV2``; the component validates
its layers, windows and the step's metadata before any kernel runs. Eager execution only:
the expert provider synchronizes with the host, so CUDA graphs and the overlap scheduler are
not supported.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
from torch import nn

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.logger import logger

from .layout import check_checkpoint, checkpoint_dir_of
from .model import MAX_NUM_TOKENS, NUM_LAYERS, BeamFullModelNvfp4Sm103Tp4


def _checkpoint_config(pretrained_config) -> tuple:
    """The checkpoint directory and its ``config.json`` mapping, read as stored and validated.

    ``layout.check_checkpoint`` is the validator routing applies too, so a checkpoint routing
    accepted is never refused here and one it would refuse never reaches construction.
    """
    checkpoint_dir = checkpoint_dir_of(pretrained_config)
    if not checkpoint_dir:
        raise ValueError(
            "the Beam NVFP4 target is built from the checkpoint's own config.json; the "
            "configuration names no checkpoint directory"
        )
    return checkpoint_dir, check_checkpoint(checkpoint_dir)


def _log_trace(phase: str, trace: list) -> None:
    """Record the stages a traced step executed, in what order, and its requests' spans."""
    layers = [r["layer"] for r in trace if r["stage"] == "layer"]
    stages = [r["stage"] for r in trace]
    spans = trace[0].get("requests", [])
    logger.info(
        f"[modeling_v2] first {phase} step: {len(spans)} requests (past, new tokens) {spans}, "
        f"stages {stages[0]}..{stages[-1]}, {len(layers)} logical "
        f"layers in order 0..{NUM_LAYERS - 1}: {layers == list(range(NUM_LAYERS))}, "
        f"{sum(r.get('mlp') == 'sparse' for r in trace)} sparse, "
        f"{sum(r.get('attention') == 'full' for r in trace)} full-attention"
    )


class _Target:
    """One engine step through all 52 layers; traces the first single- and multi-request step."""

    phase = ""

    def __init__(self, model: BeamFullModelNvfp4Sm103Tp4):
        self.model = model
        self.traced = set()

    def _run(self, attn_metadata, input_ids, position_ids, logit_rows) -> torch.Tensor:
        kind = "batched" if attn_metadata.num_seqs > 1 else "single-request"
        trace = None if kind in self.traced else []
        logits = self.model(
            input_ids, position_ids, attn_metadata, logit_rows=logit_rows, trace=trace
        )
        if trace is not None:
            self.traced.add(kind)
            _log_trace(f"{kind} {self.phase}", trace)
        return logits


class PrefillTarget(_Target):
    """A step with a context request, possibly followed by generation requests."""

    phase = "context"

    def forward(
        self,
        attn_metadata,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        *,
        return_context_logits: bool,
    ) -> torch.Tensor:
        rows = "all" if return_context_logits else None
        return self._run(attn_metadata, input_ids, position_ids, rows)


class DecodeTarget(_Target):
    """A step of generation requests: one token each."""

    phase = "generation"

    def forward(
        self,
        attn_metadata,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        *,
        return_context_logits: bool,
    ) -> torch.Tensor:
        return self._run(attn_metadata, input_ids, position_ids, None)


class ModelingV2BeamFullModelNvfp4Sm103Tp4(nn.Module):
    """The engine-facing Beam NVFP4 full model (one rank of four)."""

    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.model_config = model_config
        mapping = model_config.mapping
        self.checkpoint_dir, checkpoint_config = _checkpoint_config(model_config.pretrained_config)
        self.model = BeamFullModelNvfp4Sm103Tp4(
            checkpoint_config,
            mapping,
            torch.device("cuda", torch.cuda.current_device()),
            max_tokens=MAX_NUM_TOKENS,
        )
        self._prefill = PrefillTarget(self.model)
        self._decode = DecodeTarget(self.model)
        self.load_record: Optional[dict] = None
        logger.info(
            f"[modeling_v2] {type(self).__name__}: {NUM_LAYERS} layers, rank {mapping.rank} of "
            f"{mapping.world_size}, max {MAX_NUM_TOKENS} tokens per call"
        )

    @property
    def config(self):
        return self.model_config.pretrained_config

    @property
    def vocab_size_padded(self) -> int:
        return self.model.vocab_size

    def load_checkpoint(self, checkpoint_dir: str) -> dict:
        """Fill every parameter from ``checkpoint_dir`` (collective; see the component's load)."""
        if os.path.realpath(checkpoint_dir) != os.path.realpath(self.checkpoint_dir):
            raise ValueError(
                f"weights from {checkpoint_dir!r} do not belong to the configuration of "
                f"{self.checkpoint_dir!r}"
            )
        self.load_record = self.model.load(checkpoint_dir)
        return self.load_record

    def forward(
        self,
        attn_metadata,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        return_context_logits: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        if inputs_embeds is not None or input_ids is None:
            raise ValueError("the Beam NVFP4 target takes token ids, not input embeddings")
        if kwargs.get("spec_metadata") is not None or kwargs.get("lora_params") is not None:
            raise ValueError("speculative decoding and LoRA are not supported by this target")
        ids = input_ids.reshape(-1).to(torch.int64)
        target = self._decode if attn_metadata.num_contexts == 0 else self._prefill
        return target.forward(
            attn_metadata, ids, position_ids, return_context_logits=return_context_logits
        )


__all__ = ["DecodeTarget", "ModelingV2BeamFullModelNvfp4Sm103Tp4", "PrefillTarget"]
