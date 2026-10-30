# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Routed ModelingV2 target: the Beam NVFP4 full model on the native PyTorch runtime.

``ModelingV2BeamFullModelNvfp4Sm103Tp4`` is what ``AutoModelForCausalLM`` builds for a
``BeamForCausalLM`` NVFP4 checkpoint on four sm_103 ranks (attention TP4, MoE TP4/EP1, PP1, no
attention DP), without speculative decoding or with a DFlash 2 drafter, when ``modeling_v2``
routes it. It owns exactly one target parameter set: the full-model component
(``model.BeamFullModelNvfp4Sm103Tp4``), constructed from the checkpoint's own ``config.json``
and filled by ``weights.BeamNvfp4CheckpointLoader`` through the component's transactional
``load``; there is no second embedding, head or expert copy.

Each engine step is the native runtime's batch of one or more requests (up to the engine's
``max_batch_size``), context requests first, dispatched once on the phase:

* ``PrefillTarget``: a step with at least one context request (its prompt tokens, after any
  reused prefix), possibly followed by generation requests; the logits of each request's
  last row, or of every row when the engine asks for context logits;
* ``DecodeTarget``: a step of generation requests, one token each; their next-token logits.

Both return FP32 softcapped logits ``[rows, 151936]`` in request order; the engine's
TorchSampler selects tokens from them. The KV cache is the engine's ``KVCacheManagerV2``; the
component validates its layers, windows and the step's metadata before any kernel runs. Eager
execution only: the expert provider synchronizes with the host, so CUDA graphs and the overlap
scheduler are not supported.

With a DFlash speculative config (``routing.route`` admits only the DFlash 2 drafter on the
TRTLLM draft-attention backend) the target composes the native one-model pieces rather than a
second scheduler, verifier or cache: the drafter ``get_draft_model`` builds from the drafter
checkpoint (``modeling_dflash.DFlashForCausalLM``), the ``DFlashWorker`` ``get_spec_worker``
builds, and the drafter's layers of the engine's ``KVCacheManagerV2`` (one manager for the
target and draft layers, with one page table per request). Every step is then a
``VerifyTarget`` step: one target forward over the context rows and each generation request's
``P = runtime_draft_len + 1`` rows (the pending token and its proposals), with the completed
block outputs of the drafter's capture layers copied into the step's DFlash capture buffer and
the FP32 softcapped logits of the verification rows (each context's last row, every generation
row) handed to the worker, which accepts the proposals equal to the target's argmax, drafts the
next block and returns the native result dictionary. The drafter's logits processor is the
checkpoint's ``final_logit_softcapping``, applied once (``DFlashSoftcapLogitsProcessor``); its
embedding and vocabulary-parallel head are views of the target's own TP vocabulary slices.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import types
from typing import Optional

import torch
from torch import nn

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantAlgo

from .layout import (
    EMBED,
    HEAD,
    check_checkpoint,
    check_drafter_config,
    checkpoint_dir_of,
    read_config,
)
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

    def _run(self, attn_metadata, input_ids, position_ids, logit_rows, **kwargs):
        kind = "batched" if attn_metadata.num_seqs > 1 else "single-request"
        trace = None if kind in self.traced else []
        out = self.model(
            input_ids, position_ids, attn_metadata, logit_rows=logit_rows, trace=trace, **kwargs
        )
        if trace is not None:
            self.traced.add(kind)
            _log_trace(f"{kind} {self.phase}", trace)
        return out


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


class VerifyTarget(_Target):
    """A speculative step: contexts first, then ``P`` verification rows per generation request.

    Copies the completed block output of every capture layer into the step's DFlash capture
    buffer (``maybe_capture_hidden_states`` keeps only its own layers) and returns the logits
    of the verification rows, which must be the engine's ``gather_ids``, with the last block's
    output.
    """

    phase = "verification"

    def forward(
        self,
        attn_metadata,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        spec_metadata,
        *,
        extra_kv_tokens: int,
    ) -> tuple:
        rows_per_generation = spec_metadata.runtime_draft_len + 1
        if spec_metadata.runtime_tokens_per_gen_step != rows_per_generation:
            raise ValueError(
                f"the step verifies {spec_metadata.runtime_tokens_per_gen_step} rows per "
                f"generation request but drafts {spec_metadata.runtime_draft_len}; a linear "
                "draft verifies its proposals and the pending token"
            )
        return self._run(
            attn_metadata,
            input_ids,
            position_ids,
            "verification",
            generation_rows=rows_per_generation,
            extra_kv_tokens=extra_kv_tokens,
            block_output=spec_metadata.maybe_capture_hidden_states,
            gather_ids=spec_metadata.gather_ids,
            return_hidden=True,
        )


# Version of the disaggregated handoff: the cache identity both workers compare and the
# context-request record (``ModelingV2BeamFullModelNvfp4Sm103Tp4.export_disagg_context_request``),
# whose drafter pages travel as draft layers of the target's KV cache manager.
DISAGG_HANDOFF_VERSION = 2

# Checkpoint files whose bytes define the target's weights layout, quantization recipe and
# vocabulary; hashed once per worker, never the weights themselves.
_TARGET_MANIFEST_FILES = ("config.json", "hf_quant_config.json", "model.safetensors.index.json")
_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "generation_config.json")


def _canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _files_sha256(directory: str, names) -> str:
    """One digest over the named files of ``directory`` (name, size and bytes of each)."""
    h = hashlib.sha256()
    for name in names:
        with open(os.path.join(directory, name), "rb") as f:
            data = f.read()
        h.update(name.encode() + b"\0" + str(len(data)).encode() + b"\0" + data)
    return h.hexdigest()


def _safetensors_manifest_sha256(directory: str) -> str:
    """Digest of a checkpoint's safetensors manifests: the index when there is one, else each
    file's size and header (tensor names, dtypes, shapes and offsets), not the tensor bytes."""
    names = sorted(n for n in os.listdir(directory) if n.endswith(".safetensors"))
    if os.path.exists(os.path.join(directory, "model.safetensors.index.json")):
        return _files_sha256(directory, ["model.safetensors.index.json"])
    h = hashlib.sha256()
    for name in names:
        path = os.path.join(directory, name)
        with open(path, "rb") as f:
            (header_len,) = struct.unpack("<Q", f.read(8))
            header = f.read(header_len)
        h.update(name.encode() + b"\0" + str(os.path.getsize(path)).encode() + b"\0" + header)
    return h.hexdigest()


def kv_cache_quant_algo(model_config: ModelConfig) -> Optional[QuantAlgo]:
    """The configured KV-cache algorithm: None (BF16) or FP8; raise ValueError on any other."""
    algo = getattr(model_config.quant_config, "kv_cache_quant_algo", None)
    if algo is None:
        return None
    if QuantAlgo(algo) != QuantAlgo.FP8:
        raise ValueError(f"the Beam NVFP4 target keeps a BF16 or FP8 KV cache, not {algo}")
    return QuantAlgo.FP8


class ModelingV2BeamFullModelNvfp4Sm103Tp4(nn.Module):
    """The engine-facing Beam NVFP4 full model (one rank of four).

    The KV cache is BF16, or FP8 when the configuration's ``kv_cache_quant_algo`` is FP8
    (``kv_cache_config.dtype="fp8"``): then the target's and the drafter's caches both hold E4M3
    codes, each layer with its own FP32 scale from the scale table ``load_checkpoint`` takes.
    """

    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.model_config = model_config
        mapping = model_config.mapping
        self.checkpoint_dir, checkpoint_config = _checkpoint_config(model_config.pretrained_config)
        self.kv_cache_quant_algo = kv_cache_quant_algo(model_config)
        self.model = BeamFullModelNvfp4Sm103Tp4(
            checkpoint_config,
            mapping,
            torch.device("cuda", torch.cuda.current_device()),
            max_tokens=MAX_NUM_TOKENS,
        )
        self._prefill = PrefillTarget(self.model)
        self._decode = DecodeTarget(self.model)
        self.load_record: Optional[dict] = None
        self.kv_cache_record: Optional[dict] = None
        # The native one-model pieces, built only for a speculative config.
        self.spec_config = getattr(model_config, "spec_config", None)
        self.draft_config: Optional[ModelConfig] = None
        self.draft_model: Optional[nn.Module] = None
        self.spec_worker: Optional[nn.Module] = None
        self.use_separate_draft_kv_cache = False
        self.extra_kv_tokens = 0
        self.draft_load_record: Optional[dict] = None
        self._disagg_identity: Optional[dict] = None
        if self.spec_config is not None:
            self._build_dflash(model_config, checkpoint_config)
        logger.info(
            f"[modeling_v2] {type(self).__name__}: {NUM_LAYERS} layers, rank {mapping.rank} of "
            f"{mapping.world_size}, max {MAX_NUM_TOKENS} tokens per call, speculative decoding "
            f"{'DFlash' if self.spec_worker is not None else 'off'}, KV cache "
            f"{'FP8' if self.kv_cache_quant_algo == QuantAlgo.FP8 else 'BF16'}"
        )

    def _build_dflash(self, model_config: ModelConfig, checkpoint_config: dict) -> None:
        """Compose the native DFlash drafter, worker and draft-cache choice on this target."""
        from tensorrt_llm._torch.models.modeling_dflash import (
            DFlashSoftcapLogitsProcessor,
            dflash_final_logit_softcap,
        )
        from tensorrt_llm._torch.models.modeling_speculative import (
            _set_draft_kv_cache_quant_algo,
            external_drafter_config_kwargs,
            get_draft_model,
        )
        from tensorrt_llm._torch.speculative import (
            get_num_extra_kv_tokens,
            get_spec_worker,
            should_use_separate_draft_kv_cache,
        )

        spec_config = self.spec_config
        if not spec_config.spec_dec_mode.is_dflash():
            raise ValueError(
                f"the Beam NVFP4 target composes a DFlash drafter only, got "
                f"{type(spec_config).__name__}"
            )
        self._verify = VerifyTarget(self.model)
        self.draft_config = ModelConfig.from_pretrained(
            spec_config.speculative_model,
            **external_drafter_config_kwargs(model_config, spec_config),
        )
        _set_draft_kv_cache_quant_algo(self.draft_config, model_config)
        self.draft_config.extra_attrs = model_config.extra_attrs
        check_drafter_config(
            read_config(spec_config.speculative_model),
            checkpoint_config,
            spec_config.target_layer_ids,
        )
        draft_algo = self.draft_config.quant_config.kv_cache_quant_algo
        if (None if draft_algo is None else QuantAlgo(draft_algo)) != self.kv_cache_quant_algo:
            raise ValueError(
                f"the drafter's KV cache would be {draft_algo or 'BF16'} and the target's "
                f"{self.kv_cache_quant_algo or 'BF16'}; the Beam DFlash deployment keeps both "
                "caches in one precision"
            )
        # The draft pages live in a KV cache manager apart from the drafter (the target's,
        # when it owns the draft layers), which the worker reads through its own pool views
        # (TRTLLM draft attention).
        self.use_separate_draft_kv_cache = should_use_separate_draft_kv_cache(spec_config)
        self.draft_model = get_draft_model(model_config, self.draft_config, None, self)
        self.draft_model.logits_processor = DFlashSoftcapLogitsProcessor(
            dflash_final_logit_softcap(self.draft_config.pretrained_config)
        )
        if self.kv_cache_quant_algo == QuantAlgo.FP8:
            reason = self.draft_model.fp8_kv_cache_unsupported_reason()
            if reason is not None:
                raise ValueError(f"the drafter cannot keep an FP8 KV cache: {reason}")
        self.spec_worker = get_spec_worker(
            spec_config,
            model_config,
            model_config.mapping,
            use_separate_draft_kv_cache=self.use_separate_draft_kv_cache,
        )
        if self.spec_worker is None:
            raise ValueError(f"no speculative worker serves {spec_config.spec_dec_mode!r}")
        self.spec_worker.set_draft_model(self.draft_model)
        self.extra_kv_tokens = get_num_extra_kv_tokens(spec_config)

    @property
    def config(self):
        return self.model_config.pretrained_config

    @property
    def vocab_size_padded(self) -> int:
        return self.model.vocab_size

    def load_checkpoint(self, checkpoint_dir: str, kv_cache_scales: Optional[dict] = None) -> dict:
        """Fill every parameter from ``checkpoint_dir`` (collective; see the component's load).

        ``kv_cache_scales`` is the FP8 KV cache's scale table (``weights.read_kv_cache_scales``):
        ``"target"`` holds one scale per logical layer of the target and, with a drafter,
        ``"draft"`` one per draft layer. It is required with an FP8 KV cache and refused with a
        BF16 one; the checkpoint holds no KV-cache scales of its own.
        """
        if os.path.realpath(checkpoint_dir) != os.path.realpath(self.checkpoint_dir):
            raise ValueError(
                f"weights from {checkpoint_dir!r} do not belong to the configuration of "
                f"{self.checkpoint_dir!r}"
            )
        self._check_kv_cache_scales(kv_cache_scales)
        self.load_record = self.model.load(checkpoint_dir)
        self._apply_kv_cache_scales(kv_cache_scales)
        return self.load_record

    def _check_kv_cache_scales(self, table: Optional[dict]) -> None:
        if self.kv_cache_quant_algo != QuantAlgo.FP8:
            if table is not None:
                raise ValueError(
                    "a KV-cache scale table was given, but the KV cache is BF16; select the FP8 "
                    "KV cache (kv_cache_config.dtype='fp8') to use it"
                )
            return
        if table is None:
            raise ValueError(
                "the FP8 KV cache needs a scale table for the target"
                f"{' and the drafter' if self.draft_model is not None else ''}; the checkpoint "
                "holds no KV-cache scales and none are assumed"
            )
        sections = ["target"] + (["draft"] if self.draft_model is not None else [])
        missing = [s for s in sections if not table.get(s)]
        if missing:
            raise ValueError(f"the KV-cache scale table has no {missing} scales")

    def _apply_kv_cache_scales(self, table: Optional[dict]) -> None:
        if self.kv_cache_quant_algo != QuantAlgo.FP8:
            self.kv_cache_record = {"dtype": "bf16"}
            return
        self.model.set_kv_cache_fp8(table["target"])
        if self.draft_model is not None:
            self.draft_model.set_kv_cache_scales(table["draft"])
        self.kv_cache_record = {
            "dtype": "fp8",
            "format": "float8_e4m3fn, one FP32 scale per logical layer shared by K and V",
            "source": table.get("source"),
            "source_sha256": table.get("sha256"),
            "target_scales": list(self.model.kv_cache_scales),
            "draft_scales": (
                list(self.draft_model.kv_cache_scales) if self.draft_model is not None else None
            ),
        }
        logger.info(f"[modeling_v2] FP8 KV cache: {self.kv_cache_record}")

    def shared_vocabulary_modules(self) -> tuple:
        """Native ``Embedding`` and vocabulary-parallel ``LMHead`` over the target's own slices.

        Each module's weight is the component's registered rank-local parameter itself
        (``model.embed_tokens.weight`` / ``lm_head.weight``, rows ``[rank * V/4, (rank + 1) *
        V/4)``), not a copy: the embedding masks ids outside the slice and all-reduces, the
        head projects to the rank's vocabulary shard without gathering or capping. Requires a
        completed target load.
        """
        from tensorrt_llm._torch.modules.embedding import Embedding, LMHead
        from tensorrt_llm._torch.modules.linear import TensorParallelMode

        if not self.model.usable:
            raise RuntimeError("the target's embedding and head are shared after its load")
        mapping = self.model_config.mapping
        vocab, hidden = self.model.vocab_size, self.model.hidden
        modules = (
            Embedding(
                vocab,
                hidden,
                dtype=torch.bfloat16,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=False,
            ),
            LMHead(
                vocab,
                hidden,
                dtype=torch.bfloat16,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=False,
            ),
        )
        lo, hi = self.model.vocab_interval
        for module, name in zip(modules, (EMBED, HEAD)):
            weight = self.model.tensor(name)
            if tuple(module.weight.shape) != tuple(weight.shape):
                raise ValueError(
                    f"{type(module).__name__} holds {tuple(module.weight.shape)} rows per rank; "
                    f"the target's {name} slice is {tuple(weight.shape)}"
                )
            module.weight = weight
        embed, head = modules
        if (embed.vocab_start_index, embed.vocab_end_index) != (lo, hi):
            raise ValueError(
                f"the embedding owns ids [{embed.vocab_start_index}, {embed.vocab_end_index}) on "
                f"this rank; the target's slice is [{lo}, {hi})"
            )
        return embed, head

    def load_draft_weights(self, weights: dict, weight_mapper=None) -> None:
        """Load the drafter's own tensors, then give it the target's embedding and head.

        The drafter checkpoint carries neither (``DFlashForCausalLM.WEIGHTS_SHARED_WITH_TARGET``)
        and its own loader rejects every other missing tensor; the shared modules are views of
        the target's slices (``shared_vocabulary_modules``), handed over through the drafter's
        native ``load_weights_from_target_model``.
        """
        if self.draft_model is None:
            raise RuntimeError("this target was built without a drafter")
        self.draft_model.load_weights(weights=weights, weight_mapper=weight_mapper)
        embed, head = self.shared_vocabulary_modules()
        self.draft_model.load_weights_from_target_model(
            types.SimpleNamespace(model=types.SimpleNamespace(embed_tokens=embed), lm_head=head)
        )
        draft_embed = self.draft_model.draft_model_full.model.embed_tokens
        self.draft_load_record = {
            "tensors": len(weights),
            "embed_shares_target_storage": draft_embed.weight.data_ptr()
            == self.model.tensor(EMBED).data_ptr(),
            "head_shares_target_storage": self.draft_model.lm_head.weight.data_ptr()
            == self.model.tensor(HEAD).data_ptr(),
            "vocab_interval": list(self.model.vocab_interval),
        }
        logger.info(f"[modeling_v2] DFlash drafter loaded: {self.draft_load_record}")

    # ------------------------------------------------------------ disaggregated handoff

    @property
    def transfers_draft_kv_cache(self) -> bool:
        """Whether a disaggregated handoff must carry the drafter's context: its K/V, built
        from the target's prefill, live in the draft layers ``NUM_LAYERS..`` of the target's
        KV cache manager and move with the target's pages."""
        return self.spec_worker is not None and self.use_separate_draft_kv_cache

    def disagg_cache_identity(self) -> dict:
        """What both disaggregated workers must agree on before any KV moves between them.

        Rank-independent digests of: the target checkpoint's manifests (configuration,
        quantization recipe, weight index), the tokenizer files, the cache layout (layers,
        windows, heads per rank, head size, cache dtype), the per-layer FP8 scales actually
        applied (FP32 values, for target and drafter), the speculation settings, the drafter
        checkpoint's manifests and the handoff contract (version, transferred layers: the
        target's and, with a drafter, the draft layers that follow them in the one KV cache
        manager). Digests only, so no scale or weight leaves the worker.
        Requires a completed load; computed once.
        """
        if self._disagg_identity is not None:
            return self._disagg_identity
        if self.load_record is None or self.kv_cache_record is None:
            raise RuntimeError("the disaggregated cache identity needs a completed target load")
        component = self.model
        layout = {
            "layers": NUM_LAYERS,
            "world_size": self.model_config.mapping.world_size,
            "kv_heads_per_rank": component.local_kv_heads,
            "head_dim": component.head_dim,
            "windows": [
                component.window if component.schedule.is_sliding(layer) else None
                for layer in range(NUM_LAYERS)
            ],
            "kv_cache_dtype": self.kv_cache_record["dtype"],
        }
        scales = {
            key: None if values is None else [float(v).hex() for v in values]
            for key, values in (
                ("target", self.kv_cache_record.get("target_scales")),
                ("draft", self.kv_cache_record.get("draft_scales")),
            )
        }
        parts = {
            "target_manifest": _files_sha256(self.checkpoint_dir, _TARGET_MANIFEST_FILES),
            "tokenizer": _files_sha256(
                self.checkpoint_dir,
                [
                    n
                    for n in _TOKENIZER_FILES
                    if os.path.exists(os.path.join(self.checkpoint_dir, n))
                ],
            ),
            "layout": _canonical_sha256(layout),
            "kv_cache_scales": _canonical_sha256(scales),
            "handoff": _canonical_sha256(
                {
                    "version": DISAGG_HANDOFF_VERSION,
                    "draft_layers": [
                        NUM_LAYERS,
                        NUM_LAYERS + int(self.draft_config.pretrained_config.num_hidden_layers),
                    ]
                    if self.transfers_draft_kv_cache
                    else None,
                }
            ),
        }
        if self.spec_config is None:
            parts["speculation"] = _canonical_sha256(None)
        else:
            spec = self.spec_config
            parts["speculation"] = _canonical_sha256(
                {
                    "mode": "dflash",
                    "max_draft_len": int(spec.max_draft_len),
                    "tokens_per_gen_step": int(spec.tokens_per_gen_step),
                    "target_layer_ids": [int(i) for i in spec.target_layer_ids],
                    "mask_token_id": getattr(spec, "mask_token_id", None),
                }
            )
            drafter_dir = spec.speculative_model
            parts["drafter_manifest"] = _files_sha256(drafter_dir, ["config.json"])
            parts["drafter_weights"] = _safetensors_manifest_sha256(drafter_dir)
        self._disagg_identity = {"version": DISAGG_HANDOFF_VERSION, "parts": parts}
        logger.info(f"[modeling_v2] disaggregated cache identity: {self._disagg_identity}")
        return self._disagg_identity

    def _disagg_identity_digest(self) -> str:
        return _canonical_sha256(self.disagg_cache_identity())

    def export_disagg_context_request(self, req) -> dict:
        """The handoff record a finished context-only request leaves its generation worker.

        Called by the cache transceiver after the context forward, before the request's
        drafter slot can be recycled; it travels in the request's auxiliary slot of the same
        transfer session as the KV pages (``native.auxiliary.encode_handoff_record``). It
        binds the transfer to this worker's identity, the request's transfer id and prompt
        length and, with a drafter, the draft-context cursor whose K/V travel as the draft
        layers' pages of the same KV cache manager (``DFlashWorker.export_disagg_context``).
        """
        from tensorrt_llm._torch.disaggregation.base.transfer import get_unique_rid

        return {
            "version": DISAGG_HANDOFF_VERSION,
            "identity": self._disagg_identity_digest(),
            "transfer_id": int(get_unique_rid(req)),
            "prompt_len": int(req.prompt_len),
            "draft": (
                self.spec_worker.export_disagg_context(req.py_request_id)
                if self.spec_worker is not None
                else None
            ),
        }

    def validate_disagg_generation_request(self, req) -> Optional[str]:
        """None when ``req``'s received handoff record fits this worker, else why it does not.

        Runs on the generation worker once the transfer session completes and before the
        request is handed to generation: the record must come from a worker with this
        identity, belong to this request's transfer and prompt and, with a drafter, describe
        a draft context covering the whole prompt. Anything else fails the request before
        any received state is read.
        """
        record = getattr(req, "py_disagg_handoff", None)
        if record is None:
            return "no context handoff record arrived with the request"
        if record.get("version") != DISAGG_HANDOFF_VERSION:
            return (
                f"handoff record version {record.get('version')!r}, expected "
                f"{DISAGG_HANDOFF_VERSION}"
            )
        identity = record.get("identity") or ""
        if not identity or not self._disagg_identity_digest().startswith(identity):
            return "the context worker's cache identity differs from this worker's"
        params = getattr(req, "context_phase_params", None)
        expected_id = None if params is None else params.req_id
        if record.get("transfer_id") != expected_id:
            return (
                f"the record belongs to transfer {record.get('transfer_id')}, the request to "
                f"{expected_id}"
            )
        if record.get("prompt_len") != req.prompt_len:
            return (
                f"the record describes a {record.get('prompt_len')}-token prompt, the request "
                f"has {req.prompt_len}"
            )
        draft = record.get("draft")
        if self.transfers_draft_kv_cache:
            if draft is None:
                return "the context worker stored no draft context for this request"
            if (
                draft.get("draft_ctx_len") != req.prompt_len
                or draft.get("next_position") != req.prompt_len
            ):
                return (
                    f"the draft context covers {draft.get('draft_ctx_len')} positions (next "
                    f"position {draft.get('next_position')}); the prompt has {req.prompt_len}"
                )
        elif draft is not None:
            return "the context worker drafted but this worker has no drafter"
        return None

    def prepare_disagg_generation_request(self, req) -> None:
        """Hand a received request's draft-context cursor to the DFlash worker.

        Called by the executor once the request's KV (target and draft pages) has arrived and
        before its first generation step; the record was validated on arrival.
        """
        if self.spec_worker is None:
            return
        record = getattr(req, "py_disagg_handoff", None)
        if record is None or record.get("draft") is None:
            raise RuntimeError(
                f"request {req.py_request_id}: no validated draft-context record to import"
            )
        draft = record["draft"]
        self.spec_worker.import_disagg_context(
            req.py_request_id, draft["draft_ctx_len"], draft["next_position"]
        )

    def release_request_state(self, request_id: int) -> None:
        """Drop a received draft-context cursor that was never bound to a drafter slot."""
        if self.spec_worker is not None:
            self.spec_worker.release_request_state(request_id)

    def forward(
        self,
        attn_metadata,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        return_context_logits: bool = False,
        spec_metadata=None,
        resource_manager=None,
        **kwargs,
    ):
        if inputs_embeds is not None or input_ids is None:
            raise ValueError("the Beam NVFP4 target takes token ids, not input embeddings")
        if kwargs.get("lora_params") is not None:
            raise ValueError("LoRA is not supported by this target")
        ids = input_ids.reshape(-1).to(torch.int64)
        if self.spec_worker is None:
            if spec_metadata is not None:
                raise ValueError(
                    "speculative metadata reached a Beam target built without a speculative config"
                )
            target = self._decode if attn_metadata.num_contexts == 0 else self._prefill
            return target.forward(
                attn_metadata, ids, position_ids, return_context_logits=return_context_logits
            )
        if spec_metadata is None:
            raise ValueError(
                "the DFlash target verifies every step; the engine passed no speculative metadata"
            )
        logits, hidden = self._verify.forward(
            attn_metadata, ids, position_ids, spec_metadata, extra_kv_tokens=self.extra_kv_tokens
        )
        return self.spec_worker(
            input_ids=ids,
            position_ids=position_ids,
            hidden_states=hidden,
            logits=logits,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            draft_model=self.draft_model,
            resource_manager=resource_manager,
        )


__all__ = [
    "DecodeTarget",
    "ModelingV2BeamFullModelNvfp4Sm103Tp4",
    "PrefillTarget",
    "VerifyTarget",
]
