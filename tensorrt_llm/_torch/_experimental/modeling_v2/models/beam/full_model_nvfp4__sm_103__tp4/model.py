# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Beam full model, NVFP4 experts / sm_103 / tp4: token ids to logits through all 52 layers.

The packed NVFP4 checkpoint-layout component (``component``) with a full-depth forward. Every
operation is the component's own: the masked embedding with its all-reduce, each logical
layer's ``block`` (attention on TRTLLM attention and the call's
``KVCacheManagerV2`` slot for that layer, the dense SwiGLU on layers 0-2, the replicated FP32
router and packed W4A4 experts on layers 3-51, the four sandwich norms and the depth residuals),
the final norm, the untied head with its vocabulary all-gather and the FP32 softcap. This class
adds the order and the contract of one call::

    x = embed(input_ids)                         replicated [T, 5120] bf16
    for layer in 0 .. 51: x = block(layer, x)    in logical order, each exactly once
    logits = head(final_norm(x[rows]))           fp32 [len(rows), 151936], softcapped once

One call is one engine step of one or more requests (``_attention.check_batch_metadata``):
the step's context requests first, then its generation requests, their new tokens
concatenated into the call's T rows. A context request appends its ``n`` tokens after the
``past`` tokens the cache already holds for it (none, a reused prefix or earlier chunks), a
generation request appends ``generation_rows`` tokens: one, or in a speculative
verification step ``P`` (its pending token and ``P - 1`` proposals). Each request's
``position_ids`` must be exactly ``past .. past + n - 1``. ``logit_rows`` selects the hidden
rows that reach the final norm and the head before either runs: ``None`` (default) is the
last row of each request, in request order -- the next-token logits a sampler needs --
``"all"`` every row, ``"verification"`` the last row of each context request and every row
of each generation request (what a speculative verifier reads), or explicit row indices.
``block_output(layer, x)``, when given, receives each logical layer's completed block output
``x`` ``[T, hidden]`` (the residual stream after the block, before the next block or the
final norm) as soon as the block returns.

The cache must be a ``KVCacheManagerV2`` of the model's KV-cache dtype (BF16, or FP8 with the
per-layer scales of ``set_kv_cache_fp8``) holding all 52 logical layers in logical order
with 2 kv heads of 128 per rank, a 512-token window on every sliding layer and the whole
sequence (``max_seq_len``) on the full-attention layers 5, 11, 17, 23, 29, 35, 41, 47 and 51
(``attention_windows``). Inputs, positions, metadata and cache are validated before any kernel
runs; a call with T above ``max_tokens`` is rejected (the default ``MAX_NUM_TOKENS`` matches
the supported deployment's native ``max_num_tokens``).

Validation runs once per call: the step's metadata and request lengths, the token-id range (one
device comparison read back), the cache manager's geometry once per manager. The expert
dispatch synchronizes with the host per sparse layer. Diagnostic mode (``diagnostics``,
``TRTLLM_MODELING_V2_VALIDATE``) adds the position and ``gather_ids`` value comparisons, the
per-layer metadata re-checks and the per-sparse-layer router agreement across ranks.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence, Union

import torch

from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm.mapping import Mapping as ParallelMapping

from . import _attention
from .component import BeamCheckpointLayoutNvfp4Sm103Tp4

#: Logical decoder layers, all executed by every call.
NUM_LAYERS = 52
#: Tokens per call of the supported deployment (its native ``max_num_tokens``).
MAX_NUM_TOKENS = 8192
#: KV cache page size of the supported deployment.
TOKENS_PER_BLOCK = 32

LogitRows = Union[None, str, Sequence[int]]


class BeamFullModelNvfp4Sm103Tp4(BeamCheckpointLayoutNvfp4Sm103Tp4):
    """All 52 layers of the Beam model with packed NVFP4 experts on one of four TP ranks.

    ``config`` is the checkpoint's config.json (a mapping) and ``mapping`` this rank's
    ``tensorrt_llm.mapping.Mapping`` (world size 4, TP4, PP1, MoE TP4/EP1, no attention DP);
    every logical layer is held. ``load(checkpoint_dir)`` (collective) fills the weights; the
    forward is usable only after every rank completed it.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        mapping: ParallelMapping,
        device: Optional[torch.device] = None,
        *,
        max_tokens: int = MAX_NUM_TOKENS,
    ):
        super().__init__(config, mapping, layers=None, device=device, max_tokens=max_tokens)
        if self.layers != tuple(range(NUM_LAYERS)):
            raise ValueError(f"the full model holds logical layers 0..{NUM_LAYERS - 1}")
        self.vocab_size = config["vocab_size"]
        # The last manager check_cache_manager accepted: its layers, heads, windows and dtype
        # are fixed at its construction, so a call on the same manager need not re-read them.
        self._checked_manager = None

    def attention_windows(self, max_seq_len: int) -> list:
        """Per logical layer: the attention window its cache must keep (512 or ``max_seq_len``)."""
        return [
            self.window if self.schedule.is_sliding(layer) else max_seq_len
            for layer in range(NUM_LAYERS)
        ]

    def check_cache_manager(self, manager) -> None:
        """Raise ValueError unless ``manager`` holds every logical layer as this model needs.

        Required: the model's KV-cache dtype (BF16, or FP8 after ``set_kv_cache_fp8``),
        ``head_dim`` 128, 2 kv heads per rank on every layer, logical layers 0..51 in local
        slots 0..51, the sliding window on sliding layers and the whole sequence on
        full-attention layers. A manager that also owns the drafter's layers (one manager for
        the target and draft KV) holds exactly those draft layers after the target's, as
        layers 52.. in slots 52..; the drafter checks their own geometry. Reads host-side
        manager state only.
        """
        offsets = getattr(manager, "layer_offsets", None)
        draft_layers = tuple(getattr(manager, "draft_layer_ids", None) or ())
        num_layers = NUM_LAYERS + len(draft_layers)
        if (
            not isinstance(offsets, Mapping)
            or dict(offsets) != {layer: layer for layer in range(num_layers)}
            or draft_layers != tuple(range(NUM_LAYERS, num_layers))
        ):
            held = sorted(offsets) if isinstance(offsets, Mapping) else None
            raise ValueError(
                f"the KV cache manager must hold logical layers 0..{NUM_LAYERS - 1} in slots "
                f"0..{NUM_LAYERS - 1}, followed only by draft layers; it holds {held} with "
                f"draft layers {list(draft_layers)}"
            )
        heads = list(getattr(manager, "num_kv_heads_per_layer", None) or [])
        if (
            manager.dtype != self.kv_cache_dtype
            or manager.head_dim != self.head_dim
            or len(heads) != num_layers
            or heads[:NUM_LAYERS] != [self.local_kv_heads] * NUM_LAYERS
        ):
            raise ValueError(
                f"the KV cache must be {self.kv_cache_dtype.name} with {self.local_kv_heads} kv "
                f"heads of {self.head_dim} per layer; got {manager.dtype}, head_dim "
                f"{manager.head_dim}"
            )
        windows = list(getattr(manager, "max_attention_window_vec", None) or [])
        if len(windows) != num_layers:
            raise ValueError(f"the KV cache manager states {len(windows)} attention windows")
        for layer, window in enumerate(windows[:NUM_LAYERS]):
            if self.schedule.is_sliding(layer):
                ok = window is not None and int(window) == self.window
            else:
                ok = window is None or int(window) >= manager.max_seq_len
            if not ok:
                kind = "sliding" if self.schedule.is_sliding(layer) else "full-attention"
                raise ValueError(
                    f"logical layer {layer} is a {kind} layer; the cache keeps a window of "
                    f"{window} (max_seq_len {manager.max_seq_len})"
                )

    def _select_rows(
        self, logit_rows: LogitRows, spans: Sequence[tuple], num_contexts: int = 0
    ) -> list:
        num_tokens = sum(n for _, n in spans)
        if logit_rows is None:
            ends, end = [], 0
            for _, n in spans:
                end += n
                ends.append(end - 1)
            return ends
        if isinstance(logit_rows, str):
            if logit_rows == "all":
                return list(range(num_tokens))
            if logit_rows != "verification":
                raise ValueError(
                    "logit_rows must be None, 'all', 'verification' or row indices, got "
                    f"{logit_rows!r}"
                )
            rows, start = [], 0
            for i, (_, n) in enumerate(spans):
                rows.extend([start + n - 1] if i < num_contexts else range(start, start + n))
                start += n
            return rows
        rows = list(logit_rows)
        if (
            not rows
            or any(type(r) is not int or not 0 <= r < num_tokens for r in rows)
            or len(set(rows)) != len(rows)
        ):
            raise ValueError(
                f"logit_rows must be distinct row indices in [0, {num_tokens}), got {logit_rows!r}"
            )
        return rows

    def check_call(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attn_metadata: TrtllmAttentionMetadata,
        *,
        generation_rows: int = 1,
        extra_kv_tokens: int = 0,
    ) -> tuple:
        """Validate one call; return the flat ids, the flat positions and the request spans.

        ``input_ids``: ``[T]`` or ``[1, T]`` torch.int64 ids in ``[0, vocab)`` on the model's
        device, 1 <= T <= ``max_tokens``; ``attn_metadata``: one prepared step of its requests
        on a manager that passes ``check_cache_manager``, ``generation_rows`` rows per
        generation request and ``extra_kv_tokens`` reserved KV slots per request;
        ``position_ids``: an integer tensor of T positions, each request's
        ``past .. past + n - 1`` in request order. Returns the flat ids, the flat int64
        positions and the spans ``[(past, n), ...]`` of ``_attention.check_batch_metadata``.
        Raises ValueError otherwise.

        Every call re-validates its requests (the step's metadata changes every step); the
        manager's fixed geometry is checked once per manager. Comparing the positions' values
        with the spans reads the device back, so it runs in diagnostic mode only.
        """
        if not isinstance(input_ids, torch.Tensor) or input_ids.dtype != torch.int64:
            raise ValueError("input_ids must be a torch.int64 tensor")
        if input_ids.dim() == 2 and input_ids.shape[0] == 1:
            input_ids = input_ids[0]
        if input_ids.dim() != 1 or not 1 <= input_ids.shape[0] <= self.max_tokens:
            raise ValueError(
                f"input_ids must be [T] or [1, T] with 1 <= T <= {self.max_tokens}, "
                f"got {list(input_ids.shape)}"
            )
        if input_ids.device != self.device:
            raise ValueError("input_ids must be on the model's device")
        num_tokens = input_ids.shape[0]
        if (
            not isinstance(position_ids, torch.Tensor)
            or position_ids.is_floating_point()
            or position_ids.dtype == torch.bool
            or position_ids.numel() != num_tokens
            or position_ids.device != self.device
        ):
            raise ValueError(
                f"position_ids must be an integer tensor of {num_tokens} positions on the "
                "model's device"
            )
        if not isinstance(attn_metadata, TrtllmAttentionMetadata):
            raise ValueError(
                f"attn_metadata must be TrtllmAttentionMetadata, got {type(attn_metadata).__name__}"
            )
        manager = attn_metadata.kv_cache_manager
        if manager is None:
            raise ValueError("attn_metadata needs a KV cache manager")
        if manager is not self._checked_manager:
            self.check_cache_manager(manager)
            self._checked_manager = manager
        spans = _attention.check_batch_metadata(
            attn_metadata,
            num_tokens,
            layer_idx=0,
            num_kv_heads=self.local_kv_heads,
            head_dim=self.head_dim,
            generation_rows=generation_rows,
            extra_kv_tokens=extra_kv_tokens,
            kv_cache_dtype=self.kv_cache_dtype,
        )
        positions = position_ids.reshape(-1).to(torch.int64)
        if not self.diagnostics:
            return input_ids, positions, spans
        expected = torch.cat([torch.arange(past, past + n, dtype=torch.int64) for past, n in spans])
        if not torch.equal(positions, expected.to(self.device, non_blocking=True)):
            got, start = positions.cpu(), 0
            for i, (past, n) in enumerate(spans):
                if not torch.equal(got[start : start + n], expected[start : start + n]):
                    break
                start += n
            raise ValueError(
                f"request {i} of the call appends tokens {past}..{past + n - 1} after {past} "
                "cached tokens; its positions must be exactly those"
            )
        return input_ids, positions, spans

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attn_metadata: TrtllmAttentionMetadata,
        *,
        logit_rows: LogitRows = None,
        layer_capture: Optional[Callable[[int], Optional[dict]]] = None,
        trace: Optional[list] = None,
        generation_rows: int = 1,
        extra_kv_tokens: int = 0,
        block_output: Optional[Callable[[int, torch.Tensor], None]] = None,
        gather_ids: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> Union[torch.Tensor, tuple]:
        """One engine step through all 52 layers; FP32 softcapped logits ``[R, vocab]``.

        ``layer_capture(layer)`` may return a dict that receives that block's named
        intermediates (``block``'s capture); ``trace``, when a list, receives one record per
        executed stage in execution order (embedding with the step's request spans, each
        logical layer with its kind and window, final norm, head). ``generation_rows`` and
        ``extra_kv_tokens`` state the step's speculative shape (``check_call``);
        ``block_output`` is called with each layer's completed block output. ``gather_ids``,
        the engine's own logits rows when it states them, must name exactly the rows
        ``logit_rows`` selects. With ``return_hidden`` the result is ``(logits, x)``, ``x``
        the last block's output ``[T, hidden]`` before the final norm.
        """
        self._check_usable()
        ids, positions, spans = self.check_call(
            input_ids,
            position_ids,
            attn_metadata,
            generation_rows=generation_rows,
            extra_kv_tokens=extra_kv_tokens,
        )
        num_tokens = ids.shape[0]
        rows = self._select_rows(logit_rows, spans, attn_metadata.num_contexts)
        if gather_ids is not None:
            self._check_gather_ids(gather_ids, rows)
        x = self.embed(ids)
        if trace is not None:
            trace.append({"stage": "embed", "tokens": num_tokens, "requests": spans})
        for layer in range(NUM_LAYERS):
            capture = layer_capture(layer) if layer_capture is not None else None
            x = self.block(
                layer,
                x,
                positions,
                attn_metadata,
                capture,
                generation_rows=generation_rows,
                extra_kv_tokens=extra_kv_tokens,
                call_checked=True,
            )
            if block_output is not None:
                block_output(layer, x)
            if trace is not None:
                sliding = self.schedule.is_sliding(layer)
                trace.append(
                    {
                        "stage": "layer",
                        "layer": layer,
                        "kv_slot": _attention.local_layer_index(attn_metadata, layer),
                        "mlp": "sparse" if self.schedule.is_sparse(layer) else "dense",
                        "attention": "sliding" if sliding else "full",
                        "window": self.window if sliding else attn_metadata.max_seq_len,
                        "output_sq_norm": x.float().square().sum(),
                    }
                )
        selected = x if len(rows) == num_tokens and rows == list(range(num_tokens)) else x[rows]
        normed = self.final_norm(selected.contiguous())
        logits = self.head(normed)
        if trace is not None:
            trace.append({"stage": "final_norm", "rows": rows})
            trace.append({"stage": "head", "rows": len(rows), "vocab": int(logits.shape[-1])})
        return (logits, x) if return_hidden else logits

    def _check_gather_ids(self, gather_ids: torch.Tensor, rows: list) -> None:
        """Raise ValueError unless the engine's ``gather_ids`` are exactly ``rows``, in order.

        The tensor's kind, length and device are checked on every call; its values, one
        device comparison read back, in diagnostic mode only.
        """
        if (
            not isinstance(gather_ids, torch.Tensor)
            or gather_ids.is_floating_point()
            or gather_ids.dim() != 1
            or gather_ids.numel() != len(rows)
            or gather_ids.device != self.device
        ):
            raise ValueError(
                f"the engine gathers logits rows "
                f"{None if not isinstance(gather_ids, torch.Tensor) else list(gather_ids.shape)} "
                f"on {getattr(gather_ids, 'device', None)}; this step's verifier reads "
                f"{len(rows)} rows on {self.device}"
            )
        if not self.diagnostics:
            return
        expected = torch.tensor(rows, dtype=torch.int64).to(self.device, non_blocking=True)
        if not torch.equal(gather_ids.to(torch.int64), expected):
            raise ValueError(
                f"the engine gathers logits rows {gather_ids.tolist()}; the step's requests "
                f"need the rows {rows} (each context's last row, every generation row)"
            )


__all__ = [
    "MAX_NUM_TOKENS",
    "NUM_LAYERS",
    "TOKENS_PER_BLOCK",
    "BeamFullModelNvfp4Sm103Tp4",
]
