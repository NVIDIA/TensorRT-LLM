# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fast path ``capture_fold``: the DFlash capture written by the fused post-FFN boundary kernel.

With DFlash, the built-in decoder layer ends with
``spec_metadata.maybe_capture_hidden_states(layer_idx, hidden_states, None)``.
On the capture layers (the drafter's ``target_layer_ids``) that is a
``float -> bfloat16`` copy kernel of the layer's FP32 updated residual and a
2-D device copy of the result into the capture buffer, columns
``[i * 8192, (i + 1) * 8192)`` of ``DFlashSpecMetadata.captured_hidden_states``
for the capture index ``i``, rows ``[0, M)``. Both run on the main stream
between the layer's post-FFN boundary and the next layer's QKV projection.

On a decode step the fused boundary kernel (``ops/comm/unpool_gather_boundary``)
computes that residual itself, so this fast path hands it the capture view and the
kernel stores the BF16 rounding next to its other outputs (same
round-to-nearest-even conversion as the copy kernel, same bytes, same
addresses). The re-classed decoder layer then makes no capture call for that
layer: it calls the built-in layer without ``spec_metadata`` and calls the
built-in capture itself when the boundary did not store it -- a call the fused
path did not take, a prefill step, or a layer the step does not capture --
right where the built-in layer would have, after its last kernel.

The built-in layer reads ``spec_metadata`` only for that capture call; it
also hands its keywords on to its attention. So on a call whose capture the
boundary stores, the attention of that layer runs without ``spec_metadata``.
No attention class of this target reads it (the built-in attention passes it
on only to sparse-attention hooks, which this model does not use); an
attention class that does read it cannot be combined with this feature
(``test_modeling_v2_musespark_open_capture_fold.py`` checks the classes).

On a decode call of at most 8 rows on a layer ``install`` counted as folded
(``folded_layers``, which ``boundary_early_release`` converts like any other), a capture the
boundary could not store is still made by the built-in cast and copy, after a
boundary that may already have released the next QKV GEMM without the
``dense_gemm_prefetch`` prefetch. The layer then logs why, once per layer and reason.

Every reader of the buffer is the DFlash worker's prepare after the target
forward, on the same stream (or on a stream forked after the last capture
layer), so it still runs after the writer. The buffer address the kernel is
given is the one the built-in copy would have written at the same call, so a
CUDA graph bakes the same address either way.

This rests on built-in code: the decoder layer's forward
(``MuseSparkOpenDecoderLayer.forward``: where the capture call sits, and that
``spec_metadata`` is read only for it) and
``DFlashSpecMetadata.maybe_capture_hidden_states`` (what it stores).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from tensorrt_llm._torch.speculative.dflash import DFlashSpecMetadata
from tensorrt_llm.logger import logger

from ...ops.comm import unpool_gather_boundary as _unpool
from ...ops.comm.projection_inbox import MAX_ROWS
from ..projection_gathers.gathers import MuseSparkOpenTepDecoderLayer, MuseSparkOpenTepGatedResidual

#: The layer keyword ``DecodeTarget`` passes on decode steps.
LAYER_KWARG = "capture_fold"


def capture_layers(spec_config) -> Tuple[int, ...]:
    """The target layers a DFlash drafter captures, from the speculative config.

    DFlash only: ``destination`` folds ``DFlashSpecMetadata``'s capture and no
    other mode's.

    Raises:
        ValueError: if there is no DFlash drafter, or it names no capture layers.
    """
    mode = getattr(spec_config, "spec_dec_mode", None)
    dflash = mode is not None and mode.is_dflash()
    layers = getattr(spec_config, "target_layer_ids", None) if dflash else None
    if not layers:
        raise ValueError(
            "[musespark_open] capture_fold writes the DFlash capture from the post-FFN boundary; it needs a "
            "DFlash speculative config with target_layer_ids"
        )
    return tuple(sorted(int(layer) for layer in layers))


def destination(spec_metadata, layer_idx: int, hidden_states: torch.Tensor):
    """The view the built-in capture of ``layer_idx`` would write this call, or ``None``.

    ``None`` whenever the built-in capture would not cast the FP32 residual into
    a BF16 buffer of this layout (no DFlash metadata, no buffer, a layer the
    step does not capture, another dtype or width, more rows than the fused
    boundary takes). Host-side reads only: under a CUDA graph each is a
    per-capture constant.
    """
    return _destination(spec_metadata, layer_idx, hidden_states)[0]


def _destination(
    spec_metadata, layer_idx: int, hidden_states: torch.Tensor
) -> Tuple[Optional[torch.Tensor], Optional[str]]:
    """``destination``'s view, or why there is none when the built-in capture still writes.

    ``(view, None)`` when the boundary can store the capture. ``(None, None)``
    when the built-in capture writes nothing the fused boundary could have
    written: no metadata, no buffer, a layer the step does not capture, or more
    rows than the fused boundary takes (its built-in path then runs as well).
    ``(None, reason)`` when the built-in capture writes, through its cast and
    copy, a capture the boundary does not store.
    """
    if spec_metadata is None:
        return None, None
    if (
        type(spec_metadata).maybe_capture_hidden_states
        is not DFlashSpecMetadata.maybe_capture_hidden_states
    ):
        return None, (f"{type(spec_metadata).__name__} has its own maybe_capture_hidden_states")
    buffer = getattr(spec_metadata, "captured_hidden_states", None)
    index = getattr(spec_metadata, "_layer_to_idx", {}).get(layer_idx)
    if buffer is None or index is None:
        return None, None
    rows = hidden_states.shape[0] if hidden_states.dim() == 2 else 0
    if hidden_states.dim() == 2 and not 1 <= rows <= MAX_ROWS:
        return None, None
    hidden = _unpool.HIDDEN
    if not (
        hidden_states.dim() == 2
        and getattr(spec_metadata, "hidden_size", None) == hidden
        and hidden_states.dtype == torch.float32
        and hidden_states.shape[-1] == hidden
        and buffer.dtype == torch.bfloat16
        and buffer.dim() == 2
        and buffer.is_contiguous()
        and buffer.device == hidden_states.device
        and rows <= buffer.shape[0]
        and (index + 1) * hidden <= buffer.shape[1]
    ):
        return None, (
            f"residual {tuple(hidden_states.shape)} {hidden_states.dtype} on "
            f"{hidden_states.device}, capture buffer {tuple(buffer.shape)} {buffer.dtype} "
            f"on {buffer.device}, hidden_size {getattr(spec_metadata, 'hidden_size', None)}"
        )
    view = buffer[:rows, index * hidden : (index + 1) * hidden]
    if not _unpool.capture_fits(view, rows, hidden_states.device):
        return None, (
            f"capture view {tuple(view.shape)} stride {tuple(view.stride())} does not fit "
            "the boundary's store"
        )
    return view, None


class MuseSparkOpenTepCaptureDecoderLayer(MuseSparkOpenTepDecoderLayer):
    """The projection-gather decoder layer, whose post-FFN boundary may also write the DFlash capture.

    ``_capture_folded``: ``install`` counted this layer as folded (``folded_layers``).
    ``_capture_fold_fallbacks``: the reasons already logged for this layer.
    """

    _capture_folded: bool = False
    _capture_fold_fallbacks: frozenset = frozenset()

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata,
        precomputed_attn_input=None,
        next_layer_norm=None,
        sp_num_tokens: Optional[int] = None,
        capture_fold: bool = False,
        **kwargs,
    ):
        spec_metadata = kwargs.get("spec_metadata")
        view, reason = (
            _destination(spec_metadata, self.layer_idx, hidden_states)
            if capture_fold
            else (None, None)
        )
        if view is not None and next_layer_norm is None:
            view, reason = None, "the call has no next-layer norm, so no fused boundary"
        elif view is not None and sp_num_tokens is not None:
            view, reason = None, "a sequence-parallel call"
        if view is None:
            if reason is not None:
                self._capture_fold_fallback(reason)
            return super().forward(
                position_ids,
                hidden_states,
                attn_metadata,
                precomputed_attn_input=precomputed_attn_input,
                next_layer_norm=next_layer_norm,
                sp_num_tokens=sp_num_tokens,
                **kwargs,
            )
        join = self.add_ffn
        join._capture_fold = view
        join._capture_stored = False
        try:
            result = super().forward(
                position_ids,
                hidden_states,
                attn_metadata,
                precomputed_attn_input=precomputed_attn_input,
                next_layer_norm=next_layer_norm,
                sp_num_tokens=sp_num_tokens,
                **{key: value for key, value in kwargs.items() if key != "spec_metadata"},
            )
            stored = join._capture_stored
        finally:
            join._capture_fold = None
            join._capture_stored = False
        updated = result[0] if isinstance(result, tuple) else result
        if not stored:
            # The boundary took the built-in path: the built-in capture, where the
            # built-in layer makes it (after its last kernel).
            self._capture_fold_fallback("the post-FFN join did not take its fused boundary")
            spec_metadata.maybe_capture_hidden_states(self.layer_idx, updated, None)
        return result

    def _capture_fold_fallback(self, reason: str) -> None:
        """Log, once per reason, a built-in capture on a layer ``install`` counted as folded.

        Host-side only (under a CUDA graph: at capture). Such a call runs the
        built-in cast and copy after the boundary, which ``boundary_early_release`` (when installed)
        set to release the next QKV GEMM at entry and whose ``dense_gemm_prefetch`` prefetch it
        dropped, as for a folded layer.
        """
        if not self._capture_folded or reason in self._capture_fold_fallbacks:
            return
        self._capture_fold_fallbacks = self._capture_fold_fallbacks | {reason}
        logger.warning(
            f"[musespark_open] capture_fold: layer {self.layer_idx}'s DFlash capture falls back to the built-in "
            f"cast and copy on a decode call ({reason}); the layer was installed as folded, "
            "so with boundary_early_release those copies follow a boundary that released the next QKV early"
        )


def folded_layers(model, spec_config) -> frozenset:
    """The capture layers whose post-FFN boundary writes the capture once ``install`` ran.

    Every capture layer but the last decoder layer (which has no fused boundary).
    ``boundary_early_release`` converts these layers' QKV edges like any other: no plain copy follows
    their boundary any more.
    """
    last = len(model.layers) - 1
    return frozenset(layer for layer in capture_layers(spec_config) if 0 <= layer < last)


def install(model, spec_config) -> dict:
    """Re-class every decoder layer for ``capture_fold``; the layer keyword ``DecodeTarget`` passes.

    Run after ``projection_gathers.install`` fused the unpool site.

    Raises:
        ValueError: without a DFlash drafter's capture layers, or when a layer is
            not the projection-gather decoder layer with its fused post-FFN join.
    """
    layers = capture_layers(spec_config)
    for layer in model.layers:
        if type(layer) is not MuseSparkOpenTepDecoderLayer or not isinstance(
            layer.add_ffn, MuseSparkOpenTepGatedResidual
        ):
            raise ValueError(
                "[musespark_open] capture_fold writes the capture from the fused post-FFN boundary; layer "
                f"{layer.layer_idx} is {type(layer).__name__} with {type(layer.add_ffn).__name__}"
            )
    folded = sorted(folded_layers(model, spec_config))
    for layer in model.layers:
        layer.__class__ = MuseSparkOpenTepCaptureDecoderLayer
        layer._capture_folded = layer.layer_idx in folded
        layer._capture_fold_fallbacks = frozenset()
    logger.info(
        f"[musespark_open] capture_fold: the post-FFN boundary writes the DFlash capture of layers {folded} "
        f"on decode steps of at most {MAX_ROWS} rows"
        + (
            f"; layers {sorted(set(layers) - set(folded))} keep the built-in copy"
            if len(folded) < len(layers)
            else ""
        )
    )
    return {LAYER_KWARG: True}
