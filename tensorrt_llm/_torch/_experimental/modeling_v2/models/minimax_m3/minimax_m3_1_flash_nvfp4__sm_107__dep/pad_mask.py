# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PAD_MASK: an attention-DP rank's CUDA-graph padding requests send their rows to no expert, and LC2 reads no KV for
them.

A decode graph runs at its bucket's request count, so a rank with fewer live requests pads its batch with dummy
requests, whose rows (``spec_metadata.batch_slot_ids == dummy_slot_row``) otherwise route, dispatch and run like real
tokens. ``install`` puts a forward pre-hook on every MoE layer whose backend takes a pad mask (the MegaMoE local
backend's ``set_pad_mask``). ``PadMask.begin`` builds the step's uint8 row mask once per forward from the speculative
metadata, with device ops on static shapes, so a captured graph rebuilds it at every replay. Each hook hands the mask to
the backend's next launch when that launch has exactly the mask's rows; the backend then sends the masked rows to no
expert (ids -1, weights 0) and leaves the real rows' outputs unchanged.

Once a MoE layer takes the mask, ``arm_lc2`` also routes the LC2 decode's KV lengths through ``PadMask.lc2_seq_lens``:
a copy of the step's per-request lengths with the padding requests' set to 0, so LC2 selects no block for their rows and
writes zeros there. The copy is made at the forward's first LC2 call and reused by every later layer, which reads the
same length buffer. The real rows are unchanged. The skip needs the MoE mask: without it, the zeroed rows would still
route, all to the same experts.

``TRTLLM_M31_DISABLE_MEGAMOE_PAD_MASK=1`` installs neither; ``TRTLLM_M31_DISABLE_DEP_LC2_PAD_SKIP=1`` keeps the MoE
mask without the LC2 skip.
"""

from typing import Iterable, Optional, Tuple

import torch

from tensorrt_llm.logger import logger

from . import lc2_decode
from .m31_switches import m31_disabled

PAD_MASK = not m31_disabled("MEGAMOE_PAD_MASK")
LC2_PAD_SKIP = PAD_MASK and not m31_disabled("DEP_LC2_PAD_SKIP")

_HOOKED = "_m31_pad_mask_hooked"


def _padding_row_mask(
    spec_metadata, num_rows: int, num_contexts: int, batch_size: int, rows_per_request: int
) -> Optional[torch.Tensor]:
    """Bool mask over ``num_rows`` sampling rows marking CUDA-graph padding requests.

    Padding requests are the ones ``populate_sampling_params_for_one_model`` routed to ``dummy_slot_row``
    (``py_seq_slot is None``), so this needs no extra host-side state. Shapes are static: CUDA-graph safe. None while
    ``dummy_slot_row`` is still 0 (or the metadata has none), which is a live request's row rather than a padding
    marker, and when the rows do not match the batch layout.
    """
    slot_ids = getattr(spec_metadata, "batch_slot_ids", None)
    dummy_slot_row = getattr(spec_metadata, "dummy_slot_row", 0)
    num_gens = batch_size - num_contexts
    if slot_ids is None or dummy_slot_row <= 0 or num_gens <= 0:
        return None
    is_padding = slot_ids[num_contexts:batch_size] == dummy_slot_row
    if num_rows != num_gens * rows_per_request:
        return None
    if rows_per_request > 1:
        is_padding = is_padding.repeat_interleave(rows_per_request)
    return is_padding


class PadMask:
    """The current forward's padding requests: ``mask`` is uint8 with one entry per token row, ``requests`` bool with
    one entry per generation request; both nonzero on a padding request."""

    def __init__(self, rank: int = 0):
        self.mask: Optional[torch.Tensor] = None
        self.requests: Optional[torch.Tensor] = None
        #: The LC2 layers that skip the padding requests (set by ``arm_lc2``).
        self.lc2_layers = 0
        self._rank = rank
        self._logged = set()
        self._lc2_key = None
        self._lc2_seq_lens: Optional[torch.Tensor] = None

    def begin(self, attn_metadata, spec_metadata, num_tokens: int) -> None:
        """Build the masks for a generation-only step whose requests carry equal row counts; otherwise leave none.

        Runs in every rank's forward, including a rank whose requests are all padding and dispatches nothing, so each
        rank logs its engagement here once per row count."""
        self.end()
        if attn_metadata is None or spec_metadata is None:
            return
        num_contexts = attn_metadata.num_contexts
        batch_size = attn_metadata.num_seqs
        num_gens = batch_size - num_contexts
        if num_contexts or num_gens <= 0 or num_tokens % num_gens:
            return
        if getattr(spec_metadata, "dummy_slot_row", 0) <= 0:
            self._log_once("no mask: dummy_slot_row is not published yet")
            return
        requests = _padding_row_mask(spec_metadata, num_gens, num_contexts, batch_size, 1)
        if requests is None:
            return
        rows = num_tokens // num_gens
        self.requests = requests
        self.mask = (requests.repeat_interleave(rows) if rows > 1 else requests).to(torch.uint8)
        lc2 = f"on {self.lc2_layers} layers" if self.lc2_layers else "off"
        self._log_once(
            f"engaged at {num_tokens} rows ({num_gens} requests): MoE mask on, LC2 skip {lc2}"
        )

    def lc2_seq_lens(self, seq_lens: torch.Tensor) -> torch.Tensor:
        """``seq_lens`` with the padding requests' KV lengths set to 0 when the LC2 skip is armed and the decode covers
        exactly the mask's requests; else ``seq_lens`` itself. Made once per forward and per length buffer."""
        requests = self.requests
        if requests is None or not self.lc2_layers:
            return seq_lens
        if seq_lens.shape[0] != requests.numel():
            self._log_once(
                f"LC2 skip declined: {seq_lens.shape[0]} decode requests, {requests.numel()} in the mask"
            )
            return seq_lens
        key = (seq_lens.data_ptr(), seq_lens.shape[0])
        if self._lc2_key != key:
            self._lc2_key = key
            self._lc2_seq_lens = seq_lens.masked_fill(requests, 0)
        return self._lc2_seq_lens

    def _log_once(self, outcome: str) -> None:
        if outcome not in self._logged:
            self._logged.add(outcome)
            logger.info(f"[M3.1 DEP] rank {self._rank}: PAD_MASK {outcome}")

    def end(self) -> None:
        self.mask = self.requests = None
        self._lc2_key = self._lc2_seq_lens = None


def decline_reason(layer: torch.nn.Module) -> Optional[str]:
    """Why ``layer``'s MoE launch takes no pad mask, or None."""
    experts = getattr(getattr(layer, "block_sparse_moe", None), "experts", None)
    if experts is None:
        return "no MoE"
    backend = getattr(experts, "backend", None)
    if not hasattr(backend, "set_pad_mask"):
        return f"backend {type(backend).__name__} takes no pad mask"
    return backend.pad_mask_decline_reason()


def install(layers: Iterable[torch.nn.Module], state: PadMask) -> Tuple[int, Optional[str]]:
    """Hook every MoE layer whose launch takes a pad mask. Returns the hooked count and, when a MoE layer is left out,
    the first such layer's reason."""
    if not PAD_MASK:
        return 0, None
    hooked, left_out = 0, None
    for layer in layers:
        reason = decline_reason(layer)
        if reason is None:
            experts = layer.block_sparse_moe.experts
            if not getattr(experts, _HOOKED, False):
                experts.register_forward_pre_hook(_hand_over(state, experts.backend))
                setattr(experts, _HOOKED, True)
            hooked += 1
        elif reason != "no MoE" and left_out is None:
            left_out = reason
    return hooked, left_out


def arm_lc2(state: PadMask, moe_layers: int, lc2_layers: int) -> int:
    """Route the LC2 decode's KV lengths through ``state`` when ``install`` hooked MoE layers and LC2 runs on
    ``lc2_layers``; returns the layers that skip the padding requests (0 leaves the decode unrouted)."""
    state.lc2_layers = lc2_layers if LC2_PAD_SKIP and moe_layers and lc2_layers else 0
    lc2_decode.set_pad_rows(state if state.lc2_layers else None)
    return state.lc2_layers


def _hand_over(state: PadMask, backend):
    def hook(module, args):
        mask = state.mask
        shape = getattr(args[0], "shape", None) if args else None
        rows = shape[0] if shape else None
        backend.set_pad_mask(mask if mask is not None and mask.numel() == rows else None)

    return hook
