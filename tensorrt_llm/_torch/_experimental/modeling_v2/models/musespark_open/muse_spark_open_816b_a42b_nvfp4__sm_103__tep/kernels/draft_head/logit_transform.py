# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash drafter's logit transform on the vocab-sharded draft head (``draft_head``).

A DFlash drafter whose config declares ``output_multiplier`` or
``final_logit_softcapping`` is trained on transformed logits, and the built-in
worker applies ``DFlashForCausalLM.apply_draft_logit_transform`` to the draft
logits before the selector: in fp32 and in this order, ``.float()``, ``*
multiplier``, ``/ softcap``, ``tanh``, ``* softcap``. Call it ``t``. The
selector then takes ``torch.topk(t(logits), top_k)``: its ids, their order and
its values (the unary scores it adds to its pairwise ones).

``t`` never decreases, so a top-k of the bf16 logits has the same candidates in
the same order -- except where ``t`` maps distinct bf16 values to the same fp32
value. tanh saturates: with the Muse Spark Open drafter's multiplier and softcap
every logit of large enough magnitude transforms to exactly +-softcap. Those
compare equal for ``torch.topk``, which then keeps and orders them by vocab id,
while their bf16 values are distinct. So the draft head ranks canonical keys
and reads the unary values from a table, both built once per drafter here:

    canon   [65536] int32: 16-bit radix key of a bf16 logit -> the key of its tie
            group's representative. A tie group is the bf16 values whose
            transformed values have the same bits; ``t`` never decreases, so a
            group is a run of consecutive keys, and canonical keys compare as the
            transformed values' radix keys do. The representative is the group's
            smallest key, or its largest below +0.0 so that the -0.0 group keeps
            -0.0 (``-0.0 == +0.0`` under the selector sort's comparison, as for
            the transformed values). NaN (0xFFFF) maps to itself.
    values  [65536] fp32: key -> ``t`` of that bf16 value, computed by the
            drafter's own method on the device, so it is the built-in transform
            of every logit bit for bit (the transform is elementwise).
    lo, hi  the key interval around 0 on which ``canon`` is the identity: logits
            there need no table load.

``unsupported_reason`` checks all of it exhaustively over the 65536 keys before
the draft head installs it.
"""

from __future__ import annotations

import hashlib
import weakref
from typing import Callable, Optional

import torch

from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM

from ...ops.spec.draft_shard_topk import CANON_ENTRIES, FINITE_KEYS, canon_reason, radix_key16

__all__ = [
    "DraftLogitTransform",
    "bf16_of_key16",
    "build",
    "digest",
    "ensure_transformed",
    "same_bits",
    "tie_groups",
    "unsupported_reason",
]

#: Rows of the install-time layout check: eight requests of K = 7 positions.
LAYOUT_ROWS = 56

#: Seed of the layout check's logits, drawn on the host so no device RNG state moves.
LAYOUT_SEED = 2718

#: Key16 of +0.0 and -0.0.
POSITIVE_ZERO, NEGATIVE_ZERO = 0x8000, 0x7FFF


def bf16_of_key16(keys: torch.Tensor) -> torch.Tensor:
    """The bf16 values of 16-bit radix keys (any int dtype, values in [0, 0xFFFF])."""
    key = keys.to(torch.int32)
    bits = torch.where(key >= 0x8000, key ^ 0x8000, key ^ 0xFFFF)
    bits = torch.where(bits >= 0x8000, bits - 0x10000, bits).to(torch.int16)
    return bits.view(torch.bfloat16)


def _radix32(values: torch.Tensor) -> torch.Tensor:
    """torch's fp32 radix key of ``values`` as int64 in [0, 2**32); every NaN is the top key."""
    word = values.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    key = torch.where(word >= 0x80000000, word ^ 0xFFFFFFFF, word ^ 0x80000000)
    return torch.where(values.isnan(), torch.full_like(key, 0xFFFFFFFF), key)


def same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bitwise equality of fp32 tensors, NaN payloads aside."""
    nan = a.isnan()
    if not torch.equal(nan, b.isnan()):
        return False
    return torch.equal(
        torch.where(nan, 0.0, a).view(torch.int32), torch.where(nan, 0.0, b).view(torch.int32)
    )


class DraftLogitTransform:
    """A drafter's logit transform as the draft head's tables (see the module docstring).

    ``canon`` [65536] int32 and ``values`` [65536] fp32 live on the drafter's
    device; ``lo`` and ``hi`` are plain ints; ``multiplier`` and ``softcap`` are
    the drafter attributes the tables were built from. ``real`` is the drafter's
    own transform, the built-in computation.
    """

    def __init__(
        self,
        draft_model,
        function: Callable,
        canon: torch.Tensor,
        values: torch.Tensor,
        lo: int,
        hi: int,
    ):
        self._model = weakref.ref(draft_model)
        self.function = function
        self.canon = canon
        self.values = values
        self.lo = lo
        self.hi = hi
        self.multiplier = getattr(draft_model, "logit_multiplier", None)
        self.softcap = getattr(draft_model, "logit_softcap", None)

    @property
    def band(self) -> tuple[int, int]:
        """``(lo, hi)``: ``draft_shard_topk``'s ``canon_band``."""
        return self.lo, self.hi

    def matches(self, draft_model) -> bool:
        """Whether ``draft_model`` still declares the transform these tables hold."""
        return (
            getattr(draft_model, "logit_multiplier", None),
            getattr(draft_model, "logit_softcap", None),
        ) == (self.multiplier, self.softcap)

    def real(self, logits: torch.Tensor) -> torch.Tensor:
        """The drafter's transform of ``logits``: the built-in computation, fp32 out."""
        draft_model = self._model()
        if draft_model is None:
            raise RuntimeError("[musespark_open] draft_head: the drafter of this transform is gone")
        return self.function(draft_model, logits)


def build(draft_model, device) -> Optional[DraftLogitTransform]:
    """The tables of ``draft_model``'s logit transform on ``device``, or None for no transform.

    None when the drafter's class has no ``apply_draft_logit_transform`` or it
    returns a bf16 probe unchanged (multiplier 1, no softcap). The probe is
    bf16 because on an fp32 input the built-in transform runs in place and
    returns the input itself even when it is on. Runs the drafter's method on
    all 65536 bf16 values, so call it outside CUDA graph capture.
    ``unsupported_reason`` says whether the result can be used.
    """
    function = getattr(type(draft_model), "apply_draft_logit_transform", None)
    if function is None:
        return None
    probe = torch.zeros(4, dtype=torch.bfloat16, device=device)
    if function(draft_model, probe) is probe:
        return None
    keys = torch.arange(CANON_ENTRIES, dtype=torch.int32, device=device)
    values = function(draft_model, bf16_of_key16(keys))
    canon = torch.arange(CANON_ENTRIES, dtype=torch.int32)
    lo, hi = 0, CANON_ENTRIES - 1
    if values.dtype == torch.float32 and tuple(values.shape) == (CANON_ENTRIES,):
        first_key, last_key = FINITE_KEYS
        radix = _radix32(values.cpu())[first_key : last_key + 1]
        finite = torch.arange(first_key, last_key + 1, dtype=torch.int64)
        start = torch.ones_like(radix, dtype=torch.bool)
        start[1:] = radix[1:] != radix[:-1]
        end = torch.ones_like(start)
        end[:-1] = start[1:]
        group = torch.cumsum(start.to(torch.int64), dim=0) - 1
        first, last = finite[start], finite[end]
        representative = torch.where(first >= POSITIVE_ZERO, first, last)
        canon[first_key : last_key + 1] = representative[group].to(torch.int32)
        identity = canon == torch.arange(CANON_ENTRIES, dtype=torch.int32)
        above = (~identity[POSITIVE_ZERO:]).nonzero()
        below = (~identity[:POSITIVE_ZERO]).nonzero()
        hi = CANON_ENTRIES - 1 if above.numel() == 0 else POSITIVE_ZERO + int(above[0]) - 1
        lo = 0 if below.numel() == 0 else int(below[-1]) + 1
    return DraftLogitTransform(draft_model, function, canon.to(device), values.contiguous(), lo, hi)


def tie_groups(transform: DraftLogitTransform) -> list[tuple[int, int]]:
    """``(first, last)`` key16 of every tie group of two or more values, in key order."""
    first_key, last_key = FINITE_KEYS
    canon = transform.canon.cpu()[first_key : last_key + 1].to(torch.int64)
    start = torch.ones_like(canon, dtype=torch.bool)
    start[1:] = canon[1:] != canon[:-1]
    begins = (start.nonzero().view(-1) + first_key).tolist()
    ends = [b - 1 for b in begins[1:]] + [last_key]
    return [(b, e) for b, e in zip(begins, ends) if e > b]


def _layout_probe(rows: int, shard: int) -> torch.Tensor:
    """Host bf16 ``[rows, shard]`` logits: every bf16 value once, then values of many scales.

    The 65536 bf16 bit patterns (NaNs included) fill the first columns in a
    random order; each remaining row is normal logits of one scale up to deep
    saturation, with +-inf, +-0.0 and NaN scattered in.
    """
    g = torch.Generator().manual_seed(LAYOUT_SEED)
    scales = (1.0, 10.0, 100.0, 400.0, 1000.0, 3000.0, 1e5, 1e30)
    x = torch.randn(rows, shard, generator=g)
    x *= torch.tensor([scales[r % len(scales)] for r in range(rows)])[:, None]
    special = torch.tensor([float("inf"), float("-inf"), 0.0, -0.0, float("nan")])
    picks = torch.randint(0, shard, (rows, 40), generator=g)
    x.scatter_(1, picks, special[torch.randint(0, len(special), (rows, 40), generator=g)])
    x = x.to(torch.bfloat16)
    every = torch.arange(CANON_ENTRIES, dtype=torch.int32)[
        torch.randperm(CANON_ENTRIES, generator=g)
    ]
    flat = x.view(-1)
    count = min(CANON_ENTRIES, flat.numel())
    flat[:count] = bf16_of_key16(every[:count])
    return x


def unsupported_reason(
    transform: DraftLogitTransform, shard: int, rows: int = LAYOUT_ROWS
) -> Optional[str]:
    """Why the draft head cannot apply ``transform`` after its top-k, or None.

    Exhaustive over the 65536 bf16 values: the transform is the built-in
    ``DFlashForCausalLM`` method, returns fp32, keeps NaN, +0.0 and -0.0, never
    decreases in torch's radix order (which makes the tie groups runs of
    consecutive keys), and the canonical values compare under ``>`` as the
    transformed values do. ``canon`` is a table ``draft_shard_topk`` reads
    correctly with band ``(lo, hi)``. Last, the table gives the drafter's own
    transform of a ``[rows, shard]`` bf16 slice bit for bit (a host-seeded
    probe holding every bf16 value).
    """
    function = transform.function
    if function is not DFlashForCausalLM.apply_draft_logit_transform:
        return (
            f"the drafter's apply_draft_logit_transform is {function.__qualname__}, "
            "not DFlashForCausalLM's"
        )
    values = transform.values
    if values.dtype != torch.float32 or tuple(values.shape) != (CANON_ENTRIES,):
        return f"the drafter's logit transform returns {values.dtype}, not fp32"
    table = values.cpu()
    inputs = bf16_of_key16(torch.arange(CANON_ENTRIES, dtype=torch.int32))
    if not torch.equal(inputs.isnan(), table.isnan()):
        return "the drafter's logit transform does not map exactly the NaN logits to NaN"
    words = table.view(torch.int32)
    if int(words[POSITIVE_ZERO]) != 0 or int(words[NEGATIVE_ZERO]) != -(2**31):
        return "the drafter's logit transform does not keep +0.0 and -0.0"
    first_key, last_key = FINITE_KEYS
    radix = _radix32(table)[first_key : last_key + 1]
    drops = (radix[1:] < radix[:-1]).nonzero().view(-1)
    if drops.numel():
        at = float(inputs[first_key + int(drops[0]) + 1].float())
        return f"the drafter's logit transform decreases at the bf16 logit {at}"
    canon = transform.canon.cpu()
    reason = canon_reason(canon, transform.band, torch.device("cpu"))
    if reason is not None:
        return reason
    finite = canon[first_key : last_key + 1]
    start = torch.ones_like(finite, dtype=torch.bool)
    start[1:] = finite[1:] != finite[:-1]
    members = torch.arange(first_key, last_key + 1)[start]
    if bool((radix[1:] == radix[:-1])[start[1:]].any()):
        return "canon splits a tie group of the transform"
    if not torch.equal(_radix32(table[finite.long()]), _radix32(table[first_key : last_key + 1])):
        return "canon maps a key outside its tie group"
    shown = bf16_of_key16(finite[start]).float()
    shown_values = table[members]
    if not torch.equal(shown[1:] > shown[:-1], shown_values[1:] > shown_values[:-1]):
        return "canonical values compare differently from the transformed values"
    probe = _layout_probe(rows, shard).to(values.device)
    got = values[radix_key16(probe.view(torch.int16)).long()]
    if not same_bits(got, transform.real(probe)):
        return (
            f"the drafter's logit transform of a [{rows}, {shard}] slice differs from its "
            "transform of the 65536 bf16 values"
        )
    return None


def digest(transform: Optional[DraftLogitTransform]) -> str:
    """A short digest of the tables, equal on ranks whose draft heads would agree."""
    if transform is None:
        return "no drafter logit transform"
    h = hashlib.sha256()
    h.update(transform.canon.cpu().numpy().tobytes())
    h.update(transform.values.cpu().view(torch.int32).numpy().tobytes())
    h.update(repr((transform.lo, transform.hi, transform.multiplier, transform.softcap)).encode())
    return f"drafter logit transform tables {h.hexdigest()[:16]}"


def ensure_transformed(
    transform: Optional[DraftLogitTransform], logits: torch.Tensor
) -> torch.Tensor:
    """``logits`` with the drafter's transform applied once.

    With a transform installed, bf16 draft logits are untransformed (the
    transform always returns fp32, which ``unsupported_reason`` checks), so a
    bf16 ``logits`` gets ``transform.real`` and anything else is returned as is.
    """
    if transform is None or logits.dtype != torch.bfloat16:
        return logits
    return transform.real(logits)
