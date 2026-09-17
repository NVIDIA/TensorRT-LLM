# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What a catalog entry is, expressed as something that runs.

An entry used to be a wrapper plus a prose contract. The prose said what the
op computes, which inputs it accepts, and where it goes silently wrong -- and
nothing checked that any of it was still true. A kernel change upstream could
falsify a page of it without a single test going red.

This is the executable half of that contract:

    __call__    the guarded call a target makes
    reference   what the op is supposed to compute, in plain torch
    is_valid    the inputs the wrapper refuses before dispatch
    compare     how close the two have to be, and why
    CELLS       the input configurations this entry is certified over

`reference` and `CELLS` are the part that cannot go stale: the entry's test
drives every cell through both `__call__` and `reference` and gates them with
`compare`, so a claim that stops holding stops the build.

`CELLS` is also the answer to "can my model use this op?". A target author
reads the cell list and its comments, not a document -- and what they read is
what CI proved on the last commit.

`note` is what is left: statements that are neither a computation nor a
precondition. Keep it small. Anything that can be a cell, a `is_valid` branch,
or a test asserting a hazard belongs there instead, where it is checked.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class Cell:
    """One input configuration an entry is certified over.

    `spec` is entry-specific: the entry's test knows how to turn it into
    tensors, and nothing here interprets it. That keeps environment building
    -- a KV-cache manager, a 4-rank communicator -- in the test where it
    belongs, while the *set of configurations* stays with the op, which is
    what a target author needs to read.

    `why` is not decoration. A cell that cannot say what it covers beyond the
    ones already listed is a cell that costs CI time for nothing.
    """

    why: str
    spec: Mapping[str, Any] = field(default_factory=dict)


class OpWrapper(ABC):
    """Base for a catalog entry. Instances are callable and stateless.

    Subclasses are instantiated once at module scope under the op's own name,
    so a target still writes `flashinfer_rmsnorm(x, w, eps)` and never sees
    the class.
    """

    #: The configurations this entry is certified over. A target author reads
    #: this to decide whether the op covers their shape; CI reads it to know
    #: what to drive.
    CELLS: tuple[Cell, ...] = ()

    #: Whatever is neither computation nor precondition -- registration
    #: conditions, inert arguments, behaviour of paths this entry does not
    #: certify. Prose here is unverified by construction, which is the reason
    #: to keep as little of it as possible.
    note: str = ""

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Guard the inputs, then make the op call."""

    @abstractmethod
    def reference(self, *args: Any, **kwargs: Any) -> Any:
        """What the op is supposed to compute, in plain torch.

        Accumulate in fp32 and round once at the end: the reference has to be
        the more accurate side, or `compare`'s band is measuring the reference
        rather than the op.
        """

    def is_valid(self, *args: Any, **kwargs: Any) -> None:
        """Raise on an input the op accepts but computes wrongly.

        Default: nothing to check. Override where the op is known to take an
        input and return a plausible wrong answer -- that is the case a
        caller cannot detect for themselves.
        """
        return None

    def compare(self, out: torch.Tensor, ref: torch.Tensor) -> None:
        """Gate `out` against `reference`.

        The default is torch's dtype-aware band, which is right for an op that
        rounds once. An op that accumulates through several GEMMs will exceed
        it for reasons that are not defects, and must widen the band *and say
        what it measured* -- a tolerance without a derivation is a tolerance
        that will be widened again next time it fails.
        """
        assert out.dtype == ref.dtype, (out.dtype, ref.dtype)
        assert out.shape == ref.shape, (out.shape, ref.shape)
        torch.testing.assert_close(out, ref)
