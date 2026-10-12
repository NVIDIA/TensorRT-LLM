# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What a catalog entry is, expressed as something that runs.

An entry used to be a wrapper plus a prose contract. The prose said what the
op computes, which inputs it accepts, where it goes silently wrong, and which
architecture it had been measured on -- and nothing checked that any of it was
still true. A kernel change upstream could falsify a page of it without a
single test going red.

This is the executable form of that contract:

    __call__    the op call a target makes
    reference   what the op is supposed to compute, in plain torch
    is_valid    the inputs the op takes and answers wrongly
    CELLS       the input configurations this entry is certified over
    ARCHS       the architectures those cells were driven on
    note        what is left, and nothing that could have been one of the above

`reference` and `CELLS` are the part that cannot go stale: the entry's test
drives every cell through both `__call__` and `reference` and gates one
against the other within a band the test states and derives, so a claim that
stops holding stops the build. How close the two have to be is the test's
business, not the entry's: nothing on the way to the kernel needs it.

`CELLS` is also the answer to "can my model use this op?". A target author
reads the cell list and its reasons, not a document -- and what they read is
what CI proved on the last commit. Cells cover what the shipped targets
actually run; an input outside them is not known to work, which for most of
these ops means "nobody measured it" rather than "known broken".

`is_valid` is never called on the way to the kernel. It states, in executable
form, the inputs the op accepts and quietly answers wrongly -- the class of
mistake a caller cannot detect for themselves. Running it per call would buy a
target nothing it does not already get from reading it once, and would put its
own bugs on the hot path of a served engine. The test drives it: `validating()`
in the test tree interposes it, so every cell also proves the guard admits what
it is supposed to admit, and the rejection cases prove it refuses what it is
supposed to refuse.
"""

from __future__ import annotations

import contextlib
import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch


@contextlib.contextmanager
def true_fp32_matmul():
    """Make torch's fp32 matmul actually fp32 for the duration.

    torch 2.12 defaults ``matmul.fp32_precision`` to ``tf32`` and
    ``allow_tf32`` to True on this hardware, so a plain ``a.float() @
    b.float()`` is a *TF32* product -- ~1e-3 relative, which is 30x the error
    of the ops being tested. Left alone the reference is the inaccurate side of
    the comparison and the entry fails against a correct kernel. Measured: with
    TF32 off the op is bit-identical to torch and both sit 1.9e-5 from a
    float64 product; with TF32 on the reference alone moves by 0.035.

    Both switches have to move. Setting only ``allow_tf32`` leaves
    ``fp32_precision`` at ``tf32`` and the product is still TF32 -- which is
    exactly the failure this docstring is here to stop someone rediscovering.
    """
    prev_allow = torch.backends.cuda.matmul.allow_tf32
    prev_prec = getattr(torch.backends.cuda.matmul, "fp32_precision", None)
    torch.backends.cuda.matmul.allow_tf32 = False
    if prev_prec is not None:
        torch.backends.cuda.matmul.fp32_precision = "ieee"
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_allow
        if prev_prec is not None:
            torch.backends.cuda.matmul.fp32_precision = prev_prec


class Arch(str, enum.Enum):
    """A GPU architecture an entry can be certified on.

    An enum rather than a string so that a typo is an AttributeError here
    rather than an entry that silently claims nothing, and so that adding an
    architecture is one edit that every entry's declaration is checked against.
    """

    SM_103 = "sm_103"


@dataclass(frozen=True)
class Cell:
    """One input configuration an entry is certified over.

    `spec` is entry-specific: the entry's test knows how to turn it into
    tensors, and nothing here interprets it. That keeps environment building --
    a KV-cache manager, a 4-rank communicator -- in the test where it belongs,
    while the *set of configurations* stays with the op, which is what a target
    author needs to read.

    `why` is not decoration. A cell that cannot say what it covers beyond the
    ones already listed is a cell that costs CI time for nothing.
    """

    why: str
    spec: Mapping[str, Any] = field(default_factory=dict)


class OpWrapper(ABC):
    """Base for a catalog entry.

    An entry is a callable and nothing more: `__call__` takes every argument
    explicitly, reads nothing off `self`, and dispatches to the kernel. A
    target that calls one passes the weights, the per-step batch state and the
    configuration scalars at the call site, every time, so the call site shows
    exactly what the kernel receives.

    What that keeps true: what an entry computes is a pure function of its
    arguments, so `CELLS` covers the entry's whole input space, and `reference`
    and `is_valid` mirror `__call__` directly. A gate in the claims suite
    enforces that `__call__` reads no attribute of `self`.

    Instances carry no state, so one module-level instance per entry is the
    normal way to reach it; a target may also construct its own.
    """

    ARCHS: frozenset[Arch] = frozenset()
    CELLS: tuple[Cell, ...] = ()
    note: str = ""

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Make the op call, taking every argument explicitly. The hot path.

        Must not read state off `self` -- see the class docstring. The test
        tree interposes validation by patching this method on the subclass,
        which is why it has to live there.
        """

    @abstractmethod
    def reference(self, *args: Any, **kwargs: Any) -> Any:
        """What the op is supposed to compute, in plain torch.

        Mirrors `__call__`'s signature, parameter for parameter and name for
        name, so the two can be driven from one cell's argument list. An
        argument this computation does not need keeps its name anyway: dropping
        it would hide that the op takes it, and renaming it would break every
        call site that passes it by keyword. Say which ones are ignored in the
        docstring; nothing in this repo's lint objects to an unused parameter.

        Collectives are the one documented exception to mirroring at all. A
        collective's output is not a function of what the calling rank holds, so
        a reference given only the local `input` could state nothing; those take
        the group's inputs in rank order and say so in their own docstring.

        Accumulate in fp32 and round once at the end: the reference has to be
        the more accurate side, or the test's band measures the reference
        rather than the op.
        """

    def is_valid(self, *args: Any, **kwargs: Any) -> None:
        """Raise on an input the op accepts but computes wrongly.

        Not called on the way to the kernel -- see the module docstring. Default
        is nothing to check; override only where the op is known to take an
        input and return a plausible wrong answer. A condition the op already
        rejects loudly does not belong here: restating it only swaps its error
        for a worse one.

        Mirrors `__call__`'s signature, like `reference`, because `validating`
        forwards the call's arguments verbatim -- an override that accepts only
        the arguments it inspects raises TypeError on every call site that
        passes the others. Parameters keep the op's own names for the same
        reason `reference`'s do.
        """
        return None
