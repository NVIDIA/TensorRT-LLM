# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Debug guard for the window between staging a write and the deferred_input_prep launch.

A staged write reaches its destination only at the deferred_input_prep launch. Between the
staging call and the launch, another device operation that reads the destination
would see older bytes, and one that writes it would be overwritten by the
launch. ``DebugWindow`` (``TRTLLM_MODELING_V2_VALIDATE=1``,
see ``compat``) makes both cases fail loudly:

* a torch dispatch mode and a hook on Triton's ``JITFunction.run`` raise
  ``InputPrepHazardError`` when an operation's CUDA tensor arguments overlap a
  destination that was already staged;
* every staged destination is snapshotted when staged (level 1) or filled
  with ``POISON_BYTE`` when the launch overwrites it entirely (level 2) and
  compared right before the launch; a difference means some operation (of
  any launcher type) wrote it inside the window.

Operations issued by the stager itself run inside ``internal()`` and are not
checked.
"""

import contextlib
import traceback
from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

POISON_BYTE = 0xA5

# Operators that read only tensor metadata (shape, dtype, device).
_METADATA_ONLY_OPS = frozenset(
    {
        "aten::alias",
        "aten::detach",
        "aten::empty_like",
        "aten::full_like",
        "aten::lift_fresh",
        "aten::new_empty",
        "aten::new_empty_strided",
        "aten::new_full",
        "aten::new_ones",
        "aten::new_zeros",
        "aten::ones_like",
        "aten::record_stream",
        "aten::zeros_like",
    }
)


class InputPrepHazardError(RuntimeError):
    """An operation touched a deferred_input_prep-staged destination before the deferred_input_prep launch."""


def tensor_span(t: torch.Tensor) -> Optional[tuple]:
    """Byte range ``[lo, hi)`` covered by a CUDA tensor view, or None."""
    if not t.is_cuda or t.numel() == 0:
        return None
    last = 0
    for size, stride in zip(t.shape, t.stride()):
        if size > 1:
            last += (size - 1) * stride
    lo = t.data_ptr()
    return lo, lo + (last + 1) * t.element_size()


def _reads_only_metadata(func) -> bool:
    schema = func._schema
    if schema.name in _METADATA_ONLY_OPS:
        return True
    # View operators return an alias of an input without touching its data.
    return any(r.alias_info is not None and not r.alias_info.is_write for r in schema.returns)


def _call_site() -> str:
    for frame in reversed(traceback.extract_stack(limit=12)[:-1]):
        if "kernels/deferred_input_prep/" not in frame.filename.replace("\\", "/"):
            return f"{frame.filename.rsplit('/', 1)[-1]}:{frame.lineno} {frame.name}"
    return "deferred_input_prep"


@dataclass
class _Staged:
    lo: int
    hi: int
    label: str
    view: torch.Tensor
    snapshot: Optional[torch.Tensor]  # None: poisoned


class _GuardMode(TorchDispatchMode):
    def __init__(self, window: "DebugWindow"):
        super().__init__()
        self.window = window

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        window = self.window
        if window.internal_depth == 0 and window.staged and not _reads_only_metadata(func):
            window.check(str(func), tree_flatten((args, kwargs))[0])
        return func(*args, **kwargs)


class DebugWindow:
    """Guard state of one staged preparation (see the module docstring)."""

    def __init__(self, level: int):
        self.level = level
        self.staged: List[_Staged] = []
        self.internal_depth = 0
        self.num_checked_ops = 0
        self.num_windows = 0
        self._mode: Optional[_GuardMode] = None
        self._triton_jit = None
        self._triton_run = None

    # ---- window lifecycle ----------------------------------------------

    def open(self) -> None:
        self.close()
        self.staged = []
        self.num_windows += 1
        self._mode = _GuardMode(self)
        self._mode.__enter__()
        self._hook_triton()

    def close(self) -> None:
        if self._mode is not None:
            mode, self._mode = self._mode, None
            mode.__exit__(None, None, None)
        self._unhook_triton()

    @contextlib.contextmanager
    def internal(self):
        """Operations of the stager itself (not checked)."""
        self.internal_depth += 1
        try:
            yield
        finally:
            self.internal_depth -= 1

    # ---- staging and checks --------------------------------------------

    def register(self, dst: torch.Tensor, overwrite: bool) -> None:
        """Record a staged destination; call inside ``internal()``."""
        span = tensor_span(dst)
        if span is None:
            return
        if not dst.is_contiguous():
            raise InputPrepHazardError(
                f"deferred_input_prep debug: non-contiguous staged destination at {_call_site()}"
            )
        if overwrite and self.level >= 2:
            dst.view(torch.uint8).fill_(POISON_BYTE)
            snapshot = None
        else:
            snapshot = dst.detach().clone()
        self.staged.append(_Staged(span[0], span[1], _call_site(), dst, snapshot))

    def check(self, what: str, values: Iterable) -> None:
        self.num_checked_ops += 1
        for v in values:
            if not isinstance(v, torch.Tensor):
                continue
            span = tensor_span(v)
            if span is None:
                continue
            for s in self.staged:
                if span[0] < s.hi and s.lo < span[1]:
                    raise InputPrepHazardError(
                        f"deferred_input_prep debug guard: {what} touches a destination staged at {s.label} "
                        "before the deferred_input_prep launch (it would read stale bytes or be overwritten)"
                    )

    def verify(self) -> None:
        """Raise if any staged destination changed between staging and now."""
        if not self.staged:
            return
        with self.internal():
            flags = []
            for s in self.staged:
                cur = s.view.view(torch.uint8)
                if s.snapshot is None:
                    flags.append((cur != POISON_BYTE).any())
                else:
                    flags.append((cur != s.snapshot.view(torch.uint8)).any())
            changed = torch.stack(flags).cpu().tolist()
        hits = [s.label for s, c in zip(self.staged, changed) if c]
        self.staged = []
        if hits:
            raise InputPrepHazardError(
                "deferred_input_prep debug guard: destinations staged at "
                f"{sorted(set(hits))} were written by another operation before the deferred_input_prep launch"
            )

    # ---- Triton ---------------------------------------------------------

    def _hook_triton(self) -> None:
        try:
            from triton.runtime.jit import JITFunction
        except ImportError:
            return
        window = self
        orig = JITFunction.run

        def run(fn_self, *args, **kwargs):
            if window.internal_depth == 0 and window.staged:
                name = getattr(fn_self, "__name__", "kernel")
                window.check(f"triton:{name}", list(args) + list(kwargs.values()))
            return orig(fn_self, *args, **kwargs)

        self._triton_jit, self._triton_run = JITFunction, orig
        JITFunction.run = run

    def _unhook_triton(self) -> None:
        if self._triton_jit is not None:
            self._triton_jit.run = self._triton_run
            self._triton_jit = self._triton_run = None
