# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A test-only step-input stager that runs a window into shadow buffers.

``ShadowStager`` records every write it is offered exactly as the model's
stager does, but aims the descriptors at device clones of the buffers taken
when the window opens, and declines every write so the caller's built-in
statement still runs on the real buffers. After ``end`` (one launch into the
clones), ``compare`` lists every word where a clone differs from the real
buffer the built-in code wrote.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    runtime,
)


class ShadowStager(runtime._PrepareStager):
    """``runtime._PrepareStager`` writing into clones of ``bases`` and declining every write.

    Args:
        bases: name -> the device buffers the window may write; a write outside
            them is an error.
        staging: the step-commit block to record into (reset here).
        runner, attn_metadata, spec_metadata: the objects the window is opened on.
    """

    def __init__(
        self,
        bases: dict[str, torch.Tensor],
        staging: Any,
        runner: Any,
        attn_metadata: Any = None,
        spec_metadata: Any = None,
    ):
        staging.reset()
        super().__init__(staging, Counter(), runner, attn_metadata, spec_metadata)
        # The static-input mirror reads the real descriptor addresses; it has no place here.
        self._watch = {}
        self.input_writes = {}
        self.real: dict[str, torch.Tensor] = {}
        self.clones: dict[str, torch.Tensor] = {}
        self._ranges: list[tuple[int, int, str]] = []
        for name, t in bases.items():
            assert t.is_cuda and t.is_contiguous(), name
            self.real[name] = t
            self.clones[name] = t.clone()
            self._ranges.append((t.data_ptr(), t.data_ptr() + t.numel() * t.element_size(), name))

    def _dst(self, addr: int, nbytes: int) -> int:
        for lo, hi, name in self._ranges:
            if lo <= addr and addr + nbytes <= hi:
                return self.clones[name].data_ptr() + (addr - lo)
        raise AssertionError(f"write of {nbytes} bytes at {addr:#x} outside the registered buffers")

    def _decline(self, reason: str) -> bool:
        # Nothing to order against: the clones are written only by this window's launch.
        self.declined.append(reason)
        return False

    def _mirror_static_inputs(self, inputs: dict) -> Optional[tuple]:
        return None

    def copy(self, dst: torch.Tensor, src: torch.Tensor) -> bool:
        super().copy(dst, src)
        return False

    def overlap_inputs(self, runner: Any, **kwargs: Any) -> bool:
        super().overlap_inputs(runner, **kwargs)
        return False

    def block_offsets(self, *args: Any) -> bool:
        super().block_offsets(*args)
        return False

    def compare(self) -> list[str]:
        """Every buffer where the clone differs from the built-in result, after ``end``."""
        torch.cuda.current_stream().synchronize()
        if self.declined:
            return [f"not compared: the window could not record {self.declined}"]
        bad = []
        for name, real in self.real.items():
            clone = self.clones[name]
            if name.endswith("kv_cache_block_offsets") and self.block_cols is not None:
                # Only the readable columns of the written rows are staged.
                n, c = self.block_seqs, self.block_cols
                real, clone = real[:, :n, :, :c], clone[:, :n, :, :c]
            if not torch.equal(real, clone):
                diff = (real != clone).nonzero()[:4].tolist()
                bad.append(f"{name} differs at {diff}")
        return bad
