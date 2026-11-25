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
"""Consume a batch of PyTorch CUDA symmetric-memory signal words on the current stream."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import globaltimer


@triton.jit
def _wait_many_kernel(pad, indices, TIMEOUT_NS: tl.constexpr):
    lane = tl.program_id(0) * 32 + tl.arange(0, 32)
    word = tl.load(indices + lane)
    pending = word >= 0  # The descriptor's final warp is padded with -1.
    start = globaltimer()
    while tl.sum(pending.to(tl.int32), 0) != 0:
        # PyTorch put_signal publishes 1 with a system-scope release CAS. Each lane
        # acquires and clears its own word once. Predicating the CAS avoids touching
        # padding or consuming a later signal on an already completed lane.
        old = tl.inline_asm_elementwise(
            asm="""{
                .reg .pred active;
                setp.ne.u32 active, $2, 0;
                mov.u32 $0, 1;
                @active atom.acquire.sys.global.cas.b32 $0, [$1], 1, 0;
            }""",
            constraints="=r,l,r",
            args=[pad + word, pending.to(tl.int32)],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
        pending = old != 1
        if TIMEOUT_NS:
            if globaltimer() - start > TIMEOUT_NS and tl.sum(pending.to(tl.int32), 0) != 0:
                tl.inline_asm_elementwise(
                    "trap; mov.u32 $0, 0;", "=r", [], dtype=tl.uint32, is_pure=False, pack=1
                )
    # Match CUDA symmetric memory's wait_signal_kernel before downstream readers run.
    tl.inline_asm_elementwise(
        "fence.sc.sys; mov.u32 $0, 0;", "=r", [], dtype=tl.uint32, is_pure=False, pack=1
    )


def wait_signal_words(pad: torch.Tensor, indices: torch.Tensor, timeout_ms: int) -> None:
    _wait_many_kernel[(indices.numel() // 32,)](pad, indices, timeout_ms * 1_000_000, num_warps=1)


def warm_wait_many(device: torch.device, timeout_ms: int) -> None:
    # Trigger JIT before a peer can send or wait on real flags. All batch lengths use
    # this binary: indices select the words and grid size selects the number of warps.
    pad = torch.ones(32, dtype=torch.int32, device=device)
    indices = torch.arange(32, dtype=torch.int32, device=device)
    wait_signal_words(pad, indices, timeout_ms)
    torch.cuda.current_stream(device).synchronize()
