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
"""Teacher-forced replay of recorded token streams under forced acceptance (benchmark only).

Forced acceptance (``TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS``) commits the
target's tokens whatever the drafts were, so a request's stream is a chaotic
function of the numerics: one near-tie that flips between two runs (KV from the
context server, atomic reductions) sends the request into another stream with
other routed experts. ``TLLM_SPEC_DECODE_FORCE_REPLAY=<file>`` cuts that
feedback: a generation request that follows a recorded stream commits the
recorded tokens instead of the target's argmax, and the recorded continuation
becomes its next drafts (a drafter that is always right). The target, the
drafter and every kernel still run with the same shapes; only token values
change, so every step's inputs depend on the request alone and runs of one
configuration replay the same streams. It needs
``TLLM_SPEC_DECODE_FORCE_ACCEPT_KEY=state`` with a fractional forced count; the
accepted lengths keep their draws (keyed by the replayed tokens).

The file is a ``TLLM_SPEC_DECODE_STREAM_LOG`` record file (one JSON object per
line with ``prompt_len`` and ``token_ids``, the first id being the context
phase's token) of a run without forced acceptance, so the recorded streams are
the model's own continuations. Stream ``r`` covers the global positions
``prompt_len[r] + j`` for ``0 <= j < len[r]``. Every step a request is matched
again from its verify group (its newest committed token at global position
``p``, then its drafts): it follows stream ``r`` when the group equals
``r``'s tokens from ``j = p - prompt_len[r]`` on (the group of every step after
the first, whose drafts were taken from ``r``), or else, at its first step,
when ``j == 0`` and the newest committed token (the context phase's token) is
``r``'s first token. The lowest such ``r`` wins; a request without one keeps
the target's tokens and the drafter's drafts.
"""

import json
import os
from typing import List, NamedTuple, Optional, Tuple

import torch
import triton
import triton.language as tl

FORCE_REPLAY_ENV_VAR = "TLLM_SPEC_DECODE_FORCE_REPLAY"
#: ``length``: a request also starts replaying stream ``r`` when it reaches ``r``'s prompt length with another
#: context-phase token, if no other stream has that prompt length (default ``token``: the first-token rule only).
FORCE_REPLAY_START_ENV_VAR = "TLLM_SPEC_DECODE_FORCE_REPLAY_START"
#: Most streams one file may hold (every row is compared with every stream).
MAX_REFERENCE_STREAMS = 16


def force_replay_path() -> Optional[str]:
    """The reference file of ``TLLM_SPEC_DECODE_FORCE_REPLAY``, or None."""
    return os.environ.get(FORCE_REPLAY_ENV_VAR) or None


def replay_start_by_length() -> bool:
    """Whether ``TLLM_SPEC_DECODE_FORCE_REPLAY_START=length`` lets a request start replaying the only stream with
    its prompt length whatever its context-phase token."""
    return os.environ.get(FORCE_REPLAY_START_ENV_VAR, "token").strip().lower() == "length"


def _unique_lengths(streams: List[Tuple[int, List[int]]]) -> List[bool]:
    bases = [b for b, _ in streams]
    return [bases.count(b) == 1 for b in bases]


def load_reference_streams(path: str) -> List[Tuple[int, List[int]]]:
    """``(prompt_len, token_ids)`` of the records in a stream-log file, in file order, without
    later records of an already seen ``(prompt_len, first token)``."""
    streams, seen = [], set()
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            tokens = [int(t) for t in rec.get("token_ids") or []]
            if not tokens:
                continue
            key = (int(rec["prompt_len"]), tokens[0])
            if key not in seen:
                seen.add(key)
                streams.append((key[0], tokens))
    if not streams or len(streams) > MAX_REFERENCE_STREAMS:
        raise ValueError(
            f"{FORCE_REPLAY_ENV_VAR}: {path} holds {len(streams)} streams with token ids, "
            f"1 to {MAX_REFERENCE_STREAMS} are supported"
        )
    return streams


def start_reference(
    streams: List[Tuple[int, List[int]]], prompt_len: int, first_token: int, by_length: bool = False
) -> int:
    """The stream a request with this prompt length and context-phase token starts replaying
    (the first-step rule of the device lookup; ``by_length``: also the only stream with this prompt
    length), or -1."""
    unique = _unique_lengths(streams)
    for r, (base, tokens) in enumerate(streams):
        if base == prompt_len and (tokens[0] == first_token or (by_length and unique[r])):
            return r
    return -1


class ForceReplay(NamedTuple):
    """Device tables of the replay: the recorded streams and the per-row drafts the accept
    kernel leaves for the next-new-tokens kernel.

    ``ref_tok[r]`` holds stream ``r`` followed by copies of its last token (so the groups and
    drafts of its last steps stay inside the row); ``drafts[g]`` / ``flags[g]`` belong to
    generation row ``g`` of the current step and are rewritten by every replaying accept
    launch.
    """

    ref_base: torch.Tensor  # [R] int32, global position of each stream's first token
    ref_len: torch.Tensor  # [R] int32, recorded tokens per stream
    ref_tok: torch.Tensor  # [R, width] int32
    drafts: torch.Tensor  # [max_rows, max_draft_len] int32
    flags: torch.Tensor  # [max_rows] int32, 1 when row g replays this step
    # [R] int32, 1 when a request at stream r's prompt length starts it whatever its context token
    # (TLLM_SPEC_DECODE_FORCE_REPLAY_START=length and no other stream has that length).
    ref_start: Optional[torch.Tensor] = None

    @property
    def num_refs(self) -> int:
        return self.ref_tok.shape[0]

    @staticmethod
    def build(
        streams: List[Tuple[int, List[int]]],
        device: torch.device,
        max_rows: int,
        max_draft_len: int,
        start_by_length: bool = False,
    ) -> "ForceReplay":
        pad = 2 * (max_draft_len + 2)
        width = max(len(t) for _, t in streams) + pad
        tok = torch.empty((len(streams), width), dtype=torch.int32)
        for r, (_, tokens) in enumerate(streams):
            tok[r, : len(tokens)] = torch.tensor(tokens, dtype=torch.int32)
            tok[r, len(tokens) :] = tokens[-1]
        return ForceReplay(
            ref_base=torch.tensor([b for b, _ in streams], dtype=torch.int32).to(device),
            ref_len=torch.tensor([len(t) for _, t in streams], dtype=torch.int32).to(device),
            ref_tok=tok.to(device),
            drafts=torch.zeros((max_rows, max_draft_len), dtype=torch.int32, device=device),
            flags=torch.zeros((max_rows,), dtype=torch.int32, device=device),
            ref_start=torch.tensor(
                [int(start_by_length and u) for u in _unique_lengths(streams)], dtype=torch.int32
            ).to(device),
        )


@triton.jit
def replay_lookup(
    ref_base_ptr,
    ref_len_ptr,
    ref_tok_ptr,
    ref_stride,
    ref_start_ptr,
    num_refs,
    pos,
    group_ptr,
    group_stride,
    num_rows,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """For rows ``0 .. num_rows - 1`` with verify groups ``group_ptr[row * group_stride + c]``
    (``c < WIDTH``) whose first token sits at global position ``pos[row]`` (int64): the stream
    each row follows (-1 for none, see the module docstring), the offset ``j`` of the newest
    committed token in it and the stream's length."""
    rows = tl.arange(0, BLOCK)
    row_ok = rows < num_rows
    c = tl.arange(0, BLOCK_W)
    col_ok = c < WIDTH
    group = tl.load(
        group_ptr + rows[:, None].to(tl.int64) * group_stride + c[None, :],
        mask=row_ok[:, None] & col_ok[None, :],
        other=-1,
    )
    r = tl.arange(0, BLOCK_R)
    ref_ok = r < num_refs
    base = tl.load(ref_base_ptr + r, mask=ref_ok, other=0).to(tl.int64)
    length = tl.load(ref_len_ptr + r, mask=ref_ok, other=0).to(tl.int64)
    j = pos[:, None] - base[None, :]
    inside = row_ok[:, None] & ref_ok[None, :] & (j >= 0) & (j < length[None, :])
    rec = tl.load(
        ref_tok_ptr
        + r[None, :, None].to(tl.int64) * ref_stride
        + tl.where(inside, j, 0)[:, :, None]
        + c[None, None, :],
        mask=inside[:, :, None] & col_ok[None, None, :],
        other=-2,
    )
    same = (rec == group[:, None, :]) | ~col_ok[None, None, :]
    full = inside & (tl.min(same.to(tl.int32), axis=2) > 0)
    head = tl.sum(tl.where(c[None, None, :] == 0, rec, 0), axis=2)
    any_token = tl.load(ref_start_ptr + r, mask=ref_ok, other=0) != 0
    start = (
        inside
        & (j == 0)
        & (
            (head == tl.sum(tl.where(c[None, :] == 0, group, 0), axis=1)[:, None])
            | any_token[None, :]
        )
    )
    r_full = tl.min(tl.where(full, r[None, :], BLOCK_R), axis=1)
    r_start = tl.min(tl.where(start, r[None, :], BLOCK_R), axis=1)
    rk = tl.where(r_full < BLOCK_R, r_full, tl.where(r_start < BLOCK_R, r_start, -1))
    sel = r[None, :] == rk[:, None]
    base_k = tl.sum(tl.where(sel, base[None, :], 0), axis=1)
    len_k = tl.sum(tl.where(sel, length[None, :], 0), axis=1)
    return rk, pos - base_k, len_k


def apply_replay_drafts(
    next_draft_tokens: torch.Tensor, replay: ForceReplay, row0: int, rows: int
) -> None:
    """The next-new-tokens kernel's replay (the flagged generation rows take the recorded
    drafts) on ``next_draft_tokens`` in place, for a path that builds the next inputs with
    PyTorch ops."""
    k = next_draft_tokens.shape[-1]
    gen = next_draft_tokens[row0 : row0 + rows]
    flagged = replay.flags[:rows, None] != 0
    gen.copy_(torch.where(flagged, replay.drafts[:rows, :k].to(gen.dtype), gen))
