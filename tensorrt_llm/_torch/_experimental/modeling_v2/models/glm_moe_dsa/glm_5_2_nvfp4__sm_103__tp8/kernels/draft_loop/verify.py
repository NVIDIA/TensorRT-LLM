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
"""In-pipeline native oracle for draft_loop (diagnostics, ``TRTLLM_MODELING_V2_VALIDATE=1``).

Each verified draft_loop operation first runs the native code on the live state,
records what it produced, restores the device state, runs the draft_loop operation and
counts, on the device, every field whose value differs. The comparison ops are
ordinary stream work, so they are captured into CUDA graphs and evaluated on
every replay. Counters are logged by the next eager draft_loop forward and at exit.
The draft_loop results are the ones kept.
"""

import atexit
import functools
from typing import Dict, List, Optional

import torch

from ... import dispatch

_MAX_SNAPSHOT_NUMEL = 1 << 22


@functools.lru_cache(maxsize=1)
def verify_enabled() -> bool:
    """Whether the installed plan runs the diagnostics; read on the first draft-loop step."""
    return dispatch.validate_mode()


class _Counters:
    """Device-side mismatch counters keyed by a field name.

    One eagerly allocated pool backs all counters, so a name first seen under
    stream capture takes a free slot instead of allocating (a captured
    allocation would be re-zeroed on every replay).
    """

    _POOL = 2048

    def __init__(self):
        self.names: List[str] = []
        self.slots: Dict[str, int] = {}
        self.pool: Optional[torch.Tensor] = None
        self.last_report: Optional[dict] = None

    def ensure_pool(self, device) -> None:
        if self.pool is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("draft_loop verify counters must be created before capture")
            self.pool = torch.zeros((self._POOL,), dtype=torch.int64, device=device)

    def get(self, name: str, device) -> torch.Tensor:
        self.ensure_pool(device)
        slot = self.slots.get(name)
        if slot is None:
            slot = len(self.names)
            if slot >= self._POOL:
                raise RuntimeError("draft_loop verify counter pool exhausted")
            self.slots[name] = slot
            self.names.append(name)
        return self.pool[slot]

    def add_mismatch(self, name: str, native: torch.Tensor, candidate: torch.Tensor) -> None:
        counter = self.get(name, native.device)
        if native.shape != candidate.shape or native.dtype != candidate.dtype:
            counter.add_(1)
        else:
            # Bitwise comparison: uninitialized float scratch may hold NaN patterns.
            counter.add_((_bits(native) != _bits(candidate)).any().to(torch.int64))

    def add_one(self, name: str, device) -> None:
        self.get(name, device).add_(1)

    def report(self, force: bool = False) -> None:
        if self.pool is None or torch.cuda.is_current_stream_capturing():
            return
        host = self.pool[: len(self.names)].tolist()
        values = dict(zip(self.names, host))
        if force or values != self.last_report:
            self.last_report = values
            calls = {
                k: v for k, v in values.items() if k.endswith(":calls") or k.endswith(":declined")
            }
            bad = {k: v for k, v in values.items() if v and k not in calls}
            print(
                f"[glm_moe_dsa] draft_loop verify: calls {calls} mismatching fields {bad or 'none'}",
                flush=True,
            )


def _bits(t: torch.Tensor) -> torch.Tensor:
    if t.is_floating_point() and t.element_size() in (1, 2, 4, 8):
        return t.view(
            {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()]
        )
    return t


COUNTERS = _Counters()


@atexit.register
def _report_at_exit() -> None:
    try:
        COUNTERS.report(force=True)
    except Exception:  # noqa: BLE001 - best effort at interpreter exit
        pass


def _device_fields(md) -> List[str]:
    names = []
    for name, value in vars(md).items():
        if (
            isinstance(value, torch.Tensor)
            and value.is_cuda
            and 0 < value.numel() <= _MAX_SNAPSHOT_NUMEL
        ):
            names.append(name)
    return sorted(names)


def _native_step_tail(
    md,
    i,
    batch_size,
    num_contexts,
    runtime_draft_len,
    num_accepted_tokens,
    position_ids,
    gather_ids,
):
    """The native per-step tail of ``Eagle3OneModelWorker._forward_linear_draft_loop``."""
    from tensorrt_llm._torch.speculative.mtp import _select_mtp_position_ids

    new_position_ids = None
    if position_ids is not None:
        rows = gather_ids if i == 0 else slice(0, batch_size)
        new_position_ids = _select_mtp_position_ids(position_ids, rows) + 1
    if i == 0:
        md._seq_lens[:batch_size].fill_(1)
        md._seq_lens_cuda[:batch_size].fill_(1)
        md.on_update()
        if hasattr(md, "host_request_types"):
            md.host_request_types[: md.num_contexts].fill_(1)
        md.num_contexts = 0
        md.kv_lens_cuda[num_contexts:batch_size] -= (
            runtime_draft_len - num_accepted_tokens[num_contexts:]
        )
        md.kv_lens_cuda[:num_contexts] += 1
        md.update_for_spec_dec()
        md.use_spec_decoding = False
    else:
        md.kv_lens_cuda[:batch_size] += 1
        md.update_for_spec_dec()
    return new_position_ids


def verified_step_update(
    run_draft_loop, md, step, batch_size, draft_len, num_accepted_tokens, position_ids, gather_ids
):
    """Run the native tail, restore, run ``run_draft_loop()`` and count differences."""
    names = _device_fields(md)
    before = {name: getattr(md, name).clone() for name in names}
    native_pos = _native_step_tail(
        md, step, batch_size, 0, draft_len, num_accepted_tokens, position_ids, gather_ids
    )
    after_native = {name: getattr(md, name).clone() for name in names}
    for name in names:
        current = getattr(md, name)
        if current.shape == before[name].shape:
            current.copy_(before[name])
    result = run_draft_loop()
    if result is None:
        # draft_loop declined after the native tail already ran: keep the native state.
        for name in names:
            current = getattr(md, name)
            if current.shape == after_native[name].shape:
                current.copy_(after_native[name])
        COUNTERS.add_one("step:declined", md.kv_lens_cuda.device)
        return None
    device = md.kv_lens_cuda.device
    COUNTERS.add_one("step:calls", device)
    for name in names:
        COUNTERS.add_mismatch(f"step:{name}", after_native[name], getattr(md, name))
    if native_pos is not None:
        COUNTERS.add_mismatch("step:position_ids", native_pos, result.position_ids)
    return result


def verify_accept(
    worker,
    logits,
    spec_metadata,
    batch_size,
    draft_len,
    accepted,
    num_accepted,
    gather_ids,
    input_ids0,
) -> None:
    """Compare the fused acceptance with the native one (greedy batches only)."""
    if not spec_metadata.is_all_greedy_sample or worker.force_num_accepted_tokens != 0.0:
        return
    device = accepted.device
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    draft = spec_metadata.draft_tokens.reshape(batch_size, draft_len)
    acc_n, num_n = worker._sample_and_accept_draft_tokens_base(
        logits, draft, 0, batch_size, spec_metadata
    )
    COUNTERS.add_one("accept:calls", device)
    COUNTERS.add_mismatch("accept:accepted_tokens", acc_n, accepted)
    COUNTERS.add_mismatch("accept:num_accepted", num_n, num_accepted)
    COUNTERS.add_mismatch("accept:input_ids0", acc_n.reshape(-1), input_ids0)
    rows = (spec_metadata.batch_indices_cuda[:batch_size] * (draft_len + 1)).long() + num_n - 1
    COUNTERS.add_mismatch("accept:gather_ids", rows, gather_ids)


def verify_rows(hidden, gather_ids, rows) -> None:
    COUNTERS.add_one("rows:calls", rows.device)
    COUNTERS.add_mismatch("rows:hidden", hidden[gather_ids], rows)


def verify_finalize(
    worker, draft_tokens, accepted, num_accepted, batch_indices, batch_size, next_draft, next_new
) -> None:
    stacked = torch.stack(draft_tokens, dim=1)
    expected_new = worker._prepare_next_new_tokens(
        accepted, stacked, batch_indices, batch_size, num_accepted
    )
    COUNTERS.add_one("finalize:calls", next_draft.device)
    COUNTERS.add_mismatch("finalize:next_draft", stacked, next_draft)
    COUNTERS.add_mismatch("finalize:next_new", expected_new, next_new)
