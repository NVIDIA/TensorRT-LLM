# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU-owned planning bounds, checked against authoritative device metadata."""

import torch


def _validate_host_length_hints(hints, batches, capacity, rows):
    if hints is None:
        return None
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "host_length_hints must not be used during CUDA graph capture"
        )
    if not isinstance(hints, tuple) or len(hints) != 2:
        raise ValueError("host_length_hints must be (CPU query prefix, CPU KV bounds)")
    prefix, bounds = hints
    if (
        any(
            not isinstance(t, torch.Tensor)
            or t.device.type != "cpu"
            or t.dtype not in (torch.int32, torch.int64)
            or t.ndim != 1
            or not t.is_contiguous()
            for t in hints
        )
        or prefix.numel() != batches + 1
        or bounds.numel() != batches
    ):
        raise ValueError(
            "host_length_hints require contiguous CPU integer [B+1] and [B]"
        )
    q, kv = prefix.tolist(), bounds.tolist()
    if (
        q[0] != 0
        or q[-1] > rows
        or any(a > b for a, b in zip(q, q[1:], strict=False))
        or any(not 0 <= length <= capacity.max_context_length for length in kv)
    ):
        raise ValueError(
            "host_length_hints must fit the constructed capacity and Q extent"
        )
    return q, kv


class _HostHintValidation:
    """Allocation-free device checks; bad hints use prefix-status bit 2."""

    def __init__(self, capacity, device):
        batches = capacity.max_batch_size
        self.prefix = torch.empty(batches + 1, dtype=torch.int32, device=device)
        self.bounds = torch.empty(batches, dtype=torch.int32, device=device)
        self.lengths = torch.empty_like(self.bounds)
        self.q_bad = torch.empty(batches + 1, dtype=torch.bool, device=device)
        self.kv_bad = torch.empty(batches, dtype=torch.bool, device=device)
        self.bad = torch.empty(1, dtype=torch.bool, device=device)
        self.other_bad = torch.empty_like(self.bad)
        self.code = torch.empty(1, dtype=torch.int32, device=device)
        self.batches = None

    def stage(self, hints):
        self.batches = None if hints is None else hints[1].numel()
        if hints is not None:
            batches = self.batches
            # CPU tensors are caller-owned; complete these H2D copies before
            # returning so their lifetime/mutation needs no additional contract.
            self.prefix[: batches + 1].copy_(hints[0])
            self.bounds[:batches].copy_(hints[1])

    def check(self, state):
        if self.batches is None:
            return
        b = self.batches
        torch.ne(
            state.qo_indptr[: b + 1], self.prefix[: b + 1], out=self.q_bad[: b + 1]
        )
        torch.sub(state.kv_indptr[1 : b + 1], state.kv_indptr[:b], out=self.lengths[:b])
        torch.gt(self.lengths[:b], self.bounds[:b], out=self.kv_bad[:b])
        torch.any(self.q_bad[: b + 1], out=self.bad.view(()))
        torch.any(self.kv_bad[:b], out=self.other_bad.view(()))
        torch.logical_or(self.bad, self.other_bad, out=self.bad)
        torch.mul(self.bad, 2, out=self.code)
        state.error.bitwise_or_(self.code)
        # Never substitute the hints for real lengths, or truncate a score row.
        for tensor in (
            state.safe_qo_indptr,
            state.physical_kv_indptr,
            state.active_shape,
            state.row_ends,
        ):
            tensor.masked_fill_(self.bad, 0)
        state.row_requests.masked_fill_(self.bad, -1)
