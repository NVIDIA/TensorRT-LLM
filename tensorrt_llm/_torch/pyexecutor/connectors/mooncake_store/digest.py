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
"""Debug-only page digests: is a loaded page the bytes that were saved under its key?

Off unless `TRTLLM_MC_PAGE_DIGEST_DIR` names a directory. Each rank then appends
one JSON line per page to `<dir>/pages-<host>-<pid>-rank<r>.jsonl`:

* ``save``  -- the bytes the save thread is about to hand the store, read after
  the pass's event, so exactly what a zero-copy put reads.
* ``final`` -- the same pages again when the request retires (its pages are
  still pinned for the saves then). A full-attention page or an in-window
  sliding-window page that differs from its ``save`` digest was written after
  it was saved: a late write, e.g. drafter KV written one pass late, makes the
  stored copy stale. A passed sliding-window block may legitimately differ,
  since its page left the request when the window passed it.
* ``load``  -- the bytes a synchronous load put on the device, read right after
  the store reports the transfer complete. A ``load`` digest that differs from
  the ``save`` digest of the same key means the transfer or the addressing is
  wrong.

Two cold servers that compute the same prompt the same way also save the same
digest per key, so the log also says whether saving is deterministic.
`lanes/mc/hecate/page_digest_compare.py` joins the files.

Every page is copied to the host and hashed, so this costs a device-to-host
copy of all saved and loaded KV. Never enable it for performance runs.
"""

import hashlib
import json
import os
import socket
import threading
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from tensorrt_llm.logger import logger

#: Directory the per-rank digest logs go to; unset or empty disables them.
PAGE_DIGEST_ENV = "TRTLLM_MC_PAGE_DIGEST_DIR"

#: (key, layer group, block ordinal within the request's saves, addresses, sizes)
PageRecord = Tuple[str, int, int, List[int], List[int]]


def read_device_bytes(addresses: Sequence[int], sizes: Sequence[int]) -> List[bytes]:
    """Copy each device region to the host, synchronously."""
    import torch

    try:
        from cuda.bindings import runtime as cudart
    except ImportError:
        from cuda import cudart

    out = []
    for address, size in zip(addresses, sizes):
        host = torch.empty(int(size), dtype=torch.uint8)
        status = cudart.cudaMemcpy(
            host.data_ptr(),
            int(address),
            int(size),
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )[0]
        if status != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaMemcpy failed with {status} reading a KV page at {address:#x}")
        out.append(host.numpy().tobytes())
    return out


def page_digest(chunks: Sequence[bytes]) -> str:
    h = hashlib.blake2b(digest_size=16)
    for chunk in chunks:
        h.update(chunk)
    return h.hexdigest()


class PageDigestLog:
    """Appends page digests for one rank; see the module docstring."""

    def __init__(
        self,
        directory: str,
        rank: int,
        reader: Optional[Callable[[Sequence[int], Sequence[int]], List[bytes]]] = None,
    ) -> None:
        os.makedirs(directory, exist_ok=True)
        self.path = os.path.join(
            directory, f"pages-{socket.gethostname()}-{os.getpid()}-rank{rank}.jsonl"
        )
        self._rank = rank
        # Looked up per call, so a test can stand in for the device.
        self._reader = reader or (lambda addresses, sizes: read_device_bytes(addresses, sizes))
        self._lock = threading.Lock()
        self._file = open(self.path, "a", buffering=1)
        #: Saved pages per request, re-read when the request retires.
        self._saved: Dict[int, List[PageRecord]] = {}
        self._next_ordinal: Dict[int, Dict[bytes, int]] = {}
        logger.warning(
            f"mooncake-store rank {rank}: page digests ON ({PAGE_DIGEST_ENV}), writing "
            f"{self.path}. Every saved and loaded page is copied to the host: debug only."
        )

    @classmethod
    def from_env(cls, rank: int) -> Optional["PageDigestLog"]:
        directory = os.environ.get(PAGE_DIGEST_ENV, "")
        return cls(directory, rank) if directory else None

    def ordinal(self, request_id: int, block_hash: bytes) -> int:
        """Position of `block_hash` among the request's saved blocks, in first-seen order."""
        with self._lock:
            seen = self._next_ordinal.setdefault(request_id, {})
            return seen.setdefault(block_hash, len(seen))

    def _write(self, kind: str, request_id: int, record: PageRecord) -> None:
        key, group, ordinal, addresses, sizes = record
        digest = page_digest(self._reader(addresses, sizes))
        line = json.dumps(
            {
                "kind": kind,
                "rank": self._rank,
                "req": request_id,
                "ord": ordinal,
                "group": group,
                "key": key,
                "digest": digest,
                "t": round(time.time(), 6),
            }
        )
        with self._lock:
            self._file.write(line + "\n")

    def saved(self, request_id: int, records: Sequence[PageRecord]) -> None:
        """Digest pages about to be saved, and keep them for the retirement re-read."""
        for record in records:
            self._write("save", request_id, record)
        with self._lock:
            self._saved.setdefault(request_id, []).extend(records)

    def saved_host(self, pages, addresses, sizes) -> None:
        """Digest save snapshots, which already sit in pinned host slots.

        Not kept for the retirement re-read: the device page may legitimately
        change once the snapshot is taken.
        """
        import ctypes

        for (request_id, key), page_addresses, page_sizes in zip(pages, addresses, sizes):
            chunks = [ctypes.string_at(a, s) for a, s in zip(page_addresses, page_sizes)]
            line = json.dumps(
                {
                    "kind": "save",
                    "rank": self._rank,
                    "req": request_id,
                    "ord": -1,
                    "group": -1,
                    "key": key,
                    "digest": page_digest(chunks),
                    "t": round(time.time(), 6),
                    "snapshot": True,
                }
            )
            with self._lock:
                self._file.write(line + "\n")

    def loaded(self, request_id: int, records: Sequence[PageRecord]) -> None:
        for record in records:
            self._write("load", request_id, record)

    def retired(self, request_id: int) -> None:
        """Re-read the request's saved pages; call while they are still pinned."""
        with self._lock:
            records = self._saved.pop(request_id, [])
            self._next_ordinal.pop(request_id, None)
        for record in records:
            self._write("final", request_id, record)

    def close(self) -> None:
        with self._lock:
            if not self._file.closed:
                self._file.close()
