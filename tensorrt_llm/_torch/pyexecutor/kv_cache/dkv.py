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
"""Distributed KV (DKV): one copy of each KV page, shared by a group of ranks that run the same batch.

Every rank of the group holds every request and computes every token's KV, but the group stores
each physical page once. Its layout is ``page_split``, Helix's super-block layout: a DKV block is ``group size``
physical pages of ``tokens_per_block`` tokens, and global page ``g`` lives on rank ``g % size``.
Each rank's KV cache manager therefore allocates one slot per DKV block, which holds its own
page of the block (``KVCacheManagerV2(dkv=True)``).

A rank that reads another rank's page needs that rank's slot ids, which need not equal its own:
the native slot allocator prefers recycled slots whose GPU work has completed, and that differs
between ranks. ``Dkv.exchange_pages`` all-gathers the slots each rank allocated since the last
exchange, once per step, so every rank holds every rank's page table of every live request
(``peer_pages``). ``Dkv.agree`` takes the group minimum of a count, which keeps the prefix
every rank reuses the same.

The exchange orders host code, not GPU work. A page an owner copies in this step's cache
preparation (restored from the host tier, or the private copy of a partially reused block) lands
asynchronously on the owner's execution stream, so a peer must not read it before the group meets
on the GPU. Each rank sends whether it made such copies (``resume_copies``) with its page tables;
``Dkv.take_fence`` then tells the reader to run that GPU fence before its first read of the step.
"""

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Hashable, List, Optional, Sequence, Tuple

from tensorrt_llm._torch.utils import helix_local_len
from tensorrt_llm.logger import logger

#: ``(key, first DKV block, slot ids)``: one rank's new page-table entries of a page table, keyed
#: by ``(request id, life cycle)``.
PageUpdate = Tuple[Hashable, int, List[int]]


@dataclass
class Dkv:
    """The group that shares DKV, and every rank's page tables of the live requests.

    Args:
        size: Ranks of the group; a DKV block is ``size`` physical pages.
        rank: This rank's index in the group: it holds global page ``g`` when
            ``g % size == rank``.
        tokens_per_block: Tokens of one physical page.
        all_gather: Exchanges one picklable object per rank: every rank of the group calls it at
            the same point and gets the objects of all ranks, in group-rank order.
        check: Also check, at every exchange, that the ranks hold the same requests with the same
            numbers of blocks, and log the DKV blocks whose slot ids differ between ranks.
    """

    size: int
    rank: int
    tokens_per_block: int
    all_gather: Callable[[object], List[object]]
    check: bool = False
    # (request id, life cycle) -> one list of slot ids per rank, indexed by DKV block.
    _peer_pages: Dict[Hashable, List[List[int]]] = field(default_factory=dict)
    # A rank of the group copied pages since the last ``take_fence`` (see ``exchange_pages``).
    _fence: bool = False

    def __post_init__(self) -> None:
        if self.size < 2 or not 0 <= self.rank < self.size:
            raise ValueError(f"DKV rank {self.rank} is not in a group of {self.size}")

    @property
    def dkv_tokens_per_block(self) -> int:
        """Tokens of one DKV block: one physical page on every rank of the group."""
        return self.tokens_per_block * self.size

    def owner(self, position: int) -> int:
        """The rank that stores the token at global ``position``."""
        return (position // self.tokens_per_block) % self.size

    def local_len(self, global_len: int, rank: Optional[int] = None) -> int:
        """Tokens of the first ``global_len`` that ``rank`` (default: this rank) stores.

        For a token at ``position`` that ``rank`` stores, ``local_len(position, rank)`` is its
        index in the rank's own pages: page ``position // dkv_tokens_per_block``, row
        ``position % tokens_per_block``.
        """
        return helix_local_len(
            global_len, self.tokens_per_block, self.size, self.rank if rank is None else rank
        )

    def round_down(self, tokens: int) -> int:
        """``tokens`` rounded down to whole DKV blocks."""
        return tokens // self.dkv_tokens_per_block * self.dkv_tokens_per_block

    def agree(self, value: int) -> int:
        """The smallest ``value`` of the group (a collective)."""
        return min(int(v) for v in self.all_gather(int(value)))

    def agree_all(self, values: Sequence[Optional[int]]) -> List[Optional[int]]:
        """The group minimum of each of ``values`` with one all-gather (a collective).

        Every rank passes the same number of values; a None takes no part in its minimum (None
        when every rank passes None).
        """
        everyone = self.all_gather([None if v is None else int(v) for v in values])
        if any(len(theirs) != len(values) for theirs in everyone):
            raise RuntimeError(
                f"DKV agree: the ranks passed {[len(t) for t in everyone]} values, not "
                f"{len(values)} each"
            )
        agreed = []
        for column in zip(*everyone):
            given = [v for v in column if v is not None]
            agreed.append(min(given) if given else None)
        return agreed

    def exchange_pages(self, updates: Sequence[PageUpdate], copied: bool = False) -> None:
        """All-gather the new page-table entries of every rank (a collective, once per step).

        Args:
            updates: This rank's ``(key, first DKV block, slot ids)`` of the page tables that
                changed since the last exchange, from their first changed block; every rank lists
                the same keys and blocks, in the same order, since they run the same batch.
            copied: Whether this rank copied pages of the batch since the last exchange
                (``resume_copies``); if any rank did, ``take_fence`` returns True.
        """
        mine = [(key, int(start), [int(s) for s in slots]) for key, start, slots in updates]
        gathered = self.all_gather((mine, bool(copied)))
        if len(gathered) != self.size:
            raise RuntimeError(f"DKV exchange got {len(gathered)} ranks for {self.size}")
        everyone = [theirs for theirs, _ in gathered]
        self._fence = self._fence or any(c for _, c in gathered)
        if self.check:
            self._check(everyone)
        for rank, rank_updates in enumerate(everyone):
            for key, start, slots in rank_updates:
                tables = self._peer_pages.setdefault(key, [[] for _ in range(self.size)])
                table = tables[rank]
                if len(table) < start:
                    raise RuntimeError(
                        f"DKV: rank {rank} sent blocks from {start} of {key}, which "
                        f"has {len(table)}"
                    )
                del table[start:]
                table.extend(slots)

    def _check(self, everyone: Sequence[Sequence[PageUpdate]]) -> None:
        shapes = [[(key, start, len(slots)) for key, start, slots in u] for u in everyone]
        if any(s != shapes[0] for s in shapes[1:]):
            raise RuntimeError(f"DKV: the ranks allocated different blocks: {shapes}")
        differ = sum(
            1
            for column in zip(*everyone)
            for blocks in zip(*(slots for _, _, slots in column))
            if len(set(blocks)) > 1
        )
        if differ:
            logger.debug(f"DKV: {differ} new DKV blocks have different slot ids per rank")

    def take_fence(self) -> bool:
        """Whether a rank copied pages since the last call (the flag is cleared).

        Every rank gets the same answer: the flags travel in the exchange. True means a page an
        owner copied this step may still be in flight on the owner's execution stream: before its
        first read of another rank's pages, every rank runs one GPU collective of the group on its
        execution stream (which waits for those copies on the owners), and issues the reads after
        it.
        """
        fence, self._fence = self._fence, False
        return fence

    def peer_pages(self, key: Hashable) -> List[List[int]]:
        """Every rank's slot ids of a page table, one list per rank, indexed by DKV block."""
        return self._peer_pages[key]

    def page_of(self, key: Hashable, position: int) -> Tuple[int, int, int]:
        """``(rank, slot, row)`` of the token at global ``position`` of a page table."""
        owner = self.owner(position)
        block = position // self.dkv_tokens_per_block
        return owner, self._peer_pages[key][owner][block], position % self.tokens_per_block

    def forget(self, request_id: int) -> None:
        """Drop the page tables of a finished request (every life cycle)."""
        for key in [k for k in self._peer_pages if k[0] == request_id]:
            del self._peer_pages[key]


def resume_copies(kv_cache, fresh: bool) -> bool:
    """Whether a resume of ``kv_cache`` (the native cache) may have copied pages that another
    rank reads this step.

    A fresh cache (resumed for the first time) copies its reused pages from the host tier
    (``cached_tokens_by_level`` past the GPU level), and the page of a block its match ends inside
    into a private slot (partial reuse). A cache resumed again after a suspension may get its pages
    back from the host tier; it counts once it holds any history. Both copies run on the cache's
    stream, which the execution stream follows.

    Args:
        kv_cache: The native cache, resumed.
        fresh: Whether this was its first resume.
    """
    if not fresh:
        return int(kv_cache.history_length) > 0
    if any(int(n) for n in list(kv_cache.cached_tokens_by_level)[1:]):
        return True
    return int(kv_cache.num_committed_tokens) % int(kv_cache.manager.tokens_per_block) != 0


def gpu_tier_chunk_bytes(quota: int) -> int:
    """The physical chunk the native GPU tier maps its pools from, for a tier of ``quota`` bytes.

    Mirrors ``tierAllocationGranularity`` (storage/core.cpp): 2 MiB, doubled per doubling of the
    quota past 1 GiB, up to 32 MiB. A pool is mapped from whole chunks, one allocation each, from
    its base address on.
    """
    page = 2 << 20
    ratio = int(quota) // (page * 512)
    exponent = 0 if ratio == 0 else min(4, ratio.bit_length() - 1)
    return page << exponent


@dataclass
class PeerRegion:
    """Every rank's copy of one GPU pool, mapped into one virtual range of this rank.

    Rank ``r``'s pool starts at ``base + r * stride``; ``stride`` is a multiple of ``row_bytes``, so
    a page of ``row_bytes`` is row ``r * stride // row_bytes + i`` of the whole range.
    """

    base: int
    stride: int
    size: int
    row_bytes: int
    reserved: int = 0  # bytes of the virtual range
    # (address, size) of every chunk this rank mapped, for unmap.
    _mapped: List[Tuple[int, int]] = field(default_factory=list)
    _handles: List[object] = field(default_factory=list)

    @property
    def rows_per_rank(self) -> int:
        return self.stride // self.row_bytes


def map_peer_regions(
    dkv: Dkv, regions: Sequence[Tuple[int, int]], chunk_bytes: int, row_bytes: int
) -> List[PeerRegion]:
    """Map every rank's copy of some GPU pools into this rank's address space (a collective).

    Args:
        DKV: The group; its ``all_gather`` exchanges the handles once.
        regions: This rank's ``(base, size)`` of each pool to share. Every rank passes the same
            sizes (the managers are built from the same config), and the pools never move or grow
            afterwards (no pool rebalancing).
        chunk_bytes: The native chunk the pools are mapped from (``gpu_tier_chunk_bytes``).
        row_bytes: The unit a reader addresses (a page of one layer).

    The pools' chunks are allocated with fabric handles where the device offers them
    (``PooledPhysMemAllocator``); each is exported, imported by every peer and mapped read-write.
    This rank's own chunks are mapped a second time too, so one index addresses any rank's page.
    """
    try:
        from cuda.bindings import driver as cuda
    except ImportError:
        from cuda import cuda

    def check(ret):
        err, *rest = ret if isinstance(ret, tuple) else (ret,)
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"DKV peer mapping: {err}")
        return rest[0] if len(rest) == 1 else rest

    fabric = cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
    device = int(check(cuda.cuCtxGetDevice()))
    access = cuda.CUmemAccessDesc()
    access.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    access.location.id = device
    access.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    unit = math.lcm(chunk_bytes, row_bytes)
    out = []
    for base, size in regions:
        chunks = -(-int(size) // chunk_bytes)
        mine = []
        for c in range(chunks):
            handle = check(cuda.cuMemRetainAllocationHandle(int(base) + c * chunk_bytes))
            mine.append(bytes(check(cuda.cuMemExportToShareableHandle(handle, fabric, 0)).data))
            check(cuda.cuMemRelease(handle))
        everyone = dkv.all_gather(mine)
        if any(len(h) != chunks for h in everyone):
            raise RuntimeError("DKV peer mapping: the ranks' pools differ in size")
        stride = -(-chunks * chunk_bytes // unit) * unit
        va = int(check(cuda.cuMemAddressReserve(stride * dkv.size, chunk_bytes, 0, 0)))
        region = PeerRegion(va, stride, int(size), row_bytes, stride * dkv.size)
        for rank, handles in enumerate(everyone):
            for c, data in enumerate(handles):
                handle = check(cuda.cuMemImportFromShareableHandle(data, fabric))
                address = va + rank * stride + c * chunk_bytes
                check(cuda.cuMemMap(address, chunk_bytes, 0, handle, 0))
                check(cuda.cuMemSetAccess(address, chunk_bytes, [access], 1))
                region._mapped.append((address, chunk_bytes))
                region._handles.append(handle)
        out.append(region)
    return out


def peer_region_tensor(region: PeerRegion):
    """The whole mapped range of ``region`` as a uint8 tensor on this rank's device.

    Built through DLPack with the device named: ``torch.as_tensor`` would take the device from the
    pointer's attributes, which for imported memory name the owner's GPU.
    """
    import torch

    from tensorrt_llm._dlpack_utils import pack_strided_memory

    count = region.reserved // region.stride
    return pack_strided_memory(
        region.base, region.stride, region.stride, count, torch.uint8, torch.cuda.current_device()
    ).view(-1)


def unmap_peer_regions(regions: Sequence[PeerRegion]) -> None:
    """Unmap ``map_peer_regions``' ranges and release the imported chunks (the owners' memory is
    freed once every rank released it)."""
    try:
        from cuda.bindings import driver as cuda
    except ImportError:
        from cuda import cuda
    for region in regions:
        for address, size in region._mapped:
            cuda.cuMemUnmap(address, size)
        for handle in region._handles:
            cuda.cuMemRelease(handle)
        if region.base:
            cuda.cuMemAddressFree(region.base, region.reserved)
        region._mapped.clear()
        region._handles.clear()
        region.base = 0


def verify_peer_regions(
    dkv: Dkv,
    regions: Sequence[Tuple[int, int]],
    peers: Sequence[PeerRegion],
    chunk_bytes: int,
) -> None:
    """Check ``map_peer_regions`` end to end (a collective, before the pools hold data).

    Every rank writes a pattern of its rank and chunk at the start of each chunk of its pools, and
    then reads every rank's patterns through the mapping: a wrong chunk size or a handle mapped at
    the wrong place reads another pattern. The written bytes are zeroed again.
    """
    import torch

    from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor

    window = 64
    for (base, size), peer in zip(regions, peers):
        chunks = -(-int(size) // chunk_bytes)
        starts = torch.arange(chunks, dtype=torch.int64, device="cuda") * chunk_bytes
        offsets = (starts[:, None] + torch.arange(window, device="cuda")).reshape(-1)
        offsets = offsets[offsets < size]
        own = convert_to_torch_tensor(TensorWrapper(int(base), torch.uint8, [int(size)]))
        expect = lambda rank: ((offsets // chunk_bytes) * 31 + rank * 7) % 251 + 1  # noqa: E731
        own[offsets] = expect(dkv.rank).to(torch.uint8)
        torch.cuda.synchronize()
        dkv.all_gather(0)
        everyone = peer_region_tensor(peer)
        for rank in range(dkv.size):
            got = everyone[offsets + rank * peer.stride]
            if not torch.equal(got, expect(rank).to(torch.uint8)):
                raise RuntimeError(f"DKV peer mapping: rank {rank}'s pool reads wrong here")
        torch.cuda.synchronize()
        dkv.all_gather(0)
        own[offsets] = 0
        torch.cuda.synchronize()
