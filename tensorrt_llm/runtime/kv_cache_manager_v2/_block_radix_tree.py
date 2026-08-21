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

import hashlib
import itertools
from array import array
from itertools import chain
from typing import TYPE_CHECKING, Iterable, Iterator, NamedTuple, Sequence, TypeVar, cast

from . import rawref
from ._common import NDEBUG, BlockOrdinal, PageStatus, TokenId, TokenIdExt
from ._life_cycle_registry import AttnLifeCycle, LifeCycle, LifeCycleId, LifeCycleRegistry
from ._utils import TypedIndexList, filled_list, map_optional, typed_range, unwrap_rawref

if TYPE_CHECKING:
    from ._event_manager import KVCacheEventManager
    from ._page import CommittedPage


BlockKey = bytes
TokenBlock = list[TokenIdExt]

_SHA256_DIGEST_SIZE = hashlib.sha256().digest_size
_UINT_ITEM_SIZE = array("I").itemsize
if _UINT_ITEM_SIZE != 4:
    raise RuntimeError("Hasher requires a platform with 4-byte unsigned ints")


# id_offset is usually vocab_size. Backend-neutral (depends only on _common); the
# C++ backend exposes a native gen_multimodal_cache_key_tokens via nanobind instead.
def gen_multimodal_cache_key_tokens(
    id_offset: int, multi_modal_data_digest: bytes, num_tokens: int, token_offset: int = 0
) -> list[TokenIdExt]:
    """Create synthetic tokens used only when building multimodal KV-cache keys.

    Item-local token 0 carries the content digest; later offsets use deterministic IDs above the vocab.

    Args:
        id_offset: First synthetic id, usually ``vocab_size``, so generated ids cannot
            collide with real token ids.
        multi_modal_data_digest: Content digest of the multimodal item; must be exactly
            ``_SHA256_DIGEST_SIZE`` bytes.
        num_tokens: Number of synthetic tokens to generate. Must be positive.
        token_offset: Item-local index of the first generated token. Must be non-negative;
            only offset 0 carries the digest.

    Returns:
        The generated tokens, digest first when ``token_offset`` is 0.

    Raises:
        ValueError: If the digest length is wrong, ``num_tokens`` is not positive, or
            ``token_offset`` is negative.
    """
    if len(multi_modal_data_digest) != _SHA256_DIGEST_SIZE:
        raise ValueError(f"multi_modal_data_digest must have length {_SHA256_DIGEST_SIZE}")
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    if token_offset < 0:
        raise ValueError("token_offset must be non-negative")
    return [
        multi_modal_data_digest if token_offset + i == 0 else TokenId(id_offset + token_offset + i)
        for i in range(num_tokens)
    ]


class Hasher:
    """Incremental SHA-256 hasher used to derive block keys for the radix tree.

    Accepts ints (encoded as 4 little-endian bytes each, matching the C++ backend's
    4-byte ``TokenIdExt`` layout), raw ``bytes`` (multimodal content digests and
    reuse-scope fields), or a sequence mixing the two. Both backends must produce
    identical digests for the same logical input, so the encoding is part of the
    on-disk/cross-process contract and cannot change unilaterally.

    Args:
        data: Optional initial value, hashed immediately as if passed to ``update``.
    """

    # SECURITY INVARIANT: the block-key hash MUST stay cryptographically
    # collision-resistant and >= 256-bit. The radix tree is a globally shared,
    # cross-request/cross-tenant cache index; prefix matches are decided purely by
    # digest equality with NO re-check of the underlying tokens; and the hashed
    # input (tokens, the user-supplied cache_salt, multimodal content bytes) is
    # attacker-influenceable. A collision therefore silently reuses another
    # request's KV blocks (cross-request corruption / data leak), and cache_salt
    # tenant isolation relies entirely on this hash's collision resistance. Do NOT
    # swap in a non-cryptographic hash (xxHash, HighwayHash, ...) or truncate below
    # 256 bits without first adding a token-content equality check on match. The
    # C++ backend (blockRadixTree) mirrors this with SHA-256 (CSHA256).
    __slots__ = "_hasher"
    _hasher: "hashlib._Hash"

    def __init__(self, data: int | bytes | Sequence[int | bytes] | None = None) -> None:
        self._hasher = hashlib.sha256()
        if data is not None:
            self.update(data)

    def update(self, data: int | bytes | Sequence[int | bytes]) -> "Hasher":
        """Fold ``data`` into the running digest.

        Args:
            data: An int token id (0 <= id < 2**31), raw ``bytes``, or a sequence of
                either. An all-int sequence takes a single-call fast path; a sequence
                containing ``bytes`` (multimodal blocks) falls back to per-item hashing.

        Returns:
            This ``Hasher``, to allow chaining.
        """
        # This function is perf-critical. Expect compromised code quality.
        if type(data) is int:
            assert NDEBUG or (data >= 0 and data < (1 << 31))
            self._hasher.update(data.to_bytes(4, "little"))
        elif type(data) is bytes:
            self._hasher.update(data)
        else:
            # Hash the whole token block in one C call instead of one per token.
            # array("I", data).tobytes() packs each int as 4 native-endian bytes
            # (unsigned int); all NVIDIA GPU host platforms (x86_64, aarch64/Grace)
            # are little-endian so this is byte-identical to the per-token
            # to_bytes(4, "little") loop AND to the C++ backend's 4-byte TokenIdExt
            # layout (normal token = little-endian id, high tag bit clear). Falls
            # back to that loop for multimodal blocks (which contain bytes items).
            try:
                self._hasher.update(array("I", data).tobytes())  # type: ignore
            except (TypeError, OverflowError):
                for item in data:  # type: ignore
                    assert (
                        NDEBUG
                        or (type(item) is int and (0 <= item < (1 << 31)))
                        or type(item) is bytes
                    )
                    self._hasher.update(item.to_bytes(4, "little") if (type(item) is int) else item)  # type: ignore
        return self

    @property
    def digest(self) -> bytes:
        return self._hasher.digest()


def reuse_scope_to_bytes(reuse_scope: Iterable[int | None]) -> bytes:
    """Serialize a reuse scope to its reuse-namespace bytes.

    Backend-neutral: reads the scope's fields by iteration, so it works for both
    the pure-Python ``ReuseScope`` NamedTuple and the C++ binding without relying
    on a ``to_bytes()`` method. The layout mirrors the C++ ``emitReuseScopeBytes``:
    a mask byte (one bit per field, set when the field is present) followed by one
    little-endian ``uint64`` per present field (``signed=False``).
    """
    values = list(reuse_scope)
    mask = sum((value is not None) << i for i, value in enumerate(values))
    ret = mask.to_bytes((len(values) + 7) // 8, "little", signed=False)
    for value in values:
        if value is not None:
            ret += int(value).to_bytes(8, "little", signed=False)
    return ret


def sequence_to_blockchain_keys(
    tokens_per_block: int, reuse_scope: Iterable[int | None], tokens: Sequence[TokenIdExt]
) -> Iterator[tuple[TokenBlock, BlockKey]]:
    """Yield ``(token_block, key)`` pairs seeding a blockchain of KV-cache keys.

    The first pair is the root (``[]``, reuse-scope digest); each subsequent pair
    hashes one ``tokens_per_block`` chunk on top of the previous digest.
    """
    digest = Hasher(reuse_scope_to_bytes(reuse_scope)).digest
    yield [], digest
    iterator = iter(tokens)
    while True:
        token_block = list(itertools.islice(iterator, tokens_per_block))
        if not token_block:
            break
        digest = Hasher(digest).update(token_block).digest
        yield token_block, digest


class ReuseScope(NamedTuple):
    """Per-request namespace for prefix reuse."""

    lora_id: int | None = None
    salt: int | None = None

    def to_bytes(self) -> bytes:
        return reuse_scope_to_bytes(self)


class ReuseMatch(NamedTuple):
    """Volatile result of a KV cache prefix match.

    ``num_tokens_before_hybrid_pruning`` is retained for internal diagnostics.
    It is the prefix the attention pages alone would support, before recurrent
    snapshot availability shortens it.

    ``num_tokens_before_pruning`` is the raw token-path walk depth, before any
    pruning at all. It locates where this request's content diverges from the
    tree, independent of which pages happen to still be resident, so
    ``num_tokens_before_pruning == num_lookup_tokens`` means the whole lookup
    range matched and there is no fork here.
    """

    blocks: list["Block"]
    num_tokens: int
    num_lookup_tokens: int
    num_tokens_before_hybrid_pruning: int
    num_tokens_before_pruning: int


Child = TypeVar("Child", bound="Block | RootBlock")
Children = dict[BlockKey, Child]


def try_get_tree(block: "RootBlock | Block") -> "BlockRadixTree | None":
    node = block
    while not isinstance(node, BlockRadixTree):
        node = node._prev()
        if node is None:
            return None
    return node


def get_tree(block: "RootBlock | Block") -> "BlockRadixTree":
    tree = try_get_tree(block)
    if tree is None:
        raise ValueError("Dereferencing a dangling rawref")
    return tree


def detach_next(parent: "Block | RootBlock", key: BlockKey) -> "Block | None":
    child = parent.next.pop(key, None)
    if child is None:
        return None

    child._prev = rawref.NULL
    if isinstance(parent, RootBlock) and not parent.next:
        tree = parent._prev()
        if tree is not None and parent.key in tree.next:
            detached_root = tree.next.pop(parent.key)
            parent._prev = rawref.NULL
            assert detached_root is parent
    return child


def remove_subtree(root: "Block") -> None:
    # taking O(1) space
    # remove leaf blocks one by one, in post-order
    removed_block_hashes: list[BlockKey] = []
    tree = try_get_tree(root)
    event_manager = tree.event_manager if tree is not None else None
    block: Block = root
    while True:
        if block.next:
            block = next(iter(block.next.values()))
        else:
            removed_block_hashes.append(block.key)
            if block._prev() is None:
                assert block is root
                break
            prev_block: Block | RootBlock = block.prev
            detached = detach_next(prev_block, block.key)
            assert detached is block
            if block is root:
                break
            assert isinstance(prev_block, Block)
            block = prev_block
    if event_manager is not None:
        event_manager.add_removed_event(removed_block_hashes)


def traverse_post_order(root: "Block") -> Iterator["Block"]:
    "post-order traversal of the subtree rooted at root"
    stack: list[Iterator[Block]] = []
    block: Block | None = root
    while True:
        assert block is not None
        if block.next:
            child_iter = iter(block.next.values())
            stack.append(child_iter)
            block = next(child_iter)
        else:
            yield (last_yielded := block)
            while stack and (block := next(stack[-1], None)) is None:
                yield (last_yielded := cast(Block, last_yielded.prev))
                stack.pop()
            if not stack:
                break


def find_best_partial_match_in_next_nodes(
    block: "Block | RootBlock", tokens: TokenBlock
) -> tuple["Block | None", int]:
    """
    Among all child nodes (self.next), finds the one whose tokens have the longest leading match with the given tokens.
    Returns a tuple of (best_block, num_matched_tokens).
    If no child matches any tokens, returns (None, 0).
    """
    if len(block.next) >= 32:
        # TODO: build a database to accelerate partial matching. (TRTLLM-7784)
        # For now, it might be too slow to iterate over all children, so let's just skip.
        return None, 0
    best_block = None
    best_match_len = 0
    for b in block.next.values():
        match_len = b._partial_match_this_node(tokens)
        if match_len > best_match_len:
            best_match_len = match_len
            best_block = b
    return best_block, best_match_len


class DuplicateKeyError(Exception):
    "Another block with the same key already exists"

    key: BlockKey

    def __init__(self, key: BlockKey) -> None:
        super().__init__(f"Block with key {key.hex()} already exists")
        self.key = key


class UselessBlockError(Exception):
    block: "Block"

    def __init__(self, block: "Block") -> None:
        super().__init__(
            f"Block is useless because all its tokens are covered by another block with key = {block.key.hex()}"
        )
        self.block = block


def _add_or_get_existing(
    parent: "RootBlock | Block", tokens: Sequence[TokenIdExt]
) -> "Block | None":
    try:
        return Block(tokens, parent)
    except DuplicateKeyError as e:
        return parent.next[e.key]
    except UselessBlockError:
        return None


class RootBlock:
    __slots__ = ("__rawref__", "_prev", "key", "next", "reuse_scope")
    key: BlockKey
    reuse_scope: ReuseScope
    _prev: rawref.ref["BlockRadixTree"]
    next: Children["Block"]
    __rawref__: rawref.ref["RootBlock"]

    def __init__(self, reuse_scope: ReuseScope, prev: "BlockRadixTree") -> None:
        self.key = self.make_key(reuse_scope)
        assert self.key not in prev.next, "Root block already exists"
        self.reuse_scope = reuse_scope
        self._prev = rawref.ref(prev)
        self.next = {}
        self.__rawref__ = rawref.NULL
        prev.next[self.key] = self

    def __del__(self) -> None:
        self.__rawref__.invalidate()

    @property
    def ordinal(self) -> BlockOrdinal:
        return BlockOrdinal(-1)

    @property
    def prev(self) -> "BlockRadixTree":
        return unwrap_rawref(self._prev)

    @property
    def num_life_cycles(self) -> LifeCycleId:
        return self.prev.num_life_cycles

    @property
    def tokens_per_block(self) -> int:
        return self.prev.tokens_per_block

    @staticmethod
    def make_key(reuse_scope: ReuseScope) -> BlockKey:
        return Hasher(reuse_scope.to_bytes()).digest


class Block:
    """
    A block of tokens. Manages data for all layers.
    """

    __slots__ = ("__rawref__", "_prev", "key", "next", "ordinal", "storage", "tokens")
    key: BlockKey
    tokens: Sequence[TokenIdExt]
    ordinal: BlockOrdinal
    _prev: rawref.ref["Block | RootBlock"]
    next: Children["Block"]
    __rawref__: rawref.ref["Block"]

    # indexed with LifeCycleId
    storage: TypedIndexList[LifeCycleId, rawref.ref["CommittedPage"] | None]

    @staticmethod
    def make_key(prev_key: BlockKey, tokens: Sequence[TokenIdExt]) -> BlockKey:
        return Hasher(prev_key).update(tokens).digest

    def __init__(self, tokens: Sequence[TokenIdExt], prev: "Block | RootBlock") -> None:
        assert prev.tokens_per_block == prev.prev.tokens_per_block, "prev must be a full block"
        self.key = self.make_key(prev.key, tokens)
        self.tokens = tokens
        self.ordinal = BlockOrdinal(prev.ordinal + 1)
        self._prev = rawref.ref(prev)
        self.next = {}
        self.storage = filled_list(None, prev.num_life_cycles)
        self.__rawref__ = rawref.NULL
        # a Block is useless if all its tokens are covered by a sibling block. Raise UselessBlockError if so.
        if self.key in prev.next:
            raise UselessBlockError(prev.next[self.key])
        if len(tokens) < self.tokens_per_block:
            # @TODO: when we have the database for find_best_partial_match_in_next_nodes, we may use
            # that for faster check.
            for b in prev.next.values():
                if b.tokens[: len(tokens)] == tokens:
                    raise UselessBlockError(b)
        # A later turn may extend a partial endpoint to this longer block, replacing the
        # partial sibling. That turn may not have a committable SWA page for this block:
        # commit_min_snapshot releases out-of-window pages, while SWA scratch reuse uses
        # temporary shared storage that is not preserved. Adopt the partial sibling's
        # pages to keep the shorter endpoint reusable, retaining each page's recorded token
        # count (see CommittedPage.num_tokens_in_block).
        to_remove = []
        for k, b in prev.next.items():
            if len(b.tokens) < len(tokens) and tokens[: len(b.tokens)] == b.tokens:
                assert NDEBUG or (not b.is_full and b is not self and b.key == k and not b.next)
                to_remove.append(k)
        # Two covered siblings would be prefixes of each other; the insertion logic
        # would already have replaced the shorter one.
        assert NDEBUG or len(to_remove) <= 1
        event_manager = get_tree(prev).event_manager if to_remove else None
        # Keep RootBlock attached while covered children are replaced.  Adding
        # the replacement first prevents detach_next() from pruning an
        # otherwise-empty root before this block becomes its new child.
        prev.next[self.key] = self
        for k in to_remove:
            b = detach_next(prev, k)
            assert isinstance(b, Block)
            self._adopt_pages_from(b)
            if event_manager is not None:
                event_manager.add_removed_event(b.key)
            assert b.is_orphan  # _KVCache may still hold it.
        # prev.next keeps a strong ref to this _Block, so no need to remove self from prev.next in __del__().

    def page_coverage(self, lc_idx: LifeCycleId) -> int:
        """Return the page's recorded token count, or zero if the slot is empty.

        For attention this is prefix coverage; for SSM it is an exact checkpoint position.
        """
        page = self.get_page(lc_idx)
        return page.num_tokens_in_block if page is not None else 0

    def holds_page(self, page: "CommittedPage") -> bool:
        return self.get_page(page.life_cycle) is page

    def can_replace_page(self, lc_idx: LifeCycleId, num_tokens_in_block: int) -> bool:
        """Whether a page recording `num_tokens_in_block` may take over slot `lc_idx`.

        A slot keeps only the page with the largest recorded token count. For attention,
        greater coverage strictly dominates lesser coverage. For SSM, this deliberately
        keeps only the latest checkpoint -- two conversation turns rarely end inside the
        same block, and if they do, a reuse miss is acceptable.

        Pure; use replace_page() to install.
        """
        existing = self.get_page(lc_idx)
        return existing is None or existing.num_tokens_in_block < num_tokens_in_block

    def replace_page(self, lc_idx: LifeCycleId, page: "CommittedPage") -> None:
        """Install `page` in slot `lc_idx`, detaching whatever it supersedes.

        The superseded page may outlive this call while a request still holds it, so
        unlink_page() must clear its back-pointer: _release_pages() walks `storage`, so
        nothing would clear it later and it would dangle once this block dies.
        """
        assert NDEBUG or self.can_replace_page(lc_idx, page.num_tokens_in_block)
        existing = self.unlink_page(lc_idx)
        if existing is not None and existing.scheduled_for_eviction:
            existing.manager.exclude_from_eviction(existing)
        page.block = rawref.ref(self)
        self.storage[lc_idx] = rawref.ref(page)

    def _adopt_pages_from(self, other: "Block") -> None:
        """Move `other`'s pages into self without changing their recorded token counts."""
        assert other.ordinal == self.ordinal
        for lc_idx in typed_range(self.num_life_cycles):
            page = other.get_page(lc_idx)
            if page is None or not self.can_replace_page(lc_idx, page.num_tokens_in_block):
                continue
            # Clear the source slot directly rather than via unlink_page(), which would
            # null the back-pointer replace_page() is about to overwrite.
            other.storage[lc_idx] = None
            self.replace_page(lc_idx, page)

    def _release_pages(self) -> None:
        """Reclaim every page held by this block.

        Nulls each page's back-pointer and, for pages still scheduled for eviction,
        removes them from the eviction controller (releasing their storage slots).
        Idempotent: afterwards ``storage`` holds no pages, so it is safe to call again
        from ``__del__``.

        Cleanup is normally deferred to ``__del__``. An orphan block may remain
        referenced by a live ``_KVCache`` and retain its pages until that cache closes;
        every cache must close before ``StorageManager`` teardown.
        """
        for lc_idx in typed_range(self.num_life_cycles):
            page = self.get_page(lc_idx)
            if page is not None:
                self.unlink_page(lc_idx)
                if page.status == PageStatus.DROPPABLE:
                    if page.scheduled_for_eviction:
                        page.manager.exclude_from_eviction(page)

    def __del__(self) -> None:
        self._release_pages()
        self.__rawref__.invalidate()

    def _partial_match_this_node(self, tokens: TokenBlock) -> int:
        """
        Returns the number of leading tokens that match between the given tokens and this block's tokens.
        """
        for i, (a, b) in enumerate(zip(tokens, self.tokens)):
            if a != b:
                return i
        return min(len(tokens), len(self.tokens))

    @property
    def num_life_cycles(self) -> LifeCycleId:
        return LifeCycleId(len(self.storage))

    @property
    def prev(self) -> "Block | RootBlock":
        return unwrap_rawref(self._prev)

    def get_page(self, lc_idx: LifeCycleId) -> "CommittedPage | None":
        """Return the page in slot `lc_idx`, or None when the slot is empty.

        A non-empty slot always resolves: CommittedPage.__del__ unlinks the page from its
        block before invalidating its rawref, so `storage` never retains a dangling ref.
        """
        return map_optional(self.storage[lc_idx], lambda f: f())

    def unlink_page(
        self, lc_idx: LifeCycleId, expected_page: "CommittedPage | None" = None
    ) -> "CommittedPage | None":
        """Detach slot `lc_idx`, returning the page that was there, or None.

        The sole place a block-page link is severed.
        """
        # Called from CommittedPage.__del__, which invalidates the page's rawref only
        # afterwards, so the dying page is still reachable here.
        page = self.get_page(lc_idx)
        if page is None:
            return None
        # Only unlink when the slot still holds the expected page. During rebase
        # another block with the same key may have replaced the stored page, and
        # unlinking then would clobber the newer page's back-pointer.
        if expected_page is not None and page is not expected_page:
            return None
        page.block = rawref.NULL
        self.storage[lc_idx] = None
        return page

    @staticmethod
    def clear_stale_blocks_after_page_unlink(
        start: "Block", lc_idx: LifeCycleId, lc: LifeCycle
    ) -> None:
        assert start.get_page(lc_idx) is None
        ordinal = start.ordinal
        tree = try_get_tree(start)
        event_manager = tree.event_manager if tree is not None else None
        if type(lc) is AttnLifeCycle and (lc.window_size is None or ordinal < lc.num_sink_blocks):
            remove_subtree(start)
        elif event_manager is not None:
            event_manager.add_removed_life_cycle_event(start.key, int(lc_idx))
        # It's possible to implement more sophisticated logic to remove useless blocks for SWA, e.g.
        # check if consecutive available blocks is sufficient for window_size. (TRTLLM-8802)
        # But for simplicity, we leave it for now.
        curr = start
        while (
            (
                isinstance(curr, Block)
                and all(
                    curr.get_page(life_cycle) is None
                    for life_cycle in typed_range(curr.num_life_cycles)
                )
            )
            and not curr.next
            and curr._prev() is not None
        ):
            prev = curr.prev
            detached = detach_next(prev, curr.key)
            assert detached is curr
            if event_manager is not None:
                event_manager.add_removed_event(curr.key)
            curr = prev

    @property
    def tokens_per_block(self) -> int:
        # we assume non-leaf blocks are always full.
        prev = self.prev
        return prev.tokens_per_block if isinstance(prev, RootBlock) else len(prev.tokens)

    @property
    def is_full(self) -> bool:
        return len(self.tokens) == self.tokens_per_block

    @property
    def is_orphan(self) -> bool:
        prev = self._prev()
        assert prev is None or (self.key in prev.next and prev.next[self.key] is self)
        return prev is None


class BlockRadixTree:
    __slots__ = (
        "__rawref__",
        "_event_manager",
        "_life_cycles",
        "_tokens_per_block",
        "next",
    )
    _life_cycles: LifeCycleRegistry
    _tokens_per_block: int
    _event_manager: "KVCacheEventManager | None"
    next: Children[RootBlock]
    __rawref__: rawref.ref["BlockRadixTree"]

    def __init__(
        self,
        life_cycles: LifeCycleRegistry,
        tokens_per_block: int,
        event_manager: "KVCacheEventManager | None" = None,
    ) -> None:
        self._life_cycles = life_cycles
        self._tokens_per_block = tokens_per_block
        self._event_manager = event_manager
        self.next = {}
        self.__rawref__ = rawref.NULL

    def __del__(self) -> None:
        self.__rawref__.invalidate()

    def add_or_get_existing(self, reuse_scope: ReuseScope) -> RootBlock:
        key = RootBlock.make_key(reuse_scope)
        if key in self.next:
            return self.next[key]
        return RootBlock(reuse_scope, self)

    @property
    def tokens_per_block(self) -> int:
        return self._tokens_per_block

    @property
    def life_cycles(self) -> LifeCycleRegistry:
        return self._life_cycles

    @property
    def event_manager(self) -> "KVCacheEventManager | None":
        return self._event_manager

    @property
    def num_life_cycles(self) -> LifeCycleId:
        return self.life_cycles.size

    def clear(self) -> None:
        # taking O(1) space
        # remove leaf blocks one by one, in post-order
        # Block.__del__() handles page cleanup when the last owner releases each block.
        # detach_next() auto-prunes empty RootBlocks from the tree.
        while self.next:
            root = next(iter(self.next.values()))
            while root.next:
                remove_subtree(next(iter(root.next.values())))
        assert not self.next

    def _num_matched_tokens(self, matched: list[tuple[Block, int]]) -> int:
        if not matched:
            return 0
        return self._tokens_per_block * (len(matched) - 1) + matched[-1][1]

    # yields tuples of (block, num_matched_tokens). num_matched_tokens should be equal to
    # tokens_per_block except the last one.
    def _match_token_path(
        self,
        reuse_scope: ReuseScope,
        tokens: Sequence[TokenIdExt],
        enable_partial_match: bool = False,
    ) -> Iterator[tuple[Block, int]]:
        block: Block | RootBlock | BlockRadixTree = self
        mismatched_token_block: TokenBlock = []
        for token_block, key in sequence_to_blockchain_keys(
            self._tokens_per_block, reuse_scope, tokens
        ):
            if key in block.next:
                block = block.next[key]
                if token_block:
                    assert isinstance(block, Block)
                    yield block, len(token_block)
            else:
                mismatched_token_block = token_block
                break
        if mismatched_token_block and enable_partial_match:
            partial_block, match_len = find_best_partial_match_in_next_nodes(
                cast(Block | RootBlock, block), mismatched_token_block
            )
            if partial_block is not None:
                block = partial_block
                yield block, match_len

    def _prune_match(
        self, matched: list[tuple[Block, int]], ssm_lc_id: LifeCycleId | None
    ) -> list[tuple[Block, int]]:
        """Shorten `matched` to the prefix that is actually reusable.

        Passing ssm_lc_id=None skips the recurrent-snapshot constraint and yields
        the attention-only prefix (used for num_tokens_before_hybrid_pruning).
        """
        tokens_per_block = self._tokens_per_block
        assert all(b[1] == tokens_per_block for b in matched[:-1])

        attn_life_cycles = list(self._life_cycles.attention_life_cycles())

        # Fixed-point loop: SSM may select an earlier exact snapshot, while attention may
        # shorten the match to the coverage of a required page. Every retry strictly
        # shortens the match, so the loop terminates.
        while matched:
            # Check SSM snapshot availability first: truncating to the last reusable SSM
            # snapshot changes the matched length that all the attention checks use.
            if ssm_lc_id is not None:
                ssm_trunc = 0
                ssm_match_len = 0
                for i in reversed(range(len(matched))):
                    # An SSM page holds the recurrent state after exactly this many tokens,
                    # so reuse must stop there instead of anywhere inside the block.
                    snapshot_len = matched[i][0].page_coverage(ssm_lc_id)
                    if snapshot_len > 0 and matched[i][1] >= snapshot_len:
                        ssm_trunc = i + 1
                        ssm_match_len = snapshot_len
                        break
                matched = matched[:ssm_trunc]
                if not matched:
                    break
                matched[-1] = (matched[-1][0], ssm_match_len)

            # Only pages that are active at this candidate endpoint constrain attention
            # reuse. Full attention requires every block. SWA requires sink blocks and the
            # trailing window, but not the stale blocks between them. In particular, at an
            # exact block boundary with window_size=1, every historical block is stale.
            num_tokens = self._num_matched_tokens(matched)
            shortened = False
            for lc_idx, lc in attn_life_cycles:
                stale = lc.get_stale_range(num_tokens, tokens_per_block)
                for i in chain(range(stale.beg), range(stale.end, len(matched))):
                    block, num_matched = matched[i]
                    coverage = block.page_coverage(lc_idx)
                    if coverage >= num_matched:
                        continue
                    if coverage > 0:
                        matched = matched[: i + 1]
                        matched[-1] = (block, coverage)
                    else:
                        matched = matched[:i]
                    shortened = True
                    break
                if shortened:
                    break
            if not shortened:
                break
        return matched

    def match(
        self,
        reuse_scope: ReuseScope,
        tokens: Sequence[TokenIdExt],
        enable_partial_match: bool = False,
    ) -> ReuseMatch:
        """
        Return the currently reusable prefix match without holding pages.

        The result is volatile: callers that need to reuse the returned blocks must
        acquire ownership of the pages before depending on them.
        """
        raw_matched = list(self._match_token_path(reuse_scope, tokens, enable_partial_match))
        num_tokens_before_pruning = self._num_matched_tokens(raw_matched)
        ssm_lc_id = self._life_cycles.ssm_life_cycle_id
        # Diagnostic only: re-prune ignoring recurrent-snapshot availability to get
        # the prefix the attention pages alone support. Only hybrid models pay for
        # the second pass; without an SSM life cycle the two results are identical.
        attn_only_tokens = (
            self._num_matched_tokens(self._prune_match(list(raw_matched), None))
            if ssm_lc_id is not None
            else None
        )
        matched = self._prune_match(raw_matched, ssm_lc_id)
        num_tokens = self._num_matched_tokens(matched)
        return ReuseMatch(
            [block for block, _ in matched],
            num_tokens,
            len(tokens),
            num_tokens if attn_only_tokens is None else attn_only_tokens,
            num_tokens_before_pruning,
        )

    def _check_sanity(self) -> bool:
        raise NotImplementedError(
            "[KVCacheManager] Check if there are any unusable blocks that should have been removed."
        )
