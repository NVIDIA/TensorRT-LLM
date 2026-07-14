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
from array import array
from typing import TYPE_CHECKING, Iterable, Iterator, NamedTuple, Sequence, TypeVar, cast

from . import rawref
from ._common import NDEBUG, BlockOrdinal, PageStatus, TokenId, TokenIdExt
from ._life_cycle_registry import AttnLifeCycle, LifeCycle, LifeCycleId, LifeCycleRegistry
from ._utils import (
    TypedIndexList,
    chunked,
    div_up,
    expect_type,
    filled_list,
    find_index,
    map_optional,
    unwrap_rawref,
)

if TYPE_CHECKING:
    from ._event_manager import KVCacheEventManager
    from ._page import CommittedPage

BlockKey = bytes


class ReuseScope(NamedTuple):
    """Per-request namespace for prefix reuse."""

    lora_id: int | None = None
    salt: int | None = None

    def _mask(self) -> bytes:
        return sum((value is not None) << i for i, value in enumerate(self)).to_bytes(
            div_up(len(self), 8), "little", signed=False
        )

    def to_bytes(self) -> bytes:
        ret = self._mask()
        for value in self:
            if type(value) is int:
                ret += value.to_bytes(8, "little", signed=False)
            else:
                assert value is None, (
                    "Did you forget to update to_bytes() when adding new non-int fields to ReuseScope?"
                )
        return ret


class ReuseMatch(NamedTuple):
    """Volatile result of a KV cache prefix match."""

    blocks: list["Block"]
    num_tokens: int


# id_offset is usually vocab_size
def gen_multimodal_cache_key_tokens(
    id_offset: int, multi_modal_data_digest: bytes, num_tokens: int, token_offset: int = 0
) -> list[TokenIdExt]:
    """Create synthetic tokens used only when building multimodal KV-cache keys.

    Item-local token 0 carries the content digest; later offsets use deterministic IDs above the vocab.
    """
    assert num_tokens > 0
    assert token_offset >= 0
    return [
        multi_modal_data_digest if token_offset + i == 0 else TokenId(id_offset + token_offset + i)
        for i in range(num_tokens)
    ]


class Hasher:
    __slots__ = "_hasher"
    _hasher: "hashlib._Hash"

    def __init__(self, data: int | bytes | None | Sequence[int | bytes] = None) -> None:
        self._hasher = hashlib.sha256()
        if data is not None:
            self.update(data)

    # This function is perf-critical. Expect compromised code quality.
    def update(self, data: int | bytes | Sequence[int | bytes]) -> "Hasher":
        if type(data) is int:
            assert NDEBUG or (data >= 0 and data < (1 << 64))
            self._hasher.update(data.to_bytes(8, "little"))
        elif type(data) is bytes:
            self._hasher.update(data)
        else:
            # Hash the whole token block in one C call instead of one per token.
            # array("Q", data).tobytes() packs each int as 8 native-endian bytes;
            # all NVIDIA GPU host platforms (x86_64, aarch64/Grace) are little-endian
            # so this is byte-identical to the per-token to_bytes(8, "little") loop.
            # Falls back to that loop for multimodal blocks (which contain bytes items).
            try:
                self._hasher.update(array("Q", data).tobytes())  # type: ignore
            except (TypeError, OverflowError):
                for item in data:  # type: ignore
                    assert (
                        NDEBUG
                        or (type(item) is int and (0 <= item < (1 << 64)))
                        or type(item) is bytes
                    )
                    self._hasher.update(item.to_bytes(8, "little") if (type(item) is int) else item)  # type: ignore
        return self

    @property
    def digest(self) -> bytes:
        return self._hasher.digest()


TokenBlock = list[TokenIdExt]


def sequence_to_blockchain_keys(
    tokens_per_block: int, reuse_scope: ReuseScope, tokens: Sequence[TokenIdExt]
) -> Iterator[tuple[TokenBlock, BlockKey]]:
    digest = Hasher(reuse_scope.to_bytes()).digest
    yield [], digest
    for token_block in chunked(tokens, tokens_per_block):
        digest = Hasher(digest).update(token_block).digest
        yield token_block, digest


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


def remove_subtree(root: "RootBlock | Block") -> list[rawref.ref["CommittedPage"]]:
    # taking O(1) space
    # remove leaf blocks one by one, in post-order
    ret: list[rawref.ref["CommittedPage"]] = []
    removed_block_hashes: list[BlockKey] = []
    tree = try_get_tree(root)
    event_manager = tree.event_manager if tree is not None else None
    block: "RootBlock | Block" = root
    while True:
        if block.next:
            block = next(iter(block.next.values()))
        else:
            if isinstance(block, Block):
                removed_block_hashes.append(block.key)
                ret.extend(p for p in block.storage if p is not None)
                block.storage = filled_list(None, block.num_life_cycles)
            assert isinstance(block, RootBlock) or all(page is None for page in block.storage), (
                "Storage is not cleared, yet"
            )
            if block._prev() is None:
                assert block is root
                break
            prev_block: Block | RootBlock | BlockRadixTree = block.prev
            # Because Block.__del__() may remove RootBlock from BlockRadixTree, we need to check here.
            # It may not be in prev_block.next when block is RootBlock.
            if block.key in prev_block.next:
                prev_block.next.pop(block.key)
            if block is root:
                break
            assert not isinstance(prev_block, BlockRadixTree)
            block = prev_block
    if event_manager is not None:
        event_manager.add_removed_event(removed_block_hashes)
    return ret


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
        # If there are sibling blocks fully covered by this block, remove them.
        to_remove = []
        for k, b in prev.next.items():
            if len(b.tokens) < len(tokens) and tokens[: len(b.tokens)] == b.tokens:
                assert NDEBUG or (not b.is_full and b is not self and b.key == k and not b.next)
                to_remove.append(k)
        event_manager = get_tree(prev).event_manager if to_remove else None
        for k in to_remove:
            b = prev.next.pop(k)
            if event_manager is not None:
                event_manager.add_removed_event(b.key)
            assert b.is_orphan  # _KVCache may still hold it.
        # prev.next keeps a strong ref to this _Block, so no need to remove self from prev.next in __del__().
        prev.next[self.key] = self

    def __del__(self) -> None:
        for ref in self.storage:
            if ref is not None and ref() is not None:
                page = unwrap_rawref(ref)
                if page.status == PageStatus.DROPPABLE:
                    if page.scheduled_for_eviction:
                        page.manager.exclude_from_eviction(page)
        if self._prev() is not None and isinstance(self.prev, RootBlock) and not self.prev.next:
            self.prev.prev.next.pop(self.prev.key)
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

    def unset_page(
        self, lc_idx: LifeCycleId, lc: LifeCycle, expected_page: "CommittedPage | None" = None
    ) -> None:
        ref = self.storage[lc_idx]
        if ref is None:
            return
        if expected_page is not None and ref() is not expected_page:
            return
        ordinal = self.ordinal
        self.storage[lc_idx] = None
        tree = try_get_tree(self)
        event_manager = tree.event_manager if tree is not None else None
        if type(lc) is AttnLifeCycle and (lc.window_size is None or ordinal < lc.num_sink_blocks):
            pages = remove_subtree(self)
            for r in pages:
                if r() is not None:
                    page = unwrap_rawref(r)
                    assert page.status == PageStatus.DROPPABLE
                    if page.scheduled_for_eviction:
                        page.manager.exclude_from_eviction(page)
        elif event_manager is not None:
            event_manager.add_removed_life_cycle_event(self.key, int(lc_idx))
        # It's possible to implement more sophisticated logic to remove useless blocks for SWA, e.g.
        # check if consecutive available blocks is sufficient for window_size. (TRTLLM-8802)
        # But for simplicity, we leave it for now.
        curr = self
        while (
            (isinstance(curr, Block) and curr.storage[lc_idx] is None)
            and not curr.next
            and curr._prev() is not None
        ):
            if curr.key in curr.prev.next:
                curr.prev.next.pop(curr.key)
                if event_manager is not None:
                    event_manager.add_removed_event(curr.key)
            curr = curr.prev

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
        return self.key not in self.prev.next or self.prev.next[self.key] is not self


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

    def clear(self) -> list[rawref.ref["CommittedPage"]]:
        # taking O(1) space
        # remove leaf blocks one by one, in post-order
        ret: list[rawref.ref["CommittedPage"]] = []
        while self.next:
            block = next(iter(self.next.values()))
            ret.extend(remove_subtree(block))
        assert not self.next
        return ret

    def _num_matched_tokens(self, matched: list[tuple[Block, int]]) -> int:
        if not matched:
            return 0
        return self._tokens_per_block * (len(matched) - 1) + matched[-1][1]

    @staticmethod
    def _has_pages(block: Block, lc_list: Iterable[LifeCycleId]) -> bool:
        return all(block.storage[lc] is not None for lc in lc_list)

    @staticmethod
    def _has_page(block: Block, lc: LifeCycleId) -> bool:
        return block.storage[lc] is not None

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

    def _prune_match(self, matched: list[tuple[Block, int]]) -> list[tuple[Block, int]]:
        tokens_per_block = self._tokens_per_block
        assert all(b[1] == tokens_per_block for b in matched[:-1])

        life_cycles = self._life_cycles

        # check for full attention layers
        attn_life_cycles = list(life_cycles.attention_life_cycles())
        if any(lc.window_size is None for _, lc in attn_life_cycles):
            lc_list = [lc_idx for lc_idx, lc in attn_life_cycles if lc.window_size is None]

            def check_no_pages(b: tuple[Block, int]) -> bool:
                return not BlockRadixTree._has_pages(b[0], lc_list)

            n = find_index(matched, check_no_pages)
            matched = matched[:n]

        swa_life_cycles = tuple(
            (lc_idx, lc) for lc_idx, lc in attn_life_cycles if lc.window_size is not None
        )
        # check for SWA sink
        for lc_idx, lc in swa_life_cycles:

            def check_no_page_lc(b: tuple[Block, int]) -> bool:
                return not BlockRadixTree._has_page(b[0], lc_idx)

            n = find_index(matched[: lc.num_sink_blocks], check_no_page_lc)
            if n < lc.num_sink_blocks:
                matched = matched[:n]
        # Check SSM snapshot availability before SWA window constraints.
        # Truncating to the last reusable SSM snapshot can change the matched
        # length used by the SWA check.
        ssm_lc_id = life_cycles.ssm_life_cycle_id
        if ssm_lc_id is not None:
            from ._page import SsmCommittedPage

        while matched:
            if ssm_lc_id is not None:
                ssm_trunc = 0
                ssm_match_len = 0
                for i in reversed(range(len(matched))):
                    block = matched[i][0]
                    page = map_optional(block.storage[ssm_lc_id], lambda f: f())
                    if page is None:
                        continue
                    page = expect_type(SsmCommittedPage, page)
                    snapshot_len = page.num_tokens_in_block
                    if matched[i][1] >= snapshot_len:
                        ssm_trunc = i + 1
                        ssm_match_len = snapshot_len
                        break
                matched = matched[:ssm_trunc]
                if not matched:
                    break
                matched[-1] = (matched[-1][0], ssm_match_len)
            # SWA window check
            num_tokens = self._num_matched_tokens(matched)
            for lc_idx, lc in swa_life_cycles:
                if lc.window_size is None:
                    continue

                def check_has_page_lc(b: tuple[Block, int]) -> bool:
                    return BlockRadixTree._has_page(b[0], lc_idx)

                n = find_index(reversed(matched), check_has_page_lc)
                if n != 0:
                    matched = matched[:-n]
                    break
                _, stale_end = lc.get_stale_range(num_tokens, tokens_per_block)

                def has_no_page(b: tuple[Block, int]) -> bool:
                    return not BlockRadixTree._has_page(b[0], lc_idx)

                n = find_index(reversed(matched[stale_end:]), has_no_page)
                if len(matched) - n > stale_end:
                    matched = matched[: len(matched) - n - 1]
                    break
            else:
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
        matched = self._prune_match(
            list(self._match_token_path(reuse_scope, tokens, enable_partial_match))
        )
        return ReuseMatch([block for block, _ in matched], self._num_matched_tokens(matched))

    def _check_sanity(self) -> bool:
        raise NotImplementedError(
            "[KVCacheManager] Check if there are any unusable blocks that should have been removed."
        )
