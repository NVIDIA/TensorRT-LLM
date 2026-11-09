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
"""How the bytes of a role's buffers spread over ranks, and what one rank sends another of them;
the standard library only.

A role's layout (``RoleLayoutV2``) says how its bytes spread over ranks. The buffers of one layer
and role on all ranks together hold one space of bytes, of which each rank's buffer holds pieces
(``RoleLayoutV2.pieces``):

    ByHead(head_bytes=256, total=8)  # 8 heads of 256 B; TP2 rank 1 holds heads 4..7, bytes 1024..
    Sectioned((32, 16, 16))          # [x | B | C], each section split by byte over the ranks
    Replicated()                     # the whole buffer alike on every rank
    RingSlots(19)                    # token position p in slot p % 19, alike on every rank

A new way of laying a role out is one ``RoleLayoutV2`` subclass saying which pieces a rank holds.
What a source rank sends a destination rank of one role follows from the two layouts alone. A KV
cache manager names a layout per role (``pyexecutor.kv_cache.page_table.PageTableSpec``); only
disaggregated transfer maps pages by them, so they live with it. This module imports the standard
library alone, so that any transfer backend may read them.

``RoleLayoutV2`` is not the resharding geometry the ``get_disagg_*`` hooks return
(``disaggregation.resource.page.RoleLayout``)."""

from __future__ import annotations

import dataclasses
import numbers
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Mapping, Protocol, TypeVar

__all__ = [
    "GRID_DIMENSIONS",
    "BufferSite",
    "ByHead",
    "Own",
    "Piece",
    "RankTopology",
    "Replicated",
    "RingSlots",
    "RoleLayoutV2",
    "Sectioned",
    "check_pieces",
    "grid_place",
    "recurrent_state_split",
    "role_layout_from_dict",
]


def _integer(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        requirement = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{name} must be a {requirement} integer, got {value!r}")
    return int(value)


GRID_DIMENSIONS = ("tp", "cp")
"""Parallel dimensions a layout may split over (``RoleLayoutV2.split``): "tp" the tensor-parallel
ranks of one attention data-parallel group, "cp" the context-parallel ranks. Pipeline ranks hold
other layers and data-parallel ranks other requests, so neither splits a page."""


def grid_place(topology: RankTopology, dims: Sequence[str]) -> tuple[int, int]:
    """``(size, rank)`` of this rank in the grid of the parallel dimensions ``dims``, outer to
    inner, the last minor. Under attention data parallelism each tensor-parallel rank serves its own
    requests and holds its buffers whole, so "tp" counts this rank alone."""
    size, rank = 1, 0
    for dim in dims:
        if dim == "tp":
            n, r = (1, 0) if topology.attention_dp else (topology.tp_size, topology.tp_rank)
        elif dim == "cp":
            n, r = topology.cp_size, topology.cp_rank
        else:
            raise ValueError(f"{dim!r} is no grid dimension; one of {GRID_DIMENSIONS}")
        size, rank = size * n, rank * n + r
    return size, rank


def recurrent_state_split(topology: RankTopology) -> tuple[str, ...]:
    """The parallel dimensions recurrent state splits over, outer to inner, so that
    ``grid_place(topology, recurrent_state_split(topology))`` is the shard grid the native transfer
    and the pool sizing give recurrent state (``MambaPolicy._mamba_tp``,
    ``mamba_effective_tp_size``). Without attention data parallelism: over tensor then context
    ranks, CP minor, or over tensor ranks alone without context parallelism. Under attention data
    parallelism: over context ranks at one tensor rank, else held whole ("tp" counts this rank
    alone)."""
    if topology.cp_size > 1:
        if topology.attention_dp:
            return ("cp",) if topology.tp_size == 1 else ("tp",)
        return ("tp", "cp")
    return ("tp",)


_Positions = TypeVar("_Positions")


class BufferSite(Protocol):
    """The facts of one rank's buffer of a role that a layout in a manager's spec resolves
    against (``RoleLayoutV2.resolve``).

    Attributes:
        size: The buffer's bytes on this rank.
        kv_heads: For attention, (the layer's KV heads on this rank, the layer's KV heads over all
            ranks of the split or None if unknown); the total counts each head once, however many
            ranks hold it.
        state: Whether the buffer is recurrent state rather than attention.
        split: The parallel dimensions the buffer's layer group splits over, outer to inner
            (``GRID_DIMENSIONS``).
        ranks: The ranks of that split, ``grid_place(topology, split)[0]``.
        tokens: The token positions of a page.
    """

    size: int
    kv_heads: tuple[int, int | None]
    state: bool
    split: tuple[str, ...]
    ranks: int
    tokens: int


_ROLE_LAYOUTS: dict[str, type[RoleLayoutV2]] = {}


def _register(registry: dict, cls: type, what: str) -> None:
    """Enter ``cls`` in ``registry`` by its name; a class defined again in its own place (a module
    reloaded, a test run again) replaces itself, while another class of the same name is refused."""
    if not cls.name:
        return
    known = registry.get(cls.name)
    if known is not None and (known.__module__, known.__qualname__) != (
        cls.__module__,
        cls.__qualname__,
    ):
        raise ValueError(f"{what} {cls.name!r} is registered twice")
    registry[cls.name] = cls


def _split(split: Sequence[str], what: str) -> tuple[str, ...]:
    """``split`` as a tuple of distinct dimensions from ``GRID_DIMENSIONS``, at least one."""
    dims = tuple(str(dim) for dim in split)
    if not dims or len(set(dims)) != len(dims) or not set(dims) <= set(GRID_DIMENSIONS):
        raise ValueError(f"{what} splits over {dims}, not over distinct dims of {GRID_DIMENSIONS}")
    return dims


def _sparse(kind: object) -> dict:
    """The fields of a kind not at their defaults, as plain values (a tuple as a list, which the
    kind's constructor converts back)."""
    doc = {}
    for f in dataclasses.fields(kind):
        value = getattr(kind, f.name)
        if f.default is not dataclasses.MISSING and value == f.default:
            continue
        doc[f.name] = list(value) if isinstance(value, tuple) else value
    return doc


@dataclass(frozen=True)
class Piece:
    """Bytes of a role's space that a rank's buffer holds: ``size`` bytes from coordinate ``at`` of
    the space, at byte ``offset`` of the buffer.

    Attributes:
        at: Coordinate of the piece's first byte in the space of the role's buffers on all ranks.
        offset: Byte offset of the piece within the buffer on this rank.
        size: Positive number of bytes.
    """

    at: int
    offset: int
    size: int


@dataclass(frozen=True)
class RoleLayoutV2:
    """How the bytes of a role's buffers spread over ranks.

    The buffers of one layer and role on all ranks together hold one space of bytes. A rank's
    buffer holds pieces of it (``pieces``), by the rank's place among the ranks of the parallel
    dimensions the layout splits over (``split``); a buffer every rank holds alike splits over
    none. Two ranks' buffers correspond where their layouts lay out the same space (``space``):
    each byte of it the destination holds comes from one of the source ranks holding it.

    A subclass is one way of laying a role out, a frozen dataclass whose fields are its data,
    named (``name``) for peers. A new way is one subclass, here or in a model's own module, giving
    ``pieces`` and ``space``; one of token positions (``positional``), as ``RingSlots``, also
    gives ``positions``, ``index_of`` and ``stride``. A manager's spec
    (``page_table.PageTableSpec``) names a layout per role; each buffer's is that layout resolved
    from the buffer's facts (``resolve``).

    Attributes:
        name: The layout's name, unique among role layouts.
        positional: Whether the buffer holds token positions, of which a page holds those it has
            written: its pieces are then those of one position, which lies at the index its
            position takes (``index_of``), ``stride`` bytes per index.

    Raises:
        TypeError: On defining a positional layout that does not give ``positions``,
            ``index_of`` and ``stride``.
    """

    name: ClassVar[str] = ""
    positional: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if cls.positional and (
            cls.positions is RoleLayoutV2.positions
            or cls.index_of is RoleLayoutV2.index_of
            or cls.stride is RoleLayoutV2.stride
        ):
            raise TypeError(
                f"role layout {cls.__name__} holds token positions but does not say which a page "
                "holds (positions), where (index_of) or how far apart (stride)"
            )
        _register(_ROLE_LAYOUTS, cls, "role layout")

    @property
    def split(self) -> tuple[str, ...]:
        """The parallel dimensions the buffers spread over, outer to inner: none by default, every
        rank holding the whole space."""
        return ()

    @property
    def dtype(self) -> str | None:
        """The type of the elements by name, such as ``"bfloat16"``, or None where the layout
        states none. Where both peers' layouts state one, peers whose elements differ in type do
        not correspond (``space``); a layout stating none leaves the element type to the manager's
        spec as a whole (``quant_mode``, ``mixed_precision``) and peers' byte sizes."""
        return None

    def resolve(self, site: BufferSite) -> RoleLayoutV2:
        """The layout of one buffer, filling what a manager's spec leaves to each buffer from its
        facts (``BufferSite``): this layout itself by default."""
        return self

    def check(self, size: int) -> None:
        """Raise ValueError if a buffer of ``size`` bytes cannot be laid out so."""

    def pieces(self, size: int, ranks: int, rank: int) -> tuple[Piece, ...]:
        """The pieces of the space a buffer of ``size`` bytes holds at place ``rank`` of the
        ``ranks`` ranks of its split (``grid_place``), lying back to back over the buffer
        (``check_pieces``); for a positional layout, those of one token position."""
        raise NotImplementedError(f"role layout {self.name!r} gives no pieces")

    def space(self, size: int, ranks: int) -> tuple:
        """What the coordinates of the space mean, for buffers of ``size`` bytes over ``ranks``
        ranks: a name and the space's extents, equal on two ranks whose buffers correspond. A
        layout laying out another's space, as in another order of its bytes, gives the other's."""
        raise NotImplementedError(f"role layout {self.name!r} gives no space")

    def even(self, ranks: int) -> bool:
        """Whether pieces over ``ranks`` ranks fall on the layout's own boundaries (for a head
        split, whether its total and ``ranks`` divide one another), so that ranks holding the
        same bytes under another such count may share names: yes by default."""
        return True

    def positions(self, page: int, end: int, tokens_per_page: int) -> range:
        """For a positional layout: the token positions of page ``page`` (its position among a
        request's pages, from 0, of ``tokens_per_page`` positions each) that the buffer holds once
        the request has written every position below ``end``."""
        raise NotImplementedError(f"role layout {self.name!r} holds no token positions")

    def index_of(self, positions: _Positions, tokens_per_page: int) -> _Positions:
        """For a positional layout: the index holding each of ``positions`` (an integer or an
        integer array; the result is of the same kind) in pages of ``tokens_per_page`` positions."""
        raise NotImplementedError(f"role layout {self.name!r} holds no token positions")

    def stride(self, size: int) -> int:
        """For a positional layout: the bytes between consecutive indices of a buffer of ``size``
        bytes."""
        raise NotImplementedError(f"role layout {self.name!r} holds no token positions")

    def to_dict(self) -> dict:
        """The layout as plain values for a peer, which ``role_layout_from_dict`` reads back."""
        return {"name": self.name, **_sparse(self)}


def role_layout_from_dict(data: Mapping) -> RoleLayoutV2:
    """The role layout ``RoleLayoutV2.to_dict`` wrote, checked as its constructor checks it.

    Raises:
        ValueError: If it names no registered layout.
    """
    fields = dict(data)
    name = fields.pop("name", None)
    if name not in _ROLE_LAYOUTS:
        raise ValueError(f"unknown role layout {name!r}; known: {sorted(_ROLE_LAYOUTS)}")
    return _ROLE_LAYOUTS[name](**fields)


def _dtype(layout: RoleLayoutV2) -> None:
    """Name the layout's element type as torch does, without its module: ``"bfloat16"`` for
    ``torch.bfloat16``.

    Raises:
        ValueError: If the element type is neither a name nor a torch type.
    """
    dtype = layout.dtype
    if dtype is None or isinstance(dtype, str):
        return
    name = str(dtype)
    if not name.startswith("torch."):
        raise ValueError(f"an element type is a name or a torch type, not {dtype!r}")
    object.__setattr__(layout, "dtype", name.removeprefix("torch."))


def _typed(layout: RoleLayoutV2, space: tuple) -> tuple:
    """``space`` with the layout's element type where it states one, so that peers whose elements
    differ in type lay out different spaces."""
    return space if layout.dtype is None else (*space, layout.dtype)


@dataclass(frozen=True)
class Replicated(RoleLayoutV2):
    """The whole buffer alike on every rank, as an index-K cache is."""

    name: ClassVar[str] = "replicated"

    def pieces(self, size: int, ranks: int, rank: int) -> tuple[Piece, ...]:
        return (Piece(0, 0, size),)

    def space(self, size: int, ranks: int) -> tuple:
        return ("replicated", size)


@dataclass(frozen=True)
class Own(RoleLayoutV2):
    """Each rank's own share among the ranks of ``split``, which pairs only with the same rank's:
    the layout of a buffer whose split the manager does not describe further, or that never moves.

    Attributes:
        split: Parallel dimensions from ``GRID_DIMENSIONS``, outer to inner; the tensor-parallel
            ranks by default, those of the buffer's layer group once resolved. Normalized to a
            tuple.

    Raises:
        ValueError: If split is empty or names a dimension twice or one outside
            ``GRID_DIMENSIONS``.
    """

    name: ClassVar[str] = "own"
    split: tuple[str, ...] = ("tp",)

    def __post_init__(self) -> None:
        object.__setattr__(self, "split", _split(self.split, "an owned layout"))

    def resolve(self, site: BufferSite) -> RoleLayoutV2:
        return dataclasses.replace(self, split=site.split)

    def pieces(self, size: int, ranks: int, rank: int) -> tuple[Piece, ...]:
        return (Piece(rank * size, 0, size),)

    def space(self, size: int, ranks: int) -> tuple:
        return ("own", size, ranks)


@dataclass(frozen=True)
class Sectioned(RoleLayoutV2):
    """Consecutive sections, each this rank's share of its own part of the space, split by byte
    over the ranks of ``split``, as Mamba2's conv state ``[x | B | C]`` is: with sections of 32,
    16 and 16 bytes on each of two ranks, rank 1 holds bytes 32..63, 80..95 and 112..127 of a
    space of 64 + 32 + 32 bytes.

    Attributes:
        section_bytes: Positive bytes of each section on this rank, in order; normalized to a
            tuple.
        split: Parallel dimensions from ``GRID_DIMENSIONS``, outer to inner; the tensor-parallel
            ranks by default, those of the buffer's layer group once resolved. Normalized to a
            tuple.
        dtype: The type of the elements by name, or None.

    Raises:
        ValueError: If there is no section, a section is not a positive integer, split is empty or
            names a dimension twice or one outside ``GRID_DIMENSIONS``, or the element type is
            neither a name nor a torch type.
    """

    name: ClassVar[str] = "sectioned"
    section_bytes: tuple[int, ...]
    split: tuple[str, ...] = ("tp",)
    dtype: str | None = None

    def __post_init__(self) -> None:
        sections = tuple(_integer(n, "section bytes", 1) for n in self.section_bytes)
        if not sections:
            raise ValueError("a sectioned layout has one or more sections")
        object.__setattr__(self, "section_bytes", sections)
        object.__setattr__(self, "split", _split(self.split, "a sectioned layout"))
        _dtype(self)

    def resolve(self, site: BufferSite) -> RoleLayoutV2:
        return dataclasses.replace(self, split=site.split)

    def check(self, size: int) -> None:
        if sum(self.section_bytes) != size:
            raise ValueError(
                f"sections of {list(self.section_bytes)} bytes do not cover its {size} bytes"
            )

    def pieces(self, size: int, ranks: int, rank: int) -> tuple[Piece, ...]:
        out, base, offset = [], 0, 0
        for n in self.section_bytes:
            out.append(Piece(base + rank * n, offset, n))
            base += ranks * n
            offset += n
        return tuple(out)

    def space(self, size: int, ranks: int) -> tuple:
        return _typed(self, ("sectioned", tuple(ranks * n for n in self.section_bytes)))


@dataclass(frozen=True)
class ByHead(RoleLayoutV2):
    """Split by head over the ranks of ``split``, head-major or, with ``token_major``,
    token-major.

    The space holds the heads of all ranks: ``[head][byte]`` head-major (HND), a head's bytes
    those of a page of it, and ``[token][head][byte]`` token-major (NHD). Place ``r`` of ``n``
    ranks holds the ``size // head_bytes`` heads from ``r * total // n``: 8 heads of 256 bytes
    over TP2, rank 1 holds heads 4..7, bytes 1024..2047 of the space; 6 heads over TP4, two per
    rank, ranks hold heads 0..1, 1..2, 3..4 and 4..5, overlapping where the heads they hold
    exceed the total.

    In a manager's spec, an attention buffer's heads are by default the layer's KV heads: K of a
    layer of 8 KV heads on TP2 holds 4 heads of ``size // 4`` bytes. A buffer of heads of its own
    states their bytes (``head_bytes``), and for attention their total over all ranks
    (``total``); recurrent state's total is its heads on this rank times the ranks. A spec states
    the total alone where the layer's KV heads over all ranks are not it, as MLA's one latent head,
    which every rank holds whole (``total=1``). A buffer whose heads are unknown (for attention,
    also their total over all ranks), or whose size they do not divide, is this rank's own share
    (``Own``); with a stated total, ``resolve`` refuses it (ValueError).

    Attributes:
        token_major: Tokens outermost ([token, head, ...], NHD) rather than heads (HND).
        head_bytes: Positive bytes of one head of a page; None in a spec, for the layer's KV heads.
        total: Positive heads over the ranks of the split; None in a spec, for the layer's KV
            heads or for recurrent state, or stated where the layer's KV heads are not it.
        tokens: Positive token positions of a page of a token-major layout; resolved from the
            buffer.
        split: Parallel dimensions from ``GRID_DIMENSIONS``, outer to inner; the tensor-parallel
            ranks by default, those of the buffer's layer group once resolved. Normalized to a
            tuple.
        dtype: The type of the elements by name, or None.

    Raises:
        ValueError: If a count is not a positive integer, ``token_major`` is not a bool, split is
            empty or names a dimension twice or one outside ``GRID_DIMENSIONS``, or the element
            type is neither a name nor a torch type.
    """

    name: ClassVar[str] = "by_head"
    token_major: bool = False
    head_bytes: int | None = None
    total: int | None = None
    tokens: int | None = None
    split: tuple[str, ...] = ("tp",)
    dtype: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.token_major, bool):
            raise ValueError(f"token_major must be a bool, got {self.token_major!r}")
        for name in ("head_bytes", "total", "tokens"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, _integer(getattr(self, name), name, 1))
        object.__setattr__(self, "split", _split(self.split, "a head split"))
        _dtype(self)

    def resolve(self, site: BufferSite) -> RoleLayoutV2:
        head_bytes, total = self.head_bytes, self.total
        if head_bytes is None and not site.state:
            kv_heads, kv_total = site.kv_heads
            if kv_heads > 0 and site.size % kv_heads == 0:
                head_bytes = site.size // kv_heads
                total = kv_total if total is None else total
        elif head_bytes is not None and not site.state and total is None:
            raise ValueError(
                f"heads of {head_bytes} bytes are an attention buffer's own; state how many lie "
                "over all tensor ranks (total)"
            )
        if head_bytes is None or site.size % head_bytes or (total is None and not site.state):
            if self.total is not None:
                raise ValueError(
                    f"a split of {self.total} heads states heads a buffer of {site.size} bytes "
                    "does not hold whole"
                )
            return Own(site.split)
        if total is None:
            # Recurrent state: as many heads on every rank of the split.
            total = site.size // head_bytes * site.ranks
        return dataclasses.replace(
            self,
            head_bytes=head_bytes,
            total=total,
            tokens=site.tokens if self.token_major else None,
            split=site.split,
        )

    def check(self, size: int) -> None:
        if self.head_bytes is None or self.total is None:
            raise ValueError("a head split states its head bytes and total (resolve)")
        if size % self.head_bytes:
            raise ValueError(
                f"{size} bytes is not a whole number of heads of {self.head_bytes} bytes"
            )
        if size // self.head_bytes > self.total:
            raise ValueError(f"{size // self.head_bytes} heads held of {self.total}")
        if self.token_major and (self.tokens is None or self.head_bytes % self.tokens):
            raise ValueError(
                f"heads of {self.head_bytes} bytes are not a whole number of rows of "
                f"{self.tokens} tokens"
            )

    def pieces(self, size: int, ranks: int, rank: int) -> tuple[Piece, ...]:
        heads = size // self.head_bytes
        first = rank * self.total // ranks
        if first + heads > self.total:
            raise ValueError(
                f"rank {rank} of {ranks} holds {heads} heads from head {first}, past the "
                f"{self.total} heads"
            )
        if not self.token_major:
            return (Piece(first * self.head_bytes, 0, size),)
        if heads == self.total:
            return (Piece(0, 0, size),)
        row = self.head_bytes // self.tokens
        return tuple(
            Piece((t * self.total + first) * row, t * heads * row, heads * row)
            for t in range(self.tokens)
        )

    def space(self, size: int, ranks: int) -> tuple:
        if self.token_major:
            space = ("token_major", self.tokens, self.total, self.head_bytes // self.tokens)
        else:
            space = ("head_major", self.total, self.head_bytes)
        return _typed(self, space)

    def even(self, ranks: int) -> bool:
        return self.total % ranks == 0 or ranks % self.total == 0


@dataclass(frozen=True)
class RingSlots(RoleLayoutV2):
    """A ring of ``slots`` token positions alike on every rank: of its page's tokens, position
    ``p`` lies in slot ``p % slots``, so the buffer keeps the latest ``slots`` positions the page
    has written. Each side sizes it as it keeps, so peers may differ in its slots.

    FP4 MLA's high-precision tail on a generation worker, 19 slots in pages of 32 tokens: of page
    1 (positions 32..63) written up to position 50, it holds 32..49, position 40 in slot 2.

    Attributes:
        slots: Positive number of positions the ring keeps on this rank.
        dtype: The type of the elements by name, or None.

    Raises:
        ValueError: If slots is not a positive integer, or the element type is neither a name
            nor a torch type.
    """

    name: ClassVar[str] = "ring"
    positional: ClassVar[bool] = True
    slots: int
    dtype: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "slots", _integer(self.slots, "ring slots", 1))
        _dtype(self)

    def check(self, size: int) -> None:
        if size % self.slots:
            raise ValueError(f"{size} bytes is not a whole number of {self.slots} slots")

    def pieces(self, size: int, ranks: int, rank: int) -> tuple[Piece, ...]:
        return (Piece(0, 0, size // self.slots),)

    def space(self, size: int, ranks: int) -> tuple:
        return _typed(self, ("ring", size // self.slots))

    def positions(self, page: int, end: int, tokens_per_page: int) -> range:
        first = page * tokens_per_page
        written = min(end, first + tokens_per_page)
        return range(max(first, written - self.slots), written)

    def index_of(self, positions: _Positions, tokens_per_page: int) -> _Positions:
        return positions % self.slots

    def stride(self, size: int) -> int:
        return size // self.slots


def check_pieces(layout: RoleLayoutV2, size: int, ranks: int, rank: int) -> None:
    """Check that the pieces a ``size``-byte buffer laid out by ``layout`` holds at place ``rank``
    of ``ranks`` lie back to back over it (over one position's bytes for a positional layout),
    holding no byte of the space twice: bytes no piece describes would never move, or be moved
    twice.

    Raises:
        ValueError: Naming what does not fit.
    """
    extent = layout.stride(size) if layout.positional else size
    pieces = layout.pieces(size, ranks, rank)
    held = sum(p.size for p in pieces)
    if held != extent:
        raise ValueError(
            f"its pieces hold {held} bytes on this rank, not the {extent} bytes of its slot"
        )
    end = 0
    for p in sorted(pieces, key=lambda p: p.offset):
        if p.size <= 0 or p.offset != end:
            raise ValueError(f"its pieces do not lie back to back over its {extent} bytes")
        end += p.size
    spans = sorted((p.at, p.at + p.size) for p in pieces)
    if spans[0][0] < 0 or any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
        raise ValueError("its pieces hold a byte of the space twice, or one before its start")


@dataclass(frozen=True)
class RankTopology:
    """This rank's parallel sizes and ranks, as native RankInfo carries them; a RankInfo gives
    its own (``RankInfo.topology``).

    Attributes:
        tp_size: Positive tensor-parallel group size.
        tp_rank: Zero-based tensor-parallel rank, less than tp_size.
        pp_size: Positive pipeline-parallel group size.
        pp_rank: Zero-based pipeline-parallel rank, less than pp_size.
        cp_size: Positive context-parallel group size.
        cp_rank: Zero-based context-parallel rank, less than cp_size.
        dp_size: Positive attention data-parallel group size.
        dp_rank: Zero-based attention data-parallel rank, less than dp_size.
        attention_dp: Whether attention uses data parallelism; each tensor-parallel rank then
            occupies its own attention data-parallel rank.

    Raises:
        ValueError: If a size is not a positive integer, a rank is not a non-negative integer, any
            such value is a boolean, or a rank lies outside its size."""

    tp_size: int = 1
    tp_rank: int = 0
    pp_size: int = 1
    pp_rank: int = 0
    cp_size: int = 1
    cp_rank: int = 0
    dp_size: int = 1
    dp_rank: int = 0
    attention_dp: bool = False

    def __post_init__(self) -> None:
        for name in ("tp", "pp", "cp", "dp"):
            size = _integer(getattr(self, f"{name}_size"), f"{name}_size", 1)
            rank = _integer(getattr(self, f"{name}_rank"), f"{name}_rank")
            if rank >= size:
                raise ValueError(f"{name} rank {rank} of {size}")
            object.__setattr__(self, f"{name}_size", size)
            object.__setattr__(self, f"{name}_rank", rank)
