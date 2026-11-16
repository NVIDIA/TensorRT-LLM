# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``lg_splitk`` kernel's inline PTX and its activation-tile address map, as plain text.

Ported from the sm_103__tep ``lg_splitk`` (stage S3 / S3b of the overlap-first plan, branch
``mk/s3b``) to this target's 9..32-row split-K tactics. Pure: nothing here imports torch, CUDA or
the CuTe DSL, so a host-only test evaluates the very strings and the very address function the
kernel uses. ``ops/gemm/lg_splitk.py`` inlines the strings and calls ``b_chunk_offsets`` on DSL
integers.

Activation tile. The split-K kernel (``ops/gemm/splitk_gemm``) swaps the GEMM: its operand B is
the activation ``x [rows, K]`` (K-major); one k-tile of one token tile is ``cta_n`` rows x
``CTA_K`` (128) BF16 columns, and the shared-memory stage holding it is the K-major 128-byte
swizzle layout its TMA load writes (CuTe tiles the 8-row x 64-column swizzle atom over the rows
first, then over K):

* the 64-column half ``h`` of the k-tile is bytes ``[h * cta_n * 128, ...)``;
* within it, row ``r`` is 128 bytes at ``r * 128``;
* its 16-byte chunk ``c`` (8 BF16 columns) sits at chunk slot ``c ^ (r & 7)`` (address bits
  4..6 XOR bits 7..9: the 128-byte swizzle).

A k-tile is ``cta_n * 16`` such chunks. With ``P`` polling warps, lane ``l`` of polling warp
``p`` owns chunks ``l + 32 (p + P j)`` (``j < cta_n / (2 P)``), so a half-warp reads one 256-byte
row segment per load.

Arrival test. The arena the producer publishes into is filled with the direct transports'
sentinel word ``0xFFFEFFFE``; a 16-byte chunk has arrived when none of its eight 16-bit halves is
``0xFFFE`` -- per half, so a word whose halves land in two stores is never taken half-written.

Control block and generations. The arena's ``[rows, K]`` is cut into *units*: one K slice
(``split_k`` of them) of one token tile (``rows / cta_n`` of them), read by the CTAs of that
split rank and token tile only (one per output tile). Per unit the device control block holds a
release counter (``COUNTER_WORD + u``) and a generation tag (``GEN_WORD + u``, on its own
128-byte lines: the polling warps spin on them). The CTA that brings a unit's counter to the
unit's CTA count writes the sentinel back over the unit, zeroes the counter and stores the
call's ``tag`` into the generation with release semantics (``RETIRE_SLICE``); the next call on
the arena polls a unit only once an acquire load of its generation returns that tag
(``AWAIT_GEN``).
"""

#: K columns of one k-tile of the split-K kernel (its ``_CTA_K``).
CTA_K = 128
#: Bytes of one BF16 element.
ELEMENT_BYTES = 2
#: Bytes of one polled / stored chunk (one ``v4.b32`` access).
CHUNK_BYTES = 16
#: Chunks of one k-tile row: 128 columns x 2 bytes / 16.
CHUNKS_PER_ROW = CTA_K * ELEMENT_BYTES // CHUNK_BYTES
#: Lanes of a polling warp.
LANES = 32
#: The arena's empty word and its half (the direct transports' sentinel).
SENTINEL_WORD = 0xFFFEFFFE
SENTINEL_HALF = 0xFFFE
#: ``SENTINEL_WORD`` as a signed 32-bit integer (what ``torch.int32`` fills take).
SENTINEL_I32 = SENTINEL_WORD - (1 << 32)
#: Return codes of ``poll_tile_ptx`` and ``AWAIT_GEN``.
ARRIVED, FALLBACK = 0, 2
#: Most chunks one lane keeps in flight per polling round.
MAX_POLL_CHUNKS = 16

#: The arena's device control block, in 32-bit words: unit ``u``'s release counter at
#: ``COUNTER_WORD + u``, the fallback count (when no pinned host word is available) at
#: ``FALLBACK_WORD``, unit ``u``'s generation tag at ``GEN_WORD + u``.
MAX_UNITS = 16
COUNTER_WORD = 0
FALLBACK_WORD = 16
GEN_WORD = 32
CONTROL_WORDS = 64


def unit_index(split_rank, token_tile, split_k):
    """The unit of K slice ``split_rank`` of token tile ``token_tile`` (Python or DSL integers)."""
    return token_tile * split_k + split_rank


def counter_word(unit: int) -> int:
    """The control word of unit ``unit``'s release counter."""
    return COUNTER_WORD + unit


def gen_word(unit: int) -> int:
    """The control word of unit ``unit``'s generation tag."""
    return GEN_WORD + unit


def slice_chunk_offset(row, chunk, row_bytes, k_tile_start):
    """Byte offset from a token tile's first row of 16-byte chunk ``chunk`` of row ``row`` of the
    K slice whose first k-tile is ``k_tile_start`` (its ``k_tile_count * 16`` chunks per row are
    contiguous). The same operators as ``b_chunk_offsets``: Python or DSL integers."""
    return row * row_bytes + k_tile_start * (CTA_K * ELEMENT_BYTES) + (chunk << 4)


def chunks_per_lane(cta_n: int, poll_warps: int = 1) -> int:
    """Chunks each lane of each of ``poll_warps`` polling warps loads per k-tile."""
    if cta_n % 8 or cta_n <= 0:
        raise ValueError(f"cta_n must be a positive multiple of 8, got {cta_n}")
    total = cta_n * CHUNKS_PER_ROW // LANES
    if poll_warps <= 0 or total % poll_warps:
        raise ValueError(f"{poll_warps} polling warps do not split {total} chunks per lane")
    return total // poll_warps


def poll_group(cta_n: int, stages: int, poll_warps: int = 1) -> int:
    """K-tiles a polling warp polls per round: the largest power of two whose chunks fit
    ``MAX_POLL_CHUNKS`` per lane, at most the ring's stages (their stages are freed first)."""
    cap = max(1, min(int(stages), MAX_POLL_CHUNKS // chunks_per_lane(cta_n, poll_warps)))
    group = 1
    while group * 2 <= cap:
        group *= 2
    return group


def chunk_index(lane, poll_index, poll_warps: int, j: int):
    """The k-tile chunk lane ``lane`` of polling warp ``poll_index`` loads ``j``-th."""
    return lane + (poll_index + poll_warps * j) * LANES


def b_chunk_offsets(chunk, row_bytes, cta_n: int):
    """``(global byte offset, shared byte offset)`` of k-tile chunk ``chunk``.

    Global: from the k-tile's first column of the token tile's first row, rows ``row_bytes``
    apart. Shared: from the stage's base (1024-byte aligned), the TMA ``SWIZZLE_128B`` image
    (module docstring). Only ``>>``, ``&``, ``^``, ``<<``, ``*`` and ``+``, so ``chunk`` and
    ``row_bytes`` may be Python integers or CuTe DSL ``Int32`` values.
    """
    row = chunk >> 4
    column_chunk = chunk & 15
    half = column_chunk >> 3
    slot = column_chunk & 7
    global_offset = row * row_bytes + (column_chunk << 4)
    shared_offset = half * (cta_n * 128) + (row << 7) + ((slot ^ (row & 7)) << 4)
    return global_offset, shared_offset


def _stale_tree(words: int) -> str:
    """PTX leaving in ``m`` the half-wise unsigned minimum of ``w0 .. w{words-1}`` XOR the
    sentinel word, computed in place (the words were stored already); ``m`` has a zero half
    exactly when some word holds the sentinel half."""
    text = ""
    for i in range(words):
        text += f"xor.b32 w{i}, w{i}, 0xFFFEFFFE;\n"
    level = [f"w{i}" for i in range(words)]
    while len(level) > 1:
        nxt = []
        for a, b in zip(level[0::2], level[1::2]):
            text += f"min.u16x2 {a}, {a}, {b};\n"
            nxt.append(a)
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
    text += f"mov.b32 m, {level[0]};\n"
    return text


def poll_tile_ptx(chunks: int, backoff_ns: int = 0) -> str:
    """PTX polling ``chunks`` 16-byte chunks per lane (a group of k-tiles) into shared memory.

    Operands: ``$0`` (out, ``u32``) the code: ``ARRIVED`` (0) once a round finds no sentinel half
    in any lane's chunks, or ``FALLBACK`` (2) once the global timer passed ``$2`` ns after ``$1``
    with some chunk still holding one. ``$1`` (``u64``) the start time (``%globaltimer``), ``$2``
    (``u64``) the deadline in ns; ``$3 .. $(2 + chunks)`` (``u64``) the chunks' global addresses;
    then ``chunks`` shared addresses (``u32``).

    Each round loads every chunk with ``ld.volatile.global.v4.b32`` and stores it to its shared
    address (``st.shared.v4.b32``) before testing it: a round that still finds a sentinel half
    stores bytes the next round overwrites, and the stage is not handed to the MMA until a
    round finds none (the stores of that round are then the final bytes, one per chunk). The
    arrival and the deadline are warp votes, so the code is the same on all 32 lanes and every
    lane leaves the loop in the same round. ``backoff_ns`` > 0: ``nanosleep`` between rounds
    (fewer polls of the same lines while the producer has not stored). Labels are local to the
    ``{ }`` block, so the statement may be inlined more than once.
    """
    if chunks <= 0 or chunks > MAX_POLL_CHUNKS:
        raise ValueError(f"1..{MAX_POLL_CHUNKS} chunks per lane, got {chunks}")
    words = 4 * chunks
    g0, s0 = 3, 3 + chunks
    text = "{\n.reg .pred p_stale, p_any, p_late, p_any_late;\n"
    text += f".reg .b32 w<{words}>, m, t, n;\n.reg .u64 now, elapsed;\n"
    text += "LG_POLL:\n"
    for c in range(chunks):
        w = ", ".join(f"w{4 * c + i}" for i in range(4))
        text += f"ld.volatile.global.v4.b32 {{{w}}}, [${g0 + c}];\n"
    for c in range(chunks):
        w = ", ".join(f"w{4 * c + i}" for i in range(4))
        text += f"st.shared.v4.b32 [${s0 + c}], {{{w}}};\n"
    text += _stale_tree(words)
    # (m - 0x00010001) & ~m & 0x80008000 != 0 exactly when m has a zero half.
    text += (
        "sub.u32 t, m, 0x00010001;\nnot.b32 n, m;\nand.b32 t, t, n;\nand.b32 t, t, 0x80008000;\n"
        "setp.ne.u32 p_stale, t, 0;\n"
        "vote.sync.any.pred p_any, p_stale, 0xffffffff;\n"
        "@!p_any bra.uni LG_ARRIVED;\n"
        "mov.u64 now, %globaltimer;\nsub.u64 elapsed, now, $1;\n"
        "setp.gt.u64 p_late, elapsed, $2;\n"
        "vote.sync.any.pred p_any_late, p_late, 0xffffffff;\n"
        "@p_any_late bra.uni LG_LATE;\n"
        + (f"nanosleep.u32 {int(backoff_ns)};\n" if int(backoff_ns) > 0 else "")
        + "bra.uni LG_POLL;\n"
        f"LG_LATE:\nmov.b32 $0, {FALLBACK};\nbra.uni LG_END;\n"
        f"LG_ARRIVED:\nmov.b32 $0, {ARRIVED};\nLG_END:\n}}"
    )
    return text


def poll_tile_constraints(chunks: int) -> str:
    """The inline-asm constraint string of ``poll_tile_ptx(chunks)``."""
    return "=r,l,l" + ",l" * chunks + ",r" * chunks


def copy_chunks_ptx(chunks: int) -> str:
    """PTX copying ``chunks`` 16-byte chunks per lane from global (``$1 .. $chunks``, ``u64``,
    each offset by ``$0``: the fallback source minus the arena) to shared memory (then
    ``chunks`` ``u32`` addresses), one chunk at a time, no test: the fallback after
    ``griddepcontrol.wait``, when the source holds the producer's final values (which may contain
    a sentinel half). ``ld.volatile`` like the polls, so no L1 line of an older value serves it."""
    if chunks <= 0 or chunks > MAX_POLL_CHUNKS:
        raise ValueError(f"1..{MAX_POLL_CHUNKS} chunks per lane, got {chunks}")
    text = "{\n.reg .b32 w<4>;\n.reg .u64 a;\n"
    for c in range(chunks):
        text += f"add.u64 a, ${1 + c}, $0;\n"
        text += "ld.volatile.global.v4.b32 {w0, w1, w2, w3}, [a];\n"
        text += f"st.shared.v4.b32 [${1 + chunks + c}], {{w0, w1, w2, w3}};\n"
    return text + "}"


def copy_chunks_constraints(chunks: int) -> str:
    """The inline-asm constraint string of ``copy_chunks_ptx(chunks)``."""
    return "l" + ",l" * chunks + ",r" * chunks


#: ``fence.proxy.async.shared::cta``: this thread's generic shared-memory stores are ordered
#: before the async proxy's (the tcgen05 MMA's) reads of them.
FENCE_PROXY_ASYNC_SHARED = "fence.proxy.async.shared::cta;"

#: The global timer (ns).
GLOBALTIMER = "mov.u64 $0, %globaltimer;"

#: Lane 0 adds one to the fallback count at ``$0`` (a pinned host word, or a device word);
#: nothing when ``$0`` is 0.
COUNT_FALLBACK = (
    "{\n.reg .pred p, q;\n.reg .b32 lane, one;\nmov.u32 lane, %laneid;\nmov.b32 one, 1;\n"
    "setp.eq.u32 p, lane, 0;\nsetp.ne.u64 q, $0, 0;\nand.pred p, p, q;\n"
    "@p red.relaxed.sys.global.add.u32 [$0], one;\n}"
)

#: Every lane gets 1 when this CTA is the last of ``$2`` CTAs to add one to the counter at
#: ``$1`` (lane 0's ``atom.acq_rel.gpu`` add, its old value broadcast), else 0. Memory order:
#: every lane first executes ``fence.acq_rel.gpu`` (the add's release is then cumulative over
#: what this warp observed through the CTA's barriers -- the polling warps' reads of the
#: producer's words); in the last CTA every lane then loads the counter with ``ld.acquire.gpu``,
#: so each lane's later stores (the sentinels) come after every CTA's release.
LAST_CTA = (
    "{\n.reg .pred p, q;\n.reg .b32 lane, old, one, seen;\nmov.u32 lane, %laneid;\n"
    "mov.b32 one, 1;\nmov.b32 old, 0;\nsetp.eq.u32 p, lane, 0;\n"
    "fence.acq_rel.gpu;\n"
    "@p atom.acq_rel.gpu.global.add.u32 old, [$1], one;\n"
    "shfl.sync.idx.b32 old, old, 0, 0x1f, 0xffffffff;\n"
    "add.u32 old, old, 1;\nsetp.eq.u32 q, old, $2;\n"
    "@q ld.acquire.gpu.global.u32 seen, [$1];\n"
    "selp.b32 $0, 1, 0, q;\n}"
)

#: One 16-byte chunk of the arena back to the sentinel.
STORE_SENTINEL_X4 = (
    "{\n.reg .b32 s;\nmov.b32 s, 0xFFFEFFFE;\nst.global.v4.b32 [$0], {s, s, s, s};\n}"
)

#: Wait for the unit's generation. ``$0`` (out, ``u32``): ``ARRIVED`` (0) once an
#: ``ld.acquire.gpu`` of the word at ``$1`` returns ``$2`` on every lane (a warp vote), or
#: ``FALLBACK`` (2) once the global timer passed ``$4`` ns after ``$3`` without it. Every lane
#: loads the word itself, so each lane's later loads (the polls) are ordered after its own
#: acquire. Labels are local to the ``{ }`` block.
AWAIT_GEN = (
    "{\n.reg .pred p_ok, p_all, p_late, p_any_late;\n.reg .b32 g;\n.reg .u64 now, elapsed;\n"
    "LG_GEN:\n"
    "ld.acquire.gpu.global.b32 g, [$1];\n"
    "setp.eq.u32 p_ok, g, $2;\n"
    "vote.sync.all.pred p_all, p_ok, 0xffffffff;\n"
    "@p_all bra.uni LG_GEN_OK;\n"
    "mov.u64 now, %globaltimer;\nsub.u64 elapsed, now, $3;\n"
    "setp.gt.u64 p_late, elapsed, $4;\n"
    "vote.sync.any.pred p_any_late, p_late, 0xffffffff;\n"
    "@!p_any_late bra.uni LG_GEN;\n"
    f"mov.b32 $0, {FALLBACK};\nbra.uni LG_GEN_END;\n"
    f"LG_GEN_OK:\nmov.b32 $0, {ARRIVED};\nLG_GEN_END:\n}}"
)
AWAIT_GEN_CONSTRAINTS = "=r,l,r,l,l"

#: Retire a unit after its sentinel stores (every lane of the warp executes it): each lane's
#: ``fence.acq_rel.gpu`` orders its own sentinel stores, ``bar.warp.sync`` orders the warp's
#: lanes, then lane 0 zeroes the unit's counter at ``$0`` and stores the call's tag ``$2`` into
#: the unit's generation at ``$1`` with ``st.release.gpu`` (cumulative over what the warp
#: stored before it).
RETIRE_SLICE = (
    "{\n.reg .pred p;\n.reg .b32 lane, zero;\n"
    "fence.acq_rel.gpu;\n"
    "bar.warp.sync 0xffffffff;\n"
    "mov.u32 lane, %laneid;\nmov.b32 zero, 0;\nsetp.eq.u32 p, lane, 0;\n"
    "@p st.relaxed.gpu.global.b32 [$0], zero;\n"
    "@p st.release.gpu.global.b32 [$1], $2;\n}"
)
RETIRE_SLICE_CONSTRAINTS = "l,l,r"
