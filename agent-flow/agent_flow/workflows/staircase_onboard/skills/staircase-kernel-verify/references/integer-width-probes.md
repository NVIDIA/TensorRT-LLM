# Integer-width boundary probes

Every verify and every onboard records, for each index-typed argument of the
entry, where 32-bit (and fp32) arithmetic stops being exact -- as a hard limit
in the contract, independent of the serving envelope.

## Which arguments

Anything the kernel turns into an address or an index: row offsets, cumulative
sequence lengths, token positions, page tables and slot indices, gather and
scatter indices, top-k indices, the row count itself. Read the wrapper's
signature and the kernel's launch code for the arithmetic: `row * width + col`
in `int`, an offset accumulated in `int32_t`, a `tl.int32` program-id product,
an index stored as `float` on its way through a reduction.

## The three boundaries

| boundary | what overflows | probe at |
|---|---|---|
| total elements 2^31 | an int32 element count or flat offset | 2^31 elements (or 2^31 + 1) |
| rows x width 2^31 | `row * width` in 32 bits, for each row width the entry addresses | `2^31 / width + 1` rows |
| fp32-exact 2^24 | an integer index carried through fp32 | index `2^24 + 1` |

The smith brief computes the row counts for the widths the call sites declare
and says whether each lies inside the envelope.

## Probing without the full footprint

A 2^31-element probe of a bf16 tensor is 4 GiB, which most GPUs hold; where the
op's footprint does not fit, drive the boundary through the cheapest operand:

- the smallest dtype the op accepts for the operand that carries the count;
- a page or slot table that points a small batch at far rows of a large pool,
  rather than a large batch;
- strided or expanded views that give a large logical row count over little
  memory, where the op accepts them;
- for the rows x width boundary, the narrowest width the op supports with the
  real width's arithmetic, and the real width at the largest row count that
  fits, both stated.

Compare against a reference built from native torch ops on the rows at and
beyond the boundary, not only the first rows: the failure this exists for
wrote the rows past 2^25 out of place and left the rest correct.

When a boundary genuinely cannot be reached on this machine, record it as
`unprobed` with the reason and the largest value that was probed: the hard
limit is then that value, not the boundary.

## The contract table

```markdown
## Hard limits

| argument | arithmetic | boundary | probed at | result | receipt |
|---|---|---|---|---|---|
| `out` offset | `row * 512` in int32 | 2^31 / 512 = 4194304 rows | 4194305 rows | failed: rows >= 2^22 written out of place | sm_103 |
| `num_tokens` | element count, int64 | 2^31 elements | 2^31 + 1 | passed | sm_103 |
| `topk_idx` | through fp32 in the reduction | 2^24 | 16777217 | passed (int32 path, never fp32) | sm_103 |
```

A `failed` row is a hard limit every target must stay below, and a value of the
serving envelope above it is a blocker for the target, not for the entry. An
`unprobed` row limits the entry to what was probed. Rewrite the table only from
a run you watched, and re-read it whenever a change removes a memory cap that
kept the reachable rows small.
