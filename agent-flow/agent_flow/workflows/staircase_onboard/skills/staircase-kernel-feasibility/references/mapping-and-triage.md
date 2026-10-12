# Mapping and triage — reading a contract to place a step

How to turn one computation step plus this checkpoint's values into one of
three states, and what each state dispatches.

## Reading the certified column

A contract has four places that bound what it certifies. Read all four; the
receipt alone is not the answer.

| where | what it bounds |
|---|---|
| `receipts:` frontmatter | the **arch** the test ran under, and `world_size` for a multi-rank entry. Three-state: `passed` / `failed` / **absent = unknown** |
| **Signature** | shapes, dtypes, layout, contiguity, device per argument |
| **Metadata consumed** | which runtime fields must be populated, and with what |
| **Preconditions** | everything that must hold beforehand — buffers, weight formats, alignment, supported dtypes |

Two reading rules that have each cost a run:

**Where a contract enumerates tested shapes rather than stating a rule, the
enumeration is the column.** "Tested at head counts 8, 16, 128" does not
certify 32. A rule ("any head count divisible by 8") does.

**An arch key that does not match the arch this run targets makes the row
CERT, not OK**, however green the receipt looks. Absent means unknown.
Receipts are keyed by GPU architecture and nothing else — there is no version
pin in tree to drift against — but the freshness rule is unchanged in
substance: a receipt is valid only against the files of *its own entry*, and
an entry now spans two trees (contract and wrapper in the package, GPU test in
the tests tree), so a receipt that predates a write to either is a claim about
files nobody ran.

## The four outcomes

| situation | state | dispatch |
|---|---|---|
| entry exists, all four places cover this checkpoint's values, the receipt covers this run's arch | **OK** | still goes to chapter 1 — `staircase-kernel-verify` confirms the receipt cheaply, no GPU |
| entry exists, some value falls outside | **CERT** | `staircase-kernel-verify`, with the value and the contract line that excludes it |
| no entry, but `references/trtllm-ops.md` has a candidate callable | **GAP** | `staircase-kernel-onboard`, with the candidate and the step it serves |
| no entry and no candidate | **ceiling** | nothing — the run ends here. **First rule out a pybind binding**: the inventory is generated from the torch schema registry and lists `torch.ops.*` only, so a step served by a binding (`thop_attention` is one) searches empty and reads as a ceiling. Check the catalog's existing entries and their *uncertified* surfaces before concluding |

One more case that dispatches nobody: **a missing
`tensorrt_llm/_torch/_experimental/modeling_v2/catalog/torch/` mirror**
for pure glue — tensor layout, movement, allocation, lookup (reshape, split,
concat, copy, empty, expand, embedding, residual add). assembler adds those
itself. It is glue by default: a computation the HF reference expresses as a
named module is never *composed* from mirrors, and a missing kernel there is a
ceiling rather than a mirror to add.

**The one exception**, narrow by design: a computation may be a mirror when it is
**exactly one torch callable**, **no trtllm op serves it** (searched closed over
the inventory *and* the pybind surface the inventory cannot show), and upstream
owns its correctness. Composition is still a ceiling; a mirror chosen while a
trtllm op exists is still wrong. Precedent: `catalog/torch/gelu_tanh.py`. Such a
row is neither GAP nor ceiling — assembler adds the mirror, like any other.
<!-- PORT: verify — `catalog/torch/gelu_tanh.py` is not present in the in-tree
     catalog. Same open question as in this skill's SKILL.md: dropped in the
     move, or still to be ported. The rule stands either way. -->

## Precision in a CERT row

A CERT row becomes smith's dispatch prompt almost verbatim, so vagueness costs
a whole run. It must carry:

- the entry path
- **the value**, not the axis: `nope=192, v=256, H=16 per rank, fp8 pool`, not
  "MLA geometry"
- **the contract line that excludes it**, quoted. If you cannot quote one, the
  row is probably OK and you are dispatching on a hunch

## Two failure modes in this step

**Reading the source instead of the contract.** Nothing forbids you from
opening it — but if a contract cannot answer whether a value is covered, that
is a **defect in the contract**, and patching around it by reading source
leaves the next reader with the same problem. Record the defect; the fix is
smith rewriting the contract from an observed run, which is what
`staircase-kernel-verify` does.

**Grouping rows to "save dispatches".** One dispatch, one entry. Entries that
share an expensive test fixture are still separate dispatches — catalog tests
may not import each other, so there is no fixture to share, only the knowledge
of how to build one, and that travels in smith's reports.
