# FlashInfer header patches

FlashInfer compiles its attention kernels from headers that ship inside the
installed wheel (`flashinfer/data/include`) and caches the compiled kernels per
node. A fix in those headers therefore only takes effect when the installed
headers are edited in place *and* the kernel cache is dropped, and any
`pip install` that replaces FlashInfer silently reverts it.

So the changes live here as patch files with an apply step, rather than as a
modified copy of the package. `scripts/build_wheel.py --install` runs the apply
step after its editable install for exactly that reason.

The patch files modify, and therefore embed fragments of, FlashInfer
(https://github.com/flashinfer-ai/flashinfer), Copyright (c) FlashInfer
contributors, licensed under Apache-2.0. The copyright notice of each patched
upstream file, and the exact upstream release each directory's fragments were
copied from (tag and commit), are recorded in [NOTICE](NOTICE).

## Contents

The hunks are written against one FlashInfer release, so each release has its
own directory named for its version:

```
flashinfer-patches/0.6.16/0001-...patch
flashinfer-patches/0.6.18/0001-...patch
```

The apply script reads the `flashinfer-python` pin from `requirements.txt` and
uses the directory of that exact version. A pin with no directory is a hard
error, not a skip: applying another release's hunks, or applying none while
reporting success, both end in a build whose kernels are not the ones this tree
describes. Directories for versions the tree does not pin are kept so a pin bump
does not have to re-derive them.

Both patches carry the same change in every directory:

| file | what it changes |
| --- | --- |
| `0001-zero-fill-the-k-tail-tile.patch` | K (and its scale factors) zero-fill the out-of-range rows of a partial KV tile, as V already does |
| `0002-treat-an-empty-attention-state-as-absent.patch` | an attention state with no keys is skipped when merging, in `threadblock_sync_mdo_states` and `state_t::merge` |

Each patch is one mitigation, kept separate so a measurement can attribute a
change to one or the other.

Whether a patch is applied is decided by searching the installed headers for the
lines it adds, so those lines are the same text in every version directory. Keep
them identical when re-deriving; changing the wording changes what `--check`
looks for.

## Use

```bash
flashinfer-patches/apply-flashinfer-patches.py            # apply, drop the kernel cache
flashinfer-patches/apply-flashinfer-patches.py --revert   # back to the stock headers
flashinfer-patches/apply-flashinfer-patches.py --check    # report state, change nothing
flashinfer-patches/apply-flashinfer-patches.py --drop-cache
```

`--check` exits 0 when every patch is applied, 1 when none are, 2 when the tree
is half patched.

Exit code 4 means FlashInfer is not installed for the interpreter that ran the
script, so there is nothing to patch. That is the only outcome
`scripts/build_wheel.py` treats as a skip; every other non-zero exit fails the
build, because `patch` works a hunk at a time and a rejected patch leaves some
headers stock and others changed. `--check` reports which, `--revert` gets back
to stock.

The kernel cache is **per rank** (`FLASHINFER_WORKSPACE_BASE` carries the rank),
so on a multi-rank job apply once with a single task and then run
`--drop-cache` under the same task count as the job that will use the kernels.
Dropping only rank 0's cache leaves the other ranks replaying kernels built from
the old headers, which looks exactly like a patch that changed nothing.

Whether a patch is already applied is decided by looking for the lines it adds.
GNU patch answers `Unreversed patch detected!  Ignoring -R` and then applies a
patch *forward* with exit code 0, so a reverse dry run reports a stock tree as
patched, and `--revert` on a stock tree would apply it.

## Regenerating after a FlashInfer upgrade

Re-derive the hunks from the new sources rather than forcing the old ones, into
a new directory; do not edit the directory of a version that is still pinned
somewhere.

```bash
pip download flashinfer-python==<new> --no-deps -d <dir>     # no install needed
unzip -o <dir>/flashinfer_python-<new>-*.whl 'flashinfer/data/include/*' -d <tree>
```

Apply the old patches to a copy of the new headers, fix up whatever the rejects
say by hand, `diff -u` the result against pristine copies, and confirm both
`patch --fuzz=0` and `--reverse` are clean on an untouched tree. Then bump the
pin in `requirements.txt` in the same change.
