# Promote — merging a target's local variants into the shared catalog

A kernel task on a shared entry does not write the shared file: it lands a
**local variant** in the target's own `ops/`,
`ops/<category>/<name>__<site>.py`, with its test at
`tests/unittest/_torch/modeling_v2/models/<family>/<target>/<category>/test_modeling_v2_<name>__<site>.py`,
and the target imports the variant. Tasks on one entry then run side by side.
A `promote:<entry>[,<entry>...]` dispatch ends that: each named shared entry
becomes one entry and one test file in the catalog again, carrying everything
its variants measured, and the target imports the catalog.

Your dispatch prompt lists, per entry, the shared module and test (or that
there is none yet), every local variant and its test, a copy another target
keeps in its own `ops/` when there is one, the target files that import
a variant, and the reports of the smiths that landed the variants, whose
results say what the merge must keep that the files alone do not show (a
test one variant fixed, a helper two variants changed apart). Read all of
them before writing anything.

## The merge, per entry

1. **`__call__` and `reference` are the entry's, unchanged.** A variant adds
   cells, guards and measurements; it does not change the op call. Compare
   each variant's `__call__` and `reference` with the shared entry's (or with
   each other's, for an entry new to the catalog). Where one differs, the
   merged entry takes the form every call site can use, and every caller of
   it, in every target, is updated in the same commit; when no one form
   serves them all, write nothing and end `BLOCKED`, naming the variants and
   the difference.
2. **`CELLS` is the union of the shared entry's and every variant's**, one
   `Cell` per distinct `spec`: two cells with one `spec` are one cell, whose
   `why` says every call site it covers. A cell keeps its `large` mark.
3. **`ARCHS` is the union of theirs.** The certifying run below adds this
   device's arch only after the whole merged test file passed on it.
4. **`is_valid` refuses the union of what the variants refuse**: each guard
   was observed to fail silently on this machine, which makes it a fact about
   the op, not about one call site. Keep each guard test.
5. **The `note`** carries every measurement and hard limit the variants
   recorded. A line that said a value was uncertified goes when a merged cell
   now covers it.
6. **One test file**, `tests/unittest/_torch/modeling_v2/<category>/test_modeling_v2_<name>.py`
   (a multi-rank entry's op matrix and rank body, as its variants had them),
   parametrized off `op.CELLS`, its `_build` covering every `spec` key a
   variant used. A cell keeps the band it was certified under: never widen a
   band to merge two tests. Where two variants derived different bands for
   one kind of cell, the merged test keeps the tighter, with its derivation.

## Then

7. **The target imports the catalog**: each file the prompt lists as
   importing a variant imports
   `tensorrt_llm._torch._experimental.modeling_v2.catalog.<category>.<name>`
   instead, once, and every call site that used a variant calls the entry's
   instance. Another target that kept its own copy imports the catalog too,
   and its copy and its test go.
8. **The variants go**, modules and tests, with `git rm`. A new
   `ops/<category>/__init__.py` stays only while its directory holds an entry.
9. **Run once, at the end**, in one pytest process: every merged test file of
   this dispatch and the claims gates
   (`tests/unittest/_torch/modeling_v2/test_modeling_v2_claims.py`,
   `tests/unittest/_torch/modeling_v2/test_modeling_v2_no_stale_claims.py`).
   That is the certifying run: it post-dates every file of every merged entry,
   in both trees. A cell that passed in its variant and fails merged is a
   conflict between variants: report it and end `BLOCKED`, never land it.
10. **Commit once**, everything above in one commit through the commit helper,
    so no tree ever imports a half-promoted entry.

## Report

Per entry: the sources merged (shared, variants, another target's copy), the
cells before and after (count, and which `why`s merged), `ARCHS`, the guards
carried over, the test command and its wall time in the certifying run, and
every caller whose import changed. Decision `LANDED` when every entry is
promoted; `BLOCKED` with the entry and the conflict otherwise. An entry the
spec keeps local (`target.keep_local`) is never in a promote's list.
