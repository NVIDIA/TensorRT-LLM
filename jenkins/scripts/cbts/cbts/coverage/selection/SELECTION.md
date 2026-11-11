<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
# Coverage selection

Tier 2 maps a verified Python diff to recorded code symbols and looks up those
symbols in the coverage DB. Selection preserves recorded execution relationships
and explicit CI obligations. It does not prove that the PR is correct or that new
execution paths have sufficient assertions. Full fallback likewise cannot supply
missing tests.

## Exact lookup

1. Reconstruct the complete pre-image from the raw patch and verify it against the
   checked-out post-image. Use the executable selection patch to exclude changes
   already handled as documentation or formatting.
2. Attribute added lines in the post-image and deleted lines in the pre-image.
   Query the union of those qualified symbol names in the changed file. A removed
   function uses its old name; an adjacent surviving function is not substituted
   for a deleted-line anchor.
3. Function bodies use their recorded qualified names, including named closures
   and methods. Top-level execution uses `<module>`; class bodies use the class
   name. Function signatures, eager annotations, defaults and decorator
   applications belong to the enclosing execution scope.
4. Query `(file, qualname)` directly. Missing function or file rows contribute no
   recorded impact. No caller graph, import-target resolution, binding-consumer
   propagation, decorator interpretation, cache inference, reflection gate or
   pytest shared-state inference expands the result.

The legacy `--no-data-policy` option remains accepted for compatibility; all its
values now use exact lookup and add no tests for missing rows. The default is
`ignore`. Decision diagnostics identify `selection_policy: symbol_lookup` and
report the actual queried symbols and symbols without records.

## Explicit must-run contributions

Import-stage edits require the registered
`tests/integration/test_lists/import_sanity.yml` inventory. The sanity rule owns
no runtime files; it contributes affected stages and additive block filters.
Missing inventory, selectors or active registrations prevent narrowing.

PR-added and PR-modified tests are retained independently by the existing test
rules, even when absent from historical coverage. Coverage narrowing cannot remove
rule-kept entries. A registered parent directory can retain more tests than the
inventory's exact selectors.

These rules do not infer runtime consumers of module bindings. For example,
changing `LIMIT = 10` to `LIMIT = 20` queries the module symbol and requires import
sanity; it does not automatically select every function that reads `LIMIT`.
Likewise, changing a cached body queries that body, without inventing relationships
to cache-hit consumers. Such dependencies need explicit tests or future collection
and DB support; import success is not a runtime-behavior proof.

## Infrastructure and collection boundaries

- Missing or mismatched patches, unreadable/unparsable source, unsupported
  non-Python residual files and rule-forced fallback keep the baseline decision.
- Coverage-artifact validity and residual patch compatibility remain enforced at
  their existing boundaries. A conflict is not waived by a successful lookup.
- Entries absent from the coverage DB, untrusted captures, uninstrumented stage
  families and coarse selectors retain their existing protections.
- Shared-block pruning requires every served family to permit removal. CPU
  always-run behavior, trigger-mode filtering, and multi-GPU approval gates remain
  unchanged.

Lookup hits are retrospective candidate decisions. YAML-entry skip rates are not
expanded pytest-case counts, measured GPU savings, or guarantees of PR safety.
