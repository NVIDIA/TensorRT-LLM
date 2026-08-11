# Vendored Sources

TensorRT-LLM keeps some upstream source trees in this repository so they can be
built, packaged, and reviewed with the code that uses them. The generic
vendoring tool records where each tree came from, materializes it reproducibly,
and detects edits that are not represented by its lock entry.

The generated lock is `3rdparty/vendor_sources.lock.yaml`. It records an
upstream Git URL, an immutable commit, the source and destination directories,
the selected files, any downstream patch and its content digest, and a digest
of the materialized destination. A short branch or tag may be recorded to
explain where the commit came from, but the full commit is authoritative.

Use `scripts/vendor_sources.py` for every lock or vendor-state change. Do not
edit the YAML, generated patches, or digests by hand.

All examples below use the default lock. For an isolated test or another
consumer repository, place `--lock PATH` before the subcommand.

## Enforcement states

A vendor is in one of three states:

- **Exact**: the selected destination files are byte-for-byte copies of the
  selected upstream files.
- **Patched**: a deterministic patch records the downstream additions,
  modifications, and deletions applied after copying the upstream files.
- **Temporary divergence**: urgent destination edits are accepted for a
  bounded period. The tool records the reason, creation and expiration dates,
  affected files, and accepted digest. Pinning a commit that materializes the
  accepted tree removes the divergence and restores exact or patched
  enforcement.

Temporary divergence is an escape hatch, not a normal update path. Prefer to
validate a fix locally, export it to an upstream checkout, open the upstream
change, and pin the resulting commit.

## Inspect vendors

List the lock entries or show their enforcement state:

```bash
python scripts/vendor_sources.py list
python scripts/vendor_sources.py status
python scripts/vendor_sources.py status VENDOR
```

## Add or adopt a vendor

Create a vendor from an immutable commit and a local upstream checkout:

```bash
python scripts/vendor_sources.py create VENDOR \
  --url https://example.com/organization/repository.git \
  --branch topic-branch \
  --commit FULL_COMMIT \
  --source path/in/upstream \
  --destination path/in/tensorrt-llm \
  --include '**/*.py' \
  --repo /path/to/upstream
```

Use `--tag TAG` instead of `--branch BRANCH` for a tagged source. Branch and
tag names are informational; `FULL_COMMIT` remains the reproducibility pin.
Without `--repo`, the tool obtains the commit from the recorded URL. A normal
`create` copies the selected source into a new destination.

If the destination already contains the intended files, add `--adopt exact`
to require an exact upstream match or `--adopt patched` to generate a patch
from upstream to the existing destination. Adoption never silently accepts an
unrepresented difference. For later destination edits, use the patch or
temporary-divergence commands below.

## Sync and patch a vendor

Materialize the locked source and downstream patch into the destination:

```bash
python scripts/vendor_sources.py sync VENDOR --repo /path/to/upstream
```

After intentionally editing an exact destination, create its patch and update
its digest together:

```bash
python scripts/vendor_sources.py patch VENDOR create --repo /path/to/upstream
```

After intentionally editing a patched destination, regenerate its patch and
digest together:

```bash
python scripts/vendor_sources.py patch VENDOR refresh --repo /path/to/upstream
```

Drop a no-longer-needed patch only after the destination matches the exact
upstream selection:

```bash
python scripts/vendor_sources.py patch VENDOR drop --repo /path/to/upstream
```

Patch files are generated artifacts under `3rdparty/vendor_patches/`. Review
them, but update them through the tool.

## Round-trip a change through upstream

To test a fix in TensorRT-LLM first, edit the destination and capture the
temporary divergence described below. Export the accepted destination to a
local upstream checkout:

```bash
python scripts/vendor_sources.py export VENDOR --repo /path/to/upstream
```

Run upstream tests, review and commit the local checkout, then push it to a
branch or fork. The export command changes the selected source directory in
that checkout; it does not commit or push anything.

After the upstream commit exists, replace the recorded URL, branch or tag, and
commit with one validated from the local checkout:

```bash
python scripts/vendor_sources.py pin VENDOR \
  --url https://example.com/my-fork/repository.git \
  --branch upstream-fix \
  --commit NEW_FULL_COMMIT \
  --repo /path/to/upstream
```

When the new exact or patched materialization equals the checked-in
destination, `pin` removes temporary-divergence metadata and restores normal
enforcement. It does not silently discard a divergence that the new pin cannot
reproduce.

## Temporary divergence

Capture an urgent edit with an explicit expiration date:

```bash
python scripts/vendor_sources.py divergence VENDOR capture \
  --reason 'Urgent correctness fix' \
  --expires 2026-08-25 \
  --repo /path/to/upstream
```

The tool derives the creation date, affected files, and accepted digest. If
the urgent edit changes again, refresh those generated fields while preserving
the reason and expiration. An exception may last at most 30 days; extending it
requires a new, reviewed capture after the current exception is cleared.

```bash
python scripts/vendor_sources.py divergence VENDOR refresh \
  --repo /path/to/upstream
```

Normally, use `pin` to leave this state. To discard the temporary acceptance
without changing the destination, run:

```bash
python scripts/vendor_sources.py divergence VENDOR clear
```

The next offline check fails until the destination is synchronized or its
change is represented by a patch or a new pin. Expired divergence always
fails.

## Remove a vendor

Remove a lock entry and its generated patch while preserving the destination:

```bash
python scripts/vendor_sources.py remove VENDOR
```

The preserved destination is no longer protected by the lock. Delete or move
it separately as part of the reviewed migration that removes the vendor.

## Source access and checks

The default check is deliberately offline:

```bash
python scripts/vendor_sources.py check
```

`check --offline` is an explicit spelling of the same default behavior.

It validates the lock schema and path safety, patch and divergence metadata,
expiration dates, and the checked-in destination digest. It never invokes Git,
performs DNS resolution, or contacts a recorded URL. This is the always-run
pre-commit check, so a contributor who cannot access an internal upstream can
still develop normally.

Use the upstream check when network access is available:

```bash
python scripts/vendor_sources.py check --upstream
```

This verifies every accessible upstream and reports an inaccessible repository
as unavailable rather than failing. If a commit can be obtained, a source,
patch, or destination mismatch is an error. Trusted maintainer CI can require
access to every source:

```bash
python scripts/vendor_sources.py check --upstream --require-access
```

To verify against an existing checkout without contacting the recorded URL,
provide it explicitly:

```bash
python scripts/vendor_sources.py check VENDOR --repo /path/to/upstream
```

The checkout's configured remote may differ from the lock URL; it only needs to
contain the locked commit. Source-consuming commands accept the same `--repo`
form.

An offline digest proves that the committed destination matches the lock. It
cannot independently prove that a URL, commit, and source directory produced
that destination. Creating and pinning vendors therefore require a fetched or
local repository, and URL or commit changes require vendor CODEOWNER review.
Never put credentials in a lock URL. Run checks that use internal credentials
only in a trusted environment, not with pull-request-controlled scripts.

## License and attribution

The vendor lock is a reproducibility record, not a license manifest. Before
adding a vendor, verify that the selected upstream files carry the required
notices and follow [the Python third-party process](py-thirdparty.md) or
[the C++ third-party process](cpp-thirdparty.md), as applicable. Exact upstream
files retain their upstream copyright headers. Add an NVIDIA header only to
files that TensorRT-LLM modifies.

## PrimTS

The `flashinfer-prims-ts` entry selects the complete Python tree under
`flashinfer/attention/prims_ts` and materializes it at
`tensorrt_llm/_torch/attention_backend/prims_ts`. The `**/*.py` selection
deliberately omits upstream README files. A generated patch contains only the
TRT-LLM integration and compatibility changes; all other selected files remain
exact upstream copies.

Use the normal commands with `flashinfer-prims-ts`, for example:

```bash
python scripts/vendor_sources.py status flashinfer-prims-ts
python scripts/vendor_sources.py check flashinfer-prims-ts
```
