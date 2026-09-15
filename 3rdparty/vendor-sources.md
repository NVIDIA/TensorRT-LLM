# Vendored Sources

TensorRT-LLM keeps some upstream source trees in this repository so they can be
built, packaged, and reviewed with the code that uses them. The generic
vendoring tool records where each tree came from, materializes it reproducibly,
and rejects destination edits that are not represented by its lock entry.

The generated lock is `3rdparty/vendor_sources.lock.yaml`. It records an
upstream Git URL, an immutable commit, the source and destination directories,
the selected files, any persistent compatibility patch and its content digest,
and a digest of the materialized destination. A short branch or tag may explain
where the commit came from, but the full commit is authoritative.

Use `scripts/vendor_sources.py` for every lock or vendor-state change. Do not
edit the YAML, generated patches, or digests by hand. All examples below use the
default lock. For an isolated test or another consumer repository, place
`--lock PATH` before the subcommand.

## Lock contract

A locked vendor has one of two durable states:

- **Exact**: the selected destination files are byte-for-byte copies of the
  selected files at the locked upstream commit.
- **Patched**: applying a deterministic, persistent compatibility patch to
  those upstream files reproduces the destination exactly. Use this patch only
  for TensorRT-LLM-specific adaptations that do not belong upstream.

A destination edit is not a third state. While such an edit is pending,
`status`, the default offline `check`, and the pre-commit check intentionally
fail. Resolve it by discarding it with `sync`, recording a TensorRT-LLM-only
adaptation with `patch`, or exporting an upstream-worthy change and pinning the
resulting commit. `export` accepts this pending destination delta by default and
does not change the lock or persistent patch.

## Choose a command

```mermaid
flowchart TD
    A{Lock entry exists?}
    A -- No --> B{Destination exists?}
    B -- No --> C[create]
    B -- Yes --> D[create --adopt exact or patched]
    A -- Yes --> E{What do you need?}
    E -- Inspect --> F[list, status, or check]
    E -- Restore locked bytes --> G[sync current immutable pin]
    E -- Use a newer upstream commit --> H[Prepare matching destination, then pin]
    E -- Destination changed --> I{Should the change go upstream?}
    I -- No, TensorRT-LLM only --> J[patch create or refresh]
    I -- Yes --> K[Temporary branch, export, commit and push, then pin]
    E -- Stop vendoring --> L[remove]
```

`sync` only restores the commit and compatibility patch already recorded in the
lock. It never discovers, imports, or pins a newer upstream commit. To move to a
new upstream revision, first make the destination equal that revision plus the
existing compatibility patch, then use `pin`.

## Inspect vendors

List entries, or run the offline integrity status for all or one vendor:

```bash
python scripts/vendor_sources.py list
python scripts/vendor_sources.py status
python scripts/vendor_sources.py status VENDOR
python scripts/vendor_sources.py check VENDOR
```

`status` and the default `check` exit unsuccessfully if the destination has a
pending delta. That failure is expected during an export workflow and remains
until `pin` succeeds.

## Add or adopt a vendor

When neither the lock entry nor destination exists, create both from an
immutable commit and a local upstream checkout:

```bash
python scripts/vendor_sources.py create VENDOR \
  --url https://example.com/organization/repository.git \
  --branch main \
  --commit FULL_COMMIT \
  --source path/in/upstream \
  --destination path/in/tensorrt-llm \
  --include '**/*.py' \
  --repo /path/to/upstream
```

Use `--tag TAG` instead of `--branch BRANCH` for a tagged source. Without
`--repo`, the tool obtains the commit from the recorded URL.

If the destination already exists but has no lock entry, adopt it. Use `exact`
to require an exact upstream match:

```bash
python scripts/vendor_sources.py create VENDOR \
  --url https://example.com/organization/repository.git \
  --commit FULL_COMMIT \
  --source path/in/upstream \
  --destination path/in/tensorrt-llm \
  --include '**/*.py' \
  --adopt exact \
  --repo /path/to/upstream
```

Use `--adopt patched` instead to capture intentional TensorRT-LLM compatibility
adaptations. Adoption never silently accepts an unrepresented difference.

## Restore the current lock

Discard destination edits and reproduce the currently locked upstream commit
plus its persistent patch:

```bash
python scripts/vendor_sources.py sync VENDOR --repo /path/to/upstream
```

This overwrites the selected destination files. It does not update the lock,
look at a branch tip, or choose a newer commit.

## Maintain a TensorRT-LLM compatibility patch

After editing an exact destination for a change that must remain downstream,
create its persistent patch:

```bash
python scripts/vendor_sources.py patch VENDOR create --repo /path/to/upstream
```

After intentionally changing an already patched destination, regenerate the
patch:

```bash
python scripts/vendor_sources.py patch VENDOR refresh --repo /path/to/upstream
```

Drop a no-longer-needed patch only after the destination exactly matches the
currently locked upstream selection:

```bash
python scripts/vendor_sources.py patch VENDOR drop --repo /path/to/upstream
```

Generated patches live under `3rdparty/vendor_patches/`. Review them, but update
them only through the tool. Do not use a persistent patch for a change that
should be contributed upstream; use the export workflow instead.

## Export a destination change upstream

Start with the desired change in the TensorRT-LLM destination. The offline
check now fails by design. In a clean upstream checkout, create a temporary
branch at the currently locked commit **before** exporting:

```bash
git -C /path/to/upstream switch -c trtllm-vendor-fix LOCKED_FULL_COMMIT
python scripts/vendor_sources.py export VENDOR --repo /path/to/upstream
```

The upstream checkout's selected source must be clean before export and its
`HEAD` must equal the locked commit. `export` computes the pending destination
delta relative to the locked materialization, applies only that delta to the
raw upstream source, and leaves the vendor lock, destination, and persistent
compatibility patch unchanged.

Run the upstream tests, review the result, then commit and push the temporary
branch:

```bash
git -C /path/to/upstream add path/in/upstream
git -C /path/to/upstream commit -s -m 'Apply exported fix'
git -C /path/to/upstream push -u origin trtllm-vendor-fix
```

Finally, pin the committed revision from that checkout:

```bash
python scripts/vendor_sources.py pin VENDOR \
  --url https://example.com/my-fork/repository.git \
  --branch trtllm-vendor-fix \
  --commit NEW_FULL_COMMIT \
  --repo /path/to/upstream
```

`pin` first tries the selected files at `NEW_FULL_COMMIT` plus the existing
persistent compatibility patch. They must exactly equal the checked-in
destination. One exception is safe: if the raw new commit itself exactly equals
the destination, upstream has absorbed the compatibility patch, so `pin` drops
that patch and its metadata. Otherwise `pin` does not absorb a mismatch,
regenerate the patch, or copy candidate files into the destination. On success
it durably updates the immutable lock before removing an absorbed patch and
restores passing offline checks. If the patch cannot be removed after that
commit, `pin` succeeds with a warning and leaves a safe, unreferenced orphan;
delete the reported file manually. A failure before the durable lock commit
does not remove the existing patch. If directory synchronization fails after
the atomic replacement, the lock may already show the new pin, but the retained
patch keeps either recovered lock version reproducible.

The same rule applies when adopting a newer commit that was developed upstream
first: prepare the destination to exactly match the proposed commit plus the
existing patch, then run `pin`. Do not use `sync` to look for that commit.

## Remove a vendor

Remove a lock entry and its generated compatibility patch while preserving the
destination:

```bash
python scripts/vendor_sources.py remove VENDOR
```

The preserved destination is no longer protected by the lock. Delete or move
it separately as part of the reviewed migration that removes the vendor.

## Source access and checks

The default check is deliberately offline:

```bash
python scripts/vendor_sources.py check
python scripts/vendor_sources.py check --offline
```

It validates the lock schema and path safety, patch metadata, and the checked-in
destination digest. It never invokes Git, performs DNS resolution, or contacts
a recorded URL. This is the always-run pre-commit check. A pending destination
delta therefore blocks a commit until it is synchronized, patched, or pinned.

When network access is available, attempt verification against every recorded
upstream:

```bash
python scripts/vendor_sources.py check --upstream
```

An inaccessible repository is reported as unavailable rather than failing. If
a commit can be obtained, a source, patch, or destination mismatch is an error.
Trusted maintainer CI can require access to every source:

```bash
python scripts/vendor_sources.py check --upstream --require-access
```

To verify one vendor against an existing checkout without contacting the
recorded URL, provide it explicitly:

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

## Promote a reviewed PrimTS revision

`scripts/maintain_prims_ts.py promote` automates the maintainer follow-up after
a TRT-LLM PR has merged a PrimTS code update from a temporary FlashInfer branch.
It does not import new kernel code or implement periodic upstream refreshes.
Use a trusted checkout of this tool; do not execute a PR-supplied copy with
maintainer credentials.

### Maintenance workflow

The source-update PR reviews code changes; the promotion PR only restores the
canonical lock location. The diagram shows publication with `--auto-merge`;
the CLI remains read-only unless `--publish` is supplied.

```mermaid
flowchart TD
    canonical["Canonical FlashInfer dev branch"]
    temporary["Developer temporary branch<br/>PrimTS code changes"]
    source_pr["TRT-LLM source-update PR<br/>vendor code and pin temporary SHA"]
    pairing["Paired FlashInfer PR<br/>or explicit unpaired reason"]
    verify["promote: verify merged PR, immutable SHA,<br/>canonical target and fast-forward range"]

    canonical -->|fork| temporary
    temporary --> source_pr
    source_pr -->|merged| verify
    pairing --> verify

    subgraph publish["Maintainer: --publish --auto-merge"]
        prepare["Verify source materialization<br/>commit lock-only change with provenance and DCO"]
        advance["Fast-forward canonical branch<br/>to the reviewed source SHA"]
        followup["Create or reuse lock-only TRT-LLM PR<br/>same source SHA, patch, digests and files"]
        merge_request["Post /bot skip for no code change<br/>enable squash auto-merge with original message"]
        prepare --> advance --> followup --> merge_request
    end

    verify -->|publish requested| prepare
    merge_request -->|required checks and reviews| merged["Follow-up merged<br/>provenance preserved in squash commit"]
    merged --> rebase["Other developers rebase their temporary branches<br/>onto the promoted canonical revision"]
    rebase --> temporary
    merged -.-> refresh["Periodic refresh: manual / future tooling<br/>new upstream main plus outstanding changes<br/>new dated canonical branch"]
    refresh -.-> refresh_pr["TRT-LLM refresh PR merged<br/>all developers rebase to the dated branch"]
    refresh_pr -.-> canonical
```

Solid arrows show the development/promotion cycle; dotted arrows show periodic
refresh, which this command does not implement. A paired upstream PR can merge
before or after the TRT-LLM source-update PR. Recording its URL and observed
commits preserves provenance; it is not permission to drop a change merely
because the upstream PR later merges. Unpaired changes explicitly retain their
reason and `refresh_policy: retain`.

After a promotion follow-up merges, rebase both the pending TRT-LLM PR onto
latest main and its temporary FlashInfer branch onto the promoted canonical
revision before refreshing its vendor pin. For a periodic refresh, the
maintainer identifies already-integrated changes, carries forward outstanding
changes on newer FlashInfer main, and publishes a dated canonical branch and a
separate TRT-LLM refresh PR. That refresh may change vendor code and needs its
own validation; the promotion's no-code-change CI request does not apply.

If validation fails or another maintainer advances the relevant pins, stop and
inspect the reported state. Retrying an interrupted promotion reuses verified
worktrees, commits, PRs, and CI comments; it does not overwrite unrelated work.

### Plan and publish a promotion

Start with a dry-run (GitHub reads only):

```bash
python scripts/maintain_prims_ts.py promote \
  --trtllm-pr "$MERGED_TRTLLM_PR" \
  --canonical-repo yuxianq/flashinfer \
  --upstream-pr https://github.com/flashinfer-ai/flashinfer/pull/4829
```

Normally an upstream PR is required and may be open, draft, or merged; it must
target FlashInfer `main`. Always specify the maintainer's canonical FlashInfer
fork with `--canonical-repo OWNER/REPO`; the tool never infers the destination
fork from a previous lock that might still point to a developer's fork.
`--canonical-branch` defaults to `trtllm-prims-ts-dev`; specify it explicitly
for a dated `trtllm-prims-ts-dev-YYYYMMDD` branch. Both must match the lock in
the parent of the TRT-LLM merge commit. The tool derives the old canonical
commit from that parent and the reviewed source SHA from the merge commit's
lock. It never promotes the temporary branch's current tip. Source changes
must form a linear, fast-forward range. Finish the preceding promotion before
merging another vendor update. A developer fork may use the same branch name
as the canonical fork: the repository URL is also part of its identity.

For an existing change with no paired upstream PR, explicitly use
`--unpaired-reason` instead of `--upstream-pr`:

```bash
python scripts/maintain_prims_ts.py promote \
  --trtllm-pr 18808 \
  --canonical-repo yuxianq/flashinfer \
  --unpaired-reason "No paired FlashInfer PR for the CUTLASS DSL 4.8 API migration."
```

This records the source commits with `upstream_pr: null`, the reason, and
`refresh_policy: retain` in the commit message. A future refresh must retain
these changes until an upstream pairing or equivalence is explicitly
established; missing PR metadata must never be interpreted as permission to
drop them. This option cannot be combined with PR mappings. It does not change
the fast-forward, immutable-pin, or lock-only requirements.

One upstream PR assigns the whole newly promoted range to that PR. For multiple
PRs, repeat `--upstream-pr` and provide `--map-upstream COMMIT=PR` for each source
commit (full SHA or unambiguous prefix of at least seven characters). A terminal
invocation prompts for missing assignments; noninteractive use fails instead
of guessing. This mapping is a maintainer assertion of provenance, not proof
that a later upstream revision preserves the change's behavior.

To publish the promotion and enable squash auto-merge for its follow-up PR:

```bash
python scripts/maintain_prims_ts.py promote \
  --trtllm-pr "$MERGED_TRTLLM_PR" \
  --canonical-repo yuxianq/flashinfer \
  --upstream-pr https://github.com/flashinfer-ai/flashinfer/pull/4829 \
  --flashinfer-repo /path/to/flashinfer \
  --publish --auto-merge --wait
```

`--flashinfer-repo` is optional. If supplied, it must contain the previous and
reviewed immutable source commits; neither its checked-out branch nor dirty
files are used. Otherwise the tool fetches the reviewed source into a temporary
repository. `--repo` selects the local TRT-LLM repository. The personal TRT-LLM
fork defaults to its `fork` remote; use `--fork OWNER/REPO` to override it.
`GH_CONFIG_DIR` is respected, defaulting to `~/.config/gh`.

Publishing performs the following checks and actions:

1. Verify that current TRT-LLM main still contains the reviewed vendor entry and
   the canonical branch has not been superseded.
2. Create a separate follow-up worktree, verify source materialization, and use
   `vendor_sources.py pin` to change only the PrimTS lock URL/branch. The commit,
   compatibility patch, digest, selected source, and destination stay unchanged.
3. Make a signed-off commit with the maintainer's Git identity. Its versioned
   `PRIMTS PROMOTION V1` JSON block records both source SHAs, upstream baseline,
   originating TRT-LLM PR/merge, and upstream PR/commit mappings and snapshots.
4. Fast-forward the canonical FlashInfer branch with a non-force push, push the
   lock-only commit to the TRT-LLM personal fork, and create the follow-up PR.
5. Reverify the open PR's lock-only commit and post
   `/bot skip --comment "skip CI since no code change"` to request the bot's
   no-code-change CI path. PR creation can also start automatic GitHub checks;
   this command does not disable those checks or waive required owner reviews.
6. With `--auto-merge`, supply the signed-off commit's title/body as the final squash
   message and verify GitHub retained them. Required checks and reviews are not
   bypassed. If already eligible, the PR may merge immediately. Merge queues are
   currently rejected because this workflow requires a custom squash message.

The generated PR checklist contains only assertions the tool has verified, not
an unchecked placeholder for future approvals or checks. Those remain enforced
by GitHub. The tool does not delete branches or rewrite history. Without
`--auto-merge`, the PR description includes the required squash message for
manual merging. Preserve that complete message: it is the durable provenance
record, with no extra tracking file. Initial history predating these records
will still need classification when a future refresh tool is introduced.

`--wait` verifies the final merged commit, polling for up to `--timeout` seconds
(default 3600). Without it, the tool verifies the saved auto-merge request and
returns; there is no background monitor. If GitHub merges immediately, the tool
verifies and prints the final squash SHA. Do not change the follow-up branch
after requesting the CI skip or enabling auto-merge. A head check at enablement
does not permanently freeze the branch.

Publication spans two repositories and is not atomic. Completed pushes and PRs
are preserved if a later step fails. Repeating the same invocation recognizes
the deterministic follow-up branch/PR, verifies its lock-only change and
signed-off metadata, and resumes without creating duplicate PRs. Before posting
the CI command, retries search all PR comments for the identical command from
the authenticated user, including when a previous posting request timed out
after succeeding. A posting failure stops before enabling auto-merge. Serialize
maintainer invocations; comment deduplication is not a cross-process lock.

Interrupted worktree preparation can resume if the branch and repository match,
with either an unchanged base, the expected lock-only edit (staged or unstaged),
or a verified promotion commit. Unexpected tracked, staged, or untracked edits
are preserved and require inspection instead of automatic overwrite. Existing
upstream snapshots are preserved even if their PR heads have since advanced;
different mappings on an unfinished promotion are rejected. After a promotion
merges, reruns verify its historical squash commit and recorded provenance and
return without remote writes, even if current main or canonical branches have
advanced or the original temporary fork disappeared. The completed record is
authoritative; rerunning does not revise its provenance.

`--worktree PATH` selects the follow-up worktree; by default it is a sibling of
`--repo` named `prims-ts-promote-<PR>-<source SHA prefix>`. Git checkout skips LFS
asset downloads for this lock-only worktree without changing repository-wide
configuration. Fetch, checkout, commit, and push progress is streamed.

Hermetic CPU-only validation, without loading TRT-LLM's GPU test configuration:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest --noconftest -c /dev/null \
  -p no:cacheprovider -o 'markers=cpu_only: CPU-only test' \
  tests/unittest/others/test_maintain_prims_ts.py
```

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
`tensorrt_llm/_torch/attention/backends/prims_ts`. The `**/*.py` selection
deliberately omits upstream README files. Its persistent patch contains only
TensorRT-LLM integration and compatibility adaptations; all other selected
files remain exact upstream copies.

Use the normal commands with `flashinfer-prims-ts`, for example:

```bash
python scripts/vendor_sources.py status flashinfer-prims-ts
python scripts/vendor_sources.py check flashinfer-prims-ts
```
