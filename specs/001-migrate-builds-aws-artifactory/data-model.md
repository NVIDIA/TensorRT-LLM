# Data Model: Migrate Build Artifacts to AWS Artifactory

**Feature**: [spec.md](spec.md) | **Research**: [research.md](research.md)

This feature stores no new data. The entities below describe the artifacts, locations, and
identities that the pipelines handle, so that tasks and tests use the same names.

## Entities

### Artifact location

| Field | Values |
|---|---|
| `id` | `aws` (new) or `urm` (old) |
| `base_url` | `aws`: `https://artifactory.nvidia.com/artifactory/sw-tensorrt-llm-generic-local` · `urm`: `https://urm.nvidia.com/artifactory/sw-tensorrt-generic` |
| `repo_kind` | `aws`: local repository · `urm`: virtual repository backed by `sw-tensorrt-generic-local` |
| `read_access` | `aws`: authenticated only · `urm`: anonymous, as today |
| `write_access` | `aws`: publisher identity · `urm`: `urm-artifactory-creds`, as today |

Rule: an artifact URL is `base_url + "/" + layout_path` (see Build artifact set).

### In-scope artifact

| Kind | Name pattern | Location in the build directory | Producer | Metadata |
|---|---|---|---|---|
| Build record | `build_info.txt` | root | `L0_MergeRequest`, through the shared helper (FR-025) | none |
| Packaged build | `TensorRT-LLM.tar.gz`, `single-device-TensorRT-LLM.tar.gz`, `llvm-TensorRT-LLM.tar.gz`, `TensorRT-LLM-GH200.tar.gz`, `llvm-TensorRT-LLM-GH200.tar.gz`, `bolted-<tar>`, `unbolted-<tar>`, `tensorrt-llm[-sbsa]-release-src-<commit>.tar.gz` | root | `Build` (and `BoltProfileGen` for BOLTed canonicals) | `bolted-<tar>` carries BOLT properties such as `bolt.ref` |
| Waive list | `waive_list/waives*.txt` | `waive_list/` | `L0_MergeRequest`, through the shared helper (FR-025) | none |

Validation:

- Nothing outside this table moves in this ticket (FR-001).
- Metadata attached today is attached on `aws` too (FR-004).
- The source release package's producer is still to be identified (research R13). Its only reader
  in this repository is unreachable and stays as it is, on the reference allowlist.
- The packaged-build list is pending the owner's confirmation (FR-001, research R15).

### Build artifact set

| Field | Description |
|---|---|
| `pipeline` | Jenkins `JOB_NAME`, for example `LLM/main/L0_MergeRequest_PR` |
| `build_number` | Jenkins `BUILD_NUMBER` of the run that owns the directory |
| `legacy_path` | `ARTIFACT_PATH` / `UPLOAD_PATH` as today: `sw-tensorrt-generic/llm-artifacts/<pipeline>/<build_number>` |
| `layout_path` | `legacy_path` with a leading `sw-tensorrt-generic/` or `sw-tensorrt-generic-local/` removed: `llm-artifacts/<pipeline>/<build_number>` |
| `published_to` | `aws` or `urm`: where this run published its in-scope artifacts |

Validation:

- `layout_path` is identical on both locations (FR-003).
- Child jobs receive `legacy_path` unchanged (research R3).

### Access identity

| Field | Publisher | Reader |
|---|---|---|
| `jenkins_credential_id` | `aws-artifactory-credentials` (or a dedicated identity, research R5) | `trtllm-artifactory-credentials` |
| `required_permissions` on `aws` | exactly read, deploy, annotate (properties), and delete/overwrite (FR-010); together they cover create, overwrite, copy within the repository, and properties | read |
| `forbidden_permissions` on `aws` | administrative permissions (FR-010) | deploy, delete/overwrite (FR-011) |
| `delivered_to` | Build pods, `L0_MergeRequest` agent (through the shared helper's `aws` destination), `BoltProfileGen` cluster upload | Kubernetes test pods, Docker-build pods, the image build's `wheel` stage (BuildKit secret), Slurm nodes, every runtime (a per-job netrc in the job workspace), the BoltProfileGen cluster frontend (netrc fed over stdin), CBTS step |
| `token_expiry`, `renewal_owner` | recorded at onboarding (FR-013) | recorded at onboarding (FR-013) |

Validation:

- A secret never appears in a command line, URL, log, or file that outlives the job (FR-012).
- An expired or rejected token stops the step with an error naming the credential ID (FR-013).

### Artifact reference

One row per place in the repository that names the old location. Research R1 is the seed list.

| Field | Values |
|---|---|
| `file`, `line` | Repository location |
| `role` | `publish`, `read`, `link`, `path-derivation`, `doc` |
| `scope` | `in` (one of the three artifact kinds) or `out` |
| `disposition` | `repoint` (publish and links), `fallback` (reads that fall back, and the rollback target, kept under FR-019), `keep` (out of scope), `allowlist` (non-functional, with a reason) |
| `status` | see state transitions |

### Location switch

| Field | Value |
|---|---|
| `name` | `LLM_ARTIFACTS_PRIMARY` (Jenkins global environment variable) |
| `values` | `aws` or `urm`; unset means `aws` |
| `owner` | Jenkins administrators of the TRT-LLM CI (FR-022) |
| `scope` | Set on every Jenkins controller that runs TRT-LLM pipelines. A job started by a parent pipeline uses the parent's value, passed in `globalVars` as `llm_artifacts_primary` |
| `effect` | Publish target for in-scope artifacts, including the shared helper's destination for `build_info.txt` and `waive_list/`, and first location read; reads fall back to the other location on 404 |

### Read result

| Field | Values |
|---|---|
| `served_by` | `aws` or `urm` |
| `sha1` | The SHA-1 that the serving location reports; a downloaded file must match it (FR-014) |
| `outcome` | `ok`, `not_found_anywhere`, `primary_rejected` (401/403 from Artifactory itself), `primary_failed`, `secondary_failed` |

`fetch_build_artifact.sh` produces it for every reader (FR-005), and its exit codes map to the
outcomes: 0 `ok`, 2 `not_found_anywhere`, 3 `primary_rejected`, 4 `primary_failed` or
`secondary_failed`. BoltProfileGen takes `served_by` and `sha1` from the helper's `--exists`, and
its own parallel transfer can still fail after that, which counts as a failure on the serving
location.

Validation: the location that served each read is logged (FR-014). `primary_rejected` never triggers
a fallback. A 403 from the CDN is retried once without the redirect; if it persists, the outcome is
`primary_failed` or `secondary_failed`, never `primary_rejected`. A checksum mismatch is a failed
transfer, never a fallback (research R18). Steps that are best effort today (the merged waive-list
download and coverage-based test selection) log a `primary_rejected` outcome with the credential
message and carry on (FR-013, research R4).

## State transitions

### Branch migration state

```text
Unmigrated ──(this change merges into the branch)──▶ Migrated (primary aws, fallback urm)
Migrated ──(admin sets LLM_ARTIFACTS_PRIMARY=urm)──▶ RolledBack (primary urm, fallback aws)
RolledBack ──(admin unsets or sets aws)──▶ Migrated
Migrated ──(old repository retired; follow-up removes the fallback)──▶ Retired
```

- `Unmigrated` branches publish to and read from `urm` only, and must not notice that other branches
  migrated (FR-018).
- Which release branches leave `Unmigrated` depends on the pending FR-018 decision.
- Leaving `Unmigrated` requires the shared helper's opt-in on the library's `main` (FR-025). While
  the switch is under test, it may load the library from the library merge request's branch; it
  loads `main` again before it merges (research R17).

### Artifact reference lifecycle

```text
Inventoried ──▶ Repointed | FallbackEnabled | Allowlisted | Kept ──▶ Verified
```

`Verified` means a quickstart scenario exercised the reference, or a search confirmed it (SC-007).

### Identity readiness

```text
Requested ──▶ Onboarded (membership, token) ──▶ Permitted (permission target granted)
          ──▶ Verified (access check passes in every environment) ──▶ RenewalDue ──▶ Verified
```

`Permitted` means the publisher holds exactly read, deploy, annotate, and delete/overwrite, and the
reader holds read only (FR-010, FR-011). Cutover (merge of the switch PR) requires both identities
to be `Verified`.
