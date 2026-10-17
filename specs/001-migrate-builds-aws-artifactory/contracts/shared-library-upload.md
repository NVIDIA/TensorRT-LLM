# Contract: Shared Library Upload Destination

The interface between this repository's pipelines and `trtllm_utils.uploadArtifacts` in
`trtllm-jenkins-shared-lib` (spec FR-025). Every branch loads that library at `@main`, so this
contract is what keeps branches that have not adopted the change unaffected (FR-018).

## Interface

```text
trtllm_utils.uploadArtifacts(<pattern>, <target>[, <destination>])
```

| Argument | Meaning |
|---|---|
| `<pattern>` | Local file or glob, as today |
| `<target>` | Repository-relative target path, as today. For `aws`, the caller passes `sw-tensorrt-llm-generic-local/<layout path>` (see [artifact-locations.md](artifact-locations.md#path-mapping)). |
| `<destination>` | Optional location id: `urm` (the default; `null` also means `urm`) or `aws`. It names a location, not the spec's "destination" path. |

Today the helper is `uploadArtifacts(patternAbsPath, target)` in `vars/trtllm_utils.groovy`, always
uploading to urm. The merge request branch `feature/trtllm-17014-aws-upload-destination` adds
`destination` as a trailing parameter with default `'urm'`; any other value fails through `error()`
before a server is defined (research R2).

## Behavior

| Destination | Server | Credential | Output |
|---|---|---|---|
| absent or `urm` | Today's server | Today's credential | Unchanged |
| `aws` | `https://artifactory.nvidia.com/artifactory`, defined under its own server id so it never replaces the `Artifactory` server other code defines | `aws-artifactory-credentials`, the publishing identity (FR-010) | Any link it prints names `artifactory.nvidia.com` |

- Every existing call behaves exactly as before. No existing caller passes a destination.
- No secret appears on a command line, in a URL, or in the log (FR-012).
- The plugin reports a 401 or 403 only in its log, so the helper cannot detect one. When an `aws`
  upload gives up after `llmRetry(2, ...)`, the helper logs one note that names
  `aws-artifactory-credentials` and says what to check if the log shows 401 or 403, then rethrows
  (FR-013).
- Failure handling at the call sites does not change. Today both in-scope uploads are best effort:
  a failure is logged, and the stage continues. Spec FR-013 keeps them best effort.

## Callers

| Caller | Upload | Destination passed |
|---|---|---|
| `jenkins/L0_MergeRequest.groovy:616` | `waives*.txt` into `waive_list/` | The primary (`LLM_ARTIFACTS_PRIMARY`, unset means `aws`) |
| `jenkins/L0_MergeRequest.groovy:668` | `build_info.txt` | The primary |
| The other 14 call sites | Out of scope: test results, wheels uploaded on their own, coverage, the scan log, the image-tag map | None, so `urm`, until TRTLLMINF-472 or TRTLLMINF-295 moves them |

The log links printed after the two in-scope calls (`:617`, `:669`) name the location the run
published to (see [artifact-locations.md](artifact-locations.md#links-shown-to-people)).

## Rollout

1. The library merge request lands on the library's `main`, and no caller changes. Every upload
   still goes to urm (quickstart prerequisite 5).
2. The switch pull request in this repository starts passing the destination. It cannot merge
   before step 1, because the helper would not accept the destination.
