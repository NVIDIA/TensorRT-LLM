<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Contract: `fetch_build_artifact.sh --list`

Extends TRTLLM-17014's interface of `jenkins/scripts/fetch_build_artifact.sh`. Everything not named
here is unchanged: the environment variables (`LLM_ARTIFACTS_PRIMARY`, `ARTIFACTORY_NETRC`,
`ARTIFACTORY_CREDENTIAL_ID`, and the test-only base-URL overrides), the path mapping, credentials for
aws only, the xtrace rule, the retries, and the exit-code names.

```text
fetch_build_artifact.sh --list <directory-path>
```

| Argument | Meaning |
|---|---|
| `<directory-path>` | Legacy or layout path of a folder under `llm-artifacts/`, mapped as for files; a trailing `/` is allowed. `--list` takes no `<dest-file>`. |

## Behavior

1. Requests the folder from the primary's storage API, `<root>/api/storage/<repository>/<layout path>`,
   with the netrc only when the location is aws.
2. On 200, keeps the folder's direct children that are files; folders are skipped.
3. On 401 or 403 from the primary, reports the rejection with the credential message (TRTLLM-17014
   FR-013) and exits `FETCH_REJECTED` without asking the secondary, so that a broken credential
   cannot hide behind the other location.
4. Otherwise requests the same folder from the secondary.
5. Prints one line per file to stdout, `<location> <name>`, the primary's first, with the name as the
   storage API gives it, without its leading `/`. stderr gets
   `fetch_build_artifact: listed <n> on <primary>, <m> on <secondary> path=<layout path>`.

A 404 means that the location has no such folder: no entries, and not a failure. Transient failures
(no response, 5xx, a response it cannot parse) are retried like the other modes. The JSON is read
with text tools, as `checksums.sha1` is today, so the helper still needs neither Python nor `jq`.

## Exit codes

| Code | Name | Meaning in list mode |
|---|---|---|
| 0 | | Both locations answered, and at least one has the folder |
| 2 | `FETCH_ABSENT` | Both locations answered 404: no such folder |
| 3 | `FETCH_REJECTED` | The primary rejected the credential; the secondary was not asked |
| 4 | `FETCH_TRANSFER_FAILED` | A location's listing failed after retries, or the secondary rejected the request; entries from the location that answered are still printed |
| 64 | `FETCH_USAGE` | Usage error, an unmappable path, a `<dest-file>`, or `ARTIFACTORY_NETRC` missing when aws must be asked |

## Example

A run during dual publishing, whose `results-A10-PyTorch-3.tar.gz` replica failed:

```text
$ fetch_build_artifact.sh --list sw-tensorrt-generic/llm-artifacts/LLM/main/L0_MergeRequest_PR/1234/test-results
aws results-A10-PyTorch-1.tar.gz
aws results-A10-PyTorch-2-progress.tar.gz
aws doc-html-preview.tar.gz
aws rerun_report.html
urm results-A10-PyTorch-1.tar.gz
urm results-A10-PyTorch-2-progress.tar.gz
urm results-A10-PyTorch-3.tar.gz
urm doc-html-preview.tar.gz
urm rerun_report.html
```

stderr: `fetch_build_artifact: listed 4 on aws, 5 on urm path=llm-artifacts/LLM/main/L0_MergeRequest_PR/1234/test-results`.

## Callers

Collect Test Result and the coverage step in `jenkins/L0_MergeRequest.groovy`. They merge the
entries by name, since during dual publishing nearly every name appears once per location
([test-result-locations.md](test-result-locations.md), "Listing"). They act on `FETCH_ABSENT`,
`FETCH_REJECTED`, and `FETCH_TRANSFER_FAILED` by those names, and
`tests/unittest/scripts/test_fetch_build_artifact.py` checks the values against the helper's.

## Tests

`tests/unittest/scripts/test_fetch_build_artifact.py` serves folder answers from its local servers:
a folder on the primary only, on the secondary only, on both, and on neither; folders among the
children; a primary rejection (no request to the secondary); a secondary rejection and a secondary
failure (exit 4 with the primary's entries printed); retried transient errors; both primaries; the
netrc sent to aws only; and no credential in any output.
