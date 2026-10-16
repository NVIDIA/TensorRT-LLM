# Contract: Helper Scripts and Tool Interfaces

Command-line and environment contracts for the scripts that implement
[artifact-locations.md](artifact-locations.md). Pipelines, Slurm jobs, and tools depend on these
interfaces, and the unit tests in `tests/unittest/scripts/` pin them.

## Shared environment

| Variable | Meaning | Required |
|---|---|---|
| `LLM_ARTIFACTS_PRIMARY` | `aws` or `urm`; unset means `aws` | No |
| `ARTIFACTORY_NETRC` | Path to a netrc file with a `machine artifactory.nvidia.com` entry | Whenever `aws` may be contacted |
| `ARTIFACTORY_CREDENTIAL_ID` | Name of the CI credential behind `ARTIFACTORY_NETRC`, used in error messages | No |
| `LLM_ARTIFACTS_AWS_URL`, `LLM_ARTIFACTS_URM_URL` | Override the base URL of each location. Only the unit tests set them, to point at a local server. | No |

No script prints the contents of `ARTIFACTORY_NETRC`, passes credentials on a command line, or runs
the transfer under `set -x`. `fetch_build_artifact.sh` turns off xtrace that a caller passes on, for
example through an exported `SHELLOPTS`, because a trace would show a CDN's signed URL.

## Getting the helper

Every reader of an in-scope artifact runs `fetch_build_artifact.sh` (spec FR-005). Stages that read
before they have a source tree take it from the run's commit (research R4): `L0_Test` fetches it
once per run with `trtllm_utils.checkoutFile` and stashes it, and its Slurm launcher copies that
copy into each job's workspace. `BuildDockerImage` and `BoltProfileGen` fetch it where they need
it, and the image build bind-mounts it at its repository path. Coverage-based test selection runs
it from the checked-out tree.

A caller that runs the helper on another host, such as BoltProfileGen's cluster script or a Slurm
job's launch script, exports `LLM_ARTIFACTS_PRIMARY`, `ARTIFACTORY_NETRC`, and
`ARTIFACTORY_CREDENTIAL_ID` in that script, because Jenkins environment variables do not reach it.

## `jenkins/scripts/fetch_build_artifact.sh` (new)

Reads one in-scope artifact using the read rule.

```text
fetch_build_artifact.sh [--properties | --exists] <artifact-path> [<dest-file>]
```

| Argument | Meaning |
|---|---|
| `<artifact-path>` | Legacy or layout path of the file, mapped as in the locations contract |
| `<dest-file>` | Output file. Required unless `--exists` is given. Written to `<dest-file>.part`, then renamed. |
| `--properties` | Fetch the artifact's properties JSON instead of the file |
| `--exists` | Check existence only and write no file. On success, print `<url> <sha1>` on stdout: the URL of the location that holds the file and the SHA-1 that location reports. BoltProfileGen's parallel download, the one reader that transfers the file with its own code, uses both, and the Slurm launcher exports the URL for older copies of `slurm_install.sh`. |

| Exit code | Name | Meaning |
|---|---|---|
| 0 |  | Success. stderr gets `fetch_build_artifact: served-by=<aws\|urm> path=<layout path>`. |
| 2 | `FETCH_ABSENT` | Not found in either location |
| 3 | `FETCH_REJECTED` | Primary rejected credentials (401/403 from Artifactory itself; a 403 from the CDN is not a credential rejection). The secondary was not contacted. The message names `ARTIFACTORY_CREDENTIAL_ID` and says its token needs renewal, with no internal names (spec FR-008, FR-013). When `urm` is the primary, its rejection of the anonymous request also exits 3, and the message names no credential. |
| 4 | `FETCH_TRANSFER_FAILED` | Transfer failed after built-in retries: a connection error, 5xx, a CDN rejection that persists without the redirect, a checksum mismatch, or a non-404 error from the secondary |
| 64 | `FETCH_USAGE` | Usage error, an unmappable path ([path mapping](artifact-locations.md#path-mapping)), a base-URL override that carries credentials, or `ARTIFACTORY_NETRC` missing when needed |

The helper defines these names. A reader that acts on a particular exit code defines the same
name with the same value, and checks the code by that name, never by its number. The readers are
`jenkins/L0_Test.groovy`, `jenkins/BuildDockerImage.groovy`, BoltProfileGen's cluster script,
`artifact.py`, and `get_wheel_from_package.py`; the two Python modules prefix the name with an
underscore. Each reader defines only the names it checks. `test_fetch_build_artifact.py` fails
when a reader's value differs from the helper's, or when one of those checks compares the exit
status with a bare number (spec FR-005).

Transient errors are retried a small fixed number of times inside one call. Callers keep their
existing outer retry loops (`llmExecStepWithRetry`, `retry_command`). Those retry any failure a
bounded number of times, so a rejected credential still fails within the step's normal budget
(spec FR-013). Loops that poll until a package appears (`get_wheel_from_package.py`, the image
job's wait stage) stop at once on exit code 3. The exit-3 message avoids the phrases that the
shared library treats as transient, such as `403 Forbidden` and `URL returned error: 403`
(research R4). The merged waive-list download in `L0_Test` is best effort: after any non-zero exit
it logs the helper's message and uses the default waive list, as it does today (spec FR-013).

Download safeguards (research R18): the helper reads `checksums.sha1` from
`api/storage/<repo>/<layout path>` on the location that holds the file, compares the `.part`
file's SHA-1 after the download, and renames it only on a match. Requests follow redirects; a request
to `aws` that ends in a 403 after a redirect to another host (the CDN, `*.cloudfront.net` in
production) is retried once with `X-JFrog-Download-Redirect-To: None`. `--properties` and `--exists` download no file.

## `jenkins/scripts/check_artifactory_access.sh` (new)

Verifies an identity from the environment it runs in (spec FR-020, SC-006).

```text
check_artifactory_access.sh --role publisher|reader --probe-dir <layout-dir> [--time <artifact-path>]
check_artifactory_access.sh --role publisher --cleanup --probe-dir <layout-dir>
```

| Role | Checks, in order | Expected |
|---|---|---|
| `publisher` | Create `probe.txt`; overwrite it with a property attached (`access.check=<date>`); read its properties; server-side copy to `probe-copy.txt`; list the directory | All succeed |
| `publisher` | Delete `probe-copy.txt` | Succeeds (delete/overwrite is granted, FR-010). `probe.txt` stays for the reader checks. |
| `reader` | Read `probe.txt` with credentials, and compare its SHA-1 with the storage API's `checksums.sha1` | 200, and they match |
| `reader` | Read `probe.txt`'s properties with credentials | 200, with `access.check` |
| `reader` | Read `probe.txt` without credentials | 401 (anonymous rejected) |
| `reader` | Create a file with reader credentials | 403 (no write, FR-011) |
| `publisher` with `--cleanup` | Delete `probe.txt` | Succeeds. No other check runs. |
| either, with `--time` | Download the given artifact from `aws` and from `urm` | Prints one `TIME <location> <seconds> s <MB/s> MB/s <url>` line for each (research R14) |

Output is one line per check: `PASS <check>` or `FAIL <check>: <reason>`. Exit 0 if every check
passed, 1 otherwise, and 64 on a usage error. For one probe directory, run the publisher role first, then the reader role in
each environment, then the publisher role with `--cleanup`. The reader's storage-API checks are the
requests that the read helper's checksum and property lookups make.

## `jenkins/scripts/slurm_install.sh` (changed)

| Input | Before | After |
|---|---|---|
| `llmTarArtifactPath` | n/a | Artifact path of the packaged build (`${ARTIFACT_PATH}/<tar>`, or its `bolted-` copy), passed to the helper. Required. |
| `fetch_build_artifact.sh` | n/a | `<jobUID>-fetch_build_artifact.sh` beside this script in the job workspace, found from this script's own file name, as `bash_utils.sh` is. The launcher copies it from the `L0_Test` stash, so it comes from the run's commit. |
| `LLM_ARTIFACTS_PRIMARY`, `ARTIFACTORY_NETRC`, `ARTIFACTORY_CREDENTIAL_ID` | n/a | Exported by the job script for the helper. `ARTIFACTORY_NETRC` names the per-job netrc in the job workspace, which every Slurm job gets (research R5). |
| `llmTarfile` | Full URL on urm; required | Still exported by the launcher: the URL that `--exists` resolved on the agent. This script no longer reads it. The older `slurm_install.sh` inside a package built before this change downloads it, anonymously from urm, where such a package is. |
| Download command | `wget -nv` on the URL, inside `retry_command` | `bash <helper> "$llmTarArtifactPath" <temporary file>`, then a move into place, inside the same `retry_command --timeout 1800` |

The helper applies the read rule, the CDN retry, and the checksum check on the node, and this script
adds none of its own. `retry_command` retries any failed attempt up to its limit, including exit 3,
so a rejected credential fails within the step's budget, with the helper's message naming the
credential. The helper's command line holds no secret, since the netrc reaches it by path through
the environment, and the helper never turns on xtrace, so the job log shows no secret. Other inputs,
idempotency (a job-specific temporary file, moved into place only on success), and exit behavior are
unchanged.

## `scripts/get_wheel_from_package.py` (changed)

- The command line is unchanged: `--arch`, `--timeout`, `--artifact_path`, and the BOLT options.
- Downloads go through `fetch_build_artifact.sh`, found relative to the repository root, instead of
  a urm URL.
- It runs inside the image build, in the `wheel` stage of `docker/Dockerfile.multi`. That stage's
  `RUN` bind-mounts `jenkins/scripts/fetch_build_artifact.sh` at its repository path, mounts the
  reader netrc as the BuildKit secret `artifactory_netrc`, and receives `LLM_ARTIFACTS_PRIMARY` and
  `ARTIFACTORY_CREDENTIAL_ID` as build arguments; the credential's name lets the authentication
  error name it. `docker/Makefile` passes the secret only when `ARTIFACTORY_NETRC` is set, and each
  build argument only when its variable is set. `jenkins/BuildDockerImage.groovy` sets all three.
- Its polling loop keeps retrying on exit codes 2 and 4, because the packaged build may not be
  published yet. On exit code 3 it stops immediately with an authentication error.

## `jenkins/scripts/cbts/cbts/command/coverage/selection/artifact.py` (changed)

Run as `python3 -m cbts.command coverage selection artifact`, with
`PYTHONPATH=jenkins/scripts/cbts`.

- `build_commit` reads `build_info.txt` by running `jenkins/scripts/fetch_build_artifact.sh`, found
  relative to the module in the checked-out tree, into a temporary file. The helper gets the
  command's environment (`LLM_ARTIFACTS_PRIMARY`, `ARTIFACTORY_NETRC`, `ARTIFACTORY_CREDENTIAL_ID`),
  and its stderr goes to the job log unchanged, including `served-by=` and any credential message.
- The module keeps no copy of the read rule: no location URLs, netrc parsing, or checksum code for
  `build_info.txt`.
- Exit 0: the content. Exit 2 (absent from both locations): `None`, as for a missing file today.
- Exit 3 raises `ArtifactAuthError`, and exit 4 raises `ArtifactTransferError`. Both are new
  exceptions in `artifact.py`, derived from `ArtifactError`; the module defines no error type today.
  Exit 64 or any other status raises `ArtifactError`. None of these reads the other location.
- The click command `main` catches `ArtifactError`, prints its message to stderr, and exits 1. In
  `L0_MergeRequest`, coverage-based test selection then declines, as it does after any failure
  today (spec FR-013). It declines at once, instead of moving on to an older build as a failed read
  does today, because one helper call can retry for minutes (research R4).
- The pod that runs it (`buildpack-deps:trixie-scm`) already has bash, curl, and `sha1sum`.
- Coverage archives and pins keep their current `urm` URLs (out of scope).

## BoltProfileGen input download (changed)

The one reader that moves the bytes with its own code (spec FR-005). Its parallel ranged download
moves into the helper only if a second reader needs one (research R4).

- The cluster script resolves `unbolted-<tar>`, then `<tar>`, with `fetch_build_artifact.sh --exists`,
  and passes the URL and SHA-1 to `fetch_verified` (`jenkins/BoltProfileGen.groovy:369`).
- The parallel ranged download is kept. Its single-stream fallback sends
  `X-JFrog-Download-Redirect-To: None`, so a CDN rejection of the ranged requests ends in a direct
  download (research R18).
- After the existing archive check, the file's SHA-1 must match. A mismatch fails the download.
- `curl` sends the netrc only for `aws` URLs. The llvm-bolt download from GitHub stays anonymous.

## `scripts/package_trt_llm.py` (developer tool)

The `--addr` default names a specific pre-cutover build that remains on `urm`. It stays unchanged and
goes on the reference allowlist with that reason. Whether the tool downloads from `--addr` at all is
to be confirmed during implementation. If it does, its help text states that new-location URLs need
an Artifactory access token.
