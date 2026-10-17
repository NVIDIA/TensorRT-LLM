# Contract: Build Artifact Locations

Applies to every pipeline, script, and tool that publishes or reads an in-scope artifact
(`build_info.txt`, packaged builds, `waive_list/`). Terms are defined in
[data-model.md](../data-model.md).

## Locations

| id | Base URL | Read | Write |
|---|---|---|---|
| `aws` | `https://artifactory.nvidia.com/artifactory/sw-tensorrt-llm-generic-local` | Authenticated (reader or publisher identity) | Publisher identity (read, deploy, annotate, delete/overwrite) |
| `urm` | `https://urm.nvidia.com/artifactory/sw-tensorrt-generic` | Anonymous (unchanged) | `urm-artifactory-creds` (unchanged) |

## Path mapping

Input is a legacy path as carried in `ARTIFACT_PATH`, `UPLOAD_PATH`, `artifactPath`, `uploadPath`,
or `reuseArtifactPath`, optionally followed by a file name.

1. If the path starts with `sw-tensorrt-generic-local/` or `sw-tensorrt-generic/`, remove that
   prefix. Otherwise it must already start with `llm-artifacts/`; anything else is a usage error.
   So is an empty, `.`, or `..` segment, and whitespace, `;`, `?`, or `#`, which would change the
   meaning of the request URL. `fetch_build_artifact.sh` and `check_artifactory_access.sh` apply
   the same rule.
2. The result is the layout path, for example
   `llm-artifacts/LLM/main/L0_MergeRequest_PR/1234/TensorRT-LLM.tar.gz`.
3. The URL is `<base URL of the location>/<layout path>`.

Examples:

| Input | `aws` URL | `urm` URL |
|---|---|---|
| `sw-tensorrt-generic/llm-artifacts/LLM/main/L0_PostMerge/88/build_info.txt` | `https://artifactory.nvidia.com/artifactory/sw-tensorrt-llm-generic-local/llm-artifacts/LLM/main/L0_PostMerge/88/build_info.txt` | `https://urm.nvidia.com/artifactory/sw-tensorrt-generic/llm-artifacts/LLM/main/L0_PostMerge/88/build_info.txt` |
| `sw-tensorrt-generic-local/llm-artifacts/X/1/waive_list/waives.txt` | `.../sw-tensorrt-llm-generic-local/llm-artifacts/X/1/waive_list/waives.txt` | `.../sw-tensorrt-generic/llm-artifacts/X/1/waive_list/waives.txt` |

Property lookups use the storage API on the same location:
`<scheme://host>/artifactory/api/storage/<repo>/<layout path>?properties`.

## Primary and secondary

- Primary is the value of `LLM_ARTIFACTS_PRIMARY` (`aws` or `urm`). Unset means `aws`.
- A job started by a parent pipeline uses the parent's value instead of its own controller's.
  `L0_MergeRequest` adds it to the `globalVars` of every job it starts, as `llm_artifacts_primary`,
  and `BuildDockerImage` does the same for its image sanity test. Child jobs run on other Jenkins
  controllers, whose own setting can differ.
- Secondary is the other location.
- Pipelines resolve the value once per step and pass it to every script they run, with
  `ARTIFACTORY_NETRC` and `ARTIFACTORY_CREDENTIAL_ID`. A script on another host, such as
  BoltProfileGen's cluster script or a Slurm job's launch script, gets all three exported
  explicitly, because Jenkins environment variables do not reach it. Without the value, a script
  reads `aws` first even while CI is rolled back.

## Publish rule

- In-scope artifacts are published to the primary only. There is no dual publishing.
- `build_info.txt` and `waive_list/` go through the shared helper, with its destination set to the
  primary and its target set to `<repository of the primary>/<layout path>`
  ([shared-library-upload.md](shared-library-upload.md)).
- The layout path and file name are the same as today.
- Metadata attached today (for example BOLT properties on `bolted-<tar>`) is attached in the same
  upload request.
- A 401 or 403 on publish is reported with the credential ID and says that its token needs renewal
  (spec FR-013). The shared helper cannot see the HTTP status, so it names the credential on any
  failed `aws` upload ([shared-library-upload.md](shared-library-upload.md)). Uploads that are best
  effort today stay best effort and log that message.
- Out-of-scope artifacts keep their current publish path to `urm`.

## Read rule

For each in-scope read, including property lookups:

| Primary response | Action |
|---|---|
| 200 | Use it. Log `served-by=<primary>`. |
| 404 | Request the secondary. 200: use it and log `served-by=<secondary>`. 404: report not found. Other: report failure. |
| 401 or 403 | Fail with an authentication error that names the credential ID. Do not request the secondary. |
| Other, including connection errors and 5xx | Retry under the caller's retry policy, then fail. Do not request the secondary. |

Reads of `urm` stay anonymous. Reads of `aws` always send credentials.

Every reader applies this rule by running `jenkins/scripts/fetch_build_artifact.sh`
([helper-scripts.md](helper-scripts.md)), on CI agents, in containers, inside the image build, and
on the Slurm nodes (spec FR-005). BoltProfileGen's input download keeps its own parallel transfer,
and takes the file's location and SHA-1 from the helper. The reuse check
([below](#reuse-of-an-earlier-build)) is not a read.

Steps that are best effort today, the merged waive-list download and coverage-based test selection,
keep that behavior after an authentication error: they log the credential message and carry on,
without requesting the secondary (spec FR-013).

Two safeguards apply to every download (research R18):

- A 403 that arrives after a redirect to another host (the CDN, `*.cloudfront.net` in production),
  rather than from Artifactory, is not a credential rejection. Retry once with `X-JFrog-Download-Redirect-To: None`; a further failure is "Other".
- The downloaded file must match the SHA-1 that the serving location reports. A mismatch is "Other":
  retried, never a fallback.

## Links shown to people

| Artifact | Link target |
|---|---|
| `build_info.txt`, `waive_list/waives.txt`, packaged builds | URL on the location the run published to |
| Test results, coverage, reports, scan logs, wheels uploaded on their own | Unchanged (`urm`) |
| A directory holding both kinds (for example the build root) | One link per location |

## Reuse of an earlier build

| Where the earlier build's packaged builds exist | Action |
|---|---|
| Primary | Server-side copy within the primary |
| Secondary only | Log `reuse source is on <secondary>; rebuilding` and rebuild |
| Neither | Existing behavior: log and rebuild |

## Compatibility guarantees

- Branches without this change keep publishing to and reading from `urm` exactly as today
  (spec FR-018). The shared library changes only by an opt-in destination whose default keeps
  today's behavior (FR-025).
- Older builds that stay on `urm` remain readable through the fallback until `urm` is retired.
