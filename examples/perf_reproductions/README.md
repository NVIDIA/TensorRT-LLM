<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Reproducing Published TensorRT-LLM Benchmark Results

Each entry in this directory explains how to reproduce one published benchmark
result for TensorRT-LLM: which code and container produced it, the exact serving
configuration, how to launch it, and how to check that your run matches.

Some results are produced from a TensorRT-LLM feature branch before the code
reaches `main`. The entry then states exactly which commit and container were
used, and tracks which parts have reached `main`. The fastest path is always the
pinned NVIDIA container: you do not need to build TensorRT-LLM yourself.

For general deployment guidance, see the
[deployment guides](../../docs/source/deployment-guide/) and the recommended
configurations in [`examples/configs`](../configs/).

## Entries

| Benchmark | Model | Hardware | Scenario | Code | Entry |
|---|---|---|---|---|---|
| InferenceX | `nvidia/Qwen3.5-397B-A17B-NVFP4-V2` | GB300 NVL72 | AgentX agentic coding, disaggregated | branch | [agentx-disagg](inferencex/nvidia/Qwen3.5-397B-A17B-NVFP4-V2/GB300/agentx-disagg/README.md) |

The machine-readable registry is [`index.yaml`](index.yaml).

## Layout

```text
examples/perf_reproductions/
  README.md, index.yaml           # this page and the registry
  _common/                        # checks shared by all entries
  <benchmark>/
    _tools/                       # helpers specific to one benchmark (e.g. recipe export)
    <org>/<model>/<hardware>/<scenario>/
      README.md                   # how to reproduce this result
      source.yaml                 # pinned code, container and external references
      configs/                    # serving configuration per point and role
```

Model and hardware directory names follow
[`examples/configs/database`](../configs/database/): the Hugging Face model ID
and the GPU name (for example `GB300`).

## `source.yaml`

| Field | Meaning |
|---|---|
| `entry_id` | The entry's path below this directory. |
| `status` | `branch` (code only on a feature branch), `partial` (some features on `main`), `mainline` (reproducible from a `main` release), or `superseded` (replaced by a newer entry; kept for reference). |
| `code.commit` | Full 40-character TensorRT-LLM commit that the container was built from. This is the reference for the code. |
| `code.branch` | Branch that contains the commit, for information. It must be a branch that the repository protects against deletion and force-push. |
| `code.tag` | Optional. Required only when the commit is not on such a protected branch. |
| `image` | Container name, `tensorrt_llm` version, and digests. The digest, not the tag, identifies the image. |
| `dependencies` | Other pinned packages, such as the Dynamo wheels. |
| `external` | The published submission and the exact external recipe commit, when the benchmark has one. |
| `verify` | Log patterns that confirm the expected configuration took effect. |
| `upstream` | Per-feature status on `main`: `branch-only`, `in-review`, `equivalent-on-main`, or `mainline`, with a PR or note. |

## Adding or Updating an Entry

Before adding an entry:

- The code on the branch has passed the open-source release review.
- Behavior switches are expressed as LLM API configuration or a small number of
  profile variables, not as long lists of individual environment variables.
- The commit is reachable from a protected branch (or a tag), and the container
  is published; record its digests.
- Someone has followed the entry's README end to end in a clean environment.

When features reach `main`, update `upstream` and `status`. When a result is
reproduced from a `main` release, set `status: mainline` and point `code` and
`image` at that release. Do not delete entries; mark replaced ones `superseded`
and link the new entry.

## Validation

```bash
python3 examples/perf_reproductions/_common/validate.py              # offline checks
python3 examples/perf_reproductions/_common/validate.py --check-git  # also checks commit reachability
```

Inside a running container, `_common/check_image.py <entry>/source.yaml`
confirms the TensorRT-LLM and dependency versions, and
`_common/check_worker_logs.py` checks worker logs against the entry's
`verify.log_patterns`.
