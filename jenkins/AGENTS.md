<!--
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Jenkins instructions

Scope: this directory and its descendants. Supplements the repository-root
`AGENTS.md`.

## Stage maintenance

- Edit `jenkins/config/maintenance_stages.txt`. Blank lines and `#` comments
  are ignored. The first active line must use the exact prefix `Reason:` with
  a non-empty explanation on one physical line. This reason applies to all
  subsequent entries; each entry is one exact runtime stage name or `*` wildcard
  pattern, matched case-sensitively against the whole name. For example:

  ```text
  Reason: The affected H100 resource is temporarily unavailable.
  H100_PCIe-PyTorch-1
  ```

- The following DGX B200 names and patterns are illustrative examples, not
  entries to add by default. `DGX_B200-*` matches all DGX B200 stages, including
  single-GPU, multi-GPU, and post-merge variants. `DGX_B200-*_GPUs-*` matches only
  multi-GPU variants,
  such as `DGX_B200-2_GPUs-PyTorch-1` and `DGX_B200-4_GPUs-PyTorch-Post-Merge-1`.
  `DGX_B200-PyTorch-2` matches exactly that stage; append `*` only if runtime
  suffixes such as `-cbts` should also match. Only `*` is a wildcard, not `?`,
  character ranges, regular expressions, or exclusion rules.
- During severe resource scarcity, temporarily list selected stages to reduce
  demand while retaining representative coverage. For example, exact entries for
  `DGX_B200-PyTorch-2` through `DGX_B200-PyTorch-9` leave
  `DGX_B200-PyTorch-1` eligible to run. List each name on its own line; do not
  write a range. Ensure no other pattern matches stages you intend to retain;
  `DGX_B200-*` would disable them too. Explain the capacity constraint in the
  shared reason and remove these entries when capacity recovers.
- Matching stages show `Skip - Maintenance` and the reason, without launching
  a test agent. The nested stage is red to show missing coverage, but the skip
  alone does not fail the overall build. Previously successful stages may show
  `Skip - Reused` instead. Explicit stage selection does not override maintenance.
- Preparation merges the PR's config additions/deletions into the target branch's
  latest snapshot. After merge, older PRs with maintenance support pick it up
  without rebasing; builds that already took their snapshot are unaffected.
- To restore execution, remove the relevant entries in a recovery PR; dates in
  the reason do not expire them. Keep the file and header when no entries remain.
  Before merging, run `/bot run --stage-list "<affected-stage-names-or-patterns>"`
  on the recovery PR and verify every affected stage actually executes. The PR's
  deletions take effect for its own validation before merge. Regressions may have
  entered while coverage was disabled: fix failures or add tracked test
  waivers through the normal waiver process, then rerun and confirm the remaining
  tests pass on the latest PR head before merging the maintenance removal.

## Path-valued pytest options

- For new or modified long pytest options that take a single file or directory
  path, use `--option=value` when supported. Keep the option and value in one
  argument, for example:

  ```text
  --periodic-junit-xmlpath=${outputPath}/results.xml
  --durations-path=${durationsPath}
  ```

- Some options are registered only after pytest has started discovering its
  configuration and initial `conftest.py` files. An existing path passed as a
  separate token can influence that discovery. `--rootdir` alone does not force
  the intended configuration or conftest to load.
- Keep actual test-selection paths as positional arguments. The equals form does
  not replace shell quoting: preserve the existing escaping when a value crosses
  Groovy and shell boundaries.
- Check all affected command producers: collection, ordinary tests, sbatch,
  agents, isolation and reruns. Also check consumers that parse or rewrite the
  command, including the PerfSanity scripts.
- Validate the emitted arguments with paths both absent and already present,
  including a file outside the test tree. Check that the intended configuration
  loads and the option value is unchanged.
- Distinguish a reproduced failure from preventive consistency work. Do not claim
  that another option has the same bug without a failing case. Avoid unrelated
  command rewrites while applying this convention.

## Slurm/Pyxis environment variables

- With normal Slurm export settings, exported job variables inherit into the
  container by default. `--container-env` is not a transmission allowlist: it
  gives the named job variables priority over values defined by the image.
  Do not add an override solely to pass an exported variable into the container.
- Export values separately; `--container-env` takes variable names, not
  `NAME=value` assignments. Put all required overrides in one comma-separated
  option, for example `--container-env=BASH_ENV,OTHER_IMAGE_OVERRIDE`. Repeating
  `--container-env` causes only the last option value to reach the compute nodes.
- The Jenkins sbatch path needs `--container-env=BASH_ENV` to override the CI
  image's Bash startup hook. Preserve that override when changing the caller or
  a downstream command generator; do not append another `--container-env`.
  Image initialization can still reset variables after inheritance, so the
  profiler wrapper applies its policy after normal initialization on each Bash
  startup.
- Check the final generated `srun` arguments across ordinary and PerfSanity
  roles, including optional prechecks. For environment changes, distinguish
  default inheritance from image-value overrides and verify the required values
  inside the container, including reuse and role changes where applicable.
- `scripts/perf/local/submit.py` still contains legacy repeated
  `--container-env` declarations. Do not copy that pattern into new code or
  assume the CI fix also covers the local launcher. Validate local behavior
  separately when changing it.
