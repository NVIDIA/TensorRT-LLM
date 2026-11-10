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
