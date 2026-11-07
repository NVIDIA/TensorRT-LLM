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
