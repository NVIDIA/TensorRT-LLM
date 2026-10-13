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

# Kimi K3 on Rubin (SM107)

When compiling TensorRT-LLM for Kimi K3 on Rubin, add
`--nvrtc_dynamic_linking` and `--no-venv` to the build command:

```bash
python3 scripts/build_wheel.py \
  --cuda_architectures "107-real" \
  --nvrtc_dynamic_linking \
  --no-venv
```

- `--nvrtc_dynamic_linking` links against the dynamic NVRTC libraries used by
  the Rubin CUDA environment.
- `--no-venv` builds with the current container Python environment instead of
  creating a virtual environment.

Keep any other build options required by the target environment and append
these two flags to them.
