<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

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

# Wave 2

All plans in this folder can be implemented in parallel after Wave 1 contracts
are available. They may use local mocks while Wave 1 is still landing, but final
versions should consume the Wave 1 classifier, E8M0 helpers, source attention
op, and cache metadata contracts.

Plans:

- `03_finegrained_fp8_linear_path.md`
- `04_packed_mxfp4_expert_loader.md`
- `09_attention_kernel_microfeatures.md`
