# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
### :title KV Cache Compression
### :order 8
### :section Customization
r"""Enable NVFP4 cold-page compression for a Qwen3.5 model on one GPU.

Attention KV uses the model's normal runtime dtype on the GPU and is encoded
as NVFP4 only when KVCacheManagerV2 migrates a Page to Host memory.

Run the example with:

```bash
python llm_kv_cache_compression.py
```

A short request can remain entirely on the GPU. Use a workload with enough KV
pressure to trigger offload when validating the compression path.
"""

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import ColdPageQuantizationCompressionConfig, KvCacheConfig


def main() -> None:
    with LLM(
        model="Qwen/Qwen3.5-4B",
        backend="pytorch",
        trust_remote_code=True,
        max_seq_len=4096,
        max_batch_size=4,
        kv_cache_config=KvCacheConfig(
            use_kv_cache_manager_v2=True,
            dtype="auto",
            host_cache_size=8 << 30,
        ),
        kv_cache_compression_config=ColdPageQuantizationCompressionConfig(
            quant="nvfp4",
        ),
    ) as llm:
        outputs = llm.generate(
            ["Explain why prefix caching helps agentic workloads."],
            SamplingParams(max_tokens=128, temperature=0.0),
        )
        print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()
