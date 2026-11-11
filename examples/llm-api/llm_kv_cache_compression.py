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
### :order 6
### :section Customization
r"""Configure KV cache compression with TensorRT-LLM.

This example shows the available compression methods. Currently, only one KV
cache compression method can be enabled for each LLM instance.

NVFP4 cold-page quantization
----------------------------
Attention KV keeps its runtime dtype on the GPU. KVCacheManagerV2 encodes a
Page as NVFP4 only while the Page resides in Host or Disk storage.

```bash
python llm_kv_cache_compression.py \
    --compression-method nvfp4-cold-page \
    --model Qwen/Qwen3.5-4B
```

A short request can remain entirely on the GPU. Use a workload with enough KV
pressure to trigger offload when validating the cold-page codec path.

TurboQuant cold-page quantization
---------------------------------
Like NVFP4 cold pages, Attention KV keeps its runtime dtype on the GPU, and
KVCacheManagerV2 encodes a Page with TurboQuant only while the Page resides in
Host or Disk storage. TurboQuant stores 2 to 4 bits per value and runs on GPUs
with compute capability 8.0 or newer.

```bash
python llm_kv_cache_compression.py \
    --compression-method turboquant-cold-page \
    --model Qwen/Qwen3-8B \
    --turboquant-bits 3 \
    --turboquant-mode mse
```

TriAttention
------------
TriAttention periodically evicts less important decode tokens. It requires an
offline calibration file produced for the selected model.

```bash
python llm_kv_cache_compression.py \
    --compression-method triattention \
    --model Qwen/Qwen3-8B \
    --calibration-path /path/to/qwen3-8b-calibration.pt
```
"""

import argparse

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import (
    ColdPageQuantizationCompressionConfig,
    KvCacheConfig,
    TriAttentionKvCacheCompressionConfig,
)

_DEFAULT_MODELS = {
    "nvfp4-cold-page": "Qwen/Qwen3.5-4B",
    "turboquant-cold-page": "Qwen/Qwen3-8B",
    "triattention": "Qwen/Qwen3-8B",
}


def _generate(
    model: str,
    kv_cache_config: KvCacheConfig,
    compression_config: ColdPageQuantizationCompressionConfig
    | TriAttentionKvCacheCompressionConfig,
) -> None:
    with LLM(
        model=model,
        backend="pytorch",
        trust_remote_code=True,
        max_seq_len=4096,
        max_batch_size=4,
        kv_cache_config=kv_cache_config,
        kv_cache_compression_config=compression_config,
    ) as llm:
        outputs = llm.generate(
            ["Explain why prefix caching helps agentic workloads."],
            SamplingParams(max_tokens=128, temperature=0.0),
        )
        print(outputs[0].outputs[0].text)


def run_nvfp4_cold_page(model: str) -> None:
    """Keep active KV unchanged and quantize only Host/Disk Pages."""
    _generate(
        model,
        KvCacheConfig(
            use_kv_cache_manager_v2=True,
            dtype="auto",
            host_cache_size=8 << 30,
        ),
        ColdPageQuantizationCompressionConfig(quant="nvfp4"),
    )


def run_turboquant_cold_page(model: str, bits: float, mode: str, seed: int | None) -> None:
    """Keep active KV unchanged and store Host/Disk Pages with TurboQuant."""
    seed_option = {} if seed is None else {"turboquant_seed": seed}
    _generate(
        model,
        KvCacheConfig(
            use_kv_cache_manager_v2=True,
            dtype="auto",
            host_cache_size=8 << 30,
        ),
        ColdPageQuantizationCompressionConfig(
            quant="turboquant",
            turboquant_bits=bits,
            turboquant_mode=mode,
            **seed_option,
        ),
    )


def run_triattention(model: str, calibration_path: str) -> None:
    """Periodically compact decode KV using offline calibration."""
    _generate(
        model,
        KvCacheConfig(
            use_kv_cache_manager_v2=True,
            enable_block_reuse=True,
            dtype="auto",
        ),
        TriAttentionKvCacheCompressionConfig(
            budget=64,
            beta=32,
            eviction_mode="union",
            calibration_path=calibration_path,
        ),
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compression-method",
        choices=tuple(_DEFAULT_MODELS),
        default="nvfp4-cold-page",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model path or Hugging Face ID. Each method has a default model.",
    )
    parser.add_argument(
        "--calibration-path",
        default=None,
        help="TriAttention calibration .pt produced for the selected model.",
    )
    parser.add_argument(
        "--turboquant-bits",
        type=float,
        choices=(2.0, 2.5, 3.0, 3.5, 4.0),
        default=4.0,
        help="TurboQuant average bits per value; prod mode supports 2, 3, and 4.",
    )
    parser.add_argument(
        "--turboquant-mode",
        choices=("mse", "prod"),
        default="mse",
        help="mse minimizes reconstruction error; prod makes inner products unbiased.",
    )
    parser.add_argument(
        "--turboquant-seed",
        type=int,
        default=None,
        help="Seed of the TurboQuant rotation and QJL projection matrices "
        "(default: the configuration default).",
    )
    args = parser.parse_args()
    if args.compression_method == "triattention" and not args.calibration_path:
        parser.error("--calibration-path is required for TriAttention")
    return args


def main() -> None:
    args = parse_arguments()
    model = args.model or _DEFAULT_MODELS[args.compression_method]
    if args.compression_method == "nvfp4-cold-page":
        run_nvfp4_cold_page(model)
    elif args.compression_method == "turboquant-cold-page":
        run_turboquant_cold_page(
            model, args.turboquant_bits, args.turboquant_mode, args.turboquant_seed
        )
    else:
        run_triattention(model, args.calibration_path)


if __name__ == "__main__":
    main()
