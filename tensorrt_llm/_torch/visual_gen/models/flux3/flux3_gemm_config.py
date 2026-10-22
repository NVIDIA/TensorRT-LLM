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

"""FLUX.3 PrimsTS tactic selection."""

from __future__ import annotations

import hashlib
import os
from contextlib import AbstractContextManager, nullcontext
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from tensorrt_llm._torch.visual_gen.prims_ts.gemm import PrimsTsGemmConfig

_TACTICS = {
    ("fp8_e4m3", "linear", 3072, 9216): {
        1024: (2, False, 6, 8, False),
        8192: (2, False, 5, 4, False),
        10240: (2, False, 6, 4, False),
        32768: (2, False, 6, 4, False),
        40960: (2, False, 6, 8, False),
    },
    ("fp8_e4m3", "linear", 3072, 3072): {
        1024: (2, False, 6, 8, True),
        8192: (2, False, 6, 4, True),
        10240: (2, False, 6, 4, True),
        32768: (4, False, 6, 4, True),
        40960: (4, False, 6, 4, True),
    },
    ("fp8_e4m3", "swiglu", 3072, 18432): {
        1024: (2, False, 6, 8, False),
        8192: (4, False, 6, 8, False),
        10240: (4, False, 5, 8, False),
        32768: (2, False, 6, 8, False),
        40960: (2, False, 5, 8, False),
    },
    ("fp8_e4m3", "linear", 9216, 3072): {
        1024: (2, False, 6, 4, True),
        8192: (4, False, 6, 4, True),
        10240: (2, False, 6, 4, True),
        32768: (2, False, 6, 4, False),
        40960: (4, False, 6, 4, False),
    },
    ("fp8_e4m3", "qkv_qknorm_rope", 3072, 9216): {
        1024: (2, False, 4, 8, True),
        8192: (2, False, 4, 8, True),
        10240: (2, False, 4, 8, True),
        32768: (4, False, 4, 8, True),
        40960: (4, False, 4, 8, True),
    },
    ("nvfp4_e2m1", "linear", 3072, 9216): {
        1024: (2, True, 5, 8, False),
        8192: (2, True, 5, 8, False),
        10240: (2, True, 5, 8, False),
        32768: (2, True, 5, 8, False),
        40960: (4, True, 5, 8, False),
    },
    ("nvfp4_e2m1", "linear", 3072, 3072): {
        1024: (2, False, 5, 8, True),
        8192: (4, True, 5, 8, False),
        10240: (2, True, 5, 8, False),
        32768: (4, True, 5, 8, False),
        40960: (4, True, 5, 8, False),
    },
    ("nvfp4_e2m1", "linear", 9216, 3072): {
        1024: (4, False, 5, 4, True),
        8192: (2, True, 5, 8, False),
        10240: (2, True, 5, 8, False),
        32768: (4, True, 5, 8, False),
        40960: (4, True, 5, 8, False),
    },
    ("nvfp4_e2m1", "swiglu", 3072, 18432): {
        1024: (4, False, 5, 8, False),
        8192: (2, False, 5, 8, False),
        10240: (4, False, 5, 8, False),
        32768: (4, False, 5, 8, False),
        40960: (2, False, 5, 8, False),
    },
    ("nvfp4_e2m1", "qkv_qknorm_rope", 3072, 9216): {
        1024: (2, False, 4, 8, True),
        8192: (2, False, 4, 8, True),
        10240: (2, False, 4, 8, True),
        32768: (4, False, 4, 8, True),
        40960: (4, False, 4, 8, True),
    },
}


def autotune_enabled() -> bool:
    return os.environ.get("TRTLLM_FLUX3_PRIMS_TS_AUTOTUNE", "0") == "1"


def autotune_context(
    device: torch.device | str, rank: int, *, tune: bool = True
) -> AbstractContextManager:
    """Persist FlashInfer tactics and optionally restrict the measured M buckets."""
    if not autotune_enabled():
        return nullcontext()

    import torch
    from flashinfer import autotune

    raw_buckets = os.environ.get("TRTLLM_FLUX3_PRIMS_TS_TUNING_BUCKETS")
    buckets = None
    if raw_buckets is not None:
        try:
            buckets = tuple(sorted({int(value) for value in raw_buckets.split(",")}))
            if any(value <= 0 for value in buckets):
                raise ValueError("nonpositive bucket")
        except ValueError as exc:
            raise ValueError(
                "TRTLLM_FLUX3_PRIMS_TS_TUNING_BUCKETS must contain positive comma-separated integers"
            ) from exc

    if not tune:
        return autotune(False, tuning_buckets=buckets, round_up=True)

    root = Path(
        os.environ.get("TRTLLM_FLUX3_PRIMS_TS_CACHE_DIR", "~/.cache/tensorrt_llm/flux3_prims_ts")
    ).expanduser()
    gpu = torch.cuda.get_device_name(device).replace(" ", "_")
    cache_dir = root / gpu / f"rank{rank}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return autotune(
        True,
        cache=str(cache_dir / f"gemm-cold-l2-{_gemm_source_digest()}.json"),
        tuning_buckets=buckets,
        round_up=True,
    )


@lru_cache(maxsize=1)
def _gemm_source_digest() -> str:
    """Invalidate saved tactics when vendored kernels or their tuning policy change."""
    from tensorrt_llm._torch.visual_gen.prims_ts import gemm

    sources = sorted(Path(gemm.__file__).parent.glob("*.py"))
    if not any(source.name == "kernel.py" for source in sources):
        raise FileNotFoundError("Vendored PrimsTS GEMM kernel.py is missing")
    digest = hashlib.sha256()
    for source in sources:
        digest.update(source.name.encode())
        digest.update(source.read_bytes())
    return digest.hexdigest()[:16]


@lru_cache(maxsize=512)
def gemm_config(
    arch: int,
    operand_format: str,
    epilogue: str,
    m: int,
    n: int,
    k: int,
    output_format: str = "bf16",
    head_dim: int | None = None,
    qkv_scale: bool = False,
    tune: bool = False,
) -> PrimsTsGemmConfig | None:
    from tensorrt_llm._torch.visual_gen.prims_ts.gemm import PrimsTsGemmConfig

    if tune:
        return None
    kwargs = dict(
        arch=arch,
        operand_format=operand_format,
        output_format=output_format,
        epilogue=epilogue,
        has_bias=False,
        head_dim=head_dim,
        is_neox=False if epilogue == "qkv_qknorm_rope" else None,
        has_qkv_scale=qkv_scale,
        tile_k=128 if operand_format == "fp8_e4m3" else 256,
    )
    bucket = next((b for b in (1024, 8192, 10240, 32768, 40960) if m <= b), 40960)
    tactic = _TACTICS.get((operand_format, epilogue, k, n), {}).get(bucket)
    if tactic is not None:
        cluster, overlap, stages, warps, tma = tactic
        if operand_format == "nvfp4_e2m1" and epilogue in ("swiglu", "qkv_qknorm_rope"):
            overlap, tma = True, False
        kwargs.update(
            cluster_shape=(cluster, cluster, 1),
            tmem_overlap=overlap,
            ab_stages=stages,
            epilogue_warps=warps,
            use_tma_store=tma,
        )
    if epilogue == "qkv_qknorm_rope":
        # Separate Q, K and V planes spare the attention backend a copy of QKV.
        # The kernel writes them with direct stores.
        kwargs.update(separate_qkv_output=True, use_tma_store=False)
    return PrimsTsGemmConfig(**kwargs)
