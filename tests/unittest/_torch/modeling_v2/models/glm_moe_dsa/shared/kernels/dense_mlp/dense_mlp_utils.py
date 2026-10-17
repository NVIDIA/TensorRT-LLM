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
"""Shared helpers for the dense_mlp tests and microbenchmarks.

* ``ensure_op``: make ``trtllm::glm_moe_dsa_dense_mlp*`` available. An image built from
  a commit that contains the dense_mlp thop sources already registers the ops; for
  in-job development on an older image the same sources are JIT-compiled.
* Native oracle: the exact native BF16 ``GatedMLP`` compute (``F.linear`` ->
  TRT-LLM ``swiglu`` -> ``F.linear``) and an FP64 reference with the native
  BF16 intermediate rounding.
* Real data: GLM-5.2 dense-layer weight shards as the GEN loader lays them out
  (TP4 sub-group, fused [gate; up]) and real post-RMSNorm hidden rows.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from utils.llm_data import llm_models_root

REPO_ROOT = Path(__file__).resolve().parents[9]
DEFAULT_CKPT = os.path.join(str(llm_models_root()), "GLM-5.2-NVFP4")
DEFAULT_REAL_HIDDEN = os.environ.get("GLM_MOE_DSA_REAL_MOE_INPUTS_DIR", "")
GLM52_HIDDEN = 6144
GLM52_INTER = 12288
GLM52_DENSE_TP = 4


def ensure_op(build_dir: Optional[str] = None, verbose: bool = False) -> str:
    """Return "builtin" or "jit" after making the dense_mlp ops available."""
    import tensorrt_llm  # noqa: F401  (loads th_common and its op fragments)

    if hasattr(torch.ops.trtllm, "glm_moe_dsa_dense_mlp"):
        return "builtin"
    from torch.utils.cpp_extension import load

    libs_dir = Path(tensorrt_llm.__file__).resolve().parent / "libs"
    build_dir = build_dir or os.environ.get("GLM_MOE_DSA_DENSE_MLP_JIT_BUILD_DIR")
    if build_dir:
        Path(build_dir).mkdir(parents=True, exist_ok=True)
    load(
        name="dense_mlp_jit",
        sources=[
            str(REPO_ROOT / "cpp/tensorrt_llm/kernels/glmMoeDsa/denseMlp.cu"),
            str(REPO_ROOT / "cpp/tensorrt_llm/thop/glmMoeDsa/denseMlpOp.cpp"),
        ],
        extra_include_paths=[
            str(REPO_ROOT / "cpp/include"),
            str(REPO_ROOT / "cpp"),
        ],
        extra_cflags=["-O3", "-DENABLE_BF16", "-DENABLE_FP8"],
        extra_cuda_cflags=[
            "-O3",
            "-DENABLE_BF16",
            "-DENABLE_FP8",
            "-gencode=arch=compute_103a,code=sm_103a",
            "-lineinfo",
        ],
        extra_ldflags=[f"-L{libs_dir}", "-ltensorrt_llm", f"-Wl,-rpath,{libs_dir}"],
        build_directory=build_dir,
        is_python_module=False,
        verbose=verbose,
    )
    assert hasattr(torch.ops.trtllm, "glm_moe_dsa_dense_mlp")
    return "jit"


def native_dense_mlp(
    x: torch.Tensor, gate_up_w: torch.Tensor, down_w: torch.Tensor
) -> torch.Tensor:
    """Native BF16 GatedMLP compute without the output all-reduce."""
    from tensorrt_llm._torch.modules.swiglu import swiglu

    h1 = F.linear(x, gate_up_w)
    h2 = swiglu(h1)
    return F.linear(h2, down_w)


def native_intermediate(x: torch.Tensor, gate_up_w: torch.Tensor) -> torch.Tensor:
    from tensorrt_llm._torch.modules.swiglu import swiglu

    return swiglu(F.linear(x, gate_up_w))


def oracle_dense_mlp(
    x: torch.Tensor, gate_up_w: torch.Tensor, down_w: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """FP64 sums with the native BF16 rounding points.

    Returns ``(h, out)`` as FP64 tensors whose values are BF16-representable
    where the native path rounds (gate, up, h); ``out`` is unrounded.
    """
    inter = down_w.shape[1]
    gu = x.double() @ gate_up_w.double().t()
    gate = gu[:, :inter].to(torch.bfloat16).double()
    up = gu[:, inter:].to(torch.bfloat16).double()
    h = (gate * torch.sigmoid(gate) * up).to(torch.bfloat16).double()
    out = h @ down_w.double().t()
    return h, out


def error_stats(value: torch.Tensor, ref: torch.Tensor) -> Dict[str, float]:
    diff = value.double() - ref.double()
    scale = ref.double().pow(2).mean().sqrt().item()
    return {
        "rms": diff.pow(2).mean().sqrt().item(),
        "max": diff.abs().max().item(),
        "nrms": diff.pow(2).mean().sqrt().item() / max(scale, 1e-30),
        "ref_rms": scale,
    }


def _safetensors_index(ckpt: str) -> Dict[str, str]:
    with open(os.path.join(ckpt, "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]


def load_dense_shard(
    layer: int, shard: int, ckpt: str = DEFAULT_CKPT, tp: int = GLM52_DENSE_TP, device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return GEN-layout ``(gate_up [2I/tp, H], down [H, I/tp])`` of one rank.

    Matches the native loader: gate rows ``[I/tp * shard, ...)`` then the same
    up rows; down input columns of that shard.
    """
    from safetensors import safe_open

    index = _safetensors_index(ckpt)
    prefix = f"model.layers.{layer}.mlp."
    tensors = {}
    for name in ("gate_proj", "up_proj", "down_proj"):
        key = prefix + name + ".weight"
        with safe_open(os.path.join(ckpt, index[key]), framework="pt") as f:
            sl = f.get_slice(key)
            rows, cols = sl.get_shape()
            if name == "down_proj":
                width = cols // tp
                tensors[name] = sl[:, width * shard : width * (shard + 1)]
            else:
                width = rows // tp
                tensors[name] = sl[width * shard : width * (shard + 1), :]
    gate_up = torch.cat([tensors["gate_proj"], tensors["up_proj"]], dim=0)
    down = tensors["down_proj"].contiguous()
    assert gate_up.dtype == torch.bfloat16 and down.dtype == torch.bfloat16
    return gate_up.to(device).contiguous(), down.to(device).contiguous()


def load_real_hidden(
    fixture_dir: str = DEFAULT_REAL_HIDDEN, layer: int = 10, tokens: int = 2311
) -> Optional[torch.Tensor]:
    """Real post-RMSNorm hidden rows [tokens, 6144] (BF16) or None if absent.

    The fixture captured the routed-MoE input (``post_attention_layernorm``
    output) of the native model; it serves as a realistic activation for the
    dense MLP, which consumes the same kind of tensor in layers 0-2.
    """
    path = os.path.join(fixture_dir, f"rows{tokens}_layer{layer}.pt")
    if not os.path.exists(path):
        return None
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for key in ("hidden_states", "hidden", "x", "input"):
            if key in obj and isinstance(obj[key], torch.Tensor):
                return obj[key].to(torch.bfloat16)
        for value in obj.values():
            if (
                isinstance(value, torch.Tensor)
                and value.dim() == 2
                and value.shape[-1] == GLM52_HIDDEN
            ):
                return value.to(torch.bfloat16)
        return None
    if isinstance(obj, torch.Tensor):
        return obj.to(torch.bfloat16)
    return None
