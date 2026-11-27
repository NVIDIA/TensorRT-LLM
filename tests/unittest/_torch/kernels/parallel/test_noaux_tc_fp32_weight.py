# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The FP32 noaux_tc routing weight against the FP64 expression it replaces, bit for bit.

The launcher reads TRTLLM_DISABLE_NOAUX_TC_FP32_WEIGHT once per process, so each variant runs in its own child
process on the same seeded inputs.
"""

import os
import subprocess
import sys

import pytest
import torch

DISABLE_ENV = "TRTLLM_DISABLE_NOAUX_TC_FP32_WEIGHT"

# (num_experts, n_group, topk_group, top_k)
TOPK_CONFIGS = [
    (72, 1, 1, 6),
    (128, 1, 1, 8),
    (256, 8, 4, 8),
    (512, 1, 1, 22),
    (1024, 1, 1, 32),
]
NUM_TOKENS = [1, 37, 1024]
# 2.827 and 1/3 are not floats with at most 20 significant bits, so every weight takes the FP64 fallback.
SCALES = [2.5, 1.0, 2.827, 1.0 / 3.0]
# "negative" puts every sigmoid score near 1e-11, below the S >= 2^-13 bound of the FP32 path.
DISTRIBUTIONS = ["normal", "ties", "negative"]


def _scores(num_tokens, num_experts, dist, gen):
    scores = torch.randn(num_tokens, num_experts, generator=gen, device="cuda", dtype=torch.float32)
    bias = 0.1 * torch.randn(num_experts, generator=gen, device="cuda", dtype=torch.float32)
    if dist == "ties":
        scores[:, ::3] = scores[:, :1]
        bias.zero_()
    elif dist == "negative":
        scores -= 25.0
    return scores, bias


def _child(out_path):
    import tensorrt_llm  # noqa: F401  (registers the trtllm ops)

    results = {}
    for num_experts, n_group, topk_group, top_k in TOPK_CONFIGS:
        for num_tokens in NUM_TOKENS:
            for scale in SCALES:
                for dist in DISTRIBUTIONS:
                    gen = torch.Generator(device="cuda").manual_seed(
                        num_experts * 7919 + num_tokens
                    )
                    scores, bias = _scores(num_tokens, num_experts, dist, gen)
                    values, indices = torch.ops.trtllm.noaux_tc_op(
                        scores, bias, n_group, topk_group, top_k, scale
                    )
                    key = f"topk/{num_experts}/{n_group}/{topk_group}/{top_k}/{num_tokens}/{scale}/{dist}"
                    results[key + "/values"] = values.view(torch.int32).cpu()
                    results[key + "/indices"] = indices.cpu()

    gen = torch.Generator(device="cuda").manual_seed(0)
    scores, bias = _scores(8, 128, "normal", gen)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        torch.ops.trtllm.noaux_tc_op(scores, bias, 1, 1, 8, 2.0)
        torch.cuda.synchronize()
    kernels = sorted({e.name for e in prof.events() if "topk" in e.name or "noaux" in e.name})
    torch.save({"results": results, "kernels": kernels}, out_path)


def _run_child(tmp_path, name, disable):
    env = dict(os.environ)
    env.pop(DISABLE_ENV, None)
    if disable:
        env[DISABLE_ENV] = "1"
    out = tmp_path / f"{name}.pt"
    proc = subprocess.run(
        [sys.executable, __file__, str(out)],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    return torch.load(out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_fp32_weight_matches_fp64_expression(tmp_path):
    fp32 = _run_child(tmp_path, "fp32", disable=False)
    fp64 = _run_child(tmp_path, "fp64", disable=True)

    assert any("deepseek_v3_topk_fp32_weight_kernel" in k for k in fp32["kernels"]), fp32["kernels"]
    assert not any("fp32_weight" in k for k in fp64["kernels"]), fp64["kernels"]

    assert fp32["results"].keys() == fp64["results"].keys()
    differ = [k for k, v in fp32["results"].items() if not torch.equal(v, fp64["results"][k])]
    assert not differ, differ[:20]


if __name__ == "__main__":
    _child(sys.argv[1])
