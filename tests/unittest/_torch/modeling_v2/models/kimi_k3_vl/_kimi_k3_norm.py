# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The checks of a Kimi K3 target's ``norm.py``, each run by the target's test directory on its own module.

* ``RMSNorm`` gives the built-in RMSNorm's bits at the widths the targets normalize (hidden 7168, MoE latent 3584,
  MLA q_a 1536 and kv_a 512), on bf16 with and without a residual (the residual updated in place as the built-in
  one does), and on fp32 through the fp32 fallback.
* ``run_in_parallel`` runs both calls in order outside capture, and under capture runs ``fn1`` on the second stream
  with the replay giving the eager values.
"""

import pytest
import torch

from tensorrt_llm._torch.modules.rms_norm import RMSNorm as BuiltinRMSNorm

WIDTHS = (7168, 3584, 1536, 512)
ROWS = (1, 8, 2048)


def _pair(norm_module, width, dtype):
    torch.manual_seed(width)
    ours = norm_module.RMSNorm(width, eps=1e-6, dtype=dtype, device="cuda")
    builtin = BuiltinRMSNorm(hidden_size=width, eps=1e-6, dtype=dtype).cuda()
    weight = torch.randn(width, dtype=dtype, device="cuda")
    with torch.no_grad():
        ours.weight.copy_(weight)
        builtin.weight.copy_(weight)
    return ours, builtin


def check_rms_norm_matches_builtin(norm_module, rows, width):
    ours, builtin = _pair(norm_module, width, torch.bfloat16)
    x = torch.randn(rows, width, dtype=torch.bfloat16, device="cuda")
    assert torch.equal(ours(x), builtin(x))
    assert ours.variance_epsilon == builtin.variance_epsilon


def check_rms_norm_with_residual_matches_builtin(norm_module, rows, width):
    ours, builtin = _pair(norm_module, width, torch.bfloat16)
    x = torch.randn(rows, width, dtype=torch.bfloat16, device="cuda")
    residual = torch.randn(rows, width, dtype=torch.bfloat16, device="cuda")
    x_ours, r_ours = x.clone(), residual.clone()
    x_builtin, r_builtin = x.clone(), residual.clone()
    out_ours, sum_ours = ours(x_ours, r_ours)
    out_builtin, sum_builtin = builtin(x_builtin, r_builtin)
    assert torch.equal(out_ours, out_builtin) and torch.equal(sum_ours, sum_builtin)
    assert torch.equal(x_ours, x_builtin) and torch.equal(r_ours, r_builtin)


def check_rms_norm_fp32_matches_builtin(norm_module, width):
    ours, builtin = _pair(norm_module, width, torch.float32)
    x = torch.randn(8, width, dtype=torch.float32, device="cuda")
    residual = torch.randn(8, width, dtype=torch.float32, device="cuda")
    assert torch.equal(ours(x), builtin(x))
    out_ours, sum_ours = ours(x.clone(), residual.clone())
    out_builtin, sum_builtin = builtin(x.clone(), residual.clone())
    assert torch.equal(out_ours, out_builtin) and torch.equal(sum_ours, sum_builtin)


def check_run_in_parallel(norm_module):
    a = torch.randn(64, 64, device="cuda")
    b = torch.randn(64, 64, device="cuda")
    aux = torch.cuda.Stream()
    events = (torch.cuda.Event(), torch.cuda.Event())
    eager = norm_module.run_in_parallel(lambda: a @ a, lambda: b @ b, *events, aux)
    assert torch.equal(eager[0], a @ a) and torch.equal(eager[1], b @ b)
    assert norm_module.run_in_parallel(lambda: 1, lambda: 2, *events, None) == (1, 2)

    side_streams = []

    def on_side():
        side_streams.append(torch.cuda.current_stream())
        return b @ b

    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        captured = norm_module.run_in_parallel(lambda: a @ a, on_side, *events, aux)
    graph.replay()
    torch.cuda.synchronize()
    assert side_streams and side_streams[-1] == aux
    assert torch.equal(captured[0], eager[0]) and torch.equal(captured[1], eager[1])


def parametrize_rms_norm():
    return pytest.mark.parametrize("rows,width", [(r, w) for r in ROWS for w in WIDTHS])
