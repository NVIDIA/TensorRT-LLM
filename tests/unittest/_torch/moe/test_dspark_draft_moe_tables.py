# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSpark draft MoE (TRTLLM-Gen MXFP4 x MXFP8) on the draft route kernel's routing tables gives the
self-routing run's output bit for bit, finalized or not, eagerly and in a CUDA graph."""

from contextlib import ExitStack

import pytest
import torch
from _torch.moe.quantize_utils import MXFP4MXFP8QuantizeUtil
from transformers import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_deepseekv4 import DeepseekV4Gate
from tensorrt_llm._torch.modules.mhc.mhc_lowm import MoeUnfinalized
from tensorrt_llm._torch.moe.fused_moe import SwigluActivation, create_moe
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (100, 103),
    reason="SM100/SM103 TRTLLM-Gen MXFP4 x MXFP8 cubins required",
)

_EXPERTS, _TOP_K, _SCALE, _HIDDEN, _INTER, _CLAMP = 128, 3, 2.5, 1024, 512, 10.0


def _combined(parts, rows):
    """The un-finalized rows summed over top-k in FP32 and rounded once."""
    fc2, scales, perm = parts
    shared = torch.zeros(rows, _HIDDEN, dtype=torch.bfloat16, device="cuda")
    return MoeUnfinalized(shared, fc2, scales.view(rows, -1), perm.view(rows, -1)).finalize()


@torch.inference_mode()
def test_draft_moe_on_route_kernel_tables(monkeypatch):
    from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import MxE4m3MxE2m1BlockScaleMoERunner

    on_tables = []
    forward = MxE4m3MxE2m1BlockScaleMoERunner.forward

    def recording(self, inputs, tactic=[-1, -1], routing_tables=None, **kw):  # noqa: B006
        on_tables.append(routing_tables is not None and tactic[0] in kw.get("routing_tiles", ()))
        return forward(self, inputs, tactic=tactic, routing_tables=routing_tables, **kw)

    monkeypatch.setattr(MxE4m3MxE2m1BlockScaleMoERunner, "forward", recording)
    monkeypatch.setenv("ENABLE_PERFECT_ROUTER", "0")
    quant_config = QuantConfig(quant_algo=QuantAlgo.W4A8_MXFP4_MXFP8)
    model_config = ModelConfig(
        pretrained_config=PretrainedConfig(
            hidden_size=_HIDDEN,
            intermediate_size=_INTER,
            num_experts=_EXPERTS,
            torch_dtype=torch.bfloat16,
        ),
        mapping=Mapping(world_size=1, rank=0, tp_size=1),
        quant_config=quant_config,
        moe_backend="TRTLLM",
        max_num_tokens=256,
    )
    from tensorrt_llm._torch.autotuner import autotune

    with torch.device("cuda"), ExitStack() as stack:
        torch.manual_seed(1703)
        gate = DeepseekV4Gate(
            _HIDDEN,
            _EXPERTS,
            top_k=_TOP_K,
            n_group=1,
            topk_group=1,
            routed_scaling_factor=_SCALE,
            is_hashed=False,
            dtype=torch.bfloat16,
        )
        gate.weight.copy_(torch.randn_like(gate.weight) / _HIDDEN**0.5)
        gate.e_score_correction_bias.copy_(torch.randn(_EXPERTS, dtype=torch.float32) * 0.05)
        weights = MXFP4MXFP8QuantizeUtil(
            num_experts=_EXPERTS,
            dtype=torch.bfloat16,
            intermediate_size=_INTER,
            hidden_size=_HIDDEN,
            quant_config=quant_config,
            swiglu_alpha=1.0,
            swiglu_beta=0.0,
            swiglu_limit=_CLAMP,
        ).create_weights(input_hidden_alignment=128)
        modules = {}
        for opt_in in (False, True):
            moe = stack.enter_context(
                create_moe(
                    routing_method=gate.routing_method,
                    model_config=model_config,
                    reduce_results=False,
                    activation=SwigluActivation(clamp=_CLAMP),
                )
            )
            # As the DSpark draft MoE does; without it the runner routes itself.
            moe.backend.unfinalized_output_opt_in = opt_in
            # DeepSeek-V4.1's draft takes the CuTe FC2 (cute_fc2_rows); plain keeps the FC2 cubin.
            moe.backend.cute_fc2_rows = 16 if opt_in else 0
            moe.load_weights([weights])
            moe.post_load_weights()
            modules[opt_in] = moe
        plain, tabled = modules[False], modules[True]

        for m in (1, 5, 8):
            x = torch.randn(m, _HIDDEN, dtype=torch.bfloat16)
            logits = gate(x)

            def run(moe, finalize=True):
                return moe(x, logits, all_rank_num_tokens=[m], do_finalize=finalize)

            # The tables replace the routing of tuned tactics whose tile they cover.
            with autotune():
                run(plain), run(plain, False)
            expected = run(plain)
            expected_rows = _combined(run(plain, False), m)
            on_tables.clear()
            assert torch.equal(run(tabled).view(torch.int16), expected.view(torch.int16))
            # Un-finalized, the runner reports the tile it ran so later calls lay out only its table.
            got_rows = _combined(run(tabled, False), m)
            assert torch.equal(got_rows.view(torch.int16), expected_rows.view(torch.int16))
            assert any(on_tables), f"{m} tokens did not run on the route kernel's tables"

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                run(tabled), run(tabled, False)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            on_tables.clear()
            with torch.cuda.graph(graph):
                captured = run(tabled)
                captured_parts = run(tabled, False)
            assert all(on_tables)
            for _ in range(2):
                x.copy_(torch.randn_like(x))
                logits.copy_(gate(x))
                graph.replay()
                expected = run(plain)
                assert torch.equal(captured.view(torch.int16), expected.view(torch.int16))
                assert torch.equal(
                    _combined(captured_parts, m).view(torch.int16),
                    _combined(run(plain, False), m).view(torch.int16),
                )
