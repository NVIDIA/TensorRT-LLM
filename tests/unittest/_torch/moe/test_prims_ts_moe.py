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
"""PrimsTS MoE behavior the shared backend matrix does not cover.

The numerics of every format run in ``test_moe_backend.py`` (``PRIMS_TS``) and
selection in ``test_moe_impl.py``. Here: the vendored tree stays self-contained,
the ``moe_sort`` token map the kernels gather rows with, and replaying a
captured PrimsTS MoE as its routes change.
"""

import ast
import importlib.util
import pathlib
import re

import pytest
import torch
from _torch.moe.moe_test_utils import MoeBackendType
from _torch.moe.quantize_utils import get_test_quant_params
from transformers.configuration_utils import PretrainedConfig

from tensorrt_llm._torch.autotuner import AutoTuner
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.moe.fused_moe import RenormalizeMoeRoutingMethod
from tensorrt_llm._torch.moe.fused_moe.create_moe import create_moe_backend
from tensorrt_llm._torch.moe.fused_moe.fused_moe_prims_ts import PrimsTsFusedMoE, prims_ts_leaf
from tensorrt_llm._torch.moe.fused_moe.impl_contract import MoECommPlan, MoERunContext
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

_ESCAPING_IMPORT = re.compile(r"^\s*(?:from|import)\s+flashinfer\.prims_ts\b", re.MULTILINE)

skip_unless_prims_ts_sm = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (100, 103, 107),
    reason="PrimsTS MoE runs on SM100, SM103 and SM107",
)


@pytest.mark.cpu_only
def test_vendored_prims_ts_moe_does_not_import_the_installed_prims_ts() -> None:
    """An absolute import would bind to the older copy in the installed flashinfer."""
    spec = importlib.util.find_spec("tensorrt_llm._torch.moe.kernels.prims_ts")
    root = pathlib.Path(spec.origin).resolve().parent
    offenders = []
    for path in root.rglob("*.py"):
        relative = path.relative_to(root)
        source = path.read_text(encoding="utf-8")
        if _ESCAPING_IMPORT.search(source) or any(
            isinstance(node, ast.ImportFrom) and node.level > len(relative.parts)
            for node in ast.walk(ast.parse(source))
        ):
            offenders.append(str(relative))
    offenders.sort()
    assert not offenders, f"vendored modules escape the vendored tree: {offenders}"


@skip_unless_prims_ts_sm
@pytest.mark.parametrize(
    "local_expert_offset, local_experts", [(0, 16), (4, 8)], ids=["all_local", "ep_slice"]
)
def test_moe_sort_token_map_names_each_routed_rows_token(
    local_expert_offset: int, local_experts: int
) -> None:
    """``return_token_map`` swaps the fourth output for a permuted-row -> token map."""
    num_tokens, top_k, num_experts, tile = 37, 4, 16, 32
    torch.manual_seed(0)
    ids = torch.stack([torch.randperm(num_experts)[:top_k] for _ in range(num_tokens)])
    ids = ids.to(torch.int32).cuda()
    weights = torch.rand((num_tokens, top_k), dtype=torch.bfloat16, device="cuda")
    args = (ids, weights, num_experts, top_k, local_expert_offset, local_experts, tile)
    expanded_tables = torch.ops.trtllm.moe_sort(*args)
    token_tables = torch.ops.trtllm.moe_sort(*args, True)

    # The flag leaves the tiles alone: the tile count, padded row count and each
    # valid tile's expert and row limit. Which row of its expert's tiles a slot
    # lands in can differ from call to call.
    num_tiles = int(expanded_tables[5])
    for index in (4, 5):
        assert torch.equal(expanded_tables[index], token_tables[index]), index
    for index in (0, 1):
        assert torch.equal(expanded_tables[index][:num_tiles], token_tables[index][:num_tiles])
    assert torch.equal(expanded_tables[2] >= 0, token_tables[2] >= 0)
    # The kernels read one element past the padded rows (getRouteMapAllocCount).
    assert token_tables[3].numel() == expanded_tables[3].numel() + 1

    # Each call's fourth output inverts its own expanded -> permuted table.
    expanded = torch.arange(num_tokens * top_k, device="cuda", dtype=torch.int32)
    for tables, row_owner in ((expanded_tables, expanded), (token_tables, expanded // top_k)):
        expanded_to_permuted = tables[2].flatten()
        routed = expanded_to_permuted >= 0
        rows = expanded_to_permuted[routed].long()
        assert torch.equal(tables[3][rows], row_owner[routed])


def _create_prims_ts_backend(
    quant_config: QuantConfig | None,
    routing_method: RenormalizeMoeRoutingMethod,
    num_experts: int,
    hidden_size: int,
    inter_size: int,
) -> PrimsTsFusedMoE:
    pretrained_config = PretrainedConfig()
    pretrained_config.num_experts = num_experts
    pretrained_config.hidden_size = hidden_size
    pretrained_config.intermediate_size = inter_size
    pretrained_config.torch_dtype = torch.bfloat16
    model_config = ModelConfig(
        pretrained_config=pretrained_config,
        quant_config=quant_config,
        mapping=Mapping(),
        moe_backend=MoeBackendType.PRIMS_TS.value,
    )
    return create_moe_backend(
        moe_cls=prims_ts_leaf(None if quant_config is None else quant_config.quant_algo),
        routing_method=routing_method,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=inter_size,
        dtype=torch.bfloat16,
        reduce_results=True,
        model_config=model_config,
        init_load_balancer=False,
    )


@skip_unless_prims_ts_sm
@pytest.mark.parametrize(
    "quant_algo",
    [None, QuantAlgo.NVFP4, QuantAlgo.W4A8_MXFP4_MXFP8],
    ids=["bf16", "nvfp4", "w4a8_mxfp4_mxfp8"],
)
def test_captured_prims_ts_moe_follows_changing_routes(quant_algo: QuantAlgo | None) -> None:
    """A captured MoE reads its routing tables on device, so every replay must
    reproduce the eager result for the routes copied into its static buffers.

    The numerics against a reference are ``test_moe_backend.py``'s.
    """
    num_experts, top_k, hidden_size, inter_size, num_tokens = 8, 2, 1024, 512, 16
    dtype = torch.bfloat16
    with torch.device("cuda"):
        torch.manual_seed(0)
        AutoTuner.get().setup_distributed_state(Mapping())
        routing_method = RenormalizeMoeRoutingMethod(top_k=top_k)
        x = torch.randn((num_tokens, hidden_size), dtype=dtype)
        logits = [torch.randn((num_tokens, num_experts), dtype=dtype) for _ in range(3)]

        quantize_util_cls, quant_config, quant_kwargs = get_test_quant_params(
            quant_algo, x, MoeBackendType.PRIMS_TS
        )
        quantize_util = quantize_util_cls(
            num_experts=num_experts,
            dtype=dtype,
            intermediate_size=inter_size,
            hidden_size=hidden_size,
            quant_config=quant_config,
        )
        backend = _create_prims_ts_backend(
            quant_config, routing_method, num_experts, hidden_size, inter_size
        )
        weights = quantize_util.create_weights(**quant_kwargs)
        backend.load_weights([weights])
        backend.post_load_weights()
        backend.cuda()

        selected = torch.empty((num_tokens, top_k), dtype=torch.int32)
        scales = torch.empty((num_tokens, top_k), dtype=dtype)
        output = torch.empty((num_tokens, hidden_size), dtype=dtype)

        def route(router_logits: torch.Tensor) -> None:
            token_selected_experts, token_final_scales = routing_method.apply(router_logits)
            selected.copy_(token_selected_experts)
            scales.copy_(token_final_scales)

        with torch.inference_mode():
            x_quantized, x_sf = backend.quantize_input(x, post_quant_comm=False)
            ctx = MoERunContext(
                token_selected_experts=selected,
                token_final_scales=scales,
                x=x_quantized,
                x_sf=x_sf,
                comm_plan=MoECommPlan(
                    input_sf_swizzled=False,
                    enable_alltoall=False,
                    moe_output=output,
                    payload_in_workspace=False,
                ),
            )

            def run_eager(router_logits: torch.Tensor) -> torch.Tensor:
                route(router_logits)
                backend.run_moe(ctx)
                return output.clone()

            # Eager first, which also compiles both GEMMs outside the capture.
            eager = [run_eager(router_logits) for router_logits in logits]
            # Otherwise a replay that ignored the new routes could pass.
            assert not torch.equal(eager[0], eager[1])
            assert not torch.equal(eager[1], eager[2])

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                backend.run_moe(ctx)
            for index in (1, 2, 0):
                route(logits[index])
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(output, eager[index], rtol=0, atol=0)
