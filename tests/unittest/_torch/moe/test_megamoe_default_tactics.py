# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash MegaMoE on four GPUs under a quantized combine: the per-tile publish and bulk FC2 store
defaults match the previous default."""

import os
import pickle
import sys

import cloudpickle
import pytest
import torch
from _torch.moe.quantize_utils import NVFP4QuantizeUtil
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
from transformers import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.moe.fused_moe import (
    RenormalizeMoeRoutingMethod,
    SwigluActivation,
    create_moe,
)
from tensorrt_llm._utils import get_sm_version, mpi_rank, mpi_world_size
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

_WORLD_SIZE = 4
_NUM_EXPERTS = 384
_TOP_K = 6
_HIDDEN_SIZE = 5120
_INTERMEDIATE_SIZE = 2304

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)


def _init_worker(paths: list[str]) -> None:
    sys.path[:] = paths


# (live tokens on rank 0, expected rung) of the quantized-combine check: the per-tile publish tactic
# and the bulk FC2 store from 2048 tokens per rank.
_COMBINE_RUNG_TOKENS = ((1000, 1024), (1664, 2048))


def _run_quantized_combine() -> None:
    from tensorrt_llm._torch.moe.custom_ops import cute_dsl_megamoe_custom_op as op

    os.environ["MEGAMOE_COMBINE_FORMAT"] = "16e2m1xbf16"
    rank, world_size = mpi_rank(), mpi_world_size()
    torch.cuda.set_device(rank)
    mapping = Mapping(
        world_size=world_size,
        rank=rank,
        gpus_per_node=world_size,
        tp_size=world_size,
        moe_ep_size=world_size,
        moe_tp_size=1,
        enable_attention_dp=True,
    )
    quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=16)
    max_rung = max(rung for _, rung in _COMBINE_RUNG_TOKENS)
    model_config = ModelConfig(
        pretrained_config=PretrainedConfig(
            architectures=["DeepseekV41ForCausalLM"],
            hidden_size=_HIDDEN_SIZE,
            intermediate_size=_INTERMEDIATE_SIZE,
            num_experts=_NUM_EXPERTS,
            torch_dtype=torch.bfloat16,
        ),
        mapping=mapping,
        quant_config=quant_config,
        moe_backend="MEGAMOE_CUTEDSL",
        max_num_tokens=max_rung,
        moe_max_num_tokens=max_rung,
    )
    with torch.device(f"cuda:{rank}"), torch.inference_mode():
        torch.manual_seed(1701)
        quantizer = NVFP4QuantizeUtil(
            num_experts=_NUM_EXPERTS,
            dtype=torch.bfloat16,
            intermediate_size=_INTERMEDIATE_SIZE,
            hidden_size=_HIDDEN_SIZE,
            quant_config=quant_config,
        )
        weights = quantizer.create_weights(x_sf_global=torch.tensor(448.0 * 6.0))
        with create_moe(
            routing_method=RenormalizeMoeRoutingMethod(top_k=_TOP_K),
            model_config=model_config,
            reduce_results=False,
            activation=SwigluActivation(clamp=10.0),
        ) as moe:
            assert moe.backend.combine_format == "16e2m1xbf16"
            moe.load_weights([weights])
            moe.post_load_weights()
            del weights
            for tokens, rung in _COMBINE_RUNG_TOKENS:
                assert rung in moe.backend._maxt_buckets, moe.backend._maxt_buckets
                torch.manual_seed(1702 + rank)
                x = torch.randn((tokens, _HIDDEN_SIZE), dtype=torch.bfloat16) * 0.2
                logits = torch.randn((tokens, _NUM_EXPERTS), dtype=torch.bfloat16)
                counts = [tokens] * world_size

                def run(tactic):
                    with pytest.MonkeyPatch.context() as patch:
                        patch.setattr(
                            op,
                            "_default_megamoe_tactic_for_key",
                            lambda key, *, num_tokens: tactic,
                        )
                        patch.setattr(op, "_megamoe_default_tactic_candidates", lambda t: (t,))
                        return moe(x, logits, all_rank_num_tokens=counts).clone()

                # The previous default publishes FC2 completions in batches of 4 tiles.
                reference = run(op._MEGAMOE_NONBULK_STANDALONE_TACTIC)
                assert torch.isfinite(reference).all()
                for tactic in (
                    op._MEGAMOE_QUANTIZED_COMBINE_TACTIC,
                    op._MEGAMOE_NVFP4_COMBINE_BULK_TACTIC,
                ):
                    torch.testing.assert_close(run(tactic), reference, rtol=0, atol=0)
                default = moe(x, logits, all_rank_num_tokens=counts).clone()
                torch.testing.assert_close(default, reference, rtol=0, atol=0)


@pytest.mark.threadleak(enabled=False)
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < _WORLD_SIZE,
    reason=f"Requires {_WORLD_SIZE} GPUs",
)
@pytest.mark.skipif(get_sm_version() not in (100, 103), reason="MegaMoE CuteDSL on SM100/SM103")
def test_megamoe_quantized_combine_tactics_bitwise() -> None:
    if mpi_world_size() == _WORLD_SIZE:
        _run_quantized_combine()
        return
    if mpi_world_size() > 1:
        pytest.skip(f"Requires MPI world size {_WORLD_SIZE} or a single process")
    with MPIPoolExecutor(
        max_workers=_WORLD_SIZE, initializer=_init_worker, initargs=(sys.path,)
    ) as executor:
        results = [executor.submit(_run_quantized_combine) for _ in range(_WORLD_SIZE)]
        assert all(result.result() is None for result in results)
