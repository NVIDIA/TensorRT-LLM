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
"""GPU tests of the live-row word: its writes are ordered on the stream, and the MegaMoE
launch drops the padding rows (the route staging computes the host's live rows on the
device: test_megamoe_stage_routes.py).

The drop test runs one MegaMoE layer on four ranks twice over the same inputs: once with
every row live, once with each rank's padding requests dropped. The live rows must come
out bit for bit the same, the dropped rows zero, and the per-expert counts the router
publishes to its peers must equal the host histogram of the live rows' routes.
"""

import os
import pickle
import sys
import traceback
from types import SimpleNamespace

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

from tensorrt_llm._torch.moe import live_rows
from tensorrt_llm._utils import get_sm_version, mpi_rank

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

_BLACKWELL = torch.cuda.is_available() and get_sm_version() in (100, 103)
requires_blackwell = pytest.mark.skipif(
    not _BLACKWELL, reason="the MegaMoE Blackwell kernel's top-k reduce masks dropped routes"
)

# One MegaMoE layer of the multi-GPU module tests' CI shape (experts, top-k, hidden, intermediate).
_SHAPE = (60, 4, 2048, 1408)
_WORLD = 4
# Requests of every rank's forward and rows per request: four, as a one-model MTP target
# forward with three draft tokens.
_REQUESTS, _ROWS_PER_REQUEST = 8, 4
# Real requests per rank: a full batch, graph padding behind five and behind three real
# requests, and an attention-DP placeholder rank.
_REAL_REQUESTS = (8, 5, 0, 3)
# The top-k reduce has one store path per combine wire format; each must zero the dropped rows.
_COMBINE_FORMATS = ("bf16", "32e4m3xe8m0", "16e2m1xbf16")


@requires_blackwell
def test_megamoe_cutedsl_live_row_word_writes_are_stream_ordered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cuda.bindings import driver

    device = torch.device("cuda", torch.cuda.current_device())
    word = live_rows.LiveRowWord()
    word.write(live_rows.encode(3, 8))  # before the word exists: it is allocated holding this
    tensor = word.tensor(device)
    seen = [tensor.clone()]
    expected = [live_rows.encode(3, 8)]
    for value in (live_rows.encode(5, 8), live_rows.ALL_ROWS, live_rows.NO_ROWS, live_rows.NO_ROWS):
        # Work enqueued before a write reads the old value even while the host runs ahead.
        torch.cuda._sleep(2_000_000)
        seen.append(tensor.clone())
        expected.append(expected[-1])
        word.write(value)
        seen.append(tensor.clone())
        expected.append(value)
    torch.cuda.synchronize()
    assert [int(t.item()) for t in seen] == expected

    # Without stream memory operations the word keeps every row and is never written.
    monkeypatch.setattr(
        driver, "cuStreamWriteValue32", lambda *_: (driver.CUresult.CUDA_ERROR_NOT_SUPPORTED,)
    )
    unsupported = live_rows.LiveRowWord()
    unsupported.write(live_rows.encode(3, 8))
    assert int(unsupported.tensor(device).item()) == live_rows.ALL_ROWS
    unsupported.write(live_rows.NO_ROWS)
    assert (
        int(unsupported.tensor(device).item()) == live_rows.ALL_ROWS
        and not unsupported._stream_writes_supported
    )


def _init_worker(custom_paths: list) -> None:
    for custom_path in custom_paths:
        if custom_path.endswith("tests/unittest") and custom_path not in sys.path:
            sys.path.append(custom_path)


def _published_counts(fused_moe: torch.nn.Module, rank: int, rows: int) -> torch.Tensor:
    """This rank's per-expert counts as its router published them to every peer (itself too)."""
    from tensorrt_llm._torch.moe.custom_ops import cute_dsl_megamoe_custom_op as op

    backend = fused_moe.backend
    max_tokens = backend._select_launch_max_tokens(rows)
    buffers = [b for key, b in backend._launch_buffers.items() if key[0] == max_tokens and key[-1]]
    kernels = [
        kernel
        for _, kernel in op.Sm100MegaMoENvfp4Runner.kernel_cache.values()
        if kernel.max_tokens_per_rank == max_tokens
    ]
    assert len(buffers) == 1 and len(kernels) == 1, (len(buffers), len(kernels))
    router = kernels[0].token_comm._router
    offset = kernels[0]._device_workspace.offset(router.sizes_by_rank_region)
    width = router.expert_count_padded
    matrix = buffers[0].shared_workspace[offset : offset + _WORLD * width * 4]
    return matrix.view(torch.int32).view(_WORLD, width)[rank, : router.expert_count].cpu().clone()


def _drop_worker(real_requests_by_rank: tuple, combine_format: str) -> None:
    try:
        _drop_worker_impl(real_requests_by_rank, combine_format)
    except Exception:
        traceback.print_exc()
        raise


def _drop_worker_impl(real_requests_by_rank: tuple, combine_format: str) -> None:
    from _torch.moe.moe_test_utils import MoeBackendType
    from _torch.moe.quantize_utils import get_test_quant_params
    from _torch.moe.test_moe_module import (
        _create_mapping_for_parallel_mode,
        _create_model_config,
        _ensure_dist_for_megamoe,
    )

    from tensorrt_llm._torch.moe.fused_moe import (
        DEFAULT_MOE_ACTIVATION,
        RenormalizeMoeRoutingMethod,
        create_moe,
    )
    from tensorrt_llm._torch.moe.fused_moe.interface import MoEWeightLoadingMode
    from tensorrt_llm._torch.nccl_window_tensor_scope import nccl_window_tensor_scope
    from tensorrt_llm._torch.utils import ActivationType, model_extra_attrs
    from tensorrt_llm.models.modeling_utils import QuantAlgo

    os.environ.pop("TRTLLM_FORCE_COMM_METHOD", None)
    os.environ["MEGAMOE_COMBINE_FORMAT"] = combine_format
    rank = mpi_rank()
    mapping = _create_mapping_for_parallel_mode(_WORLD, "DEP")
    mapping.rank = rank
    torch.cuda.set_device(rank)
    backend_name = MoeBackendType.MEGAMOE_CUTEDSL.value
    _ensure_dist_for_megamoe(backend_name, rank, _WORLD)
    num_experts, top_k, hidden_size, intermediate_size = _SHAPE
    rows = _REQUESTS * _ROWS_PER_REQUEST
    dtype = torch.bfloat16

    with torch.device(f"cuda:{rank}"):
        # The same inputs and weights on every rank, as in the multi-GPU module tests.
        torch.manual_seed(0)
        routing_method = RenormalizeMoeRoutingMethod(top_k=top_k, force_enable_pytorch_op=True)
        x = torch.randn((rows, hidden_size), dtype=dtype, device="cuda")
        router_logits = torch.randn((rows, num_experts), dtype=dtype, device="cuda")
        quantize_util_cls, quant_config, quant_kwargs = get_test_quant_params(
            QuantAlgo.NVFP4, x, MoeBackendType.MEGAMOE_CUTEDSL
        )
        quant_kwargs.pop("ref_cls", None)
        quantize_util = quantize_util_cls(
            num_experts=num_experts,
            dtype=dtype,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
            quant_config=quant_config,
            bias=False,
            swiglu_gptoss_style=False,
            swiglu_alpha=None,
            swiglu_beta=None,
            swiglu_limit=None,
            num_local_experts=num_experts // mapping.moe_ep_size,
            activation_type=ActivationType.Swiglu,
        )
        model_cfg = _create_model_config(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            mapping=mapping,
            quant_config=quant_config,
            moe_backend=backend_name,
            max_num_tokens=max(256, rows),
        )
        with (
            nccl_window_tensor_scope(x),
            create_moe(
                routing_method=routing_method,
                reduce_results=True,
                model_config=model_cfg,
                bias=False,
                activation=DEFAULT_MOE_ACTIVATION,
                weight_loading_mode=getattr(
                    quantize_util, "weight_loading_mode", MoEWeightLoadingMode.VANILLA
                ),
            ) as fused_moe,
        ):
            fused_moe.create_weights()
            fused_moe.load_weights([quantize_util.create_weights(**quant_kwargs)])
            fused_moe.post_load_weights()
            fused_moe.cuda(f"cuda:{rank}")

            word = live_rows.LiveRowWord()
            attrs = {live_rows.LIVE_ROW_WORD_ATTR: word}

            def forward(value: int) -> tuple:
                word.write(value)
                with torch.inference_mode(), model_extra_attrs(attrs):
                    output = fused_moe.forward(
                        x, router_logits, all_rank_num_tokens=[rows] * _WORLD
                    ).clone()
                torch.cuda.synchronize()
                return output, _published_counts(fused_moe, rank, rows)

            real = real_requests_by_rank[rank]
            requests = [SimpleNamespace(is_dummy=index >= real) for index in range(_REQUESTS)]
            value = live_rows.forward_word(
                requests,
                num_contexts=0,
                seq_lens=torch.full((_REQUESTS,), _ROWS_PER_REQUEST, dtype=torch.int32),
            )
            live = live_rows.live_row_count(value, rows)
            assert live == real * _ROWS_PER_REQUEST

            out_all, counts_all = forward(live_rows.ALL_ROWS)
            out_again, _ = forward(live_rows.ALL_ROWS)
            assert torch.equal(out_all, out_again), "the launch must be deterministic"
            out_drop, counts_drop = forward(value)

            with torch.inference_mode():
                routes, _ = routing_method.apply(router_logits)
            routes = routes.long().cpu()
            histogram_all = torch.bincount(routes.flatten(), minlength=num_experts).to(torch.int32)
            histogram_live = torch.bincount(routes[:live].flatten(), minlength=num_experts).to(
                torch.int32
            )
            assert torch.equal(counts_all, histogram_all)
            assert torch.equal(counts_drop, histogram_live)
            assert torch.equal(out_drop[:live], out_all[:live])
            assert not out_drop[live:].any()


@requires_blackwell
@pytest.mark.skipif(torch.cuda.device_count() < _WORLD, reason=f"needs {_WORLD} GPUs")
@pytest.mark.threadleak(enabled=False)  # the pool's manager thread persists by design
@pytest.mark.parametrize("combine_format", _COMBINE_FORMATS)
def test_megamoe_cutedsl_drops_padding_rows(combine_format: str) -> None:
    with MPIPoolExecutor(
        initializer=_init_worker, initargs=(sys.path,), max_workers=_WORLD
    ) as executor:
        results = list(
            executor.map(_drop_worker, [_REAL_REQUESTS] * _WORLD, [combine_format] * _WORLD)
        )
    assert results == [None] * _WORLD
