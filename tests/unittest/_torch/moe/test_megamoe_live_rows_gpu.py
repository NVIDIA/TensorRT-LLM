# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MegaMoE CuteDSL on four ranks drops the padding rows of a forward end to end.

One layer runs twice over the same inputs: every row live, then each rank's padding requests
dropped. Live rows must match bit for bit, the dropped rows must be zero (a shared expert fused into
the kernel included: it reads the same word), and the per-expert counts the router publishes must
equal the histogram of the live rows' routes. Generic routing stages through megamoe_stage_inputs;
DeepSeek-V4 routing through its fused route+quant.
"""

import contextlib
import os
import pickle
import sys
import traceback
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import cloudpickle
import pytest
import torch
from mpi4py import MPI

from tensorrt_llm._torch.moe import live_rows
from tensorrt_llm._utils import get_sm_version, mpi_rank

if torch.cuda.is_available():
    # The 4-GPU MoE cases' shared pool; test_moe_module queries the device when imported.
    from _torch.moe.test_moe_module import moe_multi_gpu_executor  # noqa: F401

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
# DeepSeek-V4 routing takes the fused route+quant launch, which stages the bucket itself; its
# reference routing (gate_forward) takes 256 or 384 experts (and the rest of the CI shape).
_DSV4_SHAPE = (256, 6, 2048, 1408)
_ROUTINGS = ("renormalize", "dsv4")
# DeepSeek-V4 with its FP8 shared expert fused into the kernel (megamoe_fuse_shared_expert), whose
# intermediate size the fusion needs as a multiple of 256 (gate/up rows in 512-row clusters).
_FUSED_SHARED = "dsv4_fused_shared"
_SHARED_INTERMEDIATE = 1024
_CASES = [(f, r) for f in _COMBINE_FORMATS for r in _ROUTINGS] + [("bf16", _FUSED_SHARED)]


@contextmanager
def _forward_kernels(runner_class):
    """Observe the descriptor returned to each actual runner forward, then restore the method."""
    kernels = []
    original = runner_class._resolve_kernel_descriptor

    def resolve(runner, *args, **kwargs):
        result = original(runner, *args, **kwargs)
        kernels.append(result[0])
        return result

    with patch.object(runner_class, "_resolve_kernel_descriptor", resolve):
        yield kernels


def _published_counts(
    fused_moe: torch.nn.Module, rank: int, rows: int, kernels: list
) -> torch.Tensor:
    """This rank's per-expert counts as its router published them to every peer (itself too)."""
    backend = fused_moe.backend
    max_tokens = backend._select_launch_max_tokens(rows)
    buffers = [b for key, b in backend._launch_buffers.items() if key[0] == max_tokens and key[-1]]
    assert len(buffers) == 1 and len(kernels) == 1, (len(buffers), len(kernels))
    (kernel,) = kernels
    assert kernel.max_tokens_per_rank == max_tokens
    router = kernel.token_comm._router
    offset = kernel._device_workspace.offset(router.sizes_by_rank_region)
    width = router.expert_count_padded
    matrix = buffers[0].shared_workspace[offset : offset + _WORLD * width * 4]
    return matrix.view(torch.int32).view(_WORLD, width)[rank, : router.expert_count].cpu().clone()


def _drop_worker(real_requests_by_rank: tuple, combine_format: str, routing: str) -> None:
    try:
        _drop_worker_impl(real_requests_by_rank, combine_format, routing)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        patch.stopall()  # the pool's workers serve the module's other cases


@contextmanager
def _megamoe_layer(
    combine_format: str, routing: str, rows: int = _REQUESTS * _ROWS_PER_REQUEST, adaptive=False
):
    """One MegaMoE CuteDSL layer on this rank, the same inputs and weights on every rank; with
    ``adaptive``, under an adaptive load balancer with two spare slots per rank, its routes mapped
    in the dispatch router (home route tables), and the balancer is yielded too."""
    from _torch.moe.moe_test_utils import MoeBackendType
    from _torch.moe.quantize_utils import FP8BlockScalesQuantizeUtil, get_test_quant_params
    from _torch.moe.test_moe_module import (
        _create_mapping_for_parallel_mode,
        _create_model_config,
        _ensure_dist_for_megamoe,
    )

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
    from tensorrt_llm._torch.moe.fused_moe import (
        DEFAULT_MOE_ACTIVATION,
        RenormalizeMoeRoutingMethod,
        create_moe,
    )
    from tensorrt_llm._torch.moe.fused_moe.interface import MoEWeightLoadingMode
    from tensorrt_llm._torch.moe.fused_moe.routing import DeepSeekV4MoeRoutingMethod
    from tensorrt_llm._torch.nccl_window_tensor_scope import nccl_window_tensor_scope
    from tensorrt_llm._torch.utils import ActivationType
    from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

    os.environ.pop("TRTLLM_FORCE_COMM_METHOD", None)
    os.environ["MEGAMOE_COMBINE_FORMAT"] = combine_format
    rank = mpi_rank()
    mapping = _create_mapping_for_parallel_mode(_WORLD, "DEP")
    mapping.rank = rank
    torch.cuda.set_device(rank)
    backend_name = MoeBackendType.MEGAMOE_CUTEDSL.value
    _ensure_dist_for_megamoe(backend_name, rank, _WORLD)
    shared = routing == _FUSED_SHARED
    dsv4 = routing == "dsv4" or shared
    num_experts, top_k, hidden_size, intermediate_size = _DSV4_SHAPE if dsv4 else _SHAPE
    dtype = torch.bfloat16

    with torch.device(f"cuda:{rank}"):
        torch.manual_seed(0)
        x = torch.randn((rows, hidden_size), dtype=dtype, device="cuda")
        if dsv4:
            bias = torch.randn(num_experts, device="cuda") * 0.1
            routing_method = DeepSeekV4MoeRoutingMethod(
                top_k=top_k,
                n_group=1,
                topk_group=1,
                routed_scaling_factor=1.5,
                callable_e_score_correction_bias=lambda: bias,
                callable_tid2eid=lambda: None,
                is_hashed=False,
            )
            router_logits = torch.randn((rows, num_experts), device="cuda")
        else:
            routing_method = RenormalizeMoeRoutingMethod(top_k=top_k, force_enable_pytorch_op=True)
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
        if shared:
            # DeepSeek-V4's single shared expert, fused under attention DP.
            model_cfg.pretrained_config.architectures = ["DeepseekV41ForCausalLM"]
            model_cfg.pretrained_config.n_shared_experts = 1
            model_cfg = model_cfg.copy_with(megamoe_fuse_shared_expert=True)
        balancer = contextlib.nullcontext()
        if adaptive:
            from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import (
                maybe_create_moe_load_balancer,
            )
            from tensorrt_llm.llmapi.llm_args import MoeAdaptiveLoadBalancerConfig

            model_cfg.moe_load_balancer = MoeAdaptiveLoadBalancerConfig(
                num_slots=num_experts + 2 * _WORLD
            )
            balancer = maybe_create_moe_load_balancer(model_cfg, mapping)
        with (
            nccl_window_tensor_scope(x),
            balancer as moe_balancer,
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
            if shared:
                shared_quant = QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES, group_size=128)
                weights = FP8BlockScalesQuantizeUtil(
                    num_experts=1,
                    dtype=dtype,
                    intermediate_size=_SHARED_INTERMEDIATE,
                    hidden_size=hidden_size,
                    quant_config=shared_quant,
                ).create_weights(use_e8m0_scale=True)
                mlp = GatedMLP(
                    hidden_size=hidden_size,
                    intermediate_size=_SHARED_INTERMEDIATE,
                    bias=False,
                    dtype=dtype,
                    config=ModelConfig(mapping=mapping, quant_config=shared_quant),
                    overridden_tp_size=1,
                    reduce_output=False,
                )
                mlp.gate_up_proj.load_weights(
                    [
                        {k: weights[f"0.{w}.{k}"] for k in ("weight", "weight_scale")}
                        for w in ("w1", "w3")
                    ]
                )
                mlp.down_proj.load_weights(
                    [{k: weights[f"0.w2.{k}"] for k in ("weight", "weight_scale")}]
                )
                # Before the Linear transforms, as DeepseekV4ForCausalLM.post_load_weights does.
                assert fused_moe.fuse_shared_expert(mlp)
            if not adaptive:
                yield fused_moe, x, router_logits, routing_method, rank
                return
            from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb.transport import (
                find_backends,
                prepare_layers,
            )

            moe_balancer.register_weight_slots_after_to_cuda()
            moe_balancer.finalize_model()
            prepare_layers(
                find_backends(fused_moe, moe_balancer.layers), moe_balancer.layers, x.device
            )
            yield fused_moe, x, router_logits, routing_method, rank, moe_balancer


def _drop_worker_impl(real_requests_by_rank: tuple, combine_format: str, routing: str) -> None:
    from tensorrt_llm._torch.moe.fused_moe.mega_moe import mega_moe_cute_dsl
    from tensorrt_llm._torch.utils import model_extra_attrs

    rows = _REQUESTS * _ROWS_PER_REQUEST
    shared = routing == _FUSED_SHARED
    dsv4 = routing == "dsv4" or shared
    num_experts = (_DSV4_SHAPE if dsv4 else _SHAPE)[0]
    with _megamoe_layer(combine_format, routing) as (
        fused_moe,
        x,
        router_logits,
        routing_method,
        rank,
    ):
        word = live_rows.LiveRowWord()
        attrs = {live_rows.LIVE_ROW_WORD_ATTR: word}
        # The DeepSeek-V4 case must take the fused route+quant path, under the word.
        fused_words = []
        fused = mega_moe_cute_dsl.dsv4_route_nvfp4_quant

        def spy(*args, **kwargs):
            fused_words.append(kwargs.get("live_row_word"))
            return fused(*args, **kwargs)

        patcher = patch.object(mega_moe_cute_dsl, "dsv4_route_nvfp4_quant", spy)
        patcher.start()

        def forward(value: int) -> tuple:
            word.write(value)
            from tensorrt_llm._torch.moe.custom_ops import cute_dsl_megamoe_custom_op as op

            with (
                torch.inference_mode(),
                model_extra_attrs(attrs),
                _forward_kernels(op.Sm100MegaMoENvfp4Runner) as kernels,
            ):
                output = fused_moe.forward(
                    x, router_logits, all_rank_num_tokens=[rows] * _WORLD
                ).clone()
            torch.cuda.synchronize()
            return output, _published_counts(fused_moe, rank, rows, kernels)

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
        assert torch.equal(out_all, out_again), "two all-rows forwards differ"
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
        # The fused shared expert reads the same word: the dropped rows get no output of any expert.
        assert not out_drop[live:].any()
        if dsv4:
            assert fused_words and all(w is not None for w in fused_words), fused_words
        else:
            assert not fused_words


@requires_blackwell
@pytest.mark.skipif(torch.cuda.device_count() < _WORLD, reason=f"needs {_WORLD} GPUs")
@pytest.mark.threadleak(enabled=False)  # the pool's manager thread persists by design
@pytest.mark.parametrize("combine_format,routing", _CASES)
def test_megamoe_cutedsl_drops_padding_rows(
    moe_multi_gpu_executor, combine_format: str, routing: str
) -> None:
    results = list(
        moe_multi_gpu_executor.map(
            _drop_worker,
            [_REAL_REQUESTS] * _WORLD,
            [combine_format] * _WORLD,
            [routing] * _WORLD,
        )
    )
    assert results == [None] * _WORLD


def _adaptive_shared_worker(real_requests_by_rank: tuple) -> None:
    try:
        _adaptive_shared_worker_impl(real_requests_by_rank)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        patch.stopall()


def _adaptive_shared_worker_impl(real_requests_by_rank: tuple) -> None:
    from _torch.moe.test_megamoe_adaptive_route_e2e import _fixed_order, _order_bound, _pick_terms

    from tensorrt_llm._torch.utils import model_extra_attrs

    rows, rank = _REQUESTS * _ROWS_PER_REQUEST, mpi_rank()
    real = real_requests_by_rank[rank]
    requests = [SimpleNamespace(is_dummy=index >= real) for index in range(_REQUESTS)]
    seq_lens = torch.full((_REQUESTS,), _ROWS_PER_REQUEST, dtype=torch.int32)
    value = live_rows.forward_word(requests, num_contexts=0, seq_lens=seq_lens)
    live = live_rows.live_row_count(value, rows)
    word = live_rows.LiveRowWord()

    def forward(layer, x, logits, value) -> torch.Tensor:
        word.write(value)
        with torch.inference_mode(), model_extra_attrs({live_rows.LIVE_ROW_WORD_ATTR: word}):
            output = layer.forward(x, logits, all_rank_num_tokens=[rows] * _WORLD).clone()
        torch.cuda.synchronize()
        return output

    with (
        _megamoe_layer("bf16", _FUSED_SHARED) as (plain, x, logits, routing_method, _),
        _megamoe_layer("bf16", _FUSED_SHARED, adaptive=True) as (adaptive, *_, balancer),
    ):
        stats = balancer.statistics(x.device)
        stats["storage"].zero_()
        balancer.observe_scheduled_batch(real, True, False, lambda start, stop: (0, 0, 0))
        got = forward(adaptive, x, logits, value)
        balancer.observe_scheduled_batch(0, True, False, lambda start, stop: (0, 0, 0))
        want = forward(plain, x, logits, value)
        with torch.inference_mode():
            routes, _ = routing_method.apply(logits)
        picks = torch.bincount(routes[:live].long().flatten(), minlength=_DSV4_SHAPE[0]).float()
        # The adaptive kernel maps and counts the live rows' routes; the padding rows add nothing.
        assert torch.equal(stats["rows"][0].cpu(), picks.cpu()), "the router counted other routes"
        # The fused shared expert reads the word in the adaptive build too: the dropped rows are zero.
        assert not got[live:].any() and not want[live:].any()
        if _fixed_order(adaptive.backend, rows, plain.backend):
            assert torch.equal(got[:live], want[:live])
            return

        # Another tactic or an arrival-order sum: within the order bound of the k + 1 terms, the
        # shared expert's (no pick kept) and each pick's (less the shared expert's).
        def run():
            return [forward(plain, x, logits, value)]

        (shared,) = _pick_terms([plain.backend], run, [None])
        terms = [[t[0] - shared[0]] for t in _pick_terms([plain.backend], run)] + [shared]
        bound = _order_bound(terms, 0)[:live]
        assert bool(((got[:live].float() - want[:live].float()).abs() <= bound).all())


@requires_blackwell
@pytest.mark.skipif(torch.cuda.device_count() < _WORLD, reason=f"needs {_WORLD} GPUs")
@pytest.mark.threadleak(enabled=False)  # the pool's manager thread persists by design
def test_an_adaptive_route_keeps_the_fused_shared_experts_live_rows(moe_multi_gpu_executor) -> None:
    """A layer whose fused shared expert bounds its rows by the live-row word, under an adaptive
    load balancer whose routes its dispatch router maps: its live rows return what the same layer
    without the balancer returns, its padding rows are zero, and its statistics count the live
    rows' routes alone."""
    results = list(moe_multi_gpu_executor.map(_adaptive_shared_worker, [_REAL_REQUESTS] * _WORLD))
    assert results == [None] * _WORLD
