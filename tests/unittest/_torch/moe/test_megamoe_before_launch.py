# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression: the fused MegaMoE kernel ran alongside the EPLB statistic allreduce and hung.

Both MegaMoE backends run one layer under online EPLB on four ranks with multi-stream on; every
forward must issue the statistic-allreduce wait once and before every fused launch.
"""

import pytest
import torch

from tensorrt_llm._torch.utils import ActivationType

_WORLD = 4
_FORWARDS = 8
# backend -> (quantization, rows per rank). CuteDSL's rows stay below the 64-row bucket, so it
# stages its inputs (an exact-size bucket may read them in place).
_EPLB_BACKENDS = {
    "MEGAMOE_DEEPGEMM": ("W4A8_MXFP4_MXFP8", 64),
    "MEGAMOE_CUTEDSL": ("NVFP4", 60),
}
_STAGING_OPS = ("megamoe_prepare", "megamoe_stage_inputs", "megamoe_stage_routes")

if torch.cuda.is_available():
    import pickle
    import sys

    import cloudpickle

    # The 4-GPU MoE cases' shared pool; test_moe_module queries the device when imported.
    from _torch.moe.test_moe_module import moe_multi_gpu_executor  # noqa: F401
    from mpi4py import MPI

    # The pool's workers get this module's worker by value.
    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)


def _eplb_worker(backend_name: str) -> dict:
    import gc
    import traceback
    from unittest.mock import patch

    try:
        return _eplb_worker_impl(backend_name)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        patch.stopall()
        # Regression: a second load balancer in one process hit a GDR bus error when this case's
        # buffers were freed after it opened its GDRCopy handle; free them before the next case.
        gc.collect()


def _eplb_worker_impl(backend_name: str) -> dict:
    from unittest.mock import patch

    from _torch.moe.moe_test_utils import MoeBackendType, MoeModelConfig
    from _torch.moe.quantize_utils import get_test_quant_params
    from _torch.moe.test_moe_module import (
        _create_mapping_for_parallel_mode,
        _create_model_config,
        _create_moe_load_balancer,
        _ensure_dist_for_megamoe,
    )

    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream
    from tensorrt_llm._torch.moe.custom_ops import cute_dsl_megamoe_custom_op as op
    from tensorrt_llm._torch.moe.fused_moe import (
        DEFAULT_MOE_ACTIVATION,
        RenormalizeMoeRoutingMethod,
        create_moe,
    )
    from tensorrt_llm._torch.moe.fused_moe.interface import MoEWeightLoadingMode
    from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import MoeLoadBalancerIterContext
    from tensorrt_llm._torch.nccl_window_tensor_scope import nccl_window_tensor_scope
    from tensorrt_llm._utils import mpi_rank
    from tensorrt_llm.models.modeling_utils import QuantAlgo

    quant_name, rows = _EPLB_BACKENDS[backend_name]
    model = MoeModelConfig(8, 2, 512, 512)
    num_slots, dtype = 16, torch.bfloat16
    backend = MoeBackendType[backend_name]
    rank = mpi_rank()
    mapping = _create_mapping_for_parallel_mode(_WORLD, "DEP")
    mapping.rank = rank
    torch.cuda.set_device(rank)
    _ensure_dist_for_megamoe(backend.value, rank, _WORLD)
    with torch.device(f"cuda:{rank}"):
        torch.manual_seed(0)
        routing_method = RenormalizeMoeRoutingMethod(top_k=model.top_k)
        x = torch.randn((rows, model.hidden_size), dtype=dtype, device="cuda")
        router_logits = torch.randn((rows, model.num_experts), dtype=dtype, device="cuda")
        quantize_util_cls, quant_config, quant_kwargs = get_test_quant_params(
            QuantAlgo[quant_name], x, backend
        )
        quant_kwargs.pop("ref_cls", None)
        quantize_util = quantize_util_cls(
            num_experts=model.num_experts,
            dtype=dtype,
            intermediate_size=model.intermediate_size,
            hidden_size=model.hidden_size,
            quant_config=quant_config,
            bias=False,
            swiglu_gptoss_style=False,
            swiglu_alpha=None,
            swiglu_beta=None,
            swiglu_limit=None,
            num_local_experts=model.num_experts // mapping.moe_ep_size,
            activation_type=ActivationType.Swiglu,
        )
        model_cfg = _create_model_config(
            num_experts=model.num_experts,
            hidden_size=model.hidden_size,
            intermediate_size=model.intermediate_size,
            dtype=dtype,
            mapping=mapping,
            quant_config=quant_config,
            moe_backend=backend.value,
            enable_eplb=True,
            num_slots=num_slots,
            layer_updates_per_iter=1,
            max_num_tokens=256,
        )
        balancer = _create_moe_load_balancer(model_cfg, True)
        with (
            nccl_window_tensor_scope(x),
            balancer,
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
            if QuantAlgo[quant_name] == QuantAlgo.W4A8_MXFP4_MXFP8:  # backend-layout weights
                weights, _, _ = quantize_util.prepare_weights_from_backend(
                    fused_moe, **quant_kwargs
                )
            else:
                weights = quantize_util.create_weights(**quant_kwargs)
            fused_moe.load_weights(
                [{k: v.to("cpu") if isinstance(v, torch.Tensor) else v for k, v in weights.items()}]
            )
            fused_moe.post_load_weights()
            fused_moe.cuda(f"cuda:{rank}")
            balancer.register_weight_slots_after_to_cuda()
            balancer.finalize_model()
            balancer.set_iter_info(enable_statistic=True, enable_update_weights=True)

            # What each forward enqueues, in host order (the order on the launch stream).
            trace = []

            def recorded(name, function):
                def call(*args, **kwargs):
                    trace.append(name)
                    return function(*args, **kwargs)

                return call

            wait = fused_moe._load_balancer_wait_for_statistic_allreduce

            fused_moe._load_balancer_wait_for_statistic_allreduce = recorded("wait", wait)
            impl = fused_moe.backend
            impl.run_moe = recorded("run_moe", impl.run_moe)
            if backend_name == "MEGAMOE_DEEPGEMM":
                impl._launch_kernel = recorded("launch", impl._launch_kernel)
            else:
                patch.object(
                    op,
                    "run_cute_dsl_megamoe_nvfp4_blackwell",
                    recorded("launch", op.run_cute_dsl_megamoe_nvfp4_blackwell),
                ).start()
            # Read by name: an attribute named ``ops`` next to the global ``torch`` makes
            # cloudpickle ship torch.ops, which is not a module, as a submodule of this worker.
            trtllm_ops = getattr(torch, "ops").trtllm
            for name in _STAGING_OPS:
                if hasattr(trtllm_ops, name):
                    patch.object(
                        trtllm_ops, name, recorded("stage", getattr(trtllm_ops, name))
                    ).start()

            counts = [rows] * _WORLD
            traces, outputs = [], []
            with torch.inference_mode(), with_multi_stream(True):
                for i in range(_FORWARDS + 1):
                    trace.clear()
                    with MoeLoadBalancerIterContext(balancer):
                        out = fused_moe.forward(x, router_logits, all_rank_num_tokens=counts)
                    torch.cuda.synchronize()
                    if i:  # the first forward primes the kernels
                        traces.append(list(trace))
                        outputs.append(out.float())
            # Stop the balancer as an executor's shutdown does: a weight-update thread left to the
            # finalizer would copy placements into buffers this case frees (bus error next case).
            assert balancer.shutdown(), "the load balancer did not shut down"

    return dict(
        rank=rank,
        takes_before_launch=bool(getattr(impl, "takes_before_launch", False)),
        traces=traces,
        finite=all(bool(torch.isfinite(out).all()) for out in outputs),
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < _WORLD,
    reason=f"needs {_WORLD} GPUs",
)
@pytest.mark.threadleak(enabled=False)  # the pool's manager thread persists by design
@pytest.mark.parametrize("backend_name", sorted(_EPLB_BACKENDS))
def test_online_eplb_statistic_wait_order(moe_multi_gpu_executor, backend_name):
    """DeepGEMM gets the wait before run_moe; CuteDSL issues it inside run_moe after its staging."""
    from _torch.moe.moe_test_utils import MoeBackendType, MoeModelConfig
    from _torch.moe.test_moe_module import (
        MEGAMOE_IGNORE_COMM_METHOD,
        should_skip_MegaMoECuteDsl,
        should_skip_MegaMoEDeepGemm,
    )

    from tensorrt_llm._torch.moe.fused_moe import RenormalizeMoeRoutingMethod
    from tensorrt_llm.models.modeling_utils import QuantAlgo

    skip = {
        "MEGAMOE_DEEPGEMM": should_skip_MegaMoEDeepGemm,
        "MEGAMOE_CUTEDSL": should_skip_MegaMoECuteDsl,
    }[backend_name]
    reason = skip(
        "DEP",
        MEGAMOE_IGNORE_COMM_METHOD,
        MoeBackendType[backend_name],
        QuantAlgo[_EPLB_BACKENDS[backend_name][0]],
        torch.bfloat16,
        MoeModelConfig(8, 2, 512, 512),
        RenormalizeMoeRoutingMethod,
        False,
    )
    if reason:
        pytest.skip(reason)
    results = list(moe_multi_gpu_executor.map(_eplb_worker, [backend_name] * _WORLD))
    for result in results:
        assert result["finite"], result["rank"]
        assert len(result["traces"]) == _FORWARDS
        for trace in result["traces"]:
            assert trace.count("wait") == 1 and "launch" in trace, trace
            waited = trace.index("wait")
            assert "launch" not in trace[:waited], trace
            if not result["takes_before_launch"]:
                assert trace[0] == "wait" and trace[1] == "run_moe", trace
            else:
                assert trace.index("run_moe") < waited, trace
                assert "stage" in trace[:waited] and "stage" not in trace[waited:], trace
