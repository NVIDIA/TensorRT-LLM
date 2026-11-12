# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The dep16 target's MegaMoE routed experts (``kimi_k3_nvfp4__sm_107__dep16/megamoe.py``)
against the V1 fused-MoE path they replace (``create_moe`` -> ``ConfigurableMoE`` -> the MegaMoE
CuTe DSL impl), one process per GPU at EP4 with attention DP (4 GPUs of one node, SM100 / SM103 /
SM107).

One MoE layer of random checkpoint-format NVFP4 experts (E2M1 codes and power-of-two E4M3 block
scales drawn directly, weight global scales 1, per-expert power-of-two FC2 input scales) streams
expert by expert into both paths, which must build the same kernel operands bit for bit
(``mega_fc{1,2}_weight{,_sf}``, the alphas, ``fc1_norm_const``, the input scales).

Then, for every per-rank token count below (each rank's ``x`` padded to the largest count, as
attention DP pads), both paths run the same input and the same ``noaux_tc`` routing; the new
module's output must be V1's bit for bit (same kernel, same staging), with the routing given as a
``route`` callable, as a precomputed routing over the padded rows, and with ``x`` written late on
another stream (``input_ready_event``). V1 is first checked to repeat itself bit for bit.

* ``exact_bucket``: every rank holds exactly one bucket of tokens (on SM107 the generation-phase
  kernel takes the inputs in place; elsewhere they are staged).
* ``staged_adp``: uneven counts below the bucket, one rank with no tokens.
* ``top_bucket``: the top bucket, one rank exactly at it.

The first case runs inside ``autotune()``, where both paths prime the other buckets.

    pytest -s test_k3_megamoe.py          # a pool of 4 MPI workers
    mpirun -n 4 python3 test_k3_megamoe.py
"""

import importlib
import os
import pickle
import sys
import traceback
from contextlib import nullcontext

import pytest

torch = pytest.importorskip("torch")

try:
    import cloudpickle
    from mpi4py import MPI
except ImportError:  # the test is skipped below
    cloudpickle = MPI = None

if cloudpickle is not None:
    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

K3 = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl"
WORLD = 4
NUM_EXPERTS, TOP_K, HIDDEN, INTER = 64, 8, 1024, 512
MAX_NUM_TOKENS = 512
SITU_BETA, SITU_LINEAR_BETA = 4.0, 6.0
ROUTED_SCALING_FACTOR = 2.5
SV = 16
FP4_RANGE = 448.0 * 6.0
X_INPUT_SCALE = 4.0 / FP4_RANGE
W2_INPUT_SCALE = 16.0 / FP4_RANGE

# Per-rank token counts of each case.
CASES = {
    "exact_bucket": [256, 256, 256, 256],
    "staged_adp": [37, 0, 200, 129],
    "top_bucket": [300, 5, 512, 17],
}

MEGA_TENSORS = (
    "mega_fc1_weight",
    "mega_fc1_weight_sf",
    "mega_fc2_weight",
    "mega_fc2_weight_sf",
    "fc31_alpha",
    "fc2_alpha",
    "fc1_norm_const",
    "fc31_input_scale",
    "fc2_input_scale",
)


def _supported() -> bool:
    if MPI is None or not torch.cuda.is_available() or torch.cuda.device_count() < WORLD:
        return False
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor in (100, 103, 107)


pytestmark = pytest.mark.threadleak(enabled=False)


def _same_bits(a, b) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    a, b = a.contiguous(), b.contiguous()
    if a.element_size() == 2:
        a, b = a.view(torch.int16), b.view(torch.int16)
    elif a.element_size() == 4:
        a, b = a.view(torch.int32), b.view(torch.int32)
    return torch.equal(a, b.to(a.device))


def _init_dist(rank: int) -> None:
    """A NCCL WORLD group over the pool (the MegaMoE barrier-reset fence runs on WORLD)."""
    import torch.distributed as dist

    from tensorrt_llm._utils import mpi_comm

    if dist.is_initialized():
        return
    store = None
    if rank == 0:
        store = dist.TCPStore(
            host_name="127.0.0.1",
            port=0,
            world_size=WORLD,
            is_master=True,
            wait_for_workers=False,
        )
    port = mpi_comm().bcast(store.port if store is not None else None, root=0)
    if store is None:
        store = dist.TCPStore(
            host_name="127.0.0.1",
            port=port,
            world_size=WORLD,
            is_master=False,
            wait_for_workers=False,
        )
    dist.init_process_group(backend="nccl", store=store, rank=rank, world_size=WORLD)


def _rand_nvfp4(n: int, k: int, gen):
    """``(packed E2M1 uint8 [n, k / 2], E4M3 scales [n, k / 16])`` on the host."""
    codes = torch.randint(0, 16, (n, k), dtype=torch.uint8, device="cuda", generator=gen)
    exps = torch.randint(-2, 1, (n, k // SV), device="cuda", generator=gen).float()
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    return packed.cpu(), torch.exp2(exps).to(torch.float8_e4m3fn).cpu()


def _checkpoint():
    """Every expert's ``(w1, w3, w2)`` (gate, up, down), the same on every rank."""
    gen = torch.Generator(device="cuda").manual_seed(0)
    return [
        (
            _rand_nvfp4(INTER, HIDDEN, gen),
            _rand_nvfp4(INTER, HIDDEN, gen),
            _rand_nvfp4(HIDDEN, INTER, gen),
        )
        for _ in range(NUM_EXPERTS)
    ]


def _expert_tensors(ckpt, expert: int) -> dict:
    (w1, w1_sf), (w3, w3_sf), (w2, w2_sf) = ckpt[expert]
    one = torch.tensor(1.0, dtype=torch.float32)
    x_scale = torch.tensor(X_INPUT_SCALE, dtype=torch.float32)
    w2_scale = torch.tensor(W2_INPUT_SCALE * 2.0 ** -(expert % 2), dtype=torch.float32)
    return dict(
        w1_weight=w1,
        w1_weight_scale=w1_sf,
        w1_weight_scale_2=one,
        w1_input_scale=x_scale,
        w2_weight=w2,
        w2_weight_scale=w2_sf,
        w2_weight_scale_2=one,
        w2_input_scale=w2_scale,
        w3_weight=w3,
        w3_weight_scale=w3_sf,
        w3_weight_scale_2=one,
        w3_input_scale=x_scale,
    )


def _build(rank: int):
    """``(v1 layer, new layer, routing method)`` with the same checkpoint loaded into both."""
    # The V1 impl's switches at their defaults (the new module has them as constants).
    os.environ.pop("MEGAMOE_COMBINE_FORMAT", None)
    os.environ.pop("MEGAMOE_TACTIC_AUTOTUNE", None)
    os.environ["MEGAMOE_PRIME_LADDER"] = "1"
    from transformers.configuration_utils import PretrainedConfig

    megamoe = importlib.import_module(f"{K3}.kimi_k3_nvfp4__sm_107__dep16.modeling")
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.moe.fused_moe import ConfigurableMoE, SiTuActivation, create_moe
    from tensorrt_llm._torch.moe.fused_moe.routing import DeepSeekV3MoeRoutingMethod
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

    mapping = Mapping(
        world_size=WORLD,
        rank=rank,
        tp_size=WORLD,
        moe_ep_size=WORLD,
        moe_tp_size=1,
        enable_attention_dp=True,
    )
    gen = torch.Generator(device="cuda").manual_seed(1)
    bias = 0.1 * torch.randn(NUM_EXPERTS, device="cuda", generator=gen)
    routing_method = DeepSeekV3MoeRoutingMethod(
        top_k=TOP_K,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=ROUTED_SCALING_FACTOR,
        callable_e_score_correction_bias=lambda: bias,
        is_fused=True,
    )
    pretrained = PretrainedConfig()
    pretrained.num_experts = NUM_EXPERTS
    pretrained.hidden_size = HIDDEN
    pretrained.intermediate_size = INTER
    pretrained.torch_dtype = torch.bfloat16
    model_config = ModelConfig(
        pretrained_config=pretrained,
        mapping=mapping,
        quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4),
        moe_backend="MEGAMOE_CUTEDSL",
        max_num_tokens=MAX_NUM_TOKENS,
    )
    with torch.device(f"cuda:{rank}"):
        v1 = create_moe(
            routing_method=routing_method,
            num_experts=NUM_EXPERTS,
            hidden_size=HIDDEN,
            intermediate_size=INTER,
            dtype=torch.bfloat16,
            reduce_results=False,
            accepts_global_routed_output=True,
            model_config=model_config,
            layer_idx=0,
            communication_method=None,
            activation=SiTuActivation(gate_softcap=SITU_BETA, linear_softcap=SITU_LINEAR_BETA),
            allow_backend_degradation=False,
        )
        assert isinstance(v1, ConfigurableMoE), type(v1)
        assert type(v1.backend).__name__ == "TrtllmCutedslMegaMoeNvfp4Impl", type(v1.backend)
        v1.create_weights()
        new = megamoe.K3MegaMoE(
            num_experts=NUM_EXPERTS,
            top_k=TOP_K,
            hidden_size=HIDDEN,
            intermediate_size=INTER,
            mapping=mapping,
            layer_idx=0,
            situ_beta=SITU_BETA,
            situ_linear_beta=SITU_LINEAR_BETA,
            max_num_tokens=MAX_NUM_TOKENS,
        )

    ckpt = _checkpoint()
    impl, method = v1.backend, v1.backend.quant_method
    method.prepare_streaming_expert_load(impl)
    for slot, expert in enumerate(impl.initial_local_expert_ids):
        method.load_streaming_nvfp4_expert(
            impl, global_expert_id=expert, local_slot_id=slot, **_expert_tensors(ckpt, expert)
        )
    impl.process_weights_after_loading()

    backend = new.backend
    backend.prepare_streaming_load()
    for slot, expert in enumerate(backend.initial_local_expert_ids):
        backend.load_expert(
            global_expert_id=expert, local_slot_id=slot, **_expert_tensors(ckpt, expert)
        )
    backend.finalize()
    torch.cuda.synchronize()
    return v1, new, routing_method


def _row(case: str, check: str, ok: bool, detail: str = "") -> dict:
    return {"case": case, "check": check, "ok": bool(ok), "detail": detail}


def _check_weights(v1, new) -> list:
    impl, backend = v1.backend, new.backend
    rows = [
        _row(
            "weights",
            "experts",
            list(impl.initial_local_expert_ids) == list(backend.initial_local_expert_ids),
            f"v1={impl.slot_start}.. new={backend.slot_start}..",
        ),
        _row(
            "weights",
            "buckets",
            list(impl._maxt_buckets) == new.buckets,
            f"v1={impl._maxt_buckets} new={new.buckets}",
        ),
    ]
    for name in MEGA_TENSORS:
        rows.append(_row("weights", name, _same_bits(getattr(backend, name), getattr(impl, name))))
    return rows


def _check_case(case: str, v1, new, routing_method, rank: int, tune: bool) -> list:
    SortedRouting = importlib.import_module(f"{K3}.ops.moe.k3r_experts").SortedRouting
    from tensorrt_llm._torch.autotuner import autotune

    counts = CASES[case]
    num_tokens, rows_padded = counts[rank], max(counts)
    gen = torch.Generator(device="cuda").manual_seed(1000 + 10 * list(CASES).index(case) + rank)
    x = (torch.randn(rows_padded, HIDDEN, device="cuda", generator=gen) * 0.5).to(torch.bfloat16)
    logits = torch.randn(rows_padded, NUM_EXPERTS, device="cuda", generator=gen)

    def route(router_logits, local_expert_offset, local_num_experts):
        del local_expert_offset, local_num_experts
        ids, scales = routing_method.apply(router_logits)
        return SortedRouting(ids, scales, None)

    outputs = {}
    with autotune() if tune else nullcontext(), torch.inference_mode():
        # Clone at once: every launch of the bucket writes the same shared combine region.
        outputs["v1"] = v1(x, logits, all_rank_num_tokens=counts).clone()
        outputs["v1_again"] = v1(x, logits, all_rank_num_tokens=counts).clone()
        outputs["route"] = new(x, logits, route=route, all_rank_num_tokens=counts).clone()
        padded_routing = route(logits, 0, 0)
        outputs["routing"] = new(x, routing=padded_routing, all_rank_num_tokens=counts).clone()
        side, ready = torch.cuda.Stream(), torch.cuda.Event()
        x_late = torch.empty_like(x)
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            torch.cuda._sleep(20_000_000)
            x_late.copy_(x)
            ready.record()
        outputs["late_input"] = new(
            x_late, logits, route=route, all_rank_num_tokens=counts, input_ready_event=ready
        ).clone()
    torch.cuda.synchronize()

    want = outputs["v1"]
    bucket = new._select_bucket(max(counts))
    path = "direct" if num_tokens == bucket and bucket in new._direct_input_buckets else "staged"
    rows = [
        _row(case, "shape", tuple(want.shape) == (num_tokens, HIDDEN), str(tuple(want.shape))),
        _row(
            case,
            "v1 non-trivial",
            num_tokens == 0
            or (bool(torch.isfinite(want.float()).all()) and want.float().abs().sum().item() > 0),
        ),
        _row(case, "v1 repeats", _same_bits(outputs["v1_again"], want)),
    ]
    for name in ("route", "routing", "late_input"):
        got = outputs[name]
        mismatched = (
            int((got.float() != want.float()).sum().item()) if got.shape == want.shape else -1
        )
        rows.append(
            _row(
                case,
                f"{name} == v1",
                _same_bits(got, want),
                f"rank {rank} n={num_tokens} bucket={bucket} {path}; mismatched={mismatched}",
            )
        )
    return rows


def _rank_job(cases):
    try:
        from tensorrt_llm._utils import mpi_rank

        rank = mpi_rank()
        torch.cuda.set_device(rank)
        _init_dist(rank)
        v1, new, routing_method = _build(rank)
        rows = _check_weights(v1, new)
        for i, case in enumerate(cases):
            rows += _check_case(case, v1, new, routing_method, rank, tune=i == 0)
        return rows
    except Exception:
        traceback.print_exc()
        raise


@pytest.mark.skipif(not _supported(), reason=f"needs {WORLD} SM100/SM103/SM107 GPUs and mpi4py")
@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
def test_k3_megamoe_matches_v1(mpi_pool_executor):
    per_rank = list(mpi_pool_executor.map(_rank_job, [list(CASES)] * WORLD))
    failures = []
    for rank, rows in enumerate(per_rank):
        for row in rows:
            print(f"rank={rank} {row}", flush=True)
            if not row["ok"]:
                failures.append((rank, row))
    assert not failures, failures


if __name__ == "__main__":
    results = _rank_job(list(CASES))
    bad = [row for row in results if not row["ok"]]
    for row in results:
        print(row, flush=True)
    sys.exit(1 if bad else 0)
