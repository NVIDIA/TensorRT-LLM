# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the adaptive EPLB balancer and of the hooks the MoE modules call on it."""

import json
import multiprocessing
import os
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from _torch.moe import adaptive_eplb_reference as reference

import tensorrt_llm
from tensorrt_llm._torch.moe.fused_moe import moe_load_balancer as base
from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb import balancer as module
from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb import calibrate
from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb.balancer import AdaptiveMoeLoadBalancer
from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb.policy import (
    ExpertStatistics,
    TimeModel,
    build_route_tables,
    statistics_buffers,
)
from tensorrt_llm.llmapi.llm_args import MoeAdaptiveLoadBalancerConfig

pytestmark = pytest.mark.cpu_only

NUM_EXPERTS, NUM_SLOTS, EP_SIZE, TOP_K = 8, 12, 4, 2
ASSIGNMENTS = [(2 * r + j) % 8 for r in range(4) for j in range(3)]  # rank r, slot j < 3


@pytest.fixture(autouse=True)
def cpu_ops(monkeypatch):
    """The route and statistics ops on CPU tensors: their Python versions."""
    monkeypatch.setattr(torch.ops.trtllm, "adaptive_eplb_route_into", reference.route_into, False)
    monkeypatch.setattr(torch.ops.trtllm, "adaptive_eplb_record", reference.record, False)


def _balancer(ep_rank: int = 1, **fields) -> AdaptiveMoeLoadBalancer:
    config = MoeAdaptiveLoadBalancerConfig(num_slots=NUM_SLOTS, **fields)
    config.setup(ep_rank=ep_rank, ep_size=EP_SIZE)
    return AdaptiveMoeLoadBalancer(ep_rank=ep_rank, ep_size=EP_SIZE, config=config, mapping=None)


def _observe(balancer, generation: int, context: int = 0, warmup: bool = False) -> None:
    """The executor's call before ``generation`` generation and ``context`` context requests."""
    balancer.observe_scheduled_batch(generation, not context, warmup, lambda start, stop: (0, 0, 0))


def _counts(*values: float) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)


def test_a_layer_built_outside_the_balancer_registers_nothing():
    """``unbalanced_layer`` hides the balancer while a layer outside it is built, drops that layer's
    repeat count and restores the balancer, also on error. A layer that does not fit is refused."""
    balancer = _balancer()
    with balancer:
        balancer.set_repeated_for_next_layer(3)
        with balancer.unbalanced_layer():
            assert base.get_moe_load_balancer() is None
        assert base.get_moe_load_balancer() is balancer
        layer = balancer.add_layer(NUM_EXPERTS, TOP_K, NUM_SLOTS // EP_SIZE)
        assert layer.repeated_count == 1
        assert balancer.layers == [layer]
        with pytest.raises(KeyError), balancer.unbalanced_layer():
            raise KeyError("construction failed")
        assert base.get_moe_load_balancer() is balancer
    with pytest.raises(RuntimeError, match="only the current"):
        with balancer.unbalanced_layer():
            pass
    with pytest.raises(ValueError, match="multiple of `ep_size`"):
        balancer.add_layer(6, TOP_K, NUM_SLOTS // EP_SIZE)
    with pytest.raises(ValueError, match="local slots"):
        balancer.add_layer(NUM_EXPERTS, TOP_K, 2)


@pytest.mark.parametrize(
    "arch,ep_size,cluster_size,error",
    [
        ("LlamaForCausalLM", 4, 1, "does not support model architecture"),
        ("DeepseekV3ForCausalLM", 1, 1, "requires expert parallelism"),
        ("DeepseekV3ForCausalLM", 4, 2, "is incompatible with smart routing"),
    ],
)
def test_factory_refuses_what_adaptive_eplb_cannot_balance(arch, ep_size, cluster_size, error):
    mapping = SimpleNamespace(moe_ep_rank=0, moe_ep_size=ep_size, moe_cluster_size=cluster_size)
    model_config = SimpleNamespace(
        mapping=mapping,
        pretrained_config=SimpleNamespace(architectures=[arch]),
        moe_load_balancer=MoeAdaptiveLoadBalancerConfig(num_slots=NUM_SLOTS),
    )
    with pytest.raises(ValueError, match=f"adaptive EPLB {error}"):
        base.maybe_create_moe_load_balancer(model_config, mapping)


def test_statistics_record_nothing_while_the_flag_is_clear():
    off = torch.zeros(1, dtype=torch.int32)
    stats = ExpertStatistics.of(statistics_buffers(1, 4, "cpu"), 0, off)
    stats.record(torch.tensor([[0, 1], [1, 3]], dtype=torch.int32))
    assert not stats.rows.any() and stats.forwards.item() == 0


def test_layer_route_stages_into_the_backend_and_records_the_statistics():
    balancer = _balancer()
    layer = balancer.add_layer(NUM_EXPERTS, TOP_K, NUM_SLOTS // EP_SIZE)
    bucket = SimpleNamespace(
        topk_idx=torch.full((4, TOP_K), 9, dtype=torch.int64),
        topk_weights=torch.full((4, TOP_K + 1), 9.0),
        bucket=4,
        live_row_word=None,
    )
    asked = []

    def staging(num_tokens: int, top_k: int, device: torch.device):
        asked.append((num_tokens, top_k))
        return bucket if num_tokens < 4 else None

    layer.prepare_device_state(torch.int64, "cpu", staging)
    _observe(balancer, 2)
    experts = torch.tensor([[7, 0], [0, 5]], dtype=torch.int32)
    weights = torch.tensor([[0.5, 0.25], [1.0, 2.0]])
    layer.update_statistic_with_local_ids(experts, True, True, allreduce=object())
    slots = layer.route(experts, token_final_scales=weights)
    # The slots fill the bucket's first rows and -1 / 0 the other rows, the columns after top_k
    # stay untouched, and the same launch adds the ids to the statistics.
    assert slots.data_ptr() == bucket.topk_idx.data_ptr()
    assert slots.tolist() == [[10, 0], [0, 7]]
    assert bucket.topk_idx[2:].tolist() == [[-1, -1], [-1, -1]]
    assert torch.equal(bucket.topk_weights[:2, :TOP_K], weights)
    assert (bucket.topk_weights[2:, :TOP_K] == 0).all()
    assert (bucket.topk_weights[:, TOP_K] == 9).all()
    first = _counts(2, 0, 0, 0, 0, 1, 0, 1)
    assert torch.equal(layer.statistics.rows, first) and layer.statistics.forwards.item() == 1
    assert asked == [(2, TOP_K)]
    # When the backend declines (a forward it does not stage), the route returns its own tensor;
    # without the weights the backend is not asked.
    four = torch.tensor([[1, 2]] * 4, dtype=torch.int32)
    own = layer.route(four, token_final_scales=torch.ones(4, TOP_K))
    assert own.data_ptr() != bucket.topk_idx.data_ptr() and own.tolist() == [[1, 3]] * 4
    called = []
    plain = layer.route(experts, offset_by_ep_rank=True, overlap_fn=lambda: called.append(1))
    assert called == [1] and plain.dtype == torch.int64 and plain.tolist() == [[10, 0], [0, 7]]
    assert asked == [(2, TOP_K), (4, TOP_K)]
    # The statistics of ids other than the routed ones are added before the route.
    layer.update_local_statistic(four, True, True)
    layer.route(experts)
    second = first + _counts(0, 4, 4, 0, 0, 0, 0, 0)
    assert torch.equal(layer.statistics.rows, second)
    # Under the staging's live-row word (1 real of 2 requests) the padding row is staged as -1 / 0
    # and not counted, as the kernel drops it; a record outside a route counts every row.
    bucket.live_row_word = torch.tensor([(2 << 16) | 1], dtype=torch.int32)
    layer.update_statistic_with_local_ids(experts, False, True)
    assert layer.route(experts, token_final_scales=weights).tolist() == [[10, 0], [-1, -1]]
    padded = torch.tensor([[0.5, 0.25]] + [[0.0] * 2] * 3)
    assert torch.equal(bucket.topk_weights[:, :TOP_K], padded)
    layer.update_local_statistic(experts, False, True)
    layer.route(four)
    assert torch.equal(layer.statistics.rows, second + _counts(1, 0, 0, 0, 0, 0, 0, 1) + first)
    assert layer.statistics.forwards.item() == 2
    # Installed tables route the forwards enqueued after them: a replica of expert 1 on slot 5
    # takes all of its rows; an id outside the experts goes to slot num_slots.
    layout, shares = layer.home_layout(), [1.0] * NUM_SLOTS
    layout[5], shares[1] = 1, 0.0
    tables = build_route_tables(layout, shares, NUM_EXPERTS, NUM_SLOTS).packed()
    balancer.route_tables("cpu")[layer.layer_idx].copy_(torch.tensor(tables, dtype=torch.int32))
    ids = torch.tensor([[1, 1], [9, -1]], dtype=torch.int32)
    assert layer.route(ids).tolist() == [[5, 5], [NUM_SLOTS, NUM_SLOTS]]


FIELDS = dict(zip(TimeModel.__dataclass_fields__, (1.0, 2.0, 3.0, 4.0, 0.9, 8)))


def _prepare(monkeypatch, path=None, tokens=(8, 16), domain=(None, 1), others=(), **fake):
    """``prepare_runtime`` of rank ``fake["rank"]`` (0) of ``fake["layers"]`` (1) layers on fakes,
    under ``time_model`` ``file`` with ``path``, else ``fake["mode"]`` (auto) and the config
    ``fake["config"]``; see the other ``fake`` keys below; the other ranks vote ``others`` unless
    ``fake["allgather"]`` does -> the balancer, its layer, what the start-up saw and the warnings."""
    monkeypatch.delenv("TRTLLM_CACHE_DIR", raising=False)
    mode = "file" if path else fake.get("mode", "auto")
    config = dict(time_model_file=path, time_model=mode, **fake.get("config", {}))
    balancer = _balancer(ep_rank=fake.get("rank", 0), **config)
    balancer.mapping = SimpleNamespace(gpus_per_node=4)
    layers = fake.get("layers", 1)
    layer = balancer.add_layer(NUM_EXPERTS, TOP_K, NUM_SLOTS // EP_SIZE) if layers else None
    seen = {}

    def time_model(expert_bytes, rates, tile_rows):
        seen["priced"] = (expert_bytes, rates)
        tiles = {
            k: FIELDS[k] * tile_rows / 8 for k in ("per_extra_tile_us", "hot_slot_extra_tile_us")
        }
        return dict(FIELDS, per_active_slot_us=4e12 / rates[0], tile_rows=tile_rows, **tiles)

    weights = {"w": torch.arange(8.0)}
    backend = SimpleNamespace(
        decode_time_model=time_model,
        decode_calibration_identity=lambda t: dict(
            top_k=2,
            tactics={str(n): tactic(n) for n in t if n <= fake.get("top", 8192)},
            tile_rows={str(n): fake.get("tiles", {}).get(n, 8) for n in t},
        ),
        slot_id_dtype=torch.int32,
        slot_weight_planes=lambda: [("w", 16)],
        get_parameter=weights.__getitem__,
    )

    def tactic(tokens):
        if fake.get("error"):
            raise fake["error"]
        if tokens > fake.get("top", 8192):
            raise RuntimeError(f"{tokens} tokens exceed the largest launch")
        return "a"

    def prepare_layers(backends, layers, device):
        # CPU tensors here: the route op runs as its Python version.
        for handle in layers:
            handle.prepare_device_state(backends[handle.layer_idx].slot_id_dtype, "cpu")

    transport = SimpleNamespace(allgather=fake.get("allgather", lambda vote: [vote, *others]))
    monkeypatch.setattr(
        module, "find_backends", lambda model, handles: {h.layer_idx: backend for h in handles}
    )
    monkeypatch.setattr(module, "prepare_layers", prepare_layers)
    monkeypatch.setattr(module, "PeerSlotTransport", lambda mapping: transport)
    monkeypatch.setattr(module, "nvlink_domain_error", lambda comm: domain)
    monkeypatch.setattr(module, "peer_copy_gap", lambda mapping: fake.get("gap"))
    monkeypatch.setattr(module, "_plan_layer", lambda i, b: SimpleNamespace(expert_bytes=1000 + i))
    device = f"NVIDIA {fake.get('device', 'GB300')}"
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: device)
    monkeypatch.setattr(module, "get_sm_version", lambda: 103)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    gpu = SimpleNamespace(multi_processor_count=152)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index=None: gpu)
    monkeypatch.setattr(
        module, "migrate_model_weights", lambda *a: seen.update(moved=a) or "stream"
    )
    monkeypatch.setattr(module, "time_layer_copy", lambda *args: 2.0)
    monkeypatch.setattr(balancer, "start_runtime", lambda *args: seen.update(start=args))
    warnings = []
    monkeypatch.setattr(module.logger, "warning", warnings.append)
    balancer.prepare_runtime("model", list(tokens))
    return balancer, layer, seen, warnings


def _calibration(tmp_path, **identity) -> str:
    """A calibration file of buckets of 8 and 16 tokens per GPU for the fakes of :func:`_prepare`."""
    mine = dict(device="NVIDIA GB300", sm=103, sm_count=152, ep_size=EP_SIZE, hosts=1, top_k=2)
    mine.update(num_slots=NUM_SLOTS, tile_rows={"8": 8, "16": 8})
    doc = dict(zip(("schema", "version"), module.CALIBRATION))
    doc.update(identity=dict(dict(mine, tactics={"8": "a", "16": "a"}), **identity))
    doc.update(rates=dict(hbm_bytes_per_s=4e12, fp4_dense_flops_per_s=1e16))
    doc.update(buckets=[dict(tokens=8, scale=0.8, per_active_slot_us=0, route_us=1.5)])
    doc["buckets"].append(dict(tokens=16, scale=1.1))  # without its own costs: the model's
    doc.update(measurement=dict(trtllm_version=tensorrt_llm.__version__))
    (tmp_path / "calibration.json").write_text(json.dumps(doc))
    return str(tmp_path / "calibration.json")


def test_prepare_runtime_moves_the_weights_and_leaves_an_uncached_calibration_to_the_warmup(
    monkeypatch,
):
    balancer, _, seen, warnings = _prepare(monkeypatch)
    assert "start" not in seen and not warnings and "CUDA graph warmup" in balancer.off_reason
    assert (balancer.transport, balancer.copy_stream) == (seen["moved"][1], "stream")
    _observe(balancer, 4, warmup=True)
    assert not balancer.recording and not warnings and balancer._pending is not None
    # Serving begins with no warmup having calibrated it (no CUDA graphs): off, said aloud, for good.
    _observe(balancer, 4)
    assert not balancer.recording and "no CUDA graph warmup" in warnings[0]
    balancer.before_capture()
    assert "start" not in seen


def test_a_calibration_file_prices_every_decode_bucket_at_its_own_scale(monkeypatch, tmp_path):
    path = _calibration(tmp_path, device="NVIDIA B200")  # no built-in calibration for it
    _, _, seen, warnings = _prepare(monkeypatch, path, device="B200")
    model = seen["start"][0]
    assert seen["priced"] == (1000, (4e12, 1e16)) and not warnings
    # A bucket's read cost from the calibration's held-out checks also solves its shares by time.
    assert model.scales == ((16, 0.8, 0.0, 1.5, True, 8, True), (32, 1.1, 1.0, 0.0, False, 8, True))
    assert model.route_us == 0.0 and not model.time_shares


@pytest.mark.parametrize(
    "identity, tokens, domain, others, gap, why",
    [
        (dict(ep_size=8), (8, 16), (None, 1), (), None, "ep_size"),
        ({}, (8, 32), (None, 1), (), None, "bucket 32"),
        (dict(tactics={"8": "a", "16": "b"}), (8, 16), (None, 1), (), None, "tactics of bucket 16"),
        ({}, (8, 16), ("no NVLink", 2), (), None, "no NVLink"),
        ({}, (8, 16), (None, 1), (), "2 nodes outside one MNNVL domain", "MNNVL domain"),
        ({}, (8, 16), (None, 1), [("rank 1 failed", "", 1, "", None)], None, "(EP rank 1)"),
        ({}, (8, 16), (None, 1), [(None, "another file", 1, "", None)], None, "different files"),
    ],
)
def test_adaptive_eplb_stays_off_on_every_rank_when_a_check_fails(
    monkeypatch, tmp_path, identity, tokens, domain, others, gap, why
):
    """No weights move, no runtime starts; the layers route by home layouts, recording nothing."""
    path = _calibration(tmp_path, **identity)
    balancer, layer, seen, warnings = _prepare(monkeypatch, path, tokens, domain, others, gap=gap)
    assert "moved" not in seen and "start" not in seen and why in warnings[0]
    assert balancer.runtime is None and why in balancer.off_reason
    _observe(balancer, 4)
    assert not balancer.recording
    experts = torch.tensor([[1, 2], [7, 4]], dtype=torch.int32)
    layer.update_statistic_with_local_ids(experts, True, True)
    # Expert e is home on slot e // 2 * 3 + e % 2 of the 3 slots per rank.
    assert layer.route(experts).tolist() == [[1, 3], [10, 6]]
    assert layer.statistics.forwards.item() == 0
    assert balancer.shutdown() is True


def test_a_decode_bucket_beyond_the_largest_launch_needs_no_calibration(monkeypatch, tmp_path):
    """The backend's identity leaves out a bucket no single launch takes; the others start."""
    balancer, _, seen, warnings = _prepare(
        monkeypatch, _calibration(tmp_path), tokens=(8, 16, 10240)
    )
    assert balancer.off_reason is None and "start" in seen and not warnings


def test_without_a_time_model_adaptive_eplb_stays_off(monkeypatch):
    balancer, _, seen, warnings = _prepare(monkeypatch, mode="off")
    assert "moved" not in seen and "time_model is off" in balancer.off_reason and warnings
    balancer.before_capture()  # nothing to calibrate
    with pytest.raises(ValueError, match="time_model_file"):
        _balancer(time_model_file="calibration.json")


@pytest.mark.parametrize(
    "broken",
    [
        lambda doc: [doc],
        lambda doc: dict(doc, identity=dict(doc["identity"], tactics=["a"])),
        lambda doc: dict(doc, measurement="a version"),
        lambda doc: dict(doc, buckets=[]),
        lambda doc: dict(doc, rates=dict(doc["rates"], hbm_bytes_per_s=0)),
        lambda doc: dict(doc, buckets=[dict(tokens=float("inf"), scale=0.8)]),
        lambda doc: dict(doc, buckets=[dict(tokens=8, scale=0.8, route_us=-1.0)]),
    ],
)
def test_a_file_of_another_layout_ends_in_the_vote(monkeypatch, tmp_path, broken):
    path = Path(_calibration(tmp_path))
    path.write_text(json.dumps(broken(json.loads(path.read_text()))))
    balancer, _, seen, warnings = _prepare(monkeypatch, str(path), tokens=())
    assert "start" not in seen and str(path) in balancer.off_reason and warnings


def _vote(rank, path, error, barrier, folder):
    """Rank ``rank`` of the test below, run from this file in a new process raising ``error``."""

    def allgather(vote):
        (folder / f"vote{rank}").write_text(json.dumps(vote))
        barrier.wait()
        return [tuple(json.loads((folder / f"vote{r}").read_text())) for r in range(2)]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(torch.ops.trtllm, "adaptive_eplb_route_into", reference.route_into, False)
        mp.setattr(torch.ops.trtllm, "adaptive_eplb_record", reference.record, False)
        balancer, *_ = _prepare(mp, path, rank=rank, allgather=allgather, error=error)
        (folder / f"off{rank}").write_text(balancer.off_reason)


@pytest.mark.parametrize("error", [None, ArithmeticError("an error of any type")])
def test_every_rank_votes_whatever_one_rank_raises(monkeypatch, tmp_path, error):
    """Rank 1 of two processes reads a broken file or raises ``error``: both vote, none waits."""
    good, bad = _calibration(tmp_path), tmp_path / "broken.json"
    bad.write_text(json.dumps(dict(json.loads(Path(good).read_text()), measurement="v1")))
    for name in [key for key in os.environ if key.startswith(("OMPI_", "PMIX_", "PMI_"))]:
        monkeypatch.delenv(name)  # each rank starts MPI alone, not as a rank of this process's job
    context = multiprocessing.get_context("spawn")  # a fork would share this process's MPI state
    barrier = context.Barrier(2, timeout=120)
    votes = [(0, good, None), (1, good if error else str(bad), error)]
    run = "import runpy; runpy.run_path(test)['_vote'](*vote, barrier, folder)"
    spawn = [dict(test=__file__, vote=v, barrier=barrier, folder=tmp_path) for v in votes]
    ranks = [context.Process(target=exec, args=(run, scope)) for scope in spawn]
    for process in ranks:
        process.start()
    for process in ranks:
        process.join(300)
    assert [process.exitcode for process in ranks] == [0, 0]
    assert all("(EP rank 1)" in (tmp_path / f"off{rank}").read_text() for rank in range(2))


def test_a_model_without_a_balanced_layer_serves_with_adaptive_eplb_off(monkeypatch):
    """No balanced layer, or ranks balancing different numbers: the ranks vote, nothing starts."""
    balancer, _, seen, _ = _prepare(monkeypatch, layers=0)
    assert balancer.off_reason == f"{module.NO_LAYER} (EP rank 0)" and "start" not in seen
    assert balancer.shutdown() is True
    balancer, *_ = _prepare(monkeypatch, others=[(None, "", 2, "", None)])
    assert balancer.off_reason == "the EP ranks balance different MoE layers or decode kernels"


@pytest.mark.parametrize(
    "views, why",
    [
        ([(True, "a", None), (True, "a", None)], None),
        ([(True, "a", ("c", 1)), (True, "b", ("c", 1))], None),
        ([(True, "a", ("c", 1)), (True, "b", ("c", 2))], "NVLink fabric"),
        ([(True, "a", None), (True, "b", None)], "NVLink fabric"),
        ([(True, "a", None), (False, "a", None)], "all links up"),
    ],
)
def test_the_ep_group_must_be_one_nvlink_domain(views, why):
    got, hosts = module.nvlink_domain_error(SimpleNamespace(allgather=lambda own: views))
    assert hosts == len({host for _, host, _ in views})
    assert got is None if why is None else why in got


def test_the_nvml_fabric_query_fills_this_gpus_view(monkeypatch):
    sent, done = [], module.pynvml.NVML_GPU_FABRIC_STATE_COMPLETED

    def fabric_info(handle, info):  # the real pynvml structure, filled as NVML fills it
        info._obj.clusterUuid[:2], info._obj.cliqueId, info._obj.state = (7, 9), 5, done

    monkeypatch.setattr(module.pynvml, "nvmlInit", lambda: None)
    monkeypatch.setattr(module.pynvml, "nvmlDeviceGetHandleByIndex", lambda index: "gpu")
    monkeypatch.setattr(module.pynvml, "nvmlDeviceGetGpuFabricInfoV", fabric_info)
    monkeypatch.setattr(module.MnnvlMemory, "supports_mnnvl", staticmethod(lambda: True))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    module.nvlink_domain_error(SimpleNamespace(allgather=lambda own: sent.append(own) or [own]))
    assert sent[0][0] is True and sent[0][2] == ("0709" + "00" * 14, 5)


@pytest.mark.parametrize("read", [0.0, 2.0, 4.0])
def test_a_read_cost_is_fitted_where_the_held_out_designs_confirm_it(read):
    rng = np.random.default_rng(0)  # 96 designs of 8 GPUs: slots with rows, other terms, zeros
    terms = np.dstack([rng.integers(1, 17, (96, 8)), rng.uniform(20, 140, (96, 8))]) @ np.eye(2, 4)
    y = 50.0 + 0.9 * (read * terms[..., 0] + terms[..., 1]).max(1) + rng.normal(0, 0.5, 96)
    theta, fitted, own = calibrate.read_cost(terms, np.r_[4.0, 1, 0, 0], y)
    assert theta == pytest.approx(read, abs=0.1) and (fitted < own or theta == 4.0)


def test_a_slot_read_is_priced_where_the_designs_resolve_no_row_cost():
    # DEP4, 100 slots per GPU, top-6, tile 256: a slot read costs 5 us, a row 0.01 us, noise 1 us.
    tile, unit = 256, np.r_[0.0, 256, 1, 128]  # the default: no prior read cost
    for t in range(1, 17):
        g = SimpleNamespace(ep=4, M=100, T=t, K=6, tile=tile)
        xs = calibrate.make_designs(g, 128, np.random.default_rng([4, t]))
        noise = np.random.default_rng([9, t]).normal(0.0, 1.0, len(xs))
        y = 60.0 + (5.0 * (xs > 0).sum(2) + 0.01 * xs.sum(2)).max(1) + noise
        terms = calibrate.model_terms(xs, tile)
        theta = calibrate.read_cost(terms, unit, y)[0]
        p = (terms @ np.r_[theta, unit[1:]]).max(1)
        scale = calibrate.fit_bucket(p, y, np.random.default_rng([4, t, 1]), draws=20)[0]
        assert scale * theta == pytest.approx(5.0, abs=0.3), t


def test_a_bucket_is_refused_when_its_held_out_designs_disagree():
    measure = calibrate
    rng = np.random.default_rng(0)
    p = rng.uniform(40.0, 160.0, 128)
    y = 70.0 + 0.9 * p + rng.normal(0.0, 0.5, 128)
    scale, se, z = measure.fit_bucket(p, y, np.random.default_rng(1), draws=200)
    assert abs(scale - 0.9) < 3 * se and abs(z) <= 3.0
    y[3::4] += 0.2 * p[3::4]  # the held-out designs scale differently
    assert abs(measure.fit_bucket(p, y, np.random.default_rng(1), draws=200)[2]) > 3.0


def test_start_runtime_needs_torch_distributed(monkeypatch):
    balancer = _balancer(ep_rank=0)
    balancer.add_layer(NUM_EXPERTS, TOP_K, NUM_SLOTS // EP_SIZE)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    with pytest.raises(RuntimeError, match="needs torch.distributed"):
        balancer.start_runtime(TimeModel(1.0, 1.0, 1.0, 1.0, 1.0, 8), 4)
    assert balancer.runtime is None


def test_forwards_count_on_the_device_without_iter_context():
    balancer = _balancer(ep_rank=0)
    layers = [balancer.add_layer(NUM_EXPERTS, TOP_K, NUM_SLOTS // EP_SIZE) for _ in range(2)]
    for layer in layers:
        layer.set_initial_weight_assignments(ASSIGNMENTS)
        layer.prepare_device_state(torch.int32, "cpu")
    balancer.finalize_model()
    experts = torch.tensor([[1, 2]], dtype=torch.int32)

    def forward():
        for layer in layers:
            layer.update_statistic_with_local_ids(experts, True, True)
            layer.route(experts)

    def forwards() -> list[int]:
        return [layer.statistics.forwards.item() for layer in layers]

    _observe(balancer, 4)
    with base.MoeLoadBalancerIterContext(balancer):
        forward()
    forward()  # the first replay after a graph capture runs without the context
    with base.MoeLoadBalancerIterContext(balancer):
        forward()
    assert forwards() == [3, 3]
    # Warmup, context and mixed batches are not recorded: the flag the op reads is cleared.
    for generation, context, warmup in ((4, 0, True), (0, 2, False), (4, 1, False)):
        _observe(balancer, generation, context, warmup)
        assert not balancer.recording
        forward()
    assert forwards() == [3, 3]
    _observe(balancer, 1)
    forward()
    assert forwards() == [4, 4]
    with pytest.raises(RuntimeError):
        balancer.end_iter()
    assert balancer.shutdown() is True


def test_a_moe_layer_runs_a_forward_through_the_load_balancer_hooks():
    from tensorrt_llm._torch.moe.fused_moe.impl_blocks import MoEEplbWeightLayoutMixin
    from tensorrt_llm._torch.moe.fused_moe.interface import MoE
    from tensorrt_llm._torch.moe.fused_moe.quantization import FusedMoEMethodBase

    balancer = _balancer(ep_rank=0)
    layer = balancer.add_layer(NUM_EXPERTS, TOP_K, NUM_SLOTS // EP_SIZE)
    # A MoE backend's load balancer hooks: the mixin's and the MoE layer's.
    hooks = {name: f for name, f in vars(MoE).items() if name.startswith("_load_balancer_")}
    state = dict(layer_load_balancer=layer, allreduce=None, _load_balancer_overlap_fn=None)
    module = type("Module", (MoEEplbWeightLayoutMixin,), {**hooks, **state})()
    assert isinstance(layer, base.MoeLayerLoadBalancer) and layer.get_load_expert_ids() == [0, 1]
    assert layer.home_layout() == [0, 1, -1, 2, 3, -1, 4, 5, -1, 6, 7, -1]
    assert not layer.is_dynamic_routing() and layer.records_statistics()
    assert layer.stages_routes and not layer.produces_replica_plan
    # Construction, then a backend's weight load, then its post-load.
    assert layer.get_repeat_count() == 1
    layer.set_initial_weight_assignments(ASSIGNMENTS)
    assert not FusedMoEMethodBase.need_load_shared_weights(module)
    module._add_raw_shared_weights_for_unmap([torch.zeros(4)])
    layer.set_initial_weight_assignments(ASSIGNMENTS)
    balancer.register_weight_slots_after_to_cuda()
    balancer.finalize_model()
    layer.prepare_device_state(torch.int32, "cpu")
    _observe(balancer, 1)
    # One forward of two chunks.
    experts = torch.tensor([[0, 5], [7, 7]], dtype=torch.int32)
    for first, last in ((True, False), (False, True)):
        module._load_balancer_start_wait_gpu_stage(first)
        module._load_balancer_done_wait_gpu_stage(first)
        module._load_balancer_update_statistic(experts, first, last)
        slots = module._load_balancer_route(experts, True)
        module._load_balancer_start_set_cpu_stage(last)
        module._load_balancer_done_set_cpu_stage(last)
    assert slots.tolist() == [[0, 7], [10, 10]]
    assert torch.equal(layer.statistics.rows, _counts(2, 0, 0, 0, 0, 2, 0, 4))
    assert layer.statistics.forwards.item() == 1


def _timer(read=6.0, held_out=1.0, touch=None, tiles=None, off=(8, 16)):
    """A ``calibrate.LiveTimer`` stand-in: the times of a model with read cost ``read`` at scale
    0.01 plus noise, every fourth design of the buckets ``off`` ``held_out`` times slower, a route
    cost of 2 us, a bucket's tile ``tiles[tokens]`` rows (8); it calls ``touch()`` while it times."""

    class Timer:
        made = []

        def __init__(self, backend, rank, num_experts, settings, gather):
            self.reps, self.rng, self.host_us = settings.reps, np.random.default_rng(0), 1.0
            Timer.made.append(self)

        def designs(self, tokens, top_k, xs):
            tile = (tiles or {}).get(tokens, 8)
            unit = np.r_[read, FIELDS["per_extra_tile_us"] * tile / 8, 3, 4 * tile / 8]
            truth = (calibrate.model_terms(xs, tile) @ unit).max(1)
            truth[3::4] *= held_out if tokens in off else 1.0
            (touch or (lambda: None))()
            # Noise of 0.002 us: the fitted read cost then spreads by about 0.04 over noise draws,
            # well inside the tests' 5% (0.3); at 0.02 it spread by 0.33.
            return 5.0 + 0.01 * truth + self.rng.normal(0.0, 0.002, (self.reps, len(xs)))

        def fixed(self, tokens, top_k, xs):
            parts = np.zeros((self.reps, len(xs), 3)) + [0.0, 0.0, 2.0]
            return parts + self.rng.normal(0.0, 0.01, parts.shape)

    return Timer


SMALL = dict(calibration_designs=32, calibration_fixed_designs=4)


def test_a_miss_calibrates_in_the_warmup_and_caches_it_for_the_next_start(monkeypatch, tmp_path):
    timer = _timer()
    monkeypatch.setattr(calibrate, "LiveTimer", timer)
    config = dict(SMALL, calibration_cache_dir=str(tmp_path))
    balancer, _, seen, warnings = _prepare(monkeypatch, config=config)
    assert "start" not in seen and not timer.made and not list(tmp_path.iterdir())
    balancer.before_capture()
    model, *_ = seen["start"]
    assert balancer.off_reason is None and len(timer.made) == 1 and not warnings
    assert [read for _, _, read, *_ in model.scales] == pytest.approx([6.0, 6.0], rel=0.05)
    assert [scale for _, scale, *_ in model.scales] == pytest.approx([0.01, 0.01], rel=0.05)
    assert model.route_us == pytest.approx(2.0, abs=0.05) and model.time_shares
    # The cache, and beside it every bucket's design times to replay the fit with.
    cached, times = sorted(tmp_path.iterdir(), key=lambda p: p.name.endswith(".times.json"))
    buckets = {str(b["tokens"]) for b in json.loads(cached.read_text())["buckets"]}
    assert set(json.loads(times.read_text())) == buckets
    # The next start of the same run takes the cached calibration and calibrates nothing.
    balancer, _, seen, _ = _prepare(monkeypatch, config=config)
    balancer.before_capture()
    assert seen["start"][0] == model and balancer.off_reason is None and len(timer.made) == 1
    # Another identity (here its decode buckets) misses: its file is another one.
    balancer, _, seen, _ = _prepare(monkeypatch, tokens=(8,), config=config)
    balancer.before_capture()
    assert (
        "start" in seen and len(timer.made) == 2 and len(list(tmp_path.glob("*[0-9a-f].json"))) == 2
    )


def test_buckets_of_different_tiles_calibrate_and_price_each_at_its_own(monkeypatch, tmp_path):
    tiles = {8: 8, 16: 16}  # tokens per GPU: rows of the MMA tile of the bucket's tactic
    monkeypatch.setattr(calibrate, "LiveTimer", _timer(tiles=tiles))
    balancer, _, seen, _ = _prepare(monkeypatch, config=SMALL, tiles=tiles)
    balancer.before_capture()
    model, *_ = seen["start"]
    assert [bucket[5] for bucket in model.scales] == [8, 16]
    assert [read for _, _, read, *_ in model.scales] == pytest.approx([6.0, 6.0], rel=0.05)
    small, large = model.at(16), model.at(32)  # rows per GPU: tokens times top_k
    assert (small.tile_rows, large.tile_rows) == (8, 16)
    assert large.per_extra_tile_us == pytest.approx(2 * small.per_extra_tile_us)


def test_a_bucket_failing_its_hold_out_check_has_no_model_and_the_others_serve(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(calibrate, "LiveTimer", _timer(held_out=1.5, off=(16,)))
    config = dict(SMALL, calibration_cache_dir=str(tmp_path))
    balancer, _, seen, warnings = _prepare(monkeypatch, config=config)
    balancer.before_capture()
    model, *_ = seen["start"]
    assert [trusted for *_, trusted in model.scales] == [True, False]
    assert (
        len(warnings) == 1
        and "adaptive EPLB bucket 16" in warnings[0]
        and "held False" in warnings[0]
    )
    (cached,) = tmp_path.glob("*[0-9a-f].json")
    assert [b["held"] for b in json.loads(cached.read_text())["buckets"]] == [True, False]
    assert not model.at(32).trusted and model.at(16).trusted


def test_every_bucket_failing_its_hold_out_check_keeps_adaptive_eplb_off(monkeypatch, tmp_path):
    monkeypatch.setattr(calibrate, "LiveTimer", _timer())
    monkeypatch.setattr(calibrate, "fit_bucket", lambda p, y, rng: (0.01, 0.001, 9.0))  # z beyond 3
    config = dict(SMALL, calibration_cache_dir=str(tmp_path))
    balancer, _, seen, warnings = _prepare(monkeypatch, config=config)
    balancer.before_capture()
    assert "start" not in seen and "held-out designs" in balancer.off_reason and warnings
    assert [p.name.endswith(".times.json") for p in tmp_path.iterdir()] == [True]  # no cache


def test_an_unwritable_cache_keeps_the_calibration_in_memory(monkeypatch, tmp_path):
    monkeypatch.setattr(calibrate, "LiveTimer", _timer())
    (tmp_path / "taken").write_text("")  # a file where the cache directory would be
    config = dict(SMALL, calibration_cache_dir=str(tmp_path / "taken"))
    balancer, _, seen, warnings = _prepare(monkeypatch, config=config)
    balancer.before_capture()
    assert "start" in seen and "in memory only" in warnings[0]


@pytest.mark.parametrize("touched", ["route_tables", "statistics", "weights"])
def test_the_calibration_must_leave_the_weights_and_the_balancer_state_unchanged(
    monkeypatch, tmp_path, touched
):
    tensors = {}
    monkeypatch.setattr(calibrate, "LiveTimer", _timer(touch=lambda: tensors[touched].add_(1)))
    monkeypatch.setattr(calibrate, "DIGEST_WORDS", 2)  # the weights by their digest
    balancer, _, seen, _ = _prepare(monkeypatch, config=SMALL)
    tensors.update(
        route_tables=balancer.route_tables("cpu"), statistics=balancer._statistics["rows"]
    )
    tensors["weights"] = balancer._pending[0].get_parameter("w")
    with pytest.raises(RuntimeError, match="changed the weights or balancer state"):
        balancer.before_capture()
    assert "start" not in seen


def test_a_state_change_on_another_rank_raises_on_every_rank_after_the_vote(monkeypatch):
    def allgather(vote):  # EP rank 1 found its state changed; this rank's is not
        after = isinstance(vote, tuple) and len(vote) == 2 and type(vote[0]) is bool
        return [vote, (True, None) if after else vote]

    monkeypatch.setattr(calibrate, "LiveTimer", _timer())
    balancer, _, seen, _ = _prepare(monkeypatch, config=SMALL, allgather=allgather)
    with pytest.raises(RuntimeError, match=r"balancer state: \[1\]"):
        balancer.before_capture()
    assert "start" not in seen


def test_a_snapshot_widens_one_chunk_of_a_large_tensor_at_a_time(monkeypatch):
    live, peak, widen = set(), [0], torch.Tensor.long

    def long(self, *args, **kwargs):  # the int64 chunks alive at once
        out = widen(self, *args, **kwargs)
        live.add(id(out))
        weakref.finalize(out, live.discard, id(out))
        peak[0] = max(peak[0], len(live))
        return out

    monkeypatch.setattr(torch.Tensor, "long", long)
    monkeypatch.setattr(calibrate, "DIGEST_WORDS", 4)
    calibrate.snapshot([torch.arange(64, dtype=torch.int32)])  # 16 chunks
    assert 0 < peak[0] <= 2


@pytest.mark.parametrize("scale", [0.0, -0.5, float("nan"), float("inf")])
def test_a_bucket_of_a_degenerate_scale_has_no_model_and_the_others_serve(monkeypatch, scale):
    fits = iter([(0.01, 0.001, 0.0), (scale, 0.001, 0.0)])  # the buckets of 8 and 16 tokens
    monkeypatch.setattr(calibrate, "fit_bucket", lambda p, y, rng: next(fits))
    monkeypatch.setattr(calibrate, "LiveTimer", _timer())
    balancer, _, seen, _ = _prepare(monkeypatch, config=SMALL)
    balancer.before_capture()
    assert [trusted for *_, trusted in seen["start"][0].scales] == [True, False]


def test_a_file_bucket_out_of_range_has_no_model_and_the_others_serve(monkeypatch, tmp_path):
    path = Path(_calibration(tmp_path))
    doc = json.loads(path.read_text())
    doc["buckets"][0].update(scale=-1.0, route_us=float("inf"))
    path.write_text(json.dumps(doc))
    _, _, seen, _ = _prepare(monkeypatch, str(path))
    assert [trusted for *_, trusted in seen["start"][0].scales] == [False, True]


def test_a_cache_that_does_not_parse_is_stale_and_recalibrated(monkeypatch, tmp_path):
    timer = _timer()
    monkeypatch.setattr(calibrate, "LiveTimer", timer)
    config = dict(SMALL, calibration_cache_dir=str(tmp_path))
    _prepare(monkeypatch, config=config)[0].before_capture()
    (cached,) = tmp_path.glob("*[0-9a-f].json")
    doc = json.loads(cached.read_text())
    doc["identity"]["tile_rows"] = list(doc["identity"]["tile_rows"])  # a field of another type
    cached.write_text(json.dumps(doc))
    balancer, _, seen, _ = _prepare(monkeypatch, config=config)
    balancer.before_capture()
    assert "start" in seen and len(timer.made) == 2


def test_the_calibration_settings_leave_every_fit_its_points():
    MoeAdaptiveLoadBalancerConfig(
        num_slots=NUM_SLOTS, calibration_designs=12, calibration_fixed_designs=2, calibration_reps=1
    )
    with pytest.raises(ValueError, match="calibration_designs"):
        MoeAdaptiveLoadBalancerConfig(num_slots=NUM_SLOTS, calibration_designs=11)
    with pytest.raises(ValueError, match="at least 2"):
        MoeAdaptiveLoadBalancerConfig(
            num_slots=NUM_SLOTS, calibration_fixed_designs=1, calibration_reps=1
        )


def test_one_slow_pass_of_one_design_leaves_the_read_cost(monkeypatch):
    class Timer(_timer()):
        def designs(self, tokens, top_k, xs):
            y = super().designs(tokens, top_k, xs)
            y[0, 5] *= 15.0  # one stalled launch: one pass of one design 15 times slower
            return y

    monkeypatch.setattr(calibrate, "LiveTimer", Timer)
    balancer, _, seen, _ = _prepare(monkeypatch, config=SMALL)
    balancer.before_capture()
    model, *_ = seen["start"]
    assert [read for _, _, read, *_ in model.scales] == pytest.approx([6.0, 6.0], rel=0.05)


def test_forced_routes_realize_a_design_with_distinct_slots_per_token():
    g = SimpleNamespace(ep=4, M=5, T=6, K=3, tile=4)
    for x in calibrate.make_designs(g, 20, np.random.default_rng(0)):
        ids = np.stack([calibrate.forced_routes(x, rank, g.K) for rank in range(g.ep)])
        assert ids.shape == (g.ep, g.T, g.K)
        assert (np.bincount(ids.ravel(), minlength=x.size).reshape(x.shape) == x).all()
        assert (np.diff(np.sort(ids, -1), axis=-1) > 0).all()
