# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: the fused drafter hops (``drafter_hops``) on the published DFlash 2 drafter at TP8.

Run with eight MPI ranks (one per GPU) on two NVLink-connected four-GPU GB300
nodes, with ``MUSESPARK_REFERENCE_DRAFTER=<Muse-Spark-Open-DFlash2 checkpoint>``
and ``TRTLLM_ENABLE_PDL=1`` set on every rank, running
``python -m pytest <this file>``.

Anywhere else every test skips.

The drafter is built TP8 (``Mapping(tp_size=8)``, the default all-reduce
strategy AUTO, which is the MNNVL one-shot across two nodes) with its real
weights and the drafter fast paths ``drafter_block_conv``,
``drafter_block_decode`` and ``drafter_fixed_gemms``. One
request's arena holds a context of real-looking K/V (the drafter's own context
projection of random captured rows), and each step runs the block decode of
``K + 1 = 8`` rows (the decode width) on fresh noise. Every arm --
the built-in hops, ``drafter_hops`` without and with the early trigger -- must give the
same bits: the block decode's output and the block's K/V columns in the arena
(written by the attention of every layer, so they see every hop's output of
the layers before). Eagerly, under CUDA-graph replay (2,000 replays per arm
with new noise each, compared on the device), with eager forwards between
replays, and with a rank delayed before random replays.

``MODELING_V2_TIMING=1`` also times the decode graph per arm (alternating arm
order, max over ranks) -- the in-model drafter span.
"""

import os
import statistics

import pytest
import torch

try:
    from mpi4py import MPI
except ImportError:  # pragma: no cover - the test needs MPI ranks
    MPI = None

_DRAFTER_DIR = os.environ.get("MUSESPARK_REFERENCE_DRAFTER")


def _world():
    try:
        from tensorrt_llm._utils import mpi_comm

        comm = mpi_comm()
        return comm, comm.Get_size(), comm.Get_rank()
    except Exception:  # noqa: BLE001 - no MPI launch means no group
        return None, 1, 0


COMM, WORLD, RANK = _world()

pytestmark = pytest.mark.skipif(
    WORLD != 8
    or not _DRAFTER_DIR
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 3), (10, 7)),
    reason="needs 8 MPI ranks on sm_103 or sm_107 and MUSESPARK_REFERENCE_DRAFTER",
)

CTX = 3_000
BLOCK = 8
SLOT = 1
REPLAYS = 2_000
ARMS = ("builtin", "drafter_hops", "drafter_hops_early")


def _agree(ok: bool) -> bool:
    return COMM.allreduce(int(ok)) == WORLD


@pytest.fixture(scope="module")
def env():
    from safetensors.torch import load_file
    from transformers import AutoConfig

    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
        spec_worker as drafter_worker,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.drafter_block_decode import (  # noqa: E501
        block_decode as draft,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.drafter_hops import (  # noqa: E501
        hops as drafter_hops,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.allreduce.drafter_hop import (  # noqa: E501
        DrafterHopWorkspace,
    )
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM
    from tensorrt_llm.mapping import Mapping

    local = int(os.environ.get("SLURM_LOCALID", RANK % torch.cuda.device_count()))
    torch.cuda.set_device(local)
    mapping = Mapping(world_size=WORLD, rank=RANK, gpus_per_node=4, tp_size=WORLD)
    pretrained = AutoConfig.from_pretrained(_DRAFTER_DIR)
    with torch.device("cuda", local):
        model = DFlashForCausalLM(
            ModelConfig(pretrained_config=pretrained, mapping=mapping, attn_backend="TRTLLM")
        )
    model.load_weights(load_file(os.path.join(_DRAFTER_DIR, "model.safetensors")))
    assert model.has_block_conv and model.dflash_attention_backend == "VANILLA"
    draft.install(model, True, True)
    drafter_worker.install(model, None, mapping, False, False, True)
    norms = [layer.post_attention_layernorm for layer in model.model.layers]
    norms += [layer.input_layernorm for layer in model.model.layers[1:]]
    eps = sorted({float(n.variance_epsilon) for n in norms + [model.model.norm]})
    sites = 2 * len(model.model.layers)
    pre_wait = [tensor for _, tensor in drafter_hops.pre_wait_operands(model)]
    arms = {
        "builtin": None,
        "drafter_hops": drafter_hops.DrafterHopCall(
            DrafterHopWorkspace(mapping, sites=sites, eps=eps, pre_wait=pre_wait)
        ),
        "drafter_hops_early": drafter_hops.DrafterHopCall(
            DrafterHopWorkspace(
                mapping, sites=sites, eps=eps, early_trigger=True, pre_wait=pre_wait
            )
        ),
    }

    g = torch.Generator(device="cuda").manual_seed(1234)
    captured = (
        torch.randn(
            CTX, pretrained.hidden_size * len(model.target_layer_ids), generator=g, device="cuda"
        )
        * 2.0
    ).to(torch.bfloat16)
    with torch.inference_mode():
        k, v = model.precompute_context_kv(
            model.project_target_hidden(captured), torch.arange(CTX, device="cuda")
        )
    # Slot SLOT holds the served-shape request; slots 1 and 2 the two-request shapes.
    shape = (3, model._num_attn_layers, CTX + 256, model._num_kv_heads, model._head_dim)
    state = dict(
        model=model,
        arms=arms,
        pool_k=torch.zeros(shape, dtype=torch.bfloat16, device="cuda"),
        pool_v=torch.zeros(shape, dtype=torch.bfloat16, device="cuda"),
        noise=torch.empty(1, BLOCK, pretrained.hidden_size, dtype=torch.bfloat16, device="cuda"),
        positions=torch.arange(CTX, CTX + BLOCK, device="cuda").unsqueeze(0),
        num_ctx=torch.tensor([CTX], device="cuda"),
        batch_idx=torch.tensor([SLOT], device="cuda"),
        hidden=pretrained.hidden_size,
        mapping=mapping,
    )
    for slot in (1, 2):
        state["pool_k"][slot, :, :CTX] = k.permute(1, 0, 2, 3)
        state["pool_v"][slot, :, :CTX] = v.permute(1, 0, 2, 3)
    COMM.Barrier()
    return state


def _set_arm(env, arm):
    model = env["model"]
    hop = env["arms"][arm]
    if hop is None:
        model.__dict__.pop("_drafter_hop", None)
    else:
        model._drafter_hop = hop


def _forward(env):
    return env["model"].dflash_forward(
        noise_embedding=env["noise"],
        query_positions=env["positions"],
        num_ctx_per_req=env["num_ctx"],
        ctx_k_cache=env["pool_k"],
        ctx_v_cache=env["pool_v"],
        ctx_cache_batch_idx=env["batch_idx"],
    )


def _block_kv(env):
    return (
        env["pool_k"][SLOT, :, CTX : CTX + BLOCK].clone(),
        env["pool_v"][SLOT, :, CTX : CTX + BLOCK].clone(),
    )


def _noise(env, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(1, BLOCK, env["hidden"], generator=g, device="cuda").to(torch.bfloat16)


def _eager(env, arm, seed):
    _set_arm(env, arm)
    env["noise"].copy_(_noise(env, seed))
    env["pool_k"][SLOT, :, CTX : CTX + BLOCK].fill_(float("nan"))
    env["pool_v"][SLOT, :, CTX : CTX + BLOCK].fill_(float("nan"))
    calls = 0 if env["arms"][arm] is None else env["arms"][arm].calls
    out = _forward(env).clone()
    hops = 0 if env["arms"][arm] is None else env["arms"][arm].calls - calls
    return (out, *_block_kv(env)), hops


def _bits_equal(a, b) -> bool:
    return all(torch.equal(x.view(torch.int16), y.view(torch.int16)) for x, y in zip(a, b))


@torch.inference_mode()
def test_every_arm_is_the_builtin_hops_bit_for_bit_in_eager_decode(env):
    bad = []
    for seed in range(12):
        want, hops = _eager(env, "builtin", seed)
        assert hops == 0
        for arm in ARMS[1:]:
            got, hops = _eager(env, arm, seed)
            if hops != 10:
                bad.append(f"{arm} seed {seed}: {hops} fused hops, not 10")
            if not _bits_equal(got, want):
                diff = int((got[0].view(torch.int16) != want[0].view(torch.int16)).sum())
                bad.append(f"{arm} seed {seed}: {diff} output values differ")
    torch.cuda.synchronize()
    assert _agree(not bad), f"rank {RANK}: {bad[:6]}"


@torch.inference_mode()
def test_graph_replays_of_every_arm_match_the_builtin_over_2000_steps(env):
    """Each arm's decode captured once, replayed REPLAYS times with new noise; on-device compare."""
    graphs, outs = {}, {}
    for arm in ARMS:
        _set_arm(env, arm)
        _forward(env)
        torch.cuda.synchronize()
        COMM.Barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs[arm] = _forward(env)
        graphs[arm] = graph
    _set_arm(env, "builtin")
    g = torch.Generator(device="cuda").manual_seed(99)
    pool = torch.randn(64, 1, BLOCK, env["hidden"], generator=g, device="cuda").to(torch.bfloat16)
    mismatches = {arm: torch.zeros((), dtype=torch.int64, device="cuda") for arm in ARMS[1:]}
    host = torch.Generator().manual_seed(RANK * 7 + 1)
    shared = torch.Generator().manual_seed(5)
    for step in range(REPLAYS):
        env["noise"].copy_(pool[step % 64])
        env["noise"][0, step % BLOCK, step % env["hidden"]] += float(step % 17)
        graphs["builtin"].replay()
        want = (outs["builtin"].clone(), *_block_kv(env))
        for arm in ARMS[1:]:
            late = int(torch.randint(0, 2 * WORLD, (1,), generator=shared))
            if late == RANK and step % 5 == 0:
                torch.cuda._sleep(int(torch.randint(1_000, 400_000, (1,), generator=host)))
            graphs[arm].replay()
            got = (outs[arm], *_block_kv(env))
            for a, b in zip(got, want):
                mismatches[arm] += (a.view(torch.int16) != b.view(torch.int16)).sum()
        if step % 250 == 249:
            # Eager forwards of every arm between replays: the sites stay in order.
            for arm in ARMS[1:]:
                _eager(env, arm, 10_000 + step)
    torch.cuda.synchronize()
    counts = {arm: int(m) for arm, m in mismatches.items()}
    assert _agree(all(c == 0 for c in counts.values())), f"rank {RANK}: {counts}"


class _Shape:
    """Static inputs of a block decode of ``B`` requests x ``block`` rows (slots 1.., contexts CTX - 37 i)."""

    def __init__(self, env, B, block):
        self.env, self.B, self.block = env, B, block
        self.slots = list(range(1, 1 + B))
        self.ctx = [CTX - 37 * i for i in range(B)]
        self.noise = torch.empty(B, block, env["hidden"], dtype=torch.bfloat16, device="cuda")
        self.positions = torch.stack([torch.arange(c, c + block, device="cuda") for c in self.ctx])
        self.num_ctx = torch.tensor(self.ctx, device="cuda")
        self.batch_idx = torch.tensor(self.slots, device="cuda")

    @property
    def rows(self):
        return self.B * self.block

    def fill(self, seed):
        g = torch.Generator(device="cuda").manual_seed(seed)
        noise = torch.randn(self.B, self.block, self.env["hidden"], generator=g, device="cuda")
        self.noise.copy_(noise.to(torch.bfloat16))

    def forward(self):
        return self.env["model"].dflash_forward(
            noise_embedding=self.noise,
            query_positions=self.positions,
            num_ctx_per_req=self.num_ctx,
            ctx_k_cache=self.env["pool_k"],
            ctx_v_cache=self.env["pool_v"],
            ctx_cache_batch_idx=self.batch_idx,
        )

    def block_kv(self):
        ks = [
            self.env["pool_k"][s, :, c : c + self.block].clone()
            for s, c in zip(self.slots, self.ctx)
        ]
        vs = [
            self.env["pool_v"][s, :, c : c + self.block].clone()
            for s, c in zip(self.slots, self.ctx)
        ]
        return torch.stack(ks), torch.stack(vs)

    def eager(self, arm, seed):
        """(output, block K, block V) of one forward on ``arm``, and the fused hops it took."""
        _set_arm(self.env, arm)
        self.fill(seed)
        for s, c in zip(self.slots, self.ctx):
            self.env["pool_k"][s, :, c : c + self.block].fill_(float("nan"))
            self.env["pool_v"][s, :, c : c + self.block].fill_(float("nan"))
        hop = self.env["arms"][arm]
        calls = 0 if hop is None else hop.calls
        out = self.forward().clone()
        return (out, *self.block_kv()), (0 if hop is None else hop.calls - calls)


def _expected_hops(env, arm, shape) -> int:
    hop = env["arms"][arm]
    return 10 if hop is not None and hop.takes(shape.rows, shape.block, torch.bfloat16) else 0


def _bits_differing(a, b) -> int:
    return sum(int((x.view(torch.int16) != y.view(torch.int16)).sum()) for x, y in zip(a, b))


@torch.inference_mode()
def test_other_block_shapes_match_the_builtin_in_eager_decode(env):
    """B = 2 x block 8 (16 rows: the built-in path), B = 2 x block 4 (two requests per hop), B = 1 x block 4.

    Each forward is followed by a served-shape forward on the same arm, which
    must still take its 10 fused hops and match the built-in.
    """
    served = _Shape(env, 1, BLOCK)
    bad = []
    for B, block in ((2, 8), (2, 4), (1, 4)):
        shape = _Shape(env, B, block)
        for seed in range(8):
            want, _ = shape.eager("builtin", seed)
            for arm in ARMS[1:]:
                got, hops = shape.eager(arm, seed)
                if hops != _expected_hops(env, arm, shape):
                    bad.append(f"{arm} B{B}xb{block} seed {seed}: {hops} fused hops")
                if _bits_differing(got, want):
                    bad.append(
                        f"{arm} B{B}xb{block} seed {seed}: {_bits_differing(got, want)} differ"
                    )
                want8, _ = served.eager("builtin", 50 + seed)
                got8, hops8 = served.eager(arm, 50 + seed)
                if hops8 != 10 or _bits_differing(got8, want8):
                    bad.append(f"{arm} served after B{B}xb{block} seed {seed}: {hops8} hops")
    torch.cuda.synchronize()
    assert _agree(not bad), f"rank {RANK}: {bad[:6]}"


@torch.inference_mode()
def test_graph_replays_mixing_block_shapes_match_the_builtin(env):
    """One graph per (shape, arm), 600 replays of randomly drawn shapes with delayed ranks."""
    shapes = [_Shape(env, 1, BLOCK), _Shape(env, 2, 8), _Shape(env, 2, 4), _Shape(env, 1, 4)]
    graphs, outs = {}, {}
    for index, shape in enumerate(shapes):
        for arm in ARMS:
            _set_arm(env, arm)
            shape.fill(7)
            shape.forward()
            torch.cuda.synchronize()
            COMM.Barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outs[(index, arm)] = shape.forward()
            graphs[(index, arm)] = graph
    _set_arm(env, "builtin")
    mismatches = {arm: 0 for arm in ARMS[1:]}
    shared = torch.Generator().manual_seed(9)
    host = torch.Generator().manual_seed(100 + RANK)
    for step in range(600):
        index = int(torch.randint(0, len(shapes), (1,), generator=shared))
        shape = shapes[index]
        shape.fill(20_000 + step)
        graphs[(index, "builtin")].replay()
        want = (outs[(index, "builtin")].clone(), *shape.block_kv())
        for arm in ARMS[1:]:
            if int(torch.randint(0, 2 * WORLD, (1,), generator=shared)) == RANK:
                torch.cuda._sleep(int(torch.randint(1_000, 300_000, (1,), generator=host)))
            graphs[(index, arm)].replay()
            mismatches[arm] += _bits_differing((outs[(index, arm)], *shape.block_kv()), want)
        if step % 100 == 99:
            for arm in ARMS[1:]:
                shapes[0].eager(arm, 30_000 + step)
    torch.cuda.synchronize()
    _set_arm(env, "builtin")
    assert _agree(all(m == 0 for m in mismatches.values())), f"rank {RANK}: {mismatches}"


class _CaptureAbort(Exception):
    pass


@torch.inference_mode()
def test_a_capture_failing_inside_the_block_decode_gives_the_site_order_back(env):
    """The decode raises in layer 2's MLP hop while captured; the next forwards resume at site 0."""
    arm = "drafter_hops_early"
    hop = env["arms"][arm]
    served = _Shape(env, 1, BLOCK)
    _set_arm(env, arm)
    served.fill(3)
    served.forward()
    torch.cuda.synchronize()
    COMM.Barrier()

    def failing_mlp(layer_idx, *args):
        if layer_idx == 2:
            raise _CaptureAbort()
        return type(hop).mlp(hop, layer_idx, *args)

    hop.mlp = failing_mlp
    failed = torch.cuda.CUDAGraph()
    try:
        with pytest.raises(_CaptureAbort):
            with torch.cuda.graph(failed):
                served.forward()
    finally:
        del hop.mlp
    del failed
    stopped_at = hop.workspace.next_site
    bad = []
    for seed in range(4):
        want, _ = served.eager("builtin", 700 + seed)
        got, hops = served.eager(arm, 700 + seed)
        if hops != 10 or _bits_differing(got, want):
            bad.append(f"eager seed {seed}: {hops} hops, {_bits_differing(got, want)} differ")
    graphs, outs = {}, {}
    for name in ("builtin", arm):
        _set_arm(env, name)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs[name] = served.forward()
        graphs[name] = graph
    for step in range(50):
        served.fill(800 + step)
        graphs["builtin"].replay()
        want = (outs["builtin"].clone(), *served.block_kv())
        graphs[arm].replay()
        diff = _bits_differing((outs[arm], *served.block_kv()), want)
        if diff:
            bad.append(f"replay {step}: {diff} differ")
    torch.cuda.synchronize()
    _set_arm(env, "builtin")
    # Sites 0..4 (layers 0 and 1, layer 2's attention hop) were captured before the raise.
    assert _agree(stopped_at == 5 and not bad), f"rank {RANK}: stopped at {stopped_at}; {bad[:6]}"


@torch.inference_mode()
def test_install_refuses_pre_wait_parameters_it_cannot_read_as_they_are(env):
    """A non-contiguous finish base kernel is refused; a misaligned norm weight fails the workspace on every rank."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.drafter_hops import (  # noqa: E501
        hops as drafter_hops,
    )

    model = env["model"]
    _set_arm(env, "builtin")
    errors = []
    conv = model.mlp_convs[0]
    saved_base = conv.base_kernel
    strided = torch.empty(
        *saved_base.shape[:-1], 2 * saved_base.shape[-1], dtype=saved_base.dtype, device="cuda"
    )
    strided = strided[..., ::2]
    strided.copy_(saved_base)
    conv.base_kernel = torch.nn.Parameter(strided, requires_grad=False)
    try:
        drafter_hops.install(model)
        errors.append("a non-contiguous base kernel was installed")
    except ValueError as exc:
        if "MLP conv 0 finish base kernel is not contiguous" not in str(exc):
            errors.append(f"wrong refusal: {exc}")
    finally:
        conv.base_kernel = saved_base
        model.__dict__.pop("_drafter_hop", None)
    norm = model.model.norm
    saved_weight = norm.weight
    buffer = torch.empty(saved_weight.numel() + 8, dtype=saved_weight.dtype, device="cuda")
    shifted = buffer[1 : 1 + saved_weight.numel()]
    shifted.copy_(saved_weight)
    norm.weight = torch.nn.Parameter(shifted, requires_grad=False)
    try:
        drafter_hops.install(model)
        errors.append("a misaligned norm weight was installed")
    except RuntimeError as exc:
        if "failed on at least one rank" not in str(exc) or "16-byte aligned" not in str(exc):
            errors.append(f"wrong failure: {exc}")
    finally:
        norm.weight = saved_weight
        model.__dict__.pop("_drafter_hop", None)
    COMM.Barrier()
    assert _agree(not errors), f"rank {RANK}: {errors}"


@pytest.mark.skipif(os.environ.get("MODELING_V2_TIMING") != "1", reason="set MODELING_V2_TIMING=1")
@torch.inference_mode()
def test_microbenchmark_drafter_span(env):
    """The block decode (5 layers, 10 hop sites) as one CUDA graph per arm; us per forward."""
    graphs = {}
    for arm in ARMS:
        _set_arm(env, arm)
        _forward(env)
        torch.cuda.synchronize()
        COMM.Barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _forward(env)
        for _ in range(20):
            graph.replay()
        graphs[arm] = graph
    torch.cuda.synchronize()
    samples = {arm: [] for arm in ARMS}
    for rnd in range(4):
        order = ARMS if rnd % 2 == 0 else tuple(reversed(ARMS))
        for arm in order:
            for _ in range(5):
                COMM.Barrier()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(50):
                    graphs[arm].replay()
                end.record()
                end.synchronize()
                samples[arm].append(start.elapsed_time(end) * 1000.0 / 50)
    result = {}
    for arm in ARMS:
        result[arm] = COMM.allreduce(statistics.median(samples[arm]), op=MPI.MAX)
        result[arm + "_min"] = COMM.allreduce(min(samples[arm]), op=MPI.MAX)
    if RANK == 0:
        print("\ndrafter block decode (5 layers, 10 hop sites), us per forward, max over ranks:")
        for arm in ARMS:
            print(
                f"  {arm:8s} median {result[arm]:8.2f}  min {result[arm + '_min']:8.2f}  "
                f"delta vs builtin {result[arm] - result['builtin']:+7.2f}"
            )
    _set_arm(env, "builtin")
