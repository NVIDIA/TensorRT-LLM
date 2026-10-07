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
"""Multi-GPU tests for Wan with parallel_config.tp_layout='token_sharded'.

Each case compares the token-sharded model against, with the same weights: the plain-TP model
whose all-reduces are replaced by the same reduce-scatter + all-gather (``_EmulatedAllReduce``;
the blocks' output must match bitwise), the single-GPU model (relative L2), and the plain-TP
model (only the collectives' reduction order differs).

Run with (needs >= 4 GPUs; the TP8 block case needs 8):
    pytest tests/unittest/_torch/visual_gen/multi_gpu/test_wan_token_sharded_tp.py -v
"""

import os

os.environ["TLLM_DISABLE_MPI"] = "1"

import functools
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.visual_gen.config import (
    AttentionConfig,
    DiffusionModelConfig,
    TorchCompileConfig,
)
from tensorrt_llm._torch.visual_gen.mapping import VisualGenMapping
from tensorrt_llm._torch.visual_gen.parallel.token_sharded_modules import (
    TokenShardedAdapter,
    TokenShardedRow,
    classify,
    convert_to_token_sharded_tp,
)
from tensorrt_llm._torch.visual_gen.parallel.token_sharded_tp import TokenShardedTP
from tensorrt_llm._utils import is_sm_100f
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo
from tensorrt_llm.visual_gen.args import ParallelConfig

from .test_wan_tp import (
    _WAN_I2V_TEST_CONFIG,
    _WAN_T2V_TEST_CONFIG,
    _copy_ref_weights_to_tp,
    _make_model_config,
    _stabilize_model_weights,
    run_test_in_distributed,
)


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


def _rel_l2(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _check(ok, msg):
    """Assert on every rank together (a rank-local failure must not strand the peers in
    the next collective); the failing rank's message is reported."""
    flag = torch.tensor([1 if ok else 0], dtype=torch.int32, device="cuda")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    assert ok and flag.item() == 1, (
        f"rank {dist.get_rank()}: {msg if not ok else 'another rank failed'}"
    )


class _EmulatedAllReduce(nn.Module):
    """A row-parallel Linear's all-reduce done as the helper's reduce-scatter + all-gather.

    Put into an all-reduce TP model, it makes that model reduce exactly like token-sharded TP, so
    the two must agree bitwise. Needs ``tp.begin(B, S)`` for the forward's shape.
    """

    def __init__(self, tp):
        super().__init__()
        self.tp = tp

    def uses_nccl_symmetric_memory_window(self):
        return False

    def forward(self, output, all_reduce_params=None):
        rows = self.tp.reduce_scatter(output.reshape(-1, output.shape[-1]))
        return self.tp.all_gather(rows).reshape(output.shape)


class _ExactAllReduce(nn.Module):
    """fp32 all-reduce of the bf16 partials, rounded once (a ~correctly rounded sum)."""

    def uses_nccl_symmetric_memory_window(self):
        return False

    def forward(self, output, all_reduce_params=None):
        y = output.float()
        dist.all_reduce(y)
        return y.to(output.dtype)


def _replace_all_reduce(model, make):
    """Replace the all-reduces that token-sharded TP turns into reduce-scatters (per
    ``classify``: row projections and MLP down-projections) with ``make()``."""
    for name, kind in classify(model).items():
        if kind not in ("row", "mlp"):
            continue
        module = model.get_submodule(name)
        linear = module.down_proj if kind == "mlp" else module
        assert linear.all_reduce is not None, name
        linear.all_reduce = make()


_WAN_BLOCK_CONVERSION = {
    "attn1.qkv_proj": "column",
    "attn1.to_out.0": "row",
    "attn2.to_q": "column",
    "attn2.to_out.0": "row",
    "ffn": "mlp",
}


def _check_wan_conversion(ts_model, ar_model):
    """The rules convert exactly these five modules of every Wan block (T2V and I2V: the
    image K/V projections read the image embeddings, not the stream), and nothing else."""
    want = {
        f"blocks.{i}.{name}": kind
        for i in range(len(ar_model.blocks))
        for name, kind in _WAN_BLOCK_CONVERSION.items()
    }
    assert classify(ar_model) == want
    converted = {n for n, m in ts_model.named_modules() if isinstance(m, TokenShardedAdapter)}
    assert converted == set(want)


def _capture_head_input(model):
    """Record the input of the output head (the transformer blocks' output)."""
    box = {}

    def hook(mod, args):
        box["x"] = args[0].detach().clone()

    model.norm_out.register_forward_pre_hook(hook)
    return box


def _use_eager_per_token_adaln(model):
    """Make an all-reduce Wan model run per-token AdaLN eagerly, as token-sharded TP does.

    The fused per-token AdaLN kernel (SM100, hidden size % 256 == 0) is an optimization of
    the all-reduce path only; token-sharded TP applies per-token modulation on the shard with the
    model's LayerNorm module, so the bitwise wiring check compares like with like.
    """
    for blk in model.blocks:
        blk._pertoken_adaln._eligible = False
    model._pertoken_adaln_runtime._runtime_key = None  # re-resolve eligibility


def _amplify_modulation(model, seed=11):
    """Make the per-block modulation and the block branches matter.

    ``_stabilize_model_weights`` makes every block nearly the identity (all Linears with
    gain ~0.01) and the modulation ~0.01, so a modulation mix-up changes the output by
    less than bf16 noise. Unit-scale AdaLN tables, a ~0.35-gain block and a time path
    that depends visibly on the timestep keep activations bounded but make a wrong
    table an O(10%) error.
    """
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for blk in model.blocks:
            table = blk.scale_shift_table
            table.copy_((torch.randn(table.shape, generator=gen) * 0.3).to(table.dtype))
            for p in blk.parameters():
                if p.ndim == 2:
                    p.mul_(30)
        emb = model.condition_embedder
        for lin in (emb.time_embedder.linear_1, emb.time_embedder.linear_2, emb.time_proj):
            lin.weight.mul_(50)


def _build_models(rank, world_size, config_dict):
    """Single-GPU reference, token-sharded model and plain-TP model with the same weights."""
    from tensorrt_llm._torch.visual_gen.models.wan.transformer_wan import WanTransformer3DModel

    device = torch.device(f"cuda:{rank}")
    torch.manual_seed(123)
    ref = WanTransformer3DModel(_make_model_config(config_dict, tp_size=1))
    ref = ref.to(device).to(torch.bfloat16)
    _stabilize_model_weights(ref)
    _amplify_modulation(ref)
    torch.manual_seed(123)
    ts_cfg = _make_model_config(config_dict, tp_size=world_size, tp_layout="token_sharded")
    ts_model = WanTransformer3DModel(ts_cfg).to(device).to(torch.bfloat16)
    _copy_ref_weights_to_tp(ref, ts_model, rank, world_size, config_dict)
    assert ts_model.sharder.token_sharded_tp
    assert isinstance(ts_model.blocks[0].attn1.to_out[0], TokenShardedRow)
    torch.manual_seed(123)
    ar_cfg = _make_model_config(config_dict, tp_size=world_size)
    ar_model = WanTransformer3DModel(ar_cfg).to(device).to(torch.bfloat16)
    _copy_ref_weights_to_tp(ref, ar_model, rank, world_size, config_dict)
    assert not ar_model.sharder.token_sharded_tp
    _check_wan_conversion(ts_model, ar_model)
    return ref, ts_model, ar_model


def _inputs(device, batch, thw, timestep, image_dim=None, seed=456):
    t, h, w = thw
    gen = torch.Generator(device="cpu").manual_seed(seed)
    inputs = dict(
        hidden_states=(torch.randn(batch, 16, t, h, w, generator=gen) * 0.1).to(
            device, torch.bfloat16
        ),
        timestep=timestep.to(device, torch.bfloat16),
        encoder_hidden_states=(torch.randn(batch, 8, 128, generator=gen) * 0.1).to(
            device, torch.bfloat16
        ),
    )
    if image_dim is not None:
        inputs["encoder_hidden_states_image"] = (
            torch.randn(batch, 4, image_dim, generator=gen) * 0.1
        ).to(device, torch.bfloat16)
    return inputs


# =============================================================================
# Model vs emulated all-reduce, plain TP and single GPU
# =============================================================================

# Token-sharded TP vs single GPU (rel-L2). A modulation mix-up must move the reference by more than
# _SENSITIVITY x this bound.
_SINGLE_GPU_REL_L2 = 2e-2
_SENSITIVITY = 10
# Token-sharded TP vs the real all-reduce TP model (reduction order of the collectives only).
_ALL_REDUCE_REL_L2 = 5e-3


def _logic_vs_single_gpu(
    rank,
    world_size,
    *,
    config_dict,
    batch,
    thw,
    timestep,
    image_dim=None,
    mixed_up_timestep=None,
    case="",
):
    """One case. ``mixed_up_timestep``: timesteps whose reference output stands for a
    modulation mix-up (per-sample timesteps swapped, per-token made uniform)."""
    device = torch.device(f"cuda:{rank}")
    ref, ts_model, ar_model = _build_models(rank, world_size, config_dict)
    inputs = _inputs(device, batch, thw, timestep, image_dim)
    with torch.no_grad():
        # The tp=1 reference is only trusted on rank 0: without a device mesh its Mapping
        # reports the global rank as tp_rank, so on other ranks the (always row-parallel)
        # ffn.down_proj drops its bias. Every rank compares against rank 0's output.
        ref_out = ref(**inputs)
        dist.broadcast(ref_out, 0)
        ts_blocks = _capture_head_input(ts_model)
        ts_out = ts_model(**inputs)
        ar_out = ar_model(**inputs)
        _use_eager_per_token_adaln(ar_model)
        emu = TokenShardedTP(ar_model.model_config.visual_gen_mapping.tp_group_pg)
        emu.begin(batch, thw[0] * (thw[1] // 2) * (thw[2] // 2))
        _replace_all_reduce(ar_model, lambda: _EmulatedAllReduce(emu))
        emu_blocks = _capture_head_input(ar_model)
        emu_out = ar_model(**inputs)
        mixed_err = None
        if mixed_up_timestep is not None:
            mixed = dict(inputs, timestep=mixed_up_timestep.to(device, torch.bfloat16))
            mixed_out = ref(**mixed)
            dist.broadcast(mixed_out, 0)
            mixed_err = _rel_l2(mixed_out, ref_out)
    ok = ts_out.shape == ref_out.shape == inputs["hidden_states"].shape
    _check(ok and bool(torch.isfinite(ts_out).all()), f"{case}: shape/finite {ts_out.shape}")
    # Token-sharded TP only changes the transformer blocks: their output must be bitwise equal. The
    # replicated output head may still round differently when the all-reduce path hands it
    # a differently strided tensor (expand_timesteps broadcasts temb per token there).
    _check(
        torch.equal(ts_blocks["x"], emu_blocks["x"]),
        f"{case}: token-sharded blocks != plain-TP blocks with the same collectives (rel-L2 "
        f"{_rel_l2(ts_blocks['x'], emu_blocks['x']):.3e})",
    )
    err = _rel_l2(ts_out, emu_out)
    _check(err <= 1e-3, f"{case}: output vs all-reduce TP with the same collectives {err:.3e}")
    err = _rel_l2(ts_out, ref_out)
    _check(err <= _SINGLE_GPU_REL_L2, f"{case}: token-sharded TP vs single GPU rel-L2 {err:.3e}")
    if mixed_err is not None:
        _check(
            mixed_err >= _SENSITIVITY * _SINGLE_GPU_REL_L2,
            f"{case}: a modulation mix-up moves the reference only by rel-L2 {mixed_err:.3e}",
        )
    err = _rel_l2(ts_out, ar_out)
    _check(err <= _ALL_REDUCE_REL_L2, f"{case}: token-sharded TP vs all-reduce TP rel-L2 {err:.3e}")


def _logic_cases(rank, world_size, cases):
    """Several cases in one spawn (the spawn and import dominate the cost).

    Each case starts without a device mesh, as a fresh spawn does: the tp=1 reference's
    VisualGenMapping derives its TP rank from the process-global mesh the previous case's
    TP models built, and would otherwise get an invalid rank.
    """
    from tensorrt_llm._torch.device_mesh import DeviceMeshTopologyImpl

    for kwargs in cases:
        DeviceMeshTopologyImpl.device_mesh = None
        DeviceMeshTopologyImpl.tp_mesh = None
        _logic_vs_single_gpu(rank, world_size, **kwargs)


_T2V_EXPAND = dict(_WAN_T2V_TEST_CONFIG, expand_timesteps=True)


def _per_token_timesteps(thw):
    """[1, S] per-token timesteps: a ramp, so every token has its own modulation."""
    t, h, w = thw
    return torch.linspace(0.0, 0.9, t * (h // 2) * (w // 2))[None]


_TP2_CASES = [
    # T2V, B=1.
    dict(
        case="t2v_b1",
        config_dict=_WAN_T2V_TEST_CONFIG,
        batch=1,
        thw=(2, 4, 4),
        timestep=torch.tensor([0.5]),
    ),
    # I2V: image cross-attention (add_k_proj / add_v_proj read the image embeddings).
    dict(
        case="i2v",
        config_dict=_WAN_I2V_TEST_CONFIG,
        batch=1,
        thw=(2, 4, 4),
        timestep=torch.tensor([0.5]),
        image_dim=_WAN_I2V_TEST_CONFIG["image_dim"],
    ),
    # Per-token timesteps (temb [B, S, 6, D]), unpadded (S=8) and padded (S=9).
    *[
        dict(
            case=f"per_token_S{thw[0] * thw[1] * thw[2] // 4}",
            config_dict=_T2V_EXPAND,
            batch=1,
            thw=thw,
            timestep=_per_token_timesteps(thw),
            mixed_up_timestep=_per_token_timesteps(thw).flip(-1),
        )
        for thw in ((2, 4, 4), (1, 6, 6))
    ],
    # expand_timesteps with per-sample timesteps: the table stays [B, 6, D].
    dict(
        case="expand_uniform",
        config_dict=_T2V_EXPAND,
        batch=2,
        thw=(2, 4, 4),
        timestep=torch.tensor([0.3, 0.7]),
        mixed_up_timestep=torch.tensor([0.7, 0.3]),
    ),
]


class TestWanTokenShardedTP:
    def test_tp2_cases(self):
        """T2V B=1, I2V, per-token temb (S=8, padded S=9) and expand_timesteps, TP2, one spawn."""
        run_test_in_distributed(
            world_size=2, test_fn=functools.partial(_logic_cases, cases=_TP2_CASES)
        )

    def test_t2v_tp3_b2_distinct_timesteps(self):
        """B=2 with distinct timesteps, TP3 (heads 2+1+1): S=8 pads to 9 and rank 1 straddles
        both samples, so a per-shard table mix-up fails the bitwise check."""
        run_test_in_distributed(
            world_size=3,
            test_fn=functools.partial(
                _logic_vs_single_gpu,
                case="tp3_b2",
                config_dict=_WAN_T2V_TEST_CONFIG,
                batch=2,
                thw=(2, 4, 4),
                timestep=torch.tensor([0.3, 0.7]),
                mixed_up_timestep=torch.tensor([0.7, 0.3]),
            ),
        )

    def test_t2v_tp4_b2_interior_padding(self):
        """B=2, TP4, S=9 -> S_pad=10: per-sample interior padding."""
        run_test_in_distributed(
            world_size=4,
            test_fn=functools.partial(
                _logic_vs_single_gpu,
                case="tp4_b2_padded",
                config_dict=_WAN_T2V_TEST_CONFIG,
                batch=2,
                thw=(1, 6, 6),
                timestep=torch.tensor([0.3, 0.7]),
                mixed_up_timestep=torch.tensor([0.7, 0.3]),
            ),
        )


# =============================================================================
# Blocks compiled as the pipeline does (torch.compile per block)
# =============================================================================


def _compiled_break_reasons(model, all_inputs):
    """Compile every block as the pipeline does; run the inputs; return the outputs, the
    set of graph-break reasons and the number of graphs compiled."""
    import torch._dynamo
    from torch._dynamo.utils import counters

    torch._dynamo.reset()
    counters.clear()
    model.blocks = nn.ModuleList(
        [torch.compile(b, dynamic=None, fullgraph=False) for b in model.blocks]
    )
    with torch.no_grad():
        outs = [model(**inp) for inp in all_inputs]
    return outs, {str(k) for k in counters["graph_break"]}, counters["stats"]["unique_graphs"]


def _graphs_compiled(model, all_inputs, num_blocks):
    """Graphs compiled when the model runs its first ``num_blocks`` blocks, each compiled as
    the pipeline does (the model's blocks are restored afterwards)."""
    import torch._dynamo
    from torch._dynamo.utils import counters

    blocks = model.blocks
    try:
        model.blocks = nn.ModuleList([torch.compile(b) for b in list(blocks)[:num_blocks]])
        torch._dynamo.reset()
        counters.clear()
        with torch.no_grad():
            for inp in all_inputs:
                model(**inp)
        return counters["stats"]["unique_graphs"]
    finally:
        model.blocks = blocks


def _logic_compiled_blocks(rank, world_size):
    device = torch.device(f"cuda:{rank}")
    _, ts_model, ar_model = _build_models(rank, world_size, _WAN_T2V_TEST_CONFIG)
    shapes = [(2, 4, 4), (1, 6, 6), (2, 4, 4)]  # S=8, S=9 (padded at TP2, B=1), revisit
    all_inputs = [
        _inputs(device, 1, thw, torch.tensor([0.5]), seed=10 + i) for i, thw in enumerate(shapes)
    ]
    with torch.no_grad():
        eager = [ts_model(**inp) for inp in all_inputs]
    # One compiled block graph serves every block: the converted modules share one adapter
    # class per base class, so a second block adds no more graphs than in plain TP (a fresh
    # class per swap would recompile every block on type guards).
    extra = {
        name: _graphs_compiled(model, all_inputs, 2) - _graphs_compiled(model, all_inputs, 1)
        for name, model in (("token-sharded", ts_model), ("plain", ar_model))
    }
    _check(extra["token-sharded"] <= extra["plain"], f"graphs added by a second block: {extra}")
    outs, ts_reasons, ts_graphs = _compiled_break_reasons(ts_model, all_inputs)
    for i, (out, ref) in enumerate(zip(outs, eager)):
        err = _rel_l2(out, ref)
        _check(err <= 1e-2, f"compiled vs eager token-sharded TP, input {i}: rel-L2 {err:.3e}")
    # Token-sharded TP adds no graph break of its own: every break reason also occurs on the
    # all-reduce path (the pre-existing QK-norm all-reduce / mesh lookup break).
    _, ar_reasons, ar_graphs = _compiled_break_reasons(ar_model, all_inputs)
    _check(
        ts_reasons <= ar_reasons, f"token-sharded TP-only graph breaks: {ts_reasons - ar_reasons}"
    )
    _check(
        ts_graphs <= ar_graphs,
        f"token-sharded TP compiled {ts_graphs} graphs, plain TP {ar_graphs}",
    )


class TestWanTokenShardedTPCompile:
    def test_compiled_blocks_tp2(self):
        """Per-block torch.compile at TP2, two shapes and a revisit: matches eager and adds no
        graph break beyond plain TP's."""
        run_test_in_distributed(world_size=2, test_fn=_logic_compiled_blocks)


# =============================================================================
# CUDAGraphRunner-wrapped forward
# =============================================================================


def _logic_cuda_graph(rank, world_size):
    from tensorrt_llm._torch.visual_gen.cuda_graph_runner import (
        CUDAGraphRunner,
        CUDAGraphRunnerConfig,
    )

    device = torch.device(f"cuda:{rank}")
    _, ts_model, _ = _build_models(rank, world_size, _WAN_T2V_TEST_CONFIG)
    # Key A: B=2, S=8 (unpadded); key B: B=1, S=9 (padded at TP2). Each key is captured
    # on first use and replayed after switching keys.
    key_a = [_inputs(device, 2, (2, 4, 4), torch.tensor([0.3, 0.7]), seed=s) for s in (1, 2)]
    key_b = [_inputs(device, 1, (1, 6, 6), torch.tensor([0.4]), seed=s) for s in (3, 4)]
    inputs = [key_a[0], key_a[1], key_b[0], key_a[0], key_b[1], key_b[0]]
    eager_forward = ts_model.forward
    runner = CUDAGraphRunner(CUDAGraphRunnerConfig(use_cuda_graph=True))
    ts_model.forward = runner.wrap(ts_model.forward)
    try:
        with torch.no_grad():
            for i, inp in enumerate(inputs):
                graph_out = ts_model(**inp).clone()
                eager_out = eager_forward(**inp)
                _check(
                    torch.equal(graph_out, eager_out),
                    f"call {i}: graph vs eager rel-L2 {_rel_l2(graph_out, eager_out):.3e}",
                )
        _check(len(runner.graphs) == 2, f"{len(runner.graphs)} graphs captured")
    finally:
        # As BasePipeline.cleanup(): release the graphs (they hold the NCCL communicator's
        # collectives) before the process group is destroyed.
        runner.clear()
        torch.cuda.synchronize()


class TestWanTokenShardedTPCudaGraph:
    def test_cuda_graph_tp2(self):
        """CUDAGraphRunner at TP2, two keys (one padded) with switches and revisits: bitwise
        equal to eager."""
        run_test_in_distributed(world_size=2, test_fn=_logic_cuda_graph)


# =============================================================================
# One D=5120 static-NVFP4 WanBlock (FP4 all-gathers) vs the all-reduce block
# =============================================================================

_D5120_BLOCK = dict(
    num_attention_heads=40,
    attention_head_dim=128,
    ffn_dim=13824,
    eps=1e-6,
    cross_attn_norm=True,
)
_E2M1_MAX, _FP8_MAX = 6.0, 448.0
_ATTN1_QKV = ("attn1.to_q", "attn1.to_k", "attn1.to_v")  # fused into attn1.qkv_proj


def _block_config(vgm, quant_algo, token_sharded):
    world_size = vgm.tp_size
    cfg = DiffusionModelConfig(
        pretrained_config=SimpleNamespace(**_D5120_BLOCK),
        quant_config=QuantConfig(quant_algo=quant_algo) if quant_algo else QuantConfig(),
        torch_compile=TorchCompileConfig(enable=False),
        attention=AttentionConfig(backend="VANILLA"),
        visual_gen_mapping=vgm,
        parallel=ParallelConfig(
            tp_size=world_size, tp_layout="token_sharded" if token_sharded else None
        ),
        skip_create_weights_in_init=False,
    )
    cfg.mapping = vgm.to_llm_mapping()
    return cfg


def _full_block_weights(d, ffn):
    """Unsharded bf16 weights for one WanBlock (checkpoint names), seeded identically."""
    gen = torch.Generator().manual_seed(2026)

    def lin(n, k):
        return {
            "weight": (torch.randn(n, k, generator=gen) * k**-0.5).to(torch.bfloat16),
            "bias": (torch.randn(n, generator=gen) * 0.02).to(torch.bfloat16),
        }

    w = {f"attn{i}.{p}": lin(d, d) for i in (1, 2) for p in ("to_q", "to_k", "to_v", "to_out.0")}
    w["ffn.up_proj"] = lin(ffn, d)
    w["ffn.down_proj"] = lin(d, ffn)
    params = {
        "scale_shift_table": torch.randn(1, 6, d, generator=gen) * d**-0.5,
        "norm2.weight": 1.0 + 0.1 * torch.randn(d, generator=gen),
        "norm2.bias": 0.1 * torch.randn(d, generator=gen),
    }
    for i in (1, 2):
        for n in ("norm_q", "norm_k"):
            params[f"attn{i}.{n}.weight"] = 1.0 + 0.1 * torch.randn(d, generator=gen)
    return w, params


def _to_nvfp4(entry, act_amax, weight_amax=None):
    """ModelOpt static-NVFP4 checkpoint entry for a bf16 Linear (1-pass amax calibration).

    weight_amax: shared amax for projections fused at load time (ModelOpt calibrates
    q/k/v with one weight_scale_2).
    """
    weight = entry["weight"].cuda()
    if weight_amax is None:
        weight_amax = weight.float().abs().amax()
    weight_scale_2 = (weight_amax.float().cuda() / (_FP8_MAX * _E2M1_MAX)).reshape(1)
    fp4, sf = torch.ops.trtllm.fp4_quantize(weight, 1.0 / weight_scale_2, 16, False, False)
    return {
        "weight": fp4.cpu(),
        "weight_scale": sf.view(torch.float8_e4m3fn).reshape(weight.shape[0], -1).cpu(),
        "weight_scale_2": weight_scale_2.cpu(),
        "input_scale": (act_amax.float().cpu() / (_FP8_MAX * _E2M1_MAX)).reshape(1),
        "bias": entry["bias"],
    }


def _as_model(block):
    """A one-block model whose ``blocks`` container the converter's rules walk."""
    model = nn.Module()
    model.blocks = nn.ModuleList([block])
    return model


def _load_block(block, linear_weights, params):
    from tensorrt_llm._torch.visual_gen.models.wan.utils_wan import get_nvfp4_input_scale

    for name, module in block.named_modules():
        if not isinstance(module, Linear):
            continue
        if name == "attn1.qkv_proj":
            module.load_weights([linear_weights[f"attn1.to_{p}"] for p in ("q", "k", "v")])
        else:
            module.load_weights([linear_weights[name]])
        module.post_load_weights()
    for name, value in params.items():
        if ".norm_q." in name or ".norm_k." in name:
            owner = block.get_submodule(name.rsplit(".", 1)[0])
            owner.load_weights({"weight": value})
        else:
            block.get_parameter(name).data.copy_(value)
    # Same wiring as WanTransformer3DModel.post_load_weights.
    block._norm1_fp4_scale = get_nvfp4_input_scale(block.attn1.qkv_proj)
    block._norm2_fp4_scale = get_nvfp4_input_scale(block.attn2.to_q)
    block._norm3_fp4_scale = get_nvfp4_input_scale(block.ffn.up_proj)


def _calibrate_amax(vgm, device, bf16_weights, params, block_inputs):
    """One bf16 all-reduce-TP pass recording each Linear's input amax (max over ranks)."""
    from tensorrt_llm._torch.visual_gen.models.wan.transformer_wan import WanBlock

    block = WanBlock(_block_config(vgm, None, False), 0).to(device)
    _load_block(block, bf16_weights, params)
    amax = {}

    def record(name):
        def hook(mod, args):
            a = args[0].float().abs().amax()
            amax[name] = torch.maximum(amax.get(name, a), a)

        return hook

    handles = [
        m.register_forward_pre_hook(record(n))
        for n, m in block.named_modules()
        if isinstance(m, Linear)
    ]
    # The fused up-projection + GELU path does not call up_proj as a module.
    handles.append(block.ffn.register_forward_pre_hook(record("ffn.up_proj")))
    with torch.no_grad():
        block(*block_inputs)
    for h in handles:
        h.remove()
    for v in amax.values():
        dist.all_reduce(v, op=dist.ReduceOp.MAX)  # row-parallel inputs are rank-local
    amax["attn1.to_q"] = amax["attn1.to_k"] = amax["attn1.to_v"] = amax["attn1.qkv_proj"]
    return amax


def _logic_nvfp4_block(rank, world_size, *, batch, thw):
    from tensorrt_llm._torch.visual_gen.models.wan.transformer_wan import (
        WanBlock,
        WanRotaryPosEmbed,
    )

    device = torch.device(f"cuda:{rank}")
    d, ffn, text_len = 5120, _D5120_BLOCK["ffn_dim"], 64
    t, h, w = thw
    seq = t * h * w
    gen = torch.Generator().manual_seed(7)
    x = torch.randn(batch, seq, d, generator=gen).to(device, torch.bfloat16)
    enc = (torch.randn(batch, text_len, d, generator=gen) * 0.5).to(device, torch.bfloat16)
    temb = (torch.randn(batch, 6, d, generator=gen) * 0.1).to(device)  # distinct per sample
    rope = WanRotaryPosEmbed(128, (1, 2, 2), max_seq_len=1024).to(device)
    freqs_cos, freqs_sin = rope(torch.empty(batch, 16, t, 2 * h, 2 * w, device=device))
    timestep = torch.full((batch,), 0.5, device=device)
    block_inputs = (x, enc, temb, freqs_cos, freqs_sin, timestep)

    vgm = VisualGenMapping(world_size=world_size, rank=rank, tp_size=world_size)
    bf16_weights, params = _full_block_weights(d, ffn)
    amax = _calibrate_amax(vgm, device, bf16_weights, params, block_inputs)
    qkv_amax = max(bf16_weights[f"attn1.to_{p}"]["weight"].float().abs().amax() for p in "qkv")
    fp4_weights = {
        name: _to_nvfp4(entry, amax[name], qkv_amax if name in _ATTN1_QKV else None)
        for name, entry in bf16_weights.items()
    }

    ar_block = WanBlock(_block_config(vgm, QuantAlgo.NVFP4, False), 0).to(device)
    _load_block(ar_block, fp4_weights, params)
    ts_cfg = _block_config(vgm, QuantAlgo.NVFP4, True)
    tp = TokenShardedTP.from_model_config(ts_cfg)
    ts_block = WanBlock(ts_cfg, 0).to(device)
    convert_to_token_sharded_tp(_as_model(ts_block), tp)
    _load_block(ts_block, fp4_weights, params)
    for blk in (ar_block, ts_block):
        assert all(
            s is not None
            for s in (blk._norm1_fp4_scale, blk._norm2_fp4_scale, blk._norm3_fp4_scale)
        )

    # A silent BF16 fallback must fail: every consumer receives an Fp4QuantizedTensor.
    seen = {}

    def expect_fp4(name):
        def hook(mod, args):
            seen.setdefault(name, []).append(type(args[0]).__name__)

        return hook

    # ffn: the fused GELU path calls up_proj's quant method directly (no module call),
    # so the check sits on the MLP input.
    for name, mod in (
        ("attn1.qkv_proj", ts_block.attn1.qkv_proj),
        ("attn2.to_q", ts_block.attn2.to_q),
        ("ffn", ts_block.ffn),
    ):
        mod.register_forward_pre_hook(expect_fp4(name))

    with torch.no_grad():
        ref = ar_block(*block_inputs)
        plan = tp.begin(batch, seq)
        # The block sees this rank's [n, g, D] sample groups and the shard's table.
        out = ts_block(
            tp.local_view(tp.shard(x)), enc, tp.per_sample_table(temb), *block_inputs[3:]
        )
        out = tp.unshard(out.reshape(plan.local_rows, -1))
        _replace_all_reduce(_as_model(ar_block), lambda: _EmulatedAllReduce(tp))
        emu = ar_block(*block_inputs)
        _replace_all_reduce(_as_model(ar_block), _ExactAllReduce)
        exact = ar_block(*block_inputs)
    fp4 = ["Fp4QuantizedTensor"]
    expected = {"attn1.qkv_proj": fp4, "attn2.to_q": fp4, "ffn": fp4}
    _check(seen == expected, f"boundary inputs {seen}")
    desc = f"tp={world_size} B={batch} S={seq} padded={tp.plan.is_padded}"
    _check(torch.equal(out, emu), f"{desc}: token-sharded TP != emulated all-reduce block")
    cos = torch.nn.functional.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0)
    err = _rel_l2(out, ref)
    _check(
        cos.item() >= 0.999 and err <= 2e-2, f"{desc}: vs all-reduce cos={cos:.6f} rel-L2={err:.3e}"
    )
    # Precision against a correctly rounded all-reduce: token-sharded TP's reduce-scatter may use a
    # different NCCL algorithm than the all-reduce (ring vs NVLS at TP8), but must stay
    # within 1.5x of the all-reduce block's error.
    ts_err, ar_err = _rel_l2(out, exact), _rel_l2(ref, exact)
    _check(
        ts_err <= 1.5 * ar_err + 1e-6,
        f"{desc}: vs fp32-exact all-reduce: token-sharded {ts_err:.3e}, plain TP {ar_err:.3e}",
    )


class TestWanTokenShardedTPNVFP4Block:
    @pytest.mark.parametrize(
        "world_size,batch,thw",
        [(2, 2, (1, 16, 16)), (3, 2, (3, 5, 7)), (4, 2, (3, 5, 7)), (8, 2, (4, 15, 16))],
        ids=["tp2_S256", "tp3_S105", "tp4_S105_padded", "tp8_S960"],
    )
    def test_static_nvfp4_block(self, world_size, batch, thw):
        """D=5120 static-NVFP4 WanBlock: bitwise vs the emulated all-reduce block, close to the
        plain one, and within 1.5x of its error vs an fp32-exact all-reduce."""
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
            pytest.skip("NVFP4 requires SM100+")
        run_test_in_distributed(
            world_size=world_size,
            test_fn=functools.partial(_logic_nvfp4_block, batch=batch, thw=thw),
        )


# =============================================================================
# The same D=5120 WanBlock with FP8 block scales (FP8 + scales all-gathers into DeepGEMM)
# =============================================================================


def _to_fp8_block(entry):
    """128x128 block-scale FP8 checkpoint entry for a bf16 Linear: the ``weight`` /
    ``weight_scale`` pair ``FP8BlockScalesLinearMethod`` loads."""
    weight = entry["weight"].cuda()
    n, k = weight.shape
    blocks = weight.float().view(n // 128, 128, k // 128, 128).permute(0, 2, 1, 3)
    scale = blocks.abs().amax(dim=(2, 3), keepdim=True).clamp(min=1e-12) / _FP8_MAX
    q = (blocks / scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return {
        "weight": q.permute(0, 2, 1, 3).reshape(n, k).cpu(),
        "weight_scale": scale.reshape(n // 128, k // 128).cpu(),
        "bias": entry["bias"],
    }


def _logic_fp8_block_scales_block(rank, world_size, *, batch, thw):
    from tensorrt_llm._torch.visual_gen.models.wan.transformer_wan import (
        WanBlock,
        WanRotaryPosEmbed,
    )
    from tensorrt_llm._torch.visual_gen.parallel.token_sharded_tp import (
        Fp8BlockScaledActivation,
        fp8_block_scale_prequant_ok,
    )

    device = torch.device(f"cuda:{rank}")
    d, ffn, text_len = 5120, _D5120_BLOCK["ffn_dim"], 64
    t, h, w = thw
    seq = t * h * w
    gen = torch.Generator().manual_seed(7)
    x = torch.randn(batch, seq, d, generator=gen).to(device, torch.bfloat16)
    enc = (torch.randn(batch, text_len, d, generator=gen) * 0.5).to(device, torch.bfloat16)
    temb = (torch.randn(batch, 6, d, generator=gen) * 0.1).to(device)  # distinct per sample
    rope = WanRotaryPosEmbed(128, (1, 2, 2), max_seq_len=1024).to(device)
    freqs_cos, freqs_sin = rope(torch.empty(batch, 16, t, 2 * h, 2 * w, device=device))
    timestep = torch.full((batch,), 0.5, device=device)
    block_inputs = (x, enc, temb, freqs_cos, freqs_sin, timestep)

    vgm = VisualGenMapping(world_size=world_size, rank=rank, tp_size=world_size)
    bf16_weights, params = _full_block_weights(d, ffn)
    fp8_weights = {name: _to_fp8_block(entry) for name, entry in bf16_weights.items()}

    ar_block = WanBlock(_block_config(vgm, QuantAlgo.FP8_BLOCK_SCALES, False), 0).to(device)
    _load_block(ar_block, fp8_weights, params)
    ts_cfg = _block_config(vgm, QuantAlgo.FP8_BLOCK_SCALES, True)
    tp = TokenShardedTP.from_model_config(ts_cfg)
    ts_block = WanBlock(ts_cfg, 0).to(device)
    convert_to_token_sharded_tp(_as_model(ts_block), tp)
    _load_block(ts_block, fp8_weights, params)
    # The norms emit bf16 (no static NVFP4 scale) and all three consumers take the FP8 pair.
    norm_scales = (ts_block._norm1_fp4_scale, ts_block._norm2_fp4_scale, ts_block._norm3_fp4_scale)
    consumers = (ts_block.attn1.qkv_proj, ts_block.attn2.to_q, ts_block.ffn.up_proj)
    _check(
        all(s is None for s in norm_scales) and all(map(fp8_block_scale_prequant_ok, consumers)),
        "the FP8 block-scale rule engages for attn1.qkv_proj, attn2.to_q and ffn.up_proj",
    )

    # A silent BF16 gather must fail: the adapters quantize inside, so the hook sits on the
    # helper's all-gather, which must move the FP8 pair at all three block boundaries.
    moved = []
    gather = tp.all_gather

    def spy(act):
        moved.append(type(act))
        return gather(act)

    with torch.no_grad():
        ref = ar_block(*block_inputs)
        plan = tp.begin(batch, seq)
        tp.all_gather = spy
        out = ts_block(
            tp.local_view(tp.shard(x)), enc, tp.per_sample_table(temb), *block_inputs[3:]
        )
        tp.all_gather = gather
        out = tp.unshard(out.reshape(plan.local_rows, -1))
        _replace_all_reduce(_as_model(ar_block), lambda: _EmulatedAllReduce(tp))
        emu = ar_block(*block_inputs)
        _replace_all_reduce(_as_model(ar_block), _ExactAllReduce)
        exact = ar_block(*block_inputs)
    _check(moved == [Fp8BlockScaledActivation] * 3, f"boundary all-gathers moved {moved}")
    desc = f"tp={world_size} B={batch} S={seq} padded={tp.plan.is_padded}"
    _check(torch.equal(out, emu), f"{desc}: token-sharded TP != emulated all-reduce block")
    cos = torch.nn.functional.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0)
    err = _rel_l2(out, ref)
    _check(
        cos.item() >= 0.999 and err <= 2e-2, f"{desc}: vs all-reduce cos={cos:.6f} rel-L2={err:.3e}"
    )
    ts_err, ar_err = _rel_l2(out, exact), _rel_l2(ref, exact)
    _check(
        ts_err <= 1.5 * ar_err + 1e-6,
        f"{desc}: vs fp32-exact all-reduce: token-sharded {ts_err:.3e}, plain TP {ar_err:.3e}",
    )


class TestWanTokenShardedTPFP8BlockScalesBlock:
    @pytest.mark.parametrize(
        "world_size,batch,thw",
        [(2, 2, (1, 16, 16)), (3, 2, (3, 5, 7)), (4, 2, (3, 5, 7)), (8, 2, (4, 15, 16))],
        ids=["tp2_S256", "tp3_S105", "tp4_S105_padded", "tp8_S960"],
    )
    def test_fp8_block_scales_block(self, world_size, batch, thw):
        """D=5120 FP8 block-scale WanBlock (FP8 + scales all-gathers into DeepGEMM): bitwise vs
        the emulated all-reduce block, close to the plain one, and within 1.5x of its error vs
        an fp32-exact all-reduce."""
        if not torch.cuda.is_available() or not is_sm_100f():
            pytest.skip("The FP8 block-scale DeepGEMM path requires an SM100-family GPU")
        run_test_in_distributed(
            world_size=world_size,
            test_fn=functools.partial(_logic_fp8_block_scales_block, batch=batch, thw=thw),
        )


# =============================================================================
# VSA is rejected
# =============================================================================


def _logic_vsa_rejected(rank, world_size):
    from tensorrt_llm._torch.visual_gen.models.wan.transformer_wan import WanTransformer3DModel
    from tensorrt_llm.visual_gen.sparse_attention import VideoSparseAttentionConfig

    cfg = _make_model_config(
        dict(_WAN_T2V_TEST_CONFIG, attention_head_dim=128),
        tp_size=world_size,
        backend="CUTEDSL",
        tp_layout="token_sharded",
    )
    cfg.attention = AttentionConfig(
        backend="CUTEDSL", sparse_attention_config=VideoSparseAttentionConfig(vsa_sparsity=0.9)
    )
    with pytest.raises(ValueError, match="does not support Video Sparse Attention"):
        WanTransformer3DModel(cfg)


class TestWanTokenShardedTPRejections:
    def test_vsa_rejected(self):
        """Token-sharded TP with VSA raises."""
        run_test_in_distributed(world_size=2, test_fn=_logic_vsa_rejected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
