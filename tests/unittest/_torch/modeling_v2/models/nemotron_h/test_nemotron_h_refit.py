# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-flight weight updates on the Nemotron-3 Ultra BF16 dep16 and dep32 targets, on CPU.

An RL weight sync (llmapi/rlhf_utils.py) loads the checkpoint as size-based buckets of keys with
``allow_partial_loading``. These tests drive the targets' weight tables at small dims with
Ultra's structure, and engine MoE layers that record what they are fed: buckets land bit for bit
where a whole load puts them, a bucket with names the target does not take fails before anything
is written, a sync reports the tensors it left stale on the rank that holds them, and the bf16
copies the decode graphs read are refreshed in place.
"""

import importlib
import inspect
import random
import re
from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

pytestmark = pytest.mark.cpu_only

_PKG = "tensorrt_llm._torch._experimental.modeling_v2.models.nemotron_h"
TARGETS = {"dep16": 16, "dep32": 32}

# Ultra's structure at small dims: one of each block kind, mamba and moe twice.
HIDDEN, VOCAB = 32, 64
HEADS, SSM_HEAD_DIM, GROUPS, STATE, CONV_K = 4, 8, 2, 4, 4
D_INNER = HEADS * SSM_HEAD_DIM
GN = GROUPS * STATE
Q_HEADS, KV_HEADS, HEAD_DIM = 4, 2, 8
MOE_INTER, LATENT, SHARED = 16, 8, 16
WINDOW = 2  # routed experts per rank
KINDS = ["mamba", "moe", "attention", "mamba", "moe"]


def _shapes(experts: int) -> list[tuple[str, tuple[int, ...]]]:
    """(checkpoint key pattern, shape) of every tensor the table or the engine feed reads."""
    return [
        (r"backbone\.embeddings\.weight", (VOCAB, HIDDEN)),
        (r"backbone\.norm_f\.weight", (HIDDEN,)),
        (r"backbone\.layers\.\d+\.norm\.weight", (HIDDEN,)),
        (r"backbone\.layers\.\d+\.mixer\.in_proj\.weight", (2 * D_INNER + 2 * GN + HEADS, HIDDEN)),
        (r"backbone\.layers\.\d+\.mixer\.conv1d\.weight", (D_INNER + 2 * GN, 1, CONV_K)),
        (r"backbone\.layers\.\d+\.mixer\.conv1d\.bias", (D_INNER + 2 * GN,)),
        (r"backbone\.layers\.\d+\.mixer\.(A_log|D|dt_bias)", (HEADS,)),
        (r"backbone\.layers\.\d+\.mixer\.norm\.weight", (D_INNER,)),
        (r"backbone\.layers\.\d+\.mixer\.out_proj\.weight", (HIDDEN, D_INNER)),
        (r"backbone\.layers\.\d+\.mixer\.q_proj\.weight", (Q_HEADS * HEAD_DIM, HIDDEN)),
        (r"backbone\.layers\.\d+\.mixer\.[kv]_proj\.weight", (KV_HEADS * HEAD_DIM, HIDDEN)),
        (r"backbone\.layers\.\d+\.mixer\.o_proj\.weight", (HIDDEN, Q_HEADS * HEAD_DIM)),
        (r"backbone\.layers\.\d+\.mixer\.gate\.weight", (experts, HIDDEN)),
        (r"backbone\.layers\.\d+\.mixer\.gate\.e_score_correction_bias", (experts,)),
        (r"backbone\.layers\.\d+\.mixer\.fc1_latent_proj\.weight", (LATENT, HIDDEN)),
        (r"backbone\.layers\.\d+\.mixer\.fc2_latent_proj\.weight", (HIDDEN, LATENT)),
        (r"backbone\.layers\.\d+\.mixer\.shared_experts\.up_proj\.weight", (SHARED, HIDDEN)),
        (r"backbone\.layers\.\d+\.mixer\.shared_experts\.down_proj\.weight", (HIDDEN, SHARED)),
        (r"backbone\.layers\.\d+\.mixer\.experts\.\d+\.up_proj\.weight", (MOE_INTER, LATENT)),
        (r"backbone\.layers\.\d+\.mixer\.experts\.\d+\.down_proj\.weight", (LATENT, MOE_INTER)),
        (r"lm_head\.weight", (VOCAB, HIDDEN)),
    ]


def _weights(target: str):
    return importlib.import_module(f"{_PKG}.nemotron3_ultra_bf16__sm_100__{target}.weights")


def _modeling(target: str):
    return importlib.import_module(f"{_PKG}.nemotron3_ultra_bf16__sm_100__{target}.modeling")


def _core_cls(target: str):
    return getattr(_modeling(target), f"NemotronUltraBf16{target.capitalize()}ModelingV2")


def _shell_cls(target: str):
    return getattr(_modeling(target), f"ModelingV2Nemotron3UltraBf16Sm100{target.capitalize()}")


class _EngineMoe(nn.Module):
    """An engine MoE layer that records what it is fed, by engine key, the last value winning."""

    def __init__(self) -> None:
        super().__init__()
        self.fed: dict[str, torch.Tensor] = {}
        self.partial: list[bool] = []

    def load_weights(self, weights, allow_partial_loading: bool = False) -> None:
        (feed,) = weights
        self.partial.append(allow_partial_loading)
        self.fed.update({k: v.clone() for k, v in feed.items()})


class _Core(nn.Module):
    def __init__(self, W, ranks: int, rank: int, backend: str = "MEGAMOE_CUTEDSL") -> None:
        super().__init__()
        experts = WINDOW * ranks
        cfg = SimpleNamespace(
            layers_block_type=list(KINDS),
            hidden_size=HIDDEN,
            vocab_size=VOCAB,
            mamba_num_heads=HEADS,
            mamba_head_dim=SSM_HEAD_DIM,
            n_groups=GROUPS,
            ssm_state_size=STATE,
            conv_kernel=CONV_K,
            num_attention_heads=Q_HEADS,
            num_key_value_heads=KV_HEADS,
            head_dim=HEAD_DIM,
            n_routed_experts=experts,
            num_experts_per_tok=4,
            routed_scaling_factor=5.0,
            moe_intermediate_size=MOE_INTER,
            moe_latent_size=LATENT,
            moe_shared_expert_intermediate_size=SHARED,
        )
        self.model_config = SimpleNamespace(
            pretrained_config=cfg,
            mapping=SimpleNamespace(
                enable_attention_dp=True,
                moe_tp_size=1,
                moe_ep_size=ranks,
                moe_ep_rank=rank,
                rank=rank,
            ),
            torch_dtype=torch.bfloat16,
            moe_backend_requested=backend,
        )
        d = self.dims = W.MODEL_WEIGHTS.dims(self)
        self.w = W.MODEL_WEIGHTS.declare(d)
        self.routed_experts = nn.ModuleDict({str(i): _EngineMoe() for i in d.moe_layers})


class _Shell(nn.Module):
    def __init__(self, W, ranks: int, rank: int) -> None:
        super().__init__()
        self.model = _Core(W, ranks, rank)
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False, dtype=torch.bfloat16)


@pytest.fixture(autouse=True)
def _no_cuda(monkeypatch):
    """The loads synchronize and release the CUDA cache; neither exists here."""
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda *a, **k: None)


def _checkpoint(W, ranks: int, seed: int) -> dict[str, torch.Tensor]:
    """Every tensor a rank reads, every rank's routed experts and the LM head, in the
    checkpoint's dtypes: the router's score-correction bias in fp32, the rest in bf16."""
    core = _Core(W, ranks, 0)
    d = W.MODEL_WEIGHTS.dims(core)
    keys = {row[0] for rows in W.MODEL_WEIGHTS.manifest(core).values() for row, _ in rows}
    for i in d.moe_layers:
        for e in range(WINDOW * ranks):
            p = f"backbone.layers.{i}.mixer.experts.{e}"
            keys.update((f"{p}.up_proj.weight", f"{p}.down_proj.weight"))
    keys.add("lm_head.weight")
    shapes = _shapes(WINDOW * ranks)
    g = torch.Generator().manual_seed(seed)
    out = {}
    for key in sorted(keys):
        (shape,) = [s for pattern, s in shapes if re.fullmatch(pattern, key)]
        fp32 = key.endswith(".e_score_correction_bias")
        out[key] = torch.randn(shape, generator=g).to(torch.float32 if fp32 else torch.bfloat16)
    return out


def _state(model: _Shell) -> dict[str, torch.Tensor]:
    core = model.model
    out = {f"w.{k}": v.detach().clone() for k, v in core.w.items()}
    out["lm_head"] = model.lm_head.weight.detach().clone()
    for i, moe in core.routed_experts.items():
        out.update({f"moe{i}.{k}": v for k, v in moe.fed.items()})
    return out


def _assert_same(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> None:
    assert a.keys() == b.keys(), sorted(a.keys() ^ b.keys())[:8]
    differ = [k for k in a if not torch.equal(a[k], b[k])]
    assert not differ, differ[:8]


def _buckets(checkpoint: dict, count: int, seed: int) -> list[dict]:
    """Size-agnostic buckets: every key once, in shuffled order, so an expert's up and down
    tensors and a fused weight's q/k/v rows land in different buckets."""
    keys = sorted(checkpoint)
    random.Random(seed).shuffle(keys)
    return [{k: checkpoint[k] for k in keys[j::count]} for j in range(count)]


def _sync(W, model: _Shell, buckets: list[dict]) -> set[str]:
    fed = set()
    for bucket in buckets:
        fed |= W.MODEL_WEIGHTS.load(model, bucket, allow_partial=True)
    return fed


@pytest.mark.parametrize("rank", [0, -1], ids=["first_rank", "last_rank"])
@pytest.mark.parametrize("target", TARGETS)
def test_bucketed_update_lands_where_a_whole_load_does(target, rank):
    W, ranks = _weights(target), TARGETS[target]
    rank %= ranks
    old, new = _checkpoint(W, ranks, seed=1), _checkpoint(W, ranks, seed=2)
    whole = _Shell(W, ranks, rank)
    W.MODEL_WEIGHTS.load(whole, new)
    synced = _Shell(W, ranks, rank)
    W.MODEL_WEIGHTS.load(synced, old)
    fed = _sync(W, synced, _buckets(new, 7, seed=rank))
    _assert_same(_state(synced), _state(whole))
    assert W.MODEL_WEIGHTS.refit_keys(synced.model) <= fed
    for moe in synced.model.routed_experts.values():
        assert moe.partial[0] is False and all(moe.partial[1:])


@pytest.mark.parametrize(
    "rename",
    [
        # HF transformers 5.x: `model.` for `backbone.` and a layer's experts stacked by global id
        pytest.param(lambda k: "model." + k[len("backbone.") :], id="model_prefix"),
        pytest.param(
            lambda k: re.sub(r"(\.mixer\.experts)\.\d+\.(up|down)_proj\.weight$", r"\1.\2_proj", k),
            id="stacked_experts",
        ),
    ],
)
@pytest.mark.parametrize("target", TARGETS)
def test_a_sync_refuses_names_the_target_does_not_take(target, rename):
    """A sender with other names would otherwise load nothing and leave the policy stale; the
    bucket fails before anything is written."""
    W, ranks = _weights(target), TARGETS[target]
    model = _Shell(W, ranks, 0)
    W.MODEL_WEIGHTS.load(model, _checkpoint(W, ranks, seed=3))
    before = _state(model)
    new = _checkpoint(W, ranks, seed=4)
    bucket = {k: v for k, v in new.items() if ".norm." in k}
    bucket.update({rename(k): v for k, v in new.items() if ".mixer.experts." in k})
    with pytest.raises(ValueError, match="neither loads nor expects"):
        W.MODEL_WEIGHTS.load(model, bucket, allow_partial=True)
    _assert_same(_state(model), before)


@pytest.mark.parametrize("target", TARGETS)
def test_a_full_sync_may_carry_the_mtp_module_and_other_ranks_experts(target):
    """Every rank receives every checkpoint key; the ones it does not load are not an error."""
    W, ranks = _weights(target), TARGETS[target]
    model = _Shell(W, ranks, 0)
    W.MODEL_WEIGHTS.load(model, _checkpoint(W, ranks, seed=5))
    bucket = dict(_checkpoint(W, ranks, seed=6))
    bucket["mtp.layers.0.mixer.in_proj.weight"] = torch.zeros(2, 2, dtype=torch.bfloat16)
    fed = W.MODEL_WEIGHTS.load(model, bucket, allow_partial=True)
    other_window = [k for k in bucket if ".mixer.experts." in k and k not in fed]
    assert other_window and "mtp.layers.0.mixer.in_proj.weight" not in fed
    assert fed == W.MODEL_WEIGHTS.refit_keys(model.model)


def _bind_refit_hooks(target: str, core: "_Core") -> None:
    """The core's refit hooks, as the engine calls them on the real target."""
    cls = _core_cls(target)
    for name in ("pre_reload_weights", "record_refit_fed", "_report_refit"):
        setattr(core, name, MethodType(getattr(cls, name), core))
    core._refit_fed = None
    core._bf16_copies = {}
    core._decode = object()  # the targets exist: post_load_weights only refreshes


def _engine_sync(target: str, model: "_Shell", buckets: list[dict]) -> None:
    """One weight update the way llmapi/rlhf_utils.py runs it: pre_reload_weights, the shell's
    load_weights per bucket (with the arguments ModelLoader passes after inspecting it), then
    post_load_weights."""
    core = model.model
    core.pre_reload_weights()
    load = MethodType(_shell_cls(target).load_weights, model)
    for bucket in buckets:
        load(bucket, weight_mapper=None, allow_partial_loading=True)
    _core_cls(target).post_load_weights(core)


@pytest.mark.parametrize("target", TARGETS)
def test_the_shell_takes_the_loaders_partial_arguments(target):
    """ModelLoader passes `weight_mapper` and `allow_partial_loading` only to a load_weights
    whose signature names them; without the second it refuses every weight update."""
    args = inspect.getfullargspec(_shell_cls(target).load_weights).args
    assert "weight_mapper" in args and "allow_partial_loading" in args


@pytest.mark.parametrize("rank", [0, -1], ids=["first_rank", "last_rank"])
@pytest.mark.parametrize("target", TARGETS)
def test_a_sync_reports_the_tensors_it_left_stale(target, rank, monkeypatch):
    """Rank 0 reports the tensors every rank holds; each rank reports its own routed experts."""
    W, M, ranks = _weights(target), _modeling(target), TARGETS[target]
    rank %= ranks
    model = _Shell(W, ranks, rank)
    _bind_refit_hooks(target, model.model)
    W.MODEL_WEIGHTS.load(model, _checkpoint(W, ranks, seed=5))
    new = _checkpoint(W, ranks, seed=6)
    expert = WINDOW * rank + 1  # in this rank's window
    skipped = {
        "backbone.layers.0.mixer.D",
        f"backbone.layers.1.mixer.experts.{expert}.down_proj.weight",
    }
    warnings = []
    monkeypatch.setattr(M.logger, "warning", warnings.append)
    _engine_sync(target, model, _buckets({k: v for k, v in new.items() if k not in skipped}, 4, 1))
    (message,) = warnings
    reported = skipped if rank == 0 else {k for k in skipped if ".mixer.experts." in k}
    assert f"rank {rank}:" in message and f"left {len(reported)} checkpoint tensors" in message
    assert all(k in message for k in reported)
    assert model.model._refit_fed is None
    # A complete update reports nothing.
    warnings.clear()
    _engine_sync(target, model, _buckets(new, 3, 2))
    assert not warnings


@pytest.mark.parametrize("target", TARGETS)
def test_weight_updates_refuse_the_trtllm_moe_leaf(target):
    """TRTLLM's BF16 leaf would re-stage every routed expert; the core refuses before the MoE
    layers' hooks run (route() refuses an RL engine on it at build)."""
    W, ranks = _weights(target), TARGETS[target]
    core = _Core(W, ranks, 0, backend="TRTLLM")
    _bind_refit_hooks(target, core)
    with pytest.raises(NotImplementedError, match="TRTLLM"):
        core.pre_reload_weights()


@pytest.mark.parametrize("target", TARGETS)
def test_a_sync_converts_dtypes_a_whole_load_rejects(target):
    W, ranks = _weights(target), TARGETS[target]
    model = _Shell(W, ranks, 0)
    W.MODEL_WEIGHTS.load(model, _checkpoint(W, ranks, seed=7))
    key = "backbone.layers.0.mixer.out_proj.weight"
    sent = torch.randn(HIDDEN, D_INNER, dtype=torch.float32)
    W.MODEL_WEIGHTS.load(model, {key: sent}, allow_partial=True)
    assert torch.equal(model.model.w["l0_out_proj_w"], sent.to(torch.bfloat16))
    whole = _checkpoint(W, ranks, seed=8)
    whole[key] = sent
    with pytest.raises(AssertionError):
        W.MODEL_WEIGHTS.load(_Shell(W, ranks, 0), whole)
    # Routed experts go to the engine MoE layer converted: its loader would reinterpret the bytes.
    up = "backbone.layers.1.mixer.experts.0.up_proj.weight"
    sent = torch.randn(MOE_INTER, LATENT, dtype=torch.float32)
    W.MODEL_WEIGHTS.load(model, {up: sent}, allow_partial=True)
    fed = model.model.routed_experts["1"].fed["0.w1.weight"]
    assert fed.dtype == torch.bfloat16 and torch.equal(fed, sent.to(torch.bfloat16))
    whole = _checkpoint(W, ranks, seed=9)
    whole[up] = sent
    with pytest.raises(AssertionError):
        W.MODEL_WEIGHTS.load(_Shell(W, ranks, 0), whole)


@pytest.mark.parametrize("target", TARGETS)
def test_post_load_weights_refreshes_the_bf16_copies_in_place(target):
    """The first call builds the targets; a later one only refreshes the bf16 copies the decode
    graphs captured, keeping their addresses."""
    M = _modeling(target)
    core_cls = getattr(M, f"NemotronUltraBf16{target.capitalize()}ModelingV2")
    core = SimpleNamespace(
        w={"l0_D": torch.randn(HEADS), "l0_dt_bias": torch.randn(HEADS)},
        _bf16_copies={},
        _decode=object(),
        _refit_fed=None,
    )
    core._report_refit = lambda: None
    copies = {k: core_cls.bf16_copy(core, k) for k in core.w}
    assert core_cls.bf16_copy(core, "l0_D") is copies["l0_D"]
    addresses = {k: v.data_ptr() for k, v in copies.items()}
    for v in core.w.values():
        v.copy_(torch.randn(HEADS))
    core_cls.post_load_weights(core)
    for k, v in copies.items():
        assert v.data_ptr() == addresses[k]
        assert torch.equal(v, core.w[k].to(torch.bfloat16))
