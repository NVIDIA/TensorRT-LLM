# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ULL target's model, host-side: its layer schedule, the generation step's route check, the
layer-0 dense slice and its loader, the folded MLA projection's layout, the fused accept's
checkpoint record, the feature profile and the target class (no GPU)."""

import ast
import dataclasses
import inspect
import pathlib
from types import SimpleNamespace as NS

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4__sm_107__helix16_moetp16ep1 import (  # noqa: E501
    modeling as decode,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4__sm_107__helix16_moetp16ep1 import (  # noqa: E501
    modeling as layer0,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4__sm_107__helix16_moetp16ep1 import (  # noqa: E501
    modeling as mlain,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k3_vl.ops.common import k3_ull_features as features

pytestmark = pytest.mark.cpu_only

_T = "tensorrt_llm._torch.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4__sm_107__helix16_moetp16ep1"  # noqa: E501
_T_DIR = pathlib.Path(layer0.__file__).resolve().parent

#: Kimi K3's MLA layers (0-indexed).
MLA_LAYERS = tuple(range(3, 91, 4)) + (91, 92)


def _k3_text_config(num_layers: int = 93, mla=MLA_LAYERS):
    kda = [i + 1 for i in range(num_layers) if i not in mla]
    return NS(
        num_hidden_layers=num_layers,
        first_k_dense_replace=1,
        linear_attn_config={
            "kda_layers": kda,
            "full_attn_layers": [i + 1 for i in mla if i < num_layers],
            "num_heads": 96,
            "head_dim": 128,
        },
    )


# ------------------------------------------------------------------------------------------------
# Layer schedule
# ------------------------------------------------------------------------------------------------


def test_layer_kinds_of_kimi_k3():
    from importlib import import_module

    modeling = import_module(f"{_T}.modeling")
    kinds = modeling.layer_kinds(_k3_text_config())
    assert len(kinds) == 93
    assert kinds[0] == modeling.OURS
    assert [i for i, k in enumerate(kinds) if k == modeling.MOE] == list(MLA_LAYERS)
    assert kinds.count(modeling.CHAINED) == 68
    # The chain's constants agree (chain.py exports them).
    chain = import_module(_T.rpartition(".")[0] + ".ops.chain.k3_ull_chain")
    assert (chain.CHAIN_OURS, chain.CHAIN_CHAINED, chain.CHAIN_MOE) == (
        modeling.OURS,
        modeling.CHAINED,
        modeling.MOE,
    )


@pytest.mark.parametrize(
    "cfg, match",
    [
        (_k3_text_config(mla=(1,) + MLA_LAYERS), "layer 1 must be a KDA layer"),
        (_k3_text_config(num_layers=91), "last an MLA layer"),
        (_k3_text_config(mla=(0,) + MLA_LAYERS), "layer 0"),
    ],
)
def test_layer_kinds_rejects_other_schedules(cfg, match):
    from importlib import import_module

    modeling = import_module(f"{_T}.modeling")
    with pytest.raises(ValueError, match=match):
        modeling.layer_kinds(cfg)


def test_small_schedule_for_the_gpu_test():
    """The 4-layer model of ``test_k3_ull_small_model_graph.py``: KDA + dense, the chain's entry
    layer, a chained layer whose consumer stores the MLA input, the MLA layer (the last)."""
    from importlib import import_module

    modeling = import_module(f"{_T}.modeling")
    kinds = modeling.layer_kinds(_k3_text_config(num_layers=4))
    assert kinds == (modeling.OURS, modeling.CHAINED, modeling.CHAINED, modeling.MOE)


# ------------------------------------------------------------------------------------------------
# The step route
# ------------------------------------------------------------------------------------------------


class _Chain:
    rows, q = (8, 16, 24, 32), 8

    def supports(self, rows, q):
        return q == self.q and rows in self.rows


@pytest.mark.parametrize("requests", [1, 2, 3, 4])
def test_plan_step_serves_the_generation_steps(requests):
    md = NS(num_contexts=0, num_seqs=requests)
    assert decode.plan_step(md, 8 * requests, _Chain()) == decode.Step(8 * requests, requests, 8)


@pytest.mark.parametrize(
    "num_contexts, num_seqs, rows, match",
    [
        (1, 1, 100, "generation-only"),
        (0, 2, 9, "same number of tokens"),
        (0, 1, 1, "not a step the decode chain serves"),
        (0, 5, 40, "not a step the decode chain serves"),
        (0, 0, 8, "same number of tokens"),
    ],
)
def test_plan_step_rejects(num_contexts, num_seqs, rows, match):
    md = NS(num_contexts=num_contexts, num_seqs=num_seqs)
    with pytest.raises(RuntimeError, match=match):
        decode.plan_step(md, rows, _Chain())


# ------------------------------------------------------------------------------------------------
# Layer 0's dense slice
# ------------------------------------------------------------------------------------------------


def test_dense_slice_of_each_rank_is_its_tp4_sub_slice():
    """Rank r takes sub-slice r // 4 of TP4 shard r % 4 (the deployment's TP4 dense MLP within a
    node of 4 GPUs, split once more over the 4 nodes): every 16th of the intermediate exactly
    once."""
    inter, world, per_node = 33792, 16, 4
    tp = layer0.dense_tp_size(inter, world, per_node)
    assert tp == 4
    shard, sub = inter // tp, inter // world
    seen = set()
    for rank in range(world):
        index = layer0.dense_slice_index(rank, world, tp)
        lo = index * sub
        tp_rank, sub_rank = rank % tp, rank // tp
        assert lo == tp_rank * shard + sub_rank * sub
        seen.add(index)
    assert seen == set(range(world))


def test_dense_tp_size_within_a_node():
    assert layer0.dense_tp_size(33792, 16, 8) == 8
    assert layer0.dense_tp_size(33792, 16, 16) == 16
    assert layer0.dense_tp_size(48, 16, 4) == 4


def _fp8_block(n, k, seed):
    gen = torch.Generator().manual_seed(seed)
    codes = (torch.randn(n, k, generator=gen) * 2).to(torch.float8_e4m3fn)
    scale = torch.exp2(torch.randint(-8, -2, (-(-n // 128), -(-k // 128)), generator=gen).float())
    return codes, scale


@pytest.mark.parametrize("dim, lo, hi", [(0, 100, 228), (0, 0, 128), (1, 130, 300), (1, 256, 384)])
def test_checkpoint_slice_dequantizes_the_covering_blocks(dim, lo, hi):
    from importlib import import_module

    weights = import_module(f"{_T}.weights")
    codes, scale = _fp8_block(256, 384, seed=dim * 7 + lo)
    ckpt = {"w.weight": codes, "w.weight_scale": scale}
    full = (
        codes.float() * scale.repeat_interleave(128, 0).repeat_interleave(128, 1)[:256, :384]
    ).to(torch.bfloat16)
    got = weights._checkpoint_slice("w.weight", ckpt, lo, hi, dim)
    want = full[lo:hi] if dim == 0 else full[:, lo:hi]
    assert got.dtype == torch.bfloat16
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))


def test_checkpoint_slice_of_a_bf16_tensor():
    from importlib import import_module

    weights = import_module(f"{_T}.weights")
    w = torch.randn(64, 96).to(torch.bfloat16)
    assert torch.equal(weights._checkpoint_slice("w", {"w": w}, 10, 40, 1), w[:, 10:40])
    with pytest.raises(ValueError):
        weights._checkpoint_slice("w", {"w": w}, 40, 10, 0)


# ------------------------------------------------------------------------------------------------
# The folded MLA input projection
# ------------------------------------------------------------------------------------------------


def _packed_scale(n, k, seed):
    gen = torch.Generator().manual_seed(seed)
    words = -(-(k // 128) // 4)
    return torch.randint(0, 2**31 - 1, (words, n), generator=gen, dtype=torch.int32).t()


def test_folded_in_proj_layout():
    k = 1024
    n_a, n_g = 2112, 768
    w_a = torch.randn(n_a, k).to(torch.float8_e4m3fn)
    w_g = torch.randn(n_g, k).to(torch.float8_e4m3fn)
    kv_a = NS(weight=w_a, packed_weight_scale=_packed_scale(n_a, k, 0))
    g = NS(weight=w_g, packed_weight_scale=_packed_scale(n_g, k, 1))
    fold = mlain.FoldedInProj(kv_a, g)
    assert (fold.n_qkv, fold.pad_rows, fold.n_gate, fold.sigmoid_tile0) == (2112, 2176, 768, 17)
    assert tuple(fold.weight.shape) == (2944, k)
    as_bytes = fold.weight.view(torch.uint8)
    assert torch.equal(as_bytes[:n_a], w_a.view(torch.uint8))
    assert not as_bytes[n_a:2176].any()
    assert torch.equal(as_bytes[2176:], w_g.view(torch.uint8))
    assert fold.scale.shape == (2944, 2) and fold.scale.stride(0) == 1
    assert torch.equal(fold.scale[:n_a], kv_a.packed_weight_scale)
    assert (fold.scale[n_a:2176] == 0x7F7F7F7F).all()
    assert torch.equal(fold.scale[2176:], g.packed_weight_scale)


def test_folded_in_proj_rejects():
    k = 1024
    w_a = torch.randn(2112, k).to(torch.float8_e4m3fn)
    s = _packed_scale(2112, k, 0)
    assert not mlain.foldable(w_a, torch.randn(100, k).to(torch.float8_e4m3fn), s, s)
    assert not mlain.foldable(w_a.to(torch.bfloat16), w_a, s, s)
    assert not mlain.foldable(w_a, w_a, s.t().contiguous().t().contiguous(), s)


# ------------------------------------------------------------------------------------------------
# The fused accept's KDA record
# ------------------------------------------------------------------------------------------------


def test_fused_accept_records_the_checkpoint_state():
    """The DSpark fused accept writes ``prev_num_accepted_tokens`` of the cache manager's
    checkpoint state (the next step's checkpoint index), as for the runtime's replay state."""
    from importlib import import_module

    cache = import_module(_T.rpartition(".")[0] + ".ops.cache.k3_ull_cache_manager")
    dflash = import_module(f"{_T}.modeling")

    state = object.__new__(cache.K3UllKdaCheckpointState)
    state.prev_num_accepted_tokens = torch.zeros(6, dtype=torch.int32)
    mgr = object.__new__(cache.K3UllCacheManager)
    mgr.kda_context_only = False
    mgr.local_num_mamba_layers = 2
    mgr._speculative_state = state
    mask = torch.zeros(4, dtype=torch.bool)
    mgr._dummy_request_mask_buffer = NS(tensor=mask)
    slots = torch.tensor([3, 1, 4, 2], dtype=torch.int32)
    md = NS(kv_cache_manager=mgr, mamba_metadata=NS(state_indices=slots))
    fusable, replay = dflash.K3DSparkWorker._kda_replay_operands(md)
    assert fusable
    assert replay[0] is state.prev_num_accepted_tokens
    assert replay[1] is slots and replay[2] is mask


# ------------------------------------------------------------------------------------------------
# Feature profile and static checks
# ------------------------------------------------------------------------------------------------


def _sources():
    return sorted(_T_DIR.rglob("*.py"))


def _readers():
    """The target's files and the family files that read its profile (the kernels and ops it
    brought into the family import ``k3_ull_features``)."""
    family = _T_DIR.parent
    profile = pathlib.Path(features.__file__).resolve()
    return _sources() + [
        p
        for p in sorted(family.rglob("*.py"))
        if _T_DIR not in p.parents and p != profile and "k3_ull_features" in p.read_text()
    ]


def test_every_feature_field_is_read():
    """No profile field without a reader in the target or the family code it runs."""
    text = "\n".join(p.read_text() for p in _readers())
    unread = [f.name for f in dataclasses.fields(features.K3Features) if f".{f.name}" not in text]
    assert unread == []


def test_profile_and_retired_env():
    assert features.active() is features.PROFILE
    assert features.PROFILE.ull3_mla_gate_at == "exchange:16,start"
    for name in ("TRTLLM_K3_FUSE_MOE_RESIDUAL", "TRTLLM_K3_ULL_HYGIENE", "TLLM_SPEC_HEAD_FUSION"):
        assert name in features.RETIRED_ENV


def test_imports_no_sibling_target():
    """The target imports itself and the family's shared code (``ops/``, ``kernels/``,
    ``common/``, ``dspark/``), never another target of the family."""
    family = _T.rsplit(".", 1)[0]
    targets = {p.name for p in _T_DIR.parent.iterdir() if p.is_dir() and "__sm_" in p.name}
    outside = []
    for p in _sources():
        for node in ast.walk(ast.parse(p.read_text())):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level:
                module = ".".join(
                    [*_T.split(".")[: len(_T.split(".")) + 1 - node.level], node.module or ""]
                )
            else:
                module = node.module or ""
            if not module.startswith(f"{family}."):
                continue
            top = module[len(family) + 1 :].split(".")[0]
            if top in targets and top != _T.rsplit(".", 1)[1]:
                outside.append(f"{p.relative_to(_T_DIR)}:{node.lineno} {module}")
    assert not outside


def test_modeling_v2_imports_resolve_in_this_tree():
    """Every absolute modeling_v2 import of the target, the MR8 copy
    and its function-level imports included, names a module of this tree."""
    import tensorrt_llm

    root = pathlib.Path(tensorrt_llm.__file__).resolve().parent.parent
    missing = []
    for p in sorted(_T_DIR.rglob("*.py")):
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                modules = [node.module or ""]
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            else:
                continue
            for module in modules:
                if not module.startswith("tensorrt_llm._torch.modeling_v2"):
                    continue
                base = root.joinpath(*module.split("."))
                if not (base.with_suffix(".py").is_file() or base.is_dir()):
                    missing.append(f"{p.relative_to(_T_DIR)}:{node.lineno} {module}")
    assert not missing, missing


def test_no_kda_or_moe_modules():
    """The KDA + MoE layers run on the chain: no KDA / MoE module of the __default target here."""
    names = {p.relative_to(_T_DIR).as_posix() for p in _sources()}
    assert not names & {"kda.py", "moe.py", "moe_runtime.py"}


def test_target_class():
    from importlib import import_module

    from tensorrt_llm._torch.modeling_v2 import _router_index

    modeling = import_module(f"{_T}.modeling")
    cache = import_module(_T.rpartition(".")[0] + ".ops.cache.k3_ull_cache_manager")
    target = modeling.ModelingV2KimiK3Nvfp4Sm107Helix16Moetp16ep1
    assert target in _router_index._V2_DISABLED_BY_DEFAULT
    # The KV cache manager with KDA checkpoints, declared through the text model's runtime hooks
    # (RuntimeHooks.kv_cache_manager_cls), which the target class returns as its own.
    assert modeling.K3UllCacheManager is cache.K3UllCacheManager
    hooks_source = inspect.getsource(modeling.KimiK3Helix16UllForCausalLM.runtime_hooks.fget)
    assert "kv_cache_manager_cls=K3UllCacheManager" in hooks_source
    assert "self.llm.runtime_hooks" in inspect.getsource(target.runtime_hooks.fget)
    assert set(target.SUPPORTED_SMS) == {(10, 7), (10, 3)}
    assert target.mamba_metadata_cls is decode.K3UllMambaMetadata
    assert target.get_preferred_kv_cache_manager_version() == "V2"
    assert target.get_preferred_transceiver_runtime() == "PYTHON"
