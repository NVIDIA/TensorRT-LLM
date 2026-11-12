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
"""Engine names the dep16 target's glue probes, and the values its folded code runs.

The glue reaches main's engine through duck-typed probes (``hasattr``, ``getattr(..., None)``,
``inspect.signature``). A probe for a name the runtime object does not have returns its fallback,
so the switch behind it stops working without an error. Each test pins the probes of one switch
against the class the dep16 glue gets at runtime, and the values the target folded into its code
(the c383 GEN recipe: attention DP 16, MegaMoE EP16).

CPU only: classes and signatures, no kernels.
"""

import ast
import dataclasses
import importlib
import inspect
import textwrap
from types import SimpleNamespace

import pytest
from torch import nn

K3_RUBIN = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl"
DEP16 = f"{K3_RUBIN}.kimi_k3_nvfp4__sm_107__dep16"


def _k3_module(name):
    """A module of the Kimi K3 family code (``dspark.drafter``, ``kernels.k3r_spec...``)."""
    return importlib.import_module(f"{K3_RUBIN}.{name}")


def _dep16(name):
    """A module of the dep16 target (``modeling``, ``weights``)."""
    return importlib.import_module(f"{DEP16}.{name}")


def _bare(cls):
    """An instance of the ``nn.Module`` ``cls`` without running its constructor."""
    instance = cls.__new__(cls)
    nn.Module.__init__(instance)
    return instance


def _is_property(cls, name):
    return isinstance(inspect.getattr_static(cls, name, None), property)


def _text_and_outer():
    modeling = _dep16("modeling")
    text = _bare(modeling.KimiK3DefaultForCausalLM)
    outer = _bare(modeling.ModelingV2KimiK3Nvfp4Sm107Dep16)
    outer.llm = text
    return text, outer


def test_runtime_hooks_reach_the_engine():
    """The engine arms its hooks only from the ``RuntimeHooks`` it reads off the outermost model
    (``runtime_hooks_of``), the target's registered class. The text model declares none (no fused
    decode head); the target class adds the 4 GiB left outside the PyTorch cap, the c383 GEN
    KV-balanced admission planner and the family's cache manager (KDA mid-chunk snapshots off);
    everything else stays at the default."""
    from tensorrt_llm._torch.pyexecutor.runtime_hooks import DEFAULT_RUNTIME_HOOKS, runtime_hooks_of

    planner_cls = _dep16("modeling").K3Dep16KvBalancedAdmission
    text, outer = _text_and_outer()

    assert text.spec_decode_prologue is None
    assert text.runtime_hooks is None
    assert runtime_hooks_of(text) == DEFAULT_RUNTIME_HOOKS

    hooks = runtime_hooks_of(outer)
    assert hooks.decode_prologue is None
    assert hooks.torch_mem_safety_margin == 4 * 1024**3
    assert type(hooks.kv_balanced_admission_planner) is planner_cls
    # One planner for the process, not one per read.
    assert runtime_hooks_of(outer).kv_balanced_admission_planner is (
        hooks.kv_balanced_admission_planner
    )
    cache = importlib.import_module(
        "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.cache."
        "k3_sm107_cache_manager"
    )
    assert hooks.kv_cache_manager_cls is cache.KimiK3Sm107CacheManager
    assert hooks.kv_cache_manager_cls.KDA_MIDCHUNK_SNAPSHOT_ALIGNMENT == 0
    rest = dataclasses.replace(
        hooks,
        torch_mem_safety_margin=None,
        kv_balanced_admission_planner=None,
        kv_cache_manager_cls=None,
    )
    assert rest == DEFAULT_RUNTIME_HOOKS


def test_dspark_path_is_the_fused_spec_tail():
    """The drafter and worker run the fused spec tail, not the v2 tail."""
    path_cls = _k3_module("dspark.drafter").K3DSparkPath
    text, _ = _text_and_outer()
    assert text.dspark_path == path_cls(spec_tail_fusion=True, tail_v2=False)
    assert text.dspark_path.spec_tail_fusion is True
    assert text.dspark_path.tail_v2 is False


def test_drafter_declarations_reach_the_engine():
    """The KV cache creator reads the drafter's ``draft_kv_dtype`` (FP8 E4M3 draft layers) and
    ``context_kv_only`` (the context server's reserve) off ``model.draft_model`` of the outermost
    model. The target class forwards the text model's drafter, built as a generation server's
    (not context-KV-only)."""
    import torch

    drafter_mod = _k3_module("dspark.drafter")
    shell = _k3_module("dspark.shell")
    assert drafter_mod.K3DSparkDrafter.draft_kv_dtype == torch.float8_e4m3fn
    text, outer = _text_and_outer()
    assert text.dspark_context_kv_only is False
    drafter = SimpleNamespace(draft_kv_dtype=torch.float8_e4m3fn, context_kv_only=False)
    text.__dict__["draft_model"] = drafter
    assert _is_property(type(outer), "draft_model")
    assert outer.draft_model is drafter
    # The shell hands the target's flag to the drafter it builds.
    source = inspect.getsource(shell.KimiK3SpecForCausalLM._build_dspark)
    assert "context_kv_only=self.dspark_context_kv_only" in source


def test_folded_values_match_the_constants_record():
    """``constants.py`` stays as the record of what the glue was folded from; the folded code
    runs the same values."""
    modeling = _dep16("modeling")
    record = _dep16("modeling").CONSTANTS
    outer_cls = modeling.ModelingV2KimiK3Nvfp4Sm107Dep16
    assert outer_cls.K3_CONSTANTS is record
    text, outer = _text_and_outer()
    assert (text.spec_decode_prologue is not None) == record.spec_head_fusion
    assert text.dspark_path.spec_tail_fusion == record.spec_tail_fusion
    assert text.dspark_path.tail_v2 == record.dspark_tail_v2
    assert text.dspark_context_kv_only == record.dspark_context_kv_only
    assert outer.runtime_hooks.torch_mem_safety_margin == record.torch_mem_safety_margin
    assert record.router_bf16_gate is False
    assert record.megamoe_experts is _dep16("modeling").K3MegaMoE


def test_loader_does_not_take_model_defaults_from_k3():
    """The model loader takes model defaults from the class registered for the checkpoint's
    architecture, before it routes a Modeling V2 target. For Kimi K3 that class declares none, so
    every engine setting of a K3 deployment is the recipe's; the dep16 target declares none
    either."""
    from tensorrt_llm._torch.models.modeling_utils import get_registered_model_class

    builtin = get_registered_model_class("KimiK3ForConditionalGeneration")
    assert builtin is not None
    assert not hasattr(builtin, "get_model_defaults")
    assert not hasattr(_dep16("modeling").ModelingV2KimiK3Nvfp4Sm107Dep16, "get_model_defaults")


def _gate_config(**overrides):
    fields = dict(
        num_experts_per_token=4,
        num_experts=32,
        routed_scaling_factor=2.5,
        moe_router_activation_func="sigmoid",
        moe_renormalize=True,
        hidden_size=64,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_router_is_fp32():
    """The gate GEMM runs on FP32 operands (the upcast BF16 input and the FP32 weight) with FP32
    output: under attention DP the BF16 min-latency router GEMM is not used."""
    import torch

    moe_runtime = _dep16("modeling")
    gate = moe_runtime.K3MoEGate(_gate_config(), device="cpu")
    assert gate.weight.dtype == torch.float32
    assert gate.e_score_correction_bias.dtype == torch.float32
    torch.manual_seed(0)
    with torch.no_grad():
        gate.weight.normal_()
    x = torch.randn(3, 64).to(torch.bfloat16)
    logits = gate.compute_logits(x)
    assert logits.dtype == torch.float32
    assert logits.shape == (3, 32)
    torch.testing.assert_close(logits, x.float() @ gate.weight.t(), rtol=0, atol=0)

    runtime = _bare(moe_runtime.K3MoERuntime)
    runtime.gate = gate
    torch.testing.assert_close(runtime._router_logits(x), logits, rtol=0, atol=0)


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"moe_router_activation_func": "softmax"}, "sigmoid"),
        ({"moe_renormalize": False}, "sigmoid"),
        ({"num_expert_group": 2}, "ungrouped"),
        ({"num_experts": 2048}, "1024 experts"),
    ],
)
def test_gate_refuses_routing_it_does_not_run(overrides, match):
    moe_runtime = _dep16("modeling")
    with pytest.raises(ValueError, match=match):
        moe_runtime.K3MoEGate(_gate_config(**overrides), device="cpu")


def test_router_takes_noaux_tc(monkeypatch):
    """Every forward routes with ``noaux_tc_op`` (the routed experts then run ``moe_sort``, no
    sort tables handed on): the CuTe routing tail serves only non-attention-DP experts, so the
    dep16 glue carries no route-tail target."""
    import torch

    moe_runtime = _dep16("modeling")
    calls = []

    def noaux_tc(logits, bias, groups, topk_group, top_k, scale):
        calls.append((groups, topk_group, top_k, scale))
        n = logits.shape[0]
        return torch.ones(n, top_k, dtype=torch.float16), torch.zeros(n, top_k, dtype=torch.int64)

    monkeypatch.setattr(moe_runtime, "noaux_tc_op", noaux_tc)
    gate = moe_runtime.K3MoEGate(_gate_config(), device="cpu")
    ids, weights = gate.route(torch.zeros(2, 32))
    assert calls == [(1, 1, 4, 2.5)]
    assert ids.dtype == torch.int32 and weights.dtype == torch.float32
    assert ids.shape == weights.shape == (2, 4)

    for name in ("_router", "_route_tail_target", "_fused_latent_input"):
        assert not hasattr(moe_runtime.K3MoERuntime, name), name
    assert not hasattr(moe_runtime, "_k3_fc12_options")
    source = inspect.getsource(moe_runtime)
    assert "k3_route_tail" not in source
    assert "SortedRouting(*gate.route(lg), None)" in inspect.getsource(
        moe_runtime.K3MoERuntime.forward
    )


def test_moe_runtime_runs_under_attention_dp_only():
    moe_runtime = _dep16("modeling")
    model_config = SimpleNamespace(mapping=SimpleNamespace(enable_attention_dp=False))
    with pytest.raises(NotImplementedError, match="attention DP"):
        moe_runtime.K3MoERuntime(model_config, SimpleNamespace(), 0, {})


def test_megamoe_takes_what_the_moe_runtime_hands_it():
    """The routed experts are the target's own MegaMoE module; every keyword ``K3MoERuntime``
    builds it with binds to its constructor."""
    moe_runtime = _dep16("modeling")
    megamoe = _dep16("modeling").K3MegaMoE
    assert moe_runtime.K3MegaMoE is megamoe
    assert megamoe.__module__ == f"{DEP16}.modeling"
    tree = ast.parse(textwrap.dedent(inspect.getsource(moe_runtime.K3MoERuntime.__init__)))
    builds = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "K3MegaMoE"
    ]
    assert len(builds) == 1
    (build,) = builds
    assert not build.args
    kwargs = {kw.arg: object() for kw in build.keywords}
    assert None not in kwargs
    inspect.signature(megamoe.__init__).bind(object(), **kwargs)
    for name in ("mapping", "max_num_tokens", "skip_create_weights_in_init", "dtype"):
        assert name in kwargs, name


def test_attn_res_takes_the_shared_sm_reserve():
    """The persistent attention-residual op takes ``cute_dsl_utils.sm_reserve()`` as its
    ``sm_reserve`` argument."""
    import torch

    from tensorrt_llm._torch import cute_dsl_utils

    assert _dep16("modeling").sm_reserve is cute_dsl_utils.sm_reserve
    schema = torch.ops.trtllm.attn_res_add_rmsnorm_persistent_fwd.default._schema
    assert [arg.name for arg in schema.arguments][-1] == "sm_reserve"


def test_fp8_kv_layer_selector_on_the_cache_policy():
    """The MLA layers take their FP8 KV-cache fallback layers from the selector of
    ``kimi_k3_cache_policy`` that the cache manager reads too."""
    from tensorrt_llm._torch import kimi_k3_cache_policy

    assert _dep16("modeling")._kimi_k3_cache_policy is kimi_k3_cache_policy
    selector = getattr(kimi_k3_cache_policy, "get_kimi_k3_fp8_kv_layer_ids", None)
    assert callable(selector)
    assert "nvfp4_enabled" in inspect.signature(selector).parameters


def test_fp4_mla_probe_on_the_family_mla_state():
    """The FP4 MLA paths (packed decode, split-K context) key off ``uses_fp4_mla_attention`` of
    the layer's ``K3MlaAttnState`` roles."""
    state = _k3_module("ops.attention.k3_mla_state").K3MlaAttnState
    assert _dep16("modeling").K3MlaAttnState is state
    assert _is_property(state, "uses_fp4_mla_attention")


def test_kda_inforward_snapshot_plan_on_the_private_metadata():
    """In-forward KDA snapshots (context, block reuse): ``Mamba2Metadata.prepare`` hands the plan
    to ``_prepare_inforward_split``, which the step metadata of the dep16 model
    (``K3MambaMetadata``) turns into a ``kda_snapshot`` plan on the live slots with the plan's
    staging rows; the KDA layers read those rows through ``inforward_snapshot_stage_views``."""
    import torch

    from tensorrt_llm._torch.modules.kimi_kda.cache_manager import KimiK3HybridCacheManager
    from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata
    from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import (
        InforwardSnapshotEntry,
        MambaHybridCacheManagerV2,
    )

    modeling = _dep16("modeling")
    metadata_cls = _dep16("modeling").K3MambaMetadata
    assert modeling.KimiK3DefaultForCausalLM.mamba_metadata_cls is metadata_cls
    assert modeling.ModelingV2KimiK3Nvfp4Sm107Dep16.mamba_metadata_cls is metadata_cls
    assert issubclass(metadata_cls, Mamba2Metadata)

    assert issubclass(KimiK3HybridCacheManager, MambaHybridCacheManagerV2)
    for name in ("inforward_snapshot_stage_views", "state_index", "finish_inforward_snapshots"):
        assert callable(getattr(MambaHybridCacheManagerV2, name, None)), name
    assert "_prepare_inforward_split(" in inspect.getsource(Mamba2Metadata.prepare)
    assert list(inspect.signature(metadata_cls._prepare_inforward_split).parameters) == list(
        inspect.signature(Mamba2Metadata._prepare_inforward_split).parameters
    )

    metadata = metadata_cls.__new__(metadata_cls)
    metadata._state_indices_long = torch.zeros(4, dtype=torch.long)
    metadata.kda_snapshot = None
    manager = SimpleNamespace(conv_state_shape=(12, 3), state_index={11: 7, 13: 2}.__getitem__)
    attn_metadata = SimpleNamespace(seq_lens=torch.tensor([200, 130, 96]), request_ids=[11, 12, 13])
    plan = {
        11: InforwardSnapshotEntry(row=1, point=1152, num_before=128, num_after=72),
        13: InforwardSnapshotEntry(row=0, point=64, num_before=64, num_after=32),
    }
    metadata._prepare_inforward_split(attn_metadata, manager, plan, 3, 3)
    snapshot = metadata.kda_snapshot
    assert snapshot.num_snapshots == 2
    assert snapshot.slots.tolist() == [7, 2]
    assert snapshot.stage_rows.tolist() == [1, 0]
    assert snapshot.token_indices.tolist() == [128, 330 + 64]
    assert snapshot.chunks.tolist() == [2, -1, 1]
    with pytest.raises(RuntimeError, match="generation requests"):
        metadata._prepare_inforward_split(attn_metadata, manager, plan, 3, 4)
    misaligned = {13: InforwardSnapshotEntry(row=0, point=48, num_before=48, num_after=48)}
    with pytest.raises(RuntimeError, match="does not match the plan"):
        metadata._prepare_inforward_split(attn_metadata, manager, misaligned, 3, 3)


# The FP8 block-scale projections of K3's MLA layer and the KDA layer's o_proj; the weight load
# swaps each one for ``_Fp8BlockScaleWeightReadLinear``.
_K3_MLA_PROJECTIONS = ("kv_a_proj_with_mqa", "q_b_proj", "kv_b_proj", "o_proj", "g_proj")
# Read through ``getattr(proj, name, None)``; the bias-free FP8 projection has none.
_OPTIONAL_PROJECTION_READS = frozenset({"bias"})


def _projection_reads():
    import re

    source = inspect.getsource(_dep16("modeling"))
    names = "|".join(_K3_MLA_PROJECTIONS)
    reads = set(re.findall(rf"(?:self|layer)\.(?:{names})\.(\w+)\b", source))
    reads |= set(re.findall(r"\bproj\.(\w+)\b", source))
    reads |= set(
        re.findall(rf"getattr\((?:(?:self|layer)\.(?:{names})|kv_b_proj), \"(\w+)\"", source)
    )
    kda = inspect.getsource(_dep16("modeling"))
    reads |= set(re.findall(r"self\.o_proj\.(\w+)\b", kda))
    reads |= set(re.findall(r"getattr\((?:self\.)?o_proj, \"(\w+)\"", kda))
    return reads


def test_fp8_weight_read_linear_serves_the_k3_projection_reads():
    """The dep16 MLA chains and the KDA MTP verify's o_proj read attributes and methods of their
    projections, and on K3 those are ``_Fp8BlockScaleWeightReadLinear``: every one read exists
    on the target's module (a missing one silently drops the path behind a ``getattr`` probe)."""
    import torch

    reads = _projection_reads()
    assert {
        "weight",
        "weight_scale",
        "in_features",
        "supports_prequantized_input",
        "fp8w_lowm_partial_n_tile",
        "forward_prequantized",
        "takes_prequantized_input",
    } <= reads, reads

    linear = _dep16("modeling")._Fp8BlockScaleWeightReadLinear
    projection = linear(torch.zeros(256, 128, dtype=torch.float8_e4m3fn), torch.ones(2, 1), 256)
    # Set by ``from_linear``, the constructor K3's swap uses.
    projection.quant_config, projection.tp_size, projection.tp_rank = None, 1, 0
    missing = sorted(
        attr
        for attr in reads
        if attr not in _OPTIONAL_PROJECTION_READS and not hasattr(projection, attr)
    )
    assert not missing, missing
