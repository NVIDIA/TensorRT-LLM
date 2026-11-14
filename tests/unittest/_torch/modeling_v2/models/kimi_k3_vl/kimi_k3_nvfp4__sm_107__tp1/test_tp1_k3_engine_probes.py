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
"""What the engine reads off the tp1 target (every recipe's context servers), and the values its
code runs.

The engine reads the runtime hooks and the drafter's declarations off the outermost model; the
DSpark shell reads the DSpark path and the context-KV-only flag off the text model. Each test pins
those against the tp1 glue (``kimi_k3_nvfp4__sm_107__tp1``) and its constants record. The probes
the tp1 glue shares in shape with the helix8 glue (the routing tail, the MLA projection reads) and
the family's own probes are in ``common/test_k3_engine_probes.py``.

CPU only: classes and signatures, no kernels.
"""

import dataclasses
import importlib
import inspect
from types import SimpleNamespace

from torch import nn

K3_RUBIN = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl"
TP1 = f"{K3_RUBIN}.kimi_k3_nvfp4__sm_107__tp1"


def _k3_module(name):
    """A module of the Kimi K3 family code (``dspark.drafter``, ``kernels.k3r_spec...``)."""
    return importlib.import_module(f"{K3_RUBIN}.{name}")


def _tp1(name):
    """A module of the tp1 target (``modeling``, ``weights``)."""
    return importlib.import_module(f"{TP1}.{name}")


def _bare(cls):
    """An instance of the ``nn.Module`` ``cls`` without running its constructor."""
    instance = cls.__new__(cls)
    nn.Module.__init__(instance)
    return instance


def _is_property(cls, name):
    return isinstance(inspect.getattr_static(cls, name, None), property)


def _text_and_outer():
    modeling = _tp1("modeling")
    text = _bare(modeling.KimiK3DefaultForCausalLM)
    outer = _bare(modeling.ModelingV2KimiK3Nvfp4Sm107Tp1)
    outer.llm = text
    return text, outer


def test_runtime_hooks_reach_the_engine():
    """The engine arms its hooks only from the ``RuntimeHooks`` it reads off the outermost model
    (``runtime_hooks_of``), the target's registered class. The text model declares none (no fused
    decode head); the target class adds the 4 GiB left outside the PyTorch cap and the context
    servers' recurrent-state cache policies (recurrent pages of prompts under 16384 tokens at the
    lowest priority, exact demotion) and their cache manager (the family's, with KDA mid-chunk
    snapshots at the 64-token prefill chunk); everything else stays at the default."""
    from tensorrt_llm._torch.pyexecutor.runtime_hooks import (
        DEFAULT_RUNTIME_HOOKS,
        HybridCachePolicy,
        runtime_hooks_of,
    )

    text, outer = _text_and_outer()
    assert text.spec_decode_prologue is None
    assert text.runtime_hooks is None
    assert runtime_hooks_of(text) == DEFAULT_RUNTIME_HOOKS

    hooks = runtime_hooks_of(outer)
    assert hooks.decode_prologue is None
    assert hooks.torch_mem_safety_margin == 4 * 1024**3
    assert hooks.kv_balanced_admission_planner is None
    assert hooks.hybrid_cache == HybridCachePolicy(low_priority_snapshot_prompt_len=16384)
    assert hooks.exact_context_demotion is True
    cache = _k3_module("ops.cache.k3_sm107_cache_manager")
    assert hooks.kv_cache_manager_cls is cache.KimiK3Sm107MidchunkCacheManager
    assert hooks.kv_cache_manager_cls.KDA_MIDCHUNK_SNAPSHOT_ALIGNMENT == 64
    rest = dataclasses.replace(
        hooks,
        torch_mem_safety_margin=None,
        hybrid_cache=DEFAULT_RUNTIME_HOOKS.hybrid_cache,
        exact_context_demotion=DEFAULT_RUNTIME_HOOKS.exact_context_demotion,
        kv_cache_manager_cls=None,
    )
    assert rest == DEFAULT_RUNTIME_HOOKS


def test_dspark_path_is_the_plain_spec_tail():
    """The drafter and worker of a context server run the plain DSpark tail (no fused spec tail,
    no v2 tail)."""
    path_cls = _k3_module("dspark.drafter").K3DSparkPath
    text, _ = _text_and_outer()
    assert text.dspark_path == path_cls(spec_tail_fusion=False, tail_v2=False)


def test_drafter_declarations_reach_the_engine():
    """The KV cache creator reads the drafter's ``draft_kv_dtype`` (FP8 E4M3 draft layers) and
    ``context_kv_only`` (the context server's reserve) off ``model.draft_model`` of the outermost
    model. The target class forwards the text model's drafter, built context-KV-only (tp1 is the
    context server)."""
    import torch

    drafter_mod = _k3_module("dspark.drafter")
    shell = _k3_module("dspark.shell")
    assert drafter_mod.K3DSparkDrafter.draft_kv_dtype == torch.float8_e4m3fn
    text, outer = _text_and_outer()
    assert text.dspark_context_kv_only is True
    drafter = SimpleNamespace(draft_kv_dtype=torch.float8_e4m3fn, context_kv_only=True)
    text.__dict__["draft_model"] = drafter
    assert _is_property(type(outer), "draft_model")
    assert outer.draft_model is drafter
    # The shell hands the target's flag to the drafter it builds.
    source = inspect.getsource(shell.KimiK3SpecForCausalLM._build_dspark)
    assert "context_kv_only=self.dspark_context_kv_only" in source


def test_folded_values_match_the_constants_record():
    """``CONSTANTS`` records the optimizations the target's code is built for; the code runs the
    same values."""
    modeling = _tp1("modeling")
    record = _tp1("modeling").CONSTANTS
    assert modeling.ModelingV2KimiK3Nvfp4Sm107Tp1.K3_CONSTANTS is record
    text, outer = _text_and_outer()
    assert (text.spec_decode_prologue is not None) == record.spec_head_fusion
    assert text.dspark_path.spec_tail_fusion == record.spec_tail_fusion
    assert text.dspark_path.tail_v2 == record.dspark_tail_v2
    assert text.dspark_context_kv_only == record.dspark_context_kv_only
    assert outer.runtime_hooks.torch_mem_safety_margin == record.torch_mem_safety_margin
    assert record.router_bf16_gate is True
    assert record.megamoe_experts is None
    assert not record.ld_fused and not record.fc12_die_major and record.shared_sm == 0


def test_moe_runtime_carries_no_generation_only_paths():
    """The context target's routed experts run the CUTEDSL kernels: no FC12 option, no fused
    latent-down input (``ld_fused``), no MegaMoE."""
    moe_runtime = _tp1("modeling")
    assert not hasattr(moe_runtime, "_k3_fc12_options")
    assert not hasattr(moe_runtime.K3MoERuntime, "_fused_latent_input")
    assert "K3MegaMoE" not in inspect.getsource(moe_runtime)
