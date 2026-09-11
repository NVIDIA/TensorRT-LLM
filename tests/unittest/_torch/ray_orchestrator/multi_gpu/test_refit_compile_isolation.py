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
"""Isolate refit x torch.compile x piecewise CUDA graph on Qwen3.5 397B at TP8.

Why this exists alongside ``test_llm_update_weights_multi_gpu.py``:

1. The existing PCG regression compares **refit against refit** (eager-refit vs
   PCG-refit). A defect that breaks both legs equally passes. Here the reference
   loads real weights and never refits, so refit bugs cannot cancel.

2. Its metric -- top-20 index overlap averaged over four *free-running* tokens --
   is self-amplifying: two engines that diverge on one argmax then compare
   different contexts at every later position, so a single flipped token reads
   as ~25-48%. That is why the reported numbers are bimodal and why they cannot
   separate a benign near-tie flip from corruption.

   Greedy token equality is not usable either, for the same reason: the
   measured noise floor is two runs of the SAME engine flipping a near-tie
   ("the power of AI" vs "the power of artificial intelligence", top-1 margin
   0.125). The primary metric here is therefore the **first generated token's
   logits over the full vocabulary** -- both sides have consumed an identical
   prompt at that point, so nothing has compounded. Healthy runs sit at
   cosine >= 0.996; the corruption this test was written for sits at 0.13.

torch.compile and PCG are separate switches
(``torch_compile_enabled = bool(torch_compile_config is not None)``;
``enable_piecewise_cuda_graph`` defaults False). The refit hooks used to be
gated on **PCG** rather than on compile, so ``TorchCompileConfig()`` -- compile
on, PCG off -- got no refit lifecycle at all and loaded nothing. Row B is what
caught that; it stays in the matrix as the regression guard.

Run matrix (reference is R0; every comparison changes one variable):

    R0   real weights, eager                 -> ground truth
    R0p  real weights, eager (repeat)        -> noise floor
    R1   real weights, compile, no PCG       -> compile alone, no refit
    R2   real weights, compile + PCG         -> GEN-ONLY WITH PCG, no refit
    A    dummy->refit, eager                 -> refit alone
    B    dummy->refit, compile, no PCG       -> refit x compile
    C    dummy->refit, compile + PCG         -> + capture/replay

R2 answers "does gen-only with PCG+compile emit garbage?" directly -- it is the
shape the gen-only proxy recipe runs, which is believed healthy.

Each engine build also emits a ``*_noreplay`` artifact for the PCG rows: the
same engine, same weights, same allocator state, with PCG replay switched off at
runtime (``set_piecewise_cuda_graph_flag(False)``, which makes every
``PiecewiseRunner`` fall through to ``default_callable``). R2 vs R2_noreplay and
C vs C_noreplay isolate capture/replay with literally everything else held
fixed, and cost one extra generate rather than a second engine build.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import pytest
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

# The tests tree is not a package (no __init__.py); imports are rooted at
# tests/unittest, which the runner puts on PYTHONPATH.
from _torch.ray_orchestrator.multi_gpu.test_llm_update_weights_multi_gpu import (
    _QWEN35_35B_TP8,
    Qwen35_35BTP8WorkerExtension,
    _attach_to_tp8_ray_cluster,
    _mxfp8_397b_model_dir,
    _qwen35_35b_model_kwargs,
    _qwen35_35b_selected_checkpoint_names,
    _qwen35_35b_weight_group,
    _return_local_weight,
    RefQwen35MXFP8ModelWithIPCHandles,
)
from utils.util import skip_pre_blackwell

from tensorrt_llm import LLM
from tensorrt_llm._ray_utils import control_action_decorator
from tensorrt_llm._torch.utils import get_device_uuid
from tensorrt_llm.llmapi import KvCacheConfig, MoeConfig, SamplingParams, TorchCompileConfig
from tensorrt_llm.llmapi.rlhf_utils import WorkerExtension

# Ray leaves daemon threads (ray_print_logs, ray_listen_error_messages) alive
# for the life of the driver; pytest-threadleak fails the test on them after
# the work has already succeeded.
pytestmark = [pytest.mark.threadleak(enabled=False)]

ARTIFACT_DIR = os.environ.get(
    "TLLM_REFIT_ISOLATION_ARTIFACT_DIR",
    "/lustre/fsw/coreai_mlperf_training/erinh/pcg-test/artifacts",
)

# Full 60-layer/512-expert 397B by default -- the reduced 4-layer shape produces
# semantically meaningless text, which makes token-level comparison hard to
# sanity check by eye. Set TLLM_REFIT_ISO_REDUCED=1 for the fast smoke shape.
_REDUCED = bool(os.environ.get("TLLM_REFIT_ISO_REDUCED"))
# MXFP8 is what the production recipe runs (precision: fp8, is_mx: true).
# There is no real-weight reference in this mode: the engine is quantised but
# the checkpoint is BF16, so load_format="auto" cannot serve it. The reference
# is therefore A (eager refit), which still isolates compile/PCG.
_MXFP8 = bool(os.environ.get("TLLM_REFIT_ISO_MXFP8"))

_MAX_SEQ_LEN = 2048
_MAX_NUM_TOKENS = 2048
_MAX_TOKENS = 64
_LONG_PROMPT_TOKENS = 1024
# Weights are ~94 GiB/GPU at TP8 BF16 on the full model; the rest of the 288 GiB
# is available, so this only has to be enough KV for two short sequences.
_KV_FRACTION = 0.05 if _REDUCED else 0.25

_EXTENSION = (
    "_torch.ray_orchestrator.multi_gpu.test_refit_compile_isolation._RefitIsolationExtension"
)


# --------------------------------------------------------------------------- #
#  Full-model weight selection
# --------------------------------------------------------------------------- #
def _full_is_text_weight(name: str) -> bool:
    """Every text weight of the full model: no vision tower, no MTP head."""
    return not name.startswith("model.visual.") and ".mtp." not in name


def _full_selected_checkpoint_names(model_dir: str) -> List[str]:
    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    return sorted(name for name in weight_map if _full_is_text_weight(name))


def _full_weight_group(name: str) -> str:
    """Collapse only the expert index, keeping the layer index.

    The reduced-shape helper also collapses ``.layers.N.`` into ``.layers.*.``,
    which on the full model would put 60 layers x 512 experts of gate_up_proj in
    one bucket -- hundreds of GiB materialised on every worker at once. Keeping
    the layer index bounds a bucket at one layer's expert stack (~8.6 GiB for
    gate_up, ~4.3 GiB for down) while still exercising multi-bucket mapper state.
    """
    return re.sub(r"(\.experts\.)\d+\.", r"\1*.", name)


def _selected_checkpoint_names(model_dir: str) -> List[str]:
    if _REDUCED:
        return _qwen35_35b_selected_checkpoint_names(model_dir)
    return _full_selected_checkpoint_names(model_dir)


def _weight_group(name: str) -> str:
    return _qwen35_35b_weight_group(name) if _REDUCED else _full_weight_group(name)


def _model_kwargs() -> dict:
    kwargs = (
        _qwen35_35b_model_kwargs()
        if _REDUCED
        # Drop the MTP head; the rollout recipe does not speculate.
        else {"text_config": {"mtp_num_hidden_layers": 0}}
    )
    if _MXFP8:
        # Same quantisation scope as the production fp8 recipe
        # (precision: fp8, is_mx: true): routed experts only. linear_attn
        # cannot be MXFP8 at all -- the GDN mixer's in_proj_ba projects to
        # N=16 and the MXFP8 GEMM requires N % 32 == 0.
        kwargs = dict(kwargs)
        kwargs["quantization_config"] = {
            "quant_method": "mxfp8",
            "weight_block_size": [1, 32],
            "ignored_layers": [
                "*self_attn*",
                "*linear_attn*",
                "*mlp.gate",
                "*mlp.shared_expert*",
                "*visual*",
                "*vision*",
                "*embed_tokens*",
                "lm_head",
            ],
        }
    return kwargs


# --------------------------------------------------------------------------- #
#  Worker extension
# --------------------------------------------------------------------------- #
class _RefitIsolationExtension(Qwen35_35BTP8WorkerExtension):
    """Adds a full-model refit bucket loader and a runtime PCG replay switch."""

    def snapshot_param_identity(self) -> Dict[str, Tuple[int, int, int]]:
        """Record ``(id, data_ptr, nbytes)`` for every parameter.

        Diffing this across a refit answers the question every CUDA-graph
        staleness theory depends on: which tensors does refit actually rebind?
        ``pre_reload_weights`` allocates a fresh ``Parameter`` and
        ``register_parameter``s it (``linear.py:565-570``,
        ``fused_moe/quantization.py:695-702``) for every entry in
        ``rebuild_tensor_metadata`` -- which is populated only for parameters a
        transform replaced. Anything captured in a CUDA graph whose address
        moves here is replaying against freed memory.
        """
        model = self.engine.model_engine.model
        snap = {
            f"param:{name}": (id(p), p.data_ptr(), p.numel() * p.element_size())
            for name, p in model.named_parameters()
        }
        snap.update(
            {
                f"buffer:{name}": (id(b), b.data_ptr(), b.numel() * b.element_size())
                for name, b in model.named_buffers()
            }
        )
        # Plain tensor attributes are the blind spot. Dynamo lifts parameters
        # and buffers as graph inputs, but a bare tensor attribute (e.g.
        # Qwen3.5's ``_fused_norm_weight``, cached by
        # ``_precompute_fused_norm_weights``) can be baked into the FX graph as
        # a constant -- so rebinding one is invisible to the compiled callable
        # in a way that rebinding a Parameter is not.
        registered = set()
        for mod in model.modules():
            registered.update(id(t) for t in mod.parameters(recurse=False))
            registered.update(id(t) for t in mod.buffers(recurse=False))
        for mod_name, mod in model.named_modules():
            for attr, value in list(vars(mod).items()):
                if isinstance(value, torch.Tensor) and id(value) not in registered:
                    key = f"attr:{mod_name}.{attr}" if mod_name else f"attr:{attr}"
                    snap[key] = (
                        id(value),
                        value.data_ptr(),
                        value.numel() * value.element_size(),
                    )
        return snap

    def piecewise_capture_stats(self) -> Dict[str, int]:
        """How many PiecewiseRunner entries currently hold a live CUDA graph.

        The `unwrap_only` refit lifecycle deliberately does NOT clear the
        piecewise captures, on the reasoning that refit never moves a tensor so
        the captures stay valid. That keeps PWCG hot instead of falling back to
        the uncaptured path until something recaptures -- but it is a claim
        about performance, not correctness, and a run that silently stopped
        replaying would still produce correct output. This measures it:
        `captured` should stay non-zero across a refit.
        """
        engine = self.engine.model_engine
        backend = getattr(engine, "_torch_compile_backend", None)
        if backend is None:
            return {"runners": 0, "entries": 0, "captured": 0}
        runners = list(getattr(backend, "_piecewise_runners", []) or [])
        entries = captured = 0
        for runner in runners:
            for entry in getattr(runner, "entries", {}).values():
                entries += 1
                if getattr(entry, "cuda_graph", None) is not None:
                    captured += 1
        return {"runners": len(runners), "entries": entries, "captured": captured}

    def set_piecewise_replay(self, enable: bool) -> None:
        """Toggle PCG replay without rebuilding or recapturing.

        With ``enable=False`` every ``PiecewiseRunner.__call__`` short-circuits
        to ``default_callable`` -- the same FX submodule executed eagerly -- so a
        comparison against ``enable=True`` differs only by graph replay.
        """
        from tensorrt_llm._torch.utils import set_piecewise_cuda_graph_flag

        set_piecewise_cuda_graph_flag(enable)

    @control_action_decorator
    def update_weights_nemo_rl_style(
        self, model_dir: str, weight_group: Optional[str] = None
    ) -> None:
        """Replicate NeMo-RL's refit sequence exactly, lifecycle and all.

        Mirrors ``NcclExtension.update_weights_from_collective``
        (RL/nemo_rl/models/generation/trtllm/trtllm_backend.py): synchronize,
        ``begin_update_weights``, the ``pre_reload_weights`` walk, ``reload``,
        then ``_finalize_weight_update`` -- and, critically, **no**
        ``begin_weight_update`` / ``finish_weight_update``, so
        ``_remove_torch_compile()`` never runs.

        This exists to demonstrate the production failure directly instead of
        inferring it from the tekit-side rows: with torch.compile enabled this
        path should hit the guard in ``ModelLoader.reload`` (or, without the
        guard, silently load nothing and generate garbage).
        """
        model_engine = self.engine.model_engine
        model = model_engine.model
        if weight_group is None:
            model_engine.model_loader.finalize_update_weights()
            for module in model.modules():
                if hasattr(module, "process_weights_after_loading") and not getattr(
                    module, "_weights_removed", False
                ):
                    module.process_weights_after_loading()
                if hasattr(module, "post_load_weights") and not getattr(
                    module, "_weights_removed", False
                ):
                    module.post_load_weights()
            if os.environ.get("TLLM_REFIT_ISO_NEMO_RL_FIXED"):
                ext = _rl_nccl_extension()
                if ext is not None:
                    ext._restore_compiled_model_after_refit(self)
                    print("[REFIT_ISO] used RL_3 _restore_compiled_model_after_refit",
                          flush=True)
                else:
                    model_engine.restore_compiled_model_after_refit(
                        self.engine.resource_manager
                    )
            if hasattr(model, "_nemo_rl_style_begun"):
                delattr(model, "_nemo_rl_style_begun")
            return

        torch.cuda.synchronize()
        if not hasattr(model, "_nemo_rl_style_begun"):
            # TLLM_REFIT_ISO_NEMO_RL_FIXED mirrors the fix applied to
            # NcclExtension in RL_3: unwrap torch.compile before any weight
            # loading, because the wrapper renames parameter paths and
            # load_weights matches by path.
            if os.environ.get("TLLM_REFIT_ISO_NEMO_RL_FIXED"):
                ext = _rl_nccl_extension()
                if ext is not None:
                    # The real RL_3 implementation, not a copy of it.
                    fn = getattr(ext, "_unwrap_compiled_model_for_refit", None) or ext._release_compiled_model_for_refit
                    fn(self)
                    print("[REFIT_ISO] used RL_3 unwrap helper",
                          flush=True)
                else:
                    model_engine.unwrap_compiled_model_for_refit()
            model_engine.model_loader.begin_update_weights()
            for module in model.modules():
                if hasattr(module, "pre_reload_weights") and not getattr(
                    module, "_weights_removed", False
                ):
                    module.pre_reload_weights()
            setattr(model, "_nemo_rl_style_begun", True)

        weights = dict(self._load_full_group(model_dir, weight_group))
        model_engine.model_loader.reload(model, weights, allow_partial_loading=True)
        torch.cuda.current_stream().synchronize()

    @control_action_decorator
    def update_mxfp8_weights_from_full_checkpoint(
        self, model_dir: str, weight_group: Optional[str] = None
    ) -> None:
        """Full-model MXFP8 bucket: quantise BF16 checkpoint bytes on the worker.

        Mirrors ``update_mxfp8_weights_from_local_checkpoint`` but over the
        full-model weight filter. The checkpoint stores routed experts as one
        stacked tensor per layer (``mlp.experts.gate_up_proj``); the engine
        wants per-expert ``gate_proj``/``up_proj``/``down_proj`` plus UE8M0
        scales, so expand and quantise in expert chunks to bound peak memory.
        """
        if weight_group is None:
            WorkerExtension.update_weights.__wrapped__(self, None)
            return

        checkpoint_weights = self._load_full_group(model_dir, weight_group)
        local_weights: List[Tuple[str, torch.Tensor]] = []
        for name, weight in checkpoint_weights:
            match = re.fullmatch(
                r"(?P<prefix>.*\.mlp\.experts)\.(?P<projection>gate_up_proj|down_proj)", name
            )
            if match is None:
                local_weights.append((name, weight))
                continue

            prefix, projection = match.group("prefix"), match.group("projection")
            for start in range(0, weight.shape[0], 8):
                end = min(start + 8, weight.shape[0])
                if projection == "gate_up_proj":
                    half = weight.shape[1] // 2
                    projections = (
                        ("gate_proj", weight[start:end, :half]),
                        ("up_proj", weight[start:end, half:]),
                    )
                else:
                    projections = (("down_proj", weight[start:end]),)
                for proj_name, proj_weight in projections:
                    data, scale = RefQwen35MXFP8ModelWithIPCHandles._quantize_2d(proj_weight)
                    for offset, expert_id in enumerate(range(start, end)):
                        wn = f"{prefix}.{expert_id}.{proj_name}.weight"
                        local_weights.append((wn, data[offset]))
                        local_weights.append(
                            (wn.removesuffix(".weight") + ".weight_scale_inv", scale[offset])
                        )

        device_uuid = get_device_uuid(self.device_id)
        handles = {
            device_uuid: [
                (name, (_return_local_weight, (weight, None, None, None, None, None, None)))
                for name, weight in local_weights
            ]
        }
        WorkerExtension.update_weights.__wrapped__(self, handles)

    def _load_full_group(self, model_dir: str, weight_group: str):
        with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
            weight_map = json.load(f)["weight_map"]
        selected_by_shard: Dict[str, List[str]] = {}
        for name in _full_selected_checkpoint_names(model_dir):
            if _full_weight_group(name) == weight_group:
                selected_by_shard.setdefault(weight_map[name], []).append(name)
        device = torch.device("cuda", self.device_id)
        out: List[Tuple[str, torch.Tensor]] = []
        for shard, names in sorted(selected_by_shard.items()):
            with safe_open(os.path.join(model_dir, shard), framework="pt", device="cpu") as ckpt:
                out.extend((n, ckpt.get_tensor(n).to(device)) for n in sorted(names))
        return out

    @control_action_decorator
    def update_weights_from_full_checkpoint(
        self, model_dir: str, weight_group: Optional[str] = None
    ) -> None:
        """Load one full-model bucket beside each worker (no cross-node IPC)."""
        if weight_group is None:
            WorkerExtension.update_weights.__wrapped__(self, None)
            return

        with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
            weight_map = json.load(f)["weight_map"]

        selected_by_shard: Dict[str, List[str]] = {}
        for name in _full_selected_checkpoint_names(model_dir):
            if _full_weight_group(name) == weight_group:
                selected_by_shard.setdefault(weight_map[name], []).append(name)

        device = torch.device("cuda", self.device_id)
        local_weights: List[Tuple[str, torch.Tensor]] = []
        for shard, names in sorted(selected_by_shard.items()):
            with safe_open(os.path.join(model_dir, shard), framework="pt", device="cpu") as ckpt:
                local_weights.extend(
                    (name, ckpt.get_tensor(name).to(device)) for name in sorted(names)
                )

        device_uuid = get_device_uuid(self.device_id)
        handles = {
            device_uuid: [
                (name, (_return_local_weight, (weight, None, None, None, None, None, None)))
                for name, weight in local_weights
            ]
        }
        WorkerExtension.update_weights.__wrapped__(self, handles)


# --------------------------------------------------------------------------- #
#  Prompts / sampling
# --------------------------------------------------------------------------- #
def _rl_nccl_extension():
    """RL_3's NcclExtension class, when its checkout is on PYTHONPATH.

    Used to validate the **actual** NeMo-RL refit-lifecycle helpers against a
    real engine rather than a replica of them. The methods only touch
    ``self.engine.model_engine`` / ``self.engine.resource_manager``, so they can
    be invoked unbound with this module's worker extension as ``self`` -- which
    avoids needing a trainer peer, an NCCL model-update group or ZMQ.
    """
    try:
        from nemo_rl.models.generation.trtllm.trtllm_backend import NcclExtension
    except Exception as exc:  # noqa: BLE001 - optional, reported by the caller
        print(f"[REFIT_ISO] RL NcclExtension unavailable: {exc!r}", flush=True)
        return None
    return NcclExtension


def _prompts(model_dir: str) -> List[List[int]]:
    """One short and one long prompt, as token ids.

    The long prompt matters: PCG only ever executes on context batches whose
    padded token count is in ``capture_num_tokens``, so a five-token prompt
    barely exercises it.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    short = tokenizer.encode("The future of AI is")
    filler = (
        "A distributed inference engine keeps a paged key-value cache so that "
        "attention over long conversations does not recompute the whole prefix. "
    )
    long_ids = tokenizer.encode(filler * 64)[:_LONG_PROMPT_TOKENS]
    del tokenizer
    return [short, long_ids]


def _sampling_params() -> SamplingParams:
    # temperature == 0 is an explicit greedy control
    # (SamplingParams.params_imply_explicit_greedy), so decoding is argmax with
    # no RNG: two correct engines must emit identical token ids.
    return SamplingParams(temperature=0, return_generation_logits=True, max_tokens=_MAX_TOKENS)


# --------------------------------------------------------------------------- #
#  Engine spec
# --------------------------------------------------------------------------- #
class _Engine:
    def __init__(self, refit: bool, compile_kind: Optional[str], rows: List[Tuple[str, Optional[bool]]]):
        self.refit = refit
        self.compile_kind = compile_kind  # None | "compile" | "pcg"
        self.rows = rows  # (artifact_tag, pcg_replay_override or None)

    def compile_config(self) -> Optional[TorchCompileConfig]:
        if self.compile_kind is None:
            return None
        if self.compile_kind == "compile":
            return TorchCompileConfig()
        return TorchCompileConfig(enable_piecewise_cuda_graph=True)


_ENGINES: Dict[str, _Engine] = {
    "eager_real": _Engine(False, None, [("R0", None), ("R0p", None)]),
    "compiled_real": _Engine(False, "compile", [("R1", None)]),
    "pcg_real": _Engine(False, "pcg", [("R2", True), ("R2_noreplay", False)]),
    "eager_refit": _Engine(True, None, [("A", None)]),
    "compiled_refit": _Engine(True, "compile", [("B", None)]),
    "pcg_refit": _Engine(True, "pcg", [("C", True), ("C_noreplay", False)]),
}


if _MXFP8:
    # Drop the load_format="auto" rows; keep the refit rows and use A as ref.
    _ENGINES = {k: v for k, v in _ENGINES.items() if v.refit}
_REFERENCE_TAG = "A" if _MXFP8 else "R0"


def _artifact_path(tag: str) -> str:
    suffix = ("_reduced" if _REDUCED else "") + ("_mxfp8" if _MXFP8 else "")
    if os.environ.get("TLLM_REFIT_ISO_NO_CUDA_GRAPHS"):
        suffix += "_nocg"
    if os.environ.get("TLLM_REFIT_ISO_NO_UB"):
        suffix += "_noub"
    return os.path.join(ARTIFACT_DIR, f"{tag}{suffix}.pt")


def _save(tag: str, token_ids: List[List[int]], logits: List[torch.Tensor]) -> None:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    torch.save({"tag": tag, "token_ids": token_ids, "logits": logits}, _artifact_path(tag))
    print(f"[REFIT_ISO] wrote {_artifact_path(tag)}", flush=True)
    for i, ids in enumerate(token_ids):
        print(f"[REFIT_ISO] {tag} prompt{i} first16={ids[:16]}", flush=True)


def _generate(llm, prompts, sampling_params):
    outputs = llm.generate(prompts, sampling_params)
    token_ids = [list(o.outputs[0].token_ids) for o in outputs]
    logits = [o.outputs[0].generation_logits.detach().float().cpu() for o in outputs]
    return token_ids, logits


def _try_snapshot(llm) -> Optional[dict]:
    """Best-effort parameter-identity snapshot; diagnostics must never fail a row."""
    try:
        return llm._collective_rpc("snapshot_param_identity", ())[0]
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        print(f"[REFIT_ISO][param-identity] snapshot unavailable: {exc!r}", flush=True)
        return None


def _report_param_identity(before: dict, after: dict) -> None:
    """Print which parameters refit rebound (new object) or moved (new address).

    A CUDA graph captured before refit bakes the *address*. Any parameter whose
    ``data_ptr`` changes here and is reachable from a captured graph replays
    against memory the caching allocator has since handed to something else.
    """
    moved, rebound, resized = [], [], []
    for name, (obj, ptr, nbytes) in after.items():
        if name not in before:
            continue
        old_obj, old_ptr, old_nbytes = before[name]
        if old_nbytes != nbytes:
            resized.append(name)
        if old_ptr != ptr:
            moved.append(name)
        if old_obj != obj:
            rebound.append(name)
    print(
        f"[REFIT_ISO][param-identity] total={len(after)} "
        f"address_moved={len(moved)} object_rebound={len(rebound)} resized={len(resized)}",
        flush=True,
    )
    for label, names in (("address_moved", moved), ("object_rebound", rebound),
                         ("resized", resized)):
        if names:
            kinds = {}
            for n in names:
                kinds.setdefault(n.split(":", 1)[0], []).append(n)
            for kind, group in kinds.items():
                print(
                    f"[REFIT_ISO][param-identity] {label} kind={kind} "
                    f"count={len(group)} e.g. {group[:6]}",
                    flush=True,
                )


def _run_engine(engine: _Engine) -> None:
    model_dir = _mxfp8_397b_model_dir()
    if not os.path.isdir(model_dir):
        pytest.skip(f"Model directory {model_dir} does not exist")

    prompts = _prompts(model_dir)
    sampling_params = _sampling_params()

    llm_kwargs: Dict[str, object] = {}
    compile_config = engine.compile_config()
    if compile_config is not None:
        llm_kwargs["torch_compile_config"] = compile_config
    # Isolation control. A (eager refit) passes *with* decode CUDA graphs on, so
    # if B (compiled refit) becomes clean once they are off, the fault is the
    # compiled callable captured inside a decode CUDA graph rather than anything
    # in the FX graph itself -- which both Dynamo probes have already exonerated.
    if os.environ.get("TLLM_REFIT_ISO_NO_CUDA_GRAPHS"):
        llm_kwargs["cuda_graph_config"] = None
    # Userbuffers ride in on TorchCompileConfig (enable_userbuffers defaults
    # True) and tekit_2's build_custom_passes has no non-MPI gate, unlike
    # 2690b38e48 on user/zongfeij/rl. R2 is clean with UB on, so UB alone is not
    # the fault, but this separates UB from the rest of compile.
    if os.environ.get("TLLM_REFIT_ISO_NO_UB") and compile_config is not None:
        compile_config.enable_userbuffers = False

    with LLM(
        model=model_dir,
        ray_worker_extension_cls=_EXTENSION,
        orchestrator_type="ray",
        tensor_parallel_size=_QWEN35_35B_TP8,
        # MEP8 matches the production recipe's moe_expert_parallel_size.
        moe_expert_parallel_size=_QWEN35_35B_TP8,
        load_format="dummy" if engine.refit else "auto",
        pipeline_parallel_size=1,
        max_batch_size=2,
        max_seq_len=_MAX_SEQ_LEN,
        max_num_tokens=_MAX_NUM_TOKENS,
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=False,
            free_gpu_memory_fraction=_KV_FRACTION,
            mamba_ssm_cache_dtype="float32",
        ),
        model_kwargs=_model_kwargs(),
        moe_config=MoeConfig(backend="CUTLASS"),
        enable_chunked_prefill=True,
        disable_mm_encoder=True,
        **llm_kwargs,
    ) as llm:
        # Production refits once per RL step, so the lifecycle has to be
        # repeatable -- a one-shot test would miss state that only breaks on
        # reuse (stale first_pre_reload_weights, captures not re-established,
        # mapper state carried across cycles).
        refit_cycles = int(os.environ.get("TLLM_REFIT_ISO_CYCLES", "1"))
        if engine.refit:
            groups = sorted({_weight_group(n) for n in _selected_checkpoint_names(model_dir)})
            print(f"[REFIT_ISO] refit buckets: {len(groups)} cycles: {refit_cycles}",
                  flush=True)
            if os.environ.get("TLLM_REFIT_ISO_NEMO_RL_STYLE"):
                # Demonstrate the production refit sequence, which never calls
                # the TensorRT-LLM refit lifecycle (see the method docstring).
                rpc = "update_weights_nemo_rl_style"
            elif _REDUCED:
                rpc = ("update_mxfp8_weights_from_local_checkpoint" if _MXFP8
                       else "update_weights_from_local_checkpoint")
            else:
                rpc = ("update_mxfp8_weights_from_full_checkpoint" if _MXFP8
                       else "update_weights_from_full_checkpoint")
            # Match NeMo-RL: capture over the dummy model before refit.
            llm.generate(prompts, sampling_params)
            for cycle in range(refit_cycles):
                # Diagnostic only: never let it fail the row it is measuring.
                before = _try_snapshot(llm)
                bucket_start = time.time()
                for group in groups:
                    llm._collective_rpc(rpc, (model_dir, group))
                bucket_s = time.time() - bucket_start
                # The final (weight_group=None) call is where
                # finalize_weight_update and finish_weight_update run -- the
                # unwrap/recompile/recapture lifecycle. Timing it separately
                # from bucket streaming is what makes its cost visible, which is
                # the number the "don't recapture" work has to beat.
                finalize_start = time.time()
                llm._collective_rpc(rpc, (model_dir, None))
                finalize_s = time.time() - finalize_start
                print(
                    f"[REFIT_ISO][timing] cycle={cycle} buckets={len(groups)} "
                    f"stream_s={bucket_s:.1f} finalize_and_lifecycle_s={finalize_s:.1f}",
                    flush=True,
                )
                after = _try_snapshot(llm)
                if before and after:
                    _report_param_identity(before, after)
                try:
                    stats = llm._collective_rpc("piecewise_capture_stats", ())[0]
                    print(f"[REFIT_ISO][pcg-hot] after refit: {json.dumps(stats)}", flush=True)
                except Exception as exc:  # noqa: BLE001 - diagnostic only
                    print(f"[REFIT_ISO][pcg-hot] unavailable: {exc!r}", flush=True)
                if refit_cycles > 1:
                    # Generate between cycles so a cycle that leaves the engine
                    # in a bad state shows up here rather than being masked by
                    # the next refit repairing it.
                    ids, _ = _generate(llm, prompts, sampling_params)
                    print(f"[REFIT_ISO] after cycle {cycle} first8={ids[0][:8]}",
                          flush=True)

        for tag, replay in engine.rows:
            if replay is not None:
                llm._collective_rpc("set_piecewise_replay", (replay,))
            token_ids, logits = _generate(llm, prompts, sampling_params)
            assert all(torch.isfinite(t).all() for t in logits), f"{tag}: non-finite logits"
            _save(tag, token_ids, logits)
        # Leave the global flag as we found it for any later engine in-process.
        if any(replay is not None for _, replay in engine.rows):
            llm._collective_rpc("set_piecewise_replay", (True,))


@pytest.mark.high_cuda_memory
@skip_pre_blackwell
@pytest.mark.parametrize("engine_key", list(_ENGINES))
def test_refit_isolation_engine(monkeypatch, engine_key):
    """Build one engine for ``engine_key`` and persist its greedy outputs."""
    _attach_to_tp8_ray_cluster(monkeypatch)
    _run_engine(_ENGINES[engine_key])


# --------------------------------------------------------------------------- #
#  Report
# --------------------------------------------------------------------------- #
def _compare(ref: dict, test: dict) -> List[dict]:
    """Primary signal is position 0; token equality is secondary colour.

    Measured noise floor (R0 vs R0', two generates on the *same* engine, full
    397B, TP8): position-0 ``rel_l2`` 0.017-0.087 and one prompt flipping at
    token 20 on a top-1 margin of 0.125. So greedy token equality is **not** a
    valid oracle here -- run-to-run numeric jitter flips near-ties, and whether
    it does is luck.

    Position 0 is the clean comparison: both engines have consumed the
    identical prompt, so nothing has compounded yet and the batching-order
    jitter shows up directly as a logit distance rather than through an argmax.
    Corruption looks nothing like the floor -- it collapses ``pos0_cosine`` and
    drives ``pos0_rel_l2`` towards 1.
    """
    reports = []
    for i, (ref_ids, test_ids) in enumerate(zip(ref["token_ids"], test["token_ids"])):
        ref_logits, test_logits = ref["logits"][i], test["logits"][i]
        a0, b0 = test_logits[0].float(), ref_logits[0].float()
        top2_0 = torch.topk(b0, 2).values
        report = {
            "prompt": i,
            # --- primary: position 0, identical prompt, nothing compounded ---
            "pos0_top1_match": bool(a0.argmax() == b0.argmax()),
            "pos0_rel_l2": ((a0 - b0).norm() / b0.norm().clamp_min(1e-6)).item(),
            "pos0_cosine": torch.nn.functional.cosine_similarity(a0, b0, dim=0).item(),
            "pos0_max_abs_diff": (a0 - b0).abs().max().item(),
            "pos0_ref_margin": (top2_0[0] - top2_0[1]).item(),
            # --- secondary ---
            "exact_match": ref_ids == test_ids,
            "ref_len": len(ref_ids),
            "test_len": len(test_ids),
        }
        if not report["exact_match"]:
            common = min(len(ref_ids), len(test_ids))
            d = next((k for k in range(common) if ref_ids[k] != test_ids[k]), common)
            report["first_divergence"] = d
            ref_logits, test_logits = ref["logits"][i], test["logits"][i]
            if d < min(ref_logits.shape[0], test_logits.shape[0]):
                a, b = test_logits[d].float(), ref_logits[d].float()
                top2 = torch.topk(b, 2).values
                # The number that separates a near-tie flip from corruption.
                report["ref_top1_margin"] = (top2[0] - top2[1]).item()
                report["max_abs_diff"] = (a - b).abs().max().item()
                report["rel_l2"] = ((a - b).norm() / b.norm().clamp_min(1e-6)).item()
                report["cosine"] = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
                report["chosen_rank_in_ref"] = (b > b[test_ids[d]]).sum().item()
        reports.append(report)
    return reports


def _print_decoded(tags: List[str]) -> None:
    """Decode each artifact's greedy continuation so garbage is visible by eye."""
    tokenizer = AutoTokenizer.from_pretrained(_mxfp8_397b_model_dir())
    for tag in dict.fromkeys(tags):
        path = _artifact_path(tag)
        if not os.path.exists(path):
            continue
        data = torch.load(path, weights_only=False)
        for i, ids in enumerate(data["token_ids"]):
            text = tokenizer.decode(ids)
            print(f"[REFIT_ISO] --- {tag} prompt{i} ---", flush=True)
            print(f"[REFIT_ISO] {text!r}", flush=True)


def test_refit_isolation_decode():
    """Print decoded text for every artifact present (no assertions)."""
    tags = [tag for engine in _ENGINES.values() for tag, _ in engine.rows]
    if not any(os.path.exists(_artifact_path(tag)) for tag in tags):
        pytest.skip("no artifacts yet")
    _print_decoded(tags)


def test_refit_isolation_report():
    """Compare every persisted row against R0 and print the isolation table."""
    if not os.path.exists(_artifact_path(_REFERENCE_TAG)):
        pytest.skip(f"{_REFERENCE_TAG} reference artifact not present yet")
    ref = torch.load(_artifact_path(_REFERENCE_TAG), weights_only=False)

    tags = [tag for engine in _ENGINES.values() for tag, _ in engine.rows]
    table = {
        tag: _compare(ref, torch.load(_artifact_path(tag), weights_only=False))
        for tag in tags
        if tag != _REFERENCE_TAG and os.path.exists(_artifact_path(tag))
    }

    print(f"\n[REFIT_ISO] ==== isolation report (reference = {_REFERENCE_TAG}) ====", flush=True)
    for tag, reports in table.items():
        for report in reports:
            print(f"[REFIT_ISO] {tag} {json.dumps(report)}", flush=True)

    # Decoded text: the metrics say how far apart two runs are, but only the
    # text says whether a run is garbage in the way the e2e rollouts were.
    print("\n[REFIT_ISO] ==== decoded output ====", flush=True)
    _print_decoded([_REFERENCE_TAG] + list(table))

    summary = os.path.join(ARTIFACT_DIR, "report.json")
    with open(summary, "w") as f:
        json.dump(table, f, indent=2)
    print(f"[REFIT_ISO] wrote {summary}", flush=True)

    if "R0p" in table:
        floor = max(r["pos0_rel_l2"] for r in table["R0p"])
        print(f"[REFIT_ISO] noise floor pos0_rel_l2 = {floor:.4g}", flush=True)
        # Corruption drives pos0_rel_l2 towards 1 and collapses cosine; the
        # measured floor is ~0.09. 10x the floor separates the two regimes by
        # an order of magnitude without pretending the floor is zero.
        threshold = max(10 * floor, 0.5)
        bad = {
            tag: reports
            for tag, reports in table.items()
            if any(r["pos0_rel_l2"] > threshold or r["pos0_cosine"] < 0.9 for r in reports)
        }
        assert not bad, f"rows exceed {threshold:.3g} pos0_rel_l2 or cosine<0.9: {json.dumps(bad, indent=2)}"
