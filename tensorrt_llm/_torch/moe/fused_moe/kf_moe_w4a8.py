# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Default-off alternative kernel for the MXFP4-weight x MXFP8-activation MoE layer (gpt-oss geometry).

``TLLM_MOE_KF_DECODE=1`` routes ``mxe4m3_mxe2m1_block_scale_moe_runner`` calls that match the gpt-oss
geometry (hidden 2880, intermediate 2880, SwiGLU with clamp) and carry at most
``TLLM_MOE_KF_DECODE_MAX_TOKENS`` tokens (default 16384) to a kernel module with the contract

    run(hidden_states [T, 3072] e4m3, hidden_states_scale [T, 96] uint8, topk_ids [T, K] int32,
        topk_weights [T, K] bf16, gemm1_weights, gemm1_weights_scale, gemm1_bias, gemm1_alpha, gemm1_beta,
        gemm1_clamp_limit, gemm2_weights, gemm2_weights_scale, gemm2_bias, local_expert_offset: int,
        requant_round_up: int, output [T, 2880] bf16) -> None

on the engine's prepared trtllm-gen weight layout, i.e. the same tensors the trtllm-gen op reads.
``TLLM_MOE_KF_DECODE_MODULE`` names the module (dotted name, or a path to a .py file); the default is the
in-tree ``tensorrt_llm._torch.cute_dsl_kernels.kf_moe_w4a8``. ``TLLM_MOE_KF_DECODE_RECIPE`` selects the FC1
epilogue's MXFP8 scale recipe the trtllm-gen cubin on this GPU uses (0: floor(log2(amax)) - 8, 1:
ceil(log2(amax / 448)); default 1, measured for the sm100f cubin on SM107). Anything else, or a module that
cannot load, keeps the trtllm-gen op.

``TLLM_MOE_KF_DECODE_DEP=1`` (independent, default off) routes expert-parallel calls (fewer than 128 local
experts) to ``TLLM_MOE_KF_DECODE_DEP_MODULE`` (default ``tensorrt_llm._torch.cute_dsl_kernels.kf_moe_w4a8_dep``),
a kernel tuned for 20-40 tokens per expert that also accepts the 4-byte-aligned per-rank slices of the
per-expert alpha/beta/limit arrays. With it unset, every call behaves as before.

``TLLM_MOE_KF_DECODE_ROUTING=1`` (default off; only with ``TLLM_MOE_KF_DECODE=1``) hands the bf16 router logits of a
call that has no ``topk_ids`` (Renormalize routing, no routing bias, all 128 experts local) to the kernel module's
``run(..., router_logits=...)``, which routes in-kernel (top-4, then softmax over the four; ties to the lower expert
id, as the trtllm-gen routing) instead of the torch topk/softmax/cast kernels below. The module must set
``SUPPORTS_ROUTER_LOGITS = True``; otherwise, and for any other logits layout, the torch routing stays. Calls the
prefill route takes get their ids and weights from the same routing kernel, launched alone (the decode module's
``route``), when the module has it.

``TLLM_MOE_KF_PREFILL=1`` (independent, default off) routes calls with all 128 experts local and
``TLLM_MOE_KF_PREFILL_MIN_TOKENS`` (default 32768) to 131072 tokens to a large-batch kernel module with the same
contract: ``TLLM_MOE_KF_PREFILL_MODULE`` (default ``tensorrt_llm._torch.cute_dsl_kernels.kf_moe_w4a8_prefill``) and
``TLLM_MOE_KF_PREFILL_RECIPE`` (default 1). Where it applies, it takes the call from ``TLLM_MOE_KF_DECODE``, so
``TLLM_MOE_KF_DECODE_MAX_TOKENS=N`` with ``TLLM_MOE_KF_PREFILL_MIN_TOKENS=N+1`` splits the token range between them.

Both serve both forms of the op: the finalized call (``maybe_run``) and the unfinalized one
(``maybe_run_unfinalized``) that ``TLLM_GPTOSS_FUSE_FINALIZE_NORM`` requests, for which the module's
``run_unfinalized`` returns the unfinalized triple and the caller's fused kernel does the finalize, residual add and
RMSNorm (the expert-parallel kernel keeps finalized calls only). During the autotuner's tuning pass the prefill route,
and the decode route on the unfinalized form, leave their calls to the trtllm-gen op, so it runs the same tuning calls
as without the flags (any one of them profiles every bucket up to tune_max_num_tokens; this matters only when the
routes would otherwise take every tuning call). For the prefill route (both forms) and the decode route's unfinalized form, the module load, the
first engagement per form and the first refusal per reason each print one ``[TLLM_MOE_KF_PREFILL]`` or
``[TLLM_MOE_KF_DECODE]`` line to stderr, whatever ``TLLM_LOG_LEVEL`` says.
"""

import importlib
import importlib.util
import os
import sys
from typing import List, Optional

import torch

from tensorrt_llm.logger import logger

_ENV = "TLLM_MOE_KF_DECODE"
_DEFAULT_MODULE = "tensorrt_llm._torch.cute_dsl_kernels.kf_moe_w4a8"
_DEP_ENV = "TLLM_MOE_KF_DECODE_DEP"
_DEP_DEFAULT_MODULE = "tensorrt_llm._torch.cute_dsl_kernels.kf_moe_w4a8_dep"
_ROUTE_ENV = "TLLM_MOE_KF_DECODE_ROUTING"
_PREFILL_ENV = "TLLM_MOE_KF_PREFILL"
_PREFILL_DEFAULT_MODULE = "tensorrt_llm._torch.cute_dsl_kernels.kf_moe_w4a8_prefill"
_NUM_EXPERTS = 128  # gpt-oss-120b; fewer local experts means an expert-parallel rank
_HIDDEN, _INTER, _HIDDEN_PAD, _INTER_PAD = 2880, 2880, 3072, 2944
_TOP_K = 4
# The kernel's 32-bit row offsets overflow above ~180k tokens; keep a margin whatever MAX_TOKENS says.
_KERNEL_MAX_TOKENS = 131072
_UNFINALIZED_FORM = "unfinalized, TLLM_GPTOSS_FUSE_FINALIZE_NORM"
_state = {}


def _layout_ok(tensors, weights):
    """The kernel takes raw pointers that assume dense rows and 16-byte (32 for weights) alignment."""
    return (all(t.is_contiguous() and t.data_ptr() % 16 == 0 for t in tensors)
            and all(t.is_contiguous() and t.data_ptr() % 32 == 0 for t in weights))


def _settings():
    if "settings" not in _state:
        _state["settings"] = (
            os.environ.get(_ENV, "0") == "1",
            int(os.environ.get("TLLM_MOE_KF_DECODE_MAX_TOKENS", "16384")),
            int(os.environ.get("TLLM_MOE_KF_DECODE_RECIPE", "1")),
            os.environ.get("TLLM_MOE_KF_DECODE_MODULE", _DEFAULT_MODULE),
            os.environ.get(_DEP_ENV, "0") == "1",
            os.environ.get("TLLM_MOE_KF_DECODE_DEP_MODULE", _DEP_DEFAULT_MODULE),
            os.environ.get(_ROUTE_ENV, "0") == "1",
        )
    return _state["settings"]


def _prefill_settings():
    if "prefill_settings" not in _state:
        _state["prefill_settings"] = (
            os.environ.get(_PREFILL_ENV, "0") == "1",
            int(os.environ.get("TLLM_MOE_KF_PREFILL_MIN_TOKENS", "32768")),
            int(os.environ.get("TLLM_MOE_KF_PREFILL_RECIPE", "1")),
            os.environ.get("TLLM_MOE_KF_PREFILL_MODULE", _PREFILL_DEFAULT_MODULE),
        )
    return _state["prefill_settings"]


def _say(env, key, msg):
    """One stderr line per process and key (the logger's default level, error, would hide a warning)."""
    said = _state.setdefault("said", set())
    if (env, key) not in said:
        said.add((env, key))
        print(f"[TRT-LLM] [W] [{env}] {msg}", file=sys.stderr, flush=True)


def _tuning():
    """True inside the autotuner's tuning pass (engine warmup)."""
    if "autotuner_get" not in _state:
        try:
            from tensorrt_llm._torch.autotuner import AutoTuner
            _state["autotuner_get"] = AutoTuner.get
        except Exception:  # noqa: BLE001 - no autotuner, no tuning pass
            _state["autotuner_get"] = None
    get = _state["autotuner_get"]
    return get is not None and bool(get().is_tuning_mode)


def _prefill_window(T, local_num_experts, local_expert_offset, form):
    """With TLLM_MOE_KF_PREFILL on: True when it may take this call, False when not (first refusal per reason
    printed), None in the autotuner's tuning pass (the trtllm-gen op runs, to be tuned as without the flag)."""
    min_tokens = _prefill_settings()[1]
    if local_num_experts != _NUM_EXPERTS or local_expert_offset != 0:
        _say(_PREFILL_ENV, "refused-ep", f"refused ({form}): expert-parallel shard ({local_num_experts} local experts "
                                         f"from {local_expert_offset}); such calls keep the trtllm-gen op or "
                                         "TLLM_MOE_KF_DECODE[_DEP]")
        return False
    if T < min_tokens:
        _say(_PREFILL_ENV, "refused-min", f"refused ({form}): T={T} below MIN_TOKENS={min_tokens}; such calls keep "
                                          "the trtllm-gen op or TLLM_MOE_KF_DECODE")
        return False
    if T > _KERNEL_MAX_TOKENS:
        _say(_PREFILL_ENV, "refused-max", f"refused ({form}): T={T} above the kernel's {_KERNEL_MAX_TOKENS} tokens; "
                                          "such calls keep the trtllm-gen op")
        return False
    if _tuning():
        _say(_PREFILL_ENV, "deferred-tuning", f"deferred ({form}): autotuner tuning pass (T={T}); the trtllm-gen op "
                                              "runs and is tuned as without the flag")
        return None
    return True


def _contract(T, prefill, max_tokens, hidden_states, hidden_states_scale, gemm1_weights, gemm1_bias, gemm1_alpha,
              gemm1_beta, gemm1_clamp_limit, gemm2_weights, gemm2_bias, top_k, valid_hidden_size,
              valid_intermediate_size, local_num_experts, gated_act_type):
    """None when the call has the kernels' geometry and operands, else why not."""
    if not (hidden_states.dtype == torch.float8_e4m3fn and hidden_states.dim() == 2
            and hidden_states.shape[1] == _HIDDEN_PAD):
        return f"activations {hidden_states.dtype} {tuple(hidden_states.shape)}, not MXFP8 [T, {_HIDDEN_PAD}]"
    if not (prefill or 1 <= T <= min(max_tokens, _KERNEL_MAX_TOKENS)):
        return f"T={T} outside the decode window [1, {min(max_tokens, _KERNEL_MAX_TOKENS)}]"
    if not (top_k == _TOP_K and gated_act_type == 0):
        return f"top_k {top_k}, activation {gated_act_type}: not top-4 SwiGLU"
    if (gemm1_clamp_limit is None or gemm1_bias is None or gemm2_bias is None or gemm1_alpha is None
            or gemm1_beta is None):
        return "a bias, alpha, beta or clamp limit is missing"
    if not (valid_hidden_size == _HIDDEN and valid_intermediate_size in (None, _INTER)):
        return f"hidden {valid_hidden_size}, intermediate {valid_intermediate_size}: not {_HIDDEN}/{_INTER}"
    if not (tuple(gemm1_weights.shape) == (local_num_experts, 2 * _INTER_PAD, _HIDDEN_PAD // 2)
            and tuple(gemm2_weights.shape) == (local_num_experts, _INTER_PAD, _INTER_PAD // 2)):
        return f"weights {tuple(gemm1_weights.shape)}, {tuple(gemm2_weights.shape)}: not the prepared layout"
    if hidden_states_scale is None or hidden_states_scale.numel() != T * _HIDDEN_PAD // 32:
        return "activation scales missing or not [T, 96]"
    return None


def _kernel(dep=False, prefill=False, entry="run"):
    key, env = (("kernel_prefill", _PREFILL_ENV) if prefill else ("kernel_dep", _DEP_ENV) if dep
                else ("kernel", _ENV))
    if key not in _state:
        name = _prefill_settings()[3] if prefill else _settings()[5 if dep else 3]
        try:
            if name.endswith(".py"):
                spec = importlib.util.spec_from_file_location(f"tllm_kf_moe_w4a8_{key}", name)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            else:
                module = importlib.import_module(name)
            module.run  # noqa: B018 - every module has the finalized entry
            _state[key] = module
            _state[key + "_routes"] = bool(getattr(module, "SUPPORTS_ROUTER_LOGITS", False))
            limits = (f"min_tokens={_prefill_settings()[1]} recipe={_prefill_settings()[2]}" if prefill
                      else f"max_tokens={_settings()[1]} recipe={_settings()[2]}")
            if prefill:
                _say(env, "loaded", f"module loaded: module={name} {limits}")
            else:
                # warning level so the one engagement line shows at the default log level
                logger.warning(f"[{env}] active: module={name} {limits}")
        except Exception as e:  # noqa: BLE001 - any load failure keeps the trtllm-gen op
            _state[key] = None
            msg = f"kernel module {name} unavailable ({type(e).__name__}: {e}); keeping the trtllm-gen op"
            if prefill:
                _say(env, "unavailable", msg)
            else:
                logger.warning(f"[{env}] {msg}")
    module = _state[key]
    return None if module is None else getattr(module, entry, None)


def _maybe_run(router_logits, routing_bias, hidden_states, hidden_states_scale, gemm1_weights,
               gemm1_weights_scale, gemm1_bias, gemm1_alpha, gemm1_beta, gemm1_clamp_limit, gemm2_weights,
               gemm2_weights_scale, gemm2_bias, top_k, valid_hidden_size, valid_intermediate_size,
               local_expert_offset, local_num_experts, routing_method_type, gated_act_type, topk_weights,
               topk_ids, output, unfinalized):
    on, max_tokens, recipe, d_name, dep_on, _, route_on = _settings()
    p_on, p_min_tokens, p_recipe, p_name = _prefill_settings()
    if unfinalized:
        # The expert-parallel kernel has no unfinalized entry (and the fused finalize never runs expert-parallel).
        dep_on = False
    dep = dep_on and local_num_experts < _NUM_EXPERTS
    T = hidden_states.shape[0]
    form = _UNFINALIZED_FORM if unfinalized else "finalized"
    prefill = p_on and _prefill_window(T, local_num_experts, local_expert_offset, form)
    if prefill is None or not (on or dep or prefill):
        return None
    if prefill:
        recipe = p_recipe
    # Lines for the prefill route (both forms) and the decode route's unfinalized form; the decode routes'
    # finalized form keeps its own logging.
    tag = _PREFILL_ENV if prefill else (_ENV if unfinalized and not dep else None)
    reason = _contract(T, prefill, max_tokens, hidden_states, hidden_states_scale, gemm1_weights, gemm1_bias,
                       gemm1_alpha, gemm1_beta, gemm1_clamp_limit, gemm2_weights, gemm2_bias, top_k,
                       valid_hidden_size, valid_intermediate_size, local_num_experts, gated_act_type)
    if reason is not None:
        if tag:
            _say(tag, "refused-contract", f"refused ({form}): {reason}")
        return None
    if unfinalized and not prefill and _tuning():
        _say(_ENV, "deferred-tuning", f"deferred ({form}): autotuner tuning pass (T={T}); the trtllm-gen op runs and "
                                      "is tuned as without the flag")
        return None
    logits_in_kernel = None
    routed_alone = False
    if topk_ids is None:
        # Renormalize routing (gpt-oss): top-k over the logits, softmax over the k selected, bf16 weights.
        if router_logits is None or routing_bias is not None or routing_method_type != 1:
            if tag:
                _say(tag, "refused-routing", f"refused ({form}): routing method {routing_method_type}, bias "
                                             f"{routing_bias is not None}, logits {router_logits is not None}; needs "
                                             "renormalize top-k over logits without bias")
            return None
        routable = (route_on and on and not dep and local_num_experts == _NUM_EXPERTS
                    and int(local_expert_offset) == 0
                    and router_logits.dtype == torch.bfloat16 and tuple(router_logits.shape) == (T, _NUM_EXPERTS)
                    and router_logits.is_contiguous() and router_logits.data_ptr() % 16 == 0
                    and _kernel(False) is not None and _state.get("kernel_routes", False))
        route_alone = getattr(_state.get("kernel"), "route", None) if routable and prefill else None
        if routable and not prefill:
            # TLLM_MOE_KF_DECODE_ROUTING: the kernel routes from the logits itself
            logits_in_kernel = router_logits
            if "route_logged" not in _state:
                _state["route_logged"] = True
                logger.warning(f"[{_ROUTE_ENV}] active: top-{top_k} + softmax in the kernel from bf16 router "
                               f"logits [T, {_NUM_EXPERTS}] (no torch routing kernels)")
        elif route_alone is not None:
            # TLLM_MOE_KF_DECODE_ROUTING on a prefill-route call: the same routing kernel, launched alone
            topk_ids, topk_weights = route_alone(router_logits)
            routed_alone = True
        else:
            values, topk_ids = torch.topk(router_logits.float(), top_k, dim=-1)
            topk_ids = topk_ids.to(torch.int32)
            topk_weights = torch.softmax(values, dim=-1).to(torch.bfloat16)
    if logits_in_kernel is None and (topk_weights is None or topk_weights.dtype != torch.bfloat16):
        if tag:
            _say(tag, "refused-weights", f"refused ({form}): routing weights "
                                         f"{None if topk_weights is None else topk_weights.dtype}, not bf16")
        return None
    # The DEP kernel reads the per-expert alpha/beta/limit scalars one at a time, so 4-byte alignment suffices
    # (rank slices of the [128] arrays start at 43 / 86 floats).
    scalars = (gemm1_alpha, gemm1_beta, gemm1_clamp_limit)
    routing = (logits_in_kernel,) if logits_in_kernel is not None else (topk_ids, topk_weights)
    if not (_layout_ok((hidden_states, hidden_states_scale) + routing + (gemm1_bias, gemm2_bias)
                       + (() if dep else scalars) + ((output,) if output is not None else ()),
                       (gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale))
            and (not dep or all(t.is_contiguous() and t.data_ptr() % 4 == 0 for t in scalars))):
        if "layout_warned" not in _state:
            _state["layout_warned"] = True
            logger.warning(f"[{_ENV}] non-contiguous or misaligned MoE input; keeping the trtllm-gen op for such calls")
        if tag:
            _say(tag, "refused-layout", f"refused ({form}): non-contiguous or misaligned MoE input")
        return None
    run = _kernel(dep, prefill, "run_unfinalized" if unfinalized else "run")
    if run is None:
        if unfinalized and _state.get("kernel_prefill" if prefill else "kernel") is not None:
            _say(tag, "refused-entry", f"refused ({form}): module {p_name if prefill else d_name} has no "
                                       "run_unfinalized")
        return None
    hss = hidden_states_scale.view(T, _HIDDEN_PAD // 32)
    if unfinalized and prefill:
        rows = run(hidden_states, hss, topk_ids.contiguous(), gemm1_weights, gemm1_weights_scale, gemm1_bias,
                   gemm1_alpha, gemm1_beta, gemm1_clamp_limit, gemm2_weights, gemm2_weights_scale, gemm2_bias,
                   int(local_expert_offset), recipe)
        # The unfinalized op's triple: [FC2 rows, routing weights [T, 4] bf16, expanded_idx_to_permuted_idx
        # [T, 4] int32]. Row t * 4 + k is token t's k-th expert, so the index map is the identity; the window
        # admits all 128 experts local only, so every row is written and no index is -1.
        idx = torch.arange(T * _TOP_K, dtype=torch.int32, device=hidden_states.device).view(T, _TOP_K)
        result = [rows, topk_weights.contiguous(), idx]
    elif unfinalized:
        # The decode module returns the triple itself: [FC2 rows, routing weights, pair -> row map (-1: no row)].
        if logits_in_kernel is not None:
            result = run(hidden_states, hss, None, None, gemm1_weights, gemm1_weights_scale, gemm1_bias,
                         gemm1_alpha, gemm1_beta, gemm1_clamp_limit, gemm2_weights, gemm2_weights_scale,
                         gemm2_bias, int(local_expert_offset), recipe, router_logits=logits_in_kernel)
        else:
            result = run(hidden_states, hss, topk_ids.contiguous(), topk_weights.contiguous(), gemm1_weights,
                         gemm1_weights_scale, gemm1_bias, gemm1_alpha, gemm1_beta, gemm1_clamp_limit,
                         gemm2_weights, gemm2_weights_scale, gemm2_bias, int(local_expert_offset), recipe)
    else:
        out = output if output is not None else torch.empty(T, _HIDDEN, dtype=torch.bfloat16,
                                                            device=hidden_states.device)
        if logits_in_kernel is not None:
            run(hidden_states, hss, None, None, gemm1_weights, gemm1_weights_scale, gemm1_bias, gemm1_alpha,
                gemm1_beta, gemm1_clamp_limit, gemm2_weights, gemm2_weights_scale, gemm2_bias,
                int(local_expert_offset), recipe, out, router_logits=logits_in_kernel)
        else:
            run(hidden_states, hss, topk_ids.contiguous(), topk_weights.contiguous(), gemm1_weights,
                gemm1_weights_scale, gemm1_bias, gemm1_alpha, gemm1_beta, gemm1_clamp_limit, gemm2_weights,
                gemm2_weights_scale, gemm2_bias, int(local_expert_offset), recipe, out)
        # Same contract as the trtllm-gen op: an empty tensor when the caller supplied the output buffer.
        result = torch.empty(0, dtype=torch.bfloat16, device=out.device) if output is not None else out
    if prefill:
        _say(_PREFILL_ENV, f"active-{form}", f"active ({form}): T={T} module={p_name} min_tokens={p_min_tokens} "
                                             f"recipe={p_recipe} routing={'in-kernel' if routed_alone else 'torch'}")
    elif tag:
        _say(_ENV, f"active-{form}", f"active ({form}): T={T} module={d_name} max_tokens={max_tokens} "
                                     f"recipe={recipe} routing={'in-kernel' if logits_in_kernel is not None else 'torch'}")
    return result


def maybe_run(router_logits, routing_bias, hidden_states, hidden_states_scale, gemm1_weights,
              gemm1_weights_scale, gemm1_bias, gemm1_alpha, gemm1_beta, gemm1_clamp_limit, gemm2_weights,
              gemm2_weights_scale, gemm2_bias, top_k, valid_hidden_size, valid_intermediate_size,
              local_expert_offset, local_num_experts, routing_method_type, gated_act_type, topk_weights,
              topk_ids, output) -> Optional[torch.Tensor]:
    """The op's return value if the alternative kernel ran this call, else None (caller runs trtllm-gen)."""
    return _maybe_run(router_logits, routing_bias, hidden_states, hidden_states_scale, gemm1_weights,
                      gemm1_weights_scale, gemm1_bias, gemm1_alpha, gemm1_beta, gemm1_clamp_limit, gemm2_weights,
                      gemm2_weights_scale, gemm2_bias, top_k, valid_hidden_size, valid_intermediate_size,
                      local_expert_offset, local_num_experts, routing_method_type, gated_act_type, topk_weights,
                      topk_ids, output, False)


def maybe_run_unfinalized(router_logits, routing_bias, hidden_states, hidden_states_scale, gemm1_weights,
                          gemm1_weights_scale, gemm1_bias, gemm1_alpha, gemm1_beta, gemm1_clamp_limit,
                          gemm2_weights, gemm2_weights_scale, gemm2_bias, top_k, valid_hidden_size,
                          valid_intermediate_size, local_expert_offset, local_num_experts, routing_method_type,
                          gated_act_type, topk_weights, topk_ids) -> Optional[List[torch.Tensor]]:
    """``mxe4m3_mxe2m1_block_scale_moe_runner_unfinalized``'s triple [FC2 rows [P, >= 2880] bf16, routing weights
    [T, 4] bf16, expanded_idx_to_permuted_idx [T, 4] int32] if TLLM_MOE_KF_DECODE (all 128 experts local) or
    TLLM_MOE_KF_PREFILL ran this call, else None (caller runs trtllm-gen)."""
    return _maybe_run(router_logits, routing_bias, hidden_states, hidden_states_scale, gemm1_weights,
                      gemm1_weights_scale, gemm1_bias, gemm1_alpha, gemm1_beta, gemm1_clamp_limit, gemm2_weights,
                      gemm2_weights_scale, gemm2_bias, top_k, valid_hidden_size, valid_intermediate_size,
                      local_expert_offset, local_num_experts, routing_method_type, gated_act_type, topk_weights,
                      topk_ids, None, True)
