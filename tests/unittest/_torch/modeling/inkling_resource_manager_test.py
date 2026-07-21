#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""crit8 resource-manager contract: the model fetches its short-conv state pool
from the registered ``InklingConvStateManager`` via the ``resource_manager``
kwarg -- exactly the way ``PyTorchModelEngine`` passes it -- and produces the
same decode as the direct-pool path, and the manager frees slots on request
completion.

This validates the NEW model-side runtime plumbing WITHOUT standing up the full
LLM API server (which the separate ``inkling_llmapi_smoke_test.py`` covers):

1. ``InklingConvStateManager`` wraps an ``InklingConvStateCache`` and lives in a
   real ``ResourceManager`` container under ``ResourceManagerType.
   CONV_STATE_MANAGER``.
2. ``InklingForCausalLM.forward``'s ``_resolve_conv_runtime`` fetches that pool
   from the container and builds the per-forward context/generation split, so a
   prefill + multi-step decode driven ONLY by ``resource_manager=<container>``
   (no explicit ``conv_cache``) reproduces the crit8 direct-pool DENSE decode
   (cos >= 0.999 at the last dense layer -- the tight, routing-free proof).
3. ``ResourceManager.free_resources(request)`` releases the request's pool row
   (the KV-cache request lifetime), so slots do not leak across requests.

Run (single GPU, needs the TRTLLM CUDA extensions + the checkpoint):
    python tests/unittest/_torch/modeling/inkling_resource_manager_test.py
Override the checkpoint with INKLING_CHECKPOINT=/path/to/Inkling-NVFP4-full.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CKPT = os.environ.get(
    "INKLING_CHECKPOINT",
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/"
    "users/kleinc/hf_data/Inkling-NVFP4-full")

N_MODEL_LAYERS = 6  # layers 0-5 (0/1 dense, 2-5 MoE; 0-4 local, 5 global)
N_TOKENS = 24
P_PREFILL = 8


class _FakeRequest:
    """Minimal stand-in for LlmRequest (free_resources reads py_request_id)."""

    def __init__(self, rid):
        self.py_request_id = rid


def _resolve_via_container(container, md, input_ids):
    """Drive the exact model-side fetch that InklingForCausalLM.forward runs."""
    from tensorrt_llm._torch.models.modeling_inkling import \
        _resolve_conv_runtime
    return _resolve_conv_runtime(container, md)


def main() -> int:
    import inkling_moe_replay_test as moe
    import torch
    from inkling_attention_replay_test import _metrics
    from inkling_runtime_state_test import (_make_ml_manager, _md,
                                            _set_layer_offsets)

    from tensorrt_llm._torch.models.modeling_inkling import (  # noqa: F401
        InklingConvStateManager, InklingForConditionalGeneration)
    from tensorrt_llm._torch.pyexecutor.resource_manager import (
        ResourceManager, ResourceManagerType)

    assert torch.cuda.is_available(), "this test needs a CUDA GPU"
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    torch.manual_seed(0)

    moe.N_LAYERS = N_MODEL_LAYERS
    model, config = moe.build_reduced_model(CKPT, device)
    inner = model.model
    tc = config.pretrained_config.text_config
    kv_list = tc.num_kv_heads_per_layer()[:N_MODEL_LAYERS]
    head_dim = tc.head_dim
    dense_last = tc.dense_mlp_idx - 1

    g = torch.Generator(device="cpu").manual_seed(3)
    x_embeds = torch.randn(N_TOKENS, tc.hidden_size,
                           generator=g).to(device).bfloat16()
    input_ids = torch.zeros(N_TOKENS, dtype=torch.int32, device=device)
    pos_all = torch.arange(N_TOKENS, device=device, dtype=torch.int32)

    # --- Build the resource-manager container exactly as create_py_executor does.
    conv_mgr = InklingConvStateManager(config, max_batch_size=2, device=device)
    container = ResourceManager(
        {ResourceManagerType.CONV_STATE_MANAGER: conv_mgr})
    # The container must hand back the same manager the model will fetch.
    assert container.get_resource_manager(
        ResourceManagerType.CONV_STATE_MANAGER) is conv_mgr
    assert conv_mgr.get_max_resource_count() == 2

    _hook = {}

    def _dense_hook(_m, _in, out):
        _hook["o"] = (out[0] if isinstance(out, tuple) else out).detach()

    handle = inner.layers[dense_last].register_forward_hook(_dense_hook)

    # --- Reference: stateless whole-model prefill (the crit4/5-validated path). ---
    ref_mgr = _make_ml_manager(kv_list, head_dim, [N_TOKENS], device)
    _set_layer_offsets(inner)
    try:
        with torch.no_grad():
            md = _md(ref_mgr,
                     num_contexts=1,
                     seq_lens=[N_TOKENS],
                     num_cached=[0],
                     request_ids=[0],
                     N=N_TOKENS)
            inner.forward(md, inputs_embeds=x_embeds, position_ids=pos_all)
    finally:
        ref_mgr.shutdown()
    sref_dense = _hook["o"][P_PREFILL:].clone()

    # --- Decode driven ONLY through the ResourceManager container. Each step
    #     resolves (pool, rt) the way InklingForCausalLM.forward does, then
    #     threads them into InklingModel.forward. ---
    dec_mgr = _make_ml_manager(kv_list, head_dim, [N_TOKENS], device)
    _set_layer_offsets(inner)
    dense_outs = []
    try:
        with torch.no_grad():
            md_p = _md(dec_mgr,
                       num_contexts=1,
                       seq_lens=[P_PREFILL],
                       num_cached=[0],
                       request_ids=[0],
                       N=N_TOKENS)
            pool, rt = _resolve_via_container(container, md_p,
                                              input_ids[:P_PREFILL])
            assert pool is conv_mgr.cache, "model must fetch the manager's pool"
            inner.forward(md_p,
                          inputs_embeds=x_embeds[:P_PREFILL],
                          position_ids=pos_all[:P_PREFILL],
                          conv_cache=pool,
                          conv_rt=rt)
            for p in range(P_PREFILL, N_TOKENS):
                md_d = _md(dec_mgr,
                           num_contexts=0,
                           seq_lens=[1],
                           num_cached=[p],
                           request_ids=[0],
                           N=N_TOKENS)
                pool, rt = _resolve_via_container(container, md_d,
                                                  input_ids[p:p + 1])
                inner.forward(md_d,
                              inputs_embeds=x_embeds[p:p + 1],
                              position_ids=pos_all[p:p + 1],
                              conv_cache=pool,
                              conv_rt=rt)
                dense_outs.append(_hook["o"][:1].clone())
    finally:
        dec_mgr.shutdown()
        handle.remove()
    dec_dense = torch.cat(dense_outs, dim=0).contiguous()
    dense_max, _, dense_cos = _metrics(sref_dense, dec_dense)

    # --- Slot lifetime: request 0 holds a row; free_resources returns it. ---
    slot_before = conv_mgr.cache._slot_of.get(0)
    n_free_before = len(conv_mgr.cache._free)
    container.free_resources(_FakeRequest(0))
    freed = (0 not in conv_mgr.cache._slot_of
             and len(conv_mgr.cache._free) == n_free_before + 1)

    dense_ok = dense_cos >= 0.999 and bool(torch.isfinite(dec_dense).all())
    ok = dense_ok and slot_before is not None and freed
    print(
        f"RESMGR_CONTRACT container_fetch=OK "
        f"DENSE_decode_vs_stateless(cos={dense_cos:.6f} max_abs={dense_max:.4f} "
        f"last_dense_L{dense_last} gate>=0.999) slot_freed={freed} ok={ok}",
        flush=True)
    if ok:
        print("CRIT8_RESOURCE_MANAGER_OK", flush=True)
        return 0
    print(f"CRIT8_RESOURCE_MANAGER_MISMATCH dense_ok={dense_ok} freed={freed}",
          flush=True)
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
