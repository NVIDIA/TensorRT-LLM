# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TRTLLM_Q35_MTP_MOE_QUANT: load-time FP8-block / MXFP4 quantization of BF16 MoE experts for TRTLLM-gen."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="needs SM100/SM103",
)


def _mod():
    from tensorrt_llm._torch.modules import q35_mtp_moe_quant as m

    return m


def test_fp8_block_roundtrip():
    m = _mod()
    g = torch.Generator(device="cuda").manual_seed(0)
    w = (torch.randn((1024, 4096), generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    q, s = m.quantize_fp8_block(w)
    assert q.dtype == torch.float8_e4m3fn and s.shape == (8, 32) and s.dtype == torch.float32
    deq = m.dequantize_fp8_block(q, s)
    rel = (deq - w.float()).norm() / w.float().norm()
    assert rel < 0.04, float(rel)
    # block amax maps to (at most) 448
    blocks = q.float().abs().view(8, 128, 32, 128).amax(dim=(1, 3))
    assert torch.all(blocks <= 448) and torch.all(blocks >= 400)


def test_mxfp4_matches_trtllm_dequant():
    """Packing / scale layout must be what the TRT-LLM MXFP4 kernels read (negative control: a swapped
    nibble order no longer matches)."""
    m = _mod()
    g = torch.Generator(device="cuda").manual_seed(1)
    w = (torch.randn((256, 1024), generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    packed, scales = m.quantize_mxfp4(w)
    assert packed.shape == (256, 512) and scales.shape == (256, 32)
    ours = m.dequantize_mxfp4(packed, scales)
    ref = torch.ops.trtllm.mxfp4_dequantize_unswizzled(packed.cpu(), scales.cpu(), 32).float().cuda()
    assert torch.equal(ours, ref)
    swapped = ((packed >> 4) | (packed << 4)).contiguous()
    bad = torch.ops.trtllm.mxfp4_dequantize_unswizzled(swapped.cpu(), scales.cpu(), 32).float().cuda()
    assert not torch.equal(ours, bad)
    rel = (ours - w.float()).norm() / w.float().norm()
    assert rel < 0.2, float(rel)


def test_mxfp4_rounding_grid():
    m = _mod()
    vals = torch.tensor([0.0, 0.24, 0.25, 0.26, 0.75, 1.25, 2.5, 3.5, 5.0, 5.9, 6.0, -0.75, -5.0],
                        device="cuda")
    w = torch.zeros((1, 32), device="cuda")
    w[0, :vals.numel()] = vals
    w[0, -1] = 6.0  # block amax 6 -> scale 2^0
    packed, scales = m.quantize_mxfp4(w.to(torch.bfloat16))
    assert int(scales[0, 0]) == 127
    got = m.dequantize_mxfp4(packed, scales)[0, :vals.numel()]
    expect = torch.tensor([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 2.0, 4.0, 4.0, 6.0, 6.0, -1.0, -4.0], device="cuda")
    assert torch.equal(got, expect), got


def test_wrap_covers_the_configurable_moe_backend():
    """The model loader calls the backend's load_weights directly (with the wrapper on the ConfigurableMoE only, the
    BF16 dict would reach the quantized loader). Both must convert exactly once."""
    import types

    m = _mod()
    calls = []
    backend = types.SimpleNamespace(initial_local_expert_ids=[0, 1], layer_load_balancer=None,
                                    weight_loading_mode=None)
    backend.load_weights = lambda weights, allow_partial_loading=False: calls.append(("backend", weights[0]))
    wrapper = types.SimpleNamespace(backend=backend, weight_loading_mode=None)
    wrapper.load_weights = lambda weights, allow_partial_loading=False: backend.load_weights(weights,
                                                                                            allow_partial_loading)
    m.wrap_experts_loader(wrapper, "fp8block")
    E, H, I = 2, 256, 128
    w = {"gate_up_proj": torch.zeros((E, H, 2 * I), dtype=torch.bfloat16),
         "down_proj": torch.zeros((E, I, H), dtype=torch.bfloat16)}
    for target in (wrapper, backend):
        calls.clear()
        target.load_weights([dict(w)])
        assert len(calls) == 1, calls
        got = calls[0][1]
        assert set(got) == {f"{e}.{n}.{k}" for e in range(E) for n in ("w1", "w3", "w2")
                            for k in ("weight", "weight_scale_inv")}
        assert got["0.w1.weight"].dtype == torch.float8_e4m3fn and got["0.w1.weight"].shape == (I, H)
        assert got["1.w2.weight"].shape == (H, I)


def _bf16_moe_reference(x, router_logits, w1, w3, w2, top_k):
    probs = torch.softmax(router_logits.float(), dim=-1)
    topv, topi = probs.topk(top_k, dim=-1)
    topv = topv / topv.sum(-1, keepdim=True)
    out = torch.zeros_like(x, dtype=torch.float32)
    for t in range(x.shape[0]):
        for j in range(top_k):
            e = int(topi[t, j])
            h = x[t].float()
            a = torch.nn.functional.silu(w1[e].float() @ h) * (w3[e].float() @ h)
            out[t] += topv[t, j] * (w2[e].float() @ a)
    return out


@pytest.mark.parametrize("via_backend", [False, True])
@pytest.mark.parametrize("mode", ["fp8block", "mxfp4"])
@pytest.mark.parametrize("tokens", [1, 7])
def test_quantized_trtllm_gen_moe_vs_bf16(mode, tokens, via_backend, monkeypatch):
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.modules.fused_moe import (MoEWeightLoadingMode, RenormalizeMoeRoutingMethod,
                                                       create_moe)
    from tensorrt_llm._torch.autotuner import autotune
    from transformers import PretrainedConfig

    m = _mod()
    monkeypatch.setenv("TRTLLM_Q35_MTP_MOE_QUANT", mode)  # as in-model (fp8block: tileN >= 32 only)
    E, H, I, K = 32, 4096, 1024, 4
    g = torch.Generator(device="cuda").manual_seed(2)
    w1 = (torch.randn((E, I, H), generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    w3 = (torch.randn((E, I, H), generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    w2 = (torch.randn((E, H, I), generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    # checkpoint fused layout after the Qwen3.5 mapper: gate_up [E, H, 2I] (gate first), down [E, I, H]
    weights = {"gate_up_proj": torch.cat([w1, w3], dim=1).transpose(1, 2).contiguous(),
               "down_proj": w2.transpose(1, 2).contiguous()}
    cfg = PretrainedConfig()
    cfg.num_experts, cfg.hidden_size, cfg.intermediate_size, cfg.torch_dtype = E, H, I, torch.bfloat16
    moe = create_moe(routing_method=RenormalizeMoeRoutingMethod(top_k=K), reduce_results=False,
                     model_config=ModelConfig(pretrained_config=cfg, moe_backend="TRTLLM"),
                     override_quant_config=m.quant_config_for(mode),
                     weight_loading_mode=MoEWeightLoadingMode.VANILLA, dtype=torch.bfloat16)
    m.wrap_experts_loader(moe, mode)
    # via_backend: the model loader's path (the weight mapper hands the expert tensors to the
    # ConfigurableMoE's backend module, which it finds through named_modules()).
    (getattr(moe, "backend", moe) if via_backend else moe).load_weights([weights])
    if hasattr(moe, "post_load_weights"):
        moe.post_load_weights()
    moe.cuda()
    x = (torch.randn((tokens, H), generator=g, device="cuda")).to(torch.bfloat16)
    logits = torch.randn((tokens, E), generator=g, device="cuda").to(torch.bfloat16)
    with torch.inference_mode(), autotune():
        moe.forward(x, logits)
    with torch.inference_mode():
        out = moe.forward(x, logits).float()
    ref = _bf16_moe_reference(x, logits, w1, w3, w2, K)
    rel = (out - ref).norm() / ref.norm()
    print(f"{mode} tokens={tokens}: relative L2 error vs BF16 reference {float(rel):.4f}")
    # Gross-error guard (a wrong layout / scale gives ~1.0+): FP8 weights + dynamic FP8 activations
    # (two GEMMs) land at ~5-8 % on Gaussian weights, MXFP4 weights at ~15-20 %.
    assert rel < (0.10 if mode == "fp8block" else 0.25), float(rel)


def _fp8block_shard_moe(E, H, I, K, seed, cuda_first=False):
    """FP8-block TRTLLM-gen experts with one rank's MoE TP4 x EP2 shard shapes, loaded from BF16 weights the model
    loader's way. Returns the module and the BF16 weights (w1, w3, w2) for the reference. cuda_first: move the
    module to the GPU before loading, as the model loader does (a later .cuda() copies each parameter's view only)."""
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.modules.fused_moe import (MoEWeightLoadingMode, RenormalizeMoeRoutingMethod,
                                                       create_moe)
    from transformers import PretrainedConfig

    m = _mod()
    g = torch.Generator(device="cuda").manual_seed(seed)
    w1 = (torch.randn((E, I, H), generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    w3 = (torch.randn((E, I, H), generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    w2 = (torch.randn((E, H, I), generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    weights = {"gate_up_proj": torch.cat([w1, w3], dim=1).transpose(1, 2).contiguous(),
               "down_proj": w2.transpose(1, 2).contiguous()}
    cfg = PretrainedConfig()
    cfg.num_experts, cfg.hidden_size, cfg.intermediate_size, cfg.torch_dtype = E, H, I, torch.bfloat16
    moe = create_moe(routing_method=RenormalizeMoeRoutingMethod(top_k=K), reduce_results=False,
                     model_config=ModelConfig(pretrained_config=cfg, moe_backend="TRTLLM",
                                              max_num_tokens=8192),
                     override_quant_config=m.quant_config_for("fp8block"),
                     weight_loading_mode=MoEWeightLoadingMode.VANILLA, dtype=torch.bfloat16)
    m.wrap_experts_loader(moe, "fp8block")
    if cuda_first:
        moe.cuda()
    getattr(moe, "backend", moe).load_weights([weights])
    if hasattr(moe, "post_load_weights"):
        moe.post_load_weights()
    moe.cuda()
    return moe, (w1, w3, w2), g


@pytest.mark.parametrize("tokens", [7, 28, 112, 1024, 8192])
def test_fp8block_moe_tp4_shard_shapes(tokens, monkeypatch):
    """The in-model shapes of the MTP experts under MoE TP4 x EP2 on one rank (256 local experts, intermediate
    1024 / 4 = 256 rows = 2 FP8 blocks, top-10) at decode and prefill M, with autotuning as in-model (tileN >= 32).
    Single-rank module with the shard's shapes (a multi-rank Mapping needs MPI)."""
    from tensorrt_llm._torch.autotuner import autotune

    monkeypatch.delenv("TRTLLM_Q35_FP8_MOE_MIN_TILEN", raising=False)
    monkeypatch.setenv("TRTLLM_Q35_MTP_MOE_QUANT", "fp8block")

    E, H, I, K = 256, 4096, 256, 10
    moe, (w1, w3, w2), g = _fp8block_shard_moe(E, H, I, K, seed=5)
    x = (torch.randn((tokens, H), generator=g, device="cuda")).to(torch.bfloat16)
    logits = torch.randn((tokens, E), generator=g, device="cuda").to(torch.bfloat16)
    with torch.inference_mode(), autotune():
        moe.forward(x, logits)
    torch.cuda.synchronize()
    with torch.inference_mode():
        out = moe.forward(x, logits).float()
    torch.cuda.synchronize()
    n = min(tokens, 64)  # the Python reference is slow; check the first rows
    ref = _bf16_moe_reference(x[:n], logits[:n], w1, w3, w2, K)
    rel = (out[:n] - ref).norm() / ref.norm()
    print(f"fp8block E{E} I{I} top{K} tokens={tokens}: relative L2 error {float(rel):.4f}")
    assert rel < 0.10, float(rel)


def _runner_inputs(tokens, H=4096):
    """Positional inputs of FP8BlockScaleMoERunner.get_valid_tactics (only the hidden-states shape is read)."""
    x = torch.empty((tokens, H), dtype=torch.float8_e4m3fn, device="cuda")
    return [None, None, x, None, None, None, None, None, None, None]


@pytest.mark.parametrize("tokens", [1, 2, 7, 28, 112, 1024])
def test_fp8block_runner_tactics_skip_small_tiles(tokens, monkeypatch):
    """Small-tile (tileN 8/16) FP8 block-scale MoE tactics are excluded (known issue). With
    TRTLLM_Q35_MTP_MOE_QUANT=fp8block the runner offers the AutoTuner only tileN >= 32; the cache-miss fallback
    is the widest tile. Negative controls: flag unset, and TRTLLM_Q35_FP8_MOE_MIN_TILEN=0, keep tiles 8 / 16."""
    from tensorrt_llm._torch.custom_ops import trtllm_gen_custom_ops as ops

    def tiles(min_env=None, quant="fp8block"):
        monkeypatch.delenv("TRTLLM_Q35_FP8_MOE_MIN_TILEN", raising=False)
        monkeypatch.delenv("TRTLLM_Q35_MTP_MOE_QUANT", raising=False)
        if quant:
            monkeypatch.setenv("TRTLLM_Q35_MTP_MOE_QUANT", quant)
        if min_env is not None:
            monkeypatch.setenv("TRTLLM_Q35_FP8_MOE_MIN_TILEN", min_env)
        # one rank of TEP8 MOE_TP=4 x EP=2 (256 local experts, I 256) and of MOE_TP=2 x EP=4 (128 local, I 512)
        out = set()
        for local, inter, offset in ((256, 256, 256), (128, 512, 384)):
            runner = ops.FP8BlockScaleMoERunner(512, 10, None, None, None, inter, offset, local, None, 1, 0)
            out |= {int(t[0]) for t in runner.get_valid_tactics(_runner_inputs(tokens), None)}
        return out

    restricted = tiles()
    assert restricted and min(restricted) >= 32, restricted
    stock = tiles(quant=None)
    if tokens <= 112:
        assert {8, 16} <= stock, stock  # the small tiles exist for decode M (negative control)
    assert tiles(min_env="0") == stock
    assert restricted == {t for t in stock if t >= 32}, (restricted, stock)
    runner = ops.FP8BlockScaleMoERunner(512, 10, None, None, None, 256, 256, 256, None, 1, 0)
    assert runner.get_fallback_tactic(4096, tokens)[0] >= 32


@pytest.mark.parametrize("tokens", [1, 2, 7, 28, 112])
def test_quantized_mtp_experts_tune_and_run_large_tiles_only(tokens, monkeypatch):
    """In-model path of the quantized MTP experts under autotune(): every tactic the AutoTuner profiles and the
    tactic the forward then runs have tileN >= 32, and the output matches the BF16 reference."""
    from tensorrt_llm._torch.autotuner import AutoTuner, autotune
    from tensorrt_llm._torch.custom_ops import trtllm_gen_custom_ops as ops

    monkeypatch.delenv("TRTLLM_Q35_FP8_MOE_MIN_TILEN", raising=False)
    monkeypatch.setenv("TRTLLM_Q35_MTP_MOE_QUANT", "fp8block")
    E, H, I, K = 256, 4096, 256, 10
    moe, (w1, w3, w2), g = _fp8block_shard_moe(E, H, I, K, seed=9)
    used = []
    orig_forward = ops.FP8BlockScaleMoERunner.forward

    def recording_forward(self, inputs, tactic=[-1, -1], output=None):
        used.append(list(tactic))
        return orig_forward(self, inputs, tactic=tactic, output=output)

    monkeypatch.setattr(ops.FP8BlockScaleMoERunner, "forward", recording_forward)
    AutoTuner.get().clear_cache()  # other tests' profiles of this runner would be cache hits
    x = (torch.randn((tokens, H), generator=g, device="cuda")).to(torch.bfloat16)
    logits = torch.randn((tokens, E), generator=g, device="cuda").to(torch.bfloat16)
    with torch.inference_mode(), autotune():
        moe.forward(x, logits)
    torch.cuda.synchronize()
    profiled = {tuple(t) for t in used}
    assert len(profiled) > 1, profiled  # it did profile
    assert all(t[0] >= 32 for t in profiled), sorted(profiled)
    used.clear()
    with torch.inference_mode():
        out = moe.forward(x, logits).float()
    torch.cuda.synchronize()
    assert len(used) == 1 and used[0][0] >= 32, used
    n = min(tokens, 64)
    ref = _bf16_moe_reference(x[:n], logits[:n], w1, w3, w2, K)
    rel = (out[:n] - ref).norm() / ref.norm()
    print(f"fp8block E{E} I{I} top{K} tokens={tokens}: tactic {used[0]}, relative L2 error {float(rel):.4f}")
    assert rel < 0.10, float(rel)


def test_stale_small_tile_tactic_is_replaced(monkeypatch):
    """A small tile chosen without the restriction (e.g. an AutoTuner cache file written before it) is replaced by
    the cache-miss fallback at run time; without fp8block it is kept (negative control)."""
    from tensorrt_llm._torch.autotuner import AutoTuner
    from tensorrt_llm._torch.custom_ops import trtllm_gen_custom_ops as ops

    E, H, I, K = 256, 4096, 256, 10
    moe, _, g = _fp8block_shard_moe(E, H, I, K, seed=11)
    used = []
    orig_forward = ops.FP8BlockScaleMoERunner.forward

    def recording_forward(self, inputs, tactic=[-1, -1], output=None):
        used.append(list(tactic))
        if tactic[0] < 32:  # record the choice without launching the excluded small-tile tactic
            x = inputs[2]
            return output if output is not None else torch.zeros(
                (x.shape[0], x.shape[1]), dtype=torch.bfloat16, device=x.device)
        return orig_forward(self, inputs, tactic=tactic, output=output)

    monkeypatch.setattr(ops.FP8BlockScaleMoERunner, "forward", recording_forward)
    tuner = AutoTuner.get()
    orig_choose = type(tuner).choose_one

    def stale_choose(self, custom_op, runners, *args, **kwargs):
        if custom_op == "trtllm::fp8_block_scale_moe_runner":
            return runners[0], [8, 0]
        return orig_choose(self, custom_op, runners, *args, **kwargs)

    monkeypatch.setattr(type(tuner), "choose_one", stale_choose)
    x = (torch.randn((7, H), generator=g, device="cuda")).to(torch.bfloat16)
    logits = torch.randn((7, E), generator=g, device="cuda").to(torch.bfloat16)
    monkeypatch.delenv("TRTLLM_Q35_FP8_MOE_MIN_TILEN", raising=False)
    monkeypatch.setenv("TRTLLM_Q35_MTP_MOE_QUANT", "fp8block")
    with torch.inference_mode():
        moe.forward(x, logits)
    torch.cuda.synchronize()
    assert used and used[-1][0] >= 32, used
    monkeypatch.delenv("TRTLLM_Q35_MTP_MOE_QUANT")
    with torch.inference_mode():
        moe.forward(x, logits)
    torch.cuda.synchronize()
    assert used[-1] == [8, 0], used


def test_guard_flag_parsing(monkeypatch):
    """TRTLLM_Q35_MTP_MOE_QUANT_GUARD: w2 by default with fp8block, nothing for other modes, 0 / all / lists."""
    m = _mod()
    monkeypatch.delenv("TRTLLM_Q35_MTP_MOE_QUANT_GUARD", raising=False)
    assert m.mtp_moe_quant_guard("fp8block") == ["w2_weight"]
    assert m.mtp_moe_quant_guard("mxfp4") == [] and m.mtp_moe_quant_guard(None) == []
    for off in ("0", "off", "none", ""):
        monkeypatch.setenv("TRTLLM_Q35_MTP_MOE_QUANT_GUARD", off)
        assert m.mtp_moe_quant_guard("fp8block") == [], off
    monkeypatch.setenv("TRTLLM_Q35_MTP_MOE_QUANT_GUARD", "all")
    assert sorted(m.mtp_moe_quant_guard("fp8block")) == sorted(
        ["w3_w1_weight", "w3_w1_weight_scaling_factor", "w2_weight", "w2_weight_scaling_factor"])
    monkeypatch.setenv("TRTLLM_Q35_MTP_MOE_QUANT_GUARD", " w2, w31 ,w2")
    assert m.mtp_moe_quant_guard("fp8block") == ["w2_weight", "w3_w1_weight"]
    assert m.mtp_moe_quant_guard("mxfp4") == []
    monkeypatch.setenv("TRTLLM_Q35_MTP_MOE_QUANT_GUARD", "w4")
    with pytest.raises(ValueError):
        m.mtp_moe_quant_guard("fp8block")


def _storage_view(p, n_slots):
    """The first n_slots expert slots of p's storage (incl. any slot after the parameter's last expert); never
    resizes the storage (as_strided checks the bounds, Tensor.set_ would grow a too-small storage)."""
    slot = p[0].numel() * p.element_size()
    assert p.untyped_storage().nbytes() >= p.storage_offset() * p.element_size() + n_slots * slot
    return p.data.as_strided((n_slots, *p.shape[1:]), p.stride(), p.storage_offset())


def test_guard_pad_keeps_parameter_shape_and_values():
    import types

    m = _mod()
    g = torch.Generator(device="cuda").manual_seed(3)
    w = torch.nn.Parameter((torch.randn((3, 256, 128), generator=g, device="cuda") * 8).to(torch.float8_e4m3fn),
                           requires_grad=False)
    s = torch.nn.Parameter(torch.rand((3, 2, 1), generator=g, device="cuda"), requires_grad=False)
    backend = types.SimpleNamespace(w2_weight=w, w2_weight_scaling_factor=s)
    before_w, before_s = w.data.clone(), s.data.clone()
    slot = w[0].numel() * w.element_size()
    assert w.untyped_storage().nbytes() == 3 * slot
    assert m.guard_pad_expert_params(backend, ["w2_weight", "no_such_tensor"]) == ["w2_weight"]
    assert backend.w2_weight is w and w.shape == (3, 256, 128) and w.is_contiguous() and w.storage_offset() == 0
    assert torch.equal(w.data.view(torch.uint8), before_w.view(torch.uint8))
    assert w.untyped_storage().nbytes() == 4 * slot
    assert torch.all(_storage_view(w, 4)[3].view(torch.uint8) == 0)
    # untouched tensor: same storage, same values (negative control for the padding)
    assert s.untyped_storage().nbytes() == s.numel() * s.element_size() and torch.equal(s.data, before_s)


@pytest.mark.parametrize("tokens", [1, 7])
def test_guarded_fp8block_experts_match_unguarded(tokens, monkeypatch):
    """The default guard (w2) adds one zero slot after the last local expert of the loaded backend's w2_weight and
    changes no output (bitwise vs the same experts loaded with TRTLLM_Q35_MTP_MOE_QUANT_GUARD=0, same tactic);
    negative control: the unguarded module has no extra slot."""
    monkeypatch.delenv("TRTLLM_Q35_FP8_MOE_MIN_TILEN", raising=False)
    monkeypatch.setenv("TRTLLM_Q35_MTP_MOE_QUANT", "fp8block")
    E, H, I, K = 256, 4096, 256, 10
    outs, slots = {}, {}
    for guard in ("w2", "0"):
        monkeypatch.setenv("TRTLLM_Q35_MTP_MOE_QUANT_GUARD", guard)
        moe, _, g = _fp8block_shard_moe(E, H, I, K, seed=13, cuda_first=True)
        backend = getattr(moe, "backend", moe)
        w2 = backend.w2_weight
        assert w2.shape == (E, H, I)
        slots[guard] = w2.untyped_storage().nbytes() // (H * I)
        if guard != "0":
            assert torch.all(_storage_view(w2, E + 1)[E].view(torch.uint8) == 0)
        x = (torch.randn((tokens, H), generator=g, device="cuda")).to(torch.bfloat16)
        logits = torch.randn((tokens, E), generator=g, device="cuda").to(torch.bfloat16)
        with torch.inference_mode():
            outs[guard] = moe.forward(x, logits).clone()
        torch.cuda.synchronize()
        del moe, backend, w2
        torch.cuda.empty_cache()
    assert slots == {"w2": E + 1, "0": E}, slots
    assert torch.equal(outs["w2"], outs["0"])


# ----------------------------------------------------------------------------------------------
# nvfp4 (Step 2: NVFP4 MTP experts on the target's trtllm-gen FP4 path)
# ----------------------------------------------------------------------------------------------
def test_nvfp4_e2m1_rounding_grid():
    """Round to nearest, ties to the even code (cvt.rn.satfinite.e2m1x2): 0.25 -> 0, 0.75 -> 1, 1.25 -> 1,
    1.75 -> 2, 2.5 -> 2, 3.5 -> 4, 5 -> 4, above 5 -> 6; negative values keep the sign, -0 stays 0."""
    m = _mod()
    vals = [0.0, 0.24, 0.25, 0.26, 0.75, 1.25, 1.3, 1.75, 2.5, 2.6, 3.5, 5.0, 5.1, -0.75, -0.1]
    grid = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.5, 2.0, 2.0, 3.0, 4.0, 4.0, 6.0, -1.0, 0.0]
    w = torch.zeros((2, 16), device="cuda")
    w[0, :15] = torch.tensor(vals, device="cuda")
    w[0, 15] = 6.0  # block amax 6 with weight_scale_2 = 1 -> block scale exactly 1
    w[1, :15] = torch.tensor(vals, device="cuda") * 0.5
    w[1, 15] = 3.0  # block amax 3 -> block scale 0.5 (exact in E4M3): same codes, half the values
    packed, sf, ws2 = m.quantize_nvfp4(w, amax=6.0 * 448.0)
    assert float(ws2) == 1.0
    assert sf.view(torch.float8_e4m3fn).float()[:, 0].tolist() == [1.0, 0.5]
    got = m.dequantize_nvfp4(packed, sf, ws2)
    assert got[0, :15].tolist() == grid, got[0]
    assert got[1, :15].tolist() == [v * 0.5 for v in grid], got[1]
    assert got[:, 15].tolist() == [6.0, 3.0]
    # -0.1 rounds to -0 (code 8, the sign kept as the hardware conversion does), -0.75 to -1 (code 8 | 2)
    codes = torch.stack([packed[0] & 0x0F, packed[0] >> 4], dim=-1).view(-1)
    assert int(codes[14]) == 0x8 and int(codes[13]) == 0xA and int(codes[0]) == 0


def test_nvfp4_matches_trtllm_fp4_quantize():
    """Same codes and block scales as the TRT-LLM NVFP4 quantization kernel (linear scale layout) for the same
    global scale: the trtllm-gen weights must be in the format the kernels themselves produce. Negative control: a
    swapped nibble order differs."""
    m = _mod()
    g = torch.Generator(device="cuda").manual_seed(21)
    w = (torch.randn((512, 4096), generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    w[3, 100] = 0.9  # an outlier block
    w[5, :64] = 0  # all-zero blocks
    packed, sf, ws2 = m.quantize_nvfp4(w)
    gs = (1.0 / ws2).reshape(1).float().cuda()
    q_ref, sf_ref = torch.ops.trtllm.fp4_quantize(w, gs, 16, False, False)
    sf_ref = sf_ref.view(torch.uint8).view(512, 256)
    assert torch.equal(sf.view(torch.uint8), sf_ref)
    same = (packed == q_ref.view(torch.uint8)).float().mean().item()
    assert same > 0.9999, same  # reciprocal rounding can move rare boundary values by one code
    # negative control: dropping the sign of -0 (codes 8 -> 0) is a visible format difference (~3 % of the codes)
    lo, hi = packed & 0x0F, packed >> 4
    plus0 = (torch.where(lo == 8, 0, lo) | (torch.where(hi == 8, 0, hi) << 4)).to(torch.uint8)
    assert (plus0 == q_ref.view(torch.uint8)).float().mean().item() < 0.99
    ours = m.dequantize_nvfp4(packed, sf, ws2)
    ref = m.dequantize_nvfp4(q_ref.view(torch.uint8), sf_ref, ws2)
    assert (ours - ref).abs().max().item() <= ours.abs().max().item() * 0.2
    swapped = ((packed >> 4) | (packed << 4)).contiguous()
    assert not torch.equal(swapped, q_ref.view(torch.uint8))
    rel = (ours - w.float()).norm() / w.float().norm()
    assert rel < 0.12, float(rel)


def test_nvfp4_mse_block_scales():
    """TRTLLM_Q35_MTP_MOE_NVFP4_MSE: least-squares block scales lower the weight error (never raise it per block:
    the amax scale is a candidate) and keep the format (same per-tensor scale, E4M3 scales <= the amax scale)."""
    m = _mod()
    g = torch.Generator(device="cuda").manual_seed(31)
    w = (torch.randn((256, 2048), generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    p0, s0, g0 = m.quantize_nvfp4(w, mse=False)
    p1, s1, g1 = m.quantize_nvfp4(w, mse=True)
    assert float(g0) == float(g1)
    assert torch.all(s1.view(torch.float8_e4m3fn).float() <= s0.view(torch.float8_e4m3fn).float())
    d0 = m.dequantize_nvfp4(p0, s0, g0).view(256, 128, 16)
    d1 = m.dequantize_nvfp4(p1, s1, g1).view(256, 128, 16)
    wf = w.float().view(256, 128, 16)
    e0 = (d0 - wf).square().sum(-1)
    e1 = (d1 - wf).square().sum(-1)
    assert torch.all(e1 <= e0 * (1 + 1e-6) + 1e-12)
    assert e1.sum() < 0.95 * e0.sum(), (float(e1.sum()), float(e0.sum()))


def test_nvfp4_convert_keys_and_scales(monkeypatch):
    """Per-expert checkpoint-format keys; w1 / w3 share one weight_scale_2; input scales from the AMAX env."""
    m = _mod()
    monkeypatch.setenv("TRTLLM_Q35_MTP_MOE_NVFP4_AMAX", "64,512")
    E, H, I = 2, 256, 128
    g = torch.Generator(device="cuda").manual_seed(4)
    gate_up = (torch.randn((E, H, 2 * I), generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    gate_up[1, 0, 0] = 1.0  # w1 of expert 1 has the larger amax -> w3 shares it
    down = (torch.randn((E, I, H), generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    out = m.convert_bf16_experts({"gate_up_proj": gate_up, "down_proj": down}, [0, 1], "nvfp4",
                                 torch.device("cuda"))
    assert set(out) == {f"{e}.{n}.{k}" for e in range(E) for n in ("w1", "w3", "w2")
                        for k in ("weight", "weight_scale", "weight_scale_2", "input_scale")}
    assert out["0.w1.weight"].shape == (I, H // 2) and out["0.w1.weight"].dtype == torch.uint8
    assert out["0.w1.weight_scale"].shape == (I, H // 16) and out["0.w1.weight_scale"].dtype == torch.float8_e4m3fn
    assert out["0.w2.weight"].shape == (H, I // 2) and out["0.w2.weight_scale"].shape == (H, I // 16)
    assert float(out["1.w1.weight_scale_2"]) == float(out["1.w3.weight_scale_2"]) == pytest.approx(1.0 / 2688.0)
    assert float(out["0.w1.input_scale"]) == pytest.approx(64.0 / 2688.0)
    assert float(out["0.w2.input_scale"]) == pytest.approx(512.0 / 2688.0)
    w1 = gate_up[1].transpose(0, 1)[:I].float()
    d1 = m.dequantize_nvfp4(out["1.w1.weight"], out["1.w1.weight_scale"], out["1.w1.weight_scale_2"])
    assert ((d1 - w1).norm() / w1.norm()).item() < 0.15


def _nvfp4_moe_emulation(m, weights, x, logits, E, K, fc1_amax, fc2_amax):
    """What the trtllm-gen NVFP4 MoE computes, in fp32 torch: x quantized to NVFP4 with the FC1 global range, the
    converted (dequantized) NVFP4 weights, the SwiGLU output quantized with the FC2 global range, renormalized top-k."""
    conv = m.convert_bf16_experts(weights, list(range(E)), "nvfp4", x.device)
    cache = {}

    def deq(e, n):
        if (e, n) not in cache:
            cache[(e, n)] = m.dequantize_nvfp4(conv[f"{e}.{n}.weight"], conv[f"{e}.{n}.weight_scale"],
                                               conv[f"{e}.{n}.weight_scale_2"].to(x.device))
        return cache[(e, n)]

    xq = m.dequantize_nvfp4(*m.quantize_nvfp4(x.float(), amax=fc1_amax))
    probs = torch.softmax(logits.float(), dim=-1)
    topv, topi = probs.topk(K, dim=-1)
    topv = topv / topv.sum(-1, keepdim=True)
    out = torch.zeros((x.shape[0], x.shape[1]), dtype=torch.float32, device=x.device)
    for t in range(x.shape[0]):
        for j in range(K):
            e = int(topi[t, j])
            h = torch.nn.functional.silu(deq(e, "w1") @ xq[t]) * (deq(e, "w3") @ xq[t])
            hq = m.dequantize_nvfp4(*m.quantize_nvfp4(h[None], amax=fc2_amax))[0]
            out[t] += topv[t, j] * (deq(e, "w2") @ hq)
    return out


@pytest.mark.parametrize("via_backend", [False, True])
@pytest.mark.parametrize("tokens", [1, 7, 28])
def test_nvfp4_trtllm_gen_moe_vs_bf16(tokens, via_backend, monkeypatch):
    """BF16 experts converted to NVFP4 load into the trtllm-gen FP4 MoE (its own scale reconciliation, shuffles
    and alphas) and match the BF16 reference within NVFP4 weight + activation error."""
    from tensorrt_llm._torch.autotuner import autotune
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.modules.fused_moe import (MoEWeightLoadingMode, RenormalizeMoeRoutingMethod,
                                                       create_moe)
    from transformers import PretrainedConfig

    m = _mod()
    monkeypatch.setenv("TRTLLM_Q35_MTP_MOE_QUANT", "nvfp4")
    monkeypatch.setenv("TRTLLM_Q35_MTP_MOE_NVFP4_AMAX", "16,64")
    E, H, I, K = 64, 4096, 512, 10
    g = torch.Generator(device="cuda").manual_seed(6)
    w1 = (torch.randn((E, I, H), generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    w3 = (torch.randn((E, I, H), generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    w2 = (torch.randn((E, H, I), generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    weights = {"gate_up_proj": torch.cat([w1, w3], dim=1).transpose(1, 2).contiguous(),
               "down_proj": w2.transpose(1, 2).contiguous()}
    cfg = PretrainedConfig()
    cfg.num_experts, cfg.hidden_size, cfg.intermediate_size, cfg.torch_dtype = E, H, I, torch.bfloat16
    moe = create_moe(routing_method=RenormalizeMoeRoutingMethod(top_k=K), reduce_results=False,
                     model_config=ModelConfig(pretrained_config=cfg, moe_backend="TRTLLM", max_num_tokens=8192),
                     override_quant_config=m.quant_config_for("nvfp4"),
                     weight_loading_mode=MoEWeightLoadingMode.VANILLA, dtype=torch.bfloat16)
    m.wrap_experts_loader(moe, "nvfp4")
    moe.cuda()  # as the model loader: parameters on the GPU before load_weights
    (getattr(moe, "backend", moe) if via_backend else moe).load_weights([weights])
    if hasattr(moe, "post_load_weights"):
        moe.post_load_weights()
    backend = getattr(moe, "backend", moe)
    assert backend.has_nvfp4
    assert float(backend.fc31_input_scale) == pytest.approx(2688.0 / 16.0)
    assert float(backend.fc2_input_scale) == pytest.approx(2688.0 / 64.0)
    x = torch.randn((tokens, H), generator=g, device="cuda").to(torch.bfloat16)
    logits = torch.randn((tokens, E), generator=g, device="cuda").to(torch.bfloat16)
    with torch.inference_mode(), autotune():
        moe.forward(x, logits)
    torch.cuda.synchronize()
    with torch.inference_mode():
        out = moe.forward(x, logits).float()
    torch.cuda.synchronize()
    ref = _bf16_moe_reference(x, logits, w1, w3, w2, K)
    rel = (out - ref).norm() / ref.norm()
    emu = _nvfp4_moe_emulation(m, weights, x, logits, E, K, 16.0, 64.0)
    rel_emu = (out - emu).norm() / emu.norm()
    print(f"nvfp4 tokens={tokens} via_backend={via_backend}: relative L2 error vs BF16 reference {float(rel):.4f}, "
          f"vs NVFP4 emulation {float(rel_emu):.4f}")
    # vs BF16: four NVFP4 roundings (x, w1 / w3, SwiGLU output, w2; ~9.5 % each on Gaussian data) compound to ~0.23;
    # a wrong layout / scale gives ~1. vs the emulation of the same NVFP4 arithmetic: only accumulation order and rare
    # rounding flips of the SwiGLU-output quantization.
    assert rel < 0.30, float(rel)
    assert rel_emu < 0.06, float(rel_emu)


def test_calib_probe_matches_reference(monkeypatch):
    """TRTLLM_Q35_MTP_MOE_CALIB: x amax and the routed SwiGLU-output amax of a BF16 (CUTLASS) MoE equal a torch
    reference; an input-only record (no logits) leaves the h statistics alone."""
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.modules.fused_moe import (MoEWeightLoadingMode, RenormalizeMoeRoutingMethod,
                                                       create_moe)
    from transformers import PretrainedConfig

    m = _mod()
    E, H, I, K = 16, 512, 256, 4
    g = torch.Generator(device="cuda").manual_seed(8)
    w1 = (torch.randn((E, I, H), generator=g, device="cuda") * 0.05).to(torch.bfloat16)
    w3 = (torch.randn((E, I, H), generator=g, device="cuda") * 0.05).to(torch.bfloat16)
    w2 = (torch.randn((E, H, I), generator=g, device="cuda") * 0.05).to(torch.bfloat16)
    cfg = PretrainedConfig()
    cfg.num_experts, cfg.hidden_size, cfg.intermediate_size, cfg.torch_dtype = E, H, I, torch.bfloat16
    moe = create_moe(routing_method=RenormalizeMoeRoutingMethod(top_k=K), reduce_results=False,
                     model_config=ModelConfig(pretrained_config=cfg, moe_backend="CUTLASS"),
                     weight_loading_mode=MoEWeightLoadingMode.FUSED_GATE_UP_PROJ, dtype=torch.bfloat16)
    moe.load_weights([{"gate_up_proj": torch.cat([w1, w3], dim=1).transpose(1, 2).contiguous(),
                       "down_proj": w2.transpose(1, 2).contiguous()}])
    moe.cuda()
    x = torch.randn((5, H), generator=g, device="cuda").to(torch.bfloat16)
    logits = torch.randn((5, E), generator=g, device="cuda").to(torch.bfloat16)
    m._calib_state.clear()
    m.mtp_moe_calib_record(60, moe, x, logits, K)
    m.mtp_moe_calib_record(7, moe, x * 2, None, K)
    st60 = m._calib_state[(x.device, 60)].tolist()
    st7 = m._calib_state[(x.device, 7)].tolist()
    assert st60[0] == pytest.approx(float(x.float().abs().max()))
    assert st7[0] == pytest.approx(2 * float(x.float().abs().max())) and st7[1] == 0 and st7[5] == 0
    top = logits.float().topk(K, dim=-1).indices
    ref = 0.0
    for t in range(x.shape[0]):
        for e in top[t].tolist():
            h = torch.nn.functional.silu(w1[e].float() @ x[t].float()) * (w3[e].float() @ x[t].float())
            ref = max(ref, float(h.abs().max()))
    assert st60[1] == pytest.approx(ref, rel=0.02)  # bf16 GEMM rounding
    assert st60[5] == x.shape[0] * K and st60[3] == 1 and st60[4] == x.shape[0]
    buf, cnt = m._calib_state[("ring", x.device, 60)]
    assert int(cnt) == 1 and buf[0, 0].item() == x.shape[0]
    m._calib_state.clear()
